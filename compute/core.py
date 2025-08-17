import logging
import math
import itertools
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from . import config
from .utils import lag_column, compute_diff, filter_rows

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # set to INFO in production if too verbose
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


def _extract_stream_list(streams: Dict[str, Any], key: str) -> List:
    """
    Handles different shapes of Strava stream responses:
    - streams[key] might be {'data': [...]} (common with key_by_type=true)
    - or streams[key] might be a plain list already
    - or key might be absent -> return []
    """
    val = streams.get(key)
    if val is None:
        return []
    if isinstance(val, dict) and 'data' in val:
        return val.get('data') or []
    if isinstance(val, list):
        return val
    # Sometimes API returns other structure; coerce to empty
    return []


def _pad_lists_to_max(data_dict: Dict[str, List], max_len: int):
    """Extend all lists in dict to max_len with np.nan (or False for boolean 'moving')."""
    for k, lst in data_dict.items():
        if k == 'moving':
            pad_val = False
        else:
            pad_val = np.nan
        if len(lst) < max_len:
            data_dict[k] = list(lst) + [pad_val] * (max_len - len(lst))
    return data_dict


def prepare_data(stream_data: Dict[str, Any], keep_moving_filter: bool = True) -> pd.DataFrame:
    """
    Robustly convert Strava stream response into a DataFrame ready for processing.

    stream_data: dict returned by Strava /activities/{id}/streams with key_by_type=true
    keep_moving_filter: if False, skip filtering on moving==True (useful for debugging)
    """
    logger.info("Preparing data: extracting stream arrays")

    # keys we care about (map multiple possible key names to canonical column names)
    key_map = {
        'time': 'time',
        'cadence': 'cadence',
        'velocity_smooth': 'speed',
        'distance': 'distance',
        'altitude': 'altitude',
        'grade_smooth': 'grade_smooth',
        'latlng': 'latlng',
        'moving': 'moving'
    }

    # Extract lists robustly
    extracted = {}
    lengths = []
    for raw_key, canon in key_map.items():
        lst = _extract_stream_list(stream_data, raw_key)
        extracted[canon] = lst
        lengths.append(len(lst))

    max_len = max(lengths) if lengths else 0
    logger.debug(f"Detected stream lengths per key: { {k: len(v) for k, v in extracted.items()} }, max_len={max_len}")

    # Pad shorter lists so DataFrame construction aligns by index
    extracted = _pad_lists_to_max(extracted, max_len)

    # Special handling: latlng is a list of [lat,lng] — split into columns
    lat_list = []
    lng_list = []
    if 'latlng' in extracted:
        for v in extracted['latlng']:
            if isinstance(v, (list, tuple)) and len(v) == 2:
                lat_list.append(v[0])
                lng_list.append(v[1])
            else:
                lat_list.append(np.nan)
                lng_list.append(np.nan)
        extracted['lat'] = lat_list
        extracted['lng'] = lng_list
        # drop original latlng from extracted dict for DataFrame columns
        extracted.pop('latlng', None)

    # Build DataFrame
    df = pd.DataFrame(extracted)

    logger.info(f"Initial rows (after building DF): {len(df)}")

    # Coerce types
    if 'moving' in df.columns:
        # moving might be 0/1 or True/False strings or booleans
        df['moving'] = df['moving'].apply(lambda x: bool(x) if not pd.isna(x) else False)
    else:
        df['moving'] = False

    for col in ['distance', 'altitude', 'cadence', 'speed', 'time']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep only moving rows if requested
    if keep_moving_filter:
        logger.info("Filtering moving rows")
        before = len(df)
        df = df[df['moving'] == True].copy()
        logger.info(f"Remaining rows after filtering moving==True: {len(df)} (was {before})")
    else:
        logger.info("Skipping moving filter (keep_moving_filter=False)")

    # Compute diffs (uses your utils.compute_diff). If compute_diff expects named columns, ensure match.
    logger.info("Computing distance_diff and altitude_diff")
    # if compute_diff expects 'Distance'/'Altitude' capitalization, we can provide both names (safe)
    # But our utils.compute_diff uses the column name passed. We'll call with the lowercase names.
    df['distance_diff'] = compute_diff(df, 'distance')
    df['altitude_diff'] = compute_diff(df, 'altitude')

    # Filter out tiny movements (same as KNIME step: Distance_diff > 1)
    before = len(df)
    df = df[df['distance_diff'] > 1].copy()
    logger.info(f"Remaining rows after distance_diff > 1: {len(df)} (was {before})")

    return df


def calculate_gradient(df: pd.DataFrame, window: int = config.GRADIENT_WINDOW) -> pd.DataFrame:
    logger.info(f"Calculating gradient with window: {window}")
    if df.empty:
        logger.info("calculate_gradient: empty dataframe in, returning empty df.")
        df['gradient_raw'] = pd.Series(dtype=float)
        df['MA_gradient_raw'] = pd.Series(dtype=float)
        return df

    df['gradient_raw'] = (df['altitude_diff'] / df['distance_diff']) * 100
    # Rule engine: clamp unrealistic extremes -> NaN
    invalid_gradients = df[(df['gradient_raw'] > 30) | (df['gradient_raw'] < -30)].shape[0]
    if invalid_gradients > 0:
        logger.info(f"Setting {invalid_gradients} extreme gradients to NaN")
    df.loc[(df['gradient_raw'] > 30) | (df['gradient_raw'] < -30), 'gradient_raw'] = np.nan

    df['MA_gradient_raw'] = df['gradient_raw'].rolling(window=window, center=True, min_periods=1).mean()
    df['MA_gradient_raw'] = df['MA_gradient_raw'].interpolate(method='linear')
    logger.info("Gradient calculation complete")
    return df


def segment_ride(df: pd.DataFrame, time_limit: float = config.TIME_LIMIT_SEC, grad_limit: float = config.GRAD_LIMIT_PCT) -> pd.DataFrame:
    logger.info(f"Segmenting ride with time_limit={time_limit} sec and grad_limit={grad_limit}%")
    if df.empty:
        logger.info("segment_ride: empty dataframe in, returning empty df.")
        # ensure segment columns exist even if empty
        df['segment_id'] = pd.Series(dtype=int)
        df['segment_start'] = pd.Series(dtype=bool)
        df['time_since_segment'] = pd.Series(dtype=float)
        df['grad_baseline'] = pd.Series(dtype=float)
        return df

    segment_id = 0
    time_since_segment = 0.0
    grad_base = 0.0
    last_time = None

    segment_ids, segment_starts, time_since_segments, grad_bases = [], [], [], []

    for idx, row in df.iterrows():
        t = row.get('time', None)
        grad = row.get('MA_gradient_raw', np.nan)

        td = 0.0 if last_time is None or pd.isna(t) or t < last_time else float(t - last_time)
        last_time = t if not pd.isna(t) else last_time

        if idx == 0:
            segment_id = 1
            grad_base = grad if not pd.isna(grad) else 0.0
            is_start = True
            time_since_segment = 0.0
        else:
            time_since_segment += td
            time_trigger = time_since_segment >= time_limit
            grad_trigger = (not pd.isna(grad)) and (abs(float(grad) - float(grad_base)) >= grad_limit)

            if time_trigger or grad_trigger:
                segment_id += 1
                time_since_segment = 0.0
                grad_base = grad if not pd.isna(grad) else grad_base
                is_start = True
            else:
                is_start = False

        segment_ids.append(int(segment_id))
        segment_starts.append(bool(is_start))
        time_since_segments.append(float(time_since_segment))
        grad_bases.append(float(grad_base) if not pd.isna(grad_base) else float('nan'))

        if idx % 1000 == 0:
            logger.debug(f"Row {idx}: segment_id={segment_id}, time_since_segment={time_since_segment}, grad_base={grad_base}")

    df = df.copy()
    df['segment_id'] = segment_ids
    df['segment_start'] = segment_starts
    df['time_since_segment'] = time_since_segments
    df['grad_baseline'] = grad_bases

    logger.info(f"Segmentation complete: total segments={segment_id}")
    return df


def calculate_cadence_elevation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating cadence and elevation gain")
    if df.empty:
        df['cadence_nonzero'] = pd.Series(dtype=float)
        df['elev_gain'] = pd.Series(dtype=float)
        return df

    df = df.copy()
    df['cadence_nonzero'] = df['cadence'].apply(lambda x: x if (not pd.isna(x) and x > 0) else float('nan'))
    df['elev_gain'] = df['altitude_diff'].apply(lambda x: x if (not pd.isna(x) and x > 0) else 0.0)
    logger.info("Cadence and elevation calculation complete")
    return df


def aggregate_segments(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating segments")
    if df.empty:
        logger.info("aggregate_segments: empty dataframe in, returning empty aggregated df.")
        return pd.DataFrame(columns=[
            'segment_id', 'cadence_nonzero_mean', 'speed_mean', 'elev_gain_mean',
            'distance_sum', 'gradient_mean'
        ])

    agg_df = df.groupby('segment_id').agg(
        cadence_nonzero_mean=('cadence_nonzero', 'mean'),
        speed_mean=('speed', 'mean'),
        elev_gain_mean=('elev_gain', 'mean'),
        distance_sum=('distance_diff', 'sum'),
        gradient_mean=('MA_gradient_raw', 'mean'),
    ).reset_index()

    logger.info(f"Aggregation complete: {len(agg_df)} segments")
    return agg_df


def compute_scores(df: pd.DataFrame, a=config.EXERTION_A, b=config.EXERTION_B, c=config.EXERTION_C) -> pd.DataFrame:
    logger.info("Computing performance and exertion scores")
    if df.empty:
        logger.info("compute_scores: empty df in, returning empty df.")
        return df
    df = df.copy()
    # avoid division by zero
    df['speed_mean'] = df['speed_mean'].replace({0: np.nan})
    df['Exertion_Score'] = ((df['elev_gain_mean'] * (df['gradient_mean'] ** a)) + (df['cadence_nonzero_mean'] ** b)) / (df['speed_mean'] ** c)
    df['Performance_Score'] = df['speed_mean'] / df['Exertion_Score']
    logger.info("Score computation complete")
    return df


def cadence_binning(df: pd.DataFrame, bin_size: int = config.BIN_SIZE) -> pd.DataFrame:
    logger.info("Binning cadence values")
    if df.empty:
        return pd.DataFrame(columns=['cadence_bin', 'Performance_Score', 'mean_cadence'])
    df = df.copy()
    # use floor division; guard NaNs
    df['cadence_bin'] = df['cadence_nonzero_mean'].apply(lambda x: (math.floor(x / bin_size) * bin_size) if not pd.isna(x) else np.nan)
    bin_df = (
        df.groupby('cadence_bin', as_index=False)
        .agg({
            'Performance_Score': 'mean',
            'cadence_nonzero_mean': 'mean',
        })
        .rename(columns={'cadence_nonzero_mean': 'mean_cadence'})
    )
    logger.info("Cadence binning complete")
    return bin_df


def optimal_cadence(df: pd.DataFrame) -> dict:
    """
    Compute the optimal cadence given an aggregated segments DataFrame.
    Returns a dict with optimal_cadence and details
    """
    logger.info("Computing optimal cadence from aggregated segments")
    binned_df = cadence_binning(df)

    if binned_df.empty:
        logger.warning("No data available for optimal cadence computation")
        return {
            "optimal_cadence": None,
            "performance_score": None,
            "exertion_score": None,
            "details": []
        }

    # drop NaN bins before choosing best
    candidate = binned_df.dropna(subset=['cadence_bin', 'Performance_Score'])
    if candidate.empty:
        logger.warning("No valid cadence bins after dropping NaNs")
        return {
            "optimal_cadence": None,
            "performance_score": None,
            "exertion_score": None,
            "details": binned_df.to_dict(orient='records')
        }

    best_row = candidate.loc[candidate['Performance_Score'].idxmax()]
    result = {
        "optimal_cadence": float(best_row['cadence_bin']),
        "performance_score": float(best_row['Performance_Score']),
        "exertion_score": None,
        "details": binned_df.to_dict(orient="records"),
    }
    logger.info(f"Optimal cadence found: {result['optimal_cadence']}")
    return result
