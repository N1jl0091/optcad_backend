# compute/core.py
import pandas as pd
import numpy as np
import logging
from . import config
from .utils import lag_column, compute_diff, filter_rows

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)


def _extract_array(val):
    """
    Given a stream value, return a list/iterable of primitives.
    Strava streams sometimes provide {'data': [...]}, or raw lists.
    """
    if val is None:
        return []
    if isinstance(val, dict) and 'data' in val:
        return val['data'] or []
    if isinstance(val, (list, tuple, pd.Series, np.ndarray)):
        return list(val)
    # scalar -> repeat? We treat as single-element array
    return [val]


def prepare_data(stream_data) -> pd.DataFrame:
    """
    Robust conversion of Strava stream payload into a DataFrame for processing.
    Accepts either:
      - dict of arrays (or dict-of-{'data': arrays}), or
      - a pandas DataFrame (passes through with minimal normalization).
    """
    logger.info("Preparing data: extracting stream arrays")

    # If user passed an actual DataFrame already, use it
    if isinstance(stream_data, pd.DataFrame):
        df = stream_data.copy()
        logger.info(f"Input is already DataFrame with {len(df)} rows")
    else:
        # Normalize streams dict -> arrays
        keys = ['time', 'cadence', 'velocity_smooth', 'distance', 'altitude', 'grade_smooth', 'latlng', 'moving']
        arrays = {}
        max_len = 0
        for k in keys:
            raw = stream_data.get(k, None)
            arr = _extract_array(raw)
            arrays[k] = arr
            if len(arr) > max_len:
                max_len = len(arr)
        logger.debug(f"Detected stream lengths per key: { {k: len(arrays[k]) for k in arrays} }, max_len={max_len}")

        # pad all arrays to max_len with None
        for k, arr in arrays.items():
            if len(arr) < max_len:
                arrays[k] = arr + [None] * (max_len - len(arr))

        # Build DataFrame with canonical column names used by pipeline (lowercase)
        df = pd.DataFrame({
            'time': arrays.get('time', [None] * max_len),
            'cadence': arrays.get('cadence', [None] * max_len),
            'velocity_smooth': arrays.get('velocity_smooth', [None] * max_len),
            'distance': arrays.get('distance', [None] * max_len),
            'altitude': arrays.get('altitude', [None] * max_len),
            'grade_smooth': arrays.get('grade_smooth', [None] * max_len),
            'latlng': arrays.get('latlng', [None] * max_len),
            'moving': arrays.get('moving', [None] * max_len),
        })
        logger.info(f"Initial rows (after building DF): {len(df)}")

    # Convert types safely
    # Normalize 'moving' -> boolean (some entries might be 0/1/True/False/None)
    def _to_bool(v):
        if v is None:
            return False
        # If a list/tuple -> take truthiness (non-empty -> True). Streams shouldn't give lists here.
        if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
            return bool(v)
        # Allow strings like "True"/"true"
        if isinstance(v, str):
            return v.lower() in ("true", "1", "t", "yes", "y")
        try:
            return bool(v)
        except Exception:
            return False

    # Ensure numeric columns
    for col in ['distance', 'altitude', 'cadence', 'velocity_smooth', 'time']:
        if col not in df.columns:
            df[col] = np.nan

    # Coerce numeric columns
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
    df['cadence'] = pd.to_numeric(df['cadence'], errors='coerce')
    df['velocity_smooth'] = pd.to_numeric(df.get('velocity_smooth', pd.Series(dtype=float)), errors='coerce')
    # time sometimes already numeric
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

    # Convert moving robustly
    df['moving'] = df['moving'].apply(_to_bool)

    # Log counts before any filtering
    logger.info(f"Rows before filtering: {len(df)}")
    logger.info(f"Counts by moving flag: {int(df['moving'].sum())} moving, {len(df) - int(df['moving'].sum())} not moving")

    # 1) Filter moving rows
    df = df[df['moving'].astype(bool)].copy()
    logger.info(f"Remaining rows after filtering moving==True: {len(df)}")

    # 2) Compute diffs (use utils.compute_diff)
    logger.info("Computing distance_diff and altitude_diff")
    # Ensure column names match the util usage
    df['distance_diff'] = compute_diff(df, 'distance')
    df['altitude_diff'] = compute_diff(df, 'altitude')

    # 3) Filter tiny distances (keeps only meaningful moves)
    prev_len = len(df)
    df = df[df['distance_diff'] > 1].copy()
    logger.info(f"Remaining rows after distance_diff > 1: {len(df)} (was {prev_len})")

    return df


def calculate_gradient(df: pd.DataFrame, window: int = config.GRADIENT_WINDOW) -> pd.DataFrame:
    logger.info(f"Calculating gradient with window: {window}")
    if df.empty:
        logger.warning("calculate_gradient called with empty DataFrame")
        df['gradient_raw'] = pd.Series(dtype=float)
        df['MA_gradient_raw'] = pd.Series(dtype=float)
        return df

    df['gradient_raw'] = (df['altitude_diff'] / df['distance_diff']) * 100
    # clamp absurd values
    invalid_gradients = df[(df['gradient_raw'] > 30) | (df['gradient_raw'] < -30)].shape[0]
    if invalid_gradients > 0:
        logger.info(f"Setting {invalid_gradients} extreme gradients to NaN")
    df.loc[(df['gradient_raw'] > 30) | (df['gradient_raw'] < -30), 'gradient_raw'] = np.nan

    df['MA_gradient_raw'] = df['gradient_raw'].rolling(window=window, center=True, min_periods=1).mean()
    df['MA_gradient_raw'] = df['MA_gradient_raw'].interpolate(method='linear', limit_direction='both')
    logger.info("Gradient calculation complete")
    return df


def segment_ride(df: pd.DataFrame, time_limit: float = config.TIME_LIMIT_SEC,
                 grad_limit: float = config.GRAD_LIMIT_PCT) -> pd.DataFrame:
    logger.info(f"Segmenting ride with time_limit={time_limit} sec and grad_limit={grad_limit}%")
    if df.empty:
        logger.warning("segment_ride called with empty DataFrame")
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

    for idx, row in df.reset_index(drop=True).iterrows():
        t = row.get('time', None)
        grad = row.get('MA_gradient_raw', np.nan)

        if last_time is None or pd.isna(t) or (isinstance(t, (int, float)) and t < last_time):
            td = 0.0
        else:
            td = 0.0 if pd.isna(t) else (t - last_time)

        if not pd.isna(t):
            last_time = t

        if idx == 0:
            segment_id = 1
            time_since_segment = 0.0
            grad_base = 0.0 if pd.isna(grad) else grad
            is_start = True
        else:
            time_since_segment += td
            time_trigger = time_since_segment >= time_limit
            grad_trigger = (not pd.isna(grad)) and (abs(grad - grad_base) >= grad_limit)

            if time_trigger or grad_trigger:
                segment_id += 1
                time_since_segment = 0.0
                if not pd.isna(grad):
                    grad_base = grad
                is_start = True
            else:
                is_start = False

        segment_ids.append(segment_id)
        segment_starts.append(bool(is_start))
        time_since_segments.append(float(time_since_segment))
        grad_bases.append(float(grad_base if not pd.isna(grad_base) else 0.0))

        if idx % 1000 == 0:
            logger.debug(f"Row {idx}: segment_id={segment_id}, time_since_segment={time_since_segment}, grad_base={grad_base}")

    df = df.reset_index(drop=True)
    df['segment_id'] = pd.Series(segment_ids, index=df.index)
    df['segment_start'] = pd.Series(segment_starts, index=df.index)
    df['time_since_segment'] = pd.Series(time_since_segments, index=df.index)
    df['grad_baseline'] = pd.Series(grad_bases, index=df.index)

    logger.info(f"Segmentation complete: total segments={int(df['segment_id'].nunique()) if 'segment_id' in df.columns else 0}")
    return df


def calculate_cadence_elevation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating cadence and elevation gain")
    if df.empty:
        logger.warning("calculate_cadence_elevation called with empty DataFrame")
        return df
    df = df.copy()
    df['cadence_nonzero'] = df['cadence'].apply(lambda x: x if (pd.notna(x) and x > 0) else float('nan'))
    df['elev_gain'] = df['altitude_diff'].apply(lambda x: x if (pd.notna(x) and x > 0) else 0)
    logger.info("Cadence and elevation calculation complete")
    return df


def aggregate_segments(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating segments")
    if 'segment_id' not in df.columns:
        raise KeyError("'segment_id' missing in DataFrame; segmentation must provide it before aggregation.")
    agg_df = df.groupby('segment_id').agg(
        cadence_nonzero_mean=('cadence_nonzero', 'mean'),
        speed_mean=('velocity_smooth' if 'velocity_smooth' in df.columns else 'speed', 'mean'),
        elev_gain_mean=('elev_gain', 'mean'),
        distance_sum=('distance_diff', 'sum'),
        gradient_mean=('MA_gradient_raw', 'mean'),
    ).reset_index()
    logger.info(f"Aggregation complete: {len(agg_df)} segments")
    return agg_df


def compute_scores(df: pd.DataFrame, a=config.EXERTION_A, b=config.EXERTION_B, c=config.EXERTION_C) -> pd.DataFrame:
    logger.info("Computing performance and exertion scores")
    if df.empty:
        logger.warning("compute_scores called with empty DataFrame")
        return df
    df = df.copy()
    # Avoid divide-by-zero: replace zero speeds with small epsilon
    df['speed_mean'] = df['speed_mean'].replace({0: np.nan})
    df['Exertion_Score'] = ((df['elev_gain_mean'] * (df['gradient_mean'] ** a)).fillna(0) + (df['cadence_nonzero_mean'] ** b).fillna(0)) / (df['speed_mean'] ** c)
    df['Performance_Score'] = df['speed_mean'] / df['Exertion_Score']
    logger.info("Score computation complete")
    return df


def cadence_binning(df: pd.DataFrame, bin_size=config.BIN_SIZE) -> pd.DataFrame:
    logger.info("Binning cadence values")
    if df.empty:
        return pd.DataFrame(columns=['cadence_bin', 'Performance_Score', 'mean_cadence'])
    df = df.copy()
    df['cadence_bin'] = (df['cadence_nonzero_mean'] // bin_size) * bin_size
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
    logger.info("Computing optimal cadence from aggregated segments")
    binned_df = cadence_binning(df)
    if binned_df.empty:
        logger.warning("No data available for optimal cadence computation")
        return {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []}
    best_idx = int(binned_df['Performance_Score'].idxmax())
    best_bin = binned_df.iloc[best_idx]
    result = {
        "optimal_cadence": float(round(best_bin['cadence_bin'], 1)),
        "performance_score": float(round(best_bin['Performance_Score'], 3)),
        "exertion_score": None,
        "details": binned_df.where(pd.notnull(binned_df), None).to_dict(orient='records'),
    }
    logger.info(f"Optimal cadence found: {result['optimal_cadence']} rpm")
    return result

