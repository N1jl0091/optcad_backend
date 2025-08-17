# compute/core.py
import pandas as pd
import numpy as np
import logging
from . import config
from .utils import lag_column, compute_diff, filter_rows

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Change to INFO for less verbosity in prod
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def prepare_data(stream_data: dict) -> pd.DataFrame:
    """
    Convert stream data (dict of lists) into DataFrame and prepare it for computation.
    """
    logger.info("Preparing data: extracting stream arrays")
    # safe copy/normalization to ensure keys exist as lists
    # Build dictionary where missing keys are filled with empty lists of max length
    lengths = {}
    for k, v in stream_data.items():
        try:
            lengths[k] = len(v)
        except Exception:
            lengths[k] = 0
    max_len = max([l for l in lengths.values()] + [0])
    logger.debug(f"Detected stream lengths per key: {lengths}, max_len={max_len}")

    # Build dataframe column-wise; if a column is scalar or shorter, pad with NaN
    data = {}
    for k in ['time', 'cadence', 'velocity_smooth', 'distance', 'altitude', 'grade_smooth', 'latlng', 'moving']:
        vals = stream_data.get(k, [])
        # If 'latlng' is list of [lat,lng], convert to two columns later; here keep as-is
        if vals is None:
            vals = []
        # Pad/truncate to max_len
        if isinstance(vals, list):
            if len(vals) < max_len:
                vals = vals + [np.nan] * (max_len - len(vals))
            else:
                vals = vals[:max_len]
        else:
            # not a list -> replicate or build list
            vals = [vals] * max_len
        data[k] = vals

    df = pd.DataFrame({
        'time': data.get('time', []),
        'cadence': data.get('cadence', []),
        'speed': data.get('velocity_smooth', []),
        'Distance': data.get('distance', []),
        'Altitude': data.get('altitude', []),
        'moving': data.get('moving', [])
    })

    logger.info(f"Initial rows (after building DF): {len(df)}")

    # Ensure types
    df['moving'] = df['moving'].astype(bool, errors='ignore')
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    df['Altitude'] = pd.to_numeric(df['Altitude'], errors='coerce')
    df['cadence'] = pd.to_numeric(df['cadence'], errors='coerce')
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

    # 1. Filter moving rows
    logger.info("Filtering moving rows")
    before = len(df)
    try:
        df = df[df['moving'] == True].copy()
    except Exception:
        # fallback if 'moving' has weird values
        df = df[df['moving'].astype(bool, errors='ignore')].copy()
    logger.info(f"Remaining rows after filtering moving==True: {len(df)} (was {before})")

    # 2. diffs
    logger.info("Computing distance_diff and altitude_diff")
    df['distance_diff'] = compute_diff(df, 'Distance')
    df['altitude_diff'] = compute_diff(df, 'Altitude')

    before = len(df)
    df = df[df['distance_diff'] > 1].copy()
    logger.info(f"Remaining rows after distance_diff > 1: {len(df)} (was {before})")

    return df


def calculate_gradient(df: pd.DataFrame, window: int = config.GRADIENT_WINDOW) -> pd.DataFrame:
    logger.info(f"Calculating gradient with window: {window}")
    if df.empty:
        logger.warning("Empty DataFrame passed to calculate_gradient")
        df['gradient_raw'] = pd.Series(dtype=float)
        df['MA_gradient_raw'] = pd.Series(dtype=float)
        return df

    df['gradient_raw'] = (df['altitude_diff'] / df['distance_diff']) * 100
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
        logger.info("Empty DataFrame passed to segment_ride")
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
        t = row.get('time', np.nan)
        grad = row.get('MA_gradient_raw', np.nan)
        td = 0.0 if last_time is None or pd.isna(t) or (not pd.isna(last_time) and t < last_time) else (t - last_time if not pd.isna(t) and last_time is not None else 0.0)
        last_time = t if not pd.isna(t) else last_time

        if idx == 0:
            segment_id = 1
            grad_base = grad if not pd.isna(grad) else 0.0
            is_start = True
            time_since_segment = 0.0
        else:
            time_since_segment += td
            time_trigger = time_since_segment >= time_limit
            grad_trigger = not pd.isna(grad) and abs(grad - grad_base) >= grad_limit

            if time_trigger or grad_trigger:
                segment_id += 1
                time_since_segment = 0.0
                grad_base = grad if not pd.isna(grad) else grad_base
                is_start = True
            else:
                is_start = False

        segment_ids.append(segment_id)
        segment_starts.append(is_start)
        time_since_segments.append(time_since_segment)
        grad_bases.append(grad_base)

        if idx % 1000 == 0 and idx > 0:
            logger.debug(f"Row {idx}: segment_id={segment_id}, time_since_segment={time_since_segment}, grad_base={grad_base}")

    logger.info(f"Segmentation complete: total segments={segment_id}")
    df['segment_id'] = segment_ids
    df['segment_start'] = segment_starts
    df['time_since_segment'] = time_since_segments
    df['grad_baseline'] = grad_bases
    return df


def calculate_cadence_elevation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating cadence and elevation gain")
    if df.empty:
        logger.info("Empty DataFrame passed to calculate_cadence_elevation")
        df['cadence_nonzero'] = pd.Series(dtype=float)
        df['elev_gain'] = pd.Series(dtype=float)
        return df

    df = df.copy()
    df['cadence_nonzero'] = df['cadence'].apply(lambda x: x if (pd.notna(x) and x > 0) else float('nan'))
    df['elev_gain'] = df['altitude_diff'].apply(lambda x: x if (pd.notna(x) and x > 0) else 0)
    logger.info("Cadence and elevation calculation complete")
    return df


def aggregate_segments(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating segments")
    if 'segment_id' not in df.columns:
        logger.error("'segment_id' not present in DataFrame at aggregation step")
        raise KeyError('segment_id')

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
        logger.info("Empty DataFrame passed to compute_scores")
        df['Exertion_Score'] = pd.Series(dtype=float)
        df['Performance_Score'] = pd.Series(dtype=float)
        return df

    df = df.copy()
    # guard against zero speed
    df['speed_mean'] = df['speed_mean'].replace(0, np.nan)
    df['Exertion_Score'] = ((df['elev_gain_mean'] * df['gradient_mean'] ** a) + (df['cadence_nonzero_mean'] ** b)) / (df['speed_mean'] ** c)
    df['Performance_Score'] = df['speed_mean'] / df['Exertion_Score']
    logger.info("Score computation complete")
    return df


def cadence_binning(df: pd.DataFrame, bin_size=config.BIN_SIZE) -> pd.DataFrame:
    logger.info("Binning cadence values")
    if df.empty:
        logger.info("Empty DataFrame passed to cadence_binning")
        return pd.DataFrame(columns=['cadence_bin', 'Performance_Score', 'mean_cadence'])
    df = df.copy()
    df['cadence_bin'] = (df['cadence_nonzero_mean'] // bin_size) * bin_size
    bin_df = (df.groupby('cadence_bin', as_index=False)
              .agg({'Performance_Score': 'mean', 'cadence_nonzero_mean': 'mean'})
              .rename(columns={'cadence_nonzero_mean': 'mean_cadence'}))
    logger.info("Cadence binning complete")
    return bin_df


def optimal_cadence(df: pd.DataFrame) -> dict:
    logger.info("Computing optimal cadence from aggregated segments")
    binned_df = cadence_binning(df)
    if binned_df.empty:
        logger.warning("No data available for optimal cadence computation")
        return {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []}

    best_bin = binned_df.loc[binned_df['Performance_Score'].idxmax()]
    result = {
        "optimal_cadence": float(round(best_bin['cadence_bin'], 1)),
        "performance_score": float(round(best_bin['Performance_Score'], 3)),
        "exertion_score": None,
        "details": binned_df.to_dict(orient="records")
    }
    logger.info(f"Optimal cadence found: {result['optimal_cadence']} rpm")
    return result
