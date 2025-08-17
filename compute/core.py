import pandas as pd
import numpy as np
import logging
from . import config
from .utils import lag_column, compute_diff, filter_rows

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to INFO or DEBUG as needed
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def prepare_data(stream_data: dict) -> pd.DataFrame:
    """
    Convert stream data from Strava API into a proper DataFrame and prepare it for computation.
    Filters out non-moving rows, computes diffs, and ensures numeric types.
    """
    logger.info("Preparing data: converting stream_data to DataFrame")
    df = pd.DataFrame(stream_data)

    # Ensure all required columns exist
    required_columns = ['time', 'distance', 'altitude', 'moving', 'cadence', 'speed']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan  # Fill missing columns with NaN

    # Convert types
    df['moving'] = df['moving'].astype(bool)
    df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
    df['altitude'] = pd.to_numeric(df['altitude'], errors='coerce')
    df['cadence'] = pd.to_numeric(df['cadence'], errors='coerce')
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')

    # Filter moving rows
    logger.info("Filtering moving rows")
    df = df[df['moving']].copy()
    logger.info(f"Remaining rows after filtering: {len(df)}")

    # Compute differences
    logger.info("Computing distance differences")
    df['distance_diff'] = compute_diff(df, 'distance')
    logger.info("Computing altitude differences")
    df['altitude_diff'] = compute_diff(df, 'altitude')

    # Filter out small distance differences
    df = df[df['distance_diff'] > 1].copy()
    logger.info(f"Remaining rows after distance diff filter: {len(df)}")

    return df



def calculate_gradient(df: pd.DataFrame, window: int = config.GRADIENT_WINDOW) -> pd.DataFrame:
    logger.info(f"Calculating gradient with window: {window}")
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

    segment_id = 0
    time_since_segment = 0.0
    grad_base = 0.0
    last_time = None

    segment_ids, segment_starts, time_since_segments, grad_bases = [], [], [], []

    for idx, row in df.iterrows():
        t, grad = row['time'], row['MA_gradient_raw']
        td = 0.0 if last_time is None or pd.isna(t) or t < last_time else t - last_time
        last_time = t if not pd.isna(t) else last_time

        if idx == 0:
            segment_id = 1
            grad_base = grad if not pd.isna(grad) else 0.0
            is_start = True
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

        if idx % 1000 == 0:
            logger.debug(
                f"Row {idx}: segment_id={segment_id}, time_since_segment={time_since_segment}, grad_base={grad_base}"
            )

    logger.info(f"Segmentation complete: total segments={segment_id}")
    df['segment_id'] = segment_ids
    df['segment_start'] = segment_starts
    df['time_since_segment'] = time_since_segments
    df['grad_baseline'] = grad_bases
    return df


def calculate_cadence_elevation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating cadence and elevation gain")
    df = df.copy()
    df['cadence_nonzero'] = df['cadence'].apply(lambda x: x if x > 0 else float('nan'))
    df['elev_gain'] = df['altitude_diff'].apply(lambda x: x if x > 0 else 0)
    logger.info("Cadence and elevation calculation complete")
    return df


def aggregate_segments(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating segments")
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
    df = df.copy()
    df['Exertion_Score'] = (
        (df['elev_gain_mean'] * df['gradient_mean'] ** a) + (df['cadence_nonzero_mean'] ** b)
    ) / (df['speed_mean'] ** c)
    df['Performance_Score'] = df['speed_mean'] / df['Exertion_Score']
    logger.info("Score computation complete")
    return df


def cadence_binning(df: pd.DataFrame, bin_size=config.BIN_SIZE) -> pd.DataFrame:
    logger.info("Binning cadence values")
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
    """
    Compute the optimal cadence given an aggregated segments DataFrame.
    """
    logger.info("Computing optimal cadence from aggregated segments")

    # Bin cadence
    binned_df = cadence_binning(df)

    if binned_df.empty:
        logger.warning("No data available for optimal cadence computation")
        return {
            "optimal_cadence": None,
            "performance_score": None,
            "exertion_score": None,
            "details": []
        }

    # Pick cadence bin with highest performance score
    best_bin = binned_df.loc[binned_df['Performance_Score'].idxmax()]

    result = {
        "optimal_cadence": round(best_bin['cadence_bin'], 1),
        "performance_score": round(best_bin['Performance_Score'], 3),
        "exertion_score": None,  # not in bin_df, so left out or we can add if needed
        "details": binned_df.to_dict(orient="records")
    }

    logger.info(f"Optimal cadence found: {result['optimal_cadence']} rpm")
    return result
