import pandas as pd
import numpy as np
import logging
from . import config
from .utils import lag_column, compute_diff, filter_rows

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to INFO to reduce verbosity
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def _coerce_series_to_bool(s: pd.Series) -> pd.Series:
    """
    Take a Series that may contain booleans, 0/1, 'True'/'False', None, or other objects
    and return a boolean mask of the same length.
    """
    # If it's already boolean dtype, return as-is
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    # Try numeric coercion (0/1)
    num = pd.to_numeric(s, errors="coerce")
    if not num.isna().all():
        return num.fillna(0).astype(bool)

    # Strings like 'true' / 'false'
    lower = s.astype(str).str.lower()
    mask = lower.isin(["true", "1", "t", "y", "yes"])
    # For anything else, treat non-empty values as True
    mask = mask | (~s.isna() & (s.astype(str).str.strip() != ""))
    return mask.fillna(False)


def prepare_data(stream_data) -> pd.DataFrame:
    """
    Convert stream data (dict-of-lists or a DataFrame-like object) into a usable DataFrame.
    Robust to:
      - dict where values are lists
      - DataFrame where each cell might be a list (single-row with list cells)
      - dict where values are {'data': [...]}
    Normalizes column names to lowercase expected by downstream functions.
    Returns a DataFrame with numeric distance/altitude and boolean moving column.
    """
    logger.info("Preparing data: start")

    # Accept both dict-of-lists and DataFrame inputs
    if isinstance(stream_data, pd.DataFrame):
        df = stream_data.copy()
        logger.debug("Input is a pandas DataFrame")
    else:
        # If stream_data is dict-of-objects, extract 'data' if present
        normalized = {}
        for k, v in stream_data.items():
            if isinstance(v, dict) and "data" in v:
                normalized[k] = v["data"]
            else:
                normalized[k] = v
        df = pd.DataFrame(normalized)
        logger.debug("Built DataFrame from dict-like streams")

    logger.info(f"Initial rows (built DF): {len(df)} columns: {list(df.columns)}")

    # If the DataFrame has single row where each cell is a list, expand it:
    if len(df) == 1 and any(isinstance(x, (list, tuple, np.ndarray, pd.Series)) for x in df.iloc[0].values):
        logger.debug("Detected single-row DataFrame with list-like cells; expanding into rows")
        row = df.iloc[0].to_dict()
        try:
            df = pd.DataFrame({k: (v if v is not None else []) for k, v in row.items()})
            logger.info(f"Expanded to rows; new length: {len(df)}")
        except Exception as exc:
            logger.exception("Failed expanding single-row list-like DataFrame; continuing with original df")

    # Lowercase column names for consistency
    df.columns = [str(c).lower() for c in df.columns]

    # Map velocity_smooth -> speed if necessary
    if "velocity_smooth" in df.columns and "speed" not in df.columns:
        df["speed"] = df["velocity_smooth"]

    # Ensure required columns exist
    required = ["time", "distance", "altitude", "moving", "cadence", "speed"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    logger.info(f"Columns present after normalization: {list(df.columns)}")
    logger.info(f"Initial row count before filtering: {len(df)}")

    # Coerce types: numeric for distance/altitude/speed/cadence
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce")
    df["cadence"] = pd.to_numeric(df["cadence"], errors="coerce")
    df["speed"] = pd.to_numeric(df["speed"], errors="coerce")

    # Normalize moving column to boolean mask
    try:
        moving_mask = _coerce_series_to_bool(df["moving"])
    except Exception as exc:
        logger.exception("Failed to coerce 'moving' column to boolean mask; defaulting to True for non-null")
        moving_mask = ~df["moving"].isna()

    # Log counts before filtering
    total_rows = len(df)
    moving_count = int(moving_mask.sum())
    logger.info(f"Filtering moving rows: {moving_count} of {total_rows} flagged as moving")
    # Filter moving rows (but only if mask produced at least one True)
    if moving_count == 0:
        logger.warning("No rows marked as moving. Proceeding without moving filter.")
    else:
        df = df.loc[moving_mask].copy()
        logger.info(f"Remaining rows after filtering moving==True: {len(df)}")

    # Compute differences (distance and altitude)
    logger.info("Computing distance_diff and altitude_diff")
    # compute_diff expects df and a column name; use names as used here
    df["distance_diff"] = compute_diff(df, "distance")
    df["altitude_diff"] = compute_diff(df, "altitude")

    # Filter out tiny distance diffs (<=1m)
    before = len(df)
    df = df[df["distance_diff"] > 1].copy()
    after = len(df)
    logger.info(f"Remaining rows after distance_diff > 1: {after} (was {before})")

    return df


def calculate_gradient(df: pd.DataFrame, window: int = config.GRADIENT_WINDOW) -> pd.DataFrame:
    logger.info(f"Calculating gradient with window: {window}")
    if len(df) == 0:
        logger.warning("calculate_gradient called with empty DataFrame")
        df["gradient_raw"] = pd.Series(dtype=float)
        df["MA_gradient_raw"] = pd.Series(dtype=float)
        return df

    df["gradient_raw"] = (df["altitude_diff"] / df["distance_diff"]) * 100
    invalid_gradients = df[(df["gradient_raw"] > 30) | (df["gradient_raw"] < -30)].shape[0]
    if invalid_gradients > 0:
        logger.info(f"Setting {invalid_gradients} extreme gradients to NaN")
    df.loc[(df["gradient_raw"] > 30) | (df["gradient_raw"] < -30), "gradient_raw"] = np.nan

    df["MA_gradient_raw"] = df["gradient_raw"].rolling(window=window, center=True, min_periods=1).mean()
    df["MA_gradient_raw"] = df["MA_gradient_raw"].interpolate(method="linear")
    logger.info("Gradient calculation complete")
    return df


def segment_ride(df: pd.DataFrame, time_limit: float = config.TIME_LIMIT_SEC, grad_limit: float = config.GRAD_LIMIT_PCT) -> pd.DataFrame:
    logger.info(f"Segmenting ride with time_limit={time_limit} sec and grad_limit={grad_limit}%")

    if len(df) == 0:
        logger.warning("segment_ride called with empty DataFrame")
        df["segment_id"] = pd.Series(dtype=int)
        df["segment_start"] = pd.Series(dtype=bool)
        df["time_since_segment"] = pd.Series(dtype=float)
        df["grad_baseline"] = pd.Series(dtype=float)
        return df

    segment_id = 0
    time_since_segment = 0.0
    grad_base = 0.0
    last_time = None

    segment_ids, segment_starts, time_since_segments, grad_bases = [], [], [], []

    for idx, row in df.iterrows():
        t, grad = row.get("time"), row.get("ma_gradient_raw") if "ma_gradient_raw" in row else row.get("MA_gradient_raw", np.nan)

        # robust delta-time computation
        if last_time is None or pd.isna(t) or (not pd.isna(last_time) and t < last_time):
            td = 0.0
        else:
            td = float(t - last_time)

        if not pd.isna(t):
            last_time = t

        if idx == 0:
            segment_id = 1
            grad_base = grad if not pd.isna(grad) else 0.0
            is_start = True
        else:
            time_since_segment += td
            time_trigger = time_since_segment >= time_limit
            grad_trigger = (not pd.isna(grad)) and (abs(grad - grad_base) >= grad_limit)

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
            logger.debug(f"Row {idx}: segment_id={segment_id}, time_since_segment={time_since_segment}, grad_base={grad_base}")

    logger.info(f"Segmentation complete: total segments={segment_id}")
    df["segment_id"] = segment_ids
    df["segment_start"] = segment_starts
    df["time_since_segment"] = time_since_segments
    df["grad_baseline"] = grad_bases
    return df


def calculate_cadence_elevation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating cadence and elevation gain")
    if len(df) == 0:
        logger.warning("calculate_cadence_elevation called with empty DataFrame")
        return df
    df = df.copy()
    df["cadence_nonzero"] = df["cadence"].apply(lambda x: x if pd.notna(x) and x > 0 else float("nan"))
    df["elev_gain"] = df["altitude_diff"].apply(lambda x: x if pd.notna(x) and x > 0 else 0)
    logger.info("Cadence and elevation calculation complete")
    return df


def aggregate_segments(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Aggregating segments")
    if "segment_id" not in df.columns:
        logger.error("aggregate_segments called but 'segment_id' missing from DataFrame")
        return pd.DataFrame()
    agg_df = df.groupby("segment_id").agg(
        cadence_nonzero_mean=("cadence_nonzero", "mean"),
        speed_mean=("speed", "mean"),
        elev_gain_mean=("elev_gain", "mean"),
        distance_sum=("distance_diff", "sum"),
        gradient_mean=("MA_gradient_raw", "mean"),
    ).reset_index()
    logger.info(f"Aggregation complete: {len(agg_df)} segments")
    return agg_df


def compute_scores(df: pd.DataFrame, a=config.EXERTION_A, b=config.EXERTION_B, c=config.EXERTION_C) -> pd.DataFrame:
    logger.info("Computing performance and exertion scores")
    if df.empty:
        logger.warning("compute_scores called with empty DataFrame")
        return df
    df = df.copy()
    # Avoid division by zero by replacing zero speeds with NaN
    df["speed_mean"] = df["speed_mean"].replace({0: np.nan})
    df["Exertion_Score"] = ((df["elev_gain_mean"] * df["gradient_mean"] ** a) + (df["cadence_nonzero_mean"] ** b)) / (df["speed_mean"] ** c)
    df["Performance_Score"] = df["speed_mean"] / df["Exertion_Score"]
    logger.info("Score computation complete")
    return df


def cadence_binning(df: pd.DataFrame, bin_size=config.BIN_SIZE) -> pd.DataFrame:
    """
    Create cadence bins from aggregated segment rows.

    Returns a DataFrame with:
      - cadence_bin: bin floor (e.g. 60.0)
      - Performance_Score: mean performance score inside the bin
      - mean_cadence: mean cadence inside the bin
      - count: how many segments contributed to this bin
    """
    logger.info("Binning cadence values")
    if df.empty:
        logger.info("cadence_binning received empty df")
        return pd.DataFrame(columns=["cadence_bin", "Performance_Score", "mean_cadence", "count"])

    df = df.copy()
    # floor to nearest bin
    df['cadence_bin'] = (df['cadence_nonzero_mean'] // bin_size) * bin_size

    bin_df = (
        df.groupby('cadence_bin', as_index=False)
          .agg(
              Performance_Score=('Performance_Score', 'mean'),
              mean_cadence=('cadence_nonzero_mean', 'mean'),
              count=('cadence_nonzero_mean', 'size')
          )
          .sort_values('cadence_bin')
    )

    # Make sure types are native python floats/ints for JSON safety later
    bin_df['Performance_Score'] = bin_df['Performance_Score'].astype(float)
    bin_df['mean_cadence'] = bin_df['mean_cadence'].astype(float)
    bin_df['count'] = bin_df['count'].astype(int)

    logger.debug(f"Cadence bins:\n{bin_df.to_dict(orient='records')}")
    logger.info("Cadence binning complete")
    return bin_df


def optimal_cadence(df: pd.DataFrame) -> dict:
    """
    Compute the optimal cadence. Apply sensible filters:
      - bins must have mean_cadence >= config.CADENCE_MIN (if set)
      - bins must have at least config.MIN_BIN_COUNT segments (optional; default 1)
    If no bins survive filtering, fall back to the highest scoring bin with a warning.
    """
    logger.info("Computing optimal cadence from aggregated segments")

    binned_df = cadence_binning(df)

    if binned_df.empty:
        logger.warning("No binned data available for optimal cadence computation")
        return {
            "optimal_cadence": None,
            "performance_score": None,
            "exertion_score": None,
            "details": []
        }

    # Config-driven filters (safe defaults)
    min_cad = getattr(config, "CADENCE_MIN", 0) or 0
    min_count = getattr(config, "MIN_BIN_COUNT", 1) or 1

    logger.info(f"Filtering bins: mean_cadence >= {min_cad}, count >= {min_count}")
    filtered = binned_df[(binned_df['mean_cadence'] >= float(min_cad)) & (binned_df['count'] >= int(min_count))]

    if filtered.empty:
        logger.warning("No cadence bins passed filters; falling back to unfiltered best bin")
        chosen_row = binned_df.loc[binned_df['Performance_Score'].idxmax()]
    else:
        chosen_row = filtered.loc[filtered['Performance_Score'].idxmax()]

    result = {
        "optimal_cadence": float(chosen_row['mean_cadence']),
        "performance_score": float(chosen_row['Performance_Score']),
        "exertion_score": None,
        "details": binned_df.to_dict(orient="records")
    }

    logger.info(f"Optimal cadence selected: {result['optimal_cadence']} rpm (perf={result['performance_score']}, count_in_bin={int(chosen_row['count'])})")
    return result