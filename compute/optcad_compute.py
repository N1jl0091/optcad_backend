# compute/optcad_compute.py
from .core import (
    prepare_data,
    calculate_gradient,
    segment_ride,
    calculate_cadence_elevation,
    aggregate_segments,
    compute_scores,
    optimal_cadence
)
from .config import TIME_LIMIT_SEC, GRAD_LIMIT_PCT, GRADIENT_WINDOW, BIN_SIZE, EXERTION_A, EXERTION_B, EXERTION_C
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def process_activity_stream(streams: dict) -> dict:
    """
    Robust wrapper for computing OptCad results from raw Strava streams.
    Always returns JSON-serializable dict with keys:
      - segments: list (possibly empty)
      - optimal_cadence: dict with optimal_cadence and details (or None)
      - message: human-friendly string explaining the result (optional)
    """
    # Convert streams to DataFrame (keep original keys/casing consistent with core.prepare_data)
    df = pd.DataFrame({
        'time': streams.get('time', []),
        'cadence': streams.get('cadence', []),
        'speed': streams.get('velocity_smooth', []),
        'distance': streams.get('distance', []),
        'altitude': streams.get('altitude', []),
        'moving': streams.get('moving', [])
    })

    logger.debug(f"Initial stream lengths: time={len(df['time']) if 'time' in df else 0}, cadence={len(df['cadence']) if 'cadence' in df else 0}")

    # 1. Prepare data
    df = prepare_data(df)

    # If prepare_data removed everything, return early
    if df.empty:
        msg = "Insufficient moving data after filtering (no rows)."
        logger.warning(msg)
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": msg
        }

    # 2. Calculate gradient
    df = calculate_gradient(df, window=GRADIENT_WINDOW)

    # 3. Segment ride
    df = segment_ride(df, time_limit=TIME_LIMIT_SEC, grad_limit=GRAD_LIMIT_PCT)

    # If segmentation produced 0 rows (shouldn't happen if df not empty), guard:
    if 'segment_id' not in df.columns or df['segment_id'].nunique() == 0:
        msg = "No segments created (insufficient continuity or data)."
        logger.warning(msg)
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": msg
        }

    # 4. Compute cadence and elevation
    df = calculate_cadence_elevation(df)

    # 5. Aggregate segments
    agg_df = aggregate_segments(df)

    if agg_df.empty:
        msg = "No segments passed aggregation filters (e.g., cadence_min)."
        logger.warning(msg)
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": msg
        }

    # 6. Compute scores
    agg_df = compute_scores(agg_df, a=EXERTION_A, b=EXERTION_B, c=EXERTION_C)

    # 7. Compute optimal cadence
    opt_cad = optimal_cadence(agg_df)

    # Convert segment DataFrame to list of dicts
    segments = agg_df.to_dict(orient='records')

    # Ensure opt_cad.details is JSON-serializable (list of dicts)
    if isinstance(opt_cad.get("details"), pd.DataFrame):
        opt_cad["details"] = opt_cad["details"].to_dict(orient='records')

    return {
        "segments": segments,
        "optimal_cadence": opt_cad,
        "message": "Computation completed"
    }
