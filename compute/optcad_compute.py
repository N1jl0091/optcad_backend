import logging
from typing import Any, Dict
import pandas as pd

from .core import (
    prepare_data,
    calculate_gradient,
    segment_ride,
    calculate_cadence_elevation,
    aggregate_segments,
    compute_scores,
    optimal_cadence,
)
from .config import (
    TIME_LIMIT_SEC,
    GRAD_LIMIT_PCT,
    GRADIENT_WINDOW,
    BIN_SIZE,
    EXERTION_A,
    EXERTION_B,
    EXERTION_C,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _build_dataframe_from_streams(streams: Dict[str, Any]) -> pd.DataFrame:
    """
    Accepts normalized stream dict-of-lists and builds a DataFrame.
    Keeps consistent column names expected by core.prepare_data.
    """
    df = pd.DataFrame({
        "time": streams.get("time", []),
        "cadence": streams.get("cadence", []),
        "speed": streams.get("velocity_smooth", streams.get("speed", [])),
        "distance": streams.get("distance", []),
        "altitude": streams.get("altitude", []),
        "moving": streams.get("moving", []),
    })
    return df


def _to_serializable_opt(opt: dict) -> dict:
    """Ensure optimal cadence result is JSON serializable."""
    # details might already be a list of dicts; if it's a DataFrame convert it
    if isinstance(opt.get("details"), pd.DataFrame):
        opt["details"] = opt["details"].to_dict(orient="records")
    return opt


def process_activity_stream(streams: dict) -> dict:
    """
    Main wrapper that accepts Strava streams (dict of lists OR dict of {'data': [...]})
    and returns a JSON-serializable dict:
      { "segments": [...], "optimal_cadence": {...}, "message": "..." }
    This function is importable as compute.optcad_compute.process_activity_stream
    """
    logger.info("Starting process_activity_stream")
    # Normalize: if values are objects with 'data', extract them (but activities.py already normalizes)
    normalized = {}
    for k, v in streams.items():
        if isinstance(v, dict) and "data" in v:
            normalized[k] = v["data"]
        else:
            normalized[k] = v

    # Build DataFrame used by core.prepare_data (that function expects a DataFrame or dict in some versions)
    try:
        df = _build_dataframe_from_streams(normalized)
        logger.debug(f"Initial rows (built DF): {len(df)}")
    except Exception as exc:
        logger.exception("Failed to build DataFrame from streams")
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": f"Failed to build DataFrame from streams: {exc}"
        }

    # 1. Prepare data
    try:
        df_prepared = prepare_data(df)
    except Exception as exc:
        logger.exception("prepare_data raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": f"prepare_data error: {exc}"
        }

    if df_prepared.empty:
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

    # 2. Gradient
    try:
        df_grad = calculate_gradient(df_prepared, window=GRADIENT_WINDOW)
    except Exception as exc:
        logger.exception("calculate_gradient raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": f"calculate_gradient error: {exc}"
        }

    # 3. Segment
    try:
        df_segmented = segment_ride(df_grad, time_limit=TIME_LIMIT_SEC, grad_limit=GRAD_LIMIT_PCT)
    except Exception as exc:
        logger.exception("segment_ride raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": f"segment_ride error: {exc}"
        }

    if "segment_id" not in df_segmented.columns or df_segmented["segment_id"].nunique() == 0:
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

    # 4. cadence/elevation
    try:
        df_cad = calculate_cadence_elevation(df_segmented)
    except Exception as exc:
        logger.exception("calculate_cadence_elevation raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": f"calculate_cadence_elevation error: {exc}"
        }

    # 5. aggregate
    try:
        agg_df = aggregate_segments(df_cad)
    except Exception as exc:
        logger.exception("aggregate_segments raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": f"aggregate_segments error: {exc}"
        }

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

    # 6. scores
    try:
        scored = compute_scores(agg_df, a=EXERTION_A, b=EXERTION_B, c=EXERTION_C)
    except Exception as exc:
        logger.exception("compute_scores raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {
                "optimal_cadence": None,
                "performance_score": None,
                "exertion_score": None,
                "details": []
            },
            "message": f"compute_scores error: {exc}"
        }

    # 7. optimal cadence
    try:
        opt = optimal_cadence(scored)
        opt = _to_serializable_opt(opt)
    except Exception as exc:
        logger.exception("optimal_cadence raised an exception")
        opt = {
            "optimal_cadence": None,
            "performance_score": None,
            "exertion_score": None,
            "details": []
        }

    # Make segments serializable
    segments = scored.reset_index(drop=True).to_dict(orient="records")

    logger.info(f"process_activity_stream complete: segments={len(segments)} optimal={opt.get('optimal_cadence')}")
    return {"segments": segments, "optimal_cadence": opt, "message": "Computation completed"}
