# compute/optcad_compute.py
import logging
from typing import Any, Dict, List

import pandas as pd

from .core import (
    prepare_data,
    calculate_gradient,
    segment_ride,
    calculate_cadence_elevation,
    aggregate_segments,
    compute_scores,
    optimal_cadence,
    cadence_binning,
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
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def _df_to_records_safe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of serializable dicts (convert numpy types)."""
    if df is None or df.empty:
        return []
    # Use pandas json-friendly conversion via to_dict then cast any non-serializable types
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    return records


def _safe_opt_cad_serializable(opt: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure optimal cadence structure is JSON-serializable."""
    if not isinstance(opt, dict):
        return {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []}
    details = opt.get("details", [])
    if isinstance(details, pd.DataFrame):
        details = details.where(pd.notnull(details), None).to_dict(orient="records")
    # ensure basic keys exist
    return {
        "optimal_cadence": opt.get("optimal_cadence"),
        "performance_score": opt.get("performance_score"),
        "exertion_score": opt.get("exertion_score"),
        "details": details or [],
    }


def process_activity_stream(streams: dict) -> dict:
    """
    Robust wrapper for computing OptCad results from raw Strava streams.

    Returns a JSON-serializable dict:
      - segments: list of per-segment aggregated dicts (may be empty)
      - optimal_cadence: dict with optimal cadence info (or None fields)
      - message: human-readable status
    """
    logger.info("Starting process_activity_stream")

    # Build DataFrame from incoming streams dict (tolerant to missing keys)
    df = pd.DataFrame({
        "time": streams.get("time", []),
        "cadence": streams.get("cadence", []),
        "speed": streams.get("velocity_smooth", []),
        "distance": streams.get("distance", []),
        "altitude": streams.get("altitude", []),
        "moving": streams.get("moving", []),
    })

    logger.debug("Initial stream lengths: " + ", ".join(
        f"{k}={len(df[k])}" for k in df.columns if k in df
    ))
    logger.info(f"Initial rows (built DF): {len(df)}")

    # 1) Prepare data
    try:
        df_prepared = prepare_data(df)
    except Exception as e:
        logger.exception("prepare_data raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": f"prepare_data failed: {e}",
        }

    if df_prepared is None or df_prepared.empty:
        msg = "Insufficient data after prepare_data (no moving rows or filtered out)."
        logger.warning(msg)
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": msg,
        }

    logger.info(f"Rows after prepare_data: {len(df_prepared)}")
    logger.debug(f"Columns after prepare_data: {df_prepared.columns.tolist()}")

    # 2) Calculate gradient (smoothing)
    try:
        df_grad = calculate_gradient(df_prepared, window=GRADIENT_WINDOW)
    except Exception as e:
        logger.exception("calculate_gradient raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": f"calculate_gradient failed: {e}",
        }

    logger.info(f"Rows after calculate_gradient: {len(df_grad)}")
    if "MA_gradient_raw" in df_grad.columns:
        valid_grads = df_grad["MA_gradient_raw"].dropna()
        logger.info(f"Gradient stats: count={len(valid_grads)}, min={valid_grads.min() if len(valid_grads) else None}, max={valid_grads.max() if len(valid_grads) else None}")

    # 3) Segment ride
    try:
        df_seg = segment_ride(df_grad, time_limit=TIME_LIMIT_SEC, grad_limit=GRAD_LIMIT_PCT)
    except Exception as e:
        logger.exception("segment_ride raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": f"segment_ride failed: {e}",
        }

    if df_seg is None:
        msg = "segment_ride returned None"
        logger.error(msg)
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": msg,
        }

    logger.info(f"Rows after segmentation: {len(df_seg)}")
    if "segment_id" in df_seg.columns:
        n_segments = int(df_seg["segment_id"].nunique())
        logger.info(f"Unique segments created: {n_segments}")
    else:
        logger.warning("segment_id column missing after segmentation")
        n_segments = 0

    if n_segments == 0:
        msg = "No segments created (segment_id missing or zero unique segments)."
        logger.warning(msg)
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": msg,
        }

    # 4) Calculate cadence & elevation derived columns
    try:
        df_ce = calculate_cadence_elevation(df_seg)
    except Exception as e:
        logger.exception("calculate_cadence_elevation raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": f"calculate_cadence_elevation failed: {e}",
        }

    logger.info(f"Rows after cadence/elevation derivation: {len(df_ce)}")
    if "cadence_nonzero" in df_ce.columns:
        logger.info(f"cadence_nonzero count non-null: {df_ce['cadence_nonzero'].notnull().sum()}")

    # 5) Aggregate per segment
    try:
        agg_df = aggregate_segments(df_ce)
    except Exception as e:
        logger.exception("aggregate_segments raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": f"aggregate_segments failed: {e}",
        }

    logger.info(f"Segments aggregated: {len(agg_df)}")
    logger.debug(f"Aggregated columns: {agg_df.columns.tolist()}")

    if agg_df is None or agg_df.empty:
        msg = "No aggregated segments (agg_df empty)."
        logger.warning(msg)
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": msg,
        }

    # 6) Compute scores
    try:
        scored_df = compute_scores(agg_df, a=EXERTION_A, b=EXERTION_B, c=EXERTION_C)
    except Exception as e:
        logger.exception("compute_scores raised an exception")
        return {
            "segments": [],
            "optimal_cadence": {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []},
            "message": f"compute_scores failed: {e}",
        }

    logger.info("Score computation complete")
    logger.debug(f"Score columns: {scored_df.columns.tolist()}")

    # 7) Optimal cadence
    try:
        opt_cad = optimal_cadence(scored_df)
    except Exception as e:
        logger.exception("optimal_cadence raised an exception")
        opt_cad = {"optimal_cadence": None, "performance_score": None, "exertion_score": None, "details": []}

    opt_cad_safe = _safe_opt_cad_serializable(opt_cad)

    # Convert segment DataFrame to serializable list of dicts
    segments = _df_to_records_safe(scored_df)

    logger.info("process_activity_stream completed successfully")
    return {
        "segments": segments,
        "optimal_cadence": opt_cad_safe,
        "message": "Computation completed",
    }
