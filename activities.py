# activities.py — replace /compute endpoint with this implementation
import traceback
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import requests
import logging
from config import SESSIONS
from compute.optcad_compute import process_activity_stream
import pandas as pd

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/compute")
def compute_activity(session_id: str, activity_id: str):
    logger.info(f"Computing activity {activity_id} for session_id={session_id}")

    token_data = SESSIONS.get(session_id)
    if not token_data:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=401, detail="Invalid session ID")

    access_token = token_data.get("access_token")
    if not access_token:
        logger.warning("Session missing access_token")
        raise HTTPException(status_code=401, detail="Invalid session (no token)")

    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    stream_types = ",".join(["time","cadence","velocity_smooth","distance","altitude","grade_smooth","latlng","moving"])

    logger.debug(f"Fetching streams from Strava API: {url} for types {stream_types}")
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"keys": stream_types, "key_by_type": "true", "resolution": "high"},
            timeout=15
        )
    except Exception as e:
        logger.exception("HTTP error when fetching streams")
        raise HTTPException(status_code=502, detail="Failed to fetch streams from Strava")

    if response.status_code != 200:
        logger.error(f"Failed to fetch stream data: {response.status_code} {response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch stream data")

    stream_raw = response.json()
    logger.info("Stream data fetched successfully")
    logger.debug(f"Raw stream keys: {list(stream_raw.keys())}")

    # normalize to dict-of-lists expected by compute
    normalized = {}
    lengths = {}
    for k, v in stream_raw.items():
        if isinstance(v, dict) and "data" in v:
            normalized[k] = v["data"]
        else:
            normalized[k] = v
        try:
            lengths[k] = len(normalized[k]) if normalized[k] is not None else 0
        except Exception:
            lengths[k] = "unknown"
    logger.debug(f"Detected stream lengths per key: {lengths}, max_len={max([l for l in lengths.values() if isinstance(l, int)], default=0)}")

    try:
        # call the wrapper that returns plain python objects
        result = process_activity_stream(normalized)
    except Exception as e:
        # Print full traceback — crucial for debugging hidden threadpool exceptions
        logger.error("Exception in process_activity_stream; full traceback below")
        logger.error(traceback.format_exc())
        # Also raise as HTTPException so frontend gets a controlled error
        raise HTTPException(status_code=500, detail="Internal processing error")

    # Defensive serialization: convert any pandas DataFrame inside `result` to plain dicts
    def _make_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(item) for item in obj]
        return obj

    safe_result = _make_serializable(result)

    logger.info(f"Computation finished: segments={len(safe_result.get('segments', [])) if isinstance(safe_result.get('segments'), list) else 'unknown'}, optimal={safe_result.get('optimal_cadence')}")
    return JSONResponse(safe_result)
