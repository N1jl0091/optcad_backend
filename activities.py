# activities.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import requests
import logging
from config import SESSIONS
from compute.optcad_compute import process_activity_stream

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
# avoid duplicate handlers in reload environments
if not logger.handlers:
    logger.addHandler(handler)
else:
    logger.handlers = [handler]

router = APIRouter()


@router.get("/activities")
def list_activities(session_id: str):
    """Return a compact list of recent 'Ride' activities for the authenticated athlete."""
    logger.info("list_activities called")
    logger.debug("session_id=%s", session_id)

    token_data = SESSIONS.get(session_id)
    if not token_data:
        logger.warning("Invalid session ID provided to /activities: %s", session_id)
        raise HTTPException(status_code=401, detail="Invalid session ID")

    access_token = token_data.get("access_token")
    if not access_token:
        logger.error("Session exists but access_token missing for session: %s", session_id)
        raise HTTPException(status_code=401, detail="Invalid session data")

    url = "https://www.strava.com/api/v3/athlete/activities"
    logger.debug("Fetching activities from Strava API: %s", url)

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"per_page": 50},
            timeout=15,
        )
    except Exception as e:
        logger.exception("HTTP request to Strava failed")
        raise HTTPException(status_code=500, detail=f"Failed to fetch activities: {e}")

    if response.status_code != 200:
        logger.error("Failed to fetch activities: %s %s", response.status_code, response.text)
        raise HTTPException(status_code=500, detail="Failed to fetch activities from Strava")

    activities = response.json()
    logger.info("Fetched %d activities from Strava", len(activities))

    # Keep compatibility with previous frontend: return only rides with compact fields
    rides = [
        {
            "id": a.get("id"),
            "name": a.get("name"),
            "distance": a.get("distance"),
            "start_date": a.get("start_date"),
            "type": a.get("type"),
        }
        for a in activities
        if a.get("type") == "Ride"
    ][:30]

    logger.info("Returning %d bike rides", len(rides))
    return JSONResponse(rides)


@router.get("/compute")
def compute_activity(session_id: str, activity_id: str):
    """
    Fetch streams for the activity from Strava, normalize them into dict-of-lists,
    and pass to process_activity_stream which returns a JSON-serializable dict.
    """
    logger.info("compute_activity called")
    logger.debug("session_id=%s activity_id=%s", session_id, activity_id)

    token_data = SESSIONS.get(session_id)
    if not token_data:
        logger.warning("Invalid session ID provided to /compute: %s", session_id)
        raise HTTPException(status_code=401, detail="Invalid session ID")

    access_token = token_data.get("access_token")
    if not access_token:
        logger.error("Session exists but access_token missing for session: %s", session_id)
        raise HTTPException(status_code=401, detail="Invalid session data")

    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    stream_types = ",".join([
        "time", "cadence", "velocity_smooth", "distance",
        "altitude", "grade_smooth", "latlng", "moving"
    ])

    logger.debug("Fetching streams from Strava API: %s types=%s", url, stream_types)
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"keys": stream_types, "key_by_type": "true", "resolution": "high"},
            timeout=20,
        )
    except Exception as e:
        logger.exception("HTTP request to Strava streams endpoint failed")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stream data: {e}")

    if response.status_code != 200:
        logger.error("Failed to fetch stream data: %s %s", response.status_code, response.text)
        raise HTTPException(status_code=500, detail="Failed to fetch stream data from Strava")

    stream_data_raw = response.json()
    logger.info("Stream data fetched successfully")
    logger.debug("Raw stream keys: %s", list(stream_data_raw.keys()))

    # Normalize to dict-of-lists expected by compute.prepare_data / process_activity_stream
    normalized = {}
    lengths = {}
    for k, v in stream_data_raw.items():
        if isinstance(v, dict) and 'data' in v:
            normalized[k] = v.get('data') or []
        else:
            normalized[k] = v or []
        try:
            lengths[k] = len(normalized[k])
        except Exception:
            lengths[k] = "unknown"

    max_len = max([l for l in lengths.values() if isinstance(l, int)], default=0)
    logger.debug("Detected stream lengths per key: %s, max_len=%s", lengths, max_len)

    # Defensive: if no time/distance data present, return helpful error
    if max_len == 0:
        logger.error("No stream data arrays found for activity %s", activity_id)
        raise HTTPException(status_code=500, detail="No stream data available for activity")

    # Ensure all expected keys exist in normalized dict (fill with empty lists)
    expected_keys = ["time", "cadence", "velocity_smooth", "distance", "altitude", "grade_smooth", "latlng", "moving"]
    for key in expected_keys:
        if key not in normalized:
            normalized[key] = []

    # Call the compute wrapper (should return JSON-serializable python types)
    try:
        result = process_activity_stream(normalized)
    except Exception as e:
        logger.exception("Error running process_activity_stream for activity %s", activity_id)
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")

    # result expected shape: {"segments": [...], "optimal_cadence": {...}, "message": "..."}
    seg_count = len(result.get("segments", [])) if isinstance(result.get("segments"), list) else "unknown"
    opt = result.get("optimal_cadence")
    logger.info("Computation finished: segments=%s, optimal=%s", seg_count, opt)

    return JSONResponse(result)
