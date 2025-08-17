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
if not logger.handlers:
    logger.addHandler(handler)
else:
    # Avoid duplicate handlers in some reload environments
    logger.handlers = [handler]

router = APIRouter()


@router.get("/activities")
def list_activities(session_id: str):
    """Return a small list of recent 'Ride' activities for the authenticated athlete."""
    logger.info(f"Listing activities for session_id={session_id}")

    token_data = SESSIONS.get(session_id)
    if not token_data:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=401, detail="Invalid session ID")

    access_token = token_data["access_token"]
    url = "https://www.strava.com/api/v3/athlete/activities"
    logger.debug(f"Fetching activities from Strava API: {url}")

    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"per_page": 50}
    )

    if response.status_code != 200:
        logger.error(f"Failed to fetch activities: {response.status_code} {response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch activities")

    activities = response.json()
    logger.info(f"Fetched {len(activities)} activities from Strava")

    # Filter to Rides and return compact dicts (backwards-compatible with previous frontend)
    rides = [
        {
            "id": a.get("id"),
            "name": a.get("name"),
            "distance": a.get("distance"),
            "start_date": a.get("start_date")
        }
        for a in activities
        if a.get("type") == "Ride"
    ][:30]

    logger.info(f"Returning {len(rides)} bike rides")
    return JSONResponse(rides)


@router.get("/compute")
def compute_activity(session_id: str, activity_id: str):
    """
    Fetch streams for the activity from Strava, normalize streams, and
    pass to process_activity_stream wrapper which returns a JSON-serializable dict.
    """
    logger.info(f"Computing activity {activity_id} for session_id={session_id}")

    token_data = SESSIONS.get(session_id)
    if not token_data:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=401, detail="Invalid session ID")

    access_token = token_data["access_token"]
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    stream_types = ",".join([
        "time", "cadence", "velocity_smooth", "distance",
        "altitude", "grade_smooth", "latlng", "moving"
    ])

    logger.debug(f"Fetching streams from Strava API: {url} for types {stream_types}")
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"keys": stream_types, "key_by_type": "true", "resolution": "high"}
    )

    if response.status_code != 200:
        logger.error(f"Failed to fetch stream data: {response.status_code} {response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch stream data")

    stream_data_raw = response.json()
    logger.info("Stream data fetched successfully")
    logger.debug(f"Raw stream keys: {list(stream_data_raw.keys())}")

    # Normalize to dict-of-lists expected by compute.prepare_data
    normalized = {}
    lengths = {}
    for k, v in stream_data_raw.items():
        if isinstance(v, dict) and 'data' in v:
            normalized[k] = v['data'] or []
        else:
            # if Strava returned a list directly (rare), use it
            normalized[k] = v or []
        try:
            lengths[k] = len(normalized[k])
        except Exception:
            lengths[k] = "unknown"

    logger.debug(f"Detected stream lengths per key: {lengths}, max_len={max([l for l in lengths.values() if isinstance(l, int)], default=0)}")

    try:
        result = process_activity_stream(normalized)
    except Exception as e:
        logger.exception("Error running process_activity_stream")
        raise HTTPException(status_code=500, detail=f"Computation failed: {e}")

    # result is already JSON-serializable: contains list "segments" and dict "optimal_cadence"
    seg_count = len(result.get("segments", []))
    opt = result.get("optimal_cadence")
    logger.info(f"Computation finished: segments={seg_count}, optimal={opt}")
    return JSONResponse(result)
