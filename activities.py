# activities.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import requests
import logging
from config import SESSIONS
from compute.optcad_compute import process_activity_stream
from fastapi.responses import Response
import io
import csv

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
def list_activities(session_id: str, activity_name: str | None = None):
    """Return a compact list of recent 'Ride' activities or search by exact activity name."""
    logger.info("list_activities called")
    logger.debug("session_id=%s activity_name=%s", session_id, activity_name)

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

    # If user supplied activity_name, search pages until found (avoid missing older activities)
    if activity_name:
        logger.info("Searching for activity by exact name: %s", activity_name)
        per_page = 100
        max_pages = 5  # search up to 500 activities (tunable)
        found = None
        for page in range(1, max_pages + 1):
            try:
                response = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={"per_page": per_page, "page": page},
                    timeout=15,
                )
            except Exception as e:
                logger.exception("HTTP request to Strava failed while searching")
                raise HTTPException(status_code=500, detail=f"Failed to fetch activities: {e}")

            if response.status_code != 200:
                logger.error("Failed to fetch activities (search): %s %s", response.status_code, response.text)
                raise HTTPException(status_code=500, detail="Failed to fetch activities from Strava")

            activities = response.json()
            logger.debug("Fetched page %s with %d activities", page, len(activities))

            # Look for an exact name match (case sensitive). Change to .lower() if you want case-insensitive.
            for a in activities:
                if a.get("type") == "Ride" and a.get("name") == activity_name:
                    found = a
                    break
            if found:
                break

            # stop early if fewer than per_page items returned (no more pages)
            if len(activities) < per_page:
                break

        if found:
            rides = [{
                "id": found.get("id"),
                "name": found.get("name"),
                "distance": found.get("distance"),
                "start_date": found.get("start_date"),
                "type": found.get("type"),
            }]
            logger.info("Found activity by name; returning 1 ride")
            return JSONResponse(rides)
        else:
            logger.info("Activity not found by name: %s", activity_name)
            return JSONResponse([])  # frontend already expects array

    # --- fallback: list recent rides as before ---
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
    
@router.get("/activity-stream")
def activity_stream(session_id: str, activity_id: str):
    """
    Download CSV of Strava activity streams for given session/activity.
    Returns: CSV file attachment.
    """
    logger.info("CSV request for activity=%s session=%s", activity_id, session_id)

    token_data = SESSIONS.get(session_id)
    if not token_data:
        logger.warning("Invalid session for CSV request: %s", session_id)
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
    logger.debug("Fetching streams for CSV from Strava: %s types=%s", url, stream_types)

    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"keys": stream_types, "key_by_type": "true", "resolution": "high"},
        timeout=20
    )

    if resp.status_code != 200:
        logger.error("Failed to fetch streams for CSV: %s %s", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="Failed to fetch stream data")

    stream_raw = resp.json()

    # Normalize stream arrays: Strava returns {key: {"data":[...]}} or sometimes lists directly
    keys = ["time", "cadence", "velocity_smooth", "distance", "altitude", "grade_smooth", "latlng", "moving"]
    arrays = {}
    for k in keys:
        v = stream_raw.get(k)
        if isinstance(v, dict) and "data" in v:
            arrays[k] = v.get("data") or []
        else:
            arrays[k] = v or []

    max_len = max([len(arr) for arr in arrays.values()] + [0])
    if max_len == 0:
        logger.warning("No stream arrays found to build CSV for activity %s", activity_id)
        raise HTTPException(status_code=500, detail="No stream data available for activity")

    # Build CSV in-memory
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time", "cadence", "speed", "distance", "altitude", "grade_smooth", "lat", "lng", "moving"])

    for i in range(max_len):
        time_v = arrays["time"][i] if i < len(arrays["time"]) else ""
        cadence_v = arrays["cadence"][i] if i < len(arrays["cadence"]) else ""
        speed_v = arrays["velocity_smooth"][i] if i < len(arrays["velocity_smooth"]) else ""
        distance_v = arrays["distance"][i] if i < len(arrays["distance"]) else ""
        altitude_v = arrays["altitude"][i] if i < len(arrays["altitude"]) else ""
        grade_v = arrays["grade_smooth"][i] if i < len(arrays["grade_smooth"]) else ""
        lat = ""
        lng = ""
        if i < len(arrays["latlng"]):
            coords = arrays["latlng"][i]
            if isinstance(coords, (list, tuple)) and len(coords) == 2:
                lat, lng = coords
        moving_v = arrays["moving"][i] if i < len(arrays["moving"]) else ""
        writer.writerow([time_v, cadence_v, speed_v, distance_v, altitude_v, grade_v, lat, lng, moving_v])

    csv_text = buf.getvalue()
    buf.close()

    filename = f"activity_{activity_id}_streams.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    logger.info("Returning CSV for activity %s (rows=%d)", activity_id, max_len)
    return Response(content=csv_text, media_type="text/csv", headers=headers)
