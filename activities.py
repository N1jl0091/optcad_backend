from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import requests
import logging
from config import SESSIONS
from compute.core import (
    prepare_data,
    calculate_gradient,
    segment_ride,
    calculate_cadence_elevation,
    aggregate_segments,
    compute_scores,
    optimal_cadence
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

router = APIRouter()

@router.get("/activities")
def list_activities(session_id: str):
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
        logger.error(f"Failed to fetch activities: {response.status_code}, {response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch activities")
    
    logger.info(f"Fetched {len(response.json())} activities")
    return JSONResponse(response.json())

@router.get("/compute")
def compute_activity(session_id: str, activity_id: str):
    logger.info(f"Computing activity {activity_id} for session_id={session_id}")
    
    token_data = SESSIONS.get(session_id)
    if not token_data:
        logger.warning(f"Invalid session ID: {session_id}")
        raise HTTPException(status_code=401, detail="Invalid session ID")
    
    access_token = token_data["access_token"]
    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    stream_types = ",".join(["time","cadence","velocity_smooth","distance","altitude","grade_smooth","latlng","moving"])
    
    logger.debug(f"Fetching streams from Strava API: {url} for types {stream_types}")
    response = requests.get(
        url, 
        headers={"Authorization": f"Bearer {access_token}"}, 
        params={"keys": stream_types,"key_by_type":"true","resolution":"high"}
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to fetch stream data: {response.status_code}, {response.text}")
        raise HTTPException(status_code=500, detail="Failed to fetch stream data")
    
    stream_data = response.json()
    logger.info("Stream data fetched successfully")
    
    try:
        logger.info("Preparing data")
        data = prepare_data(stream_data)

        logger.info("Calculating gradient")
        grad = calculate_gradient(data)

        logger.info("Segmenting ride")
        segments_idx = segment_ride(data)

        logger.info("Calculating cadence and elevation")
        segments = calculate_cadence_elevation(data)
        
        logger.info("Aggregating segments")
        segments = aggregate_segments(segments)

        logger.info("Computing scores")
        segments = compute_scores(segments)

        logger.info("Determining optimal cadence")
        opt_cad = optimal_cadence(segments)

        logger.info(f"Computation complete: {len(segments)} segments, optimal cadence={opt_cad}")

    except Exception as e:
        logger.exception(f"Error during computation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return JSONResponse({"segments": segments, "optimal_cadence": opt_cad})
