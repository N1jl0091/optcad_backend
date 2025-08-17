# streams.py
from fastapi import APIRouter
from optcad_compute import process_activity_stream

router = APIRouter(prefix="/streams", tags=["streams"])

@router.post("/process")
def process_streams(streams: dict):
    """
    Accept Strava activity streams and process them into
    segments, scores, and optimal cadence.
    """
    result = process_activity_stream(streams)
    return result
