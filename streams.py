from fastapi import APIRouter
from compute.optcad_compute import process_activity_stream  # <--- fixed

router = APIRouter(prefix="/streams", tags=["streams"])

@router.post("/process")
def process_streams(streams: dict):
    result = process_activity_stream(streams)
    return result
