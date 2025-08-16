from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from config import SESSIONS, DROPBOX_ACCESS_TOKEN, DROPBOX_UPLOAD_FOLDER
import requests
import csv
import uuid
import dropbox
from io import StringIO

router = APIRouter()

@router.get("/activity-stream")
def get_activity_stream(session_id: str, activity_id: str):
    token_data = SESSIONS.get(session_id)
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid session ID")

    access_token = token_data["access_token"]

    # Strava streams request
    stream_types = ",".join([
        "time", "cadence", "velocity_smooth", "distance",
        "altitude", "grade_smooth", "latlng", "moving"
    ])

    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"keys": stream_types, "key_by_type": "true", "resolution": "high"}
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch stream data")

    stream_data = response.json()

    # Generate CSV in memory
    fields = ["time", "cadence", "speed", "distance", "altitude", "grade_smooth", "lat", "lng", "moving"]
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(fields)

    row_count = len(stream_data.get("time", {}).get("data", []))
    for i in range(row_count):
        lat, lng = ("", "")
        if "latlng" in stream_data and i < len(stream_data["latlng"].get("data", [])):
            coords = stream_data["latlng"]["data"][i]
            if isinstance(coords, list) and len(coords) == 2:
                lat, lng = coords
        row = [
            stream_data.get("time", {}).get("data", [])[i] if i < len(stream_data.get("time", {}).get("data", [])) else "",
            stream_data.get("cadence", {}).get("data", [])[i] if i < len(stream_data.get("cadence", {}).get("data", [])) else "",
            stream_data.get("velocity_smooth", {}).get("data", [])[i] if i < len(stream_data.get("velocity_smooth", {}).get("data", [])) else "",
            stream_data.get("distance", {}).get("data", [])[i] if i < len(stream_data.get("distance", {}).get("data", [])) else "",
            stream_data.get("altitude", {}).get("data", [])[i] if i < len(stream_data.get("altitude", {}).get("data", [])) else "",
            stream_data.get("grade_smooth", {}).get("data", [])[i] if i < len(stream_data.get("grade_smooth", {}).get("data", [])) else "",
            lat, lng,
            stream_data.get("moving", {}).get("data", [])[i] if i < len(stream_data.get("moving", {}).get("data", [])) else "",
        ]
        writer.writerow(row)

    # Upload to Dropbox
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    dropbox_path = f"{DROPBOX_UPLOAD_FOLDER}/activity_{activity_id}.csv"

    try:
        dbx.files_upload(output.getvalue().encode("utf-8"), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
    except dropbox.exceptions.ApiError as e:
        raise HTTPException(status_code=500, detail=f"Dropbox upload failed: {e}")

    return JSONResponse({"message": "CSV uploaded to Dropbox", "dropbox_path": dropbox_path})
