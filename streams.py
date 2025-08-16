from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from config import SESSIONS
import requests
import csv
import uuid

router = APIRouter()

@router.get("/activity-stream")
def get_activity_stream(session_id: str, activity_id: str):
    print(f"Processing stream for activity: {activity_id} with session: {session_id}")

    token_data = SESSIONS.get(session_id)
    if not token_data:
        return JSONResponse({"error": "Invalid session ID"}, status_code=401)

    access_token = token_data["access_token"]

    # Request all desired stream types
    stream_types = ",".join([
        "time",
        "cadence",
        "velocity_smooth",
        "distance",
        "altitude",
        "grade_smooth",
        "latlng",
        "moving"
    ])

    url = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"

    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "keys": stream_types,
            "key_by_type": "true",
            "resolution": "high"
        }
    )

    if response.status_code != 200:
        print(f"Error fetching stream data: {response.text}")
        return JSONResponse({"error": "Failed to fetch stream data"}, status_code=500)

    stream_data = response.json()
    print("Stream data retrieved. Generating CSV...")

    # Prepare CSV file
    filename = f"/tmp/stream_{uuid.uuid4().hex}.csv"
    fields = ["time", "cadence", "speed", "distance", "altitude", "grade_smooth", "lat", "lng", "moving"]

    row_count = len(stream_data.get("time", {}).get("data", []))
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fields)

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
                lat,
                lng,
                stream_data.get("moving", {}).get("data", [])[i] if i < len(stream_data.get("moving", {}).get("data", [])) else "",
            ]
            writer.writerow(row)

    print(f"CSV file generated at {filename}")
    return FileResponse(filename, media_type="text/csv", filename="activity_stream.csv")
