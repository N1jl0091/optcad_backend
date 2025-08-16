from fastapi import APIRouter
from fastapi.responses import JSONResponse
from config import SESSIONS
import requests

router = APIRouter()

@router.get("/activities")
def list_activities(session_id: str):
    print(f"Fetching activities for session: {session_id}")
    
    token_data = SESSIONS.get(session_id)
    if not token_data:
        print("Invalid or expired session ID.")
        return JSONResponse(content={"error": "Invalid session ID"}, status_code=401)

    access_token = token_data["access_token"]
    
    print("Requesting activities from Strava...")
    strava_response = requests.get(
        "https://www.strava.com/api/v3/athlete/activities",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"per_page": 100, "page": 1}  # Request up to 100 most recent activities
    )

    if strava_response.status_code != 200:
        print(f"Failed to fetch activities: {strava_response.text}")
        return JSONResponse(content={"error": "Failed to fetch activities"}, status_code=500)

    all_activities = strava_response.json()

    # Get the 30 most recent Rides
    rides = [
        {
            "id": a.get("id"),
            "name": a.get("name"),
            "distance": a.get("distance"),
            "start_date": a.get("start_date"),
        }
        for a in all_activities if a.get("type") == "Ride"
    ][:30]

    print(f"Returning {len(rides)} bike rides")
    return JSONResponse(content=rides)