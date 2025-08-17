# auth.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
import requests
import uuid
import os
from config import SESSIONS, STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, REDIRECT_URI

router = APIRouter()

# Frontend URL can be set via env or fallback
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://N1jl0091.github.io/optcad_frontend/activities.html"
)

@router.get("/auth")
def auth_redirect():
    """Redirect user to Strava authorization"""
    if not STRAVA_CLIENT_ID or not STRAVA_CLIENT_SECRET:
        return JSONResponse({"error": "Server configuration error"}, status_code=500)

    url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={STRAVA_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=activity:read_all"
    )
    return RedirectResponse(url)

@router.get("/callback")
def auth_callback(code: str):
    """Handle Strava OAuth callback"""
    if not STRAVA_CLIENT_ID or not STRAVA_CLIENT_SECRET:
        return JSONResponse({"error": "Server configuration error"}, status_code=500)

    # Exchange code for token
    token_response = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code"
        }
    )

    try:
        token_data = token_response.json()
    except Exception:
        return JSONResponse({"error": "Failed to parse token response"}, status_code=500)

    # Store the full token response in session
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = token_data

    # Redirect user to frontend with session_id
    redirect_url = f"{FRONTEND_URL}?session_id={session_id}"
    return RedirectResponse(redirect_url)
