# auth.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
import requests
from typing import Dict
import os

router = APIRouter()

# In-memory session storage (for demo; replace with DB in prod)
SESSIONS: Dict[str, Dict] = {}

STRAVA_CLIENT_ID = os.environ.get("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET")
STRAVA_REDIRECT_URI = os.environ.get("STRAVA_REDIRECT_URI", "https://optcadbackend-production.up.railway.app/auth/callback")

@router.get("/auth")
def auth_redirect():
    """Redirect user to Strava authorization"""
    if not STRAVA_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Missing Strava client ID")
    
    url = (
        f"https://www.strava.com/oauth/authorize?"
        f"client_id={STRAVA_CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={STRAVA_REDIRECT_URI}&"
        f"scope=activity:read_all&"
        f"approval_prompt=auto"
    )
    return RedirectResponse(url)

@router.get("/auth/callback")
def auth_callback(code: str):
    """Handle Strava OAuth callback"""
    if not STRAVA_CLIENT_ID or not STRAVA_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Missing Strava credentials")
    
    # Exchange code for access token
    token_url = "https://www.strava.com/oauth/token"
    resp = requests.post(token_url, data={
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code"
    })
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to get access token")
    
    token_data = resp.json()
    # Generate session ID (simple example)
    import uuid
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "expires_at": token_data.get("expires_at")
    }
    
    # Redirect to frontend with session_id in query string
    return RedirectResponse(f"{STRAVA_REDIRECT_URI}?session_id={session_id}")
