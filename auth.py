# auth.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
import requests
from typing import Dict
import os
import uuid
import logging

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# In-memory session storage (for demo; replace with DB in prod)
SESSIONS: Dict[str, Dict] = {}

STRAVA_CLIENT_ID = os.environ.get("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET")
STRAVA_REDIRECT_URI = os.environ.get(
    "STRAVA_REDIRECT_URI",
    "https://optcadbackend-production.up.railway.app/auth/callback"
)

FRONTEND_URL = os.environ.get(
    "FRONTEND_URL",
    "https://n1jl0091.github.io/optcad_frontend/activities.html"
)


@router.get("/auth")
def auth_redirect():
    """Redirect user to Strava authorization"""
    logger.info("Initiating Strava OAuth flow")
    
    if not STRAVA_CLIENT_ID:
        logger.error("Missing STRAVA_CLIENT_ID")
        raise HTTPException(status_code=500, detail="Missing Strava client ID")
    
    url = (
        f"https://www.strava.com/oauth/authorize?"
        f"client_id={STRAVA_CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={STRAVA_REDIRECT_URI}&"
        f"scope=activity:read_all&"
        f"approval_prompt=auto"
    )
    logger.debug(f"Redirecting to Strava OAuth URL: {url}")
    return RedirectResponse(url)


@router.get("/auth/callback")
def auth_callback(code: str):
    """Handle Strava OAuth callback"""
    logger.info(f"Received OAuth callback with code: {code}")
    
    if not STRAVA_CLIENT_ID or not STRAVA_CLIENT_SECRET:
        logger.error("Missing Strava credentials")
        raise HTTPException(status_code=500, detail="Missing Strava credentials")
    
    # Exchange code for access token
    token_url = "https://www.strava.com/oauth/token"
    logger.debug(f"Exchanging code for access token at {token_url}")
    
    resp = requests.post(token_url, data={
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code"
    })
    
    if resp.status_code != 200:
        logger.error(f"Failed to get access token: {resp.status_code} {resp.text}")
        raise HTTPException(status_code=500, detail="Failed to get access token")
    
    token_data = resp.json()
    logger.debug(f"Access token received: {token_data}")

    # Generate session ID
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "expires_at": token_data.get("expires_at")
    }
    
    logger.info(f"Session created: {session_id}")
    
    # Redirect to frontend with session_id
    redirect_url = f"{FRONTEND_URL}?session_id={session_id}"
    logger.debug(f"Redirecting user to frontend: {redirect_url}")
    
    return RedirectResponse(redirect_url)
