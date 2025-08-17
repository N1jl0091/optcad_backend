from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
import requests
import uuid
import logging
from config import SESSIONS, STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, REDIRECT_URI, FRONTEND_URL

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@router.get("/auth")
def auth_redirect():
    """Redirect user to Strava authorization page"""
    logger.info("Initiating Strava OAuth flow")
    
    if not STRAVA_CLIENT_ID or not STRAVA_CLIENT_SECRET:
        logger.error("Missing Strava credentials")
        raise HTTPException(status_code=500, detail="Server configuration error")
    
    url = (
        f"https://www.strava.com/oauth/authorize?"
        f"client_id={STRAVA_CLIENT_ID}&"
        f"response_type=code&"
        f"redirect_uri={REDIRECT_URI}&"
        f"scope=activity:read_all&"
        f"approval_prompt=auto"
    )
    logger.debug(f"Redirecting user to Strava OAuth URL: {url}")
    return RedirectResponse(url)


@router.get("/auth/callback")
def auth_callback(code: str):
    """Handle Strava OAuth callback"""
    logger.info(f"Received OAuth callback with code: {code}")
    
    if not STRAVA_CLIENT_ID or not STRAVA_CLIENT_SECRET:
        logger.error("Missing Strava credentials")
        raise HTTPException(status_code=500, detail="Server configuration error")
    
    # Exchange code for access token
    token_url = "https://www.strava.com/oauth/token"
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
    
    # Create session ID
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data["refresh_token"],
        "expires_at": token_data.get("expires_at")
    }
    logger.info(f"Session created: {session_id}")
    
    # Redirect to frontend with session ID
    redirect_url = f"{FRONTEND_URL}?session_id={session_id}"
    logger.debug(f"Redirecting user to frontend: {redirect_url}")
    
    return RedirectResponse(redirect_url)
