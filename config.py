import os

# Strava credentials (must be set in environment variables)
STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

# Backend endpoint for OAuth callback
REDIRECT_URI = os.getenv(
    "STRAVA_REDIRECT_URI",
    "https://optcadbackend-production.up.railway.app/auth/callback"
)

# Frontend URL for redirect after login
FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://n1jl0091.github.io/optcad_frontend/activities.html"
)

# In-memory session storage (demo only; replace with DB in production)
SESSIONS = {}
