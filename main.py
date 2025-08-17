# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from auth import router as auth_router
from activities import router as activities_router
from streams import router as streams_router
from compute.config import TIME_LIMIT_SEC  # Example import to confirm config is loaded

app = FastAPI()

# CORS setup to allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://n1jl0091.github.io"],  # Your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(activities_router)
app.include_router(streams_router)

print(f"Starting app. TIME_LIMIT_SEC from config: {TIME_LIMIT_SEC}")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
