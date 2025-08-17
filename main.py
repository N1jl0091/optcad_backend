from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from auth import router as auth_router
from activities import router as activities_router
from streams import router as streams_router  # ✅ correct

app = FastAPI()

from fastapi import Request
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = logging.getLogger("uvicorn.error")
    logger.error("Unhandled exception occurred: %s", exc)
    logger.error(traceback.format_exc())
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://n1jl0091.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(activities_router)
app.include_router(streams_router)  # ✅ now works

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
