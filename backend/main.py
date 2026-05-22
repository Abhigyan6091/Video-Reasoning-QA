"""
FastAPI application entry-point.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────
    settings.ensure_directories()
    await init_db()
    yield
    # ── Shutdown ───────────────────────────────────────────────────


app = FastAPI(
    title="Video Question Generator",
    description="AI-powered reasoning question generator from MP4 videos",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ─────────────────────────────────────────────────────
from backend.routes.api import router as api_router  # noqa: E402

app.include_router(api_router, prefix="/api")

# ── Generated Media ────────────────────────────────────────────────
settings.ensure_directories()
app.mount("/frames", StaticFiles(directory=str(settings.FRAMES_DIR)), name="frames")
app.mount("/explanations", StaticFiles(directory=str(settings.EXPLANATION_AUDIO_DIR)), name="explanations")

# ── Static Frontend ───────────────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
