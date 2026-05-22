"""
Centralized application configuration via environment variables.
"""
from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    FRAMES_DIR: Path = BASE_DIR / "frames"
    AUDIO_DIR: Path = BASE_DIR / "audio_temp"
    EXPLANATION_AUDIO_DIR: Path = BASE_DIR / "explanations"
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "app.db"
    FAISS_INDEX_DIR: Path = DATA_DIR / "faiss_index"

    # ── API Keys ───────────────────────────────────────────────────────
    GEMINI_API_KEY: str = ""

    # ── Ollama Config ──────────────────────────────────────────────────
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_VISION_MODEL: str = "llava:7b"
    OLLAMA_TEXT_MODEL: str = "qwen2.5:7b"

    # ── Server ─────────────────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = False

    # ── Whisper ────────────────────────────────────────────────────────
    WHISPER_MODEL: str = "base"

    # ── Processing Tuning ──────────────────────────────────────────────
    KEYFRAME_INTERVAL_SEC: float = 2.0
    MAX_KEYFRAMES_PER_SCENE: int = 10
    MAX_UPLOAD_SIZE_MB: int = 500

    # ── Question Engine ────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.85
    QUESTIONS_PER_CATEGORY: int = 5
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"debug", "dev", "development"}:
                return True
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_directories(self) -> None:
        """Create all required data directories."""
        for d in (self.UPLOAD_DIR, self.FRAMES_DIR, self.AUDIO_DIR,
                  self.EXPLANATION_AUDIO_DIR, self.DATA_DIR, self.FAISS_INDEX_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
