# ── Stage 1: Build ─────────────────────────────────────────────────
FROM python:3.10-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-dri \
    libglib2.0-0 \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip "setuptools<70" wheel
# Install heavy dependencies with specific version pins to avoid conflicts
RUN python -m pip install --no-cache-dir "tokenizers>=0.21,<0.22" "transformers>=4.48.0" sentence-transformers==3.4.1
# Install whisper first with absolute version and no isolation to avoid pkg_resources issue
RUN python -m pip install --no-cache-dir openai-whisper==20240930 --no-build-isolation
RUN python -m pip install --no-cache-dir -r requirements.txt

# Application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create data directories
RUN mkdir -p uploads frames audio_temp explanations data/faiss_index

# ── Runtime ────────────────────────────────────────────────────────
EXPOSE 9005

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9005/api/videos')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "9005"]
