# 🎬 Video Question Generator (VQG)

AI-powered system that generates unique, reasoning-based questions from videos using **local AI models**. Now featuring multi-language support and high-quality audio explanations.

## ✨ Features

- **Local AI Pipeline** — Fully offline-capable using Ollama (Llava for vision, Gemma/Qwen for text).
- **Multi-Language Reasoning** — Generate questions and explanations in **English**, **Hindi**, or **Hinglish (Mixed)**.
- **Audio Explanations (TTS)** — High-quality AI voiceovers for every answer, generated via Edge-TTS.
- **Deep Video Understanding** — Analyzes scenes for objects, actions, emotions, and anomalies.
- **7 Question Categories** — Temporal, Causal, Counterfactual, Contradiction, Emotion, Multi-Scene, Symbolic.
- **Uniqueness Filter** — Embedding-based deduplication using FAISS removes near-duplicate questions.
- **Premium UI** — Modern, dark-themed interface with real-time processing feedback.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (HTML/CSS/JS) — Dark theme, drag-drop upload  │
├─────────────────────────────────────────────────────────┤
│  FastAPI Backend (Port 9005)                             │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Upload  │→│ Scenes   │→│ Frames   │→│ Audio      │  │
│  │ Ingest  │ │ Segment  │ │ Extract  │ │ Transcribe │  │
│  └─────────┘ └──────────┘ └──────────┘ └────────────┘  │
│       ↓                                                  │
│  ┌──────────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Ollama (VLM) │→│ Scene    │→│ FAISS Memory      │  │
│  │ Understanding│  │ Graphs   │  │ (Embeddings)      │  │
│  └──────────────┘  └──────────┘  └───────────────────┘  │
│       ↓                                                  │
│  ┌──────────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Question     │→│ Audio    │→│ Uniqueness Filter │  │
│  │ Generator    │  │ (TTS)    │  │ (Cosine Dedup)    │  │
│  └──────────────┘  └──────────┘  └───────────────────┘  │
├─────────────────────────────────────────────────────────┤
│  SQLite (Metadata) + FAISS (Vectors) + MP3 (Audio)       │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start (Local Setup)

### 1. Requirements
- **Ollama**: [Download and install Ollama](https://ollama.ai/)
- **Models**: `ollama pull llava:7b` and `ollama pull gemma4:e4b` (or `qwen2.5:7b`)
- **FFmpeg**: Required for audio/video processing.

### 2. Startup (Linux)
The easiest way to start is using the provided startup script:
```bash
./startup.sh
```
This script will:
- Check if Ollama is running.
- Verify required models are available.
- Start the backend on **http://localhost:9005**.

### 3. Docker (Alternative)
```bash
cp .env.example .env
docker-compose up --build
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload MP4 with language preference |
| `POST` | `/api/process/{id}` | Start processing pipeline |
| `GET`  | `/api/progress/{id}`| SSE progress stream |
| `POST` | `/api/regenerate/{id}`| Regenerate with language/audio |
| `GET`  | `/explanations/{id}.mp3` | Serve audio explanations |

## 📁 Project Structure

```
Video Question Generator/
├── backend/
│   ├── main.py               # FastAPI app entry point
│   ├── database.py            # SQLite async database layer
│   ├── pipelines/             # Processing pipeline (ingestion -> generation)
│   ├── services/
│   │   └── tts.py             # Audio synthesis via Edge-TTS
│   ├── question_engine/       # AI reasoning & filtering logic
│   └── routes/api.py          # REST API endpoints
├── frontend/                  # SPA (HTML/CSS/JS)
├── data/                      # Database & FAISS index
├── explanations/              # Generated MP3 voiceovers
├── requirements.txt
├── Dockerfile
└── startup.sh
```

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.10, FastAPI |
| Vision/Text | **Ollama** (Local Llava & Gemma) |
| Audio TTS | **Edge-TTS** (High-quality voices) |
| Transcription | OpenAI Whisper (Local) |
| Vector Search | FAISS |
| Database | SQLite (Async) |
| Frontend | Vanilla HTML/CSS/JS (Premium Dark) |


