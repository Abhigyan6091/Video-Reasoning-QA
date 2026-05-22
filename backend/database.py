"""
SQLite database layer – async via aiosqlite.

Tables:
  videos   – uploaded video metadata
  scenes   – detected scenes per video
  events   – structured event data per scene (objects, actions, emotions …)
  questions – generated questions with category, difficulty, embedding id
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from backend.config import settings

DB_PATH = str(settings.DB_PATH)

# ────────────────────────────────────────────────────────────────────
# Schema
# ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    duration    REAL,
    width       INTEGER,
    height      INTEGER,
    fps         REAL,
    status      TEXT DEFAULT 'uploaded',   -- uploaded | processing | completed | failed
    error_msg   TEXT,
    transcript  TEXT,                       -- JSON object with text and segments
    language    TEXT DEFAULT 'english',    -- english | hindi | mixed
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenes (
    id          TEXT PRIMARY KEY,
    video_id    TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    scene_idx   INTEGER NOT NULL,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    summary     TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    scene_id    TEXT NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    video_id    TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    objects     TEXT,   -- JSON array
    actions     TEXT,   -- JSON array
    interactions TEXT,  -- JSON array
    emotions    TEXT,   -- JSON array
    anomalies   TEXT,   -- JSON array
    summary     TEXT,
    timestamp   TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id              TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    question_text   TEXT NOT NULL,
    answer_text     TEXT,
    category        TEXT NOT NULL,  -- temporal | causal | counterfactual | contradiction | emotion | multi_scene | symbolic
    persona         TEXT,           -- detective | psychologist | director | …
    difficulty      TEXT NOT NULL,  -- easy | medium | hard | expert
    difficulty_score REAL,
    scenes_involved TEXT,           -- JSON array of scene ids
    novelty_score   REAL,
    embedding_id    INTEGER,        -- index into FAISS
    mc_options      TEXT,           -- JSON array of MCQ options
    correct_option  INTEGER,        -- 0-indexed position of correct answer
    explanation     TEXT,           -- Detailed explanation linking audio and visual
    visual_refs     TEXT,           -- JSON array of frame references for visual questions
    quality_score   REAL,           -- Quality score from validation
    audio_path      TEXT,           -- Path to generated audio explanation
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scenes_video  ON scenes(video_id);
CREATE INDEX IF NOT EXISTS idx_events_scene  ON events(scene_id);
CREATE INDEX IF NOT EXISTS idx_events_video  ON events(video_id);
CREATE INDEX IF NOT EXISTS idx_questions_video ON questions(video_id);
CREATE INDEX IF NOT EXISTS idx_questions_cat   ON questions(category);
CREATE INDEX IF NOT EXISTS idx_questions_diff  ON questions(difficulty);
"""


async def init_db() -> None:
    """Create tables if they don't exist."""
    settings.ensure_directories()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await _migrate_schema(db)
        await db.commit()


async def _migrate_schema(db: aiosqlite.Connection) -> None:
    """Apply lightweight migrations for databases created by older builds."""
    async with db.execute("PRAGMA table_info(videos)") as cur:
        columns = {row[1] for row in await cur.fetchall()}

    if "transcript" not in columns:
        await db.execute("ALTER TABLE videos ADD COLUMN transcript TEXT")
    
    if "language" not in columns:
        await db.execute("ALTER TABLE videos ADD COLUMN language TEXT DEFAULT 'english'")

    async with db.execute("PRAGMA table_info(questions)") as cur:
        question_columns = {row[1] for row in await cur.fetchall()}

    if "visual_refs" not in question_columns:
        await db.execute("ALTER TABLE questions ADD COLUMN visual_refs TEXT")
    
    if "audio_path" not in question_columns:
        await db.execute("ALTER TABLE questions ADD COLUMN audio_path TEXT")


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return uuid.uuid4().hex


def _json_dumps(obj: Any) -> str | None:
    return json.dumps(obj) if obj is not None else None


def _json_loads(s: str | None) -> Any:
    return json.loads(s) if s else None


def _row_to_dict(cursor: aiosqlite.Cursor, row: tuple) -> dict:
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


# ────────────────────────────────────────────────────────────────────
# CRUD – Videos
# ────────────────────────────────────────────────────────────────────

async def create_video(filename: str, filepath: str,
                       duration: float | None = None,
                       width: int | None = None,
                       height: int | None = None,
                       fps: float | None = None,
                       language: str = "english") -> dict:
    vid = _uuid()
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO videos (id, filename, filepath, duration, width, height, fps, language, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (vid, filename, filepath, duration, width, height, fps, language, now, now),
        )
        await db.commit()
    return {"id": vid, "filename": filename, "status": "uploaded"}


async def update_video_status(video_id: str, status: str,
                              error_msg: str | None = None) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE videos SET status = ?, error_msg = ?, updated_at = ? WHERE id = ?",
            (status, error_msg, _now(), video_id),
        )
        await db.commit()


async def update_video_metadata(video_id: str, duration: float,
                                width: int, height: int, fps: float) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE videos SET duration=?, width=?, height=?, fps=?, updated_at=? WHERE id=?",
            (duration, width, height, fps, _now(), video_id),
        )
        await db.commit()


async def update_video_transcript(video_id: str, transcript: dict) -> None:
    """Store transcript for a video."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE videos SET transcript = ?, updated_at = ? WHERE id = ?",
            (_json_dumps(transcript), _now(), video_id),
        )
        await db.commit()


async def get_video(video_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = _row_to_dict  # type: ignore[assignment]
        async with db.execute("SELECT * FROM videos WHERE id = ?", (video_id,)) as cur:
            row = await cur.fetchone()
            return row  # type: ignore[return-value]


async def get_video_transcript(video_id: str) -> dict | None:
    """Return the parsed transcript object for a video, if one has been stored."""
    video = await get_video(video_id)
    if not video or not video.get("transcript"):
        return None

    transcript = video["transcript"]
    return _json_loads(transcript) if isinstance(transcript, str) else transcript


async def get_latest_transcript_for_filename(
    filename: str,
    exclude_video_id: str | None = None,
) -> dict | None:
    """Return the newest non-empty transcript stored for the same uploaded filename."""
    query = """
        SELECT id, transcript
        FROM videos
        WHERE filename = ?
          AND transcript IS NOT NULL
        ORDER BY updated_at DESC, created_at DESC
    """

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = _row_to_dict  # type: ignore[assignment]
        async with db.execute(query, (filename,)) as cur:
            rows = await cur.fetchall()

    for row in rows:
        if exclude_video_id and row.get("id") == exclude_video_id:
            continue

        transcript = _json_loads(row.get("transcript"))
        if transcript and _transcript_has_text(transcript):
            transcript["reused_from_video_id"] = row.get("id")
            return transcript

    return None


def _transcript_has_text(transcript: dict) -> bool:
    if (transcript.get("text") or "").strip():
        return True
    return any((seg.get("text") or "").strip() for seg in transcript.get("segments", []))


async def list_videos() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = _row_to_dict  # type: ignore[assignment]
        async with db.execute("SELECT * FROM videos ORDER BY created_at DESC") as cur:
            return await cur.fetchall()  # type: ignore[return-value]


# ────────────────────────────────────────────────────────────────────
# CRUD – Scenes
# ────────────────────────────────────────────────────────────────────

async def create_scenes(video_id: str,
                        scene_list: list[dict]) -> list[str]:
    """Bulk-insert scenes. Each dict: {scene_idx, start_time, end_time}."""
    ids: list[str] = []
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        for s in scene_list:
            sid = _uuid()
            ids.append(sid)
            await db.execute(
                "INSERT INTO scenes (id, video_id, scene_idx, start_time, end_time, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sid, video_id, s["scene_idx"], s["start_time"], s["end_time"], now),
            )
        await db.commit()
    return ids


async def update_scene_summary(scene_id: str, summary: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE scenes SET summary = ? WHERE id = ?", (summary, scene_id)
        )
        await db.commit()


async def get_scenes(video_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = _row_to_dict  # type: ignore[assignment]
        async with db.execute(
            "SELECT * FROM scenes WHERE video_id = ? ORDER BY scene_idx", (video_id,)
        ) as cur:
            return await cur.fetchall()  # type: ignore[return-value]


# ────────────────────────────────────────────────────────────────────
# CRUD – Events
# ────────────────────────────────────────────────────────────────────

async def create_event(scene_id: str, video_id: str,
                       objects: list | None = None,
                       actions: list | None = None,
                       interactions: list | None = None,
                       emotions: list | None = None,
                       anomalies: list | None = None,
                       summary: str | None = None,
                       timestamp: str | None = None) -> str:
    eid = _uuid()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO events (id, scene_id, video_id, objects, actions, interactions, "
            "emotions, anomalies, summary, timestamp, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (eid, scene_id, video_id,
             _json_dumps(objects), _json_dumps(actions),
             _json_dumps(interactions), _json_dumps(emotions),
             _json_dumps(anomalies), summary, timestamp, _now()),
        )
        await db.commit()
    return eid


async def get_events(video_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = _row_to_dict  # type: ignore[assignment]
        async with db.execute(
            "SELECT * FROM events WHERE video_id = ? ORDER BY timestamp", (video_id,)
        ) as cur:
            rows = await cur.fetchall()
    # Parse JSON fields back to lists
    for row in rows:
        for field in ("objects", "actions", "interactions", "emotions", "anomalies"):
            row[field] = _json_loads(row[field])  # type: ignore[index]
    return rows  # type: ignore[return-value]


# ────────────────────────────────────────────────────────────────────
# CRUD – Questions
# ────────────────────────────────────────────────────────────────────

async def create_questions(video_id: str,
                           question_list: list[dict]) -> list[str]:
    """Bulk-insert questions."""
    ids: list[str] = []
    now = _now()
    async with aiosqlite.connect(DB_PATH) as db:
        for q in question_list:
            qid = q.get("id") or _uuid()
            ids.append(qid)
            await db.execute(
                "INSERT INTO questions (id, video_id, question_text, answer_text, "
                "category, persona, difficulty, difficulty_score, scenes_involved, "
                "novelty_score, embedding_id, mc_options, correct_option, explanation, "
                "visual_refs, quality_score, audio_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (qid, video_id,
                 q["question_text"], q.get("answer_text"),
                 q["category"], q.get("persona"),
                 q["difficulty"], q.get("difficulty_score"),
                 _json_dumps(q.get("scenes_involved")),
                 q.get("novelty_score"), q.get("embedding_id"),
                 _json_dumps(q.get("mc_options")),
                 q.get("correct_option"),
                 q.get("explanation"),
                 _json_dumps(q.get("visual_refs")),
                 q.get("quality_score"),
                 q.get("audio_path"),
                 now),
            )
        await db.commit()
    return ids


async def get_questions(video_id: str,
                        category: str | None = None,
                        difficulty: str | None = None) -> list[dict]:
    query = "SELECT * FROM questions WHERE video_id = ?"
    params: list[Any] = [video_id]
    if category:
        query += " AND category = ?"
        params.append(category)
    if difficulty:
        query += " AND difficulty = ?"
        params.append(difficulty)
    query += " ORDER BY category, difficulty_score"

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = _row_to_dict  # type: ignore[assignment]
        async with db.execute(query, params) as cur:
            rows = await cur.fetchall()
    for row in rows:
        row["scenes_involved"] = _json_loads(row["scenes_involved"])  # type: ignore[index]
        row["mc_options"] = _json_loads(row["mc_options"])  # type: ignore[index]
        row["visual_refs"] = _json_loads(row.get("visual_refs")) or []  # type: ignore[index]
    return rows  # type: ignore[return-value]


async def delete_questions(video_id: str,
                           category: str | None = None) -> int:
    """Delete questions for a video (optionally filtered by category). Returns deleted count."""
    query = "DELETE FROM questions WHERE video_id = ?"
    params: list[Any] = [video_id]
    if category:
        query += " AND category = ?"
        params.append(category)
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(query, params)
        await db.commit()
        return cur.rowcount
