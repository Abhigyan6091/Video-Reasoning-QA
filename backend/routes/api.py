"""
API routes for the Video Question Generator.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from backend.config import settings
from backend import database as db
from backend.pipelines.orchestrator import (
    process_video,
    get_progress,
    transcribe_video_to_db,
    regenerate_questions_from_existing,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ────────────────────────────────────────────────────────────────────
# Upload
# ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    language: str = "english"
):
    """Upload an MP4 video for processing."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Only MP4 files are accepted")

    logger.info("Received upload request for: %s", file.filename)
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    logger.info("File read complete: %s (%.2f MB)", file.filename, size_mb)

    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_UPLOAD_SIZE_MB} MB")

    from backend.pipelines.ingestion import ingest_video
    logger.info("Starting ingestion for %s (language: %s)", file.filename, language)
    video = await ingest_video(file.filename, contents, language=language)
    logger.info("Ingestion complete for %s. Video ID: %s", file.filename, video['id'])

    return JSONResponse(content={
        "status": "uploaded",
        "video": video,
        "message": f"Video '{file.filename}' uploaded ({size_mb:.1f} MB). "
                   f"Call POST /api/process/{video['id']} to start processing.",
    })


# ────────────────────────────────────────────────────────────────────
# Process
# ────────────────────────────────────────────────────────────────────

@router.post("/process/{video_id}")
async def start_processing(video_id: str):
    """Trigger the full processing pipeline for an uploaded video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    if video["status"] == "processing":
        raise HTTPException(409, "Video is already being processed")

    # Run pipeline in background
    asyncio.create_task(process_video(video_id))

    return {"status": "processing", "video_id": video_id,
            "message": "Processing started. Use GET /api/progress/{video_id} for updates."}


# ────────────────────────────────────────────────────────────────────
# Progress (SSE)
# ────────────────────────────────────────────────────────────────────

@router.get("/progress/{video_id}")
async def stream_progress(video_id: str):
    """Server-Sent Events stream of processing progress."""
    async def event_generator() -> AsyncGenerator[str, None]:
        last_idx = 0
        while True:
            progress = get_progress(video_id)
            while last_idx < len(progress):
                msg = progress[last_idx]
                data = json.dumps(msg)
                yield f"data: {data}\n\n"
                last_idx += 1

                # Stop streaming if complete or errored
                if msg.get("step") in ("completed", "error"):
                    return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ────────────────────────────────────────────────────────────────────
# Get Video Info
# ────────────────────────────────────────────────────────────────────

@router.get("/videos")
async def list_videos():
    """List all uploaded videos."""
    videos = await db.list_videos()
    return {"videos": videos}


@router.get("/videos/{video_id}")
async def get_video_info(video_id: str):
    """Get details for a specific video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    return {"video": video}


# ────────────────────────────────────────────────────────────────────
# Questions
# ────────────────────────────────────────────────────────────────────

@router.get("/questions/{video_id}")
async def get_questions(
    video_id: str,
    category: str | None = Query(None, description="Filter by category"),
    difficulty: str | None = Query(None, description="Filter by difficulty"),
):
    """Retrieve generated questions with optional filters."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    questions = await db.get_questions(video_id, category=category, difficulty=difficulty)

    # Group by category for easier frontend consumption
    grouped: dict[str, list] = {}
    for q in questions:
        cat = q.get("category", "other")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(q)

    return {
        "video_id": video_id,
        "total": len(questions),
        "grouped": grouped,
        "questions": questions,
    }


@router.get("/scenes/{video_id}")
async def get_scenes(video_id: str):
    """Get detected scenes for a video."""
    scenes = await db.get_scenes(video_id)
    return {"video_id": video_id, "scenes": scenes}


# ────────────────────────────────────────────────────────────────────
# Transcript
# ────────────────────────────────────────────────────────────────────

@router.get("/transcript/{video_id}")
async def get_transcript(video_id: str):
    """Get transcript for a video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    transcript = await db.get_video_transcript(video_id)
    if not transcript:
        raise HTTPException(404, "No transcript available for this video")

    return {
        "video_id": video_id,
        "filename": video.get("filename"),
        "transcript": transcript,
    }


@router.get("/export-transcript/{video_id}")
async def export_transcript(
    video_id: str,
    format: str = Query("json", description="Export format: json or txt"),
):
    """Export transcript as JSON or plain text."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    transcript_data = await db.get_video_transcript(video_id)
    if not transcript_data:
        raise HTTPException(404, "No transcript available for this video")

    if format == "txt":
        segments = transcript_data.get("segments", [])
        if segments:
            text_content = "\n".join(
                f"[{_format_timestamp(seg.get('start', 0))}] {seg.get('text', '').strip()}"
                for seg in segments
                if seg.get("text")
            )
        else:
            text_content = transcript_data.get("text", "")

        return StreamingResponse(
            io.BytesIO(text_content.encode()),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=transcript_{video_id}.txt"},
        )
    else:
        # Return as JSON
        return JSONResponse(
            content={"video_id": video_id, "transcript": transcript_data},
            headers={"Content-Disposition": f"attachment; filename=transcript_{video_id}.json"},
        )


def _format_timestamp(seconds: float | int | None) -> str:
    seconds = int(seconds or 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@router.post("/transcript/{video_id}/regenerate")
async def regenerate_transcript(video_id: str):
    """Re-run only audio extraction and transcription for a video."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    try:
        transcript = await transcribe_video_to_db(video_id)
    except Exception as exc:
        logger.exception("Transcript regeneration failed for %s", video_id)
        raise HTTPException(500, f"Transcript regeneration failed: {exc}") from exc

    return {
        "video_id": video_id,
        "filename": video.get("filename"),
        "transcript": transcript,
    }


# ────────────────────────────────────────────────────────────────────
# Regenerate
# ────────────────────────────────────────────────────────────────────

@router.post("/regenerate/{video_id}")
async def regenerate_questions(
    video_id: str,
    category: str | None = Query(None, description="Regenerate only this category"),
    difficulty: str | None = Query(None, description="Regenerate only this difficulty"),
):
    """Delete existing questions (optionally filtered) and regenerate."""
    video = await db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")

    # Delete existing
    deleted = await db.delete_questions(video_id, category=category)

    # Re-run question generation from stored transcript/visual analysis in background.
    asyncio.create_task(
        regenerate_questions_from_existing(
            video_id,
            category=category,
            difficulty=difficulty,
        )
    )

    return {
        "status": "regenerating",
        "deleted": deleted,
        "message": f"Deleted {deleted} questions. Regeneration started.",
    }


# ────────────────────────────────────────────────────────────────────
# Export
# ────────────────────────────────────────────────────────────────────

@router.get("/export/{video_id}")
async def export_questions(
    video_id: str,
    format: str = Query("json", description="Export format: json or csv"),
    category: str | None = Query(None),
    difficulty: str | None = Query(None),
):
    """Export questions as JSON or CSV with MCQ options and explanations."""
    questions = await db.get_questions(video_id, category=category, difficulty=difficulty)

    if not questions:
        raise HTTPException(404, "No questions found")

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "question_text", "mc_option_a", "mc_option_b", "mc_option_c", "mc_option_d",
            "correct_option", "explanation", "answer_text", "category", "persona",
            "difficulty", "difficulty_score", "quality_score", "novelty_score", "visual_refs",
        ])
        writer.writeheader()
        for q in questions:
            options = q.get("mc_options", [])
            correct_idx = q.get("correct_option", 0)
            writer.writerow({
                "question_text": q.get("question_text", ""),
                "mc_option_a": options[0] if len(options) > 0 else "",
                "mc_option_b": options[1] if len(options) > 1 else "",
                "mc_option_c": options[2] if len(options) > 2 else "",
                "mc_option_d": options[3] if len(options) > 3 else "",
                "correct_option": correct_idx,
                "explanation": q.get("explanation", ""),
                "answer_text": q.get("answer_text", ""),
                "category": q.get("category", ""),
                "persona": q.get("persona", ""),
                "difficulty": q.get("difficulty", ""),
                "difficulty_score": q.get("difficulty_score", ""),
                "quality_score": q.get("quality_score", ""),
                "novelty_score": q.get("novelty_score", ""),
                "visual_refs": json.dumps(q.get("visual_refs", [])),
            })
        content = output.getvalue()
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=questions_{video_id}.csv"},
        )
    else:
        return JSONResponse(
            content={"video_id": video_id, "questions": questions},
            headers={"Content-Disposition": f"attachment; filename=questions_{video_id}.json"},
        )
