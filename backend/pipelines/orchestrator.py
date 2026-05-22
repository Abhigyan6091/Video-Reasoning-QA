"""
Pipeline orchestrator – coordinates the full video processing pipeline
and emits progress updates for the frontend via SSE.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import AsyncGenerator

from backend.config import settings
from backend import database as db
from backend.pipelines.ingestion import ingest_video
from backend.pipelines.scene_segmentation import detect_scenes
from backend.pipelines.frame_extraction import extract_keyframes
from backend.pipelines.audio_processing import extract_audio, transcribe_audio, transcript_has_text
from backend.pipelines.video_understanding import analyze_all_scenes
from backend.scene_graph.graph_builder import build_scene_graph, get_graph_context_for_questions
from backend.scene_graph.memory import add_to_index, load_index
from backend.question_engine.generator import generate_all_questions
from backend.question_engine.difficulty import estimate_all
from backend.question_engine.uniqueness import deduplicate_across_categories
from backend.question_engine.quality_filter import filter_questions
from backend.services.tts import generate_explanation_audio

logger = logging.getLogger(__name__)

# Global progress store: video_id -> list of progress messages
_progress: dict[str, list[dict]] = {}


def get_progress(video_id: str) -> list[dict]:
    return _progress.get(video_id, [])


def _emit(video_id: str, step: str, pct: int, detail: str = ""):
    msg = {"step": step, "percent": pct, "detail": detail}
    if video_id not in _progress:
        _progress[video_id] = []
    _progress[video_id].append(msg)
    logger.info(f"[{video_id}] {step} ({pct}%) {detail}")


def _representative_frame_url(video_id: str, scene_id: str) -> str:
    scene_frame_dir = Path(settings.FRAMES_DIR) / video_id / scene_id
    if not scene_frame_dir.exists():
        return ""

    frames = sorted(scene_frame_dir.glob("*.jpg"))
    if not frames:
        return ""

    rel = frames[0].relative_to(settings.FRAMES_DIR)
    return f"/frames/{rel.as_posix()}"


async def process_video(video_id: str) -> dict:
    """
    Run the full processing pipeline:
      1. Scene segmentation
      2. Frame extraction
      3. Audio extraction + transcription
      4. Video understanding (Gemini)
      5. Scene graph construction
      6. Memory indexing
      7. Question generation
      8. Difficulty estimation
      9. Uniqueness filtering
     10. Persist questions

    Returns summary dict with question counts.
    """
    _progress[video_id] = []

    try:
        await db.update_video_status(video_id, "processing")
        video = await db.get_video(video_id)
        if not video:
            raise ValueError(f"Video {video_id} not found")

        video_path = video["filepath"]

        # ── 1. Scene Segmentation ──────────────────────────────────
        _emit(video_id, "scene_segmentation", 5, "Detecting scene boundaries…")
        scenes = await detect_scenes(video_id, video_path)
        _emit(video_id, "scene_segmentation", 15,
              f"Found {len(scenes)} scenes")

        # ── 2. Frame Extraction ────────────────────────────────────
        _emit(video_id, "frame_extraction", 20, "Extracting keyframes…")
        frame_map = await asyncio.to_thread(
            extract_keyframes, video_path, video_id, scenes
        )
        total_frames = sum(len(v) for v in frame_map.values())
        _emit(video_id, "frame_extraction", 30,
              f"Extracted {total_frames} keyframes")

        # ── 3. Audio Processing ────────────────────────────────────
        transcript = await transcribe_video_to_db(video_id, video_path, emit_progress=True)

        # ── 4. Video Understanding ─────────────────────────────────
        _emit(video_id, "video_understanding", 55, "Analyzing scenes with AI…")
        analyses = await analyze_all_scenes(video_id, scenes, frame_map, transcript)
        _emit(video_id, "video_understanding", 70,
              f"Analyzed {len(analyses)} scenes")

        # ── 5. Scene Graph Construction ────────────────────────────
        _emit(video_id, "graph_construction", 72, "Building scene graphs…")
        graph_data = build_scene_graph(scenes, analyses)
        graph_context = get_graph_context_for_questions(graph_data)
        _emit(video_id, "graph_construction", 75, "Graphs constructed")

        # ── 6. Memory Indexing ─────────────────────────────────────
        _emit(video_id, "memory_indexing", 77, "Indexing scene embeddings…")
        load_index()
        scene_texts = [a.get("summary", "") for a in analyses]
        scene_meta = [{"video_id": video_id, "scene_id": s["id"],
                       "scene_idx": s["scene_idx"], "type": "scene_summary"}
                      for s in scenes]
        add_to_index(scene_texts, scene_meta)
        _emit(video_id, "memory_indexing", 80, "Scenes indexed in vector store")

        # ── 7. Question Generation ─────────────────────────────────
        _emit(video_id, "question_generation", 82, "Generating reasoning questions…")
        raw_questions = await generate_all_questions(
            graph_context=graph_context,
            transcript=transcript.get("text", ""),
            target_language=video.get("language", "english"),
        )
        _emit(video_id, "question_generation", 88,
              f"Generated {len(raw_questions)} raw questions")

        # ── 8. Quality Filtering ───────────────────────────────────
        _emit(video_id, "quality_filter", 89, "Filtering low-quality questions…")
        high_quality_questions = filter_questions(
            raw_questions,
            graph_context=graph_context,
            transcript=transcript.get("text", ""),
            threshold=0.6
        )
        _emit(video_id, "quality_filter", 91,
              f"{len(high_quality_questions)} high-quality questions retained")

        # ── 9. Difficulty Estimation ───────────────────────────────
        _emit(video_id, "difficulty_estimation", 92, "Estimating difficulty…")
        scored_questions = estimate_all(high_quality_questions, len(scenes))

        # ── 10. Uniqueness Filtering ───────────────────────────────
        _emit(video_id, "uniqueness_filter", 94, "Removing duplicate questions…")
        unique_questions = deduplicate_across_categories(scored_questions)
        _emit(video_id, "uniqueness_filter", 96,
              f"{len(unique_questions)} unique questions after filtering")

        # ── 11. Persist Questions ──────────────────────────────────
        _emit(video_id, "saving", 97, "Saving questions to database…")

        # Also index question embeddings
        q_texts = [q["question_text"] for q in unique_questions]
        q_meta = [{"video_id": video_id, "question_text": q["question_text"],
                    "category": q["category"], "type": "question"}
                   for q in unique_questions]
        if q_texts:
            emb_ids = add_to_index(q_texts, q_meta)
            for q, eid in zip(unique_questions, emb_ids):
                q["embedding_id"] = eid

        # Ensure every question has an ID for stable file naming
        for q in unique_questions:
            if "id" not in q:
                q["id"] = str(uuid.uuid4())

        # ── 12. Audio Explanations ─────────────────────────────────
        _emit(video_id, "audio_processing", 98, "Generating audio explanations…")
        lang = video.get("language", "english")
        tts_tasks = [
            generate_explanation_audio(q["id"] if "id" in q else f"{video_id}_{i}", q["explanation"], lang)
            for i, q in enumerate(unique_questions)
        ]
        audio_paths = await asyncio.gather(*tts_tasks)
        for q, path in zip(unique_questions, audio_paths):
            q["audio_path"] = path

        await db.create_questions(video_id, unique_questions)
        await db.update_video_status(video_id, "completed")

        # ── Summary ───────────────────────────────────────────────
        summary = {
            "video_id": video_id,
            "scenes": len(scenes),
            "total_questions": len(unique_questions),
            "by_category": {},
            "by_difficulty": {},
        }
        for q in unique_questions:
            cat = q.get("category", "unknown")
            diff = q.get("difficulty", "unknown")
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            summary["by_difficulty"][diff] = summary["by_difficulty"].get(diff, 0) + 1

        _emit(video_id, "completed", 100,
              f"Done! {len(unique_questions)} questions generated.")

        return summary

    except Exception as e:
        logger.exception(f"Pipeline failed for video {video_id}")
        await db.update_video_status(video_id, "failed", str(e))
        _emit(video_id, "error", -1, str(e))
        raise


async def transcribe_video_to_db(
    video_id: str,
    video_path: str | None = None,
    emit_progress: bool = False,
) -> dict:
    """Extract audio, transcribe it, store the result, and recover from empty runs."""
    video = await db.get_video(video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found")

    video_path = video_path or video["filepath"]
    if emit_progress:
        _emit(video_id, "audio_processing", 35, "Extracting audio…")

    transcript: dict
    try:
        audio_path = await asyncio.to_thread(extract_audio, video_path, video_id)
        if emit_progress:
            _emit(video_id, "audio_processing", 40, "Transcribing audio…")

        transcript = await asyncio.to_thread(transcribe_audio, audio_path)
        if transcript_has_text(transcript):
            transcript["status"] = "ready"
        else:
            transcript["status"] = "empty"
            transcript["warning"] = "Whisper returned no text even though audio extraction succeeded."

        if emit_progress:
            segment_count = len(transcript.get("segments", []))
            _emit(
                video_id,
                "audio_processing",
                50,
                f"Transcribed {segment_count} segments ({transcript.get('language', 'unknown')} detected)",
            )
    except Exception as e:
        logger.warning(f"Audio processing failed: {e}. Trying cached transcript fallback.")
        error_msg = str(e)
        if emit_progress:
            _emit(video_id, "audio_processing", 50, f"Transcription error: {error_msg}")
        transcript = {
            "text": "",
            "segments": [],
            "language": "error",
            "status": "unavailable",
            "error": error_msg,
        }

    if not transcript_has_text(transcript):
        cached = await db.get_latest_transcript_for_filename(
            video["filename"],
            exclude_video_id=video_id,
        )
        if cached:
            transcript = cached
            transcript["status"] = "reused"
            transcript["warning"] = (
                "The fresh transcription was empty, so this transcript was reused "
                "from a previous upload of the same filename."
            )
            if emit_progress:
                _emit(video_id, "audio_processing", 50, "Reused transcript from matching earlier upload")

    await db.update_video_transcript(video_id, transcript)
    return transcript


async def regenerate_questions_from_existing(
    video_id: str,
    category: str | None = None,
    difficulty: str | None = None,
) -> dict:
    """Regenerate questions from stored transcript, scene summaries, events, and frames."""
    _progress[video_id] = []
    _emit(video_id, "question_generation", 10, "Loading stored transcript and visual analysis…")
    video = await db.get_video(video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found")

    scenes = await db.get_scenes(video_id)
    events = await db.get_events(video_id)
    transcript = await db.get_video_transcript(video_id) or {}

    if not scenes or not events:
        logger.info("No stored analysis for %s; falling back to full processing.", video_id)
        return await process_video(video_id)

    _emit(video_id, "graph_construction", 25, "Rebuilding long-context video map…")
    events_by_scene = {event["scene_id"]: event for event in events}
    analyses = []
    for scene in scenes:
        event = events_by_scene.get(scene["id"], {})
        analyses.append({
            "scene_id": scene["id"],
            "scene_idx": scene["scene_idx"],
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "summary": scene.get("summary") or event.get("summary", ""),
            "objects": event.get("objects") or [],
            "actions": event.get("actions") or [],
            "interactions": event.get("interactions") or [],
            "emotions": event.get("emotions") or [],
            "anomalies": event.get("anomalies") or [],
            "mood": "neutral",
            "representative_frame": _representative_frame_url(video_id, scene["id"]),
        })

    graph_data = build_scene_graph(scenes, analyses)
    graph_context = get_graph_context_for_questions(graph_data)

    _emit(video_id, "question_generation", 45, "Generating long-context MCQs…")
    raw_questions = await generate_all_questions(
        graph_context=graph_context,
        transcript=transcript.get("text", ""),
        categories=[category] if category else None,
        target_language=video.get("language", "english"),
    )
    high_quality_questions = filter_questions(
        raw_questions,
        graph_context=graph_context,
        transcript=transcript.get("text", ""),
        threshold=0.6,
    )
    _emit(video_id, "quality_filter", 70, f"{len(high_quality_questions)} questions passed quality checks")
    scored_questions = estimate_all(high_quality_questions, len(scenes))
    unique_questions = deduplicate_across_categories(scored_questions)

    if difficulty:
        unique_questions = [q for q in unique_questions if q.get("difficulty") == difficulty]

    q_texts = [q["question_text"] for q in unique_questions]
    q_meta = [
        {
            "video_id": video_id,
            "question_text": q["question_text"],
            "category": q["category"],
            "type": "question",
        }
        for q in unique_questions
    ]
    if q_texts:
        _emit(video_id, "memory_indexing", 85, "Indexing regenerated questions…")
        load_index()
        emb_ids = add_to_index(q_texts, q_meta)
        for q, eid in zip(unique_questions, emb_ids):
            q["embedding_id"] = eid

    # Ensure every question has an ID for stable file naming
    for q in unique_questions:
        if "id" not in q:
            q["id"] = str(uuid.uuid4())

    # Indexing ...
    
    # ── Audio Explanations ─────────────────────────────────
    _emit(video_id, "audio_processing", 90, "Generating audio explanations…")
    lang = video.get("language", "english")
    tts_tasks = [
        generate_explanation_audio(q["id"] if "id" in q else f"{video_id}_regen_{i}", q["explanation"], lang)
        for i, q in enumerate(unique_questions)
    ]
    audio_paths = await asyncio.gather(*tts_tasks)
    for q, path in zip(unique_questions, audio_paths):
        q["audio_path"] = path

    _emit(video_id, "saving", 95, "Saving regenerated MCQs…")
    await db.create_questions(video_id, unique_questions)
    await db.update_video_status(video_id, "completed")
    _emit(video_id, "completed", 100, f"Done! {len(unique_questions)} MCQs regenerated.")

    return {
        "video_id": video_id,
        "total_questions": len(unique_questions),
        "by_category": {},
        "by_difficulty": {},
    }
