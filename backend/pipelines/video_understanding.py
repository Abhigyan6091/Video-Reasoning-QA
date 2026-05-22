"""
Video understanding – sends keyframes + transcript to Gemini API
for structured scene analysis.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import ollama

from backend.config import settings
from backend import database as db

logger = logging.getLogger(__name__)


def _frame_url(frame_path: str) -> str:
    path = Path(frame_path)
    try:
        rel = path.relative_to(settings.FRAMES_DIR)
    except ValueError:
        return ""
    return f"/frames/{rel.as_posix()}"


_SCENE_ANALYSIS_PROMPT = """You are an expert video analyst specializing in educational and technical content. Your goal is to extract the CORE MESSAGE and INTENT of this scene.

Scene info:
- Scene index: {scene_idx}
- Time range: {start_time:.1f}s – {end_time:.1f}s

Transcript for this scene (PRIMARY SOURCE OF TRUTH):
\"\"\"{transcript}\"\"\"

Analyze the keyframes and provide a detailed JSON response. 

CRITICAL INSTRUCTIONS:
1. USE THE TRANSCRIPT to understand what is being taught or discussed.
2. Focus visual analysis ONLY on elements that support the transcript.
3. IGNORE incidental background logos, watermarks, UI chrome, or random objects unless they are specifically mentioned/demonstrated by the narrator.
4. If the narrator is explaining a concept (e.g., embeddings), describe how the visual helps illustrate that concept.

{{
  "objects": ["Only conceptually relevant objects mentioned or demonstrated"],
  "actions": ["Primary technical actions or demonstrations"],
  "interactions": ["Interactions that further the educational/narrative goal"],
  "emotions": ["Relevant emotional tone of the speaker or characters"],
  "anomalies": ["Structural or logical inconsistencies, not random visual noise"],
  "important_events": ["Key conceptual milestones reached in this scene"],
  "summary": "A 2-3 sentence summary focusing on WHAT is being explained and HOW it is shown.",
  "setting": "The conceptual or technical environment",
  "mood": "Overall educational/professional tone"
}}

Return ONLY valid JSON, no markdown fences or extra text."""


async def analyze_scene(scene: dict, frame_paths: list[str],
                        transcript_text: str) -> dict:
    """
    Send scene keyframes + transcript to Ollama for structured analysis.
    Returns parsed analysis dict and also stores as an event in DB.
    """
    prompt = _SCENE_ANALYSIS_PROMPT.format(
        scene_idx=scene["scene_idx"],
        start_time=scene["start_time"],
        end_time=scene["end_time"],
        transcript=transcript_text or "(no speech in this scene)",
    )

    try:
        client = ollama.AsyncClient(host=settings.OLLAMA_URL)
        
        # Add up to 5 keyframes to avoid token limits
        response = await client.chat(
            model=settings.OLLAMA_VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': frame_paths[:min(len(frame_paths), 5)]
            }],
            options={'temperature': 0}
        )
        
        raw = response['message']['content'].strip()

        # Clean up potential markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        if raw.startswith("json"):
            raw = raw[4:]

        analysis = json.loads(raw.strip())
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Scene analysis failed for scene {scene['scene_idx']}: {e}")
        analysis = {
            "objects": [], "actions": [], "interactions": [],
            "emotions": [], "anomalies": [], "important_events": [],
            "summary": f"Scene from {scene['start_time']:.1f}s to {scene['end_time']:.1f}s. "
                       f"Transcript: {transcript_text[:200] if transcript_text else 'none'}",
            "setting": "unknown", "mood": "neutral",
        }

    # Store event in DB
    await db.create_event(
        scene_id=scene["id"],
        video_id=scene["video_id"],
        objects=analysis.get("objects"),
        actions=analysis.get("actions"),
        interactions=analysis.get("interactions"),
        emotions=analysis.get("emotions"),
        anomalies=analysis.get("anomalies"),
        summary=analysis.get("summary"),
        timestamp=f"{scene['start_time']:.1f}s",
    )

    # Update scene summary
    await db.update_scene_summary(scene["id"], analysis.get("summary", ""))

    return analysis


async def analyze_all_scenes(video_id: str, scenes: list[dict],
                             frame_map: dict[str, list[str]],
                             transcript: dict) -> list[dict]:
    """
    Analyze all scenes sequentially (respects API rate limits).
    Returns list of analysis dicts.
    """
    from backend.pipelines.audio_processing import get_transcript_for_scene

    analyses = []
    for scene in scenes:
        scene_transcript = get_transcript_for_scene(
            transcript, scene["start_time"], scene["end_time"]
        )
        frame_paths = frame_map.get(scene["id"], [])
        analysis = await analyze_scene(scene, frame_paths, scene_transcript)
        analysis["scene_id"] = scene["id"]
        analysis["scene_idx"] = scene["scene_idx"]
        analysis["start_time"] = scene["start_time"]
        analysis["end_time"] = scene["end_time"]
        analysis["representative_frame"] = _frame_url(frame_paths[0]) if frame_paths else ""
        analysis["keyframes"] = [_frame_url(path) for path in frame_paths[:3]]
        analyses.append(analysis)

    return analyses
