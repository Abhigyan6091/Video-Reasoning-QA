"""
Audio extraction and transcription.
Uses FFmpeg to extract audio and Whisper for transcription.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from backend.config import settings

# Lazy-loaded whisper model
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(settings.WHISPER_MODEL)
    return _whisper_model


def extract_audio(video_path: str, video_id: str) -> str:
    """
    Extract audio track from video using FFmpeg.
    Returns path to the extracted WAV file.
    """
    out_dir = settings.AUDIO_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "audio.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz sample rate (whisper needs this)
        "-ac", "1",               # mono
        str(audio_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return str(audio_path)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using Whisper.
    Returns dict with 'text' (full transcript) and 'segments' (timestamped).
    """
    model = _get_whisper()
    result = model.transcribe(audio_path, language=None, verbose=False)

    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })

    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
        "language": result.get("language", "unknown"),
    }


def transcript_has_text(transcript: dict) -> bool:
    """Return True when Whisper produced usable words in text or segments."""
    if (transcript.get("text") or "").strip():
        return True
    return any((seg.get("text") or "").strip() for seg in transcript.get("segments", []))


def get_transcript_for_scene(transcript: dict,
                             start_time: float,
                             end_time: float) -> str:
    """
    Extract transcript text that falls within a scene's time range.
    """
    parts: list[str] = []
    for seg in transcript.get("segments", []):
        # Include segment if it overlaps with the scene
        if seg["end"] > start_time and seg["start"] < end_time:
            parts.append(seg["text"])
    return " ".join(parts).strip()
