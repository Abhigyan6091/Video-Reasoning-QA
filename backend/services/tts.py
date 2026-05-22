"""
Text-to-Speech (TTS) service using edge-tts.
Generates premium-sounding voiceovers for question explanations.
"""
import logging
import asyncio
from pathlib import Path
import edge_tts

from backend.config import settings

logger = logging.getLogger(__name__)

# Map internal language codes to edge-tts voices
VOICE_MAP = {
    "english": "en-US-GuyNeural",
    "hindi": "hi-IN-MadhurNeural",
    "mixed": "hi-IN-SwaraNeural", # Swara is good for Hinglish/Mixed
}

async def generate_explanation_audio(question_id: str, text: str, language: str = "english") -> str | None:
    """
    Generate an MP3 file for the given text and return the relative URL/path.
    """
    if not text:
        return None

    voice = VOICE_MAP.get(language.lower(), VOICE_MAP["english"])
    
    filename = f"{question_id}.mp3"
    output_path = settings.EXPLANATION_AUDIO_DIR / filename
    
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(output_path))
        
        # Return the "URL" path for static serving
        return f"/explanations/{filename}"
    except Exception as e:
        logger.error(f"TTS generation failed for {question_id}: {e}")
        return None
