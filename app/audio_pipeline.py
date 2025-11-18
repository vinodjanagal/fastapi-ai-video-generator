# file: app/audio_pipeline.py
import uuid
import logging
from pathlib import Path
from typing import Optional



# Import your existing SpeechT5 TTS module; based on your shared code it's app.audio_generator
from app.audio_generator import initialize_tts_models, generate_audio_from_text

import uuid
from app.audio_generator import generate_audio_from_text

async def generate_narration_audio(text: str) -> str:
    filename = f"narration_{uuid.uuid4().hex[:8]}.wav"
    return await generate_audio_from_text(text, filename)

logger = logging.getLogger("app.audio_pipeline")


async def generate_narration_audio(text: str, filename: Optional[str] = None) -> str:
    """
    High-level wrapper:
      - ensures the SpeechT5 models are initialized,
      - generates a unique filename if not provided,
      - writes the file to static/audio/ and returns the file path as string.
    """
    await initialize_tts_models()

    if not filename:
        uid = uuid.uuid4().hex[:8]
        filename = f"narration_{uid}.wav"

    logger.info(f"[audio_pipeline] Generating narration audio -> {filename}")
    out_path = await generate_audio_from_text(text, filename)
    logger.info(f"[audio_pipeline] Audio ready -> {out_path}")
    return out_path
