import asyncio
import logging
from pathlib import Path
from gtts import gTTS

logger = logging.getLogger("app.audio_generator")

# This function is not needed for gTTS as there are no models to load.
async def initialize_tts_models():
    logger.info("Using gTTS (API-based), no models to load.")
    pass


async def generate_audio_from_text(text: str, output_filename: str) -> str:
    """
    Generates audio using gTTS and saves it to the EXACT path specified.
    This function no longer makes its own decisions about file locations.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")
    
    # The 'output_filename' argument is now expected to be a full, ready-to-use path.
    # We create a Path object from it to ensure the parent directory exists.
    out_path = Path(output_filename)
    
    # Ensure the directory where we want to save the file actually exists.
    # For example, it ensures 'static/videos/' is created before trying to save a file in it.
    out_path.parent.mkdir(parents=True, exist_ok=True)
        
    logger.info(f"Generating audio with gTTS for text: {text[:50]}...")
    
    def _blocking_gtts():
        # gTTS runs in a separate thread, so we pass it the string version of the path.
        tts = gTTS(text, lang='en', slow=False)
        tts.save(str(out_path))

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _blocking_gtts)
    
    logger.info(f"✅ Audio saved to: {out_path}")
    return str(out_path)