
import asyncio
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf

logger = logging.getLogger("app.audio_generator")
logger.setLevel(logging.INFO)

# ==============================================================
# GLOBALS
# ==============================================================
processor: Optional[SpeechT5Processor] = None
model: Optional[SpeechT5ForTextToSpeech] = None
vocoder: Optional[SpeechT5HifiGan] = None
speaker_embeddings: Optional[torch.Tensor] = None

device = torch.device("cpu")
logger.info(f"Audio generator using device: {device}")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "static" / "audio"
RESOURCES_DIR = PROJECT_ROOT / "resources"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# LOAD MODELS (OFFLINE MODE)
# ==============================================================
async def initialize_tts_models(force_reload: bool = False):
    """
    Load SpeechT5 + local speaker embedding.
    No internet. Fully offline.
    """
    global processor, model, vocoder, speaker_embeddings

    if processor and model and vocoder and speaker_embeddings and not force_reload:
        logger.info("TTS models already initialized.")
        return

    logger.info("Loading SpeechT5 models (offline)...")

    try:
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        logger.info("Core SpeechT5 models loaded.")
    except Exception as e:
        logger.error("Failed to load SpeechT5 model components.", exc_info=True)
        raise RuntimeError("TTS core model load failure.") from e

    # -------------------------------
    # Load *local* speaker embeddings
    # -------------------------------
    embedding_path = RESOURCES_DIR / "spk_embeds.pt"
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Speaker embedding file not found at: {embedding_path}\n"
            f"Expected file: spk_embeds.pt"
        )

    try:
        speaker_embeddings = torch.load(embedding_path, map_location=device)
        logger.info(f"Loaded local speaker embeddings: {embedding_path}")
    except Exception as e:
        logger.error("Failed loading local speaker embeddings.", exc_info=True)
        raise RuntimeError("Failed to load speaker embeddings.") from e


# ==============================================================
# TEXT-TO-SPEECH
# ==============================================================
async def generate_audio_from_text(text: str, output_filename: str) -> str:
    """
    Generate a narration WAV file using SpeechT5 + local embedding.
    """

    # --- Ensure models are initialized ---
    if model is None:
        await initialize_tts_models()

    # --- Safe null check (avoids tensor boolean crash) ---
    if (
        processor is None
        or model is None
        or vocoder is None
        or speaker_embeddings is None
    ):
        raise RuntimeError("TTS models not fully loaded.")

    if not text or not text.strip():
        raise ValueError("Cannot generate audio: text is empty.")

    logger.info(f"[TTS] Generating audio → {output_filename}")

    out_path = AUDIO_DIR / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _infer():
        inputs = processor(text=text, return_tensors="pt").to(device)

        # Use first embedding vector
        spk = speaker_embeddings[0].unsqueeze(0).to(device)

        with torch.no_grad():
            speech = model.generate_speech(
                inputs["input_ids"],
                spk,
                vocoder=vocoder
            )

        sf.write(str(out_path), speech.cpu().numpy(), samplerate=16000)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _infer)

    logger.info(f"Audio saved: {out_path}")
    return str(out_path)
