import asyncio
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from huggingface_hub import hf_hub_download

import soundfile as sf

logger = logging.getLogger("app.audio_generator")
logger.setLevel(logging.INFO)

# Device
device = torch.device("cpu")
logger.info(f"Audio generator configured to use device: {device}")

# Global model state
processor: Optional[SpeechT5Processor] = None
model: Optional[SpeechT5ForTextToSpeech] = None
vocoder: Optional[SpeechT5HifiGan] = None
speaker_embedding: Optional[torch.Tensor] = None

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_ROOT / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

_EMBEDDING_CANDIDATES = [
    "cmu_us_slt_arctic-xvectors.npy",
    "cmu-arctic-xvectors.npy",
    "cmu_us_slt_arctic_xvectors.npy",
    "xvectors.npy",
]

def _insert_light_commas(text: str, every_n_words: int = 8) -> str:
    words = text.strip().split()
    if len(words) <= every_n_words:
        return text.strip()
    out = []
    for i, w in enumerate(words):
        out.append(w)
        if (i + 1) % every_n_words == 0 and i + 1 < len(words):
            if not out[-1].endswith((".", "!", "?", ",")):
                out[-1] = out[-1] + ","
    return " ".join(out)

def _apply_light_emphasis(text: str) -> str:
    s = text.strip()
    if not s:
        return s
    if s[-1] not in ".!?":
        s = s + "."
    if len(s.split()) <= 6 and not s.endswith("!"):
        s = s[:-1] + "!"
    return s

def _preprocess_text_for_tts(text: str) -> str:
    if not text or not text.strip():
        return text
    t = " ".join(text.strip().split())
    t = _insert_light_commas(t, every_n_words=8)
    t = _apply_light_emphasis(t)
    return t

async def initialize_tts_models(force_reload: bool = False):
    global processor, model, vocoder, speaker_embedding
    if all([processor, model, vocoder, speaker_embedding]) and not force_reload:
        logger.info("TTS models already loaded.")
        return
    logger.info("Loading SpeechT5 models from Hugging Face...")
    try:
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        logger.info("Core SpeechT5 models loaded.")
    except Exception as e:
        logger.exception("Failed to load core SpeechT5 models.")
        processor = model = vocoder = None
        return
    tried_sources: List[str] = []
    speaker_loaded = False
    repo_id = "Matthijs/cmu-arctic-xvectors"
    for fname in _EMBEDDING_CANDIDATES:
        try:
            tried_sources.append(f"{repo_id}/{fname}")
            emb_path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
            logger.info(f"Downloaded embedding file from {repo_id}: {fname}")
            arr = np.load(emb_path)
            if arr.ndim == 2:
                idx = 7306 if arr.shape[0] > 7306 else 0
                vec = arr[idx]
            else:
                vec = arr
            speaker_embedding = torch.tensor(vec).unsqueeze(0).to(device)
            speaker_loaded = True
            break
        except Exception:
            continue
        except Exception as e:
            logger.warning(f"Could not load embedding {fname} from {repo_id}: {e}", exc_info=True)
            continue
    if not speaker_loaded:
        alt_repo = "Matthijs/cmu-arctic-xvectors"
        try:
            emb_path = hf_hub_download(repo_id=alt_repo, filename="cmu_us_arctic-xvectors.npy", repo_type="dataset")
            arr = np.load(emb_path)
            speaker_embedding = torch.tensor(arr[0]).unsqueeze(0).to(device)
            speaker_loaded = True
        except Exception:
            pass
    if not speaker_loaded:
        logger.warning(
            f"Could not download speaker embedding. Using random vector. Searched: {tried_sources}"
        )
        fallback_dim = 512
        rand_vec = np.random.normal(scale=0.75, size=(1, fallback_dim)).astype(np.float32)
        speaker_embedding = torch.tensor(rand_vec).to(device)
    logger.info("Speaker embedding ready (may be fallback). TTS is usable.")

async def generate_audio_from_text(text: str, output_filename: str) -> str:
    if not all([processor, model, vocoder, speaker_embedding]):
        raise RuntimeError("TTS models not initialized. Call initialize_tts_models() first.")
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")
    tts_text = _preprocess_text_for_tts(text)
    logger.info(f"Generating audio for text (first 120 chars): {tts_text[:120]!r}")
    out_path = AUDIO_DIR / output_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _sync_infer(text_input: str, out_file: Path):
        inputs = processor(text=text_input, return_tensors="pt", max_length=600, truncation=True).to(device)
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)
        audio_np = speech.cpu().numpy()
        samplerate = 16000
        sf.write(str(out_file), audio_np, samplerate=samplerate)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _sync_infer, tts_text, out_path)
    logger.info(f"✅ Audio saved to: {out_path}")
    return str(out_path)

if __name__ == "__main__":
    async def _test():
        await initialize_tts_models()
        sample = "Push forward. Do the work. The small steps win the race."
        try:
            p = await generate_audio_from_text(sample, "test_slt_energy.wav")
            print("Generated:", p)
        except Exception as e:
            print("Failed:", e)
    asyncio.run(_test())