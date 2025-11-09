# app/video_engines/speecht5_engine.py

import argparse
import io
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
from transformers import (SpeechT5ForTextToSpeech, SpeechT5HifiGan,
                          SpeechT5Processor)

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [speecht5_engine] - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# --- Environment Fix for FFmpeg ---
def find_ffmpeg_path():
    if 'CONDA_PREFIX' in os.environ:
        path = Path(os.environ['CONDA_PREFIX']) / "Library" / "bin" / "ffmpeg.exe"
        if path.is_file():
            logger.info(f"Found ffmpeg in Conda environment: {path}")
            return str(path.parent)
    path = shutil.which("ffmpeg")
    if path:
        logger.info(f"Found ffmpeg in system PATH: {path}")
        return str(Path(path).parent)
    return None

ffmpeg_dir = find_ffmpeg_path()
if ffmpeg_dir:
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
    AudioSegment.converter = str(Path(ffmpeg_dir) / "ffmpeg.exe")
else:
    logger.critical("CRITICAL: ffmpeg executable not found.")

# --- Globals ---
SPEAKER_VOICES = {
    "atlas": 0,
    "nova": 0,
    "echo": 0,
    "breeze": 0,
}


tts_processor: Optional[SpeechT5Processor] = None
tts_model: Optional[SpeechT5ForTextToSpeech] = None
tts_vocoder: Optional[SpeechT5HifiGan] = None
speaker_embeddings_tensor: Optional[torch.Tensor] = None

def initialize_models(device_str: str = "cpu"):
    global tts_processor, tts_model, tts_vocoder, speaker_embeddings_tensor
    if tts_model is not None: return
    device = torch.device(device_str)
    logger.info(f"Initializing SpeechT5 models on device: {device}")
    try:
        tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device).eval()
        tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device).eval()
        
        script_dir = Path(__file__).resolve().parent.parent.parent
        embedding_path = script_dir / "resources" / "spk_embeds.pt"
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Speaker embedding file not found at: {embedding_path}")
        
        speaker_embeddings_tensor = torch.load(embedding_path, map_location=device)
        logger.info(f"Loaded speaker embeddings tensor with shape: {speaker_embeddings_tensor.shape}")
        
        logger.info("SpeechT5 models and local speaker embeddings initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {e}") from e

def generate_audio(
    text: str,
    output_path: Path,
    voice_name: str,  # <<< THIS IS THE FIX
    target_loudness_dbfs: float = -14.0,
    min_duration_ms: int = 500,
    max_text_length: int = 1000
) -> Tuple[str, float]:
    
    if not text or not text.strip(): raise ValueError("Input text cannot be empty.")
    if len(text) > max_text_length: raise ValueError(f"Input text exceeds max length of {max_text_length} chars.")
    
    voice_name_lower = voice_name.lower()
    if voice_name_lower not in SPEAKER_VOICES:
        raise KeyError(f"Voice '{voice_name}' not found. Available: {list(SPEAKER_VOICES.keys())}")
    
    if any(x is None for x in [tts_processor, tts_model, tts_vocoder, speaker_embeddings_tensor]):
        raise RuntimeError("Model or embeddings not initialized properly.")

    device = tts_model.device
    
    inputs = tts_processor(text=text, return_tensors="pt").to(device)
    speaker_id = SPEAKER_VOICES[voice_name_lower]
    
    if speaker_id >= speaker_embeddings_tensor.shape[0]:
        raise IndexError(f"Speaker ID {speaker_id} is out of bounds for the loaded embeddings tensor which has size {speaker_embeddings_tensor.shape[0]}.")

    
    # Fix extra dimensions (ensure shape is [1, 512])
    speaker_embeddings = speaker_embeddings_tensor[speaker_id]

    # Flatten unnecessary dimensions    
    speaker_embeddings = speaker_embeddings.squeeze()

    # Ensure it's [1, 512] for SpeechT5
    if speaker_embeddings.dim() == 1:
        speaker_embeddings = speaker_embeddings.unsqueeze(0)


    with torch.no_grad():
        speech = tts_model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embeddings,
            vocoder=tts_vocoder
        )
        waveform = speech.cpu().numpy()

    if waveform.size == 0: raise ValueError("Generated audio waveform was empty.")
    waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

    buffer = io.BytesIO()
    sampling_rate = 16000
    sf.write(buffer, waveform, sampling_rate, format='WAV')
    buffer.seek(0)

    audio_segment = AudioSegment.from_file(buffer, format="wav")
    
    if target_loudness_dbfs is not None:
        audio_segment = pydub_normalize(audio_segment, headroom=abs(target_loudness_dbfs))
    
    if len(audio_segment) < min_duration_ms:
        silence = AudioSegment.silent(duration=(min_duration_ms - len(audio_segment)))
        audio_segment += silence

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_segment.export(str(output_path), format="mp3")
    duration = len(audio_segment) / 1000.0
    
    logger.info(f"Generated audio for voice '{voice_name}' (ID {speaker_id}): {output_path.name} ({duration:.2f}s)")
    return str(output_path), duration



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpeechT5 Audio Generation Engine")
    # Make --text NOT required
    parser.add_argument("--text", type=str, help="Text to synthesize.")
    parser.add_argument("--text-file", type=str, help="Path to a text file to read for synthesis.")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--voice", type=str, default="nova", choices=list(SPEAKER_VOICES.keys())
    )
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()

    # --- THIS IS THE CRITICAL LOGIC ---
    text_to_synthesize = ""
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text_to_synthesize = f.read()
    elif args.text:
        text_to_synthesize = args.text
    else:
        # If neither is provided, exit with an error
        logger.critical("Error: You must provide either --text or a valid --text-file.")
        print(json.dumps({"status": "FAILED", "error": "No text input provided."}))
        sys.exit(1)
    

    try:
        initialize_models(args.device)
        # Now we use the new variable that is guaranteed to have the text
        final_path, duration = generate_audio(
            text=text_to_synthesize,
            output_path=Path(args.output_path),
            voice_name=args.voice 
        )
        result = {"status": "COMPLETED", "file": final_path, "duration_sec": duration}
        print(json.dumps(result))
        sys.exit(0)
    except Exception as e:
        error_code = "UNKNOWN_ERROR"
        if isinstance(e, (ValueError, KeyError)): error_code = "INVALID_INPUT"
        elif isinstance(e, RuntimeError): error_code = "MODEL_RUNTIME_ERROR"
        
        result = {"status": "FAILED", "error": {"code": error_code, "message": str(e)}}
        logger.critical(f"An unrecoverable error occurred: {e}", exc_info=True)
        print(json.dumps(result))
        sys.exit(1)