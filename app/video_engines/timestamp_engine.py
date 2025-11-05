import argparse
import json
import sys
from faster_whisper import WhisperModel
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_timestamps(audio_path: str, model_size: str = "tiny.en"):
    logging.info(f"Loading Whisper model '{model_size}'...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    logging.info(f"Transcribing audio file: {audio_path}")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)

    all_words = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                all_words.append({
                    "word": word.word.strip(),
                    "start": word.start,
                    "end": word.end
                })
    
    logging.info(f"Timestamp generation complete. Found {len(all_words)} words.")
    return all_words

def main():
    parser = argparse.ArgumentParser(description="Generate word-level timestamps from an audio file.")
    parser.add_argument("--audio-path", required=True, help="Path to the input audio file.")
    args = parser.parse_args()
    
    start_time = time.time()
    try:
        timestamp_data = generate_timestamps(args.audio_path)
        output = { "status": "COMPLETED", "timestamps": timestamp_data }
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        output = { "status": "FAILED", "error": str(e) }

    print(json.dumps(output, indent=2))
    
    end_time = time.time()
    logging.info(f"Engine finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()