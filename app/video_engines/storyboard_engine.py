import argparse
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

from groq import Groq, BadRequestError
from dotenv import load_dotenv

# ✅ Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Preferred Groq Models (fallback if one is deprecated)
PREFERRED_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768"
]

def load_api_key() -> str:
    """Load Groq API key safely from environment or .env."""
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found in environment or .env file.")
    return api_key

def pick_available_model(client: Groq) -> str:
    """Automatically choose a valid available model from Groq."""
    try:
        models = client.models.list()
        available = {m.id for m in models.data}
        for model in PREFERRED_MODELS:
            if model in available:
                logging.info(f"✅ Using model: {model}")
                return model
        raise ValueError(f"No preferred models available: {PREFERRED_MODELS}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to check available models: {e}")
        return PREFERRED_MODELS[0]  # Default fallback

def calculate_scene_durations(storyboard: List[Dict], timestamp_data: Optional[List[Dict]]) -> List[Dict]:
    """Add 'start_time', 'end_time', 'duration' to each scene based on speech timestamps."""
    if not timestamp_data:
        logging.warning("⚠️ No timestamp data provided. Skipping duration calculation.")
        return storyboard

    total_time = timestamp_data[-1]["end"]
    scene_count = len(storyboard)
    average_duration = total_time / scene_count

    current_time = 0.0
    for scene in storyboard:
        scene["start_time"] = round(current_time, 2)
        scene["end_time"] = round(current_time + average_duration, 2)
        scene["duration"] = round(average_duration, 2)
        current_time += average_duration

    logging.info("✅ Added duration timing to scenes.")
    return storyboard

def generate_storyboard(quote_text: str, timestamp_data: Optional[List[Dict]] = None) -> List[Dict]:
    api_key = load_api_key()
    client = Groq(api_key=api_key)

    model_name = pick_available_model(client)

    system_prompt = """
    You are a creative director. Break the quote into 2-4 visual scenes.
    Return ONLY JSON. Format:
    [
      {"scene_description": "...", "animation_prompt": "..."},
      ...
    ]
    Rules:
    - Prompts must be visual, cinematic, no text or words inside the image.
    - No extra commentary or text outside JSON.
    """

    logging.info("🚀 Requesting Groq for storyboard...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Quote: \"{quote_text}\""}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
    except BadRequestError as e:
        raise RuntimeError(f"❌ Groq API error: {e}")

    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        storyboard = parsed.get("scenes", parsed.get("storyboard", parsed))
        if not isinstance(storyboard, list):
            raise ValueError("Storyboard is not a list.")
    except Exception as e:
        logging.error(f"❌ Failed to parse storyboard JSON: {raw}")
        raise

    logging.info(f"✅ Storyboard generated with {len(storyboard)} scenes.")
    return calculate_scene_durations(storyboard, timestamp_data)

def main():
    parser = argparse.ArgumentParser(description="Generate storyboard with optional scene durations.")
    parser.add_argument("--quote", required=True, help="Quote text to generate storyboard from.")
    parser.add_argument("--timestamps", required=False, help="Path to JSON with word-level timestamps.")
    args = parser.parse_args()

    start = time.time()
    try:
        timestamp_data = None
        if args.timestamps and os.path.exists(args.timestamps):
            with open(args.timestamps, "r") as f:
                timestamp_data = json.load(f)

        scenes = generate_storyboard(args.quote, timestamp_data)
        output = {"status": "COMPLETED", "storyboard": scenes}

    except Exception as e:
        logging.error(e, exc_info=True)
        output = {"status": "FAILED", "error": str(e)}

    print(json.dumps(output, indent=2))
    logging.info(f"✅ Finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()
