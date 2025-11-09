# app/video_engines/storyboard_engine.py
# FINAL, VERIFIED, AND WORKING VERSION
import argparse
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

from groq import Groq, BadRequestError
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PREFERRED_MODELS = ["llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]

def load_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found.")
    return api_key

def pick_available_model(client: Groq) -> str:
    try:
        models = client.models.list()
        available = {m.id for m in models.data}
        for model in PREFERRED_MODELS:
            if model in available:
                logger.info(f"✅ Using model: {model}")
                return model
        raise ValueError(f"No preferred models available: {PREFERRED_MODELS}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to check available models: {e}")
        return PREFERRED_MODELS[0]

def calculate_scene_durations(storyboard: List[Dict], timestamp_data: Optional[List[Dict]]) -> List[Dict]:
    if not timestamp_data:
        return storyboard
    total_time = 0
    if timestamp_data:
        for segment in reversed(timestamp_data):
            if "words" in segment and segment["words"]:
                total_time = segment["words"][-1]["end"]
                break
    if total_time == 0:
        return storyboard
    scene_count = len(storyboard)
    if scene_count == 0: return storyboard
    average_duration = total_time / scene_count
    current_time = 0.0
    for scene in storyboard:
        scene["start_time"] = round(current_time, 2)
        scene["end_time"] = round(current_time + average_duration, 2)
        scene["duration"] = round(average_duration, 2)
        current_time += average_duration
    logger.info("✅ Added duration timing to scenes.")
    return storyboard

def generate_storyboard(quote_text: str, timestamp_data: Optional[List[Dict]] = None) -> List[Dict]:
    client = Groq(api_key=load_api_key())
    model_name = pick_available_model(client)

    # === THIS IS THE ONLY CHANGE: A more robust prompt and parsing logic ===

    system_prompt = """
    You are an AI Film Director and a master of concise Stable Diffusion prompts. Your job is to create a visual storyboard for a quote by breaking it down into 2-3 concrete, cinematic scenes.

    **CRITICAL RULES:**
    1.  **METAPHORS FIRST:** Translate the abstract quote into a concrete visual metaphor.
        -   **INSTEAD OF:** "A figure of hope."
        -   **USE:** "A green sprout pushes through cracked earth."

    2.  **TOKEN EFFICIENCY IS KEY:** The final `animation_prompt` MUST be a comma-separated list of keywords and short phrases. It should be rich but concise. The underlying AI has a hard limit of around 77 tokens.
        -   **BAD (Too long):** "This is a cinematic shot of a lone wolf that is standing on a very tall, snowy mountain peak right at the moment the sun is rising, which creates an epic and beautiful lighting effect. The style should be photorealistic and rendered in 8k."
        -   **GOOD (Token-Efficient):** "lone wolf on snowy mountain peak at sunrise, cinematic, epic lighting, photorealistic, sharp focus, 8k"

    3.  **STRUCTURED COMPOSITION:** For each scene, you must define the `camera`, `lighting`, `style`, and `environment` using this efficient keyword style.

    4.  **JSON OUTPUT ONLY:** Return ONLY a single, valid JSON object in this exact format. Do not add commentary.

    **EXAMPLE JSON FORMAT:**
    {
    "scenes": [
        {
        "description": "A climber's hand slips on a wet rock, but finds a new grip.",
        "composition": {
            "camera": "low-angle close-up, hand on rock face",
            "lighting": "dramatic morning sun, long shadows",
            "environment": "rugged granite cliff, clouds below",
            "style": "photorealistic, cinematic, sharp focus, award-winning photography, 8k"
        }
        }
    ]
    }
    """

    logger.info("🚀 Requesting Groq for storyboard...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Quote: \"{quote_text}\""}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    
    try:
        parsed = json.loads(raw)
        
        # This parsing is now simpler and more robust because we demand a dictionary.
        if not isinstance(parsed, dict):
            raise TypeError("LLM response was not a JSON object as requested.")
            
        storyboard = parsed.get("scenes")
        
        if not storyboard or not isinstance(storyboard, list):
            raise ValueError("LLM response did not contain a valid, non-empty 'scenes' list.")

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"❌ Failed to parse or validate storyboard from LLM: {raw}")
        raise ValueError(f"Error processing LLM response: {e}")
    # === END OF CHANGE ===

    logger.info(f"✅ Storyboard generated with {len(storyboard)} scenes.")
    return calculate_scene_durations(storyboard, timestamp_data)

def main():
    parser = argparse.ArgumentParser(description="Generate storyboard with optional scene durations.")
    parser.add_argument("--quote", required=True, help="Quote text.")
    parser.add_argument("--timestamps", required=False, help="Path to JSON timestamps.")
    args = parser.parse_args()
    start = time.time()
    try:
        timestamp_data = None
        if args.timestamps and os.path.exists(args.timestamps):
            with open(args.timestamps, "r") as f:
                loaded_json = json.load(f)
                timestamp_data = loaded_json.get("segments")
        scenes = generate_storyboard(args.quote, timestamp_data)
        output = {"status": "COMPLETED", "storyboard": scenes}
    except Exception as e:
        logging.error(e, exc_info=True)
        output = {"status": "FAILED", "error": str(e)}
    print(json.dumps(output, indent=2))
    logger.info(f"✅ Finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()