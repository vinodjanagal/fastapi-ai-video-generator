
import argparse
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

from groq import Groq
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
    # This function remains unchanged
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

def generate_storyboard(quote_text: str, timestamp_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
    client = Groq(api_key=load_api_key())
    model_name = pick_available_model(client)

    # =================== V9.1 PROMPT - FULLY UPGRADED ===================
    system_prompt = """
    You are an AI Film Director and a master of concise Stable Diffusion prompts. Your job is to create a visual storyboard and a character sheet for a quote.

    **CRITICAL RULES:**
    1.  **IDENTIFY THE CHARACTER:** First, identify the main character or subject of the quote.
    2.  **CREATE A CHARACTER SHEET:** You MUST create a `character_sheet` prompt. This should be a detailed, photorealistic, "headshot" or "portrait" style description of the main character. This is for visual identity.
        - **GOOD EXAMPLE:** "photorealistic portrait of a wise, old Roman philosopher with a grey beard, stoic expression, detailed wrinkles, cinematic lighting"
        - If there is no clear character (e.g., the quote is purely abstract), return an empty string "" for the `character_sheet`.
    3.  **METAPHORS FOR SCENES:** Translate the abstract quote into 2-3 concrete, cinematic scenes.
    4.  **TOKEN EFFICIENCY:** All prompts must be a comma-separated list of keywords and short phrases.
    5.  **ADD CAMERA MOTION:** For each scene, you MUST specify a `camera_motion`. Valid options are: "static", "slow_zoom_in", "slow_zoom_out", "pan_left", "pan_right". Choose a motion that fits the mood of the scene.
    6.  **JSON OUTPUT ONLY:** Return ONLY a single, valid JSON object in this exact format. Do not add commentary.

    **EXAMPLE JSON FORMAT:**
    {
      "character_sheet": "photorealistic portrait of a weary but determined medieval king, sharp focus, detailed face, 4k",
      "scenes": [
        {
          "description": "A king overlooks a vast, misty valley from a castle battlement at dawn.",
          "composition": {
            "camera": "wide shot, from behind the king's shoulder",
            "lighting": "soft morning light, long shadows",
            "environment": "stone castle, misty mountains",
            "style": "photorealistic, cinematic, epic fantasy"
          },
          "camera_motion": "pan_left"
        }
      ]
    }
    """
    # ====================================================================

    logger.info("🚀 Requesting Groq for V7.1 storyboard...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Quote: \"{quote_text}\""}
        ],
        temperature=0.7,
        response_format={"type": "json_object"}
    )
    raw_json_output = response.choices[0].message.content
    
    try:
        parsed_data = json.loads(raw_json_output)
        if not isinstance(parsed_data, dict):
            raise TypeError("LLM response was not a JSON object.")
            
        storyboard_scenes = parsed_data.get("scenes")
        if not storyboard_scenes or not isinstance(storyboard_scenes, list):
            raise ValueError("LLM response did not contain a valid 'scenes' list.")

        # Add durations to the scenes
        timed_scenes = calculate_scene_durations(storyboard_scenes, timestamp_data)
        parsed_data["scenes"] = timed_scenes

        logger.info(f"✅ V9.1 Storyboard and Character Sheet generated with {len(timed_scenes)} scenes.")
        return parsed_data

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"❌ Failed to parse or validate storyboard from LLM: {raw_json_output}")
        raise ValueError(f"Error processing LLM response: {e}")

def main():
    # This main function is for testing and remains largely the same
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
        
        # The output is now a dictionary, not just a list of scenes
        storyboard_data = generate_storyboard(args.quote, timestamp_data)
        output = {"status": "COMPLETED", "storyboard_data": storyboard_data}

    except Exception as e:
        logging.error(e, exc_info=True)
        output = {"status": "FAILED", "error": str(e)}

    print(json.dumps(output, indent=2))
    logger.info(f"✅ Finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    main()