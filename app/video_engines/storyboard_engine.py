import argparse
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

from groq import Groq
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("phoenix.storyboard")


# Preferred Groq models
PREFERRED_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "mixtral-8x7b-32768"
]


# ---------------------------------------------------------
# API Key handling
# ---------------------------------------------------------
def load_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("❌ GROQ_API_KEY not found in environment variables.")
    return api_key


def pick_available_model(client: Groq) -> str:
    try:
        models = client.models.list()
        available = {m.id for m in models.data}

        for model in PREFERRED_MODELS:
            if model in available:
                logger.info(f"🎬 Using Groq model: {model}")
                return model

        logger.warning("⚠️ No preferred models are available. Using fallback.")
        return PREFERRED_MODELS[0]

    except Exception as e:
        logger.warning(f"⚠️ Failed to check model availability: {e}")
        return PREFERRED_MODELS[0]


# ---------------------------------------------------------
# Duration assignment (kept unchanged)
# ---------------------------------------------------------
def calculate_scene_durations(scenes: List[Dict], ts_data: Optional[List[Dict]]) -> List[Dict]:
    if not ts_data:
        return scenes

    total_time = 0.0
    for segment in reversed(ts_data):
        if "words" in segment and segment["words"]:
            total_time = segment["words"][-1]["end"]
            break

    if total_time == 0:
        return scenes

    n = len(scenes)
    if n == 0:
        return scenes

    avg = total_time / n
    now = 0.0

    for s in scenes:
        s["start_time"] = round(now, 2)
        s["end_time"] = round(now + avg, 2)
        s["duration"] = round(avg, 2)
        now += avg

    return scenes


# ---------------------------------------------------------
# UNIVERSAL STORYBOARD (Phoenix V10)
# ---------------------------------------------------------
def generate_storyboard(quote_text: str, timestamp_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
    client = Groq(api_key=load_api_key())
    model_name = pick_available_model(client)

    # ======================================================
    # UNIVERSAL SYSTEM PROMPT (Final Version)
    # ======================================================
    system_prompt = """
You are an AI Film Director and expert Stable Diffusion prompt architect.

Your job is to convert ANY input text (story, quote, idea, metaphor, abstract
thought, educational content, etc.) into a structured, cinematic storyboard.

OUTPUT MUST BE STRICTLY JSON:
{
  "character_sheet": string,
  "scenes": [
    {
      "description": "...",
      "composition": {
        "camera": "...",
        "lighting": "...",
        "environment": "...",
        "style": "..."
      },
      "camera_motion": "..."
    }
  ]
}

========================
RULES
========================

1. **DETERMINE IF ANY CHARACTER EXISTS**
   - A character may be a human, animal, robot, creature, or defined entity.
   - If the text clearly describes a person/being, create a character sheet.
   - If the text has NO character (e.g., "The universe expands"), then:
       "character_sheet": ""
   - DO NOT invent characters that do not exist.
   - DO NOT hallucinate professions, gender, or appearance not implied.

2. **CHARACTER SHEET (ONLY WHEN A CHARACTER EXISTS)**
   Must be a portrait/headshot description:
   - species (human/animal/robot/etc.)
   - approximate age only if implied
   - gender only if explicit or clearly implied
   - defining physical traits
   - emotional tone
   - proper clothing if described
   - NO repetitions
   - style: "photorealistic, cinematic lighting, sharp focus"

3. **SCENE GENERATION**
   - 2–4 scenes.
   - Scenes must reflect the meaning of the text.
   - Scenes may be metaphorical, emotional, abstract, or literal.
   - Scenes should depict environments, objects, or the character (if one exists).
   - DO NOT contradict the original meaning.

4. **COMPOSITION**
   Each scene must include:
   - "camera": e.g., "close-up", "medium shot", "wide shot"
   - "lighting": e.g., "warm lamp light", "soft ambient light"
   - "environment": e.g., "quiet workshop", "vast desert", "abstract cosmic space"
   - "style": "photorealistic, cinematic"

5. **CAMERA MOTION**
   One of:
   - static
   - slow_zoom_in
   - slow_zoom_out
   - pan_left
   - pan_right

6. **STRICT JSON ONLY**
    No explanations, no commentary, no markdown — JSON object only.
"""

    user_msg = f"Text: \"{quote_text}\""

    # Request to Groq
    logger.info("🎥 Requesting Groq for storyboard...")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.5,
        response_format={"type": "json_object"}
    )

    raw_json = response.choices[0].message.content

    # Parse JSON
    try:
        data = json.loads(raw_json)

        # Validate
        if not isinstance(data, dict):
            raise ValueError("LLM output is not a JSON object")

        scenes = data.get("scenes")
        if not scenes or not isinstance(scenes, list):
            raise ValueError("Missing or invalid 'scenes' list in LLM output")

        # Assign durations
        scenes = calculate_scene_durations(scenes, timestamp_data)
        data["scenes"] = scenes

        logger.info(f"🎬 Storyboard generated successfully with {len(scenes)} scenes.")
        return data

    except Exception as e:
        logger.error("❌ Failed to parse LLM JSON:")
        logger.error(raw_json)
        raise ValueError(f"Storyboard parsing error: {e}")


# ---------------------------------------------------------
# Testing entrypoint
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phoenix Storyboard Generator (V10 Universal)")
    parser.add_argument("--quote", required=True)
    parser.add_argument("--timestamps", required=False)
    args = parser.parse_args()

    ts_data = None
    if args.timestamps and os.path.exists(args.timestamps):
        with open(args.timestamps) as f:
            js = json.load(f)
            ts_data = js.get("segments")

    start = time.time()
    try:
        result = generate_storyboard(args.quote, ts_data)
        print(json.dumps({"status": "COMPLETED", "storyboard_data": result}, indent=2))
    except Exception as e:
        print(json.dumps({"status": "FAILED", "error": str(e)}, indent=2))

    logger.info(f"⏳ Finished in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
