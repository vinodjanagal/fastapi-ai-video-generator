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
    You are an AI Film Director and expert Cinematic Scene Designer.

    Your task: Convert ANY input text (quote, story, idea, concept, metaphor, abstract emotion,
    educational statement, etc.) into a highly cinematic 3-scene storyboard.

    You are NOT summarizing — you are designing FILM SHOTS with CHARACTER + ENVIRONMENT.

    ========================
    OUTPUT FORMAT (STRICT)
    ========================

    You must output ONLY a single JSON object of the form:

    {
    "character_sheet": "string",
    "scenes": [
        {
        "description": "string (1–3 sentences)",
        "composition": {
            "camera": "wide shot | wide establishing shot | medium shot | medium-wide shot",
            "lighting": "string describing cinematic lighting",
            "environment": "string describing full surroundings",
            "style": "photorealistic, cinematic"
        },
        "camera_motion": "static | slow_zoom_in | slow_zoom_out | pan_left | pan_right"
        }
    ]
    }

    No markdown.  
    No commentary.  
    No explanations.  
    JSON ONLY.

    ========================
    GLOBAL CINEMATIC RULES
    ========================

    1. ABSOLUTE BAN ON MICRO-SHOTS  
    Do NOT create scenes focused only on:
    - hands  
    - eyes  
    - mouth  
    - single facial features  
    - isolated objects  
    - macro / extreme close-ups  

    Each scene MUST show:
    - the character (at least half-body, ideally full)  
    - AND the surrounding environment clearly  
    - AND the action happening within that environment  

    Exception:  
    Only use a close-up if the USER TEXT explicitly demands it
    (e.g., “focus only on her eyes”, “close-up of the ring”).  
    If the user does not explicitly ask — do NOT use close-ups.

    2. **EXACTLY 3 SCENES. ALWAYS.**  
    The cinematic arc MUST follow this structure:

    **Scene 1 – Wide Establishing Shot**  
    - Show the full environment  
    - Include the character in context  
    - Introduce setting, tone, atmosphere  
    - No action close-ups  

    **Scene 2 – Medium or Medium-Wide Shot (Action Moment)**  
    - Character + main action clearly visible  
    - Keep the environment visible in background  
    - Emotional or narrative development  

    **Scene 3 – Medium-Wide or Wide Hero Shot (Climax/Resolution)**  
    - Character + environment + emotional payoff  
    - Show the scene’s “impact moment”  
    - Reveal or resolution of the idea/emotion  

    All 3 scenes must visually connect (same character, same environment style).

    3. CHARACTER SHEET  
    If the text implies a character:
        - Provide ONE cinematic description:
            - species  
            - approximate age (only if implied)  
            - gender (only if safely implied)  
            - clothing  
            - physical traits  
            - emotional tone  

        Style must ALWAYS include:
        “photorealistic, cinematic lighting, sharp focus”

    If the input text has NO character:
        - Set "character_sheet": "".

    4. SCENE FIDELITY  
    Scenes must reflect the meaning and setting of the input text.
    If a location is mentioned (library, forest, city, desert, ship, classroom, etc.):
        - it MUST appear in the “environment” field,  
        - AND be clearly described in the scene.

    5. COMPOSITION FIELDS
    - description: full scene (character + environment + action + emotion)  
    - camera: wide, wide establishing, medium, medium-wide  
        (close-up NOT allowed unless user explicitly demands it)  
    - lighting: cinematic lighting description  
    - environment: clear description of surroundings  
    - style: always “photorealistic, cinematic”

    6. CAMERA MOTION RULES  
    - slow_zoom_in → tension, discovery, emotional reveal  
    - slow_zoom_out → resolution, reveal of context  
    - pan_left / pan_right → motion across the scene  
    - static → calm or contemplative scenes  

    7. STRICT JSON  
    Output MUST be valid JSON.
    No trailing commas. No markdown. No explanations.
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
