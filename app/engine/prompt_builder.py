# app/engine/prompt_builder.py
from typing import Dict, Tuple
import logging
from app.engine.parser import reweight_parts

logger = logging.getLogger("v9_prompt_builder")

BASE_QUALITY_PROMPT = "photorealistic, cinematic, ultra-detailed, cinematic lighting, sharp focus, 8k"
BASE_NEGATIVE_PROMPT = "blurry, deformed, extra limbs, extra fingers, watermark, text, logo, bad anatomy, mutation, worst quality, low quality, jpeg artifacts"
ANTI_EYE_NEG = "macro eye close-up, extreme close-up of iris, isolated eye, iris detail"

def _looks_like_style_already(text: str) -> bool:
    lower = (text or "").lower()
    # if user or LLM already included 'photorealistic' or '8k' etc, we won't re-inject
    for token in ["photorealistic", "8k", "ultra-detailed", "cinematic", "film grain", "sharp focus"]:
        if token in lower:
            return True
    return False

def build_semantic_prompt(
    raw_prompt: str,
    user_negative_prompt: str,
    shot_type: str,
    semantic_parts: Dict[str, str],
    scene_index: int = 0,
    total_scenes: int = 1
) -> Tuple[str, str]:
    """
    Builds a layered prompt:
      - Base narrative (env + subject + action + object)
      - Shot composition (from shot_type and shot progression)
      - Motion tokens (camera)
      - Lighting & composition tokens
      - Quality tokens (only if missing)
      - Negative tokens (with anti-eye unless explicitly requested)
    """
    logger.info(f"build_semantic_prompt: parts={semantic_parts} shot_type={shot_type} idx={scene_index}/{total_scenes}")
    parts = reweight_parts(dict(semantic_parts))  # conservative reweight

    subject = parts.get("subject") or ""
    action = parts.get("action") or ""
    obj = parts.get("object") or ""
    env = parts.get("environment") or ""

    # Layer 1: Base narrative
    narrative_parts = []
    if env:
        narrative_parts.append(f"({env}:1.15)")
    if subject:
        narrative_parts.append(f"({subject}:1.35)")
    if action:
        narrative_parts.append(action)
    if obj:
        narrative_parts.append(f"({obj}:1.2)")

    # Layer 2: Shot composition (progression-friendly)
    shot_map = {
        "ECU": "(extreme close-up:1.2)",
        "CU": "(cinematic close-up:1.15)",
        "MS": "medium shot",
        "WS": "wide establishing shot"
    }
    shot_token = shot_map.get(shot_type, "")
    # Guard: if shot_token is an ECU/CU but subject is a generic 'portrait' or no subject, downgrade to MS
    if shot_token and "extreme" in shot_token.lower():
        if not subject or "portrait" in subject.lower():
            # downgrade ECU to CU to avoid macro-eye collapse
            shot_token = "(cinematic close-up:1.05)"

    # Smooth shot progression for multi-scene
    if total_scenes > 1:
        if scene_index == 0:
            # opening scene: prefer WS
            shot_token = shot_token or "wide establishing shot"
        elif scene_index == total_scenes - 1:
            shot_token = shot_token or "(intimate medium close-up:1.1)"
        else:
            shot_token = shot_token or "dynamic tracking medium shot"

    if shot_token:
        narrative_parts.append(shot_token)

    # Layer 3: motion / camera movement
    # keep this descriptive; actual engine maps these tokens to camera transforms
    if total_scenes > 1:
        if scene_index == 0:
            narrative_parts.append("camera: slow push-in")
        elif scene_index == total_scenes - 1:
            narrative_parts.append("camera: slow pull-out")
        else:
            narrative_parts.append("camera: smooth tracking")

    # Layer 4: lighting & composition
    lighting = "golden hour soft light, volumetric rays, cinematic ambience"
    composition = "balanced composition, environmental storytelling, depth of field"
    narrative_parts.append(lighting)
    narrative_parts.append(composition)

    # Layer 5: quality injection — only if raw_prompt or subject doesn't already contain quality tokens
    if not _looks_like_style_already(raw_prompt):
        narrative_parts.append(BASE_QUALITY_PROMPT)

    pos_prompt = ", ".join([p for p in narrative_parts if p])

    # Negative prompt: combine base + user + anti-eye safeguard (unless prompt explicitly requests eye)
    negative_parts = [BASE_NEGATIVE_PROMPT]
    if user_negative_prompt:
        negative_parts.append(user_negative_prompt)
    if "eye" not in (raw_prompt or "").lower():
        negative_parts.append(ANTI_EYE_NEG)

    neg_prompt = ", ".join(negative_parts)

    logger.info(f"Final Positive Prompt: {pos_prompt}")
    return pos_prompt, neg_prompt
