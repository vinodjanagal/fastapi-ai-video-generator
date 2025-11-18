from typing import Dict, Tuple
import logging

logger = logging.getLogger("v10_prompt_builder")

BASE_QUALITY_PROMPT = "photorealistic, cinematic, ultra-detailed, cinematic lighting, sharp focus, 8k"
BASE_NEGATIVE_PROMPT = "blurry, deformed, extra limbs, extra fingers, watermark, text, logo, bad anatomy, mutation, worst quality, low quality, jpeg artifacts"
ANTI_EYE_NEG = "macro eye close-up, extreme close-up of iris, isolated eye, iris detail"


def _looks_like_style_already(text: str) -> bool:
    lower = (text or "").lower()
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
    total_scenes: int = 1,
) -> Tuple[str, str]:
    logger.info(f"build_semantic_prompt: parts={semantic_parts} shot_type={shot_type} idx={scene_index}/{total_scenes}")

    subject = semantic_parts.get("subject") or ""
    action = semantic_parts.get("action") or ""
    obj = semantic_parts.get("object") or ""
    env = semantic_parts.get("environment") or ""

    narrative = []
    if env:
        narrative.append(f"({env}:1.15)")
    if subject:
        narrative.append(f"({subject}:1.35)")
    if action:
        narrative.append(action)
    if obj:
        narrative.append(f"({obj}:1.2)")

    shot_map = {"ECU": "(extreme close-up:1.2)", "CU": "(cinematic close-up:1.15)", "MS": "medium shot", "WS": "wide establishing shot"}
    shot_token = shot_map.get(shot_type, "")
    if shot_token and "extreme" in shot_token.lower():
        if not subject or "portrait" in subject.lower():
            shot_token = "(cinematic close-up:1.05)"

    if total_scenes > 1:
        if scene_index == 0:
            shot_token = shot_token or "wide establishing shot"
        elif scene_index == total_scenes - 1:
            shot_token = shot_token or "(intimate medium close-up:1.1)"
        else:
            shot_token = shot_token or "dynamic tracking medium shot"

    if shot_token:
        narrative.append(shot_token)

    if total_scenes > 1:
        if scene_index == 0:
            narrative.append("camera: slow push-in")
        elif scene_index == total_scenes - 1:
            narrative.append("camera: slow pull-out")
        else:
            narrative.append("camera: smooth tracking")

    lighting = "golden hour soft light, volumetric rays, cinematic ambience"
    composition = "balanced composition, environmental storytelling, depth of field"
    narrative.append(lighting)
    narrative.append(composition)

    if not _looks_like_style_already(raw_prompt):
        narrative.append(BASE_QUALITY_PROMPT)

    pos_prompt = ", ".join([p for p in narrative if p])

    negative_parts = [BASE_NEGATIVE_PROMPT]
    if user_negative_prompt:
        negative_parts.append(user_negative_prompt)
    if "eye" not in (raw_prompt or "").lower():
        negative_parts.append(ANTI_EYE_NEG)

    neg_prompt = ", ".join(negative_parts)
    logger.info(f"Final Positive Prompt: {pos_prompt}")

    return pos_prompt, neg_prompt