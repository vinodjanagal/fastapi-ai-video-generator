from __future__ import annotations
from typing import Dict, Tuple
import logging

logger = logging.getLogger("v6_engine.cinematics")

# ---------------------------------------------------
# SHOT TYPE CLASSIFIER
# ---------------------------------------------------
def classify_shot_type(prompt_text: str, semantic_parts: Dict[str, str]) -> str:
    text = (prompt_text or "").lower()
    action = (semantic_parts.get("action") or "").lower()

    # explicit keywords
    if "extreme close-up" in text or "macro" in text:
        return "ECU"
    if "close-up" in text or "portrait" in text or "face" in text:
        return "CU"
    if "wide shot" in text or "establishing" in text:
        return "WS"

    # verbs that imply subject-focus
    if action in {"study", "studying", "examine", "examining",
                  "hold", "holding", "reading", "look", "looking"}:
        return "CU"

    if semantic_parts.get("subject") and semantic_parts.get("environment"):
        return "MS"

    return "MS"


# ---------------------------------------------------
# PARAM TUNING
# ---------------------------------------------------
def apply_param_tuning(args, shot_type: str):
    """
    Applies baseline values if user didn't provide their own.
    EXACTLY matching your animatediff.py's expectations.
    """
    defaults = {
        "ECU": (0.20, 0.00, 6.0),
        "CU":  (0.28, 0.035, 6.5),
        "MS":  (0.35, 0.05, 7.0),
        "WS":  (0.50, 0.08, 7.5),
    }
    if args.manual:
        return

    (d_strength, d_ip, d_guidance) = defaults.get(shot_type, (0.3, 0.04, 7.0))

    if args.strength is None:
        args.strength = d_strength
    if args.ip_adapter_scale is None:
        args.ip_adapter_scale = d_ip
    if args.guidance_scale is None:
        args.guidance_scale = d_guidance


# ---------------------------------------------------
# SAFETY GUARDS  (matching your engine)
# ---------------------------------------------------
def apply_safety_guards(args):
    # Minimum steps
    if not args.manual and args.num_steps < 15:
        args.num_steps = 15

    # NOTE: Your animatediff.py does not clamp resolution,
    # so we DO NOT add it here.
    # This file stays 100% compatible.


# ---------------------------------------------------
# CONTINUITY: FIX OVER-ANCHORING
# ---------------------------------------------------
def compute_runtime_continuity(
    base_strength: float,
    base_ip_scale: float,
    base_guidance: float,
    prompt: str
) -> Tuple[float, float, float]:

    motion_verbs = {
        "studying", "examining", "holding", "inspecting",
        "reading", "looking", "slicing", "repairing"
    }

    is_motion = any(v in prompt.lower() for v in motion_verbs)

    if is_motion:
        strength = max(0.35, min(0.48, base_strength * 1.15))
        ip_scale = min(0.02, base_ip_scale * 0.6)
        guidance = min(5.0, base_guidance * 0.8)
    else:
        strength = base_strength
        ip_scale = base_ip_scale
        guidance = base_guidance

    # clamp
    strength = float(min(max(strength, 0.2), 0.9))
    ip_scale = float(min(max(ip_scale, 0.0), 0.1))
    guidance = float(min(max(guidance, 2.5), 7.5))

    return strength, ip_scale, guidance
