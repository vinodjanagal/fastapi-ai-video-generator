# app/engine/cinematics.py
from typing import Dict, Tuple
import logging
import random

logger = logging.getLogger("v6_engine.cinematics")

# ---------------------------------------------------
# SHOT TYPE CLASSIFIER (safer)
# ---------------------------------------------------
def classify_shot_type(prompt_text: str, semantic_parts: Dict[str, str]) -> str:
    text = (prompt_text or "").lower()
    action = (semantic_parts.get("action") or "").lower()

    # Explicit request for ECU/CU
    if "extreme close-up" in text or "macro" in text:
        return "ECU"
    if "close-up" in text or "large portrait" in text or "extreme portrait" in text:
        return "CU"

    # If prompt contains 'portrait' but also contains 'full' or 'figure' -> treat as MS
    if "portrait" in text and ("full" in text or "full body" in text or "figure" in text):
        return "MS"

    # verbs that imply subject-focus -> CU
    if action in {"study", "studying", "examine", "examining", "hold", "holding", "read", "reading", "look", "looking"}:
        # but only CU if subject indicates a focused object/person
        if semantic_parts.get("subject"):
            return "CU"

    # environment + subject -> MS
    if semantic_parts.get("subject") and semantic_parts.get("environment"):
        return "MS"

    # fallback
    return "MS"


# ---------------------------------------------------
# PARAM TUNING
# ---------------------------------------------------
def apply_param_tuning(args, shot_type: str):
    """
    Applies baseline values if user didn't provide their own.
    """
    defaults = {
        "ECU": (0.20, 0.00, 6.0),
        "CU":  (0.28, 0.035, 6.5),
        "MS":  (0.35, 0.05, 7.0),
        "WS":  (0.50, 0.08, 7.5),
    }
    if getattr(args, "manual", False):
        return

    (d_strength, d_ip, d_guidance) = defaults.get(shot_type, (0.3, 0.04, 7.0))

    if getattr(args, "strength", None) is None:
        args.strength = d_strength
    if getattr(args, "ip_adapter_scale", None) is None:
        args.ip_adapter_scale = d_ip
    if getattr(args, "guidance_scale", None) is None:
        args.guidance_scale = d_guidance


# ---------------------------------------------------
# SAFETY GUARDS  (matching your engine)
# ---------------------------------------------------
def apply_safety_guards(args):
    # Minimum steps
    if not getattr(args, "manual", False) and getattr(args, "num_steps", 0) < 15:
        args.num_steps = 15

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

    is_motion = any(v in (prompt or "").lower() for v in motion_verbs)

    if is_motion:
        strength = max(0.35, min(0.48, base_strength * 1.15))
        ip_scale = max(0.0, min(0.06, base_ip_scale * 0.6))
        guidance = max(2.5, min(6.5, base_guidance * 0.9))
    else:
        strength = base_strength
        ip_scale = base_ip_scale
        guidance = base_guidance

    # clamp
    strength = float(min(max(strength, 0.2), 0.9))
    ip_scale = float(min(max(ip_scale, 0.0), 0.1))
    guidance = float(min(max(guidance, 2.5), 7.5))

    return strength, ip_scale, guidance
