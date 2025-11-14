import logging
from typing import Dict

logger = logging.getLogger("v6_cinematics")

def classify_shot_type(prompt_text: str, semantic_parts: Dict[str, str]) -> str:
    """
    Intent-driven shot classification with a 4-layer priority hierarchy.
    """
    text = (prompt_text or "").lower()
    action = (semantic_parts.get("action") or "").lower()

    # 1. Highest priority: Explicit user keywords.
    if any(k in text for k in ["macro", "extreme close-up", "extreme closeup", "ecu"]):
        return "ECU"
    if any(k in text for k in ["close-up", "closeup", "portrait", "face", "hands"]):
        return "CU"
    if any(k in text for k in ["wide shot", "full scene", "establishing shot", "environment"]):
        return "WS"

    # 2. Second priority: Action implies a specific shot type.
    action_close_inspect = {
        "study", "studied", "hold", "holding", "examine", "examining", 
        "inspect", "read", "reading", "look", "looking", "slice", "slicing"
    }
    if action in action_close_inspect:
        return "CU"

    # 3. Third priority: Contextual inference.
    if semantic_parts.get("subject") and semantic_parts.get("environment"):
        return "MS"

    # 4. Final Fallbacks
    if semantic_parts.get("subject"):
        return "CU"

    return "MS"


def apply_param_tuning(args, shot_type: str):
    """
    Apply safe defaults for parameters, respecting manual CLI overrides.
    """
    logger.info(f"Classified shot type: {shot_type}. Applying tuning.")
    defaults = {
        "ECU": (0.20, 0.0, 6.0),
        "CU": (0.28, 0.035, 6.5),
        "MS": (0.35, 0.05, 7.0),
        "WS": (0.50, 0.08, 7.5)
    }
    default_strength, default_ip_scale, default_guidance = defaults.get(shot_type, (0.3, 0.04, 7.0))

    if not getattr(args, "manual", False):
        if getattr(args, "strength", None) is None: args.strength = default_strength
        if getattr(args, "ip_adapter_scale", None) is None: args.ip_adapter_scale = default_ip_scale
        if getattr(args, "guidance_scale", None) is None: args.guidance_scale = default_guidance

    logger.info(f"Final tuned parameters -> strength={args.strength}, ip_scale={args.ip_adapter_scale}, guidance={args.guidance_scale}")


def apply_safety_guards(args):
    """
    Enforce minimum safe parameters to prevent model collapse.
    """
    if not getattr(args, "manual", False) and getattr(args, "num_steps", 0) < 15:
        logger.warning(f"[safety-guard] num_steps was {args.num_steps}, too low. Forcing to 15.")
        args.num_steps = 15