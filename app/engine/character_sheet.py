# app/engine/character_sheet.py
import re
import logging
from typing import Tuple

from app.engine.prompt_builder import (
    BASE_QUALITY_PROMPT,
    BASE_NEGATIVE_PROMPT,
    ANTI_EYE_NEG,
)

logger = logging.getLogger("phoenix.character_sheet")


# --------------------------------------------
# Utility cleaning functions
# --------------------------------------------
REMOVE_PREFIXES = [
    r"^photorealistic\s+portrait\s+of\s+",
    r"^portrait\s+of\s+",
    r"^a\s+portrait\s+of\s+",
    r"^an\s+portrait\s+of\s+",
    r"^digital\s+portrait\s+of\s+",
]

ACTION_VERBS = [
    "holding", "hunched", "working", "studying",
    "finding", "leaning", "looking", "reading",
    "standing", "sitting", "focused", "examining",
]

ENV_WORDS = [
    "workshop", "room", "bench", "table", "forest", "field",
    "background", "scene", "environment", "landscape",
    "setting", "laboratory", "office", "street",
]

CAMERA_WORDS = [
    "close-up", "medium shot", "wide shot",
    "shot", "view", "angle",
]

LIGHT_WORDS = [
    "warm light", "lamp light", "soft light",
    "shadow", "lighting", "sunlight",
]

NO_IDENTITY_WORDS = [
    "expression", "focused", "emotion", "pose", "gesture",
]


def _remove_prefixes(text: str) -> str:
    clean = text.strip()
    for p in REMOVE_PREFIXES:
        clean = re.sub(p, "", clean, flags=re.I).strip()
    return clean

def _remove_trailing_noise(words: list) -> list:
    cleaned = []
    for w in words:
        lw = w.lower()
        if any(lw.startswith(v) for v in ACTION_VERBS):
            continue
        if any(lw == e for e in ENV_WORDS):
            continue
        if any(lw in c for c in CAMERA_WORDS):
            continue
        if any(lw in l for l in LIGHT_WORDS):
            continue
        if lw in NO_IDENTITY_WORDS:
            continue
        cleaned.append(w)
    return cleaned

def _clean_subject(raw: str) -> str:
    """Extract a clean identity string from raw LLM text."""
    if not raw:
        return "person"

    raw = _remove_prefixes(raw)

    # Break into words for cleanup
    words = re.split(r"[,\s]+", raw)
    words = [w.strip() for w in words if w.strip()]

    words = _remove_trailing_noise(words)

    if not words:
        return "person"

    # Reconstruct subject
    subject = " ".join(words)

    # Remove accidental duplicates
    subject = re.sub(r"\b(\w+)( \1\b)+", r"\1", subject, flags=re.I)

    # Clamp length for portrait stability
    subject = " ".join(subject.split()[:12])

    return subject.strip()


# --------------------------------------------
# MAIN BUILDER
# --------------------------------------------
def build_character_prompt(raw: str) -> Tuple[str, str]:
    """
    Phoenix V10 – Production Character Sheet Builder
    ------------------------------------------------
    - Extracts identity only
    - Removes environment / actions / camera words
    - Always output human/creature face identity
    - Safe studio portrait lighting
    - Bulletproof negative prompt
    """

    if not raw:
        raise ValueError("Character sheet text is empty")

    # 1. Extract identity
    subject = _clean_subject(raw)

    logger.info(f"[CharacterSheet] Identity extracted: {subject}")

    # 2. Final positive portrait prompt
    final_pos = (
        f"photorealistic portrait of {subject}, "
        f"cinematic lighting, soft studio lighting, "
        f"detailed facial features, {BASE_QUALITY_PROMPT}"
    )

    # 3. Final negative prompt
    final_neg = (
        f"{BASE_NEGATIVE_PROMPT}, {ANTI_EYE_NEG}, "
        f"cropped face, warped face, distorted eyes"
    )

    logger.info(f"[CharacterSheet] Final Prompt: {final_pos}")

    return final_pos, final_neg
