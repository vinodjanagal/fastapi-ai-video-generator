# app/engine/character_sheet.py
from typing import Tuple
import logging
from app.engine.parser import semantic_parser
from app.engine.prompt_builder import BASE_QUALITY_PROMPT, BASE_NEGATIVE_PROMPT

logger = logging.getLogger("v9_character_sheet")

def _is_sentence_like(text: str) -> bool:
    # If the character sheet text contains a verb or comma-separated descriptors, treat more carefully
    if not text: return False
    if len(text.split()) > 12:  # longer description -> may be narrative
        return True
    # short prompts like "photorealistic portrait of..." -> not sentence-like
    if any(c in text for c in [".", ";"]):
        return True
    return False

def build_character_prompt(raw: str) -> Tuple[str, str]:
    """
    Build a stable character prompt:
      - If raw looks like a sentence (long), we run minimal parsing to extract subject.
      - If raw is a short descriptor ("photorealistic portrait of a wise man"), we DO NOT semantic-parse aggressively;
        we anchor the subject and append quality tokens.
    """
    if not raw:
        raise ValueError("Empty character sheet prompt")

    if _is_sentence_like(raw):
        # mild parsing
        parts = semantic_parser(raw)
        subject = parts.get("subject") or raw
    else:
        # short descriptor: try to find the 'of <subject>' pattern or human hint fallback
        import re
        m = re.search(r"of\s+([A-Za-z ,]+)$", raw, flags=re.I)
        if m:
            subject = m.group(1).strip()
        else:
            # fallback: entire raw without style tokens (strip 'photorealistic', 'portrait')
            subject = raw
            subject = subject.replace("photorealistic", "").replace("portrait", "").strip()

    # Compose prompt conservatively
    # Ensure subject is explicit (e.g., "wise old Japanese martial artist")
    final_prompt = f"photorealistic portrait of {subject}, calm expression, detailed wrinkles, cinematic lighting, soft studio lighting, {BASE_QUALITY_PROMPT}"
    # Strong negative to avoid macro eye or extreme crop
    final_negative = f"{BASE_NEGATIVE_PROMPT}, macro eye close-up, extreme close-up, cropped face, iris detail"

    logger.info(f"Character prompt built: {final_prompt}")
    return final_prompt, final_negative
