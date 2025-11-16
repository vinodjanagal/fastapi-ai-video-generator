# app/engine/parser.py
from typing import Dict, Optional
import logging
import spacy
import re

logger = logging.getLogger("v7_parser")

def _load_spacy_model():
    """Loads the spaCy model with optimized pipes for our use case."""
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "textcat"])
        logger.info("spaCy en_core_web_sm loaded (optimized).")
        return nlp
    except Exception as e:
        logger.error("spaCy model not available: %s", e)
        raise RuntimeError("Please run: python -m spacy download en_core_web_sm") from e

NLP = _load_spacy_model()

# Small list for fallback human detection
_HUMAN_HINTS = [
    "man", "woman", "person", "elder", "master", "teacher", "warrior",
    "monk", "martial artist", "samurai", "fighter", "grandmaster",
    "portrait", "face", "head", "actor", "reader"
]

def _first_human_hint(text: str) -> Optional[str]:
    t = text.lower()
    for hint in _HUMAN_HINTS:
        if hint in t:
            # return a small phrase around hint if possible
            # basic regex to capture up to 3 words before/after
            m = re.search(r"((?:\w+\s){0,3}"+re.escape(hint)+r"(?:\s\w+){0,3})", text, flags=re.I)
            if m:
                return m.group(1).strip()
            return hint
    return None

def semantic_parser(prompt: str) -> Dict[str, Optional[str]]:
    """
    spaCy-based dependency-first semantic parsing that extracts subject/action/object/environment.
    Uses robust fallbacks and returns dict keys: subject, action, object, environment.
    """
    doc = NLP(prompt)

    def get_chunk_for_token(token):
        if token is None: return None
        for chunk in doc.noun_chunks:
            if token.i >= chunk.start and token.i < chunk.end:
                return chunk.text
        return token.text

    subject_tok, action_tok, object_tok, env_tok = None, None, None, None

    # 1. Find ROOT or first verb
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            action_tok = token
            break
    if action_tok is None:
        for token in doc:
            if token.pos_ == "VERB":
                action_tok = token
                break

    # 2. Collect dependents of the main verb
    if action_tok:
        for child in action_tok.children:
            if child.dep_ in {"nsubj", "nsubj:pass"}:
                subject_tok = child
            if child.dep_ in {"dobj", "obj", "pobj"}:
                object_tok = child
            if child.dep_ == "prep":
                # environment candidates
                if child.text.lower() in ["in", "at", "on", "inside", "near", "under", "within", "amid", "among"]:
                    for p_child in child.children:
                        if p_child.dep_ == "pobj":
                            env_tok = p_child
                        # sometimes environment appears as noun_chunk after prep
    # 3. Fallbacks
    if subject_tok is None:
        for token in doc:
            if token.dep_ == "nsubj":
                subject_tok = token
                break
    if subject_tok is None:
        # fallback to first noun chunk
        for chunk in doc.noun_chunks:
            subject_tok = chunk.root
            break

    if env_tok is None:
        for token in doc:
            if token.dep_ == "pobj" and token.head.text.lower() in {"in", "at", "on", "inside", "near"}:
                env_tok = token; break

    # 4. Convert tokens to text
    subject = get_chunk_for_token(subject_tok)
    action = action_tok.lemma_ if action_tok else None
    obj = get_chunk_for_token(object_tok)
    environment = get_chunk_for_token(env_tok)

    # 5. Post processing & safety
    if obj and environment and obj == environment:
        obj = None

    # If subject missing or subject is too generic and prompt contains a human hint, anchor it
    if not subject or subject.strip().lower() in {"photorealistic portrait", "portrait", "photorealistic"}:
        hint = _first_human_hint(prompt)
        if hint:
            subject = hint

    parsed = {"subject": subject, "action": action, "object": obj, "environment": environment}
    logger.info(f"semantic_parser -> {parsed}")
    return parsed


# Optional helper to slightly reweight parts for safety (callers can use if desired)
def reweight_parts(parts: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """
    Small, conservative reweighting:
    - If subject contains 'portrait' but also a human hint, prefer 'person in context' style to avoid ECU.
    - This function does NOT overwrite rich subjects, it only avoids degenerate 'portrait' alone.
    """
    subj = parts.get("subject") or ""
    if subj and "portrait" in subj.lower() and not any(w in subj.lower() for w in ["man","woman","martial","samurai","artist","master"]):
        # expand to contextualized subject
        parts["subject"] = (subj + " in full context").strip()
    return parts
