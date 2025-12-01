# app/engine/parser.py
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger("phoenix.parser")
logger.setLevel(logging.INFO)

# try to import spaCy; if not present, we fallback to a light parser
try:
    import spacy
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False


def _load_spacy_model():
    """
    Load spaCy model once. If not available, raise; caller may fallback.
    """
    try:
        nlp = spacy.load("en_core_web_sm", exclude=["ner", "textcat"])
        logger.info("spaCy loaded")
        return nlp
    except Exception as e:
        logger.error("spaCy model not available: %s", e)
        raise


# create global nlp if possible
_nlp = None
if _SPACY_AVAILABLE:
    try:
        _nlp = _load_spacy_model()
    except Exception:
        _nlp = None


# -------------------------------------------------------------------
# Helper cleaning functions
# -------------------------------------------------------------------

_CAMERA_PATTERNS = [
    r"^The camera\s+\w+\s+across[^,.]*[,.]\s*",   # "The camera pans across ..."
    r"^Camera\s+\w+\s+across[^,.]*[,.]\s*",       # "Camera glides across ..."
]


def _clean_semantic_string(text: str) -> str:
    """
    Strip obvious narrative / camera language and tidy whitespace.
    This keeps only the core scene content for subject/action/object.
    """
    if not text:
        return ""

    cleaned = text

    # Remove leading camera-narration phrases like
    # "The camera pans across the library, ..."
    for pat in _CAMERA_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

    # Collapse multiple spaces and stray spaces before punctuation
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)

    return cleaned.strip()


def _truncate_at_first_comma(text: str) -> str:
    """
    For subjects we don't want huge clauses. Keep text up to first comma.
    """
    if not text:
        return ""
    parts = text.split(",", 1)
    return parts[0].strip()


def _simple_subject_heuristic(text: str) -> Optional[str]:
    """
    Very small fallback heuristic: return first continuous chunk of words
    before a verb, after narrative cleaning. Not perfect but prevents crashes.
    """
    text = _clean_semantic_string(text)
    if not text:
        return None

    # split at common verb tokens to isolate likely subject phrase
    verbs = (
        r"\b("
        r"is|are|was|were|"
        r"find|finds|"
        r"take|takes|"
        r"pick|picks|"
        r"hold|holds|"
        r"walk|walks|"
        r"run|runs|"
        r"look|looks|"
        r"see|sees|"
        r"discover|discovers|"
        r"open|opens|"
        r"close|closes|"
        r"turn|turns|"
        r"move|moves|"
        r"reach|reaches|"
        r"approach|approaches"
        r")\b"
    )
    parts = re.split(verbs, text, flags=re.IGNORECASE)
    if parts:
        candidate = parts[0].strip()
        if candidate:
            candidate = _truncate_at_first_comma(candidate)
            # keep up to 6 words for subject candidate
            words = candidate.split()
            return " ".join(words[:6])
    return None


def _simple_action_heuristic(text: str) -> Optional[str]:
    """
    Small regex-based verb picker for fallback when spaCy is not available.
    """
    text = _clean_semantic_string(text)
    if not text:
        return None

    actions = (
        r"\b("
        r"find|finds|"
        r"take|takes|"
        r"pick|picks|"
        r"hold|holds|"
        r"walk|walks|"
        r"run|runs|"
        r"look|looks|"
        r"see|sees|"
        r"discover|discovers|"
        r"open|opens|"
        r"close|closes|"
        r"turn|turns|"
        r"move|moves|"
        r"reach|reaches|"
        r"approach|approaches|"
        r"lift|lifts|"
        r"touch|touches"
        r")\b"
    )
    m = re.search(actions, text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _simple_environment_heuristic(text: str) -> Optional[str]:
    """
    Fallback env extractor: look for phrases starting with in/on/at/inside/outside.
    """
    text = _clean_semantic_string(text)
    if not text:
        return None

    m = re.search(
        r"\b(in|on|at|inside|outside)\b\s+([^.,]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        phrase = m.group(0)
        return phrase.strip()
    return None


# -------------------------------------------------------------------
# Main semantic parser (JSON-first, as per Option A)
# -------------------------------------------------------------------
def semantic_parser(
    text: str,
    composition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns a small semantic dict: subject, action, object, environment (may be None).

    JSON-FIRST (Option A):
    - If composition["environment"] exists, it is used as the primary environment.
    - Text is still parsed for subject/action/object, and may fill env only if JSON has none.
    """
    out: Dict[str, Optional[str]] = {
        "subject": None,
        "action": None,
        "object": None,
        "environment": None,
    }

    # 1) JSON-FIRST: use storyboard composition.environment if present.
    if composition:
        env = composition.get("environment")
        if isinstance(env, str) and env.strip():
            out["environment"] = env.strip()

    # If there is no text at all, return what we got from JSON.
    if not text:
        # normalize empties to None
        for k in out:
            if isinstance(out[k], str):
                out[k] = out[k].strip() or None
        return out  # type: ignore[return-value]

    cleaned_text = _clean_semantic_string(text)

    if not cleaned_text:
        # Still normalize and bail early
        for k in out:
            if isinstance(out[k], str):
                out[k] = out[k].strip() or None
        return out  # type: ignore[return-value]

    # 2) spaCy path (preferred)
    if _nlp:
        try:
            doc = _nlp(cleaned_text)

            # SUBJECT: first noun chunk that is not about the camera.
            chunks = list(doc.noun_chunks)
            for chunk in chunks:
                # Skip camera-describing chunks
                root_lemma = chunk.root.lemma_.lower()
                if root_lemma in {"camera"}:
                    continue
                subj_text = " ".join(tok.text for tok in chunk)
                subj_text = _truncate_at_first_comma(subj_text)
                out["subject"] = subj_text
                break

            # ACTION: first "real" verb, avoiding camera verbs like "pan", "glide"
            for tok in doc:
                if tok.pos_ == "VERB":
                    lemma = tok.lemma_.lower()
                    if lemma in {"pan", "glide", "sweep"}:
                        continue
                    out["action"] = lemma
                    break

            # OBJECT: direct object if present
            dobj = next((tok for tok in doc if tok.dep_ == "dobj"), None)
            if dobj:
                out["object"] = " ".join(t.text for t in dobj.subtree)

            # ENVIRONMENT: only if JSON did NOT already give us one
            if out["environment"] is None:
                for tok in doc:
                    if tok.text.lower() in ("in", "on", "at", "inside", "outside"):
                        pobj = next(
                            (child for child in tok.children if child.dep_ == "pobj"),
                            None,
                        )
                        if pobj:
                            out["environment"] = " ".join(t.text for t in pobj.subtree)
                            break

        except Exception as e:
            logger.warning("spaCy parsing failed, falling back to heuristics: %s", e)
            out["subject"] = _simple_subject_heuristic(cleaned_text)
            if out["action"] is None:
                out["action"] = _simple_action_heuristic(cleaned_text)
            if out["environment"] is None:
                out["environment"] = _simple_environment_heuristic(cleaned_text)

    else:
        # 3) No spaCy: pure heuristic path
        out["subject"] = _simple_subject_heuristic(cleaned_text)
        out["action"] = _simple_action_heuristic(cleaned_text)
        if out["environment"] is None:
            out["environment"] = _simple_environment_heuristic(cleaned_text)

    # 4) Normalize empty strings to None
    for k in out:
        if isinstance(out[k], str):
            out[k] = out[k].strip() or None

    logger.info(f"semantic_parser: text='{text}' -> {out}")
    return out  # type: ignore[return-value]
