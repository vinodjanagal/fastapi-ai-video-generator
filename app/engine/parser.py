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


def _simple_subject_heuristic(text: str) -> Optional[str]:
    """
    Very small fallback heuristic: return first continuous chunk of words before a verb.
    Not perfect but prevents crashes.
    """
    # split at common verb tokens to isolate likely subject phrase
    verbs = r"\b(is|are|was|were|finds|find|finds|find|picks|picks up|takes|holds|walks|runs|looks|sees|discovers|opens|closes|turns)\b"
    parts = re.split(verbs, text, flags=re.IGNORECASE)
    if parts:
        candidate = parts[0].strip()
        # clamp length
        if len(candidate) > 0:
            # keep up to 6 words for subject candidate
            words = candidate.split()
            return " ".join(words[:6])
    return None


def semantic_parser(text: str) -> Dict[str, Any]:
    """
    Returns a small semantic dict: subject, action, object, environment (may be None).
    Uses spaCy if available, otherwise uses a small heuristic.
    """
    out = {"subject": None, "action": None, "object": None, "environment": None}
    if not text:
        return out

    if _nlp:
        try:
            doc = _nlp(text)
            # noun_chunks may be a generator in some spaCy versions, so convert to list
            chunks = list(doc.noun_chunks)
            if chunks:
                subject_tok = chunks[0].root
                out["subject"] = " ".join([tok.text for tok in subject_tok.subtree])
            # simple action detection: first verb
            verb = next((tok for tok in doc if tok.pos_ == "VERB"), None)
            if verb:
                out["action"] = verb.lemma_
            # object: look for direct object (dobj) dependency if present
            dobj = next((tok for tok in doc if tok.dep_ == "dobj"), None)
            if dobj:
                out["object"] = " ".join([t.text for t in dobj.subtree])
            # environment extraction: look for prepositional objects (pobj) after 'in', 'on', 'at'
            for tok in doc:
                if tok.text.lower() in ("in", "on", "at", "inside", "outside"):
                    # find the pobj child
                    pobj = next((child for child in tok.children if child.dep_ == "pobj"), None)
                    if pobj:
                        out["environment"] = " ".join([t.text for t in pobj.subtree])
                        break
        except Exception as e:
            logger.warning("spaCy parsing failed, falling back to heuristics: %s", e)
            out["subject"] = _simple_subject_heuristic(text)
    else:
        # fallback heuristic
        out["subject"] = _simple_subject_heuristic(text)
        # action: first verb-like token by regex
        m = re.search(r"\b(find|finds|takes|holds|walks|runs|looks|sees|discovers|opens|closes|turns|moves)\b", text, flags=re.IGNORECASE)
        if m:
            out["action"] = m.group(0).lower()

    # normalize empty strings to None
    for k in out:
        if isinstance(out[k], str):
            out[k] = out[k].strip() or None

    return out
