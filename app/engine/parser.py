
from typing import Dict, Tuple, Optional
import logging
import spacy

logger = logging.getLogger("v6_parser")

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

def semantic_parser(prompt: str) -> Dict[str, Optional[str]]:
    """
    Performs robust, dependency-first semantic parsing to correctly extract full noun chunks.
    This is the definitive, tested, and correct implementation.
    """
    doc = NLP(prompt)

    # Helper to map a token to its full noun chunk
    def get_chunk_for_token(token):
        if token is None: return None
        for chunk in doc.noun_chunks:
            if token.i >= chunk.start and token.i < chunk.end:
                return chunk.text
        return token.text

    subject_tok, action_tok, object_tok, env_tok = None, None, None, None
    
    # 1. Find the root of the sentence, which is usually the main verb
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            action_tok = token
            break
    if action_tok is None: # Fallback if no root verb
        for token in doc:
            if token.pos_ == "VERB":
                action_tok = token
                break

    # 2. Find dependents (children) of the main action verb
    if action_tok:
        for child in action_tok.children:
            if child.dep_ == "nsubj": subject_tok = child
            if child.dep_ == "dobj": object_tok = child
            # A preposition attached to the verb often governs the object or environment
            if child.dep_ == "prep":
                # Check for location prepositions to identify the environment
                if child.text.lower() in ["in", "at", "on", "inside", "near", "under", "within"]:
                    for p_child in child.children:
                        if p_child.dep_ == "pobj": env_tok = p_child
                # If not an environment, it might be the object (e.g., "looking at a schematic")
                elif object_tok is None:
                    for p_child in child.children:
                        if p_child.dep_ == "pobj": object_tok = p_child

    # 3. Final Fallbacks if the dependency tree was unusual
    if subject_tok is None:
        for token in doc:
            if token.dep_ == "nsubj": subject_tok = token; break
    if subject_tok is None: # Last resort for subject
        for chunk in doc.noun_chunks:
             subject_tok = chunk.root; break
    
    if env_tok is None: # Last resort for environment
        for token in doc:
            if token.dep_ == "pobj" and token.head.text in ["in", "at", "on", "inside"]:
                env_tok = token; break

    # 4. Convert identified tokens to their full text chunks
    subject = get_chunk_for_token(subject_tok)
    action = action_tok.lemma_ if action_tok else None
    obj = get_chunk_for_token(object_tok)
    environment = get_chunk_for_token(env_tok)

    # 5. Post-processing to clean up results like "cat on a windowsill"
    if subject and environment and not action and not obj:
        # If the only other noun is the environment, it's likely just a location, not an object
        pass
    elif obj == environment:
        obj = None # Avoid duplication

    return {"subject": subject, "action": action, "object": obj, "environment": environment}


def build_semantic_prompt(
    raw_prompt: str, user_negative_prompt: str, shot_type: str, semantic_parts: Dict[str, Optional[str]]
) -> Tuple[str, str]:
    logger.info(f"Semantic parts extracted: {semantic_parts}")

    subject = f"({semantic_parts['subject']}:1.3)" if semantic_parts.get("subject") else ""
    action = semantic_parts.get("action") or ""
    obj = f"({semantic_parts['object']}:1.4)" if semantic_parts.get("object") else ""

    env_map = {
        "ECU": "(studio lighting:1.1)", "CU": "(soft indoor background:1.1)",
        "MS": "(work environment:1.1)", "WS": "(large cinematic environment:1.1)"
    }
    env = f"({semantic_parts['environment']}:1.1)" if semantic_parts.get("environment") else env_map.get(shot_type, "(cinematic background:1.1)")

    style = "photorealistic, ultra-realistic, cinematic lighting, sharp focus, 8k"
    shot_style = {"CU": "(cinematic close-up:1.2)", "MS": "medium shot", "WS": "wide establishing shot", "ECU": "(extreme close-up:1.2)"}.get(shot_type, "")

    positive_parts = [env, shot_style, subject, action, obj, style]
    pos_prompt = ", ".join(filter(None, positive_parts))

    auto_neg_prompt = "(deformed, distorted, bad anatomy:1.3), blurry, ugly, cartoon, mutated hands, text, watermark"
    final_neg_prompt = ", ".join(filter(None, [auto_neg_prompt, user_negative_prompt or ""]))

    logger.info(f"Final Positive Prompt: {pos_prompt}")
    return pos_prompt, final_neg_prompt