import logging

from typing import List, Optional, Tuple
from PIL import Image
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

import sys, json, gc, re, argparse
from pathlib import Path

import spacy
from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler, AutoencoderKL

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s", stream=sys.stderr)
logger = logging.getLogger("v6_engine")
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path: sys.path.insert(0, str(project_root))

# --- V6.0 SEMANTIC PARSING ENGINE ---
try:
    NLP = spacy.load("en_core_web_sm")
    logger.info("[v6-engine] Lightweight NLP model (spaCy) loaded successfully.")
except IOError:
    logger.error("[v6-engine] CRITICAL: spaCy model 'en_core_web_sm' not found.")
    logger.error("--> Please run: python -m spacy download en_core_web_sm")
    sys.exit(1)

def semantic_parser(prompt: str) -> dict:
    doc = NLP(prompt)
    subject, action, obj, environment = None, None, None, None
    for token in doc:
        if subject is None and token.dep_ == "nsubj" and token.pos_ in ("NOUN", "PROPN"): subject = token.text
        if action is None and token.pos_ == "VERB": action = token.lemma_
        if obj is None and token.dep_ == "dobj": obj = token.text
    # V6.2 FIX: Fallback for objects in prepositional phrases
    if obj is None:
        for token in doc:
            if token.dep_ == "pobj" and token.pos_ == "NOUN": obj = token.text; break
    if subject is None: # Fallback for subject
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN"): subject = token.text; break
    for chunk in doc.noun_chunks:
        if chunk.root.head.dep_ == "prep": environment = chunk.text; break
    return {"subject": subject, "action": action, "object": obj, "environment": environment}

def build_semantic_prompt(raw_prompt: str, user_negative_prompt: str, shot_type: str, semantic_parts: dict) -> Tuple[str, str]:
    logger.info(f"[v6-parser] Semantic parts extracted: {semantic_parts}")
    # V6.2 FIX: Use parentheses for correct weighting
    subject = f"({semantic_parts['subject']}:1.3)" if semantic_parts["subject"] else ""
    action = semantic_parts["action"] or ""
    obj = f"({semantic_parts['object']}:1.4)" if semantic_parts["object"] else ""
    # V6.2 FIX: Context-aware environment fallback
    env = f"({semantic_parts['environment']}:1.1)" if semantic_parts["environment"] else {
        "ECU": "(studio lighting:1.1)", "CU": "(soft indoor background:1.1)",
        "MS": "(work environment:1.1)", "WS": "(large cinematic environment:1.1)"
    }.get(shot_type, "(cinematic background:1.1)")
    style = "photorealistic, ultra-realistic, cinematic lighting, sharp focus, 8k"
    shot_style = {
        "CU": "(cinematic close-up:1.2)", "MS": "medium shot",
        "WS": "wide establishing shot", "ECU": "(extreme close-up:1.2)"
    }.get(shot_type, "")
    positive_parts = [env, shot_style, subject, action, obj, style]
    pos_prompt = ", ".join(filter(None, positive_parts))
    # V6.2 FIX: Merge auto-generated negative with user-provided negative
    auto_neg_prompt = "(deformed, distorted, bad anatomy:1.3), blurry, ugly, cartoon, mutated hands, text, watermark, signature, wooden panel"
    final_neg_prompt = ", ".join(filter(None, [auto_neg_prompt, user_negative_prompt]))
    logger.info(f"[v6-parser] Final Positive Prompt: {pos_prompt}")
    return pos_prompt, final_neg_prompt

# --- V6.2 UTILITY FUNCTIONS (REFINED) ---
def classify_shot_type(prompt_text: str, semantic_parts: dict) -> str:
    text = prompt_text.lower()
    if any(k in text for k in ["macro", "extreme close-up"]): return "ECU"
    if any(k in text for k in ["close-up", "portrait", "face"]): return "CU"
    if any(k in text for k in ["wide shot", "full scene"]): return "WS"
    if semantic_parts.get("subject") and semantic_parts.get("environment"): return "MS"
    if semantic_parts.get("subject"): return "CU"
    return "MS"

def apply_param_tuning(args, shot_type: str):
    logger.info(f"[param-engine] Classified shot type: {shot_type}. Applying tuning.")
    if not args.manual:
        # Only set defaults if the user has not provided them via CLI
        args.strength = args.strength if args.strength is not None else {"ECU": 0.20, "CU": 0.28, "MS": 0.35, "WS": 0.50}.get(shot_type, 0.3)
        args.ip_adapter_scale = args.ip_adapter_scale if args.ip_adapter_scale is not None else {"ECU": 0.0, "CU": 0.035, "MS": 0.05, "WS": 0.08}.get(shot_type, 0.04)
        args.guidance_scale = args.guidance_scale if args.guidance_scale is not None else {"ECU": 6.0, "CU": 6.5, "MS": 7.0, "WS": 7.5}.get(shot_type, 7.0)
    logger.info(f"[param-engine] Final tuned parameters -> strength={args.strength}, ip_scale={args.ip_adapter_scale}, guidance={args.guidance_scale}")

def apply_safety_guards(args):
    if not args.manual and args.num_steps < 15:
        logger.warning(f"[safety-guard] num_steps was {args.num_steps}, too low. Forcing to 15."); args.num_steps = 15

def _maybe_swap_vae(pipe, vae_id, device, dtype):
    if not vae_id: return pipe
    logger.info(f"[vae] Trying VAE: {vae_id}")
    try:
        vae_dtype = torch.float32 if device.type == "cpu" else dtype
        try: # V6.2 FIX: Try standard subfolder first
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype, subfolder="vae")
        except Exception: # Fallback for VAEs not in a subfolder
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype)
        pipe.vae = vae.to(device)
        logger.info(f"✅ VAE swapped successfully: {vae_id}")
    except Exception as e:
        logger.error(f"CRITICAL: VAE loading failed: {e}", exc_info=True); raise
    return pipe

def _load_ip_adapter_safe(pipe, repo, device):
    # V6.2 FIX: Robust, multi-path IP-Adapter loader
    candidates = [("models", "ip-adapter_sd15.bin"), ("models", "ip-adapter_sd15_light.bin"), ("", "ip-adapter_sd15.bin")]
    for sub, weight in candidates:
        try:
            pipe.load_ip_adapter(repo, subfolder=sub, weight_name=weight)
            logger.info(f"✅ IP-Adapter loaded successfully from {repo}/{sub}/{weight}")
            return True
        except Exception:
            continue
    logger.error(f"CRITICAL: All IP-Adapter load attempts failed for repo '{repo}'.")
    return False

# --- CORE RENDERING LOGIC ---
def build_pipeline(args, device, dtype):
    logger.info(f"Loading pipeline for base model: {args.base_model}")
    motion_adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(args.base_model, motion_adapter=motion_adapter, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    pipe = _maybe_swap_vae(pipe, args.vae_model, device, dtype)
    if args.ip_adapter_scale > 0.0 and args.ip_adapter_image_path:
        if _load_ip_adapter_safe(pipe, args.ip_adapter_repo, device):
            pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    # V6.2 FIX: Add memory slicing for CPU
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.to(device)
    return pipe

def run_inference(pipe, args, device):
    gen = torch.Generator(device=device).manual_seed(args.seed)
    # V6.2 FIX: Pre-normalize all image inputs
    ip_adapter_image = Image.open(args.ip_adapter_image_path).convert("RGB").resize((args.width, args.height), Image.LANCZOS) if args.ip_adapter_image_path else None
    init_image = Image.open(args.init_image).convert("RGB").resize((args.width, args.height), Image.LANCZOS) if args.init_image else None
    kwargs = {"prompt": args.prompt, "negative_prompt": args.negative_prompt, "num_frames": args.num_frames, "num_inference_steps": args.num_steps, "guidance_scale": args.guidance_scale, "width": args.width, "height": args.height, "generator": gen}
    if args.ip_adapter_scale > 0: kwargs["ip_adapter_image"] = ip_adapter_image
    if args.init_image and 0 < args.strength < 1: kwargs["image"] = init_image; kwargs["strength"] = args.strength
    logger.info("Running inference...")
    out = pipe(**kwargs)
    frames = out.frames[0] # V6.2 FIX: Use direct access
    if not frames: raise RuntimeError("Pipeline returned an empty frame list")
    logger.info(f"✅ Generated {len(frames)} frames successfully.")
    return frames

def build_parser():
    p = argparse.ArgumentParser(description="AnimateDiff V6.2 Production-Ready Semantic Engine")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-image", type=str)
    p.add_argument("--ip-adapter-image-path", type=str)
    p.add_argument("--base-model", default="SG161222/Realistic_Vision_V5.1_noVAE")
    p.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ip-adapter-repo", default="h94/IP-Adapter")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--manual", action="store_true")
    p.add_argument("--strength", type=float, default=None)
    p.add_argument("--ip-adapter-scale", type=float, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--negative-prompt", type=str, default="", help="Optional user-provided negative prompts to merge.")
    return p

# --- MAIN ORCHESTRATOR ---
if __name__ == "__main__":
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        device, dtype = torch.device("cpu"), torch.float32
       
        semantic_parts = semantic_parser(args.prompt)
        shot_type = classify_shot_type(args.prompt, semantic_parts)
        args.prompt, args.negative_prompt = build_semantic_prompt(args.prompt, args.negative_prompt, shot_type, semantic_parts)
       
        apply_param_tuning(args, shot_type)
        apply_safety_guards(args)
       
        pipe = build_pipeline(args, device, dtype)
        frames = run_inference(pipe, args, device)
       
        if frames:
            # V6.2 FIX: Correct save logic
            preview_path = output_dir / "preview_frame_0000.png"
            frames[0].save(preview_path)
            logger.info(f"[preview] Preview saved to: {preview_path}")
            paths = []
            for i, frame in enumerate(frames):
                path = output_dir / f"frame_{i:04d}.png"; frame.save(path); paths.append(str(path))
            print(json.dumps({"status": "COMPLETED", "frame_paths": paths}))
       
        del pipe, frames, NLP; gc.collect()
       
    except Exception as e:
        logger.error(f"V6.2 Engine failed: {e}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)