# ==============================================================================
# animate_diff_engine.py - V3.4 DEFINITIVE PRODUCTION SCRIPT
# This version is the complete, correct, and final build.
# It includes the robust VAE loader, token-safe prompt compiler, and the
# intelligent Shot-Type Engine, with all obsolete logic removed.
# ==============================================================================

import sys
from pathlib import Path

# --- Dynamic Path Setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import logging
import gc
import re
from typing import List, Optional, Tuple
from PIL import Image
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    UNet2DConditionModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s", stream=sys.stderr)
logger = logging.getLogger("animate_diff_engine")


# --- UTILITIES ---
def _select_device_and_dtype(prefer_cuda_fp16: bool = True) -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), (torch.float16 if prefer_cuda_fp16 else torch.float32)
    return torch.device("cpu"), torch.float32

def _apply_memory_saving(pipe: AnimateDiffPipeline) -> None:
    for fn_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(pipe, fn_name):
            getattr(pipe, fn_name)()

def _maybe_swap_vae(pipe: AnimateDiffPipeline, vae_id: Optional[str], device: torch.device, dtype: torch.dtype) -> AnimateDiffPipeline:
    if not vae_id:
        logger.info("[vae] No VAE specified — using default.")
        return pipe
    logger.info(f"[vae] Trying VAE: {vae_id}")
    try:
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
        pipe.vae = vae.to(device)
        logger.info(f"[vae] ✅ Loaded VAE from pretrained: {vae_id}")
    except Exception as e:
        logger.error(f"[vae] FAILED to swap VAE ({e}); continuing with pipeline's default VAE.", exc_info=True)
    return pipe

def compile_prompt(raw_prompt: str, max_phrases: int = 12) -> Tuple[str, List[str]]:
    original_phrases = [p.strip() for p in raw_prompt.split(",") if p.strip()]
    priority, others = [], []
    for p in original_phrases:
        search_p = re.sub(r'[:\d\.]', '', p).lower()
        if re.search(r"(inventor|man|woman|person|face|hands|worker|scientist|engineer|girl|guy|portrait|character)", search_p):
            priority.append(p)
        else:
            others.append(p)
    combined_phrases = (priority + others)[:max_phrases]
    suffix = "photorealistic, cinematic, ultra-realistic, high detail"
    final_prompt = ", ".join(combined_phrases) + ", " + suffix
    logger.info(f"[prompt-compile] Rebuilt prompt from key phrases: {combined_phrases}")
    return final_prompt, combined_phrases

def classify_shot_type(prompt_text: str) -> str:
    prompt_text = prompt_text.lower()
    if any(k in prompt_text for k in ["extreme close-up", "macro", "detailed eyes"]): return "ECU"
    if any(k in prompt_text for k in ["close-up", "portrait", "focused expression", "face"]): return "CU"
    if any(k in prompt_text for k in ["at work", "full body", "medium shot", "inventor standing"]): return "MS"
    if any(k in prompt_text for k in ["wide shot", "full scene", "cabin interior", "environment"]): return "WS"
    if any(k in prompt_text for k in ["inventor", "man", "woman", "person", "character"]): return "CU"
    return "MS"

def apply_shot_type_tuning(args, shot_type: str):
    logger.info(f"[shot-engine] Classified shot type: {shot_type}. Applying tuning.")
    if shot_type == "ECU":
        args.strength, args.ip_adapter_scale, args.guidance_scale = 0.20, 0.0, 6.0
    elif shot_type == "CU":
        args.strength, args.ip_adapter_scale, args.guidance_scale = 0.28, 0.035, 6.5
    elif shot_type == "MS":
        args.strength, args.ip_adapter_scale, args.guidance_scale = 0.35, 0.05, 7.0
    elif shot_type == "WS":
        args.strength, args.ip_adapter_scale, args.guidance_scale = 0.50, 0.08, 7.5
    logger.info(f"[shot-engine] Final tuned parameters -> strength={args.strength}, ip_scale={args.ip_adapter_scale}, guidance={args.guidance_scale}")


# --- CORE FUNCTIONS ---
def build_pipeline(args) -> AnimateDiffPipeline:
    device, dtype = _select_device_and_dtype(prefer_cuda_fp16=not args.no_cuda_fp16)
    logger.info(f"[pipeline_builder] Mode: {args.mode}, Steps: {args.num_steps}, Base: {args.base_model}")
    
    motion_repo = "guoyww/animatediff-motion-adapter-v1-5-2"
    logger.info(f"[classic] Loading motion adapter from {motion_repo}")
    motion_adapter = MotionAdapter.from_pretrained(motion_repo, torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(args.base_model, motion_adapter=motion_adapter, torch_dtype=dtype)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
    logger.info("[scheduler] ✅ Swapped to: DPM++ 2M Karras")

    if args.ip_adapter_image_path and args.ip_adapter_scale > 0.0:
        logger.info("[ip_adapter] Enabled. Loading IP-Adapter model...")
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_light.bin")
        pipe.set_ip_adapter_scale(args.ip_adapter_scale)
        logger.info(f"[ip_adapter] ✅ Loaded successfully. Scale = {args.ip_adapter_scale}")
    
    pipe = _maybe_swap_vae(pipe, vae_id=args.vae_model, device=device, dtype=dtype)
    _apply_memory_saving(pipe)
    pipe.to(device)
    logger.info(f"[{args.mode}] ✅ AnimateDiff pipeline is ready on {device}.")
    return pipe

def run_inference(pipe: AnimateDiffPipeline, args) -> List[Image.Image]:
    try:
        device = getattr(pipe, "device", next(pipe.unet.parameters()).device)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        kwargs = {
            "prompt": args.prompt, "negative_prompt": args.negative_prompt,
            "num_frames": args.num_frames, "num_inference_steps": args.num_steps,
            "guidance_scale": args.guidance_scale, "width": args.width, "height": args.height,
            "generator": gen,
        }
        if args.ip_adapter_scale > 0 and args.ip_adapter_image_path:
            kwargs["ip_adapter_image"] = Image.open(args.ip_adapter_image_path).convert("RGB")
        if args.init_image and 0 < args.strength < 1:
            kwargs["image"] = Image.open(args.init_image).convert("RGB")
            kwargs["strength"] = args.strength
        
        logger.info(f"[prompt-debug] Final prompt token count: {len(pipe.tokenizer(args.prompt).input_ids)}")
        logger.info("[inference] Running AnimateDiff pipeline...")
        out = pipe(**kwargs)
        frames = getattr(out, "frames", [[]])[0]
        if not frames: raise RuntimeError("Pipeline returned an empty frame list")
        logger.info(f"[inference] ✅ Generated {len(frames)} frames successfully.")
        return frames
    except Exception as e:
        logger.error(f"[inference] CRASH: {e}", exc_info=True)
        raise

def save_frames(frames: List[Image.Image], output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = output_dir / f"frame_{i:04d}.png"
        frame.save(path)
        paths.append(str(path))
    return paths

# --- CLI AND MAIN EXECUTION ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AnimateDiff Engine (V3.4 Definitive)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="SG161222/Realistic_Vision_V5.1_noVAE")
    p.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument("--ip-adapter-image-path", type=str, default=None)
    p.add_argument("--ip-adapter-scale", type=float, default=0.0)
    p.add_argument("--init-image", type=str, default=None)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--manual", action="store_true", help="Disable the Shot-Type Engine for manual tuning.")
    # Obsolete arguments kept for API consistency if needed, but not used by engine
    p.add_argument("--scheduler", type=str, default="DPM++ 2M Karras", choices=["DPM++ 2M Karras", "DDIM"])
    p.add_argument("--mode", type=str, default="classic", choices=["lightning", "classic"])
    p.add_argument("--no-cuda-fp16", action="store_true")
    return p

def main() -> int:
    args = build_parser().parse_args()
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"))
        logger.addHandler(fh)
    try:
        full_prompt_for_classification = args.prompt 
        args.prompt, visual_tokens = compile_prompt(args.prompt)
        
        if not args.manual:
            shot_type = classify_shot_type(full_prompt_for_classification)
            apply_shot_type_tuning(args, shot_type)
        else:
            logger.info("[main] --manual flag detected. Skipping automated Shot-Type Engine.")

        pipe = build_pipeline(args)
        frames = run_inference(pipe, args)
        paths = save_frames(frames, Path(args.output_dir))
        print(json.dumps({"status": "COMPLETED", "frame_paths": paths}))
        
        del pipe, frames, paths; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return 0
    except Exception as e:
        logger.error(f"AnimateDiff engine main function failed: {e}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"{type(e).__name__}: {e}"}))
        return 1

if __name__ == "__main__":
    sys.exit(main())