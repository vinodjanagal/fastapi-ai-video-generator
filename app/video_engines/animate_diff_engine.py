# app/video_engines/animate_diff_engine.py
# FINAL VERSION - RESTORING YOUR CORRECT LOGIC + PYTHONPATH FIX
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from statistics import pstdev
import gc

# --- FIX: Robust import with dynamic PYTHONPATH ---
def _ensure_project_root_in_path():
    """Ensure the project root is in sys.path regardless of cwd."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # D:\revision
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# -------- logging --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [animate_diff_engine] - %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

_ensure_project_root_in_path()

try:
    from app.video_engines.heavy_config import HeavyEngineConfig
except ModuleNotFoundError as e:
    if "app" in str(e):
        logger.error("Failed to import HeavyEngineConfig. Make sure you're running from project root or use PYTHONPATH.")
        logger.error("Try: set PYTHONPATH=D:\\revision && python app\\video_engines\\animate_diff_engine.py ...")
    raise

import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from PIL import Image
# Components from Diffusers
from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    AutoencoderKL,
)
# Components from Transformers
from transformers import CLIPTextModel, CLIPTokenizer



# -------- logging --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [animate_diff_engine] - %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)
# =============== Utilities ===============


def _select_device_and_dtype(prefer_cuda_fp16: bool = True) -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available(): return torch.device("cuda"), (torch.float16 if prefer_cuda_fp16 else torch.float32)
    return torch.device("cpu"), torch.float32


def _apply_memory_saving(pipe: AnimateDiffPipeline, device: torch.device) -> None:
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        try: getattr(pipe, fn)()
        except Exception: pass
    try:
        if device.type == "cuda": pipe.enable_sequential_cpu_offload()
    except Exception: pass


def _maybe_swap_vae(pipe: AnimateDiffPipeline, vae_id: Optional[str], device: torch.device, dtype: torch.dtype) -> AnimateDiffPipeline:
    if not vae_id: return pipe
    try:
        logger.info(f"[vae] Swapping VAE -> {vae_id}")
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
        vae.to(device)
        pipe.vae = vae
    except Exception as e: logger.warning(f"[vae] VAE swap failed ({vae_id}): {e}")
    return pipe


def _image_entropy_variance(im: Image.Image) -> Tuple[float, float]:
    g = im.convert("L"); pixels = list(g.getdata()); hist = g.histogram(); total = sum(hist) or 1
    from math import log2
    entropy = 0.0
    for c in hist:
        if c: p = c / total; entropy -= p * log2(p)
    sigma = pstdev(pixels)
    return entropy, sigma


def _looks_like_noise(im: Image.Image) -> bool:
    entropy, sigma = _image_entropy_variance(im)
    logger.info(f"[quality] entropy={entropy:.2f} sigma={sigma:.2f}")
    return (entropy > 7.8 or sigma > 85.0)
# =====================================================================
# === YOUR CORRECT PIPELINE BUILDER LOGIC - FINALLY RESTORED ==========
# =====================================================================



def _build_lightning_pipeline_from_config(*,
                                          base_model: str,
                                          lightning_repo: str,
                                          lightning_file: str,
                                          device: torch.device,
                                          dtype: torch.dtype,
                                          vae_id: Optional[str]) -> AnimateDiffPipeline:
    logger.info(f"[build_from_config] 1. Reading UNet config from style model '{base_model}'")
    unet_config = UNet2DConditionModel.load_config(base_model, subfolder="unet")
    logger.info("[build_from_config] 2. Creating MotionAdapter structure FROM the UNet's config.")
    motion_adapter = MotionAdapter.from_config(unet_config, torch_dtype=dtype)
    logger.info(f"[build_from_config] 3. Loading Lightning weights into the adapter from '{lightning_repo}'")
    lightning_path = hf_hub_download(lightning_repo, lightning_file)
    motion_adapter.load_state_dict(load_file(lightning_path, device="cpu"), strict=False)
    logger.info(f"[build_from_config] 4. Assembling pipeline using all components from '{base_model}'")
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=motion_adapter,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear"
    )
   
    pipe = _maybe_swap_vae(pipe, vae_id=vae_id, device=device, dtype=dtype)
    _apply_memory_saving(pipe, device)
   
    logger.info("[build_from_config] Pipeline ready.")
    return pipe
# =============== Generation ===============



def _run_inference(pipe: AnimateDiffPipeline, prompt: str, negative_prompt: str, frames: int, steps: int, guidance: float, width: int, height: int, seed: Optional[int]) -> List[Image.Image]:

    if steps not in {1, 2, 4, 8}:
        logger.warning("[lightning] steps must be in {1,2,4,8}; forcing steps=4"); steps = 4
    if guidance is None: guidance = 1.0
    generator = torch.Generator("cpu").manual_seed(int(seed)) if seed is not None else None
    out = pipe(prompt=prompt, negative_prompt=negative_prompt, num_frames=frames, num_inference_steps=steps, guidance_scale=guidance, width=width, height=height, generator=generator)
    return out.frames[0]



def _save_frames(frames: List[Image.Image], output_dir: Path, preview: bool) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True); paths: List[str] = []
    if preview:
        p = output_dir / "preview_000.png"; frames[0].save(p); paths.append(str(p)); return paths
    for i, fr in enumerate(frames):
        fp = output_dir / f"frame_{i:04d}.png"; fr.save(fp); paths.append(str(fp))
    return paths


def generate_frames(cfg: HeavyEngineConfig, *, prompt: str, negative_prompt: str, output_dir: Path, base_model: str, motion_repo: str, motion_file: str, num_frames: Optional[int], num_steps: Optional[int], guidance_scale: Optional[float], width: Optional[int], height: Optional[int], seed: Optional[int], preview: bool, prefer_cuda_fp16: bool, fallback_sd15: bool) -> List[str]:
    # This function signature now accepts `negative_prompt: str`
    
    device, dtype = _select_device_and_dtype(prefer_cuda_fp16=prefer_cuda_fp16)
    logger.info(f"Device={device} dtype={dtype} preview={preview} mode=lightning")
    eff_frames = 1 if preview else (num_frames or cfg.ANIMATEDIFF_NUM_FRAMES)
    eff_steps = (num_steps or cfg.ANIMATEDIFF_NUM_STEPS)
    eff_guidance = guidance_scale if guidance_scale is not None else (cfg.ANIMATEDIFF_GUIDANCE_SCALE or 1.0)
    eff_width = width or cfg.IMG_WIDTH
    eff_height = height or cfg.IMG_HEIGHT
    
    # --- COMBINE NEGATIVE PROMPTS ---
    # Combine the base negative prompt with the one from the command line for the best effect.
    final_negative_prompt = f"{cfg.BASE_NEGATIVE_PROMPT}, {negative_prompt}" if negative_prompt else cfg.BASE_NEGATIVE_PROMPT
    
    pipe = _build_lightning_pipeline_from_config(
        base_model=base_model, lightning_repo=motion_repo, lightning_file=motion_file,
        device=device, dtype=dtype, vae_id=getattr(cfg, "VAE_MODEL", None),
    )
    if device.type == "cpu": logger.warning("Running on CPU — use --preview first to validate output.")
    
    logger.info(f"Generating frames: frames={eff_frames}, steps={eff_steps}, guidance={eff_guidance}, size={eff_width}x{eff_height}")
    
    # --- CHANGE #1: Use the final_negative_prompt ---
    frames = _run_inference(pipe=pipe, prompt=prompt, negative_prompt=final_negative_prompt, frames=eff_frames, steps=eff_steps, guidance=eff_guidance, width=eff_width, height=eff_height, seed=seed)
   
    if preview and fallback_sd15 and _looks_like_noise(frames[0]):
        logger.warning("[quality] Preview looks like noise. Falling back to base_model='runwayml/stable-diffusion-v1-5'.")
        pipe_sd15 = _build_lightning_pipeline_from_config(
            base_model="runwayml/stable-diffusion-v1-5", lightning_repo=motion_repo, lightning_file=motion_file,
            device=device, dtype=dtype, vae_id=getattr(cfg, "VAE_MODEL", None),
        )
        # --- CHANGE #2: Also use it in the fallback ---
        frames = _run_inference(pipe=pipe_sd15, prompt=prompt, negative_prompt=final_negative_prompt, frames=eff_frames, steps=eff_steps, guidance=eff_guidance, width=eff_width, height=eff_height, seed=seed)
        
    return _save_frames(frames, output_dir, preview=preview)


# =============== CLI ===============
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AnimateDiff Lightning Engine (Robust build from config)")
    p.add_argument("--prompt", required=True, help="Text prompt.")
    p.add_argument("--output-dir", required=True, help="Directory to save frames.")
    p.add_argument("--base-model", default="Lykon/dreamshaper-8", help="SD1.5-compatible style model.")
    p.add_argument("--motion-repo", default="ByteDance/AnimateDiff-Lightning", help="Lightning adapter repo.")
    p.add_argument("--motion-file", default="animatediff_lightning_4step_comfyui.safetensors", help="Lightning adapter filename.")
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--preview", action="store_true", help="Generate a single preview frame.")
    p.add_argument("--no-cuda-fp16", action="store_true", help="Force float32 on CUDA.")
    p.add_argument("--fallback-sd15", action="store_true", help="If preview is noise, retry with SD1.5 base.")
    p.add_argument("--negative-prompt", type=str, default="", help="Prompt of concepts to avoid.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        cfg = HeavyEngineConfig()
        paths = generate_frames(cfg=cfg, prompt=args.prompt,negative_prompt=args.negative_prompt, output_dir=Path(args.output_dir), base_model=args.base_model, motion_repo=args.motion_repo, motion_file=args.motion_file, num_frames=args.num_frames, num_steps=args.num_steps, guidance_scale=args.guidance_scale, width=args.width, height=args.height, seed=args.seed, preview=args.preview, prefer_cuda_fp16=(not args.no_cuda_fp16), fallback_sd15=bool(args.fallback_sd15))
        result = {"status": "COMPLETED", "mode": "preview" if args.preview else "full", "frame_paths": paths}
        if args.preview and paths: result["preview_path"] = paths[0]
        print(json.dumps(result))
        return 0
    except Exception as e:
        logger.error(f"An error occurred in main execution: {e}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"{type(e).__name__}: {e}"}))
        return 1
    
    
if __name__ == "__main__":
    sys.exit(main())