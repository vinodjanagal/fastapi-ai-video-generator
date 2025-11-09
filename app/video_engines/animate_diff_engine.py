
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from statistics import pstdev
from PIL import Image
import torch


# --- DYNAMIC PYTHONPATH ---
def _ensure_project_root_in_path():
    """Ensure the project root is in sys.path for robust imports."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

_ensure_project_root_in_path()

# -------- LOGGING SETUP --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [animate_diff_engine] - %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# --- SAFE IMPORTS ---
try:
    from app.video_engines.heavy_config import HeavyEngineConfig
except ModuleNotFoundError:
    logger.error("Failed to import HeavyEngineConfig. Ensure you are running from the project root.")
    sys.exit(1)

import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from PIL import Image
from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    DDIMScheduler,
    AutoencoderKL,
)

# =============== UTILITIES ===============

def _select_device_and_dtype(prefer_cuda_fp16: bool = True) -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available(): return torch.device("cuda"), (torch.float16 if prefer_cuda_fp16 else torch.float32)
    return torch.device("cpu"), torch.float32

def _apply_memory_saving(pipe: AnimateDiffPipeline, device: torch.device) -> None:
    for fn_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(pipe, fn_name):
            getattr(pipe, fn_name)()
    # CPU offload is correctly commented out as it requires 'accelerate'
    pass

def _maybe_swap_vae(pipe: AnimateDiffPipeline, vae_id: Optional[str], device: torch.device, dtype: torch.dtype) -> AnimateDiffPipeline:
    if not vae_id: return pipe
    try:
        logger.info(f"[vae] Swapping VAE -> {vae_id}")
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
        pipe.vae = vae.to(device)
    except Exception as e:
        logger.warning(f"[vae] VAE swap failed ({vae_id}): {e}")
    return pipe

def _looks_like_noise(im: Image.Image) -> bool:
    g = im.convert("L")
    pixels = list(g.getdata())
    if not pixels: return True
    from math import log2
    entropy = 0.0
    hist = g.histogram()
    total = sum(hist) or 1
    for count in hist:
        if count > 0:
            p = count / total
            entropy -= p * log2(p)
    sigma = pstdev(pixels) if len(pixels) > 1 else 0
    logger.info(f"[quality] entropy={entropy:.2f} sigma={sigma:.2f}")
    return (entropy > 7.8 or sigma > 85.0)

# =====================================================================
# === FINAL PRODUCTION-GRADE PIPELINE BUILDER =========================
# =====================================================================

def build_animdiff_pipeline(*,
                            base_model: str,
                            ip_adapter_image_path: Optional[str],
                            mode: str = "auto",
                            num_steps: int,
                            device: torch.device,
                            dtype: torch.dtype,
                            lightning_repo: str = "ByteDance/AnimateDiff-Lightning",
                            motion_repo: str = "guoyww/animatediff-motion-adapter-v1-5-2",
                            vae_id: Optional[str] = None) -> AnimateDiffPipeline:
   
    if mode == "auto":
        mode = "lightning" if num_steps <= 8 else "classic"
   
    logger.info(f"[pipeline_builder] ✅ Mode selected: {mode} (num_steps={num_steps})")
    if mode == "lightning":
        valid_steps = [1, 2, 4, 8]
        if num_steps not in valid_steps:
            logger.warning(f"[lightning] num_steps={num_steps} invalid for Lightning. Forcing to 4.")
            num_steps = 4
         
        lightning_file = f"animatediff_lightning_{num_steps}step_diffusers.safetensors"
        logger.info(f"[lightning] Building UNet config from base model '{base_model}'")
        unet_config = UNet2DConditionModel.load_config(base_model, subfolder="unet")
         
        logger.info("[lightning] Creating MotionAdapter structure from UNet config")
        motion_adapter = MotionAdapter.from_config(unet_config, torch_dtype=dtype)
         
        logger.info(f"[lightning] Downloading and loading weights from '{lightning_repo}/{lightning_file}'")
        lightning_path = hf_hub_download(repo_id=lightning_repo, filename=lightning_file)
         
        motion_adapter.load_state_dict(load_file(lightning_path, device="cpu"), strict=False)

        pipe = AnimateDiffPipeline.from_pretrained(
            base_model, motion_adapter=motion_adapter, torch_dtype=dtype
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="leading", beta_schedule="linear", prediction_type="epsilon"
        )
            
        if ip_adapter_image_path:
            logger.info("[ip_adapter] IP-Adapter is ENABLED. Loading components...")
            try:
                pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter_sd15.bin"
                )
                ip_adapter_scale = args.ip_adapter_scale  # Sensible default
                pipe.set_ip_adapter_scale(ip_adapter_scale)
                logger.info("[ip_adapter] IP-Adapter loaded successfully. Scale set to %f.", ip_adapter_scale)


            except Exception as e:
                logger.error("[ip_adapter] FAILED to load IP-Adapter. It will be disabled. Error: %s", e, exc_info=True)
                # This ensures the pipeline doesn't crash, it just won't use the IP-Adapter
                ip_adapter_image_path = None 
        else:
            logger.info("[ip_adapter] IP-Adapter is DISABLED (no --ip-adapter-image-path provided).")


    else: # Classic Mode
        logger.info(f"[classic] Loading full motion adapter from {motion_repo}")
        motion_adapter = MotionAdapter.from_pretrained(
            motion_repo, weight_name="motion_module.safetensors", torch_dtype=dtype
        )
        pipe = AnimateDiffPipeline.from_pretrained(
            base_model, motion_adapter=motion_adapter, torch_dtype=dtype
        )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="leading", beta_schedule="linear", prediction_type="epsilon"
        )
    pipe.to(device)
    pipe = _maybe_swap_vae(pipe, vae_id=vae_id, device=device, dtype=dtype)
    _apply_memory_saving(pipe, device)
    for p in pipe.motion_adapter.parameters():
        p.requires_grad = False
   
    logger.info(f"[{mode}] ✅ Pipeline ready.")
    return pipe

# =============== GENERATION & SAVING ===============

def _run_inference(pipe: AnimateDiffPipeline, *, 
                   prompt: str, 
                   negative_prompt: str, 
                   frames: int, 
                   steps: int, 
                   guidance: float, 
                   width: int, 
                   height: int, 
                   ip_adapter_image_path: Optional[str],
                   seed: int) -> List[Image.Image]:
    
    # --- Load the IP Adapter reference image ---
    ip_adapter_ref_image = None
    if ip_adapter_image_path:
        try:
            logger.info("Loading IP-Adapter reference image from: %s", ip_adapter_image_path)
            ip_adapter_ref_image = Image.open(ip_adapter_image_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to load IP-Adapter image. IP-Adapter will be inactive for this run. Error: %s", e)
            # Ensure the variable is None if loading fails, so we fall back gracefully
            ip_adapter_ref_image = None

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # --- NEW, ROBUST PIPELINE CALL ---
    # 1. Create a dictionary of arguments that are always present.
    pipeline_kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": frames,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height,
        "generator": generator,
    }

    # 2. Only add the ip_adapter_image argument if the image was successfully loaded.
    #    This prevents us from ever passing 'ip_adapter_image=None' to the pipeline.
    if ip_adapter_ref_image is not None:
        pipeline_kwargs["ip_adapter_image"] = ip_adapter_ref_image

    # 3. Call the pipeline by unpacking the keyword arguments dictionary.
    output = pipe(**pipeline_kwargs).frames[0]
    # --- END OF ROBUST CALL ---

    return output

def _save_frames(frames: List[Image.Image], output_dir: Path, preview: bool) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    if preview and frames:
        path = output_dir / "preview_000.png"
        frames[0].save(path)
        paths.append(str(path))
        return paths
   
    for i, frame in enumerate(frames):
        path = output_dir / f"frame_{i:04d}.png"
        frame.save(path)
        paths.append(str(path))
    return paths

def generate_frames(cfg: HeavyEngineConfig, *, prompt: str, negative_prompt: str, output_dir: Path, base_model: str, num_frames: int, num_steps: int, guidance_scale: float, ip_adapter_scale: float,width: int, height: int, seed: int, ip_adapter_image_path: Optional[str],preview: bool, prefer_cuda_fp16: bool, fallback_sd15: bool) -> List[str]:
    device, dtype = _select_device_and_dtype(prefer_cuda_fp16=prefer_cuda_fp16)
   
    eff_frames = 1 if preview else num_frames
   
    final_negative_prompt = f"{cfg.BASE_NEGATIVE_PROMPT}, {negative_prompt}" if negative_prompt else cfg.BASE_NEGATIVE_PROMPT
   
    pipe = build_animdiff_pipeline(
        base_model=base_model,
        ip_adapter_image_path=ip_adapter_image_path,
        num_steps=num_steps,

        device=device,
        dtype=dtype,
        vae_id=getattr(cfg, "VAE_MODEL", None)
    )
   
    logger.info(f"Generating frames: frames={eff_frames}, steps={num_steps}, guidance={guidance_scale}, size={width}x{height}")
   
    frames = _run_inference(pipe=pipe, prompt=prompt, negative_prompt=final_negative_prompt, frames=eff_frames, steps=num_steps, guidance=guidance_scale, width=width, height=height,ip_adapter_image_path=ip_adapter_image_path, seed=seed)
   
    if preview and fallback_sd15 and frames and _looks_like_noise(frames[0]):
        logger.warning("[quality] Preview is noise. Falling back to base_model='runwayml/stable-diffusion-v1-5'.")
        pipe_sd15 = build_animdiff_pipeline(
            base_model="runwayml/stable-diffusion-v1-5",
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            vae_id=getattr(cfg, "VAE_MODEL", None)
        )
        frames = _run_inference(pipe=pipe_sd15, prompt=prompt, negative_prompt=final_negative_prompt, frames=eff_frames, steps=num_steps, guidance=guidance_scale, width=width, height=height, seed=seed)
      
    return _save_frames(frames, output_dir, preview=preview)

# =============== CLI & MAIN EXECUTION ===============

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AnimateDiff Engine with Automatic Pipeline Selection")
    p.add_argument("--prompt", required=True, help="Text prompt.")
    p.add_argument("--negative-prompt", type=str, default="", help="The negative prompt to guide the AI away from unwanted results.")
    p.add_argument("--output-dir", required=True, help="Directory to save frames.")
    p.add_argument("--base-model", default="Lykon/dreamshaper-8", help="SD1.5-compatible style model.")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=1.5)
    p.add_argument(
    "--ip-adapter-scale", 
    type=float, 
    default=0.5, 
    help="Controls the influence strength of the IP-Adapter image."
)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview", action="store_true", help="Generate a single preview frame.")
    p.add_argument("--no-cuda-fp16", action="store_true", help="Force float32 on CUDA.")
    p.add_argument("--fallback-sd15", action="store_true", help="If preview is noise, retry with SD1.5 base.")
    p.add_argument(
    "--ip-adapter-image-path",
    type=str,
    default=None, # It's optional. If not provided, IP-Adapter won't be used for this run.
    help="Path to the reference image for the IP-Adapter to maintain continuity."
)
    
    p.add_argument(
    "--foundation-embedding-path",
    type=str,
    default=None,
    help="Optional: Path to a saved foundation embedding (.pt) for style consistency."
)

# -
    return p

def main() -> int:
    args = _build_parser().parse_args()
    try:
        cfg = HeavyEngineConfig()
        paths = generate_frames(
            cfg=cfg,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            output_dir=Path(args.output_dir),
            base_model=args.base_model,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            ip_adapter_scale=args.ip_adapter_scale,
            width=args.width,
            height=args.height,
            seed=args.seed,
            ip_adapter_image_path=args.ip_adapter_image_path,
            preview=args.preview,
            prefer_cuda_fp16=(not args.no_cuda_fp16),
            fallback_sd15=bool(args.fallback_sd15)
        )
        result = {"status": "COMPLETED", "mode": "preview" if args.preview else "full", "frame_paths": paths}
        if args.preview and paths:
            result["preview_path"] = paths[0]
        print(json.dumps(result))
        return 0
    except Exception as e:
        logger.error(f"An error occurred in main execution: {e}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"{type(e).__name__}: {e}"}))
        return 1

if __name__ == "__main__":
    sys.exit(main())