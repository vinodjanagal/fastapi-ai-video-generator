
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Go 3 levels up from video_engines
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from diffusers import (
    AnimateDiffPipeline,
    MotionAdapter,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)

# ======================== LOGGING SETUP ========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [animate_diff_engine] - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# ======================== UTILITIES ============================
def _select_device_and_dtype(prefer_cuda_fp16: bool = True) -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), (torch.float16 if prefer_cuda_fp16 else torch.float32)
    return torch.device("cpu"), torch.float32

def _apply_memory_saving(pipe: AnimateDiffPipeline) -> None:
    """Enable all memory optimizations for CPU-friendly runs."""
    for fn_name in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(pipe, fn_name):
            getattr(pipe, fn_name)()
    pass

def _maybe_swap_vae(pipe, vae_id: str | None, device, dtype):
    if not vae_id:
        logger.info("[vae] No custom VAE specified; using pipeline default.")
        return pipe
    try:
        logger.info(f"[vae] Attempting to swap VAE -> {vae_id}")
        try:
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
            logger.info(f"[vae] Loaded VAE from repo: {vae_id}")
        except Exception:
            logger.info(f"[vae] Repo load failed; trying single-file...")
            for fname in ["sd-vae-ft-mse-original.safetensors", "diffusion_pytorch_model.safetensors"]:
                try:
                    path = hf_hub_download(repo_id=vae_id.split("/")[0], filename=fname)
                    vae = AutoencoderKL.from_single_file(path, torch_dtype=dtype)
                    logger.info(f"[vae] Loaded VAE from single file: {path}")
                    break
                except:
                    continue
            else:
                raise RuntimeError("VAE load failed")
        pipe.vae = vae.to(device)
        logger.info(f"[vae] Successfully swapped VAE -> {vae_id}")
    except Exception as e:
        logger.error(f"[vae] VAE swap FAILED: {e}. Using default.", exc_info=True)
    return pipe

# =====================================================================
# === FINAL PRODUCTION-GRADE PIPELINE BUILDER (Corrected Version) =====
# =====================================================================

def build_pipeline(args) -> AnimateDiffPipeline:
    """Builds a fully stable AnimateDiff pipeline with configurable, high-quality components."""
    device, dtype = _select_device_and_dtype(prefer_cuda_fp16=not args.no_cuda_fp16)
    
    logger.info(f"[pipeline_builder] Mode: {args.mode}, Steps: {args.num_steps}, Base: {args.base_model}")

    # --- Step 1: Build base pipeline ---
    if args.mode == "lightning":
        valid_steps = [1, 2, 4, 8]
        num_steps = args.num_steps
        if num_steps not in valid_steps:
            logger.warning(f"[lightning] num_steps={num_steps} invalid. Forcing to 4.")
            num_steps = 4
        lightning_repo = "ByteDance/AnimateDiff-Lightning"
        lightning_file = f"animatediff_lightning_{num_steps}step_diffusers.safensors"
        logger.info(f"[lightning] Loading UNet and motion weights...")
        unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", torch_dtype=dtype)
        motion_adapter_path = hf_hub_download(repo_id=lightning_repo, filename=lightning_file)
        motion_adapter_state_dict = load_file(motion_adapter_path, device="cpu")
        motion_adapter = MotionAdapter()
        motion_adapter.load_state_dict(motion_adapter_state_dict)
        pipe = AnimateDiffPipeline.from_pretrained(
            args.base_model, unet=unet, motion_adapter=motion_adapter, torch_dtype=dtype
        )
    else:  # CLASSIC MODE
        motion_repo = "guoyww/animatediff-motion-adapter-v1-5-2"
        logger.info(f"[classic] Loading motion adapter from {motion_repo}")
        motion_adapter = MotionAdapter.from_pretrained(motion_repo, torch_dtype=dtype)
        pipe = AnimateDiffPipeline.from_pretrained(
            args.base_model, motion_adapter=motion_adapter, torch_dtype=dtype
        )

    # --- Step 2: Set Scheduler with Fallback ---
    if getattr(args, "scheduler", "DDIM") == "DPM++ 2M Karras":
        try:
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
            logger.info("[scheduler] Swapped to: DPM++ 2M Karras")
        except Exception as e:
            logger.warning(f"[scheduler] Karras failed ({e}). Using DDIM.")
            from diffusers import DDIMScheduler
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        logger.info("[scheduler] Using: DDIM")

    # --- Step 3: Load IP-Adapter ---
    if args.mode == "lightning":
        logger.info("[ip_adapter] Skipped — incompatible with Lightning mode.")
    elif getattr(args, "ip_adapter_image_path", None):
        logger.info("[ip_adapter] Enabled. Loading IP-Adapter model...")
        try:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_light.bin")
            scale = getattr(args, "ip_adapter_scale", 0.05)
            pipe.set_ip_adapter_scale(scale)
            logger.info(f"[ip_adapter] Loaded successfully. Scale = {scale}")
        except Exception as e:
            logger.error(f"[ip_adapter] FAILED: {e}", exc_info=True)
    else:
        logger.info("[ip_adapter] Not provided — skipping.")

    # --- Step 4: Swap VAE ---
    vae_id = getattr(args, "vae", "stabilityai/sd-vae-ft-mse")
    pipe = _maybe_swap_vae(pipe, vae_id=vae_id, device=device, dtype=dtype)

    # --- Finalization ---
    _apply_memory_saving(pipe)
    pipe.to(device)
    logger.info(f"[{args.mode}] AnimateDiff pipeline is ready on {device}.")
    return pipe

# ======================== INFERENCE + SAVE =============================

def run_inference(pipe, args) -> List[Image.Image]:
    """Runs AnimateDiff inference with robust, multi-modal continuity handling."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    pipeline_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "generator": generator,
    }

    # --- Triple-Safe Continuity Logic ---
    has_ip_adapter = args.ip_adapter_image_path is not None and args.ip_adapter_scale > 0.0
    has_init_image = args.init_image is not None and 0.0 < args.strength < 1.0

    if has_ip_adapter:
        logger.info(f"[continuity] IP-Adapter enabled with scale: {args.ip_adapter_scale}")
        try:
            ip_img = Image.open(args.ip_adapter_image_path).convert("RGB")
            pipeline_kwargs["ip_adapter_image"] = ip_img
        except Exception as e:
            logger.error(f"[continuity] Failed to load IP-Adapter image: {e}", exc_info=True)
            return []  # prevent crash

    if has_init_image:
        logger.info(f"[continuity] Init_image (img2vid) enabled with strength: {args.strength}")
        try:
            init_img = Image.open(args.init_image).convert("RGB").resize((args.width, args.height))
            pipeline_kwargs["image"] = init_img
            pipeline_kwargs["strength"] = args.strength
        except Exception as e:
            logger.error(f"[continuity] Failed to load init_image: {e}", exc_info=True)
            return []  # prevent crash

    logger.info("Running AnimateDiff pipeline inference...")

    try:
        output = pipe(**pipeline_kwargs)
        frames = output.frames[0]
        logger.info(f"✅ Inference completed, {len(frames)} frames generated.")
        return frames
    except Exception as e:
        logger.error(f"[inference] Pipeline execution failed: {e}", exc_info=True)
        return []
    

def save_frames(frames: List[Image.Image], output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = output_dir / f"frame_{i:04d}.png"
        frame.save(path)
        paths.append(str(path))
    return paths


# ======================== CLI =============================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AnimateDiff Engine (V2.4 Production-Locked)")
    
    # Core I/O
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--output-dir", required=True)
    
    # Models
    p.add_argument("--base-model", default="SG161222/Realistic_Vision_V5.1_noVAE")
    p.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model ID.")
    
    # Generation Params
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument("--scheduler", type=str, default="DPM++ 2M Karras", choices=["DPM++ 2M Karras", "DDIM"])
    
    # Continuity Params
    p.add_argument("--ip-adapter-image-path", type=str, default=None)
    p.add_argument("--ip-adapter-scale", type=float, default=0.0)
    p.add_argument("--init-image", type=str, default=None)
    p.add_argument("--strength", type=float, default=1.0)
    
    # Technical Params
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="classic", choices=["lightning", "classic"])
    p.add_argument("--no-cuda-fp16", action="store_true")
    
    return p

def main() -> int:
    args = build_parser().parse_args()
    try:
        pipe = build_pipeline(args)
        frames = run_inference(pipe, args)
        paths = save_frames(frames, Path(args.output_dir))
        print(json.dumps({"status": "COMPLETED", "frame_paths": paths}))
        return 0
    except Exception as e:
        logger.error(f"AnimateDiff failed: {e}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
