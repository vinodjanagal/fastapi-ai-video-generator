# app/video_engines/animate_diff_engine.py
"""
Final, stateless AnimateDiff engine (DUMB WORKER).
Receives final_positive_prompt and final_negative_prompt from orchestrator and renders.
No "brain" imports. No prompt-building logic here.
"""

import sys
import json
import gc
import argparse
import os
import random
from pathlib import Path
from typing import Optional, List
from PIL import Image
import logging
import numpy as np
import torch

from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler, AutoencoderKL

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [animate_diff] - %(message)s", stream=sys.stderr)
logger = logging.getLogger("animate_diff_engine")


# -------------------------
# Helpers (stateless)
# -------------------------
def _maybe_swap_vae(pipe, vae_id: Optional[str], device: torch.device, dtype: torch.dtype):
    """Attempt to load and swap VAE safely. If vae_id is falsy, use pipeline default."""
    logger.info(f"[vae] Requested VAE id: {vae_id}")
    if not vae_id:
        logger.info("[vae] No VAE override provided; using pipeline default VAE.")
        return pipe

    try:
        vae_dtype = torch.float32 if device.type == "cpu" else dtype
        try:
            logger.info("[vae] Trying to load VAE from subfolder='vae' ...")
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype, subfolder="vae")
        except Exception:
            logger.info("[vae] Subfolder attempt failed; trying root from_pretrained ...")
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype)
        pipe.vae = vae.to(device)
        logger.info(f"[vae] VAE swapped successfully: {vae_id}")
        return pipe
    except Exception as e:
        logger.error(f"[vae] Failed to load VAE '{vae_id}': {e}", exc_info=True)
        raise


def _load_ip_adapter_safe(pipe, repo: str) -> bool:
    """
    Attempt multiple common candidate names/locations for ip-adapter weights.
    Returns True if loaded, False otherwise.
    """
    candidates = [
        ("models", "ip-adapter_sd15_light.bin"),
        ("models", "ip-adapter_sd15.bin"),
        ("", "ip-adapter_sd15_light.bin"),
        ("", "ip-adapter_sd15.bin"),
    ]
    last_exc = None
    for sub, weight in candidates:
        try:
            pipe.load_ip_adapter(repo, subfolder=sub, weight_name=weight)
            logger.info(f"[ip-adapter] Loaded: {repo}/{sub}/{weight}")
            return True
        except Exception as e:
            last_exc = e
            continue
    logger.error(f"[ip-adapter] All attempts failed for repo '{repo}'. Last error: {last_exc}")
    return False


def _safe_open_and_resize(path: Optional[str], width: int, height: int) -> Optional[Image.Image]:
    """Open image if exists, convert to RGB, resize with LANCZOS. Return PIL image or None."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"[io] Image not found at path: {path}. Skipping IP/Init image.")
        return None
    try:
        img = Image.open(p).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        return img
    except Exception as e:
        logger.warning(f"[io] Failed to open/resize image {path}: {e}", exc_info=True)
        return None


def _extract_frames_from_output(out) -> List[Image.Image]:
    """
    Extract list of PIL.Image frames from AnimateDiff pipeline output.
    Handles multiple possible return shapes.
    """
    frames_container = getattr(out, "frames", None)
    if frames_container is None:
        # Some pipeline versions return list/tuple directly
        if isinstance(out, (list, tuple)):
            candidate = out[0] if out else []
            if isinstance(candidate, list):
                return candidate
            return list(out)
        raise RuntimeError("Pipeline returned no 'frames' attribute and not a list/tuple.")
    # If nested list: [[imgs]]
    if isinstance(frames_container, list) and frames_container and isinstance(frames_container[0], list):
        return frames_container[0]
    if isinstance(frames_container, list):
        return frames_container
    try:
        return list(frames_container)
    except Exception:
        raise RuntimeError("Unable to extract frames from pipeline output.")


# -------------------------
# Pipeline build & inference
# -------------------------
def build_pipeline(args, device, dtype):
    """Build AnimateDiff pipeline with MotionAdapter, optional VAE and optional IP-adapter."""
    logger.info(f"[pipeline] Loading pipeline for base model: {args.base_model}")
    # load motion adapter + base model
    motion_adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(args.base_model, motion_adapter=motion_adapter, torch_dtype=dtype)

    # use Karras-enabled DPMSolver variant for improved stability
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    except Exception:
        logger.debug("[pipeline] Could not switch scheduler to DPMSolverMultistepScheduler (version mismatch).")

    # VAE swap if requested
    pipe = _maybe_swap_vae(pipe, args.vae_model, device, dtype)

    # IP adapter load if requested (and set scale)
    if getattr(args, "ip_adapter_scale", None) and getattr(args, "ip_adapter_image_path", None):
        if _load_ip_adapter_safe(pipe, getattr(args, "ip_adapter_repo", "h94/IP-Adapter")):
            try:
                pipe.set_ip_adapter_scale(args.ip_adapter_scale)
            except Exception:
                logger.warning("[ip-adapter] set_ip_adapter_scale not available on this pipeline version.")

    # attempt memory-saving helpers
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        logger.debug("[pipeline] attention/vae slicing APIs not available; continuing without them.")

    pipe.to(device)
    return pipe


def run_inference(pipe, args, device):
    """Run AnimateDiff pipeline with deterministic seeding and provided args. Returns list[PIL.Image]."""
    # deterministic seeds for CPU reproducibility
    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) & 0xFFFFFFFF)
    os.environ["PYTHONHASHSEED"] = str(int(args.seed))

    gen = torch.Generator(device=device).manual_seed(int(args.seed))

    ip_img = _safe_open_and_resize(getattr(args, "ip_adapter_image_path", None), args.width, args.height)
    init_img = _safe_open_and_resize(getattr(args, "init_image", None), args.width, args.height)

    kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "width": args.width,
        "height": args.height,
        "generator": gen,
    }

    if ip_img is not None and getattr(args, "ip_adapter_scale", 0.0) and args.ip_adapter_scale > 0:
        kwargs["ip_adapter_image"] = ip_img
    if init_img is not None and getattr(args, "strength", None) is not None and 0 < args.strength < 1:
        kwargs["image"] = init_img
        kwargs["strength"] = args.strength

    logger.info("[inference] Running AnimateDiff pipeline...")
    out = pipe(**kwargs)

    frames = _extract_frames_from_output(out)
    if not frames:
        raise RuntimeError("Pipeline returned an empty frame list")
    logger.info(f"[inference] ✅ Generated {len(frames)} frames successfully.")
    return frames


# -------------------------
# CLI parser
# -------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Phoenix Engine - Stateless AnimateDiff renderer")
    p.add_argument("--prompt", required=True, help="Final positive prompt (fully formed).")
    p.add_argument("--negative-prompt", type=str, default="", help="Final negative prompt.")
    p.add_argument("--output-dir", required=True, help="Directory to save frames.")
    p.add_argument("--base-model", default="Lykon/dreamshaper-8", help="Base diffusion model id.")
    p.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse", help="Optional VAE id to swap.")
    p.add_argument("--ip-adapter-image-path", type=str, default=None, help="Path to master reference image for IP adapter.")
    p.add_argument("--ip-adapter-repo", type=str, default="h94/IP-Adapter", help="Repo id for ip-adapter weights.")
    p.add_argument("--ip-adapter-scale", type=float, default=0.0, help="IP adapter scale (0 disables).")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manual", action="store_true")
    p.add_argument("--strength", type=float, default=None, help="Init image strength (0-1).")
    p.add_argument("--guidance-scale", type=float, default=7.0)
    return p


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device = torch.device("cpu")
        dtype = torch.float32

        # Build pipeline (stateless)
        pipe = build_pipeline(args, device, dtype)

        # Run inference
        frames = run_inference(pipe, args, device)

        # Save frames robustly
        saved_paths: List[str] = []
        for i, frame in enumerate(frames):
            out_path = output_dir / f"frame_{i:04d}.png"
            try:
                frame.save(out_path)
                saved_paths.append(str(out_path))
            except Exception as e:
                logger.warning(f"[io] Failed saving frame {i}: {e}", exc_info=True)

        # Print machine-readable result
        print(json.dumps({"status": "COMPLETED", "frame_paths": saved_paths}))
        # cleanup
        try:
            del pipe, frames
        except Exception:
            pass
        gc.collect()
        sys.exit(0)

    except Exception as exc:
        logger.error(f"[engine] FAILED: {exc}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
