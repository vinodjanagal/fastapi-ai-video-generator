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
from app.engine.cinematics import compute_runtime_continuity

from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler, AutoencoderKL

# Modular imports (v6.0)
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.engine.parser import semantic_parser, build_semantic_prompt
from app.engine.cinematics import classify_shot_type, apply_param_tuning, apply_safety_guards

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s", stream=sys.stderr)
logger = logging.getLogger("v6_orchestrator")


# ---------- Utilities ----------
def _maybe_swap_vae(pipe, vae_id: Optional[str], device: torch.device, dtype: torch.dtype):
    """
    Robust VAE loader:
      - log intent even if vae_id is falsy
      - try subfolder='vae' first, then fallback
      - force torch.float32 on CPU to avoid silent precision issues
    """
    logger.info(f"[vae] Requested VAE id: {vae_id}")
    if not vae_id:
        logger.info("[vae] No VAE specified — using pipeline default.")
        return pipe

    try:
        vae_dtype = torch.float32 if device.type == "cpu" else dtype
        try:
            logger.info(f"[vae] Attempting to load VAE (subfolder='vae') with dtype={vae_dtype}")
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype, subfolder="vae")
        except Exception:
            logger.info(f"[vae] Subfolder attempt failed; trying default from_pretrained for {vae_id}")
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=vae_dtype)
        pipe.vae = vae.to(device)
        logger.info(f"[vae] ✅ VAE swapped successfully: {vae_id}")
    except Exception as e:
        logger.error(f"[vae] CRITICAL: VAE loading failed for {vae_id}: {e}", exc_info=True)
        raise
    return pipe


def _load_ip_adapter_safe(pipe, repo: str) -> bool:
    """
    Try multiple common ip-adapter locations / weight files.
    Returns True on success, False otherwise.
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
            logger.info(f"[ip-adapter] ✅ loaded: {repo}/{sub}/{weight}")
            return True
        except Exception as e:
            last_exc = e
            continue
    logger.error(f"[ip-adapter] All load attempts failed for repo '{repo}'. Last error: {last_exc}")
    return False


def _safe_open_and_resize(path: Optional[str], width: int, height: int) -> Optional[Image.Image]:
    """
    Open image if exists, convert to RGB, and resize using LANCZOS.
    Returns None if path is falsy or file missing.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"[io] Image not found at path: {path}. Skipping.")
        return None
    try:
        img = Image.open(p).convert("RGB").resize((width, height), Image.LANCZOS)
        return img
    except Exception as e:
        logger.warning(f"[io] Failed to open/resize image {path}: {e}", exc_info=True)
        return None


def _extract_frames_from_output(out) -> List[Image.Image]:
    """
    Universal extractor: handle multiple pipeline output shapes safely.
    Expected: out.frames -> list[list[PIL.Image]] or list[PIL.Image]
    """
    frames_container = getattr(out, "frames", None)
    if frames_container is None:
        # Some pipeline versions return the images directly
        if isinstance(out, (list, tuple)):
            maybe_imgs = out[0] if out else []
            if isinstance(maybe_imgs, list):
                return maybe_imgs
            # otherwise wrap
            return list(out)
        raise RuntimeError("Pipeline returned no 'frames' attribute and not a list/tuple.")
    # If frames_container is nested list, pick first list
    if isinstance(frames_container, list) and frames_container and isinstance(frames_container[0], list):
        return frames_container[0]
    # If frames_container is a single list of images
    if isinstance(frames_container, list):
        return frames_container
    # Fallback: try to iterate
    try:
        return list(frames_container)
    except Exception:
        raise RuntimeError("Unable to extract frames from pipeline output.")


# ---------- Pipeline build & inference ----------
def build_pipeline(args, device, dtype):
    logger.info(f"[pipeline] Loading pipeline for base model: {args.base_model}")
    motion_adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(args.base_model, motion_adapter=motion_adapter, torch_dtype=dtype)

    # set scheduler to a stable Karras variant
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)

    # Try swapping the VAE
    pipe = _maybe_swap_vae(pipe, args.vae_model, device, dtype)

    # Load IP-Adapter if requested
    if getattr(args, "ip_adapter_scale", 0.0) and getattr(args, "ip_adapter_image_path", None):
        if _load_ip_adapter_safe(pipe, getattr(args, "ip_adapter_repo", "h94/IP-Adapter")):
            pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    # Enable memory-friendly options for CPU runs
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        # older/newer diffusers variants may not have these APIs
        logger.debug("[pipeline] attention/vae slicing not available for this pipeline version.")

    pipe.to(device)
    return pipe


def run_inference(pipe, args, device):
    # Deterministic seeding for reproducible outputs on CPU
    torch.manual_seed(args.seed)
    np_seed = int(args.seed) & 0xFFFFFFFF
    np.random.seed(np_seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    gen = torch.Generator(device=device).manual_seed(args.seed)

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


# ---------- CLI & main ----------
def build_parser():
    p = argparse.ArgumentParser(description="AnimateDiff V7.1 Modular Semantic Engine")
    p.add_argument("--prompt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--init-image", type=str, default=None)
    p.add_argument("--ip-adapter-image-path", type=str, default=None)
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
    p.add_argument("--negative-prompt", type=str, default="")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        device = torch.device("cpu")
        dtype = torch.float32

        # Semantic decomposition
        semantic_parts = semantic_parser(args.prompt)
        shot_type = classify_shot_type(args.prompt, semantic_parts)

        # Build final prompts (merges user negative prompt)
        args.prompt, args.negative_prompt = build_semantic_prompt(args.prompt, args.negative_prompt, shot_type, semantic_parts)

        # Apply default parameter tuning and safety guards
        apply_param_tuning(args, shot_type)
        apply_safety_guards(args)

        # Build pipeline and run inference
        pipe = build_pipeline(args, device, dtype)
        frames = run_inference(pipe, args, device)

        # Save frames and preview (robust save)
        if frames:
            preview = output_dir / "preview_frame_0000.png"
            try:
                frames[0].save(preview)
                logger.info(f"[preview] Preview saved to: {preview}")
            except Exception as e:
                logger.warning(f"[io] Failed to save preview: {e}", exc_info=True)

            paths = []
            for i, frame in enumerate(frames):
                path = output_dir / f"frame_{i:04d}.png"
                try:
                    frame.save(path)
                except Exception as e:
                    logger.warning(f"[io] Failed to save frame {i}: {e}", exc_info=True)
                    continue
                paths.append(str(path))

            print(json.dumps({"status": "COMPLETED", "frame_paths": paths}))
        # cleanup
        try:
            del pipe, frames
        except Exception:
            pass
        gc.collect()
        sys.exit(0)

    except Exception as e:
        logger.error(f"V7.1 Engine failed: {e}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"{type(e).__name__}: {e}"}))
        sys.exit(1)