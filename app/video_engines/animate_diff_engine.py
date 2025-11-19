from __future__ import annotations

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from PIL import Image
import numpy as np

# CPU thread safety (deterministic behavior on constrained hardware)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [animate_diff] - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("animate_diff_engine")


# ------------------------
# Helpers
# ------------------------
def safe_open_image(path: Optional[str], width: int, height: int) -> Optional[Image.Image]:
    """Open and resize init image if path exists. Return PIL.Image or None."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"[io] init-image path does not exist: {path}")
        return None
    try:
        img = Image.open(p).convert("RGB").resize((int(width), int(height)), Image.LANCZOS)
        return img
    except Exception as e:
        logger.warning(f"[io] failed to load/resize image {path}: {e}")
        return None


def _to_uint8_array(arr: np.ndarray) -> np.ndarray:
    """Convert float or other dtypes -> uint8 HWC array safely."""
    if arr.dtype == np.uint8:
        return arr
    try:
        if np.nanmax(arr) <= 1.0:
            out = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            out = np.clip(arr, 0, 255).astype(np.uint8)
        return out
    except Exception:
        return arr.astype(np.uint8, errors="ignore")


def extract_frames(output) -> List[Image.Image]:
    """
    Robust frame extractor accepting multiple AnimateDiff output shapes:
      - output.frames (list) -> flatten
      - output == list-of-lists           -> flatten
      - output contains torch.Tensor / np.ndarray batches -> convert each frame
    Returns list of PIL.Image (RGB).
    Raises RuntimeError if no frames extracted.
    """
    raw = getattr(output, "frames", None)

    # Fallback: pipeline sometimes returns list or list-of-lists directly
    if raw is None:
        if isinstance(output, (list, tuple)):
            # Example shapes seen: [ [PIL,PIL,...] ] or [PIL, PIL, ...] or [np.ndarray] etc.
            raw = output
        else:
            raise RuntimeError("AnimateDiff produced no frames (no output.frames).")

    flattened: List[object] = []
    # Flatten one level if outer is singleton batch list, but ensure we don't over-flatten weird arrays
    if isinstance(raw, (list, tuple)) and len(raw) == 1 and isinstance(raw[0], (list, tuple)):
        # common case: [ [frame1, frame2, ...] ]
        flattened = list(raw[0])
    else:
        # could be [frame1, frame2, ...] or np arrays, tensors
        flattened = list(raw)

    result: List[Image.Image] = []
    for item in flattened:
        # Already a PIL image
        if isinstance(item, Image.Image):
            try:
                result.append(item.convert("RGB"))
            except Exception:
                # fallback: create from numpy below
                arr = np.array(item)
                result.append(Image.fromarray(_to_uint8_array(arr)).convert("RGB"))
            continue

        # Torch tensor: (C,H,W) or (B,H,W,C) or (B,C,H,W) or (H,W,C)
        if isinstance(item, torch.Tensor):
            try:
                arr = item.detach().cpu().numpy()
            except Exception:
                arr = np.array(item)
        else:
            # numpy array or other (e.g., PIL-like object convertible)
            arr = np.array(item)

        # Squeeze trivial dims
        try:
            arr = np.squeeze(arr)
        except Exception:
            pass

        # Cases:
        # - arr.ndim == 4  -> (B,H,W,C) or (B,C,H,W) or (B, H, W, 3)
        # - arr.ndim == 3  -> (H,W,C) or (C,H,W)
        # - arr.ndim == 2  -> gray -> expand channel
        if arr.ndim == 4:
            # Interpret as batch: iterate frames
            # Try formats: (B, H, W, C) or (B, C, H, W)
            b = arr.shape[0]
            # naive attempt to detect channel pos
            if arr.shape[-1] in (1, 3, 4):
                for i in range(b):
                    sub = arr[i]
                    sub = _to_uint8_array(np.squeeze(sub))
                    if sub.ndim == 2:
                        sub = np.expand_dims(sub, -1)
                    if sub.ndim == 3 and sub.shape[0] in (1, 3) and sub.shape[-1] not in (1, 3):
                        # CHW -> HWC
                        sub = np.transpose(sub, (1, 2, 0))
                    if sub.ndim == 3 and sub.shape[-1] == 1:
                        sub = np.repeat(sub, 3, axis=-1)
                    try:
                        result.append(Image.fromarray(sub.astype(np.uint8)).convert("RGB"))
                    except Exception:
                        continue
                continue
            else:
                # assume (B, C, H, W)
                for i in range(b):
                    sub = arr[i]
                    # CHW -> HWC
                    if sub.ndim == 3 and sub.shape[0] in (1, 3):
                        sub = np.transpose(sub, (1, 2, 0))
                    sub = _to_uint8_array(np.squeeze(sub))
                    if sub.ndim == 2:
                        sub = np.expand_dims(sub, -1)
                    if sub.ndim == 3 and sub.shape[-1] == 1:
                        sub = np.repeat(sub, 3, axis=-1)
                    try:
                        result.append(Image.fromarray(sub.astype(np.uint8)).convert("RGB"))
                    except Exception:
                        continue
                continue

        # If arr is 3D, check for CHW vs HWC
        if arr.ndim == 3:
            # If channels first (C,H,W)
            if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            # If single-channel last
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            arr = _to_uint8_array(np.squeeze(arr))
            try:
                result.append(Image.fromarray(arr.astype(np.uint8)).convert("RGB"))
                continue
            except Exception:
                # fallthrough and skip
                pass

        # If arr is 2D (grayscale)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, -1)
            arr = np.repeat(arr, 3, axis=-1)
            arr = _to_uint8_array(arr)
            try:
                result.append(Image.fromarray(arr.astype(np.uint8)).convert("RGB"))
                continue
            except Exception:
                pass

        logger.warning(f"[extract] skipping unsupported array shape: {getattr(arr, 'shape', type(arr))}")

    if not result:
        raise RuntimeError("No valid frames extracted from pipeline output.")

    return result


# ------------------------
# Pipeline builder
# ------------------------
def build_pipeline(cfg: Dict[str, Any]):
    """
    Build AnimateDiff pipeline with best-effort: attempt to load MotionAdapter separately,
    then create AnimateDiffPipeline either with adapter or without (in case of unified model).
    """
    from diffusers import AnimateDiffPipeline, MotionAdapter, DPMSolverMultistepScheduler, AutoencoderKL

    device = torch.device("cpu")
    dtype = torch.float32

    base_model = cfg.get("base_model", "SG161222/Realistic_Vision_V5.1_noVAE")
    motion_adapter_model = cfg.get("motion_adapter", "guoyww/animatediff-motion-adapter-v1-5-2")
    vae_model = cfg.get("vae_model", "stabilityai/sd-vae-ft-mse")

    logger.info(f"[pipeline] loading base model: {base_model}")
    logger.info(f"[pipeline] motion adapter hint: {motion_adapter_model}")

    motion_adapter = None
    try:
        motion_adapter = MotionAdapter.from_pretrained(motion_adapter_model, torch_dtype=dtype)
    except Exception as e:
        logger.warning(f"[pipeline] MotionAdapter.from_pretrained failed: {e} — continuing (adapter may be embedded in base).")
        motion_adapter = None

    try:
        if motion_adapter is not None:
            pipe = AnimateDiffPipeline.from_pretrained(base_model, motion_adapter=motion_adapter, torch_dtype=dtype)
        else:
            pipe = AnimateDiffPipeline.from_pretrained(base_model, torch_dtype=dtype)
    except Exception as e:
        logger.error(f"[pipeline] AnimateDiffPipeline.from_pretrained failed: {e}", exc_info=True)
        raise

    # Scheduler: prefer DPMSolverMultistepScheduler (dpmsolver++) with Karras sigmas
    try:
        logger.info("[scheduler] attempting DPMSolverMultistepScheduler (dpmsolver++)")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        logger.info("[scheduler] DPMSolver++ with Karras sigmas enabled.")
    except Exception as e:
        logger.warning(f"[scheduler] DPMSolver optimized setup failed: {e} — using pipeline default scheduler.")

    # VAE swap (optional)
    try:
        vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)
        pipe.vae = vae.to(device)
        logger.info("[vae] swapped OK")
    except Exception as e:
        logger.warning(f"[vae] failed to load custom VAE: {e} — continuing with pipeline VAE")

    pipe.to(device)

    # CPU-friendly: enable attention slicing if available
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # Quick diagnostic: verify motion modules on UNET (best-effort)
    try:
        unet = getattr(pipe, "unet", None)
        if unet is not None:
            motion_modules = [n for n, _ in unet.named_modules() if "motion" in n or "adapter" in n]
            if not any("motion" in n for n in motion_modules):
                logger.warning("[pipeline] UNET does not show expected motion modules — motion adapter may be inactive.")
    except Exception:
        pass

    logger.info("[pipeline] ready on CPU")
    return pipe


# ------------------------
# Inference runner
# ------------------------
def run_inference(pipe, job: Dict[str, Any]) -> List[Image.Image]:
    """
    Run pipeline with explicit kwargs and return a list of PIL.Image frames.
    This function logs a warning if the pipeline returned 1 frame when more were requested.
    """
    prompt = str(job.get("prompt", ""))
    negative_prompt = str(job.get("negative_prompt", "")) if job.get("negative_prompt") else None
    width = int(job.get("width", 384))
    height = int(job.get("height", 384))
    steps = int(job.get("num_steps", 15))
    n_frames = int(job.get("num_frames", 16))
    guidance_scale = float(job.get("guidance_scale", 6.0))
    seed = int(job.get("seed", 42))
    init_image_path = job.get("init_image", None)
    strength = job.get("strength", None)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    logger.info(f"[inference] running... frames={n_frames} steps={steps} size={width}x{height}")

    init_img = safe_open_image(init_image_path, width, height) if init_image_path else None

    call_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
        "num_frames": n_frames,
    }
    if negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt
    if init_img is not None and strength is not None:
        call_kwargs["image"] = init_img
        call_kwargs["strength"] = float(strength)

    logger.info(f"[inference] calling pipeline (kwargs keys: {list(call_kwargs.keys())})")

    try:
        output = pipe(**call_kwargs)
    except Exception as e:
        logger.error(f"[inference] pipeline call failed: {e}", exc_info=True)
        raise

    frames = extract_frames(output)

    # Diagnostic: if pipeline returned 1 frame but user requested >1, motion adapter may be inactive
    if len(frames) == 1 and n_frames > 1:
        logger.warning(f"[inference] Pipeline returned 1 frame (requested {n_frames}). MotionAdapter may be inactive or incompatible; check model+adapter compatibility.")

    logger.info(f"[inference] {len(frames)} frames extracted")
    return frames


# ------------------------
# CLI
# ------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="animate_diff_engine.py", description="AnimateDiff worker (hybrid-cleaning).")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="SG161222/Realistic_Vision_V5.1_noVAE")
    p.add_argument("--motion-adapter", default="guoyww/animatediff-motion-adapter-v1-5-2")
    p.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=15)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--init-image", default=None)
    p.add_argument("--strength", type=float, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    job: Dict[str, Any] = vars(args)
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "base_model": job.get("base_model"),
        "motion_adapter": job.get("motion_adapter"),
        "vae_model": job.get("vae_model"),
    }

    try:
        pipe = build_pipeline(cfg)
        pil_frames = run_inference(pipe, job)

        saved_paths: List[str] = []
        for i, fr in enumerate(pil_frames):
            fp = out_dir / f"frame_{i:04d}.png"
            fr.save(fp)
            saved_paths.append(str(fp))
            logger.info(f"[io] saved {fp}")

        print(json.dumps({"status": "COMPLETED", "frame_paths": saved_paths}))
        sys.exit(0)
    except Exception as e:
        logger.error(f"[engine] FAILED: {e}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": str(e)}))
        sys.exit(1)
