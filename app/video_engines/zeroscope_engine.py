# app/video_engines/zeroscope_engine.py (V19)
# Rewritten for Vinod: robust, CPU-friendly, defensive, and drop-in compatible
# - Robust model loading (local cache preferred, remote fallback)
# - CPU stability tweaks (threads, matmul precision, attention slicing)
# - Safe model offload when available
# - Very defensive frame unpacking (handles 5D/4D/torch/np/list outputs)
# - Robust saver that logs shapes and failures but never raises

import argparse
import json
import logging
import os
import sys
from typing import List, Any

import numpy as np
import torch
from diffusers import TextToVideoSDPipeline
from PIL import Image

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [zeroscope] - %(message)s')
logger = logging.getLogger("zeroscope_engine")

# ---- Global CPU constraints ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


# ---- Utilities ----
def _to_uint8_image(arr: np.ndarray) -> Image.Image:
    """Convert a numpy array (H,W,C) or (H,W) float [0..1] or uint8 into PIL Image."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        # grayscale
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    if arr.ndim == 3:
        # channel first (C,H,W)? -> move to HWC
        if arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[2]:
            try:
                arr = np.moveaxis(arr, 0, -1)
            except Exception:
                # fallback: leave as-is
                pass

        # floats
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        if arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0], mode="L")
        return Image.fromarray(arr)

    raise ValueError(f"Unsupported array to convert to image: ndim={arr.ndim} shape={arr.shape} dtype={arr.dtype}")


def save_frames(frames: Any, out_dir: str) -> List[str]:
    """Save frames from a wide variety of possible pipeline outputs.

    Accepts:
      - list of frames
      - nested lists
      - numpy array: (N,H,W,C) or (1,N,H,W,C) or (H,W,C)
      - torch.Tensor with same shapes

    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    counter = 0

    def _save_one(fobj, idx_label=""):
        nonlocal counter
        try:
            if hasattr(fobj, "cpu") and callable(getattr(fobj, "cpu")):
                try:
                    fobj = fobj.cpu().numpy()
                except Exception:
                    # leave as-is
                    pass

            # If we receive an object-dtype numpy array containing arrays
            if isinstance(fobj, np.ndarray) and fobj.dtype == object:
                # try to extract first element or stack
                if fobj.size == 1:
                    fobj = np.asarray(fobj.flat[0])
                else:
                    try:
                        fobj = np.stack(list(fobj.flat)).squeeze()
                    except Exception:
                        raise ValueError("Cannot coerce object array into image")

            if isinstance(fobj, np.ndarray):
                # if the array is 4D and looks like a batch, we should not reach here
                if getattr(fobj, "ndim", None) == 4:
                    # caller should have iterated batch; just iterate now
                    for i in range(fobj.shape[0]):
                        _save_one(fobj[i], idx_label=f"[batch_{i}]")
                    return

                img = _to_uint8_image(fobj)
            else:
                # try coercing to numpy
                arr = np.asarray(fobj)
                img = _to_uint8_image(arr)

            fname = f"frame_{counter:04d}.png"
            p = os.path.join(out_dir, fname)
            img.save(p)
            saved.append(p)
            logger.info(f"Saved {p}  {idx_label} (shape={getattr(fobj,'shape',None)}, dtype={getattr(fobj,'dtype',None)})")
            counter += 1
        except Exception as e:
            logger.error(f"Failed to save frame {counter} {idx_label}: {e}")

    # Normalize different top-level container types
    # If frames is a numpy array or torch tensor
    if isinstance(frames, torch.Tensor):
        try:
            frames = frames.cpu().numpy()
        except Exception:
            pass

    if isinstance(frames, np.ndarray):
        # If 5D: (B, T, H, W, C) -> flatten B dimension
        if frames.ndim == 5:
            logger.info(f"Detected 5D array, shape={frames.shape} -> flattening batch dimension")
            if frames.shape[0] == 1:
                frames = frames[0]
            else:
                # iterate over batch first
                for b in range(frames.shape[0]):
                    for t in range(frames.shape[1]):
                        _save_one(frames[b, t], idx_label=f"(orig_b={b},t={t})")
                return saved

        # If 4D: (T,H,W,C)
        if frames.ndim == 4:
            for t in range(frames.shape[0]):
                _save_one(frames[t], idx_label=f"(t={t})")
            return saved

        # If 3D or 2D -> single image
        _save_one(frames)
        return saved

    # If list/tuple
    if isinstance(frames, (list, tuple)):
        for idx, it in enumerate(frames):
            # If nested list-of-lists
            if isinstance(it, (list, tuple)):
                for j, sub in enumerate(it):
                    _save_one(sub, idx_label=f"(outer={idx},inner={j})")
            else:
                _save_one(it, idx_label=f"(idx={idx})")
        return saved

    # Last resort
    _save_one(frames)
    return saved


# ---- Main inference logic ----
def run_inference(args):
    # prefer local cache
    if args.model_dir and os.path.exists(args.model_dir):
        model_src = args.model_dir
        logger.info(f"Loading local ZeroScope from: {model_src}")
        local_only = True
    else:
        model_src = args.model_id
        logger.info(f"Loading HF ZeroScope: {model_src}")
        local_only = False

    # Load pipeline (robust)
    try:
        pipe = TextToVideoSDPipeline.from_pretrained(model_src, torch_dtype=torch.float32, local_files_only=local_only)
    except Exception as e:
        logger.warning(f"Local load failed: {e}")
        if local_only:
            logger.info("Attempting remote load (this may download files and require network)")
        try:
            pipe = TextToVideoSDPipeline.from_pretrained(model_src, torch_dtype=torch.float32, local_files_only=False)
        except Exception as e2:
            logger.error(f"Model load failed: {e2}")
            print(json.dumps({"error": f"Model load failed: {e2}"}))
            sys.exit(1)

    # CPU stability patches
    logger.info("Applying CPU stability mitigations...")
    try:
        torch.set_float32_matmul_precision('high')
        logger.info("- set_float32_matmul_precision('high')")
    except Exception:
        logger.info("- set_float32_matmul_precision not available")

    try:
        pipe.enable_attention_slicing()
        logger.info("- attention slicing enabled")
    except Exception:
        logger.info("- attention slicing not available")

    # Move to CPU and try offload where available
    try:
        pipe.to("cpu")
    except Exception as e:
        logger.error(f"Failed to move pipeline to cpu: {e}")

    try:
        pipe.enable_model_cpu_offload()
        logger.info("- model cpu offload enabled")
    except Exception:
        logger.info("- model cpu offload not available")

    # deterministic algorithms (best-effort; may slow)
    try:
        torch.use_deterministic_algorithms(True)
        logger.info("- deterministic algorithms enabled")
    except Exception:
        pass

    # Build RNG
    generator = torch.Generator("cpu").manual_seed(int(args.seed))

    # Prompt guard - avoid overflow but do not aggressively truncate identity
    prompt = args.prompt
    if prompt is None:
        prompt = ""
    if len(prompt) > 1000:
        logger.info("Prompt excessively long; truncating to 1000 chars")
        prompt = prompt[:1000]

    logger.info(f"Generating {args.num_frames} frames ({args.width}x{args.height}) with steps={args.num_steps} ...")

    # Call the pipeline in a defensive way
    try:
        out = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            num_frames=int(args.num_frames),
            width=int(args.width),
            height=int(args.height),
            num_inference_steps=int(args.num_steps),
            guidance_scale=float(args.guidance_scale),
            generator=generator,
        )
    except TypeError as e:
        logger.warning(f"Pipeline call signature mismatch: {e} - trying minimal call")
        try:
            out = pipe(prompt=prompt)
        except Exception as e2:
            logger.error(f"Pipeline failed: {e2}")
            print(json.dumps({"error": f"Inference failed: {e2}"}))
            sys.exit(1)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        print(json.dumps({"error": f"Inference failed: {e}"}))
        sys.exit(1)

    # Extract frames safely
    raw_frames = getattr(out, "frames", None)
    if raw_frames is None:
        # some pipelines return dict-like
        if isinstance(out, dict) and "frames" in out:
            raw_frames = out["frames"]
        elif isinstance(out, (list, tuple)):
            raw_frames = out
        else:
            logger.error("No frames found in pipeline output")
            print(json.dumps({"error": "No frames found in pipeline output"}))
            sys.exit(1)

    # Normalize into an iterable list
    final_frames = []
    try:
        # torch tensor -> numpy
        if isinstance(raw_frames, torch.Tensor):
            raw_frames = raw_frames.detach().cpu().numpy()

        if isinstance(raw_frames, np.ndarray):
            # If 5D: (B, T, H, W, C) -> flatten B
            if raw_frames.ndim == 5:
                logger.info(f"Detected 5D tensor: {raw_frames.shape} -> flattening batch dim")
                if raw_frames.shape[0] == 1:
                    raw_frames = raw_frames[0]
                else:
                    for b in range(raw_frames.shape[0]):
                        for t in range(raw_frames.shape[1]):
                            final_frames.append(raw_frames[b, t])
            # If 4D: (T, H, W, C)
            if getattr(raw_frames, "ndim", None) == 4:
                for t in range(raw_frames.shape[0]):
                    final_frames.append(raw_frames[t])
            else:
                # fallback - try to iterate
                try:
                    for item in raw_frames:
                        final_frames.append(item)
                except Exception:
                    final_frames.append(raw_frames)

        elif isinstance(raw_frames, (list, tuple)):
            for item in raw_frames:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        final_frames.append(sub)
                elif isinstance(item, (np.ndarray, torch.Tensor)) and getattr(item, "ndim", 0) > 3:
                    # batch inside list
                    for j in range(item.shape[0]):
                        final_frames.append(item[j])
                else:
                    final_frames.append(item)
        else:
            final_frames.append(raw_frames)
    except Exception as e:
        logger.error(f"Error during unpacking frames: {e}")
        final_frames.append(raw_frames)

    # Final validation: ensure a non-empty sequence
    n_frames = 0
    try:
        n_frames = len(final_frames)
    except Exception:
        # if final_frames is a numpy array -> check shape
        try:
            if isinstance(final_frames, np.ndarray):
                n_frames = final_frames.shape[0]
        except Exception:
            n_frames = 0

    if n_frames == 0:
        logger.error("No frames available after unpacking. Aborting.")
        print(json.dumps({"error": "No frames available after unpacking."}))
        sys.exit(1)

    # Save frames
    saved_paths = save_frames(final_frames, args.output_dir)

    print(json.dumps({"status": "COMPLETED", "frame_paths": saved_paths, "prompt": prompt}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--model-id", type=str, default="cerspense/zeroscope_v2_576w")
    parser.add_argument("--model-dir", type=str, default=None)

    args = parser.parse_args()
    run_inference(args)
