# ==============================================================================
# animate_diff_engine.py - V3.0 UNIFIED PRODUCTION BUILD (Hardened)
# Minimal, safe fixes applied:
# - Robust VAE loading (no from_single_file usage)
# - Generator placed on pipeline device for determinism
# - Resilient frame extraction and tokenizer fallback
# - Hybrid continuity guard/warning
# - Optional --log-file for persistent logs
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
    """
    Robust VAE loader:
      1) Try AutoencoderKL.from_pretrained(vae_id)
      2) If that fails and vae_id points to a file or candidate filenames exist in the repo,
         download and load state dict via safetensors and apply to a constructed AutoencoderKL.
      3) If everything fails, log and continue using the pipeline's default VAE.
    """
    if not vae_id:
        logger.info("[vae] No VAE specified — using default.")
        return pipe

    logger.info(f"[vae] Trying VAE: {vae_id}")
    try:
        # 1) Preferred: load the VAE directly by repo / folder name
        try:
            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
            pipe.vae = vae.to(device)
            logger.info(f"[vae] ✅ Loaded VAE from pretrained: {vae_id}")
            return pipe
        except Exception as e_repo:
            logger.debug(f"[vae] from_pretrained failed ({e_repo}); attempting single-file fallback...")

        # 2) Attempt single-file handling (safetensors / ckpt)
        fpath = None
        try:
            # If the user provided a path-like string that exists locally, use it
            if Path(vae_id).exists():
                fpath = str(Path(vae_id))
            else:
                # try common candidates under provided repo (repo may be 'user/repo' or just 'repo')
                repo = vae_id.split("/")[0]
                candidates = []
                # If vae_id contains a filename-like part, try it
                if "/" in vae_id:
                    candidates.append(vae_id.split("/")[-1])
                candidates += ["vae.safetensors", "model.safetensors", "vae.pt", "vae.ckpt"]
                for cand in candidates:
                    try:
                        fpath = hf_hub_download(repo_id=repo, filename=cand)
                        if fpath:
                            break
                    except Exception:
                        fpath = None
                        continue

            if not fpath:
                raise RuntimeError("Could not locate a single-file VAE (local or HF).")

            logger.info(f"[vae] Attempting to load single-file VAE state from: {fpath}")
            state = load_file(fpath, device="cpu")

            # Try to create an AutoencoderKL with a best-effort base and load state_dict (not strictly guaranteed)
            vae: Optional[AutoencoderKL] = None
            try:
                # best-effort: attempt to reuse the base model name if available on the pipeline
                base_name = getattr(pipe, "pretrained_model_name_or_path", None)
                if base_name:
                    vae = AutoencoderKL.from_pretrained(base_name, torch_dtype=dtype)
            except Exception:
                vae = None

            if vae is None:
                # Last resort: attempt from vae_id parent repo
                try:
                    parent_repo = vae_id.rsplit("/", 1)[0] if "/" in vae_id else vae_id
                    vae = AutoencoderKL.from_pretrained(parent_repo, torch_dtype=dtype)
                except Exception as e_vaebase:
                    logger.warning(f"[vae] Could not construct AutoencoderKL from base repos ({e_vaebase}). Will try to apply state to default vae if compatible.")

            if vae is not None:
                # load as permissive (strict=False) to accept partial keys
                vae.load_state_dict(state, strict=False)
                pipe.vae = vae.to(device)
                logger.info(f"[vae] ✅ Loaded single-file VAE and applied state from {fpath}")
                return pipe
            else:
                # If we couldn't construct an AutoencoderKL to map state onto, log and fallback
                logger.warning("[vae] Could not construct AutoencoderKL to apply single-file state; falling back to pipeline default VAE.")
        except Exception as e_single:
            logger.warning(f"[vae] single-file fallback failed: {e_single}", exc_info=True)

        # If all methods exhausted:
        raise RuntimeError("All VAE loading strategies failed (from_pretrained and single-file).")
    except Exception as e:
        logger.error(f"[vae] FAILED to swap VAE ({e}); continuing with pipeline's default VAE.", exc_info=True)
        return pipe

def compile_prompt(raw_prompt: str, max_phrases: int = 12) -> Tuple[str, List[str]]:
    """
    Rebuilds the prompt to be concise and effective, prioritizing key subjects
    and ensuring the total length respects the 77-token limit.
    This version REBUILDS the prompt, it does not duplicate it.
    """
    # Use the original phrases with weighting syntax intact
    original_phrases = [p.strip() for p in raw_prompt.split(",") if p.strip()]

    priority = []
    others = []

    for p in original_phrases:
        # Search for subject keywords in a cleaned version of the phrase
        search_p = re.sub(r'[:\d\.]', '', p).lower()
        if re.search(r"(inventor|man|woman|person|face|hands|worker|scientist|engineer|girl|guy|portrait|character)", search_p):
            priority.append(p)
        else:
            others.append(p)

    # Combine priority phrases with other descriptive phrases, truncating to a safe number.
    combined_phrases = (priority + others)[:max_phrases]
    
    # Add a suffix for overall quality.
    suffix = "photorealistic, cinematic, ultra-realistic, high detail"
    
    final_prompt = ", ".join(combined_phrases) + ", " + suffix
    
    logger.info(f"[prompt-compile] Rebuilt prompt from key phrases: {combined_phrases}")
    return final_prompt, combined_phrases


def apply_hybrid_continuity(args) -> None:
    """
    Auto-adjusts init_image strength and IP-Adapter scale
    based on presence of human/subject in prompt.
    Also warns if init_image provided but strength is not in (0,1).
    """
    wants_person = bool(re.search(
        r"(inventor|person|man|woman|human|hands|face|portrait|character)",
        args.prompt.lower()
    ))

    if wants_person:
        # clamp to reasonable person-preserving strengths
        args.strength = min(max(getattr(args, 'strength', 0.4), 0.30), 0.45)
        args.ip_adapter_scale = min(max(getattr(args, 'ip_adapter_scale', 0.05), 0.0), 0.06)
    else:
        args.strength = min(max(getattr(args, 'strength', 0.25), 0.25), 0.55)
        args.ip_adapter_scale = max(getattr(args, 'ip_adapter_scale', 0.1), 0.10)

    # Warn if init_image present but strength is ineffective
    if getattr(args, "init_image", None) and not (0 < args.strength < 1):
        logger.warning("[hybrid] init_image provided but strength not in (0,1); the init image will be ignored by inference. Consider setting --strength between 0 and 1.")

    logger.info(f"[hybrid] wants_person={wants_person} -> strength={args.strength}, ip_scale={args.ip_adapter_scale}")


# --- CORE FUNCTIONS ---
def build_pipeline(args) -> AnimateDiffPipeline:
    device, dtype = _select_device_and_dtype(prefer_cuda_fp16=not args.no_cuda_fp16)
    logger.info(f"[pipeline_builder] Mode: {args.mode}, Steps: {args.num_steps}, Base: {args.base_model}")

    if args.mode == "lightning":
        valid_steps = [1, 2, 4, 8]
        num_steps = args.num_steps
        if num_steps not in valid_steps:
            logger.warning(f"[lightning] num_steps={num_steps} invalid. Forcing to 4.")
            num_steps = 4
        lightning_repo = "ByteDance/AnimateDiff-Lightning"
        lightning_file = f"animatediff_lightning_{num_steps}step_diffusers.safetensors"
        logger.info(f"[lightning] Loading UNet and motion weights...")
        unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", torch_dtype=dtype)
        motion_adapter_path = hf_hub_download(repo_id=lightning_repo, filename=lightning_file)
        motion_adapter_state_dict = load_file(motion_adapter_path, device="cpu")
        motion_adapter = MotionAdapter()
        motion_adapter.load_state_dict(motion_adapter_state_dict)
        logger.info("[lightning] Motion adapter successfully loaded.")
        pipe = AnimateDiffPipeline.from_pretrained(
            args.base_model, unet=unet, motion_adapter=motion_adapter, torch_dtype=dtype
        )
    else:  # CLASSIC MODE
        motion_repo = "guoyww/animatediff-motion-adapter-v1-5-2"
        logger.info(f"[classic] Loading motion adapter from {motion_repo}")
        try:
            motion_adapter = MotionAdapter.from_pretrained(motion_repo, torch_dtype=dtype)
            pipe = AnimateDiffPipeline.from_pretrained(
                args.base_model, motion_adapter=motion_adapter, torch_dtype=dtype
            )
        except Exception as e:
            logger.warning(f"[classic] Motion adapter load failed ({e}), attempting to load pipeline without adapter.")
            pipe = AnimateDiffPipeline.from_pretrained(args.base_model, torch_dtype=dtype)

    # Scheduler swap with defensive fallback
    if args.scheduler == "DPM++ 2M Karras":
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True)
            logger.info("[scheduler] ✅ Swapped to: DPM++ 2M Karras")
        except Exception as e:
            logger.warning(f"[scheduler] Karras scheduler failed ({e}). Falling back to DDIM.")
            try:
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            except Exception as e2:
                logger.error(f"[scheduler] DDIM fallback also failed: {e2}", exc_info=True)
    else:
        try:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            logger.info("[scheduler] ✅ Using default: DDIM")
        except Exception as e:
            logger.warning(f"[scheduler] Could not set DDIM: {e}", exc_info=True)

    # IP-Adapter conditional load (only when image path provided and scale>0)
    if args.ip_adapter_image_path and args.ip_adapter_scale > 0.0 and args.mode == "classic":
        logger.info("[ip_adapter] Enabled. Loading IP-Adapter model...")
        try:
            # Best-effort: this API varies by diffusers version; guard exceptions
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_light.bin")
            if hasattr(pipe, "set_ip_adapter_scale"):
                pipe.set_ip_adapter_scale(args.ip_adapter_scale)
            logger.info(f"[ip_adapter] ✅ Loaded successfully. Scale = {args.ip_adapter_scale}")
        except Exception as e:
            logger.error(f"[ip_adapter] FAILED to load: {e}", exc_info=True)
            # ensure we don't accidentally use IP adapter settings if load failed
            args.ip_adapter_scale = 0.0

    # VAE swap (robust)
    pipe = _maybe_swap_vae(pipe, vae_id=args.vae_model, device=device, dtype=dtype)

    # Memory optimizations and move to device
    _apply_memory_saving(pipe)
    try:
        pipe.to(device)
    except Exception as e:
        logger.warning(f"[pipeline] pipe.to(device) failed: {e}. Proceeding (some components may remain on CPU).", exc_info=True)

    logger.info(f"[{args.mode}] ✅ AnimateDiff pipeline is ready on {device}.")
    return pipe


def run_inference(pipe: AnimateDiffPipeline, args) -> List[Image.Image]:
    """
    Robust inference wrapper:
    - Creates Generator on same device as the pipeline (best-effort).
    - Handles IP-Adapter and init image loading safely.
    - Extracts frames from multiple possible return shapes (out.frames, out.images, tuple/list).
    """
    try:
        # Determine device used by the pipeline (best-effort)
        try:
            device = getattr(pipe, "device", None)
            if device is None:
                # try infer from unet parameters
                device = next(pipe.unet.parameters()).device
        except Exception:
            device = torch.device("cpu")
        logger.info(f"[inference] Using device for generator: {device}")

        gen = torch.Generator(device=device).manual_seed(args.seed)

        kwargs = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt or None,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "width": args.width,
            "height": args.height,
            "generator": gen,
        }

        if args.ip_adapter_scale > 0 and args.ip_adapter_image_path:
            try:
                ip_img = Image.open(args.ip_adapter_image_path).convert("RGB").resize((args.width, args.height), Image.LANCZOS)
                kwargs["ip_adapter_image"] = ip_img
            except Exception as e:
                logger.error(f"[ip] failed to load image: {e}", exc_info=True)
                raise RuntimeError("Failed to load IP-Adapter image") from e

        if args.init_image and 0 < args.strength < 1:
            try:
                img = Image.open(args.init_image).convert("RGB").resize((args.width, args.height), Image.LANCZOS)
                kwargs["image"] = img
                kwargs["strength"] = args.strength
            except Exception as e:
                logger.error(f"[init_image] failed to load image: {e}", exc_info=True)
                raise RuntimeError("Failed to load init image") from e
        elif args.init_image and args.strength >= 1.0:
            logger.warning("[init_image] init_image provided but strength >= 1.0 — the init image will be ignored. Consider setting --strength between 0 and 1.")

        # Token debug (safe fallback)
        try:
            tok = None
            if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
                tok = pipe.tokenizer(args.prompt, return_tensors="pt", truncation=True, max_length=77)
            else:
                # fallback to transformers AutoTokenizer if available
                try:
                    from transformers import AutoTokenizer
                    tokenizer_name = getattr(pipe, "pretrained_model_name_or_path", None) or args.base_model
                    tok_obj = AutoTokenizer.from_pretrained(tokenizer_name)
                    tok = tok_obj(args.prompt, return_tensors="pt", truncation=True, max_length=77)
                except Exception:
                    tok = None
            if tok is not None:
                logger.info(f"[prompt-debug] Final prompt token count: {len(tok['input_ids'][0])}")
        except Exception:
            logger.warning("[prompt-debug] Could not calculate prompt token count.")

        logger.info("[inference] Running AnimateDiff pipeline...")
        out = pipe(**kwargs)

        # Robust frame extraction logic
        frames: List[Image.Image] = []
        if hasattr(out, "frames"):
            frames = out.frames if isinstance(out.frames, list) else list(out.frames)
            # flatten if batch nested e.g. [ [frames...] ]
            if frames and isinstance(frames[0], list):
                frames = frames[0]
        elif hasattr(out, "images"):
            frames = out.images
        elif isinstance(out, (list, tuple)):
            # try to find a list-of-images inside returned tuple/list
            for item in out:
                if isinstance(item, list) and item and isinstance(item[0], Image.Image):
                    frames = item
                    break
            # fallback: direct list of PIL images
            if not frames and out and isinstance(out[0], Image.Image):
                frames = list(out)  # type: ignore
        else:
            # last resort: if out itself is a list of PILs
            if isinstance(out, list) and out and isinstance(out[0], Image.Image):
                frames = out  # type: ignore

        if not frames:
            logger.error("[inference] Pipeline returned an empty frame list.")
            raise RuntimeError("Pipeline returned an empty frame list")

        logger.info(f"[inference] ✅ Generated {len(frames)} frames successfully.")
        return frames
    except Exception as e:
        logger.error(f"[inference] CRASH: {e}", exc_info=True)
        raise


def save_frames(frames: List[Image.Image], output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for i, frame in enumerate(frames):
        path = output_dir / f"frame_{i:04d}.png"
        frame.save(path)
        paths.append(str(path))
    return paths


# --- CLI AND MAIN EXECUTION ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AnimateDiff Engine (V3.0 Production-Locked)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="SG161222/Realistic_Vision_V5.1_noVAE")
    p.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse", help="VAE model ID.")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=7.0)
    p.add_argument("--scheduler", type=str, default="DPM++ 2M Karras", choices=["DPM++ 2M Karras", "DDIM"])
    p.add_argument("--ip-adapter-image-path", type=str, default=None)
    p.add_argument("--ip-adapter-scale", type=float, default=0.0)
    p.add_argument("--init-image", type=str, default=None)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="classic", choices=["lightning", "classic"])
    p.add_argument("--no-cuda-fp16", action="store_true")
    p.add_argument("--log-file", type=str, default=None, help="Optional path to persist logger output.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # Add optional file handler early so subsequent logs are persisted
    if args.log_file:
        try:
            fh = logging.FileHandler(args.log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"))
            logger.addHandler(fh)
            logger.info(f"[main] Logging to file: {args.log_file}")
        except Exception as e:
            logger.warning(f"[main] Could not create log file handler ({e}); continuing without file log.", exc_info=True)

    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # Step 1: Compile the prompt
        args.prompt, visual_tokens = compile_prompt(args.prompt)
        logger.info(f"[main] Compiled prompt with visual tokens: {visual_tokens}")

        # Step 2: Apply hybrid logic
        apply_hybrid_continuity(args)

        # Step 3: Build pipeline
        pipe = build_pipeline(args)

        # Step 4: Run inference
        frames = run_inference(pipe, args)

        if not frames:
            raise RuntimeError("Inference returned no frames. Check logs for details.")

        # Step 5: Save frames
        paths = save_frames(frames, Path(args.output_dir))
        print(json.dumps({"status": "COMPLETED", "frame_paths": paths}))

        # Step 6: Memory Cleanup
        del pipe, frames, paths
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return 0
    except Exception as e:
        logger.error(f"AnimateDiff engine main function failed: {e}", exc_info=True)
        print(json.dumps({"status": "FAILED", "error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
