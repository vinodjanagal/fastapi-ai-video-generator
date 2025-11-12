
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

def _maybe_swap_vae(pipe: AnimateDiffPipeline, vae_id: Optional[str], device: torch.device, dtype: torch.dtype) -> AnimateDiffPipeline:
    """Optional VAE swap for improved image sharpness."""
    if not vae_id:
        return pipe
    try:
        logger.info(f"[vae] Swapping VAE -> {vae_id}")
        vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
        pipe.vae = vae.to(device)
    except Exception as e:
        logger.warning(f"[vae] VAE swap failed ({vae_id}): {e}")
    return pipe

# =====================================================================
# === FINAL PRODUCTION-GRADE PIPELINE BUILDER (Corrected Version) =====
# =====================================================================

def build_pipeline(args) -> AnimateDiffPipeline:
    """Builds a fully stable AnimateDiff pipeline with correct motion adapter and scheduler."""
    device, dtype = _select_device_and_dtype(prefer_cuda_fp16=not args.no_cuda_fp16)
    num_steps = args.num_steps
    mode = args.mode

    logger.info(f"[pipeline_builder] ✅ Mode selected: {mode} (num_steps={num_steps})")
    logger.info(f"[pipeline_builder] Base model: {args.base_model}")

    # -------------------- FIX 1: Proper Motion Adapter Load --------------------
    if mode == "lightning":
        valid_steps = [1, 2, 4, 8]
        if num_steps not in valid_steps:
            logger.warning(f"[lightning] num_steps={num_steps} invalid. Forcing to 4.")
            num_steps = 4

        lightning_repo = "ByteDance/AnimateDiff-Lightning"
        lightning_file = f"animatediff_lightning_{num_steps}step_diffusers.safetensors"

        logger.info(f"[lightning] Loading UNet and motion weights...")
        unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder="unet", torch_dtype=dtype)

        motion_adapter_path = hf_hub_download(repo_id=lightning_repo, filename=lightning_file)
        motion_adapter_state_dict = load_file(motion_adapter_path, device="cpu")

        # ✅ Correctly instantiate and load the MotionAdapter
        motion_adapter = MotionAdapter()
        motion_adapter.load_state_dict(motion_adapter_state_dict)
        logger.info("[lightning] Motion adapter successfully loaded into memory.")

        pipe = AnimateDiffPipeline.from_pretrained(
            args.base_model,
            unet=unet,
            motion_adapter=motion_adapter,
            torch_dtype=dtype
        )

        # ✅ FIX 3: Use DDIM instead of Euler for stability and clarity
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="leading",
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    else:  # CLASSIC MODE
        motion_repo = "guoyww/animatediff-motion-adapter-v1-5-2"
        logger.info(f"[classic] Loading motion adapter from {motion_repo}")
        motion_adapter = MotionAdapter.from_pretrained(motion_repo, torch_dtype=dtype)
        pipe = AnimateDiffPipeline.from_pretrained(
            args.base_model, motion_adapter=motion_adapter, torch_dtype=dtype
        )

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="leading",
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    # -------------------- FIX 2: Disable IP-Adapter in Lightning Mode --------------------
    if mode == "lightning":
        logger.info("[ip_adapter] Skipped — incompatible with Lightning mode to avoid over-blurring.")
    elif args.ip_adapter_image_path:
        logger.info("[ip_adapter] Enabled. Loading IP-Adapter model...")
        try:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_light.bin")
            pipe.set_ip_adapter_scale(args.ip_adapter_scale)
            logger.info(f"[ip_adapter] Loaded successfully. Scale = {args.ip_adapter_scale}")
        except Exception as e:
            logger.error(f"[ip_adapter] FAILED: {e}", exc_info=True)
    else:
        logger.info("[ip_adapter] Not provided — skipping.")

    # Optional VAE
    from app.video_engines.heavy_config import HeavyEngineConfig
    pipe = _maybe_swap_vae(pipe, vae_id=getattr(HeavyEngineConfig(), "VAE_MODEL", None), device=device, dtype=dtype)

    _apply_memory_saving(pipe)
    pipe.to(device)
    logger.info(f"[{mode}] ✅ AnimateDiff pipeline is ready on {device}.")
    return pipe


# ======================== INFERENCE + SAVE =============================

def run_inference(pipe: AnimateDiffPipeline, args) -> List[Image.Image]:
    """
    Run AnimateDiff inference with robust IP-Adapter handling.

    Important behaviors:
    - We only pass the `ip_adapter_image` into the pipeline call when the
      `pipe` actually has an IP-Adapter loaded (or, more generally, the
      internal attributes required for IP-Adapter embedding preparation).
    - This prevents the pipeline from entering IP-Adapter code paths when
      it was intentionally not configured (e.g., Lightning mode).
    """
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

    # --- Defensive IP-Adapter handling ---
    # Only add an ip_adapter_image if:
    #  1) the user provided a path (args.ip_adapter_image_path), AND
    #  2) the pipeline actually appears to support IP-Adapter (pipe.ip_adapter is present),
    #     or the UNet has 'encoder_hid_proj' with the expected attribute.
    ip_path = getattr(args, "ip_adapter_image_path", None)
    pipeline_supports_ip = getattr(pipe, "ip_adapter", None) is not None

    # Extra defensive fallback: check the unet internals which animatediff expects
    if not pipeline_supports_ip:
        try:
            # If unet.encoder_hid_proj exists and has image_projection_layers, pipeline likely supports IP
            pipeline_supports_ip = bool(
                getattr(getattr(pipe, "unet", None), "encoder_hid_proj", None) is not None
            )
        except Exception:
            pipeline_supports_ip = False

    if ip_path:
        if pipeline_supports_ip:
            try:
                logger.info(f"Loading IP-Adapter reference image: {ip_path}")
                ip_img = Image.open(ip_path).convert("RGB")
                pipeline_kwargs["ip_adapter_image"] = ip_img
            except FileNotFoundError:
                logger.warning(f"IP-Adapter reference image not found at: {ip_path} -- continuing without it.")
            except Exception as e:
                logger.error(f"Failed to load IP-Adapter reference image: {e} -- continuing without it.")
        else:
            # Important: user asked for a reference image but pipeline doesn't support IP-Adapter.
            # We avoid passing the image to prevent NoneType attribute errors.
            logger.warning(
                "IP-Adapter reference image was provided, but the current pipeline "
                "was built without an IP-Adapter (or required UNet components are missing). "
                "Skipping ip_adapter_image to avoid runtime errors."
            )

    logger.info("Running AnimateDiff pipeline inference...")

    if args.init_image:
        try:
            logger.info(f"Loading init_image for compositional change: {args.init_image}")
            init_img = Image.open(args.init_image).convert("RGB").resize((args.width, args.height))
            pipeline_kwargs["image"] = init_img  # NOTE: The parameter is 'image', not 'init_image' for this pipeline
            pipeline_kwargs["strength"] = args.strength
        except Exception as e:
            logger.error(f"Failed to load or use init_image: {e}")

        result = pipe(**pipeline_kwargs)
        return result.frames[0]

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
    p = argparse.ArgumentParser(description="AnimateDiff Engine (Stable Production Build)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--base-model", default="SG161222/Realistic_Vision_V5.1_noVAE")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--ip-adapter-scale", type=float, default=0.12, help="IP scale (0.1-0.3 safe)")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="lightning", choices=["lightning", "classic"], help="Pipeline mode: lightning for speed, classic for quality.")
    p.add_argument("--ip-adapter-image-path", type=str, default=None)
    p.add_argument("--no-cuda-fp16", action="store_true")
    p.add_argument("--init-image", type=str, default=None)
    p.add_argument("--strength", type=float, default=0.75) # A good starting point for significant change
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
