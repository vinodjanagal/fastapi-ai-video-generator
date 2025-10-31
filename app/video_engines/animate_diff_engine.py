# app/video_engines/animate_diff_engine.py

import argparse
import json
import logging
import sys
from pathlib import Path

# ++++++++++++++++++++++ ADD THIS BLOCK +++++++++++++++++++++++++++
# --- Path Correction for Standalone Execution ---
# This makes the script runnable from the project root for testing.
try:
    # Try to import from the existing path
    from app.video_engines.heavy_config import HeavyEngineConfig
except ModuleNotFoundError:
    # If it fails, we're likely running standalone, so add the project root to the path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(project_root))
    logger = logging.getLogger(__name__) # Re-initialize logger after potential path change
    logger.info(f"Added project root to Python path: {project_root}")
    from app.video_engines.heavy_config import HeavyEngineConfig
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler,UNet2DConditionModel 
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Import our special low-resource config
from app.video_engines.heavy_config import HeavyEngineConfig

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [animate_diff_engine] - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


def generate_frames(cfg: HeavyEngineConfig, prompt: str, output_dir: Path) -> list[str]:
    logger.info(f"Initializing AnimateDiff pipeline on device: {cfg.DEVICE}")
    
    dtype = torch.float16 # We will use float16 for loading to save memory

    try:
        # --- THIS IS THE CORRECTED LOADING LOGIC ---
        # 1. Load Motion Adapter manually
        logger.info("Loading Motion Adapter...")
        # First, we need a config. The most reliable way is to get it from a compatible UNet's subfolder.
        # We will use the UNet from the base model for this.
        unet_config = UNet2DConditionModel.load_config(cfg.TXT2IMG_MODEL, subfolder="unet")
        motion_adapter = MotionAdapter.from_config(unet_config)
        
        motion_adapter_path = hf_hub_download(
            repo_id=cfg.ANIMATEDIFF_ADAPTER_REPO_ID,
            filename=cfg.ANIMATEDIFF_LIGHTNING_ADAPTER_FILENAME
        )
        motion_adapter.load_state_dict(load_file(motion_adapter_path, device="cpu"))
        logger.info("Motion Adapter loaded successfully.")

        # 2. Load the base pipeline with memory optimizations
        logger.info(f"Loading base model '{cfg.TXT2IMG_MODEL}' with fp16 variant...")
        pipe = AnimateDiffPipeline.from_pretrained(
            cfg.TXT2IMG_MODEL,
            motion_adapter=motion_adapter,
            torch_dtype=dtype,
            variant="fp16",
        )
        # ------------------------------------------------

        
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear"
        )
        
        logger.info("AnimateDiff pipeline initialized with MEMORY SAVING optimizations.")
    except Exception as e:
        logger.critical(f"Failed to initialize AnimateDiff pipeline: {e}", exc_info=True)
        raise RuntimeError("Pipeline initialization failed") from e

    # --- Generation ---
    logger.info(f"Generating {cfg.ANIMATEDIFF_NUM_FRAMES} frames for prompt: '{prompt[:100]}...'")
    logger.warning("This will be VERY slow on a CPU due to model offloading. Please be patient.")

    try:
        # No need for torch.no_grad() here, the pipeline handles it.
        output = pipe(
            prompt=prompt,
            negative_prompt=cfg.BASE_NEGATIVE_PROMPT,
            num_frames=cfg.ANIMATEDIFF_NUM_FRAMES,
            guidance_scale=cfg.ANIMATEDIFF_GUIDANCE_SCALE,
            num_inference_steps=cfg.ANIMATEDIFF_NUM_STEPS,
            width=cfg.IMG_WIDTH,
            height=cfg.IMG_HEIGHT,
        )
        
        frames = output.frames[0]
        logger.info(f"Successfully generated {len(frames)} frames in memory.")
    except Exception as e:
        logger.critical(f"Frame generation failed: {e}", exc_info=True)
        raise RuntimeError("Frame generation failed") from e

    # --- Saving Frames ---
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    logger.info(f"Saving {len(frames)} frames to {output_dir}...")
    for i, frame in enumerate(frames):
        frame_path = output_dir / f"frame_{i:04d}.png"
        frame.save(frame_path)
        frame_paths.append(str(frame_path))
        
    logger.info(f"Saved {len(frame_paths)} frames successfully.")
    return frame_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnimateDiff Frame Generation Engine (Low-Resource)")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt for the animation.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output frames.")
    
    args = parser.parse_args()
    
    try:
        config = HeavyEngineConfig()
        
        saved_frame_paths = generate_frames(
            cfg=config,
            prompt=args.prompt,
            output_dir=Path(args.output_dir)
        )

        result = {
            "status": "COMPLETED",
            "frame_paths": saved_frame_paths
        }
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        result = {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {e}"
        }
        print(json.dumps(result))
        sys.exit(1)