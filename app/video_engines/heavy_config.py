# app/video_engines/heavy_config.py

from pydantic import BaseModel, Field
from typing import Optional

class HeavyEngineConfig(BaseModel):
    """
    Configuration for the AnimateDiff engine, tuned for LOW-RESOURCE CPU execution.
    """
    
    # --- Device & Precision ---
    # We are hardcoding these for CPU-only execution to prevent any mistakes.
    DEVICE: str = "cpu"
    TORCH_DTYPE_STR: str = "float32" # CPU does not benefit from float16

    # --- Model IDs ---
    # We will use the same models, but load them carefully.
    TXT2IMG_MODEL: str = "runwayml/stable-diffusion-v1-5"
    VAE_MODEL: Optional[str] = "stabilityai/sd-vae-ft-ema"
    ANIMATEDIFF_ADAPTER_REPO_ID: str = "ByteDance/AnimateDiff-Lightning"
    ANIMATEDIFF_LIGHTNING_ADAPTER_FILENAME: str = "animatediff_lightning_4step_diffusers.safetensors"

    # --- CRITICAL: Generation Parameters (Our Levers for Control) ---
    
    # We are drastically reducing the resolution. This is the single biggest factor
    # for reducing memory usage and speeding up computation.
    IMG_WIDTH: int = Field(default=256, description="LOW-RES: Target image width.")
    IMG_HEIGHT: int = Field(default=256, description="LOW-RES: Target image height.")

    # We will generate a very small number of frames at a time.
    ANIMATEDIFF_NUM_FRAMES: int = Field(default=8, description="LOW-COUNT: Target number of frames per generation call.")
    
    # We will use the minimum number of steps for the Lightning model.
    ANIMATEDIFF_NUM_STEPS: int = Field(default=4, description="MIN-STEPS: Number of inference steps.")
    
    # A standard guidance scale.
    ANIMATEDIFF_GUIDANCE_SCALE: float = Field(default=1.0)
    
    # --- Feature Flags ---
    # We will disable almost all advanced features to save memory and CPU.
    USE_IP_ADAPTER: bool = False # IP-Adapter is too heavy for this setup.
    USE_GFPGAN_CORRECTION: bool = False
    USE_COLOR_CORRECTION: bool = False
    
    # --- Prompts ---
    BASE_NEGATIVE_PROMPT: str = "blurry, low quality, distortion, artifacts"