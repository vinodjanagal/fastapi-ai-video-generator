# app/video_engines/heavy_config.py

from pydantic import BaseModel, Field
from typing import Optional

class HeavyEngineConfig(BaseModel):
    DEVICE: str = "cpu"
    TORCH_DTYPE_STR: str = "float32"

    # Use SD1.5 base (working) but you can override to stylised model for quality
    TXT2IMG_MODEL: str = "Lykon/dreamshaper-8"

    VAE_MODEL: Optional[str] = "stabilityai/sd-vae-ft-mse"

    # Use official AnimateDiff-Lightning repo
    ANIMATEDIFF_ADAPTER_REPO_ID: str = "ByteDance/AnimateDiff-Lightning"

    # Use correct filename for the 4-step model
    ANIMATEDIFF_LIGHTNING_ADAPTER_FILENAME: str = "animatediff_lightning_4step_diffusers.safetensors"

    # Generation parameters
    IMG_WIDTH: int = Field(default=256, description="LOW-RES width for CPU preview.")
    IMG_HEIGHT: int = Field(default=256, description="LOW-RES height for CPU preview.")
    ANIMATEDIFF_NUM_FRAMES: int = Field(default=8, description="Number of frames per generation.")
    ANIMATEDIFF_NUM_STEPS: int = Field(default=4, description="Inference steps for Lightning model.")
    ANIMATEDIFF_GUIDANCE_SCALE: float = Field(default=1.0)

    USE_IP_ADAPTER: bool = False
    USE_GFPGAN_CORRECTION: bool = False
    USE_COLOR_CORRECTION: bool = False

    BASE_NEGATIVE_PROMPT: str = "blurry, low quality, distortion, artifacts"
