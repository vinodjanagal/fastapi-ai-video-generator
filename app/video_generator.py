# FILE: app/video_generator.py
import asyncio
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from moviepy import ImageSequenceClip, AudioFileClip
import logging

logger = logging.getLogger("video_generator")
logging.basicConfig(format="%(asctime)s - %(levelname)s - [video_gen] - %(message)s", level=logging.INFO)

# Default Widescreen for ZeroScope
WIDTH, HEIGHT = 576, 320
FPS = 24

def ease_in_out_quad(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2

def _safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGBA")
    except Exception:
        return None

async def apply_camera_motion(frame_paths: List[str], motion_type: str, width: int = WIDTH, height: int = HEIGHT, zoom_intensity: float = 0.20, pan_intensity: float = 0.15) -> Tuple[Optional[tempfile.TemporaryDirectory], List[str]]:
    if not frame_paths:
        return None, []

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="motion_frames_")
    temp_dir = Path(temp_dir_obj.name)
    temp_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _process_frames_for_motion, frame_paths, str(temp_dir), motion_type, width, height, zoom_intensity, pan_intensity)

    new_paths = sorted([str(p) for p in temp_dir.glob("frame_*.png")])
    return temp_dir_obj, new_paths

def _process_frames_for_motion(frame_paths, temp_dir_path, motion_type, width, height, zoom_intensity, pan_intensity):
    from PIL import Image as PILImage
    temp_dir = Path(temp_dir_path)
    n = max(1, len(frame_paths))

    for i, p in enumerate(frame_paths):
        progress = i / (n - 1) if n > 1 else 0.0
        t = ease_in_out_quad(progress)

        img = _safe_open_image(p)
        if img is None:
            img = PILImage.new("RGBA", (width, height), (0, 0, 0, 255))

        if img.size != (width, height):
            img = img.resize((width, height), PILImage.LANCZOS)

        if motion_type == "static":
            transformed = img
        elif motion_type in ["slow_zoom_in", "slow_zoom_out"]:
            if motion_type == "slow_zoom_out":
                t = 1.0 - t
            final_w = int(width * (1.0 - zoom_intensity))
            final_h = int(height * (1.0 - zoom_intensity))
            cur_w = int(width - (width - final_w) * t)
            cur_h = int(height - (height - final_h) * t)
            left = (width - cur_w) // 2
            top = (height - cur_h) // 2
            transformed = img.crop((left, top, left + cur_w, top + cur_h))
        elif motion_type == "pan_right":
            crop_w = int(width * (1.0 - pan_intensity))
            max_offset = width - crop_w
            offset = int(max_offset * t)
            transformed = img.crop((offset, 0, offset + crop_w, height))
        elif motion_type == "pan_left":
            crop_w = int(width * (1.0 - pan_intensity))
            max_offset = width - crop_w
            offset = int(max_offset * (1.0 - t))
            transformed = img.crop((offset, 0, offset + crop_w, height))
        else:
            transformed = img

        if transformed.size != (width, height):
            transformed = transformed.resize((width, height), PILImage.LANCZOS)

        transformed.save(temp_dir / f"frame_{i:05d}.png")

async def create_video_from_frames(frame_paths: List[str], output_path: str, audio_path: Optional[str]) -> str:
    if not frame_paths:
        raise ValueError("No frames provided to assemble video.")

    def _assemble():
        sorted_paths = sorted(frame_paths)
        clip = ImageSequenceClip(sorted_paths, fps=FPS)
        audio_clip = None
        try:
            if audio_path:
                audio_clip = AudioFileClip(audio_path)
                # moviepy 2.x uses with_audio; older uses set_audio
                if hasattr(clip, "with_audio"):
                    clip = clip.with_audio(audio_clip)
                else:
                    clip = clip.set_audio(audio_clip)
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac" if audio_path else None, fps=FPS, logger=None)
        finally:
            try:
                if audio_clip:
                    audio_clip.close()
            except Exception:
                pass
            try:
                clip.close()
            except Exception:
                pass

    await asyncio.get_event_loop().run_in_executor(None, _assemble)
    return output_path
