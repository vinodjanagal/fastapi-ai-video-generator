# FILE: app/video_generator.py
"""
Video assembly + camera motion utilities.
- Ensures all frames are forced to the target resolution (prevents MoviePy size errors)
- Applies per-frame camera motion (zoom/pan) to the *entire* frame sequence (not just single-image Ken Burns)
- Assembles MP4 with audio and calculated FPS
- Defaults tuned to match PhoenixDirector (384x384)
"""

import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageFilter
from moviepy import ImageSequenceClip, AudioFileClip
import logging
import math

logger = logging.getLogger(__name__)

# === Config (match PhoenixDirector) ===
FPS = 24
MIN_FPS = 1
MAX_FPS = 60
DEFAULT_WIDTH = 384
DEFAULT_HEIGHT = 384


# -------------------------
# Utilities
# -------------------------
def ease_in_out_quad(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2


def _safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGBA")
        return img
    except Exception as e:
        logger.warning(f"Failed to open image '{path}': {e}")
        return None


# -------------------------
# Camera motion
# -------------------------
async def apply_camera_motion(
    frame_paths: List[str],
    motion_type: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    zoom_intensity: float = 0.20,
    pan_intensity: float = 0.15,
) -> Tuple[Optional[tempfile.TemporaryDirectory], List[str]]:
    """
    Apply camera motion to frames and write transformed frames into a NEW temporary directory.
    Returns (tempdir_obj, new_frame_paths).
    If motion_type == "static" returns (None, frame_paths) but still verifies size.
    """
    if not frame_paths:
        return None, []

    num_frames = len(frame_paths)
    logger.info(f"Applying motion '{motion_type}' to {num_frames} frames (target {width}x{height})")

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="motion_frames_")
    temp_dir = Path(temp_dir_obj.name)
    temp_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        _process_frames_for_motion,
        frame_paths,
        str(temp_dir),
        motion_type,
        width,
        height,
        zoom_intensity,
        pan_intensity,
    )

    new_paths = sorted([str(p) for p in temp_dir.glob("frame_*.png")])
    return temp_dir_obj, new_paths


def _process_frames_for_motion(
    frame_paths: List[str],
    temp_dir_path: str,
    motion_type: str,
    width: int,
    height: int,
    zoom_intensity: float,
    pan_intensity: float,
):
    """
    Synchronous worker that enforces strict output size and returns a sequence of frames
    transformed according to the motion_type.
    """
    temp_dir = Path(temp_dir_path)
    n = max(1, len(frame_paths))
    for i, p in enumerate(frame_paths):
        progress = i / (n - 1) if n > 1 else 0.0
        eased = ease_in_out_quad(progress)

        img = _safe_open_image(p)
        if img is None:
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        # Start by forcing to target resolution to avoid aspect drift
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)

        transformed = img.copy()

        try:
            if motion_type in {"slow_zoom_in", "slow_zoom_out"}:
                t = eased if motion_type == "slow_zoom_in" else (1.0 - eased)
                final_w = int(round(width * (1.0 - zoom_intensity)))
                final_h = int(round(height * (1.0 - zoom_intensity)))
                current_w = int(round(width - (width - final_w) * t))
                current_h = int(round(height - (height - final_h) * t))

                left = int(round((width - current_w) / 2.0))
                top = int(round((height - current_h) / 2.0))
                right = left + current_w
                bottom = top + current_h

                left = max(0, left)
                top = max(0, top)
                right = min(width, right)
                bottom = min(height, bottom)

                transformed = img.crop((left, top, right, bottom))

            elif motion_type in {"pan_left", "pan_right"}:
                t = eased
                if motion_type == "pan_right":
                    t = 1.0 - eased
                crop_w = int(round(width * (1.0 - pan_intensity)))
                max_pan = max(0, width - crop_w)
                pan_offset = int(round(max_pan * t))
                left = pan_offset
                top = 0
                right = min(width, left + crop_w)
                bottom = height
                transformed = img.crop((left, top, right, bottom))

            # Final force-resize to target resolution (prevents off-by-one crops)
            if transformed.size != (width, height):
                transformed = transformed.resize((width, height), Image.LANCZOS)

        except Exception as exc:
            logger.warning(f"Motion transform failed for {p}: {exc}; using original resized frame")
            transformed = img.copy()
            if transformed.size != (width, height):
                transformed = transformed.resize((width, height), Image.LANCZOS)

        out_path = temp_dir / f"frame_{i:05d}.png"
        try:
            transformed.convert("RGB").save(out_path)
        except Exception as exc:
            logger.warning(f"Failed saving transformed frame {out_path}: {exc}")

    return


# -------------------------
# Video assembly
# -------------------------
async def create_video_from_frames(frame_paths: List[str], output_path: str, audio_path: Optional[str]) -> str:
    """
    Assemble frames into a video and sync to audio if provided.
    Ensures all frames are same size; will resize if mismatch detected.
    """
    if not frame_paths:
        raise ValueError("No frames provided to assemble video")

    def _sorted_frames(paths: List[str]) -> List[str]:
        try:
            def _key(p: str):
                stem = Path(p).stem
                parts = stem.split("_")
                for part in reversed(parts):
                    if part.isdigit():
                        return int(part)
                return stem
            return sorted(paths, key=_key)
        except Exception:
            return sorted(paths)

    def _assemble():
        sorted_paths = _sorted_frames(frame_paths)
        # Safety: validate sizes and force-resize to the first image size if needed
        try:
            with Image.open(sorted_paths[0]) as img:
                target_size = img.size
        except Exception:
            target_size = (DEFAULT_WIDTH, DEFAULT_HEIGHT)

        # Ensure all frames match target_size (in-place resize if necessary into tmp dir)
        needs_resize = False
        for p in sorted_paths:
            try:
                with Image.open(p) as img:
                    if img.size != target_size:
                        needs_resize = True
                        break
            except Exception:
                needs_resize = True
                break

        if needs_resize:
            tmp = tempfile.TemporaryDirectory(prefix="resize_frames_")
            tmpdir = Path(tmp.name)
            tmpdir.mkdir(parents=True, exist_ok=True)
            new_paths = []
            for i, p in enumerate(sorted_paths):
                try:
                    with Image.open(p) as img:
                        img = img.convert("RGB").resize(target_size, Image.LANCZOS)
                        np = tmpdir / f"frame_{i:05d}.png"
                        img.save(np)
                        new_paths.append(str(np))
                except Exception as exc:
                    logger.warning(f"Resizing failed for {p}: {exc}")
            sorted_paths = new_paths
        else:
            # ensure sorted_paths is a list of strings
            sorted_paths = list(sorted_paths)

        if audio_path:
            audio_clip = AudioFileClip(audio_path)
            audio_duration = max(audio_clip.duration, 0.001)
            fps = len(sorted_paths) / audio_duration if audio_duration > 0 else FPS
            fps = max(MIN_FPS, min(MAX_FPS, fps))
            logger.info(f"Calculated FPS: {fps:.2f} ({len(sorted_paths)} frames / {audio_duration:.2f}s)")
            clip = ImageSequenceClip(sorted_paths, fps=fps).set_audio(audio_clip)
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps, logger=None)
            clip.close()
            audio_clip.close()
            if needs_resize:
                tmp.cleanup()
        else:
            fps = max(MIN_FPS, min(MAX_FPS, FPS))
            clip = ImageSequenceClip(sorted_paths, fps=fps)
            clip.write_videofile(output_path, codec="libx264", audio=False, fps=fps, logger=None)
            clip.close()
            if needs_resize:
                tmp.cleanup()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _assemble)
    logger.info(f"Video ready: {output_path}")
    return output_path
