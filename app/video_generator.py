# file: app/video_generator.py
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import ImageSequenceClip, AudioFileClip
import logging
import math

logger = logging.getLogger(__name__)

# ================= CONFIGURATION ================= #
PROJECT_ROOT = Path(__file__).parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
AUDIO_DIR = STATIC_DIR / "audio"
VIDEO_DIR = STATIC_DIR / "video"
FONT_DIR = STATIC_DIR / "fonts"

# NOTE: Keep these defaults but the orchestrator may generate 512x512 frames for AnimateDiff.
FPS = 24
WIDTH, HEIGHT = 1080, 1920
MARGIN = 100
BG_BLUR = 40
FONT_SIZE = 80
AUTHOR_FONT_SIZE = 52
LINE_SPACING = 26
SLIDE_PIXELS = 70
FADE_DURATION = 1.1
MAX_FPS = 60
MIN_FPS = 1

STYLE_PRESETS: Dict[str, Dict[str, Any]] = {
    "dark_gradient": {
        "bg_colors": ("#0f1724", "#2b2f4a"),
        "text_color": (255, 255, 255, 255),
        "author_color": (255, 255, 0, 255),
        "font_path": FONT_DIR / "Inter Bold 700.otf",
        "author_font_path": FONT_DIR / "Inter Bold Italic 700.otf",
    },
    "yellow_punch": {
        "bg_colors": ("#facc15", "#eab308"),
        "text_color": (0, 0, 0, 255),
        "author_color": (30, 30, 30, 255),
        "font_path": FONT_DIR / "Merriweather-Bold.ttf",
        "author_font_path": FONT_DIR / "Merriweather-Italic.ttf",
    },
    "blue_calm": {
        "bg_colors": ("#3b82f6", "#60a5fa"),
        "text_color": (255, 255, 255, 255),
        "author_color": (230, 230, 250, 255),
        "font_path": FONT_DIR / "Lato-Regular.ttf",
        "author_font_path": FONT_DIR / "Lato-Italic.ttf",
    }
}


# ================= UTILS ================= #
def gradient_background(width: int, height: int, top: str, bottom: str) -> Image.Image:
    """Create a blurred gradient background (RGBA)."""
    top_rgb = tuple(int(top[i:i + 2], 16) for i in (1, 3, 5))
    bot_rgb = tuple(int(bottom[i:i + 2], 16) for i in (1, 3, 5))
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        r = y / max(height - 1, 1)
        gradient[y, :] = (1 - r) * np.array(top_rgb) + r * np.array(bot_rgb)
    pil = Image.fromarray(gradient.astype(np.uint8), "RGB")
    try:
        return pil.filter(ImageFilter.GaussianBlur(BG_BLUR)).convert("RGBA")
    except Exception:
        # fallback if GaussianBlur fails for any reason
        return pil.convert("RGBA")


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Break text into lines that fit within max_width using font.getbbox for robust measurement."""
    words = text.strip().split()
    if not words:
        return []
    lines = [words[0]]
    for word in words[1:]:
        test_line = f"{lines[-1]} {word}"
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            lines[-1] = test_line
        else:
            lines.append(word)
    return lines


def ease_in_out_quad(t: float) -> float:
    """Smooth ease in/out (0..1)."""
    t = max(0.0, min(1.0, t))
    return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2


def ease_out_cubic(t: float) -> float:
    """Ease-out cubic (0..1)."""
    t = max(0.0, min(1.0, t))
    return 1 - pow(1 - t, 3)


# -------------------------
# apply_camera_motion
# -------------------------
async def apply_camera_motion(
    frame_paths: List[str],
    motion_type: str,
    width: int = 512,
    height: int = 512,
    zoom_intensity: float = 0.20,
    pan_intensity: float = 0.15,
) -> Tuple[Optional[tempfile.TemporaryDirectory], List[str]]:
    """
    Apply camera motion to frames and write transformed frames into a NEW temporary
    directory backed by tempfile.TemporaryDirectory(). Returns (tempdir_obj, list_of_paths).
    Caller is responsible for keeping tempdir_obj alive while using the files, and
    for calling tempdir_obj.cleanup() (the runner will do that).
    """
    if not frame_paths or motion_type == "static":
        return None, frame_paths

    num_frames = len(frame_paths)
    logger.info(f"Applying '{motion_type}' motion to {num_frames} frames.")

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="motion_frames_")
    temp_dir_path = Path(temp_dir_obj.name)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    # Run the CPU-bound frame processing in a thread pool
    await loop.run_in_executor(
        None,
        _process_frames_for_motion,  # synchronous helper you already have in file
        frame_paths,
        str(temp_dir_path),
        motion_type,
        width,
        height,
        zoom_intensity,
        pan_intensity,
    )

    # Collect and return frame paths in numeric order
    new_frame_paths = sorted([str(p) for p in temp_dir_path.glob("frame_*.png")])
    return temp_dir_obj, new_frame_paths


def _safe_open_image(path: str) -> Optional[Image.Image]:
    """Open image safely and convert to RGBA; returns None on failure (logged)."""
    try:
        img = Image.open(path)
        img = img.convert("RGBA")
        return img
    except Exception as e:
        logger.warning(f"Failed to open image '{path}': {e}")
        return None


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
    Synchronous worker run in executor. Writes transformed frames into temp_dir_path.
    This function must use integer crop coordinates for Pillow compatibility.
    """
    temp_path = Path(temp_dir_path)
    num_frames = max(1, len(frame_paths))

    for i, frame_path in enumerate(frame_paths):
        progress = i / (num_frames - 1) if num_frames > 1 else 0.0
        eased = ease_in_out_quad(progress)

        img = _safe_open_image(frame_path)
        if img is None:
            # create a blank placeholder if file couldn't be opened to keep sequence length
            img = Image.new("RGBA", (width, height), (0, 0, 0, 255))

        # Ensure image has expected size; if not, resize with aspect-preserving center crop
        try:
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
        except Exception:
            img = img.convert("RGBA").resize((int(width), int(height)), Image.LANCZOS)

        transformed_img = img.copy()

        try:
            if motion_type in {"slow_zoom_in", "slow_zoom_out"}:
                t = eased
                if motion_type == "slow_zoom_out":
                    t = 1.0 - eased

                # compute current crop size (integers)
                final_w = int(round(width * (1.0 - zoom_intensity)))
                final_h = int(round(height * (1.0 - zoom_intensity)))
                current_w = int(round(width - (width - final_w) * t))
                current_h = int(round(height - (height - final_h) * t))

                left = int(round((width - current_w) / 2.0))
                top = int(round((height - current_h) / 2.0))
                right = left + current_w
                bottom = top + current_h

                # clamp coords
                left = max(0, left)
                top = max(0, top)
                right = min(width, right)
                bottom = min(height, bottom)

                transformed_img = img.crop((left, top, right, bottom)).resize((width, height), Image.LANCZOS)

            elif motion_type in {"pan_left", "pan_right"}:
                t = eased
                if motion_type == "pan_right":
                    t = 1.0 - eased

                crop_w = int(round(width * (1.0 - pan_intensity)))
                crop_h = height
                max_pan_offset = max(0, width - crop_w)
                pan_offset = int(round(max_pan_offset * t))

                left = pan_offset
                top = 0
                right = left + crop_w
                bottom = crop_h

                # clamp
                left = max(0, left)
                right = min(width, right)

                transformed_img = img.crop((left, top, right, bottom)).resize((width, height), Image.LANCZOS)

            else:
                # Unknown motion_type => pass-through
                transformed_img = img.copy()

        except Exception as e:
            logger.warning(f"Motion transform failed for frame '{frame_path}': {e}. Using original frame.")
            transformed_img = img.copy()

        out_path = temp_path / f"frame_{i:05d}.png"
        try:
            # ensure parent exists
            out_path.parent.mkdir(parents=True, exist_ok=True)
            transformed_img.save(out_path)
        except Exception as e:
            logger.warning(f"Failed saving transformed frame {out_path}: {e}")

    return


# ================= MAIN VIDEO GENERATOR ================= #
async def create_typography_video(
    text: str,
    audio_path: str,
    output_path: str,
    author_name: Optional[str] = None,
    style: str = "dark_gradient",
) -> str:
    """
    Create a typography-style MP4 with text over gradient background, synced to audio_path.
    This is CPU-heavy; run inside an executor if called from an async server.
    """
    logger.info(f"🎬 Creating typography video: {output_path} with style '{style}'")
    style_config = STYLE_PRESETS.get(style, STYLE_PRESETS["dark_gradient"])
    if style not in STYLE_PRESETS:
        logger.warning(f"Style '{style}' not found. Falling back to 'dark_gradient'.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_clip = AudioFileClip(str(audio_path))
    duration = max(audio_clip.duration, 1.5)
    total_frames = max(1, int(math.ceil(duration * FPS)))

    try:
        font = ImageFont.truetype(str(style_config["font_path"]), FONT_SIZE)
        author_font = ImageFont.truetype(str(style_config["author_font_path"]), AUTHOR_FONT_SIZE) if author_name else None
    except Exception as e:
        raise FileNotFoundError(f"Font missing for style '{style}': {style_config['font_path']}. Error: {e}")

    bg_img = gradient_background(WIDTH, HEIGHT, style_config["bg_colors"][0], style_config["bg_colors"][1])
    lines = wrap_text(text, font, WIDTH - 2 * MARGIN)
    if not lines:
        raise ValueError("Text is empty or font metrics failed to layout text.")

    stagger = max(0.6, (duration - FADE_DURATION) / max(len(lines), 1))
    timings = [(i * stagger, i * stagger + FADE_DURATION) for i in range(len(lines))]

    frame_paths: List[str] = []
    with tempfile.TemporaryDirectory(prefix="frames_") as tmpdir:
        tmp_dir = Path(tmpdir)
        logger.info(f"Rendering {total_frames} frames (typography)...")
        for i in range(total_frames):
            t = i / float(FPS)
            frame = bg_img.copy()
            layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            bboxes = [font.getbbox(line) for line in lines]
            total_h = sum((bb[3] - bb[1]) + LINE_SPACING for bb in bboxes) - LINE_SPACING
            y = int((HEIGHT - total_h) / 2)

            for idx_line, line in enumerate(lines):
                start, end = timings[idx_line]
                denom = max((end - start), 1e-6)
                progress = ease_out_cubic(min(max((t - start) / denom, 0.0), 1.0))
                opacity = int(255 * progress)
                offset = int(SLIDE_PIXELS * (1 - progress))
                bb = bboxes[idx_line]
                lw = bb[2] - bb[0]
                x = int((WIDTH - lw) / 2)
                text_fill_color = (*style_config["text_color"][:3], opacity)
                draw.text((x, y + offset), line, font=font, fill=text_fill_color)
                y += (bb[3] - bb[1]) + LINE_SPACING

            if author_name and author_font:
                at = timings[-1][0]
                ap = ease_out_cubic(min(max((t - at) / FADE_DURATION, 0.0), 1.0))
                a_op = int(255 * ap)
                ab = author_font.getbbox(f"- {author_name}")
                ax = int(WIDTH - MARGIN - (ab[2] - ab[0]))
                ay = int(HEIGHT - MARGIN - (ab[3] - ab[1]))
                author_fill_color = (*style_config["author_color"][:3], a_op)
                draw.text((ax, ay), f"- {author_name}", font=author_font, fill=author_fill_color)

            frame = Image.alpha_composite(frame, layer).convert("RGB")
            frame_path = tmp_dir / f"frame_{i:05}.png"
            frame.save(frame_path, optimize=True)
            frame_paths.append(str(frame_path))

        logger.info("🔗 Composing final MP4 (typography)")
        clip = ImageSequenceClip(frame_paths, fps=FPS)
        clip = clip.set_audio(audio_clip)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: clip.write_videofile(
            str(output_path), codec="libx264", audio_codec="aac", fps=FPS,
            preset="medium", threads=2, logger=None
        ))
        clip.close()
        audio_clip.close()

    logger.info(f"✅ Typography video ready at: {output_path}")
    return str(output_path)


async def create_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    audio_path: Optional[str],  # optional now — caller may pass None for silent video
) -> str:
    """
    Assemble frames into a video and sync to audio if provided.
    FPS is computed as len(frames) / audio_duration when audio provided (clamped).
    If audio_path is None, uses a default FPS constant.
    """

    logger.info(f"Syncing {len(frame_paths)} frames to audio: {audio_path}")

    if not frame_paths:
        raise ValueError("No frames provided to create_video_from_frames")

    def _sorted_frames(frame_paths_list: List[str]) -> List[str]:
        """
        Sort frames in numeric order using the trailing integer in the filename.
        Supports names like 'frame_00001.png' or 'scene_1_frame_0001.png'.
        Falls back to lexical sort if numeric parse fails.
        """
        def _key(p: str):
            stem = Path(p).stem
            parts = stem.split("_")
            # find last numeric segment
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            # fallback: return stem for lexical sort
            return stem

        try:
            return sorted(frame_paths_list, key=_key)
        except Exception:
            return sorted(frame_paths_list)

    def _assemble():
        if audio_path:
            audio_clip = AudioFileClip(audio_path)
            audio_duration = max(audio_clip.duration, 0.001)
            fps = len(frame_paths) / audio_duration if audio_duration > 0 else FPS
            # safety clamp
            fps = max(MIN_FPS, min(MAX_FPS, fps))
            logger.info(f"Calculated FPS: {fps:.2f} ({len(frame_paths)} frames / {audio_duration:.2f}s)")

            clip = ImageSequenceClip(_sorted_frames(frame_paths), fps=fps)
            clip = clip.set_audio(audio_clip)
            clip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                fps=fps,
                logger=None
            )
            clip.close()
            audio_clip.close()
        else:
            # silent video -> use default FPS constant and clamp
            fps = max(MIN_FPS, min(MAX_FPS, FPS))
            clip = ImageSequenceClip(_sorted_frames(frame_paths), fps=fps)
            clip.write_videofile(output_path, codec="libx264", audio=False, fps=fps, logger=None)
            clip.close()

    await asyncio.get_event_loop().run_in_executor(None, _assemble)
    logger.info(f"Video ready: {output_path}")
    return output_path
