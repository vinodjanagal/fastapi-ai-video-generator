# file: app/video_generator.py
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import ImageSequenceClip, AudioFileClip
import logging


logger = logging.getLogger(__name__)

# ================= CONFIGURATION ================= #
PROJECT_ROOT = Path(__file__).parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
AUDIO_DIR = STATIC_DIR / "audio"
VIDEO_DIR = STATIC_DIR / "video"
FONT_DIR = STATIC_DIR / "fonts"

FPS = 24
WIDTH, HEIGHT = 1080, 1920
MARGIN = 100
BG_BLUR = 40
FONT_SIZE = 80
AUTHOR_FONT_SIZE = 52
LINE_SPACING = 26
SLIDE_PIXELS = 70
FADE_DURATION = 1.1

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
    top_rgb = tuple(int(top[i:i + 2], 16) for i in (1, 3, 5))
    bot_rgb = tuple(int(bottom[i:i + 2], 16) for i in (1, 3, 5))
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        r = y / height
        gradient[y, :] = (1 - r) * np.array(top_rgb) + r * np.array(bot_rgb)
    return Image.fromarray(gradient, "RGB").filter(ImageFilter.GaussianBlur(BG_BLUR)).convert("RGBA")

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.strip().split()
    if not words: return []
    lines = [words[0]]
    for word in words[1:]:
        test_line = f"{lines[-1]} {word}"
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]
        if width <= max_width: lines[-1] = test_line
        else: lines.append(word)
    return lines

def ease_out_cubic(t: float) -> float:
    return 1 - pow(1 - t, 3)

# ================= MAIN VIDEO GENERATOR ================= #
async def create_typography_video(text: str, audio_path: str, output_path: str, author_name: str = None, style: str = "dark_gradient"):
    # This function is unchanged. Your code is correct.
    logger.info(f"🎬 Creating typography video: {output_path} with style '{style}'")
    style_config = STYLE_PRESETS.get(style, STYLE_PRESETS["dark_gradient"])
    if style not in STYLE_PRESETS:
        logger.warning(f"Style '{style}' not found. Falling back to 'dark_gradient'.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_clip = AudioFileClip(str(audio_path))
    duration = max(audio_clip.duration, 1.5)
    total_frames = int(duration * FPS)
    try:
        font = ImageFont.truetype(str(style_config["font_path"]), FONT_SIZE)
        author_font = ImageFont.truetype(str(style_config["author_font_path"]), AUTHOR_FONT_SIZE) if author_name else None
    except Exception as e:
        raise FileNotFoundError(f"Font missing for style '{style}': {style_config['font_path']}. Error: {e}")
    bg_img = gradient_background(WIDTH, HEIGHT, style_config["bg_colors"][0], style_config["bg_colors"][1])
    lines = wrap_text(text, font, WIDTH - 2 * MARGIN)
    if not lines: raise ValueError("Text is empty.")
    stagger = max(0.6, (duration - FADE_DURATION) / len(lines))
    timings = [(i * stagger, i * stagger + FADE_DURATION) for i in range(len(lines))]
    frame_paths = []
    with tempfile.TemporaryDirectory(prefix="frames_") as tmpdir:
        tmp_dir = Path(tmpdir)
        logger.info(f"Rendering {total_frames} frames...")
        for i in range(total_frames):
            t = i / FPS
            frame = bg_img.copy()
            layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            bboxes = [font.getbbox(line) for line in lines]
            total_h = sum(bb[3] - bb[1] + LINE_SPACING for bb in bboxes) - LINE_SPACING
            y = (HEIGHT - total_h) / 2
            for idx, line in enumerate(lines):
                start, end = timings[idx]
                progress = ease_out_cubic(min(max((t - start) / (end - start), 0), 1))
                opacity = int(255 * progress)
                offset = int(SLIDE_PIXELS * (1 - progress))
                bb = bboxes[idx]
                lw = bb[2] - bb[0]
                x = (WIDTH - lw) / 2
                text_fill_color = (*style_config["text_color"][:3], opacity)
                draw.text((x, y + offset), line, font=font, fill=text_fill_color)
                y += bb[3] - bb[1] + LINE_SPACING
            if author_name and author_font:
                at = timings[-1][0]
                ap = ease_out_cubic(min(max((t - at) / FADE_DURATION, 0), 1))
                a_op = int(255 * ap)
                ab = author_font.getbbox(f"- {author_name}")
                ax = WIDTH - MARGIN - (ab[2] - ab[0])
                ay = HEIGHT - MARGIN - (ab[3] - ab[1])
                author_fill_color = (*style_config["author_color"][:3], a_op)
                draw.text((ax, ay), f"- {author_name}", font=author_font, fill=author_fill_color)
            frame = Image.alpha_composite(frame, layer).convert("RGB")
            frame_path = tmp_dir / f"frame_{i:05}.png"
            frame.save(frame_path, optimize=True)
            frame_paths.append(str(frame_path))
        logger.info("🔗 Composing final MP4")
        clip = ImageSequenceClip(frame_paths, fps=FPS)
        clip = clip.set_audio(audio_clip) # Corrected line
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: clip.write_videofile(
            str(output_path), codec="libx264", audio_codec="aac", fps=FPS,
            preset="medium", threads=2, logger=None
        ))
        clip.close()
    logger.info(f"✅ Video ready at: {output_path}")
    return str(output_path)

async def create_video_from_frames(
    frame_paths: List[str],
    output_path: str,
    audio_path: str  # REQUIRED
) -> str:
    """
    Syncs frames to audio: 1 frame per word, no hold, no freeze.
    FPS = num_frames / audio_duration
    """
    logger.info(f"Syncing {len(frame_paths)} frames to audio: {audio_path}")

    if not frame_paths:
        raise ValueError("No frames")
    if not audio_path:
        raise ValueError("Audio path required")

    def _assemble():
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        if audio_duration <= 0:
            raise ValueError("Audio duration is zero")

        # YOUR GENIUS: FPS = frames / audio_duration
        fps = len(frame_paths) / audio_duration
        logger.info(f"Calculated FPS: {fps:.2f} ({len(frame_paths)} frames / {audio_duration:.2f}s)")

        video_clip = ImageSequenceClip(sorted(frame_paths), fps=fps)
        video_clip.audio = audio_clip  # Direct assignment

        video_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            logger=None
        )

        video_clip.close()
        audio_clip.close()

    await asyncio.get_event_loop().run_in_executor(None, _assemble)
    logger.info(f"Video ready: {output_path}")
    return output_path