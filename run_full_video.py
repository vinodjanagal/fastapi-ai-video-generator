# run_full_video.py
# Phoenix V10.5 Full Production Runner (corrected, production-safe)
import argparse
import asyncio
import logging
import os
import shutil
from pathlib import Path
from tqdm import tqdm  # pip install tqdm

from app.orchestrator.phoenix_director import PhoenixDirector
from app.video_generator import apply_camera_motion, create_video_from_frames
from app.audio_pipeline import generate_narration_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [RUN_FULL] - %(message)s"
)
logger = logging.getLogger("run_full")


async def render_full_video(
    quote: str,
    audio_path: str,
    output_dir: str,
    output_video: str,
):
    root = str(Path(__file__).parent.resolve())
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # where we save per-scene previews
    previews_dir = output_dir_path / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Scene previews will be saved to: {previews_dir}")

    # keep TemporaryDirectory objects alive here to avoid GC cleanup
    motion_temp_dir_objects = []

    try:
        director = PhoenixDirector(
            project_root=root,
            output_dir=str(output_dir_path),
            mode="production",
            seed=42,
        )

        logger.info("=== Phoenix V10.5 FULL RENDER START ===")

        # 1) Generate raw frames (director only produces frames + metadata)
        video_data = await director.render_video(quote)
        if not video_data:
            raise RuntimeError("Director returned no data.")

        # DIRECTOR FIX: character sheet is NOT a scene
        scene_dirs = [
            d for d in video_data.get("scene_dirs", [])
            if not Path(d).name.startswith("character_sheet_")
        ]
        scene_defs = video_data.get("scenes", [])

        if not scene_dirs or not scene_defs:
            raise RuntimeError("Director returned no valid scenes.")

        if len(scene_dirs) != len(scene_defs):
            raise RuntimeError(
                f"Scene mismatch: {len(scene_dirs)} dirs vs {len(scene_defs)} defs"
            )

        # Master progress bar: one step per scene + one for assembly
        total_steps = len(scene_dirs) + 1
        all_frames = []
        conf = director._conf()

        with tqdm(total=total_steps, desc="Overall Video Progress", unit="step") as pbar:
            # 2) Per-scene camera motion + preview saving
            for idx, (scene_dir, scene_info) in enumerate(zip(scene_dirs, scene_defs), start=1):
                pbar.set_description(f"Scene {idx}/{len(scene_dirs)}")
                motion = scene_info.get("camera_motion", "static")

                # gather raw frames (safe PNG-only)
                raw_frames = sorted(
                    [
                        str(Path(scene_dir) / f)
                        for f in os.listdir(scene_dir)
                        if f.lower().endswith(".png")
                    ]
                )

                if not raw_frames:
                    raise RuntimeError(f"No frames found in {scene_dir}")

                # Save scene preview (middle frame)
                try:
                    mid_index = len(raw_frames) // 2
                    preview_src = raw_frames[mid_index]
                    preview_dst = previews_dir / f"scene_{idx:02d}_preview.png"
                    shutil.copy(preview_src, preview_dst)
                    logger.info(f"Saved preview for scene {idx} → {preview_dst}")
                except Exception as e:
                    logger.warning(f"Failed to save preview for scene {idx}: {e}")

                # Apply camera motion (returns TemporaryDirectory object + new frames)
                temp_obj, new_frames = await apply_camera_motion(
                    frame_paths=raw_frames,
                    motion_type=motion,
                    width=conf["width"],
                    height=conf["height"],
                    zoom_intensity=0.20,
                    pan_intensity=0.20,
                )

                # ABSOLUTE SAFETY FIX
                if not new_frames:
                    raise RuntimeError(f"Camera motion returned no frames for scene {idx}")

                if temp_obj:
                    motion_temp_dir_objects.append(temp_obj)

                all_frames.extend(new_frames)
                pbar.update(1)

            if not all_frames:
                raise RuntimeError("No frames after processing all scenes.")

            # 3) Assemble final video (sync to audio)
            pbar.set_description("Assembling final MP4")
            output_path = str(Path(output_video).resolve())

            final_video = await create_video_from_frames(
                frame_paths=all_frames,
                output_path=output_path,
                audio_path=audio_path,
            )

            pbar.update(1)

        logger.info(f"=== FINAL VIDEO READY ===\n{final_video}")
        return final_video

    finally:
        # explicit cleanup of temporary motion directories
        logger.info("--- Cleaning up temporary motion directories ---")
        for temp_dir in motion_temp_dir_objects:
            try:
                temp_dir.cleanup()
                logger.info(f"Cleaned up {getattr(temp_dir, 'name', str(temp_dir))}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup temp dir {getattr(temp_dir, 'name', str(temp_dir))}: {e}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phoenix V10.5 Full Cinematic Generator")
    parser.add_argument("-q", "--quote", required=True)
    parser.add_argument("-a", "--audio", required=False, help="Optional narration. If omitted, TTS is auto-generated.")
    parser.add_argument("-o", "--output-dir", default="render_output")
    parser.add_argument("-v", "--video", default="final_output.mp4")
    args = parser.parse_args()

    async def _cli():
        audio_path = args.audio
        if not audio_path:
            logger.info("No audio provided, generating narration via SpeechT5...")
            audio_path = await generate_narration_audio(args.quote)
            logger.info(f"Generated audio: {audio_path}")
        else:
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return

        await render_full_video(
            quote=args.quote,
            audio_path=audio_path,
            output_dir=args.output_dir,
            output_video=args.video,
        )

    asyncio.run(_cli())
