# file: app/tasks.py
import os
import logging
from pathlib import Path
from typing import List

from app.database import AsyncSessionLocal
from app import crud, models
from app.video_generator import apply_camera_motion, create_video_from_frames
from app.audio_pipeline import generate_narration_audio  # new helper
from app.utils import run_subprocess_streamed  # common utility if you need it here

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "static", "videos")
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


async def process_semantic_video_generation(video_id: int):
    """
    Phoenix V10.1 – Full Production Pipeline (task)
    1) Ensure audio exists (generate if missing)
    2) Director.render_video -> raw scene frames (no motion)
    3) Apply camera motion (post-processing)
    4) Assemble final MP4 with narration audio
    5) Save + update DB
    """
    logger.info(f"TASK: Starting full Phoenix V10.1 render for video_id={video_id}")

    # --------------------------------------------------
    # 1) Fetch DB record: quote + audio
    # --------------------------------------------------
    async with AsyncSessionLocal() as db:
        video_record = await crud.get_video(db, video_id=video_id)
        if not video_record:
            raise FileNotFoundError(f"Video {video_id} not found")

        quote_text = video_record.quote.text
        audio_path = video_record.audio_path

    # If audio missing, generate now and persist
    if not audio_path or not os.path.exists(audio_path):
        logger.info("No narration audio found for job — generating via SpeechT5...")
        audio_path = await generate_narration_audio(quote_text)
        # Persist audio path
        async with AsyncSessionLocal() as db:
            vr = await crud.get_video(db, video_id=video_id)
            if vr:
                vr.audio_path = audio_path
                await crud.update_video_record(db, video=vr)
                await db.commit()
        logger.info(f"Generated and stored audio: {audio_path}")

    # --------------------------------------------------
    # 2) Generate frames using PhoenixDirector
    # NOTE: import PhoenixDirector lazily to avoid circular import at module load time
    # --------------------------------------------------
    from app.orchestrator.phoenix_director import PhoenixDirector

    director = PhoenixDirector(
        project_root=PROJECT_ROOT,
        output_dir=VIDEO_OUTPUT_DIR,
        mode="production",
        seed=42,
    )

    video_data = await director.render_video(quote_text)
    if not video_data:
        raise RuntimeError("PhoenixDirector.render_video returned NONE (failure).")

    scene_dirs = video_data.get("scene_dirs", [])
    scene_defs = video_data.get("scenes", [])
    if not scene_dirs or not scene_defs:
        raise RuntimeError("PhoenixDirector returned no scenes or empty directories.")

    # --------------------------------------------------
    # 3) Apply camera motion PER SCENE
    # --------------------------------------------------
    logger.info("TASK: Applying digital camera motion to frames...")

    all_frames: List[str] = []

    for idx, (scene_dir, scene_meta) in enumerate(zip(scene_dirs, scene_defs), start=1):
        motion = scene_meta.get("camera_motion", "static")
        logger.info(f"[Scene {idx}] Motion = {motion}")

        frames = sorted([
            str(Path(scene_dir) / f)
            for f in os.listdir(scene_dir)
            if f.lower().endswith(".png")
        ])

        if not frames:
            logger.warning(f"No frames found in {scene_dir}")
            continue

        temp_obj, transformed_frames = await apply_camera_motion(
            frame_paths=frames,
            motion_type=motion,
            width=512,
            height=512,
            zoom_intensity=0.20,
            pan_intensity=0.20,
        )

        all_frames.extend(transformed_frames)
        # Note: temp_obj should be kept alive until cleanup; director/task caller may clean up later.

    if not all_frames:
        raise RuntimeError("After camera motion, no frames remain! Something failed.")

    # --------------------------------------------------
    # 4) Assemble final MP4 synced to audio
    # --------------------------------------------------
    final_video_path = os.path.join(VIDEO_OUTPUT_DIR, f"final_video_{video_id}.mp4")
    logger.info("TASK: Assembling the final MP4 with audio...")

    final_path = await create_video_from_frames(
        frame_paths=all_frames,
        output_path=final_video_path,
        audio_path=audio_path,
    )

    # --------------------------------------------------
    # 5) Update DB with final video path
    # --------------------------------------------------
    async with AsyncSessionLocal() as db:
        video_to_update = await crud.get_video(db, video_id=video_id)
        await crud.update_video_record(
            db,
            video=video_to_update,
            status=models.VideoStatus.COMPLETED,
            video_path=final_path,
        )
        await db.commit()

    logger.info(f"TASK: Phoenix V10.1 finished. Final video saved at: {final_path}")
    return final_path
