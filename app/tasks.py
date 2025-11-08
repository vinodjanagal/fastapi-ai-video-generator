import os
import sys
import json
import uuid
import shutil
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from sqlalchemy import Float
from app.database import AsyncSessionLocal
from app import crud, models
from app import audio_generator_gtts as audio_generator
from app import video_generator

logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "static", "videos")
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# ---------- Tunables ----------
SPEECH_TIMEOUT = 86_400     # 24h
AD_FRAME_TIMEOUT = 86_400   # 24h
STORYBOARD_TIMEOUT = 86_400 # 5m is plenty for LLM call via CLI wrapper
TIMESTAMPS_TIMEOUT = 86_400 # 10m if you later split timestamps engine out
DEFAULT_FPS = 8

# ---------- Async subprocess helper (non-blocking, streamed logs) ----------
async def run_subprocess_streamed(
    cmd: List[str],
    timeout: Optional[int] = None,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
) -> Tuple[int, str, str]:
    """
    Run a subprocess without blocking the event loop.
    Streams stdout/stderr to logs and also returns full captured text.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=cwd,
    )
    stdout_chunks, stderr_chunks = [], []
    async def _pipe(stream, sink, log_fn):
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode(errors="ignore")
            sink.append(text)
            log_fn(text.rstrip("\n"))
    try:
        await asyncio.wait_for(
            asyncio.gather(
                _pipe(proc.stdout, stdout_chunks, logger.info),
                _pipe(proc.stderr, stderr_chunks, logger.warning),
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"Subprocess timed out after {timeout}s: {' '.join(cmd)}")
    rc = await proc.wait()
    return rc, "".join(stdout_chunks), "".join(stderr_chunks)

# ---------- Existing light pipeline (gTTS) ----------
async def process_video_generation(video_id: int, style: str):
    logger.info(f"TASK STARTED for video_id={video_id}, style='{style}'")
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    async with AsyncSessionLocal() as db:
        try:
            video_record = await crud.get_video(db, video_id=video_id)
            if not video_record:
                logger.error(f"Video {video_id} not found.")
                return
           
            # INITIAL STATUS
            video_record.status = models.VideoStatus.PROCESSING
            video_record.progress = 0.0
            await db.merge(video_record)
            await db.commit()
            # ---------- PROGRESS HELPER ----------
            async def update_progress(percent: float, step_name: str):
                video_record.progress = round(percent, 1)
                video_record.status = f"PROCESSING: {step_name}"
                await db.merge(video_record)
                await db.commit()
                logger.info(f"Progress: {video_record.progress}% - {step_name}")
            await crud.update_video_record(db, video=video_record, status=models.VideoStatus.PROCESSING)
            await db.commit()
            quote = video_record.quote
            uid = uuid.uuid4()
            audio_path = os.path.join(VIDEO_OUTPUT_DIR, f"quote_{quote.id}_{uid}.mp3")
            video_path = os.path.join(VIDEO_OUTPUT_DIR, f"quote_{quote.id}_{uid}.mp4")
            logger.info(f"Generating audio -> {audio_path}")
            await audio_generator.generate_audio_from_text(text=quote.text, output_filename=audio_path)
            logger.info(f"Generating video -> {video_path}")
            final_video_path = await video_generator.create_typography_video(
                audio_path=audio_path,
                text=quote.text,
                author_name=quote.author.name,
                output_path=video_path,
                style=style,
            )
            await crud.update_video_record(
                db, video=video_record, status=models.VideoStatus.COMPLETED, video_path=final_video_path
            )
            await db.commit()
            logger.info(f"TASK SUCCESS for video {video_id}. Path: {final_video_path}")
        except Exception as e:
            logger.error(f"TASK FAILED for video {video_id}: {e}", exc_info=True)
            await db.rollback()
            vr = await crud.get_video(db, video_id=video_id)
            if vr:
                await crud.update_video_record(db, video=vr, status=models.VideoStatus.FAILED, error_message=str(e))
                await db.commit()
        finally:
            vr_final = await crud.get_video(db, video_id=video_id)
            if vr_final and vr_final.status == models.VideoStatus.FAILED:
                if audio_path and os.path.exists(audio_path):
                    try: os.remove(audio_path); logger.info(f"Removed failed audio: {audio_path}")
                    except OSError as err: logger.error(f"Remove audio error: {err}")
                if video_path and os.path.exists(video_path):
                    try: os.remove(video_path); logger.info(f"Removed failed video: {video_path}")
                    except OSError as err: logger.error(f"Remove video error: {err}")
            elif vr_final and vr_final.status == models.VideoStatus.COMPLETED:
                if audio_path and os.path.exists(audio_path):
                    try: os.remove(audio_path); logger.info(f"Removed intermediate audio: {audio_path}")
                    except OSError as err: logger.error(f"Remove intermediate audio error: {err}")
            logger.info(f"TASK FINISHED for video_id={video_id}")

# ---------- SpeechT5 external engine pipeline ----------
async def process_video_generation_speecht5(video_id: int, voice_name: str, video_style: str):
    logger.info(f"TASK STARTED (SpeechT5) for video_id={video_id}, voice='{voice_name}'")
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    async with AsyncSessionLocal() as db:
        try:
            video_record = await crud.get_video(db, video_id=video_id)
            if not video_record:
                logger.error(f"Video {video_id} not found.")
                return
            await crud.update_video_record(db, video=video_record, status=models.VideoStatus.PROCESSING)
            await db.commit()
            quote = video_record.quote
            uid = uuid.uuid4()
            audio_path = os.path.join(VIDEO_OUTPUT_DIR, f"quote_{quote.id}_{uid}.mp3")
            video_path = os.path.join(VIDEO_OUTPUT_DIR, f"quote_{quote.id}_{uid}.mp4")
            script_path = os.path.join(PROJECT_ROOT, "app", "video_engines", "speecht5_engine.py")
            cmd = [sys.executable, script_path, "--text", quote.text, "--output-path", audio_path, "--voice", voice_name]
            rc, out, err = await run_subprocess_streamed(
                cmd, timeout=SPEECH_TIMEOUT, env=os.environ.copy(), cwd=PROJECT_ROOT
            )
            if rc != 0:
                raise RuntimeError(f"speecht5_engine.py failed (rc={rc}). Stderr:\n{err}")
            try:
                result = json.loads(out)
                if result.get("status") != "COMPLETED":
                    raise ValueError(result.get("error", "Unknown SpeechT5 error"))
            except Exception as e:
                raise RuntimeError(f"Invalid SpeechT5 stdout. Stdout:\n{out}\nError: {e}")
            logger.info(f"Audio OK -> {audio_path}")
            final_video_path = await video_generator.create_typography_video(
                audio_path=audio_path,
                text=quote.text,
                author_name=quote.author.name,
                output_path=video_path,
                style=video_style,
            )
            await crud.update_video_record(
                db, video=video_record, status=models.VideoStatus.COMPLETED, video_path=final_video_path
            )
            await db.commit()
            logger.info(f"TASK SUCCESS (SpeechT5) for video {video_id}. Path: {final_video_path}")
        except Exception as e:
            logger.error(f"TASK FAILED (SpeechT5) for video {video_id}: {e}", exc_info=True)
            await db.rollback()
            vr = await crud.get_video(db, video_id=video_id)
            if vr:
                await crud.update_video_record(db, video=vr, status=models.VideoStatus.FAILED, error_message=str(e))
                await db.commit()
        finally:
            vr_final = await crud.get_video(db, video_id=video_id)
            if vr_final and vr_final.status == models.VideoStatus.FAILED:
                if audio_path and os.path.exists(audio_path):
                    try: os.remove(audio_path)
                    except OSError as err: logger.error(f"Audio cleanup error: {err}")
            elif vr_final and vr_final.status == models.VideoStatus.COMPLETED:
                if audio_path and os.path.exists(audio_path):
                    try: os.remove(audio_path)
                    except OSError as err: logger.error(f"Intermediate audio cleanup error: {err}")
            logger.info(f"TASK FINISHED (SpeechT5) for video_id={video_id}")

# ---------- AnimateDiff full pipeline (single prompt) ----------
async def process_video_generation_animatediff(video_id: int, prompt: str, voice_name: str):
    logger.info(f"TASK STARTED (AnimateDiff Full) for video_id={video_id}")
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    frames_dir: Optional[str] = None
    async with AsyncSessionLocal() as db:
        try:
            video_record = await crud.get_video(db, video_id=video_id)
            if not video_record:
                raise FileNotFoundError(f"Video record {video_id} not found.")
           
            # Set initial processing state
            video_record.status = models.VideoStatus.PROCESSING
            video_record.progress = 0.0
            await db.merge(video_record)
            await db.commit()

            quote_text = video_record.quote.text
            uid = uuid.uuid4()
            # ---------- PROGRESS HELPER ----------
            async def update_progress(percent: float, step_name: str):
                video_record.progress = round(percent, 1)
                video_record.status = f"PROCESSING: {step_name}"
                await db.merge(video_record)
                await db.commit()
                logger.info(f"Progress: {video_record.progress}% - {step_name}")
            # -------------------------------------------------
            audio_path = os.path.join(VIDEO_OUTPUT_DIR, f"audio_{uid}.mp3")
            video_path = os.path.join(VIDEO_OUTPUT_DIR, f"video_{uid}.mp4")
            frames_dir = os.path.join(VIDEO_OUTPUT_DIR, f"frames_{uid}")
            # Stage 1: Speech
            audio_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "speecht5_engine.py")
            rc, out, err = await run_subprocess_streamed(
                [sys.executable, audio_script, "--text", quote_text, "--output-path", audio_path, "--voice", voice_name],
                timeout=SPEECH_TIMEOUT, env=os.environ.copy(), cwd=PROJECT_ROOT
            )
            if rc != 0:
                raise RuntimeError(f"speecht5_engine.py failed (rc={rc}). Stderr:\n{err}")
            try:
                audio_result = json.loads(out)
                if audio_result.get("status") != "COMPLETED":
                    raise ValueError(audio_result.get("error", "Speech error"))
            except Exception as e:
                raise RuntimeError(f"Invalid SpeechT5 stdout.\n{out}\nError:{e}")
            # Stage 2: Frames
            frame_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "animate_diff_engine.py")
           
            # === THE ONLY CHANGE IS HERE: Golden Configuration Applied ===
            frame_cmd = [
                sys.executable,
                frame_script,
                "--prompt", prompt,
                "--output-dir", frames_dir,
                "--num-steps", "4",
                "--guidance-scale", "1.5"
            ]
            rc2, out2, err2 = await run_subprocess_streamed(
                frame_cmd,
                timeout=AD_FRAME_TIMEOUT, env=os.environ.copy(), cwd=PROJECT_ROOT
            )
            if rc2 != 0:
                raise RuntimeError(f"animate_diff_engine.py failed (rc={rc2}). Stderr:\n{err2}")
            try:
                frame_result = json.loads(out2)
                if frame_result.get("status") != "COMPLETED":
                    raise ValueError(frame_result.get("error", "AnimateDiff error"))
                frame_paths = frame_result.get("frame_paths", [])
                if not frame_paths:
                    raise ValueError("No frame_paths from AnimateDiff.")
            except Exception as e:
                raise RuntimeError(f"Invalid AnimateDiff stdout.\n{out2}\nError:{e}")
            # Stage 3: Assemble
            final_video_path = await video_generator.create_video_from_frames(
                frame_paths=frame_paths, output_path=video_path, fps=DEFAULT_FPS, audio_path=audio_path
            )
            await crud.update_video_record(
                db, video=video_record, status=models.VideoStatus.COMPLETED, video_path=final_video_path
            )
            await db.commit()
            logger.info(f"TASK SUCCESS (AnimateDiff Full) for video {video_id}. Path: {final_video_path}")
        except Exception as e:
            logger.error(f"TASK FAILED (AnimateDiff Full) for video {video_id}: {e}", exc_info=True)
            await db.rollback()
            vr = await crud.get_video(db, video_id=video_id)
            if vr:
                await crud.update_video_record(db, video=vr, status=models.VideoStatus.FAILED, error_message=str(e))
                await db.commit()
        finally:
            logger.info(f"Cleanup intermediates for job {video_id}...")
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except OSError as err: logger.error(f"Audio cleanup error: {err}")
            if frames_dir and os.path.exists(frames_dir):
                try: shutil.rmtree(frames_dir)
                except Exception as e_clean: logger.error(f"Frames cleanup error: {e_clean}")
            logger.info(f"TASK FINISHED (AnimateDiff Full) for video_id={video_id}")

# ---------- NEW: Semantic Orchestrator ----------
async def process_semantic_video_generation(video_id: int):
    """
    Semantic pipeline:
      1) Generate master audio (and timestamps) with SpeechT5 engine.
      2) Generate storyboard with durations using storyboard_engine.py.
      3) For each scene: run animate_diff_engine.py to create frames.
      4) Concatenate all frames in order, assemble with master audio.
      5) Update DB and cleanup.
    Expects engines to output JSON on stdout:
      - speecht5_engine.py -> {"status":"COMPLETED","file": "...", "timestamps_file":"..."}  (timestamps_file optional)
      - storyboard_engine.py -> {"status":"COMPLETED","storyboard":[{scene_description,animation_prompt,start_time,end_time,duration},...]}
      - animate_diff_engine.py -> {"status":"COMPLETED","frame_paths":[...]}
    """
    logger.info(f"ORCHESTRATOR START video_id={video_id}")
    audio_path: Optional[str] = None
    timestamps_path: Optional[str] = None
    final_video_path: Optional[str] = None
    scene_temp_dirs: List[str] = []
    all_frame_paths: List[str] = []
    async with AsyncSessionLocal() as db:
        try:
            video_record = await crud.get_video(db, video_id=video_id)
            if not video_record:
                raise FileNotFoundError(f"Video {video_id} not found.")

            # INITIAL STATUS
            video_record.status = models.VideoStatus.PROCESSING
            video_record.progress = 0.0
            await db.merge(video_record)
            await db.commit()

            # ---------- PROGRESS HELPER ----------
            async def update_progress(percent: float, step_name: str):
                video_record.progress = round(percent, 1)
                video_record.status = f"PROCESSING: {step_name}"
                await db.merge(video_record)
                await db.commit()
                logger.info(f"Progress: {video_record.progress}% - {step_name}")
            # -------------------------------------------------

            quote_text = video_record.quote.text
            uid = uuid.uuid4()

            # ---------- 1) Speech (master audio + timestamps) ----------
            audio_path = os.path.join(VIDEO_OUTPUT_DIR, f"semantic_audio_{uid}.mp3")
            speech_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "speecht5_engine.py")
            speech_cmd = [sys.executable, speech_script, "--text", quote_text, "--output-path", audio_path]
            logger.info("ORCH: Running SpeechT5 (audio + timestamps)...")
            rc, out, err = await run_subprocess_streamed(
                speech_cmd, timeout=SPEECH_TIMEOUT, env=os.environ.copy(), cwd=PROJECT_ROOT
            )
            if rc != 0:
                raise RuntimeError(f"speecht5_engine.py failed (rc={rc}). Stderr:\n{err}")
            try:
                speech_json = json.loads(out)
                if speech_json.get("status") != "COMPLETED":
                    raise ValueError(speech_json.get("error", "Speech engine error"))
                timestamps_path = speech_json.get("timestamps_file")  # optional
            except Exception as e:
                raise RuntimeError(f"Invalid SpeechT5 stdout.\n{out}\nError:{e}")

            # Stage 1: Audio done → 20%
            await update_progress(20.0, "Audio generated")

            # ---------- 2) Storyboard (with durations) ----------
            storyboard_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "storyboard_engine.py")
            storyboard_cmd = [sys.executable, storyboard_script, "--quote", quote_text]
            if timestamps_path and os.path.exists(timestamps_path):
                storyboard_cmd += ["--timestamps", timestamps_path]
            logger.info("ORCH: Running Storyboard engine...")
            rc2, out2, err2 = await run_subprocess_streamed(
                storyboard_cmd, timeout=STORYBOARD_TIMEOUT, env=os.environ.copy(), cwd=PROJECT_ROOT
            )
            if rc2 != 0:
                raise RuntimeError(f"storyboard_engine.py failed (rc={rc2}). Stderr:\n{err2}")
            try:
                sb_json = json.loads(out2)
                if sb_json.get("status") != "COMPLETED":
                    raise ValueError(sb_json.get("error", "Storyboard engine error"))
                storyboard = sb_json.get("storyboard", [])
                if not storyboard:
                    raise ValueError("Storyboard returned empty.")
            except Exception as e:
                raise RuntimeError(f"Invalid storyboard stdout.\n{out2}\nError:{e}")

            # Stage 2: Storyboard done → 40%
            await update_progress(40.0, "Storyboard created")

            # ---------- 3) Per-scene AnimateDiff ----------
            ad_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "mock_animate_diff_engine.py")
            for idx, scene in enumerate(storyboard, start=1):
                prompt = scene.get("animation_prompt")
                if not prompt:
                    raise ValueError(f"Scene {idx} has no 'animation_prompt'.")
                frames_dir = os.path.join(VIDEO_OUTPUT_DIR, f"scene_{idx}_{uuid.uuid4().hex}")
                scene_temp_dirs.append(frames_dir)
                logger.info(f"ORCH: Scene {idx}/{len(storyboard)} → AnimateDiff")
               
                # === THE ONLY CHANGE IS HERE: Golden Configuration Applied ===
                ad_cmd = [
                    sys.executable,
                    ad_script,
                    "--prompt", prompt,
                    "--output-dir", frames_dir,
                    "--num-steps", "4",
                    "--guidance-scale", "1.5"
                ]
                rc3, out3, err3 = await run_subprocess_streamed(
                    ad_cmd, timeout=AD_FRAME_TIMEOUT, env=os.environ.copy(), cwd=PROJECT_ROOT
                )
                if rc3 != 0:
                    raise RuntimeError(f"animate_diff_engine.py failed (scene {idx}) rc={rc3}. Stderr:\n{err3}")
                try:
                    ad_json = json.loads(out3)
                    if ad_json.get("status") != "COMPLETED":
                        raise ValueError(ad_json.get("error", f"AnimateDiff scene {idx} error"))
                    frame_paths = ad_json.get("frame_paths", [])
                    if not frame_paths:
                        raise ValueError(f"AnimateDiff scene {idx} returned no frames.")
                    all_frame_paths.extend(frame_paths)
                except Exception as e:
                    raise RuntimeError(f"Invalid AnimateDiff stdout (scene {idx}).\n{out3}\nError:{e}")

            # Stage 3: All scenes done → 80%
            await update_progress(80.0, f"Animated {len(storyboard)} scenes")

            # ---------- 4) Assemble final video ----------
            final_video_path = os.path.join(VIDEO_OUTPUT_DIR, f"semantic_video_{uid}.mp4")
            logger.info("ORCH: Assembling final video with master audio...")
            final_path = await video_generator.create_video_from_frames(
                frame_paths=all_frame_paths,
                output_path=final_video_path,
                audio_path=audio_path,
            )

            # Stage 4: Assembly done → 90%
            await update_progress(90.0, "Video assembled")

            # ---------- 5) Update DB ----------
            await crud.update_video_record(
                db, video=video_record, status=models.VideoStatus.COMPLETED, video_path=final_path
            )
            await db.commit()

            # Stage 5: Finalize → 100%
            video_record.progress = 100.0
            video_record.status = models.VideoStatus.COMPLETED
            await db.merge(video_record)
            await db.commit()

            logger.info(f"ORCHESTRATOR SUCCESS video_id={video_id}. Path: {final_path}")

        except Exception as e:
            logger.error(f"ORCHESTRATOR FAILED video_id={video_id}: {e}", exc_info=True)
            await db.rollback()
            vr = await crud.get_video(db, video_id=video_id)
            if vr:
                vr.progress = 0.0  # Reset on failure
                await crud.update_video_record(db, video=vr, status=models.VideoStatus.FAILED, error_message=str(e))
                await db.commit()
        finally:
            # Cleanup intermediates (audio + per-scene dirs)
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except OSError as err: logger.error(f"Audio cleanup error: {err}")
            for d in scene_temp_dirs:
                if d and os.path.exists(d):
                    try: shutil.rmtree(d)
                    except Exception as err: logger.error(f"Scene dir cleanup error: {err}")
            logger.info(f"ORCHESTRATOR FINISHED video_id={video_id}")