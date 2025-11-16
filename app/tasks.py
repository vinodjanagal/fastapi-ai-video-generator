import os
import sys
import json
import uuid
import shutil
import logging
import asyncio
import tempfile
from typing import List, Dict, Optional, Tuple
from sqlalchemy import Float
from app.database import AsyncSessionLocal
from app import crud, models
from app import audio_generator_gtts as audio_generator
from app import video_generator
from app.video_generator import apply_camera_motion
from app.engine.character_sheet import build_character_prompt
from app.engine.parser import semantic_parser
from app.engine.prompt_builder import build_semantic_prompt
from app.engine.cinematics import classify_shot_type



logger = logging.getLogger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "static", "videos")
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

# --- CONSTANTS EXTRACTED FROM YOUR ADVANCED SCRIPT ---
BASE_QUALITY_PROMPT = "masterpiece, best quality, ultra-detailed, photorealistic, cinematic lighting, sharp focus, 8k"
BASE_NEGATIVE_PROMPT = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, mutation, mutilated, extra limbs, gross proportions, malformed limbs, disfigured face, ugly(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"

# ---------- Tunables ----------
SPEECH_TIMEOUT = 86_400     # 24h
AD_FRAME_TIMEOUT = 86_400   # 24h
STORYBOARD_TIMEOUT = 86_400 # 5m is plenty for LLM call via CLI wrapper
TIMESTAMPS_TIMEOUT = 86_400 # 10m if you later split timestamps engine out
DEFAULT_FPS = 8

V2_CLASSIC_STEPS = 20
V2_CLASSIC_GUIDANCE = 6.0
V2_IP_ADAPTER_SCALE = 0.12

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
            logger.debug(text.rstrip("\n"))
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

                bar = "█" * int(percent // 10) + "░" * (10 - int(percent // 10))
                # Log the new, clean format
                logger.info(f"PROGRESS: [{bar}] {percent:.1f}% - {step_name}")


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

                bar = "█" * int(percent // 10) + "░" * (10 - int(percent // 10))
                # Log the new, clean format
                logger.info(f"PROGRESS: [{bar}] {percent:.1f}% - {step_name}")


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
            frame_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "mock_animate_diff_engine.py")
           
            # === THE ONLY CHANGE IS HERE: Golden Configuration Applied ===
            frame_cmd = [
                sys.executable,
                frame_script,
                "--prompt", prompt,
                "--output-dir", frames_dir,
                "--num-steps", "20",
                "--guidance-scale", "6"
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
    Phoenix Semantic Pipeline V7.3 (Final Clean Version)
    ----------------------------------------------------
    ✔ Uses CHARACTER-SHEET builder for consistent casting
    ✔ Uses SEMANTIC BUILDER for scene prompts (enhanced prompts)
    ✔ Uses CINEMATIC SHOT classifier (hybrid logic)
    ✔ Preserves storyboard composition overrides
    ✔ Correctly passes final prompts to AnimateDiff
    ✔ Avoids ALL duplicated logic from earlier versions
    """

    logger.info(f"ORCHESTRATOR V7.3 START video_id={video_id}")

    audio_path: Optional[str] = None
    timestamps_path: Optional[str] = None
    final_video_path: Optional[str] = None
    character_reference_image_path: Optional[str] = None

    cleanup_dirs: List[str] = []
    motion_temp_dir_objects: list = []

    async with AsyncSessionLocal() as db:
        try:
            # ------------------------------------------------------------
            # LOAD VIDEO OBJECT
            # ------------------------------------------------------------
            video_record = await crud.get_video(db, video_id=video_id)
            if not video_record:
                raise FileNotFoundError(f"Video {video_id} not found.")

            async def update_progress(percent: float, step_name: str):
                video_record.progress = round(percent, 1)
                video_record.status = f"PROCESSING: {step_name}"
                await db.merge(video_record)
                await db.commit()
                bar = "█" * int(percent // 10) + "░" * (10 - int(percent // 10))
                logger.info(f"PROGRESS: [{bar}] {percent:.1f}% - {step_name}")

            await update_progress(0.0, "Initializing...")

            quote_text = video_record.quote.text
            uid = uuid.uuid4()


            # ------------------------------------------------------------
            # 1) SPEECH GENERATION
            # ------------------------------------------------------------
            await update_progress(5.0, "Generating audio...")

            audio_path = os.path.join(VIDEO_OUTPUT_DIR, f"semantic_audio_{uid}.mp3")
            speech_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "speecht5_engine.py")

            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8') as tf:
                tf.write(quote_text)
                text_file_path = tf.name

            try:
                speech_cmd = [sys.executable, speech_script,
                              "--text-file", text_file_path,
                              "--output-path", audio_path]
                rc, out, err = await run_subprocess_streamed(speech_cmd, timeout=SPEECH_TIMEOUT)
                if rc != 0:
                    raise RuntimeError(f"Speech engine failed. STDERR:\n{err}")

                speech_json = json.loads(out)
                if speech_json.get("status") != "COMPLETED":
                    raise ValueError(speech_json.get("error", "Speech engine error"))

                timestamps_path = speech_json.get("timestamps_file")

            finally:
                if os.path.exists(text_file_path):
                    os.remove(text_file_path)

            await update_progress(15.0, "Audio generated")


            # ------------------------------------------------------------
            # 2) STORYBOARD GENERATION
            # ------------------------------------------------------------
            await update_progress(20.0, "Directing scenes...")

            storyboard_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "storyboard_engine.py")
            storyboard_cmd = [sys.executable, storyboard_script, "--quote", quote_text]

            if timestamps_path:
                storyboard_cmd += ["--timestamps", timestamps_path]

            rc2, out2, err2 = await run_subprocess_streamed(storyboard_cmd, timeout=STORYBOARD_TIMEOUT)
            if rc2 != 0:
                raise RuntimeError(f"Storyboard engine failed. STDERR:\n{err2}")

            storyboard_data = json.loads(out2).get("storyboard_data", {})
            storyboard_scenes = storyboard_data.get("scenes", [])
            character_sheet_prompt = storyboard_data.get("character_sheet")

            if not storyboard_scenes:
                raise ValueError("Storyboard returned no scenes.")

            await update_progress(25.0, "Storyboard created")


            # ------------------------------------------------------------
            # 3) CHARACTER CASTING (MASTER REFERENCE IMAGE)
            # ------------------------------------------------------------
            ad_script = os.path.join(PROJECT_ROOT, "app", "video_engines", "animate_diff_engine.py")

            if character_sheet_prompt:
                await update_progress(30.0, "Casting character...")

                char_sheet_dir = os.path.join(VIDEO_OUTPUT_DIR, f"character_sheet_{uid.hex}")
                cleanup_dirs.append(char_sheet_dir)

                # USE THE NEW CHARACTER BUILDER
                final_char_prompt, final_char_negative = build_character_prompt(character_sheet_prompt)

                logger.info(f"CHARACTER SHEET PROMPT: {final_char_prompt}")

                char_cmd = [
                    sys.executable, ad_script,
                    "--prompt", final_char_prompt,
                    "--negative-prompt", final_char_negative,
                    "--output-dir", char_sheet_dir,
                    "--num-frames", "1",
                    "--num-steps", "25",
                    "--guidance-scale", "7.0",
                ]

                rc_char, out_char, err_char = await run_subprocess_streamed(char_cmd, timeout=AD_FRAME_TIMEOUT)
                if rc_char != 0:
                    raise RuntimeError(f"Character sheet generation failed. STDERR:\n{err_char}")

                char_json = json.loads(out_char)
                character_frame_paths = char_json.get("frame_paths", [])
                if not character_frame_paths:
                    raise ValueError("Character sheet returned no frames.")

                character_reference_image_path = character_frame_paths[0]

                logger.info(f"MASTER CHARACTER IMAGE: {character_reference_image_path}")


            # ------------------------------------------------------------
            # 4) SCENE BY SCENE RENDERING
            # ------------------------------------------------------------
            last_successful_frame_path = None
            all_final_frame_paths = []

            for idx, scene in enumerate(storyboard_scenes, start=1):
                progress_start = 35.0
                progress_range = 90.0 - progress_start
                scene_progress = progress_start + ((idx - 1) / len(storyboard_scenes)) * progress_range

                await update_progress(scene_progress, f"Rendering Scene {idx}/{len(storyboard_scenes)}")

                raw_description = scene.get("description", "")
                composition = scene.get("composition", {})

                # -------- NEW BRAIN --------
                semantic_parts = semantic_parser(raw_description)
                shot_type = classify_shot_type(raw_description, semantic_parts)

                final_positive_prompt, final_negative_prompt = build_semantic_prompt(
                    raw_description,
                    BASE_NEGATIVE_PROMPT,
                    shot_type,
                    semantic_parts
                )

                # Composition overrides (only add missing elements)
                additions = []
                cam = composition.get("camera", "")
                env = composition.get("environment", "")
                light = composition.get("lighting", "")
                style = composition.get("style", "")

                if cam and cam.lower() not in final_positive_prompt.lower():
                    additions.append(cam)
                if env and env.lower() not in final_positive_prompt.lower():
                    additions.append(f"Scene set in {env}")
                if light and light.lower() not in final_positive_prompt.lower():
                    additions.append(f"with {light} lighting")
                if style and style.lower() not in final_positive_prompt.lower():
                    additions.append(style)

                if additions:
                    final_positive_prompt += ", " + ", ".join(additions)

                # LOG FINAL PROMPT
                logger.info(f"FINAL PROMPT (Scene {idx}): {final_positive_prompt}")


                # -------- CALL ANIMATEDIFF --------
                base_frames_dir = os.path.join(VIDEO_OUTPUT_DIR, f"scene_{idx}_base_{uid.hex}")
                cleanup_dirs.append(base_frames_dir)

                ad_cmd = [
                    sys.executable, ad_script,
                    "--prompt", final_positive_prompt,
                    "--negative-prompt", final_negative_prompt,
                    "--output-dir", base_frames_dir,
                    "--num-steps", str(V2_CLASSIC_STEPS),
                    "--guidance-scale", str(V2_CLASSIC_GUIDANCE),
                ]

                # IP ADAPTER → ALWAYS use master reference first
                if character_reference_image_path:
                    ad_cmd.extend([
                        "--ip-adapter-image-path", character_reference_image_path,
                        "--ip-adapter-scale", "0.55"
                    ])
                elif last_successful_frame_path:
                    ad_cmd.extend([
                        "--ip-adapter-image-path", last_successful_frame_path,
                        "--ip-adapter-scale", str(V2_IP_ADAPTER_SCALE)
                    ])

                # Call renderer
                rc3, out3, err3 = await run_subprocess_streamed(ad_cmd, timeout=AD_FRAME_TIMEOUT)
                if rc3 != 0:
                    raise RuntimeError(f"AnimateDiff failed (scene {idx}). STDERR:\n{err3}")

                original_paths = json.loads(out3).get("frame_paths", [])
                if not original_paths:
                    raise ValueError(f"No frames returned for scene {idx}")

                # Camera motion
                motion_type = scene.get("camera_motion", "static")
                if motion_type != "static":
                    temp_d, transformed = await apply_camera_motion(original_paths, motion_type)
                    if temp_d:
                        motion_temp_dir_objects.append(temp_d)
                    all_final_frame_paths.extend(transformed)
                else:
                    all_final_frame_paths.extend(original_paths)

                last_successful_frame_path = original_paths[len(original_paths) // 2]


            # ------------------------------------------------------------
            # 5) ASSEMBLE VIDEO
            # ------------------------------------------------------------
            await update_progress(95.0, "Assembling video...")

            final_video_path = os.path.join(VIDEO_OUTPUT_DIR, f"semantic_video_{uid}.mp4")

            final_path = await video_generator.create_video_from_frames(
                frame_paths=all_final_frame_paths,
                output_path=final_video_path,
                audio_path=audio_path,
            )

            await crud.update_video_record(
                db,
                video=video_record,
                status=models.VideoStatus.COMPLETED,
                video_path=final_path
            )
            await db.commit()

            await update_progress(100.0, "Completed")

            logger.info(f"ORCHESTRATOR SUCCESS video_id={video_id} → {final_path}")

        except Exception as e:
            logger.error(f"ORCHESTRATOR FAILED video_id={video_id}: {e}", exc_info=True)
            await db.rollback()

            vr = await crud.get_video(db, video_id=video_id)
            if vr:
                vr.progress = 0.0
                await crud.update_video_record(
                    db, video=vr,
                    status=models.VideoStatus.FAILED,
                    error_message=str(e)
                )
                await db.commit()

        finally:
            logger.info(f"ORCHESTRATOR CLEANUP video_id={video_id}")

            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except: pass

            for d in cleanup_dirs:
                if os.path.exists(d):
                    try: shutil.rmtree(d)
                    except: pass

            for temp_dir_obj in motion_temp_dir_objects:
                try: temp_dir_obj.cleanup()
                except: pass

            logger.info(f"ORCHESTRATOR FINISHED video_id={video_id}")
