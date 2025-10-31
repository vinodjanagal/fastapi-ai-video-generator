import logging
import os
import uuid
from typing import Optional

from app.database import AsyncSessionLocal
from app import crud, models, schemas
# We are assuming these are the correct import paths for your generators
from app import audio_generator_gtts as audio_generator
from app import video_generator
import subprocess
import sys
import json
import shutil

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIDEO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "static", "videos")
# Ensure the directory exists on startup
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


async def process_video_generation(video_id: int, style: str):
    """
    A hardened background task that generates a video, ensures transactional integrity,
    and cleans up resources on failure.
    """
    logger.info(f"TASK STARTED for video_id: {video_id}, style: '{style}'")

    # Step 1: Create a new, independent database session for this task.
    db = AsyncSessionLocal()
    
    # --- Resource Tracking ---
    # We define these here to track any created files for cleanup in the `finally` block.
    audio_path: Optional[str] = None
    video_path: Optional[str] = None

    try:
        # Step 2: Fetch the record using our eager-loading CRUD function.
        # This gets the video, quote, and author all in one efficient DB query.
        video_record = await crud.get_video(db, video_id=video_id)
        if not video_record:
            logger.error(f"Video record {video_id} not found. Aborting task.")
            return

        # Step 3: "Claim" the job by setting its status to PROCESSING.
        # We commit this immediately so the API frontend can see the job has started.
        await crud.update_video_record(db, video=video_record, status=models.VideoStatus.PROCESSING)
        await db.commit()
        logger.info(f"Video {video_id} status set to PROCESSING.")

        quote = video_record.quote

        # Step 4: Perform the slow, heavy work (Audio & Video Generation).
        unique_id = uuid.uuid4()
        audio_filename = f"quote_{quote.id}_{unique_id}.mp3"
        video_filename = f"quote_{quote.id}_{unique_id}.mp4"
        
        # Store the full paths for generation and potential cleanup.
        audio_path = os.path.join(VIDEO_OUTPUT_DIR, audio_filename)
        video_path = os.path.join(VIDEO_OUTPUT_DIR, video_filename)

        logger.info(f"Generating audio for video {video_id} -> {audio_path}")
        await audio_generator.generate_audio_from_text(
            text=quote.text,
            output_filename=audio_path
        )

        logger.info(f"Generating video for video {video_id} -> {video_path}")
        final_video_path = await video_generator.create_typography_video(
            audio_path=audio_path,
            text=quote.text,
            author_name=quote.author.name,
            output_path=video_path,
            style=style
        )
        
        # --- This is a test point. To check failure, you can uncomment the next line: ---
        # raise ValueError("A deliberate error to test the failure path!")

        # Step 5: If all work is successful, finalize the record.
        await crud.update_video_record(
            db,
            video=video_record,
            status=models.VideoStatus.COMPLETED,
            video_path=final_video_path
        )
        await db.commit()
        logger.info(f"TASK SUCCESS for video {video_id}. Path: {final_video_path}")

    except Exception as e:
        # Step 6: If ANY exception occurs, handle the failure gracefully.
        logger.error(f"TASK FAILED for video {video_id}: {e}", exc_info=True)
        
        # Rollback any uncommitted changes from the failed 'try' block.
        await db.rollback()
        
        # We must fetch the record again in a clean session state to update it.
        video_record_to_fail = await crud.get_video(db, video_id=video_id)
        if video_record_to_fail:
            await crud.update_video_record(
                db,
                video=video_record_to_fail,
                status=models.VideoStatus.FAILED,
                error_message=str(e)
            )
            await db.commit()
            logger.warning(f"Set video {video_id} status to FAILED in the database.")

    finally:
        # Step 7: This block runs ALWAYS, whether the task succeeded or failed.
        # It's our ultimate safety net for cleaning up resources.
        
        # Re-fetch the record's final status to decide if we should clean up.
        final_video_record = await crud.get_video(db, video_id=video_id)
        if final_video_record and final_video_record.status == models.VideoStatus.FAILED:
            logger.warning(f"Cleaning up artifacts for failed video job {video_id}.")
            # Clean up audio file if it exists
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"Removed failed audio artifact: {audio_path}")
                except OSError as err:
                    logger.error(f"Error removing file {audio_path}: {err}")
            # Clean up video file if it exists
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info(f"Removed failed video artifact: {video_path}")
                except OSError as err:
                    logger.error(f"Error removing file {video_path}: {err}")
        
        # On success, we only clean up the intermediate audio file.
        elif final_video_record and final_video_record.status == models.VideoStatus.COMPLETED:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"Removed intermediate audio file: {audio_path}")
                except OSError as err:
                    logger.error(f"Error removing intermediate file {audio_path}: {err}")
            
        # ALWAYS close the database session to return the connection to the pool.
        await db.close()
        logger.info(f"TASK FINISHED for video_id: {video_id}. Database session closed.") 

async def process_video_generation_speecht5(video_id: int, voice_name: str, video_style: str):

    """
    A hardened background task that uses the EXTERNAL speecht5_engine.py script
    via a subprocess to generate high-quality audio, then generates the video.
    """
    logger.info(f"TASK STARTED (SpeechT5) for video_id: {video_id}, voice: '{voice_name}'")

    db = AsyncSessionLocal()

    audio_path: Optional[str] = None
    video_path: Optional[str] = None

    try:
        video_record = await crud.get_video(db, video_id= video_id)
        if not video_record:
            logger.error(f"Video record {video_id} not found. Aborting task")
            return
        
        await crud.update_video_record(db, video= video_record, status= models.VideoStatus.PROCESSING)
        await db.commit()
        logger.info(f"Video {video_id} status set to PROCESSING")

        quote = video_record.quote


        # --- Step 1: Define paths for the external script ---
        unique_id = uuid.uuid4()
        audio_filename= f"quote_{quote.id}_{unique_id}.mp3"
        video_filename= f"quote_{quote.id}_{unique_id}.mp4"

        audio_path = os.path.join(VIDEO_OUTPUT_DIR, audio_filename)
        video_path = os.path.join(VIDEO_OUTPUT_DIR, video_filename)

        script_path = os.path.join(PROJECT_ROOT, "app", "video_engines", "speecht5_engine.py")


        # --- Step 2: Build and run the audio engine subprocess ---


        logger.info(f"Executing speecht5_engine.py for video {video_id}...")
        cmd = [
            sys.executable,          # The current python interpreter (ensures same venv)
            script_path,             # Path to our engine script
            "--text", quote.text,
            "--output-path", audio_path,
            "--voice", voice_name
        ]
        
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=86400)

        # --- Step 3: Check the result of the subprocess ---
        if proc.returncode != 0:
            error_message = f"speecht5_engine.py failed!\nExit Code: {proc.returncode}\nStderr: {proc.stderr}"
            raise RuntimeError(error_message)

        try:
            result = json.loads(proc.stdout)
            if result.get("status") != "COMPLETED":
                raise ValueError(f"SpeechT5 engine reported failure: {result.get('error', 'Unknown error')}")
            logger.info(f"speecht5_engine.py completed. Audio at: {result.get('file')}")
        except (json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Failed to parse output from speecht5_engine.py. Stdout: {proc.stdout}\nError: {e}")

        # --- Step 4: Proceed with video generation ---
        logger.info(f"Generating video for video {video_id} -> {video_path}")
        final_video_path = await video_generator.create_typography_video(
            audio_path=audio_path,
            text=quote.text,
            author_name=quote.author.name,
            output_path=video_path,
            style=video_style
        )
        
        await crud.update_video_record(
            db, video=video_record, status=models.VideoStatus.COMPLETED, video_path=final_video_path
        )
        await db.commit()
        logger.info(f"TASK SUCCESS (SpeechT5) for video {video_id}. Path: {final_video_path}")

    except Exception as e:
        logger.error(f"TASK FAILED (SpeechT5) for video {video_id}: {e}", exc_info=True)
        await db.rollback()
        video_record_to_fail = await crud.get_video(db, video_id=video_id)
        if video_record_to_fail:
            await crud.update_video_record(
                db, video=video_record_to_fail, status=models.VideoStatus.FAILED, error_message=str(e)
            )
            await db.commit()
            logger.warning(f"Set video {video_id} status to FAILED in the database.")

    finally:
        # The cleanup logic is identical and works perfectly.
        final_video_record = await crud.get_video(db, video_id=video_id)
        if final_video_record and final_video_record.status == models.VideoStatus.FAILED:
            logger.warning(f"Cleaning up artifacts for failed SpeechT5 job {video_id}.")
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except OSError as err: logger.error(f"Error removing file {audio_path}: {err}")
            if video_path and os.path.exists(video_path):
                try: os.remove(video_path)
                except OSError as err: logger.error(f"Error removing file {video_path}: {err}")
        
        elif final_video_record and final_video_record.status == models.VideoStatus.COMPLETED:
            if audio_path and os.path.exists(audio_path):
                try: os.remove(audio_path)
                except OSError as err: logger.error(f"Error removing intermediate file {audio_path}: {err}")
            
        await db.close()
        logger.info(f"TASK FINISHED (SpeechT5) for video_id: {video_id}. Database session closed.")





async def process_video_generation_animatediff(video_id: int, prompt: str, voice_name: str):
    """
    Orchestrates the full AnimateDiff video generation pipeline:
    1. Generates audio via the speecht5_engine subprocess.
    2. Generates frames via the animate_diff_engine subprocess.
    3. Assembles the final video.
    4. Cleans up all intermediate files.
    """
    logger.info(f"TASK STARTED (AnimateDiff Full Pipeline) for video_id: {video_id}")
    db = AsyncSessionLocal()
    
    # --- Resource Tracking ---
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    frames_dir: Optional[str] = None

    try:
        video_record = await crud.get_video(db, video_id=video_id)
        if not video_record:
            raise FileNotFoundError(f"Video record {video_id} not found.")

        await crud.update_video_record(db, video=video_record, status=models.VideoStatus.PROCESSING)
        await db.commit()
        logger.info(f"Video {video_id} status set to PROCESSING.")

        quote_text = video_record.quote.text
        unique_id = uuid.uuid4()
        
        # --- Define Paths ---
        audio_filename = f"audio_{unique_id}.mp3"
        video_filename = f"video_{unique_id}.mp4"
        frames_dirname = f"frames_{unique_id}"
        
        audio_path = os.path.join(VIDEO_OUTPUT_DIR, audio_filename)
        video_path = os.path.join(VIDEO_OUTPUT_DIR, video_filename)
        frames_dir = os.path.join(VIDEO_OUTPUT_DIR, frames_dirname)
        
        # === STAGE 1: AUDIO GENERATION ===
        logger.info("Stage 1: Executing speecht5_engine.py...")
        audio_script_path = os.path.join(PROJECT_ROOT, "app", "video_engines", "speecht5_engine.py")
        audio_cmd = [
            sys.executable, audio_script_path,
            "--text", quote_text,
            "--output-path", audio_path,
            "--voice", voice_name
        ]
        audio_proc = subprocess.run(audio_cmd, capture_output=True, text=True, check=False, timeout=300)
        
        if audio_proc.returncode != 0:
            raise RuntimeError(f"speecht5_engine.py failed!\nStderr: {audio_proc.stderr}")
        
        audio_result = json.loads(audio_proc.stdout)
        if audio_result.get("status") != "COMPLETED":
            raise ValueError(f"SpeechT5 engine reported failure: {audio_result.get('error')}")
        logger.info("Stage 1: Audio generation successful.")

        # === STAGE 2: FRAME GENERATION ===
        logger.info("Stage 2: Executing animate_diff_engine.py...")
        frame_script_path = os.path.join(PROJECT_ROOT, "app", "video_engines", "animate_diff_engine.py")
        frame_cmd = [
            sys.executable, frame_script_path,
            "--prompt", prompt,
            "--output-dir", frames_dir
        ]
        # Using a very long timeout for the slow frame generation
        frame_proc = subprocess.run(frame_cmd, capture_output=True, text=True, check=False, timeout=7200)

        if frame_proc.returncode != 0:
            raise RuntimeError(f"animate_diff_engine.py failed!\nStderr: {frame_proc.stderr}")
            
        frame_result = json.loads(frame_proc.stdout)
        if frame_result.get("status") != "COMPLETED":
            raise ValueError(f"AnimateDiff engine reported failure: {frame_result.get('error')}")
            
        frame_paths = frame_result.get("frame_paths", [])
        if not frame_paths:
            raise ValueError("AnimateDiff engine returned no frame paths.")
        logger.info(f"Stage 2: Frame generation successful ({len(frame_paths)} frames).")

        # === STAGE 3: FINAL ASSEMBLY ===
        logger.info("Stage 3: Assembling final video from frames and audio...")
        final_video_path = await video_generator.create_video_from_frames(
            frame_paths=frame_paths,
            output_path=video_path,
            fps=8,  # Lightning model default
            audio_path=audio_path
        )
        logger.info("Stage 3: Video assembly successful.")

        # === STAGE 4: DATABASE UPDATE ===
        await crud.update_video_record(
            db, video=video_record, status=models.VideoStatus.COMPLETED, video_path=final_video_path
        )
        await db.commit()
        logger.info(f"TASK SUCCESS (AnimateDiff Full Pipeline) for video {video_id}. Path: {final_video_path}")

    except Exception as e:
        logger.error(f"TASK FAILED (AnimateDiff Full Pipeline) for video {video_id}: {e}", exc_info=True)
        await db.rollback()
        video_record_to_fail = await crud.get_video(db, video_id=video_id)
        if video_record_to_fail:
            await crud.update_video_record(
                db, video=video_record_to_fail, status=models.VideoStatus.FAILED, error_message=str(e)
            )
            await db.commit()

    finally:
        # === STAGE 5: GUARANTEED CLEANUP ===
        logger.info(f"Cleaning up intermediate files for job {video_id}...")
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Cleaned up audio file: {audio_path}")
            except OSError as err:
                logger.error(f"Error removing audio file {audio_path}: {err}")
        
        if frames_dir and os.path.exists(frames_dir):
            try:
                shutil.rmtree(frames_dir)
                logger.info(f"Cleaned up frames directory: {frames_dir}")
            except Exception as e_clean:
                logger.error(f"Failed to clean up frames directory {frames_dir}: {e_clean}")
                
        await db.close()
        logger.info(f"TASK FINISHED (AnimateDiff Full Pipeline) for video_id: {video_id}. Database session closed.")