import asyncio
import subprocess
import logging
from app.tasks import process_video_generation_speecht5, process_video_generation

# Set up basic logging to see the output from the worker
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- This is our main job function ---
# ARQ will call this function with the arguments we provide.
async def generate_audio_task(ctx, quote_text: str, audio_file_path: str):
    """
    A job that calls our standalone speecht5_engine.py script.
    """
    logging.info(f"Starting audio generation for audio file: {audio_file_path}")

    try:
        # We use asyncio.create_subprocess_exec to run the command asynchronously,
        # which is a best practice when working with async frameworks like ARQ.
        process = await asyncio.create_subprocess_exec(
            "python",
            "app/video_engines/speecht5_engine.py",
            "--text",
            quote_text,
            "--output",
            audio_file_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for the process to finish and capture the output
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logging.info(f"Successfully generated audio file: {audio_file_path}")
            return f"Success: {audio_file_path}"
        else:
            # If the script fails, log the error from stderr
            error_message = stderr.decode().strip()
            logging.error(f"Audio generation failed for {audio_file_path}. Error: {error_message}")
            raise RuntimeError(error_message)

    except Exception as e:
        logging.error(f"An unexpected error occurred during audio generation: {e}")
        # Re-raise the exception so ARQ knows the job failed
        raise



async def generate_video_task(ctx, video_id: int, style: str):
    """
    The main ARQ task for generating a complete video.
    This function will call the existing async logic from app.tasks.
    """
    logging.info(f"ARQ worker picked up video generation job for video_id: {video_id} with style '{style}'")
    
    # REMOVED: loop = asyncio.get_running_loop()

    try:
        style_parts = style.split(':')
        voice = style_parts[0]
        video_style = style_parts[1] if len(style_parts) > 1 else "dark_gradient"

        speecht5_voices = {"atlas", "nova", "echo", "breeze"}
        
        if voice in speecht5_voices:
            logging.info(f"Awaiting SpeechT5 process for video_id: {video_id}")
            
            # --- THE CORRECT WAY for async-to-async calls ---
            await process_video_generation_speecht5(
                video_id=video_id,
                voice_name=voice,
                video_style=video_style
            )
        else:
            logging.info(f"Awaiting default gTTS process for video_id: {video_id}")

            # --- THE CORRECT WAY for async-to-async calls ---
            await process_video_generation(
                video_id=video_id,
                style=style
            )
        
        logging.info(f"Successfully completed video generation for video_id: {video_id}")
        return f"Success for video_id: {video_id}"

    except Exception as e:
        logging.error(f"Video generation FAILED for video_id: {video_id}. Error: {e}", exc_info=True)
        raise


# --- This is the ARQ Worker Configuration ---
# It tells ARQ which functions are jobs and how to connect to Redis.
class WorkerSettings:
    functions = [generate_audio_task, generate_video_task]
    # This default Redis connection setting assumes Redis is running on localhost:6379
    # which is exactly what our Docker command did.


# --- This block allows us to run the worker directly with "python worker.py" ---
if __name__ == "__main__":
    from arq.worker import run_worker
    run_worker(WorkerSettings) # Just call the function directly.