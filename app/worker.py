import os
import sys
import json
import asyncio
import logging
import subprocess

from app.tasks import (
    process_video_generation_speecht5,
    process_video_generation,
    process_video_generation_animatediff,
    process_semantic_video_generation,   # internal orchestrator (we’ll wrap it)
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------- ARQ tasks ----------

async def generate_audio_task(ctx, quote_text: str, audio_file_path: str):
    """
    Runs the standalone speecht5_engine.py to generate audio.
    NOTE: speecht5_engine.py expects --output-path (not --output).
    """
    logging.info(f"Starting audio generation for: {audio_file_path}")

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "app/video_engines/speecht5_engine.py",
            "--text", quote_text,
            "--output-path", audio_file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            logging.info(f"Audio OK: {audio_file_path}")
            return f"Success: {audio_file_path}"
        else:
            err = stderr.decode(errors="ignore").strip()
            logging.error(f"Audio generation failed. rc={process.returncode}. stderr:\n{err}")
            raise RuntimeError(err)

    except Exception as e:
        logging.error(f"Unexpected error during audio generation: {e}", exc_info=True)
        raise

async def generate_video_task(ctx, video_id: int, style: str):
    """
    Main video generation task.
    """
    logging.info(f"ARQ job: generate_video_task video_id={video_id}, style='{style}'")
    try:
        style_parts = (style or "").split(":")
        voice = style_parts[0] if style_parts and style_parts[0] else "gtts"
        video_style = style_parts[1] if len(style_parts) > 1 else "dark_gradient"

        speecht5_voices = {"atlas", "nova", "echo", "breeze"}
        if voice in speecht5_voices:
            await process_video_generation_speecht5(video_id=video_id, voice_name=voice, video_style=video_style)
        else:
            await process_video_generation(video_id=video_id, style=video_style)

        logging.info(f"Completed video generation video_id={video_id}")
        return f"Success for video_id: {video_id}"

    except Exception as e:
        logging.error(f"Video generation FAILED for {video_id}: {e}", exc_info=True)
        raise

# NEW: ARQ wrapper to run the semantic orchestrator
async def generate_semantic_video_task(ctx, video_id: int):
    """
    ARQ-executable wrapper that calls the internal semantic orchestrator.
    """
    logging.info(f"ARQ job: generate_semantic_video_task video_id={video_id}")
    try:
        await process_semantic_video_generation(video_id=video_id)
        logging.info(f"Semantic video completed video_id={video_id}")
        return f"Semantic Success for video_id: {video_id}"
    except Exception as e:
        logging.error(f"Semantic video FAILED for {video_id}: {e}", exc_info=True)
        raise

class WorkerSettings:
    functions = [
        generate_audio_task,
        generate_video_task,
        # Keep these as ARQ-callables only if you invoke them directly from API:
        # process_video_generation_animatediff,   # typically called internally, but safe to expose if you want
        generate_semantic_video_task,            # ✅ expose orchestrator via wrapper
    ]
    job_timeout = 86_400  # 24h

if __name__ == "__main__":
    from arq.worker import run_worker
    run_worker(WorkerSettings)
