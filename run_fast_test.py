# file: run_fast_test.py
import asyncio
import sys
import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("FAST_TEST_RUNNER")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import tasks, crud, models

class FakeAuthor:
    name = "Test Author"
class FakeQuote:
    id = 1
    text = "This is a test quote for the fast pipeline."
    author = FakeAuthor()
class FakeVideoRecord:
    def __init__(self, quote):
        self.id = 123
        self.quote = quote
        self.status = "PENDING"
        self.progress = 0.0
        self.error_message = None
        self.video_path = None

class FakeDBSession:
    async def __aenter__(self):
        logger.info("MOCK DB: Entering async session context.")
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info("MOCK DB: Exiting async session context.")
    async def add(self, obj):
        logger.info(f"MOCK DB: Adding object to session (e.g., staging update for video {getattr(obj, 'id', 'N/A')})")
        pass
    async def merge(self, obj):
        logger.info(f"MOCK DB: Merging object (e.g., updating progress to {getattr(obj, 'progress', 'N/A')}%)")
        pass
    async def commit(self):
        logger.info("MOCK DB: Commit called.")
        pass
    async def rollback(self):
        logger.info("MOCK DB: Rollback called due to an error.")
        pass

async def main():
    TEST_OUTPUT_DIR = Path("./test_run_output")
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
    TEST_OUTPUT_DIR.mkdir()

    logger.info("--- 🚀 STARTING FAST PIPELINE TEST FOR V9.1 🚀 ---")
    original_video_output_dir = tasks.VIDEO_OUTPUT_DIR
    original_async_session = tasks.AsyncSessionLocal
    original_get_video = crud.get_video
    original_update_video = crud.update_video_record
    original_run_subprocess = tasks.run_subprocess_streamed

    try:
        logger.info("Monkeypatching external engines and database...")
        tasks.VIDEO_OUTPUT_DIR = str(TEST_OUTPUT_DIR.resolve())
        tasks.AsyncSessionLocal = FakeDBSession

        mock_storyboard_path = Path("tests/assets/mock_storyboard_v9.json")
        mock_animate_diff_path = str(Path("app/video_engines/mock_animate_diff_engine.py").resolve())
        with open(mock_storyboard_path, 'r') as f:
            mock_storyboard_output = f.read()

        async def mocked_run_subprocess_streamed(cmd, *args, **kwargs):
            script_path = cmd[1]
            if "storyboard_engine.py" in script_path:
                logger.info("INTERCEPTED & MOCKED: storyboard_engine.py call")
                return 0, mock_storyboard_output, ""
            elif "animate_diff_engine.py" in script_path:
                logger.info(f"INTERCEPTED & REROUTING to mock: {' '.join(cmd)}")
                new_cmd = [sys.executable, mock_animate_diff_path] + cmd[2:]
                return await original_run_subprocess(new_cmd, *args, **kwargs)
            return await original_run_subprocess(cmd, *args, **kwargs)
        tasks.run_subprocess_streamed = mocked_run_subprocess_streamed

        fake_video_record_instance = FakeVideoRecord(quote=FakeQuote())
        async def mock_get_video(db_session, video_id):
            logger.info(f"MOCK CRUD: get_video for id {video_id} -> returning fake record")
            return fake_video_record_instance
        crud.get_video = mock_get_video
        
        # We also need to mock the crud function itself to avoid the db.add call
        async def mock_update_video_record(db, video, **kwargs):
             logger.info(f"MOCK CRUD: Updating video {video.id} with status {kwargs.get('status')}")
             video.status = kwargs.get('status')
             video.video_path = kwargs.get('video_path')
             video.error_message = kwargs.get('error_message')
             return video
        crud.update_video_record = mock_update_video_record

        logger.info("✅ Patching complete. Running the orchestrator...")
        await tasks.process_semantic_video_generation(video_id=123)
        logger.info("--- ✅ TEST SUCCEEDED ---")
        logger.info(f"Final video should be located in: {TEST_OUTPUT_DIR}")
    except Exception as e:
        logger.error("--- ❌ TEST FAILED ---", exc_info=True)
    finally:
        logger.info("Restoring original functions...")
        tasks.VIDEO_OUTPUT_DIR = original_video_output_dir
        tasks.AsyncSessionLocal = original_async_session
        crud.get_video = original_get_video
        crud.update_video_record = original_update_video
        tasks.run_subprocess_streamed = original_run_subprocess

if __name__ == "__main__":
    asyncio.run(main())