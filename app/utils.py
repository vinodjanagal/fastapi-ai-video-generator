# file: app/utils.py
import asyncio
import logging
import sys
from typing import List, Tuple

logger = logging.getLogger("app.utils")

async def run_subprocess_with_realtime_progress(
    cmd: List[str],
    timeout_per_line: int = 7200  # Generous 1-hour idle timeout for long renders
) -> Tuple[int, str]:
    """
    The definitive Phoenix subprocess runner.
    - Streams stderr to the console in real-time for progress bars (tqdm).
    - Captures stdout in memory for parsing the final JSON result.
    - Has an idle timeout to kill stalled processes.
    - Handles KeyboardInterrupt gracefully.
    
    Returns: (return_code, captured_stdout)
    """
    logger.info(f"Executing subprocess: {' '.join(cmd)}")
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout_capture = []

    async def stream_stdout(stream):
        """Capture stdout in memory."""
        while True:
            try:
                line = await asyncio.wait_for(stream.readline(), timeout=timeout_per_line)
                if not line: break
                stdout_capture.append(line.decode('utf-8', errors='ignore'))
            except asyncio.TimeoutError:
                logger.error(f"Subprocess STALLED (no stdout): Terminating after {timeout_per_line}s.")
                if proc.returncode is None: proc.kill()
                break

    async def stream_stderr(stream):
        """Stream stderr directly to the console in real-time."""
        while True:
            try:
                line = await asyncio.wait_for(stream.readline(), timeout=timeout_per_line)
                if not line: break
                sys.stderr.write(line.decode('utf-8', errors='ignore'))
                sys.stderr.flush()
            except asyncio.TimeoutError:
                logger.error(f"Subprocess STALLED (no stderr): Terminating after {timeout_per_line}s.")
                if proc.returncode is None: proc.kill()
                break

    try:
        await asyncio.gather(
            stream_stdout(proc.stdout),
            stream_stderr(proc.stderr)
        )
    except KeyboardInterrupt:
        logger.warning("\nKeyboardInterrupt received. Terminating subprocess...")
        if proc.returncode is None: proc.kill()
        await proc.wait()
        raise

    
    rc = await proc.wait()
    clean_out = "".join(stdout_capture).strip()
    return rc, clean_out