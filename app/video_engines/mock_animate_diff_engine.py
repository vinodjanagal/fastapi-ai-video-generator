# app/video_engines/mock_animate_diff_engine.py
import argparse
import json
import os
import shutil
import time
from pathlib import Path

# THIS SCRIPT IS A "STUNT DOUBLE" FOR THE REAL animate_diff_engine.py
# Its only job is to simulate a successful run as quickly as possible.

def main():
    parser = argparse.ArgumentParser(description="MOCK AnimateDiff Engine: Simulates frame generation.")
    parser.add_argument("--prompt", type=str, required=True, help="Animation prompt (ignored, but required for compatibility).")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output frames.")
    # Add other arguments from the real script to accept them without error, even if we don't use them.
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- THE CORE MOCK LOGIC ---
    # Instead of generating frames for hours, we copy a placeholder a few times.
    # This simulates the real output in milliseconds.
    
    placeholder_src = Path(__file__).parent / "placeholder.png"
    if not placeholder_src.exists():
        # As a fallback, create a dummy file if placeholder.png is missing
        placeholder_src.touch()

    frame_paths = []
    num_mock_frames = 16 # Simulate generating 16 frames

    for i in range(num_mock_frames):
        dest_path = output_dir / f"frame_{i:04d}.png"
        shutil.copy(str(placeholder_src), str(dest_path))
        frame_paths.append(str(dest_path))

    # Simulate a little bit of work
    time.sleep(2)

    # --- THE CRITICAL CONTRACT ---
    # Print the exact JSON structure the orchestrator expects on success.
    result = {
        "status": "COMPLETED",
        "frame_paths": frame_paths
    }
    print(json.dumps(result))

if __name__ == "__main__":
    main()