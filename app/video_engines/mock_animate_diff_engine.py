# file: app/video_engines/mock_animate_diff_engine.py
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# This is an upgraded "stunt double" for the real animate_diff_engine.py
# It accepts all the arguments the real engine does to avoid crashing.

def main():
    parser = argparse.ArgumentParser(description="MOCK AnimateDiff Engine V2: Simulates frame generation.")
    
    # --- ADD ALL ARGUMENTS FROM THE REAL ENGINE'S CALLS ---
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--ip-adapter-image-path", type=str, default=None)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.5)
    # Add any other arguments your real script might use
    parser.add_argument("--base-model", type=str, default="")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- THE CORE MOCK LOGIC (Unchanged) ---
    placeholder_src = Path(__file__).parent / "placeholder.png"
    if not placeholder_src.exists():
        placeholder_src.touch()

    # Use num_frames if provided and > 0, otherwise default to 16
    num_mock_frames = args.num_frames if args.num_frames > 0 else 16

    frame_paths = []
    for i in range(num_mock_frames):
        dest_path = output_dir / f"frame_{i:04d}.png"
        shutil.copy(str(placeholder_src), str(dest_path))
        frame_paths.append(str(dest_path))

    # Simulate a little bit of work
    time.sleep(0.5)

    result = {
        "status": "COMPLETED",
        "frame_paths": frame_paths
    }
    print(json.dumps(result))
    sys.exit(0)

if __name__ == "__main__":
    main()