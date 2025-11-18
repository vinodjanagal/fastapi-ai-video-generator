import asyncio
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw
import logging

from app.video_generator import apply_camera_motion, create_video_from_frames

logging.basicConfig(level=logging.INFO)


# ----------------------------------------------------------
# Generate N simple test frames (512×512 colored rectangles)
# ----------------------------------------------------------
def generate_test_frames(n: int, out_dir: Path) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    for i in range(n):
        img = Image.new("RGB", (512, 512), (30 * i % 255, 80, 150))
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"Frame {i}", fill="white")

        p = out_dir / f"frame_{i:04d}.png"
        img.save(p)
        frame_paths.append(str(p))

    return frame_paths


# ----------------------------------------------------------
# Full async test pipeline
# ----------------------------------------------------------
async def main():
    print("\n=== TEST: apply_camera_motion + create_video_from_frames ===\n")

    with tempfile.TemporaryDirectory(prefix="test_frames_") as base_tmp:
        base_dir = Path(base_tmp)

        # 1) Create 20 test frames
        frame_dir = base_dir / "raw_frames"
        frames = generate_test_frames(20, frame_dir)
        print(f"Generated {len(frames)} test frames at: {frame_dir}")

        # 2) Apply a motion type (slow zoom-in)
        motion_type = "slow_zoom_in"
        print(f"Applying camera motion: {motion_type}")

        tempdir_obj, new_frames = await apply_camera_motion(
            frame_paths=frames,
            motion_type=motion_type,
            width=512,
            height=512,
            zoom_intensity=0.25,
            pan_intensity=0.20,
        )

        print(f"Motion applied. New frame count: {len(new_frames)}")
        print(f"Transformed frames stored in: {tempdir_obj.name}")

        # 3) Generate final test video (SILENT VIDEO)
        output_video = str(base_dir / "test_output.mp4")
        print(f"Building final test video → {output_video}")

        await create_video_from_frames(
            frame_paths=new_frames,
            output_path=output_video,
            audio_path=None,   # Silent video (supported in your corrected code!)
        )

        # Cleanup transformed temp frames
        if tempdir_obj:
            tempdir_obj.cleanup()

        print("\n=== TEST COMPLETE ===")
        print(f"Open the resulting video at:\n{output_video}\n")


if __name__ == "__main__":
    asyncio.run(main())
