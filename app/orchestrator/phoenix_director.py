import os
import argparse
import subprocess
import logging
import uuid
import glob
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("phoenix.v24.anchored")

class PhoenixDirector:
    def __init__(self, mode="production"):
        self.root = os.getcwd()
        self.output_dir = os.path.join(self.root, "phoenix_output")
        self.engine_script = os.path.join(self.root, "app", "video_engines", "animate_diff_engine.py")
        self.conf = {
            "char":  {"width": 512, "height": 512, "frames": 16, "steps": 30},
            "scene": {"width": 512, "height": 384, "frames": 16, "steps": 30}
        }
        os.makedirs(self.output_dir, exist_ok=True)

    def run_command(self, cmd):
        logger.info(f"[EXEC] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            logger.error("Command Failed.")
            raise

    def generate_character(self, prompt, uid):
        logger.info(">>> PHASE 1: ANCHORING IDENTITY")
        out_dir = os.path.join(self.output_dir, f"char_{uid}")
        sheet_prompt = f"full body character sheet, {prompt}, detailed face, neutral expression, flat background, best quality"
        cmd = [
            "python", self.engine_script, # Changed to python for Windows
            "--prompt", sheet_prompt,
            "--output-dir", out_dir,
            "--width", str(self.conf["char"]["width"]),
            "--height", str(self.conf["char"]["height"]),
            "--num-frames", str(self.conf["char"]["frames"]),
            "--num-steps", str(self.conf["char"]["steps"]),
            "--guidance-scale", "8.0"
        ]
        self.run_command(cmd)
        return prompt 

    def generate_scenes(self, identity, uid):
        logger.info(">>> PHASE 2: CINEMATIC SEQUENCING")
        scene_dirs = []
        script = [
            f"Extreme wide shot, establishing shot, {identity}, cinematic lighting, volumetric fog, 8k masterpiece",
            f"Medium camera shot, {identity}, looking at camera, intense focus, depth of field, detailed armor, masterpiece",
            f"Dynamic action shot, low angle, {identity}, running, motion blur, particle effects, dramatic lighting"
        ]
        for i, scene_prompt in enumerate(script):
            logger.info(f"--- SCENE {i+1} ---")
            out_dir = os.path.join(self.output_dir, f"scene_{uid}_{i+1}")
            scene_dirs.append(out_dir)
            cmd = [
                "python", self.engine_script, # Changed to python for Windows
                "--prompt", scene_prompt,
                "--output-dir", out_dir,
                "--width", str(self.conf["scene"]["width"]),
                "--height", str(self.conf["scene"]["height"]),
                "--num-frames", str(self.conf["scene"]["frames"]),
                "--num-steps", str(self.conf["scene"]["steps"]),
                "--seed", str(42 + i)
            ]
            self.run_command(cmd)
        return scene_dirs

    def stitch(self, scene_dirs, uid):
        logger.info(">>> PHASE 3: FINAL CUT")
        try:
            from moviepy.editor import ImageSequenceClip, concatenate_videoclips
            clips = []
            for d in scene_dirs:
                frames = sorted(glob.glob(os.path.join(d, "frames", "*.png")))
                if frames: clips.append(ImageSequenceClip(frames, fps=10))
            if clips:
                final = concatenate_videoclips(clips, method="compose")
                out = os.path.join(self.output_dir, f"PHOENIX_V24_{uid}.mp4")
                final.write_videofile(out, codec="libx264", audio=False, verbose=False, logger=None)
                logger.info(f"DONE. FILE: {out}")
        except Exception as e:
            logger.error(f"Stitch Error: {e}")

    def run(self, prompt):
        uid = uuid.uuid4().hex[:6]
        self.generate_character(prompt, uid)
        scene_dirs = self.generate_scenes(prompt, uid)
        self.stitch(scene_dirs, uid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", default="production")
    args = parser.parse_args()
    PhoenixDirector(args.mode).run(args.prompt)
