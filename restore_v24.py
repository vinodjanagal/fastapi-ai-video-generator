import os

# --- 1. THE DIRECTOR CODE (V24 ANCHORED) ---
DIRECTOR_CODE = r'''import os
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
        self.generate_scenes(prompt, uid)
        self.stitch(self.generate_scenes(prompt, uid), uid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", default="production")
    args = parser.parse_args()
    PhoenixDirector(args.mode).run(args.prompt)
'''

# --- 2. THE ENGINE CODE (T4 OPTIMIZED) ---
ENGINE_CODE = r'''import argparse
import torch
import os
import logging
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phoenix.engine")

def run(args):
    print(f">> ENGINE START: {args.width}x{args.height} | Steps: {args.num_steps}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL: CUDA GPU NOT DETECTED.")
        
    device = "cuda"
    torch.use_deterministic_algorithms(False)
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        "Lykon/dreamshaper-7",
        motion_adapter=adapter,
        torch_dtype=torch.float16
    ).to(device)
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
    pipe.enable_vae_slicing() 
    
    generator = torch.Generator("cpu").manual_seed(args.seed)
    
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        width=args.width,
        height=args.height,
        generator=generator
    )

    frames = output.frames[0]
    os.makedirs(args.output_dir, exist_ok=True)
    
    frame_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(frame_dir, f"{i:04d}.png"))
        
    print(f">> SUCCESS. Output saved to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="bad quality")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()
    run(args)
'''

# --- 3. THE SETUP SCRIPT ---
SETUP_CODE = r'''
pip install -q "protobuf==3.20.3" "moviepy==1.0.3" "imageio[ffmpeg]" 
pip install -q diffusers transformers accelerate safetensors opencv-python groq
python app/orchestrator/phoenix_director.py --prompt "$1" --mode production
'''

def write_file(path, content):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ RESTORED: {path}")

def restore():
    print(">>> PHOENIX V24 RESTORATION INITIATED...")
    write_file("app/orchestrator/phoenix_director.py", DIRECTOR_CODE)
    write_file("app/video_engines/animate_diff_engine.py", ENGINE_CODE)
    write_file("setup_and_run.sh", SETUP_CODE)
    print(">>> SYSTEM ONLINE. FILES RECONSTRUCTED.")

if __name__ == "__main__":
    restore()