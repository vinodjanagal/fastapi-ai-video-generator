# FILE: app/orchestrator/phoenix_director.py
from __future__ import annotations
import asyncio
import os
import json
import uuid
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from app.utils import run_subprocess_with_realtime_progress
from app.engine.character_sheet import build_character_prompt

logger = logging.getLogger("phoenix.director")
logger.setLevel(logging.INFO)

PROMPT_WORD_CAP = 55
COMPACT_NEGATIVE = "blurry, deformed, extra limbs, watermark, text, low quality, bad anatomy, ugly, distorted, static, still image"

# CRITICAL: This must match where you downloaded the model
LOCAL_ZEROSCOPE_PATH = os.path.abspath("zeroscope_cache")

@dataclass
class Scene:
    index: int
    description: str
    composition: Dict[str, Any]
    camera_motion: str = "static"

@dataclass
class VideoContext:
    uid: str
    quote_text: str
    scene_dirs: List[str] = field(default_factory=list)
    identity_prompt: str = "" 

def extract_identity_features(prompt: str) -> str:
    words = prompt.split(',')
    return ", ".join(words[:5]).strip()

def sanitize_for_cli(text: str) -> str:
    return text.replace("\n", " ").replace('"', "'").strip()

def clean_and_cap_prompt(prompt: str) -> str:
    if not prompt: return ""
    clean = re.sub(r"\([^)]+:[\d\.]+\)", "", prompt).replace("(", "").replace(")", "")
    parts = [p.strip() for p in clean.split(",") if p.strip()]
    final = []
    word_count = 0
    for p in parts:
        if "camera" in p.lower() or "shot" in p.lower(): continue
        wc = len(re.findall(r"\w+", p))
        if word_count + wc > PROMPT_WORD_CAP: continue
        final.append(p)
        word_count += wc
    if word_count < PROMPT_WORD_CAP - 5:
        final.append("cinematic 8k")
    return ", ".join(final)

class PhoenixDirector:
    def __init__(self, project_root: str, output_dir: str, mode: Literal["production", "dry_run"] = "production", seed: int = 42):
        self.project_root = project_root
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.mode = mode
        self.seed = seed
        self.ctx: Optional[VideoContext] = None
        self.defaults = {
            "production": { "num_steps": 25, "guidance_scale": 7.5, "num_frames": 16, "width": 576, "height": 320, "char_width": 384, "char_height": 384, "char_steps": 20 },
            "dry_run": { "num_steps": 10, "guidance_scale": 7.0, "num_frames": 8, "width": 448, "height": 256, "char_width": 256, "char_height": 256, "char_steps": 10 }
        }
        self.animate_script = os.path.join(self.project_root, "app", "video_engines", "animate_diff_engine.py")
        self.zeroscope_script = os.path.join(self.project_root, "app", "video_engines", "zeroscope_engine.py")

    def _conf(self): return self.defaults[self.mode]

    async def render_video(self, quote_text: str):
        logger.info("=== PhoenixDirector V17.2 (Local ZeroScope) START ===")
        
        # VERIFY MODEL EXISTS BEFORE STARTING
        if not os.path.exists(LOCAL_ZEROSCOPE_PATH):
            raise RuntimeError(f"CRITICAL: ZeroScope model not found at {LOCAL_ZEROSCOPE_PATH}. Please run download command.")
            
        self.ctx = VideoContext(uid=uuid.uuid4().hex, quote_text=quote_text)
        conf = self._conf()

        sb_script = os.path.join(self.project_root, "app", "video_engines", "storyboard_engine.py")
        rc, out = await run_subprocess_with_realtime_progress([os.sys.executable, sb_script, "--quote", quote_text], timeout_per_line=86400)
        if rc != 0: raise RuntimeError("Storyboard failed")
        sb_data = json.loads(out).get("storyboard_data", {})
        scenes = [Scene(i, s["description"], s.get("composition", {}), s.get("camera_motion", "static")) for i, s in enumerate(sb_data.get("scenes", []))]

        char_src = sb_data.get("character_sheet", "")
        if char_src:
            raw_char, _ = build_character_prompt(char_src)
            self.ctx.identity_prompt = extract_identity_features(raw_char)
            logger.info(f"IDENTITY LOCKED: {self.ctx.identity_prompt}")
            char_prompt = sanitize_for_cli(clean_and_cap_prompt(raw_char))
            char_out_dir = os.path.join(self.output_dir, f"character_sheet_{self.ctx.uid}")
            self.ctx.scene_dirs.append(char_out_dir)
            cmd = [
                os.sys.executable, self.animate_script,
                "--prompt", char_prompt, "--negative-prompt", COMPACT_NEGATIVE,
                "--output-dir", char_out_dir, "--num-frames", "12", "--num-steps", str(conf["char_steps"]),
                "--width", str(conf["char_width"]), "--height", str(conf["char_height"]),
                "--seed", str(self.seed),
                "--base-model", "Lykon/dreamshaper-7",
                "--motion-adapter", "guoyww/animatediff-motion-adapter-v1-5-2"
            ]
            await run_subprocess_with_realtime_progress(cmd, timeout_per_line=86400)

        for scene in scenes:
            env = scene.composition.get("environment", "")
            desc = scene.description
            light = scene.composition.get("lighting", "")
            raw_prompt = f"{env}, {self.ctx.identity_prompt}, {desc}, {light}"
            prompt = sanitize_for_cli(clean_and_cap_prompt(raw_prompt))
            logger.info(f"SCENE {scene.index} PROMPT: {prompt}")
            out_dir = os.path.join(self.output_dir, f"scene_{scene.index + 1}_{self.ctx.uid}")
            self.ctx.scene_dirs.append(out_dir)

            cmd = [
                os.sys.executable, self.zeroscope_script,
                "--prompt", prompt, "--negative-prompt", COMPACT_NEGATIVE,
                "--output-dir", out_dir,
                "--num-frames", str(conf["num_frames"]),
                "--num-steps", str(conf["num_steps"]),
                "--width", str(conf["width"]), "--height", str(conf["height"]),
                "--seed", str(self.seed + scene.index + 1),
                "--guidance-scale", str(conf["guidance_scale"]),
                "--model-dir", LOCAL_ZEROSCOPE_PATH  # FORCE LOCAL
            ]
            rc3, out3 = await run_subprocess_with_realtime_progress(cmd, timeout_per_line=14400) # 4 Hour Timeout

        return {"scene_dirs": self.ctx.scene_dirs, "scenes": [{"description": s.description, "camera_motion": s.camera_motion} for s in scenes]}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quote", required=True)
    parser.add_argument("--mode", choices=["production", "dry_run"], default="dry_run")
    parser.add_argument("--output-dir", default="phoenix_output")
    args = parser.parse_args()
    async def _run():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        d = PhoenixDirector(project_root=root, output_dir=args.output_dir, mode=args.mode)
        if args.mode == "dry_run": print(json.dumps(await d.dry_run_from_text(args.quote), indent=2))
        else: print(json.dumps(await d.render_video(args.quote), indent=2))
    asyncio.run(_run())