from __future__ import annotations
import asyncio
import os
import json
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List

from app.utils import run_subprocess_with_realtime_progress
from app.engine.parser import semantic_parser
from app.engine.prompt_builder import build_semantic_prompt, BASE_NEGATIVE_PROMPT
from app.engine.character_sheet import build_character_prompt
from app.engine.cinematics import classify_shot_type

logger = logging.getLogger("phoenix.director")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# PhoenixDirector
# ---------------------------------------------------------------------
class PhoenixDirector:
    def __init__(
        self,
        project_root: str,
        output_dir: str,
        mode: Literal["production", "dry_run"] = "production",
        seed: int = 42,
    ):

        self.project_root = project_root
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.mode = mode
        self.seed = seed
        self.ctx: Optional[VideoContext] = None

        # Golden settings for CPU-first environments
        self.defaults = {
            "production": {
                "num_steps": 15,
                "guidance_scale": 6.0,
                "num_frames": 16,
                "width": 384,
                "height": 384,
                "init_strength": 0.45,
            },
            "dry_run": {
                "num_steps": 10,
                "guidance_scale": 7.0,
                "num_frames": 8,
                "width": 256,
                "height": 256,
                "init_strength": 0.45,
            },
        }

        self.engine_script = os.path.join(
            self.project_root,
            "app",
            "video_engines",
            "animate_diff_engine.py",
        )

    # ------------------------------------------------------------------
    def _conf(self):
        return self.defaults[self.mode]

    # ------------------------------------------------------------------
    def clean_prompt(self, prompt: str) -> str:
        """
        Removes duplicated descriptors like:
            "cinematic, cinematic lighting, cinematic"
        Prevents CLIP truncation and improves clarity.
        """
        parts = [p.strip() for p in prompt.split(",")]
        seen = set()
        out = []
        for p in parts:
            norm = p.lower()
            if norm not in seen and norm != "":
                seen.add(norm)
                out.append(p)
        return ", ".join(out)

    # ------------------------------------------------------------------
    async def dry_run_from_text(self, text: str) -> Dict[str, Any]:
        sb_script = os.path.join(self.project_root, "app", "video_engines", "storyboard_engine.py")
        rc, out = await run_subprocess_with_realtime_progress(
            [os.sys.executable, sb_script, "--quote", text], timeout_per_line=86400
        )
        if rc != 0:
            raise RuntimeError("Storyboard failed during dry_run")
        return json.loads(out).get("storyboard_data", {})

    # ------------------------------------------------------------------
    async def render_video(self, quote_text: str):
        logger.info("=== PhoenixDirector V11 — FULL RENDER START ===")
        self.ctx = VideoContext(uid=uuid.uuid4().hex, quote_text=quote_text)
        conf = self._conf()

        # ===================================================================
        # 1) STORYBOARD GENERATION
        # ===================================================================
        sb_script = os.path.join(self.project_root, "app", "video_engines", "storyboard_engine.py")

        rc, out = await run_subprocess_with_realtime_progress(
            [os.sys.executable, sb_script, "--quote", quote_text],
            timeout_per_line=86400,
        )
        if rc != 0:
            raise RuntimeError("Storyboard engine failed")

        sb_data = json.loads(out).get("storyboard_data", {})
        raw_scenes = sb_data.get("scenes", [])
        char_src = sb_data.get("character_sheet")

        if not raw_scenes:
            raise RuntimeError("Storyboard produced 0 scenes — cannot proceed.")

        scenes: List[Scene] = [
            Scene(
                index=i,
                description=s["description"],
                composition=s.get("composition", {}),
                camera_motion=s.get("camera_motion", "static"),
            )
            for i, s in enumerate(raw_scenes)
        ]

        # ===================================================================
        # 2) CHARACTER SHEET → provides LAST_FRAME for continuity
        # ===================================================================
        last_frame = None

        if char_src:
            pos, neg = build_character_prompt(char_src)

            # Clean prompt (duplicate-word removal)
            pos = self.clean_prompt(pos)

            out_dir = os.path.join(self.output_dir, f"character_sheet_{self.ctx.uid}")
            self.ctx.scene_dirs.append(out_dir)

            cmd = [
                os.sys.executable,
                self.engine_script,
                "--prompt", pos,
                "--negative-prompt", neg,
                "--output-dir", out_dir,
                "--num-frames", "12",
                "--num-steps", "20",
                "--width", str(conf["width"]),
                "--height", str(conf["height"]),
                "--seed", str(self.seed),
                "--base-model", "SG161222/Realistic_Vision_V5.1_noVAE",
                "--motion-adapter", "guoyww/animatediff-motion-adapter-v1-5-2",
            ]

            rc2, out2 = await run_subprocess_with_realtime_progress(cmd, timeout_per_line=86400)
            if rc2 != 0:
                raise RuntimeError("Character sheet render failed")

            js = json.loads(out2)
            fpaths = js.get("frame_paths", [])
            if fpaths:
                last_frame = fpaths[-1]

        # ===================================================================
        # 3) RENDER EACH SCENE (full AnimateDiff motion)
        # ===================================================================
        scene_defs: List[Dict[str, Any]] = []

        for scene in scenes:
            desc = scene.description

            # Semantic parsing + shot classification
            sem = semantic_parser(desc)
            shot_type = classify_shot_type(desc, sem)

            pos, neg = build_semantic_prompt(
                desc,
                BASE_NEGATIVE_PROMPT,
                shot_type,
                sem,
                scene.index,
                len(scenes),
            )

            # Clean prompt to avoid CLIP overflow
            pos = self.clean_prompt(pos)

            # Output directory for scene
            out_dir = os.path.join(self.output_dir, f"scene_{scene.index+1}_{self.ctx.uid}")
            self.ctx.scene_dirs.append(out_dir)

            # Base command for AnimateDiff worker
            cmd = [
                os.sys.executable,
                self.engine_script,
                "--prompt", pos,
                "--negative-prompt", neg,
                "--output-dir", out_dir,
                "--num-frames", str(conf["num_frames"]),
                "--num-steps", str(conf["num_steps"]),
                "--width", str(conf["width"]),
                "--height", str(conf["height"]),
                "--seed", str(self.seed + scene.index + 1),
                "--guidance-scale", str(conf["guidance_scale"]),
                "--base-model", "SG161222/Realistic_Vision_V5.1_noVAE",
                "--motion-adapter", "guoyww/animatediff-motion-adapter-v1-5-2",
            ]

            # ----- CONTINUITY ANCHOR -----
            if last_frame:
                if scene.index == 0:
                    # Very strong anchor for the establishing scene
                    strength_val = 0.65
                else:
                    # Softer continuity for all later scenes
                    strength_val = conf.get("init_strength", 0.45)

                cmd += ["--init-image", last_frame, "--strength", str(strength_val)]

            # Run AnimateDiff
            rc3, out3 = await run_subprocess_with_realtime_progress(cmd, timeout_per_line=7200)
            if rc3 != 0:
                logger.warning(f"Scene {scene.index+1} failed — continuing with next.")
                continue

            js = json.loads(out3)
            scene_frames = js.get("frame_paths", [])
            if scene_frames:
                last_frame = scene_frames[-1]   # anchor next scene on final frame

            scene_defs.append({
                "description": scene.description,
                "camera_motion": scene.camera_motion,
            })

        # ===================================================================
        # END — Return structured scene info
        # ===================================================================
        return {
            "scene_dirs": self.ctx.scene_dirs,
            "scenes": scene_defs,
        }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quote", required=True)
    parser.add_argument("--mode", choices=["production", "dry_run"], default="production")
    parser.add_argument("--output-dir", default="phoenix_output")

    args = parser.parse_args()

    async def _run():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        director = PhoenixDirector(project_root=root, output_dir=args.output_dir, mode=args.mode)

        if args.mode == "dry_run":
            data = await director.dry_run_from_text(args.quote)
            print(json.dumps(data, indent=2))
        else:
            out = await director.render_video(args.quote)
            print(json.dumps(out, indent=2))

    asyncio.run(_run())
