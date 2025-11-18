from __future__ import annotations
import asyncio
import os
import json
import uuid
import logging
from typing import Optional, Literal, Dict, Any, List

from dataclasses import dataclass, field

# subprocess runner (2-value return)
from app.utils import run_subprocess_with_realtime_progress

# “brain” modules
from app.engine.parser import semantic_parser
from app.engine.prompt_builder import build_semantic_prompt, BASE_NEGATIVE_PROMPT
from app.engine.character_sheet import build_character_prompt
from app.engine.cinematics import classify_shot_type

logger = logging.getLogger("phoenix.director")
logger.setLevel(logging.INFO)


# ------------------------------
# Data Classes
# ------------------------------
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


# ============================================================
# PhoenixDirector V10.6 — Stable CPU Version
# (Init-Image continuity only — correct for AnimateDiff)
# ============================================================
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

        # Stable settings for CPU AnimateDiff
        self.defaults = {
            "production": {
                "num_steps": 20,
                "guidance_scale": 6.0,
                "num_frames": 16,
                "width": 384,
                "height": 384,
                "init_strength": 0.55,   # best identity retention on CPU
            },
            "dry_run": {
                "num_steps": 20,
                "guidance_scale": 7.0,
                "num_frames": 4,
                "width": 384,
                "height": 384,
            },
        }

        self.engine_script = os.path.join(
            self.project_root,
            "app",
            "video_engines",
            "animate_diff_engine.py",
        )

    def _conf(self):
        return self.defaults[self.mode]

    # ---------------------------------------------------------
    # DRY RUN (fast preview)
    # ---------------------------------------------------------
    async def dry_run_from_text(self, text: str) -> Dict[str, Any]:
        sb_script = os.path.join(
            self.project_root,
            "app",
            "video_engines",
            "storyboard_engine.py",
        )

        rc, out = await run_subprocess_with_realtime_progress(
            [os.sys.executable, sb_script, "--quote", text],
            timeout_per_line=7200
        )
        if rc != 0:
            raise RuntimeError("Storyboard failed in dry_run.")

        return json.loads(out).get("storyboard_data", {})

    # ---------------------------------------------------------
    # FULL RENDER PIPELINE
    # ---------------------------------------------------------
    async def render_video(self, quote_text: str):
        logger.info("=== PhoenixDirector V10.6 RENDER START ===")

        self.ctx = VideoContext(uid=uuid.uuid4().hex, quote_text=quote_text)
        conf = self._conf()

        # --------------------------------------------------------------------
        # 1. STORYBOARD
        # --------------------------------------------------------------------
        sb_script = os.path.join(
            self.project_root,
            "app",
            "video_engines",
            "storyboard_engine.py",
        )

        rc, out = await run_subprocess_with_realtime_progress(
            [os.sys.executable, sb_script, "--quote", quote_text],
            timeout_per_line=7200
        )
        if rc != 0:
            raise RuntimeError("Storyboard engine failed.")

        sb_data = json.loads(out).get("storyboard_data", {})
        scenes_raw = sb_data.get("scenes", [])
        char_sheet_src = sb_data.get("character_sheet")

        if not scenes_raw:
            raise RuntimeError("Storyboard produced no scenes")

        scenes = [
            Scene(
                index=i,
                description=s["description"],
                composition=s.get("composition", {}),
                camera_motion=s.get("camera_motion", "static"),
            )
            for i, s in enumerate(scenes_raw)
        ]

        # --------------------------------------------------------------------
        # 2. CHARACTER SHEET  →  first continuity frame
        # --------------------------------------------------------------------
        last_frame = None

        if char_sheet_src:
            pos, neg = build_character_prompt(char_sheet_src)

            out_dir = os.path.join(
                self.output_dir,
                f"character_sheet_{self.ctx.uid}",
            )
            self.ctx.scene_dirs.append(out_dir)

            cmd = [
                os.sys.executable,
                self.engine_script,
                "--prompt", pos,
                "--negative-prompt", neg,
                "--output-dir", out_dir,
                "--num-frames", "12",
                "--num-steps", "25",
                "--width", str(conf["width"]),
                "--height", str(conf["height"]),
                "--seed", str(self.seed),
                "--base-model", "SG161222/Realistic_Vision_V5.1_noVAE",
            ]

            rc2, out2 = await run_subprocess_with_realtime_progress(cmd, timeout_per_line=7200)
            if rc2 != 0:
                raise RuntimeError("Character sheet render failed")

            js = json.loads(out2)
            fp = js.get("frame_paths", [])
            last_frame = fp[0] if fp else None

        # --------------------------------------------------------------------
        # 3. SCENE RENDERING (Init-Image continuity ONLY)
        # --------------------------------------------------------------------
        scene_defs: List[Dict[str, Any]] = []

        for scene in scenes:
            desc = scene.description

            sem = semantic_parser(desc)
            shot = classify_shot_type(desc, sem)

            pos, neg = build_semantic_prompt(
                desc,
                BASE_NEGATIVE_PROMPT,
                shot,
                sem,
                scene.index,
                len(scenes),
            )

            out_dir = os.path.join(
                self.output_dir,
                f"scene_{scene.index + 1}_{self.ctx.uid}",
            )
            self.ctx.scene_dirs.append(out_dir)

            cmd = [
                os.sys.executable,
                self.engine_script,
                "--prompt", pos,
                "--negative-prompt", neg,
                "--output-dir", out_dir,
                "--num-steps", str(conf["num_steps"]),
                "--guidance-scale", str(conf["guidance_scale"]),
                "--num-frames", str(conf["num_frames"]),
                "--width", str(conf["width"]),
                "--height", str(conf["height"]),
                "--seed", str(self.seed + scene.index + 1),
                "--base-model", "SG161222/Realistic_Vision_V5.1_noVAE",
            ]

            # -------------------------------------------------------------
            # CONTINUITY ENGINE (Stable AnimateDiff Version)
            # -------------------------------------------------------------
            if last_frame:
                cmd += [
                    "--init-image", last_frame,
                    "--strength", str(conf["init_strength"]),
                ]

            rc3, out3 = await run_subprocess_with_realtime_progress(cmd, timeout_per_line=7200)
            if rc3 != 0:
                logger.warning(f"Scene {scene.index + 1} failed — skipping.")
                continue

            js = json.loads(out3)
            frames = js.get("frame_paths", [])
            if frames:
                last_frame = frames[-1]

            scene_defs.append({
                "description": scene.description,
                "camera_motion": scene.camera_motion,
            })

        # --------------------------------------------------------------------
        # RETURN METADATA TO run_full_video.py
        # --------------------------------------------------------------------
        return {
            "scene_dirs": self.ctx.scene_dirs,
            "scenes": scene_defs,
        }


# ---------------------------------------------------------
# CLI (optional)
# ---------------------------------------------------------
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
        if args.mode == "dry_run":
            prev = await d.dry_run_from_text(args.quote)
            print(json.dumps(prev, indent=2))
        else:
            out = await d.render_video(args.quote)
            print(json.dumps(out, indent=2))

    asyncio.run(_run())
