# app/orchestrator/phoenix_director.py

from __future__ import annotations
import os
import json
import uuid
import logging
from typing import Optional, Literal, Dict, Any, List
from dataclasses import dataclass, field

from app.tasks import run_subprocess_streamed
from app.engine.parser import semantic_parser
from app.engine.prompt_builder import build_semantic_prompt
from app.engine.character_sheet import build_character_prompt
from app.engine.cinematics import classify_shot_type

logger = logging.getLogger("phoenix.director")
logger.setLevel(logging.INFO)


# ------------------------------------------------------
# DATA CLASSES
# ------------------------------------------------------
@dataclass
class Scene:
    index: int
    description: str
    composition: Dict[str, Any]
    duration: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class VideoContext:
    uid: str
    quote_text: str
    video_id: Optional[int] = None
    audio_path: Optional[str] = None
    timestamps_path: Optional[str] = None
    scene_dirs: List[str] = field(default_factory=list)


# ------------------------------------------------------
# PHOENIX DIRECTOR (DRY-RUN + PRODUCTION)
# ------------------------------------------------------
class PhoenixDirector:
    """
    Phoenix V9.3 Director:
    -------------------------------------------
    • Identical brain for production & dry-run
    • Uses semantic_parser, classify_shot_type,
      build_semantic_prompt, build_character_prompt.
    • Dry-run produces a FAST character preview
      using lightweight settings.
    """

    def __init__(
        self,
        project_root: str,
        output_dir: str,
        mode: Literal["production", "dry_run"] = "production",
        seed: int = 42,
    ):
        self.project_root = project_root
        self.output_dir = output_dir
        self.mode = mode
        self.seed = seed
        self.ctx: Optional[VideoContext] = None

        # Golden Defaults
        self.defaults = {
            "production": {
                "num_steps": 22,
                "guidance_scale": 6.0,
                "num_frames": 16,
                "width": 512,
                "height": 512,
                "ip_scale": 0.55,
            },
            "dry_run": {
                "num_steps": 20,
                "guidance_scale": 7.0,
                "num_frames": 12,
                "width": 384,
                "height": 384,
                "ip_scale": 0.10,
            },
        }

    def _conf(self):
        return self.defaults["dry_run"] if self.mode == "dry_run" else self.defaults["production"]


    # ------------------------------------------------------
    # DRY RUN ENTRYPOINT
    # ------------------------------------------------------
    async def dry_run_from_text(self, quote_text: str) -> Optional[str]:
        """
        Phoenix V9.3 DRY-RUN:
        • Calls storyboard_engine
        • Builds scene prompts using the REAL Cinematic Brain
        • Builds CHARACTER PROMPT using character_sheet logic
        • Generates ONE preview frame from AnimateDiff
        """

        logger.info(f"Phoenix dry_run start (mode={self.mode})")
        self.ctx = VideoContext(uid=uuid.uuid4().hex, quote_text=quote_text)

        # --------------------------------------------------
        # 1) RUN STORYBOARD ENGINE (LLM → Scenes + Character)
        # --------------------------------------------------
        storyboard_script = os.path.join(
            self.project_root, "app", "video_engines", "storyboard_engine.py"
        )

        sb_cmd = [os.sys.executable, storyboard_script, "--quote", quote_text]

        rc, out, err = await run_subprocess_streamed(sb_cmd)
        if rc != 0:
            logger.error(f"Storyboard engine failed: {err}")
            return None

        try:
            parsed = json.loads(out)

            # Scenes: accept several possible formats
            scenes = (
                parsed.get("scenes")
                or parsed.get("storyboard")
                or (parsed.get("storyboard_data") or {}).get("scenes")
            )

            character_sheet_prompt = (
                parsed.get("character_sheet")
                or parsed.get("character_sheet_prompt")
                or (parsed.get("storyboard_data") or {}).get("character_sheet")
            )

            if not scenes:
                logger.error("Storyboard returned no scenes.")
                return None

        except Exception as e:
            logger.error(f"Failed to parse storyboard JSON: {e}")
            logger.error(f"STDOUT:\n{out}")
            return None

        # --------------------------------------------------
        # 2) BUILD PRODUCTION-IDENTICAL PROMPTS (Brain)
        # --------------------------------------------------
        logger.info("Cinematic Brain prompts (dry-run preview):")
        for idx, scene in enumerate(scenes, start=1):
            desc = scene.get("description", "")
            sem = semantic_parser(desc)
            shot = classify_shot_type(desc, sem)
            pos, neg = build_semantic_prompt(desc, "", shot, sem)

            logger.info(f"[Scene {idx}] {pos[:300]}")

        # --------------------------------------------------
        # 3) CHARACTER PROMPT (SPECIAL BUILDER)
        # --------------------------------------------------
        if not character_sheet_prompt:
            logger.warning("No character sheet prompt detected.")
            return None

        final_char_pos, final_char_neg = build_character_prompt(character_sheet_prompt)

        logger.info(f"Character Sheet Prompt:\n{final_char_pos}")
        logger.info(f"Character Sheet NEG:\n{final_char_neg}")

        # --------------------------------------------------
        # 4) RUN ANIMATE DIFF ENGINE (FAST PREVIEW)
        # --------------------------------------------------
        conf = self._conf()
        ad_script = os.path.join(
            self.project_root, "app", "video_engines", "animate_diff_engine.py"
        )

        char_dir = os.path.join(self.output_dir, f"dry_character_{self.ctx.uid}")
        os.makedirs(char_dir, exist_ok=True)

        ad_cmd = [
            os.sys.executable,
            ad_script,
            "--prompt", final_char_pos,
            "--negative-prompt", final_char_neg,
            "--output-dir", char_dir,
            "--num-steps", str(conf["num_steps"]),
            "--guidance-scale", str(conf["guidance_scale"]),
            "--num-frames", str(conf["num_frames"]),
            "--width", str(conf["width"]),
            "--height", str(conf["height"]),
            "--seed", str(self.seed),
        ]

        logger.info("Calling AnimateDiff for dry-run preview...")
        rc2, out2, err2 = await run_subprocess_streamed(ad_cmd)

        if rc2 != 0:
            logger.error(f"AnimateDiff dry-run failed: {err2}")
            return None

        try:
            parsed_ad = json.loads(out2)
            frame_paths = parsed_ad.get("frame_paths", [])
            if not frame_paths:
                logger.error("AnimateDiff returned no frames.")
                return None

            preview_path = frame_paths[0]
            logger.info(f"Dry-run preview ready: {preview_path}")
            return preview_path

        except Exception as e:
            logger.error(f"Failed to parse AnimateDiff output: {e}")
            logger.error(f"STDOUT:\n{out2}")
            return None
