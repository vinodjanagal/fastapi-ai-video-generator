# dry_run.py  (repo root)
import argparse
import asyncio
import logging
from pathlib import Path

from app.orchestrator.phoenix_director import PhoenixDirector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [DRY_RUN] - %(message)s"
)
logger = logging.getLogger("dry_run")


async def main(quote: str, output_dir: str):
    project_root = str(Path(__file__).parent.resolve())

    logger.info("=== Phoenix V7.3 DRY-RUN START ===")
    director = PhoenixDirector(
        project_root=project_root,
        output_dir=output_dir,
        mode="dry_run",
        seed=42,
    )

    preview = await director.dry_run_from_text(quote)
    if preview:
        logger.info(f"=== Dry-run preview generated ===\n{preview}")
    else:
        logger.error("Dry-run failed. Review logs above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phoenix V7.3 Smart Dry-Run CLI (uses real cinematic brain)."
    )
    parser.add_argument("-q", "--quote", required=True, help="Quote text to preview.")
    parser.add_argument(
        "-o", "--output-dir",
        default="dry_run_output",
        help="Directory for dry-run output frames.",
    )

    args = parser.parse_args()
    asyncio.run(main(args.quote, args.output_dir))
