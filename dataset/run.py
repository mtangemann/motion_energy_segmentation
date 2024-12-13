import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import coloredlogs
from executor import ExternalCommandFailed, execute

LOGGER = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")


def main():
    output_path = Path("output")
    os.makedirs(output_path, exist_ok=True)

    if (output_path / "done").is_file():
        LOGGER.info("Job is done already.")
        sys.exit(0)

    generate(output_path, "train", 900)
    generate(output_path, "val", 100)

    LOGGER.info("Done.")
    if not (output_path / "done").exists():
        with open(output_path / "done", "w") as done_marker:
            done_marker.write("completed\n")


def generate(output_path: Path, split: str, num_examples: int) -> None:
    os.makedirs(output_path / split, exist_ok=True)

    def num_completed_examples() -> int:
        return sum(1 for child in (output_path / split).iterdir() if child.is_dir())

    while num_completed_examples() < num_examples:
        LOGGER.info(
            "Generating... (%s, %d/%d)",
            split, num_completed_examples() + 1, num_examples,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            try:
                execute(f"python worker.py --job-dir={tempdir} --split {split}")

            except ExternalCommandFailed as error:
                LOGGER.error("Rendering failed: %s", error)

            else:
                video_id = str(hex(random.randint(0, 2**32-1)))[2:].zfill(8)
                if num_completed_examples() < num_examples:
                    shutil.copytree(tempdir, output_path / split / video_id)


if __name__ == "__main__":
    main()
