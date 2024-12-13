"""Methods for creating human subject experiments."""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import PIL.Image
from executor import execute

from dataset import RandomDotShapeIdentificationSample

LOGGER = logging.getLogger(__name__)


def generate_experiment(
    output_path: str | Path,
    train_samples: Iterable[RandomDotShapeIdentificationSample],
    test_samples: Iterable[RandomDotShapeIdentificationSample],
    catch_samples: Iterable[RandomDotShapeIdentificationSample],
) -> None:
    """Generate a human subject experiment."""
    output_path = Path(output_path)

    os.makedirs(output_path, exist_ok=True)

    _copy_files(output_path)
    _export_sample_for_instructions(output_path)

    trials_data = {
        "train": _export_samples(output_path, "train", train_samples),
        "test": _export_samples(output_path, "test", test_samples),
        "catch": _export_samples(output_path, "catch", catch_samples),
    }
    with open(output_path / "static" / "trials.json", "w") as f:
        json.dump(trials_data, f)

    LOGGER.info("Experiment generated successfully.")


def _export_samples(
    output_path: Path,
    subset: str,
    samples: Iterable[RandomDotShapeIdentificationSample],
) -> list[dict]:
    """Export the samples."""
    LOGGER.info("Exporting samples ...\n\toutput_path = %s", output_path)
    trials_data = []

    for index, trial in enumerate(samples):
        trial_path = output_path / "data" / subset / f"{index}"
        trial_path.mkdir(parents=True)
        trial.save(trial_path / "trial.pt")

        _render(
            trial.stimulus, trial_path / "video.mp4", frame_rate=trial.frame_rate
        )
        for choice_index, choice in enumerate(trial.choices):
            choice = (choice * 255).astype(np.uint8)
            PIL.Image.fromarray(choice).save(
                trial_path / f"choice_{choice_index}.png"
            )

        trials_data.append({
            "stimulus": [f"{trial_path.relative_to(output_path)}/video.mp4"],
            "choices": [
                f'{trial_path.relative_to(output_path)}/choice_{choice_index}.png'
                for choice_index in range(len(trial.choices))
            ],
            "correct_response": int(trial.target),
            "number_of_informative_dots": trial.number_of_informative_dots(),
        })
    
    return trials_data


def _copy_files(output_path: Path) -> None:
    """Copy the files required for the experiment."""
    LOGGER.info("Copying experiment files ...")
    experiment_path = Path(__file__).parent / "experiment"
    for child in experiment_path.iterdir():
        destination = output_path / child.name
        if child.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(child, destination)
        else:
            shutil.copy(child, destination)


def _export_sample_for_instructions(output_path: Path) -> None:
    LOGGER.info("Exporting sample for instructions ...")
    sample = _create_sample_for_instructions()
    _render(
        sample.stimulus,
        output_path / "static" / "example.mp4",
        frame_rate=sample.frame_rate,
    )
    for choice_index, choice in enumerate(sample.choices):
        choice = (choice * 255).astype(np.uint8)
        PIL.Image.fromarray(choice).save(
            output_path / "static" / f"example_choice_{choice_index}.png"
        )


def _create_sample_for_instructions() -> RandomDotShapeIdentificationSample:
    """Creates a sample for the experiment instructions."""
    # x,y coordinates of the corners as array of shape (2, N)
    square = np.array([
        [-1.0, -1.0, 1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0, -1.0, -1.0],
    ])

    triangle = np.array([
        [-1.0, 0.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0, -1.0],
    ])

    return RandomDotShapeIdentificationSample(
        shapes=[square, triangle],
        shapes_smooth=[False, False],
        target=0,
        orientation=0,
        scale=1/3,
        background_motion=(-0.5, 0.0),
        foreground_motion=(0.5, 0.0),
        size=(300, 256, 256),
        duration=10.0,
        random_dot_density=200,
        random_dot_lifetime=2.0,
        random_dot_seed=0,
    )


def _render(
    video: np.ndarray, output_path: Path, frame_rate: float = 30.0
) -> None:
    if video.dtype == np.float32:
        video = (video * 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tempdir:
        for frame_index, frame in enumerate(video):
            frame = PIL.Image.fromarray(frame)
            frame.save(f"{tempdir}/{frame_index:04d}.png")
    
        if output_path.name.endswith(".gif"):
            execute(
                f"ffmpeg -y -r {frame_rate} -i {tempdir}/%04d.png "
                "-filter_complex 'split [a][b]; [a] palettegen [p]; [b][p] paletteuse' "
                f"{output_path}"
            )
        else:
            execute(
                f"ffmpeg -y -r {frame_rate} -i {tempdir}/%04d.png "
                "-c:v libx264 -crf 18 -preset ultrafast -pix_fmt yuv420p "
                f"{output_path}"
            )