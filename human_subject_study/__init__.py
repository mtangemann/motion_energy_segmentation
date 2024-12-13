"""Random Dot Shape Identification."""

from pathlib import Path

from .dataset import (
    RandomDotShapeIdentificationDataset,
    RandomDotShapeIdentificationSample,
)

with open(Path(__file__).parent.parent / "VERSION") as version_file:
    __version__ = version_file.read().strip()

__all__ = [
    "RandomDotShapeIdentificationDataset",
    "RandomDotShapeIdentificationSample",
]
