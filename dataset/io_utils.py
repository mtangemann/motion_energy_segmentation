"""Utilities for loading and saving data."""

import json
import zipfile
from pathlib import Path

import numpy as np
import PIL.Image
import pycocotools.mask


def save_video(video: np.ndarray, path: Path) -> None:
    """Save a video to a zip file.
    
    The zip file will contain the frames following the pattern `%05d.png`.
    """
    with zipfile.ZipFile(path, "w") as archive:
        for frame_index, frame in enumerate(video):
            with archive.open(f"{frame_index:05d}.png", "w") as file:
                PIL.Image.fromarray(frame).save(file, format="png")


def save_segmentation(segmentation: np.ndarray, path: Path) -> None:
    """Save video segmentation masks to a json file using the COCO format."""
    json_data = []

    for frame in segmentation:
        encoded_frame = pycocotools.mask.encode(np.asfortranarray(frame))
        for mask in encoded_frame:
            mask["counts"] = mask["counts"].decode("utf-8")
        json_data.append(encoded_frame)

    with open(path, "w") as stream:
        json.dump(json_data, stream)


def save_flow(flow: np.ndarray, path: Path) -> None:
    """Save optical flow to npz."""
    min_value = flow.min()
    max_value = flow.max()
    flow = (flow - min_value) * 65535 / (max_value - min_value)
    flow = flow.astype(np.uint16)
    np.savez(path, flow=flow, min_value=min_value, max_value=max_value)
