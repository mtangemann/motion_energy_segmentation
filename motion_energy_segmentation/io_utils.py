"""Utilities for loading and saving data."""

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import PIL.Image
import pycocotools.mask
import torch
from numpy.typing import NDArray


def load_json(
    path: str | os.PathLike,
    frame_index: int | slice | None = None,
    format: str = "ignored",
) -> dict | list:
    """Loads a json file.
    
    Parameters:
        path: Path to a json file.
        frame_index: If specified, load only the frames specified by the index.
            Otherwise load the entire video.
        format: Ignored. This parameter is only used to make the function signature
            compatible with other `load_*` functions.
    
    Returns:
        The loaded json data.
    """
    del format  # Unused parameter.

    with open(path, "r") as stream:
        data = json.load(stream)

    if frame_index is not None:
        if not isinstance(data, list):
            raise ValueError("Expected json to be a list of frames.")
        data = data[frame_index]
    
    return data


def load_video(
    path: str | os.PathLike,
    frame_index: int | slice | None = None,
    format: Literal["numpy", "torch"] = "numpy",
) -> np.ndarray | torch.Tensor:
    """Loads a video from a zip file.
    
    The zip file is expected to contain the frames as png or jpg files, using the frame
    index as file name (`%05d.png` or `%05d.jpg`). For example:

    ```
    video.zip
    |- 00000.png
    |- 00001.png
    |- 00002.png
    |- ...
    ```

    Parameters:
        path: Path to a zip file containing the video frames as png or jpg images.
        frame_index: If specified, load only the frames specified by the index.
            Otherwise load the entire video.
        format: Array format. `"numpy"` or `"torch"`.

    Returns:
        The video as a uint8 array. Depending on the `format` parameter, either a
            numpy array of shape `(T, H, W, 3)` or a torch tensor of shape
            `(3, T, H, W)`. The time dimension `T` is omitted if `frame_index` is an
            integer.
    """
    if frame_index is None:
        frame_index = slice(None)

    rgb_frames = []

    with zipfile.ZipFile(path, "r") as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")
        )

        if isinstance(frame_index, int):
            names = [names[frame_index]]
        else:
            names = names[frame_index]

        for name in names:
            with archive.open(name, "r") as file:
                image = PIL.Image.open(file, formats=["png", "jpeg"])
                rgb_frames.append(np.array(image))

    if isinstance(frame_index, int):
        video = rgb_frames[0]
    else:
        video = np.stack(rgb_frames)

    if format == "numpy":
        return video
    elif format == "torch":
        return torch.from_numpy(video).movedim(-1, 0)
    else:
        raise ValueError(f"Unknown array format: {format}")


def save_video(video: np.ndarray | torch.Tensor, path: str | os.PathLike) -> None:
    """Saves a video to a zip file.
    
    The zip file will contain the frames as png images, using the frame index as file
    name (`%05d.png`). For example:

    ```
    video.zip
    |- 00000.png
    |- 00001.png
    |- 00002.png
    |- ...
    ```

    Parameters:
        video: Video as uint8 array. Either a numpy array of shape `(T, H, W, 3)` or a
            torch tensor of shape `(3, T, H, W)`.
        path: Output file path. This path must not exist yet.

    Raises:
        FileExistsError: If the given `path` exists already.
    """
    path = Path(path)

    if path.exists():
        raise FileExistsError(f"Output path exists: {path}")

    if isinstance(video, torch.Tensor):
        video = video.permute(1, 2, 3, 0).numpy()
    
    if video.dtype != np.uint8:
        raise ValueError(
            "Expected video to be a uint8 array, but got an array of type "
            f"{video.dtype}."
        )

    with zipfile.ZipFile(path, "w") as archive:
        for frame_index, frame in enumerate(video):
            with archive.open(f"{frame_index:05d}.png", "w") as file:
                PIL.Image.fromarray(frame).save(file, format="png")


def load_segmentation(
    path: str | os.PathLike,
    frame_index: int | slice | None = None,
    format: Literal["numpy", "torch", "list"] = "numpy",
) -> np.ndarray | torch.Tensor | list[np.ndarray | None] | None:
    """Loads video segmentation masks from a json file using the COCO format.
    
    This saves all masks for all frames to json uses the run-length encoding of the COCO
    dataset. For example:

    ```js
    [
        [
            { "size": [256, 256], "counts": "..." },  // frame 0, mask 1
            { "size": [256, 256], "counts": "..." }   // frame 0, mask 2
        ],
        [
            { "size": [256, 256], "counts": "..." },  // frame 1, mask 1
            { "size": [256, 256], "counts": "..." }   // frame 1, mask 2
        ]
    ]
    ```

    The background is assumed to be not explicitely encoded. I.e., all pixels that do
    not belong to any of the masks are assigned to the background (mask 0).
    
    !!! warning
        This does not support overlapping masks. If multiple masks contain the same
        pixel for a frame, the respective pixel is assigned to the last mask containing
        that pixel.

    Parameters:
        path: Path to a json file.
        frame_index: If specified, load only the frames specified by the index.
            Otherwise load the entire video.
        format: Array format. `"numpy"` or `"torch"` or `"list"`. The format must be
            `"list"` if the segmentation is not available for all frames.

    Returns:
        The segmentation as a uint8 array. Depending on the `format` parameter, either a
            numpy array of shape `(T, H, W, 1)` or a torch tensor of shape
            `(1, T, H, W)`. The time dimension `T` is omitted if `frame_index` is an
            integer.
    """
    if frame_index is None:
        frame_index = slice(None)

    with open(path, "r") as stream:
        json_data = json.load(stream)

    if isinstance(frame_index, int):
        encoded_frames = [json_data[frame_index]]
    else:
        encoded_frames = json_data[frame_index]

    segmentation_frames: list[NDArray | None] = []

    for frame in encoded_frames:
        if frame is None:
            segmentation_frames.append(None)
        else:
            masks = pycocotools.mask.decode(frame)
            masks = masks * (np.arange(masks.shape[-1], dtype=np.uint8) + 1)
            segmentation_frame = np.max(masks, axis=-1, keepdims=True)
            segmentation_frames.append(segmentation_frame)

    if isinstance(frame_index, int):
        segmentation = segmentation_frames[0]
        if segmentation is None:
            return None
        if format == "numpy" or format == "list":
            return segmentation
        if format == "torch":
            return torch.from_numpy(segmentation).movedim(-1, 0)

    if format == "list":
        return segmentation_frames

    if any(frame is None for frame in segmentation_frames):
        raise ValueError(
            "Not all frames are annotated. Use format='list' to load them."
        )

    segmentation = np.stack(segmentation_frames)  # type: ignore

    if format == "numpy":
        return segmentation
    elif format == "torch":
        return torch.from_numpy(segmentation).movedim(-1, 0)
    else:
        raise ValueError(f"Unknown array format: {format}")


def save_segmentation(
    segmentation: np.ndarray | torch.Tensor | list[np.ndarray | None],
    path: str | os.PathLike,
) -> None:
    """Saves video segmentation masks to a json file using the COCO format.
    
    See [load_segmentation](.#common.io_utils.load_segmentation) for details about the
    format.

    Parameters:
        segmentation: Segmentation in one of the following formats. (1) A uint8
            numpy array of shape `(T, H, W, 1)` or a torch tensor of shape
            `(1, T, H, W)`. Each entry is an integer that represents the segment to
            which the pixel belongs. (2) A boolean numpy array of shape `(T, H, W, N)`
            or a torch tensor of shape `(N, T, H, W)`. Each entry is a boolean that
            represents whether the pixel belongs to the segment. (3) A list of numpy
            arrays corresponding to format (1) or (2) for each frame. The list may
            contain `None` for frames that are not annotated.
        path: Output file path. This path must not exist yet.

    Raises:
        FileExistsError: If the given `path` exists already.
    """
    path = Path(path)

    if path.exists():
        raise FileExistsError(f"Output path exists: {path}")

    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.permute(1, 2, 3, 0).numpy()

    masks = _segmentation_masks(segmentation)  # type: ignore
    
    if all(frame is None for frame in masks):
        raise ValueError("All frames are empty. At least one mask must be provided.")

    num_segments = max(frame.shape[-1] for frame in masks if frame is not None)
    if num_segments == 0:
        # If there are no masks, we encode a single empty mask for each frame so that
        # the decoder can infer the size of the frames.
        masks = [
            None if frame is None else np.zeros((*frame.shape[:-1], 1), dtype=bool)
            for frame in masks
        ]

    json_data: list[Any] = []

    for frame in masks:
        if frame is None:
            json_data.append(None)
            continue

        encoded_frame = []

        for mask_index in range(frame.shape[-1]):
            mask = np.asfortranarray(frame[:, :, mask_index])

            encoded_mask = pycocotools.mask.encode(mask)
            assert isinstance(encoded_mask["counts"], bytes)
            encoded_mask["counts"] = encoded_mask["counts"].decode("utf-8")

            encoded_frame.append(encoded_mask)

        json_data.append(encoded_frame)

    with open(path, "w") as stream:
        json.dump(json_data, stream)


def _segmentation_masks(
    segmentation: np.ndarray | list[np.ndarray | None],
) -> list[np.ndarray | None]:
    masks = []

    for frame in segmentation:
        if frame is None or frame.dtype == bool:
            masks.append(frame)

        elif frame.dtype == np.uint8:            
            num_masks = np.max(frame)
            frame_masks = np.zeros((*frame.shape[:-1], num_masks), dtype=bool)

            for mask_index in range(1, num_masks + 1):
                frame_masks[..., mask_index - 1] = frame[..., 0] == mask_index

            masks.append(frame_masks)

        else:
            raise ValueError(
                "Expected segmentation of dtype bool or uint8, "
                f"but got an array of type {frame.dtype}."
            )

    return masks


def load_flow(
    path: str | os.PathLike,
    frame_index: int | slice | None = None,
    format: Literal["numpy", "torch"] = "numpy",
) -> np.ndarray | torch.Tensor:
    """Loads optical flow from a npz file.

    The optical flow format is the same as used by
    [kubric](https://github.com/google-research/kubric). The npz contains two scalars
    `min_value` and `max_value`; the uint16 `flow` array contains quantized flow in the
    specified range. I.e., the actual flow values are obtained by:

    ```python
    flow = (quantized_flow / 65535) * (max_value - min_value) + min_value
    ```

    Parameters:
        path: Path to a json file.
        frame_index: If specified, load only the frames specified by the index.
            Otherwise load the entire video.
        format: Array format. `"numpy"` or `"torch"`.

    Returns:
        The optical flow as a float32 array. Depending on the `format` parameter, either
            a numpy array of shape `(T, H, W, 2)` or a torch tensor of shape
            `(2, T, H, W)`. The time dimension `T` is omitted if `frame_index` is an
            integer.
    """
    if frame_index is None:
        frame_index = slice(None)

    data = np.load(path)
    min_value = data["min_value"]
    max_value = data["max_value"]
    flow = data["flow"][frame_index]
    flow = flow.astype(np.float32) / 65535 * (max_value - min_value) + min_value

    if format == "numpy":
        return flow
    elif format == "torch":
        return torch.from_numpy(flow).movedim(-1, 0)
    else:
        raise ValueError(f"Unknown array format: {format}")


def save_flow(flow: np.ndarray | torch.Tensor, path: str | os.PathLike) -> None:
    """Saves optical flow to a npz file.

    Parameters:
        flow: Optical flow as float array. Either a numpy array of shape `(T, H, W, 2)`
            or a torch tensor of shape `(2, T, H, W)`.
        path: Output file path. This path must not exist yet.

    Raises:
        FileExistsError: If the given `path` exists already.
    """
    path = Path(path)

    if path.exists():
        raise FileExistsError(f"Output path exists: {path}")

    if isinstance(flow, torch.Tensor):
        flow = flow.permute(1, 2, 3, 0).numpy()

    # From now, flow always is a numpy array. Mypy doesn't get this, so we need to
    # disable type checking for some lines in the following.

    if not np.issubdtype(flow.dtype, np.floating):  # type: ignore
        raise ValueError(
           f"Expected flow to be a float array, but got an array of type {flow.dtype}."
        )

    min_value = flow.min()
    max_value = flow.max()
    flow = (flow - min_value) * 65535 / (max_value - min_value)
    flow = flow.astype(np.uint16)  # type: ignore
    np.savez(path, flow=flow, min_value=min_value, max_value=max_value)


def save_depth(depth: np.ndarray | torch.Tensor, path: str | os.PathLike) -> None:
    """Saves depth maps to a npz file.

    Parameters:
        depth: Depth maps as float array. Either a numpy array of shape `(T, H, W, 1)`
            or a torch tensor of shape `(1, T, H, W)`.
        path: Output file path. This path must not exist yet.
    
    Raises:
        FileExistsError: If the given `path` exists already.
    """
    path = Path(path)

    if path.exists():
        raise FileExistsError(f"Output path exists: {path}")

    if isinstance(depth, torch.Tensor):
        depth = depth.permute(1, 2, 3, 0).numpy()

    if not np.issubdtype(depth.dtype, np.floating):  # type: ignore
        raise ValueError(
            "Expected depth to be a float array, but got an array of type "
            f"{depth.dtype}."
        )

    np.savez(path, depth=depth)


def load_depth(
    path: str | os.PathLike,
    frame_index: int | slice | None = None,
    format: Literal["numpy", "torch"] = "numpy",
) -> np.ndarray | torch.Tensor:
    """Loads depth maps from a npz file.

    Parameters:
        path: Path to a npz file.
        frame_index: If specified, load only the frames specified by the index.
            Otherwise load the entire video.
        format: Array format. `"numpy"` or `"torch"`.
    
    Returns:
        The depth maps as a float32 array. Depending on the `format` parameter, either
            a numpy array of shape `(T, H, W, 1)` or a torch tensor of shape
            `(1, T, H, W)`. The time dimension `T` is omitted if `frame_index` is an
            integer.
    """
    if frame_index is None:
        frame_index = slice(None)

    data = np.load(path)
    depth = data["depth"][frame_index]

    if format == "numpy":
        return depth
    elif format == "torch":
        return torch.from_numpy(depth).permute(3, 0, 1, 2)
    else:
        raise ValueError(f"Unknown array format: {format}")
