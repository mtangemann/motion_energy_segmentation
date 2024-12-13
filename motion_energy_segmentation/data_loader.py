import logging
import os
import zipfile
from pathlib import Path
from typing import Iterable

import torch
import torch.utils.data

from . import io_utils

LOGGER = logging.getLogger(__name__)

_LOADERS = {
    "attributes/json": io_utils.load_json,
    "backward_flow": io_utils.load_flow,
    "bounding_boxes/json": io_utils.load_json,
    "forward_flow": io_utils.load_flow,
    "rgb": io_utils.load_video,
    "segmentation": io_utils.load_segmentation,
    "segmentation/json": io_utils.load_segmentation,
    "flow/npz": io_utils.load_flow,
    "video/zip": io_utils.load_video,
}


def build_data_loader(config: dict):
    dataset_kwargs = {
        key: value for key, value in config["dataset"].items()
        if not key.startswith("_")
    }
    dataset = VideoDataset(**dataset_kwargs)
    
    data_loader_kwargs = {
        key: value for key, value in config.items()
        if not key.startswith("_") and key != "dataset"
    }

    clip_length = data_loader_kwargs["clip_length"]
    if clip_length not in [2, 9]:
        raise ValueError(f"Unsupported clip length: {clip_length}")

    data_loader = ClipLoader(dataset, **data_loader_kwargs)

    return data_loader


class VideoDataset:
    """Provides access to a video dataset.

    Format:

    ```
    ├── train
    │   ├── example1
    │   │   ├── backward_flow.npz
    │   │   ├── forward_flow.npz
    │   │   ├── rgb.png.zip
    │   │   └── segmentation.json
    │   └── example2
    │       ├── backward_flow.npz
    │       ├── forward_flow.npz
    │       ├── rgb.png.zip
    │       └── segmentation.json
    └── test
        └── example3
            ├── backward_flow.npz
            ├── forward_flow.npz
            ├── rgb.png.zip
            └── segmentation.json
    ```

    The dataset is expected to consist of several subsets, each in a separate subfolder
    (`train` and `test` in the example). The examples are individual directories
    containing the example data. Each example may contain rgb video (`rgb.png.zip`),
    forward and backward optical flow (`forward_flow.npz`, `backward_flow.npz`) and
    segmentations (`segmentation.json`). The formats used to store these data are
    described in [`io_utils`](/reference/io_utils).

    The dataset instance provides access to the data. For iterating over the data, for
    example during training, a data loader has to be used.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        subset: str,
        features: Iterable[str] | None = None,
    ) -> None:
        """Initializes the dataset instance.
        
        Parameters:
            path: Path to the directory containing the datas
            subset: Video subset to load. The given path has to contain a directory with
                this name.
            features: If specified, only load the given features (e.g., `"rgb"` or
                `"forward_flow"`).
        """

        self.path = Path(path)
        self.subset = subset
        self.features = None if features is None else set(features)

        self._build_index()

    @staticmethod
    def from_config(config: dict) -> "VideoDataset":
        return VideoDataset(**config)

    def _build_index(self) -> None:
        LOGGER.info("Building dataset index ...")
        self._index = []
        
        for example_path in sorted((self.path / self.subset).iterdir()):
            if not example_path.is_dir():
                continue

            example_info = {}

            if (example_path / "rgb.jpg.zip").is_file():
                example_info["rgb"] = example_path / "rgb.jpg.zip"

            elif (example_path / "rgb.png.zip").is_file():
                example_info["rgb"] = example_path / "rgb.png.zip"

            if (example_path / "backward_flow.npz").is_file():
                example_info["backward_flow"] = example_path / "backward_flow.npz"

            if (example_path / "forward_flow.npz").is_file():
                example_info["forward_flow"] = example_path / "forward_flow.npz"

            if (example_path / "segmentation.json").is_file():
                example_info["segmentation"] = example_path / "segmentation.json"

            example_info["num_frames"] = _get_video_length(example_info["rgb"])  # type: ignore

            self._index.append(example_info)

        LOGGER.info("Building dataset index finished")

    def __len__(self) -> int:
        """Returns the number of videos in this dataset.

        When used by a model, the number of examples additionally depends on the data
        loader. For example, if the data loader treats each frame as an individual
        example, the length of the loaded data stream is larger than the length returned
        by this dataset.

        Returns:
            The number of videos in this dataset.
        """
        return len(self._index)

    @property
    def video_lengths(self) -> list[int]:
        """Returns the lengths (number of frames) of all videos in this dataset.

        Returns:
            The lengths (number of frames) of all videos in this dataset.
        """
        return [example_info["num_frames"] for example_info in self._index]  # type: ignore

    def load(
        self,
        video_index: int,
        frame_index: int | slice | None = None,
    ) -> dict[str, torch.Tensor]:
        """Loads data for an example.
        
        Parameters:
            video_index: The video to be loaded.
            frame_index: The frame or a range of frames to be loaded. If omitted
                (default), the entire video is loaded.

        Returns:
            The loaded data as a dictionary, for example:
                ```
                {
                    "backward_flow": torch.Tensor(...)  # shape (2, T, H, W), float
                    "forward_flow": torch.Tensor(...)   # shape (2, T, H, W), float
                    "rgb": torch.Tensor(...)            # shape (3, T, H, W), float
                    "segmentation": torch.Tensor(...)   # shape (1, T, H, W), uint8
                }
                ```

                The time dimension `T` is omitted if `frame_index` is an integer.
        """
        example_info = self._index[video_index]

        example = dict()

        for feature, path in example_info.items():
            if self.features is not None and feature not in self.features:
                continue
            if feature not in _LOADERS:
                continue

            data = _LOADERS[feature](path, frame_index, format="torch")  # type: ignore
            assert isinstance(data, torch.Tensor)
            example[feature] = data

        return example


    def __getitem__(
        self,
        video_index: int | slice | tuple[int | slice, int | slice | None],
        frame_index: int | slice | None = None
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Loads data for an example.
        
        Alias for [`load`](.#common.datasets.StandardVideoDataset.load).

        For compatibility with PyTorch data loaders this allows passing all arguments
        as a single tuple. Do not use this otherwise.
        """
        # If this is used with a PyTorch data loader, all arguments are passed as single
        # tuple that needs to be unpacked first.
        if isinstance(video_index, tuple):
            video_index, frame_index = video_index

        if isinstance(video_index, slice):
            video_indices = range(*video_index.indices(len(self)))
            return [self.load(i, frame_index) for i in video_indices]

        return self.load(video_index, frame_index)


def _get_video_length(path: Path) -> int:
    """Returns the number of frames for a video in a zip file."""
    with zipfile.ZipFile(path, "r") as archive:
        return sum(
            1 for name in archive.namelist()
            if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg")
        )


class ClipLoader:
    """Loads overlapping clips from a video dataset."""

    def __init__(
        self,
        dataset: VideoDataset,
        clip_length: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        repeat: bool = True,
        num_workers: int = 0,
    ) -> None:
        """Initializes the data loader.

        Parameters:
            dataset: A video dataset. Needs to provide a list of all video lengths as
                `video_lengths` property.
            clip_length: Length of each video clip (number of frames).
            batch_size: Number of examples per batch.
            drop_last: If True, the last batch will be excluded if it contains less than
                `batch_size` examples.
            repeat: If True, repeat the dataset.
            num_workers: Number of data loading workers.
        """
        self._batch_sampler = _ClipBatchSampler(
            dataset, clip_length, batch_size, drop_last, shuffle  # type: ignore
        )
        self._base_loader = torch.utils.data.DataLoader(
            dataset,  # type: ignore
            batch_sampler=self._batch_sampler,
            num_workers=num_workers,
        )
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.repeat = repeat

    @property
    def num_examples(self) -> int:
        """Returns the number of unique examples (clips)."""
        return len(self._batch_sampler)

    @property
    def num_batches(self) -> int:
        """Returns the number of batches (ignoring repeats)."""
        count = self.num_examples // self.batch_size
        if self.num_examples % self.batch_size > 0 and not self.drop_last:
            count = count + 1
        return count

    def __iter__(self):
        if not self.repeat:
            yield from self._base_loader
        else:
            while True:
                yield from self._base_loader


class _ClipBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: VideoDataset,
        clip_length: int,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
    ) -> None:
        super().__init__(dataset)  # type: ignore

        if not hasattr(dataset, "video_lengths"):
            raise ValueError(
                f"The given dataset does not provide video lengths: {dataset}"
            )

        self.clips = []
        for video_index, video_length in enumerate(dataset.video_lengths):
            self.clips += [
                (video_index, slice(start, start + clip_length))
                for start in range(video_length - clip_length + 1)
            ]

        self._base_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(self.clips) if shuffle else self.clips,  # type: ignore
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self):
        for batch in self._base_sampler:
            yield [self.clips[index] for index in batch]
