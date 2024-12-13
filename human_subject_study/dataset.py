"""Random Dot Shape Identification dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageDraw
import torch
import torch.nn.functional as F
from idsprites import InfiniteDSprites

from random_utils import sample_shape


@dataclass
class RandomDotShapeIdentificationSample:
    """A sample from the Random Dot Shape Identification dataset.

    All properties are expressed using "stimulus coordinates" if not specified
    otherwise. On all axes (x, y, t), the stimulus spans the range [-1, 1]. I.e., the
    origin (0, 0, 0) is at the center of the middle frame of the video.
    """

    shapes: list[npt.NDArray]
    """The shapes used in the trial."""

    shapes_smooth: list[bool]
    """Indicates whether the shapes are smoothed."""

    orientation: float
    """The orientation of the shapes in radians."""

    scale: float
    """The scale of the shapes."""

    background_motion: tuple[float, float]
    """The motion of the background."""

    foreground_motion: tuple[float, float]
    """The motion of the foreground object."""

    target: int
    """The index of the target shape."""

    size: tuple[int, int, int]
    """The size of the video as (T, H, W)."""

    duration: float
    """The duration of the video in seconds."""

    random_dot_density: float
    """The density of the random dots in dots per unit square."""

    random_dot_lifetime: float
    """The lifetime of the random dots."""

    random_dot_seed: int
    """The seed used to generate the random dots."""

    _stimulus: npt.NDArray | None = None
    _stimulus_debug: npt.NDArray | None = None
    _stimulus_clean: npt.NDArray | None = None
    _distractor_clean: npt.NDArray | None = None
    _choices: list[npt.NDArray[np.uint8]] | None = None
    _dot_data: dict[str, npt.NDArray] | None = None

    @property
    def frame_rate(self) -> float:
        """The frame rate of the video in frames per second."""
        return self.size[0] / self.duration

    @property
    def foreground_speed(self) -> float:
        """Speed of the foreground object in pixels per frame."""
        return np.linalg.norm(self.foreground_motion).item()

    @property
    def background_speed(self) -> float:
        """Speed of the background in pixels per frame."""
        return np.linalg.norm(self.background_motion).item()

    @property
    def motion_difference(self) -> float:
        """Difference between the foreground and background motion."""
        return np.linalg.norm(
            np.array(self.foreground_motion) - np.array(self.background_motion)
        ).item()

    @property
    def stimulus(self) -> npt.NDArray[np.uint8]:
        """The rendered stimulus of the trial with shape (T, H, W)."""
        if self._stimulus is None:
            self.render_stimulus()
        assert self._stimulus is not None
        return self._stimulus

    @property
    def stimulus_debug(self) -> npt.NDArray[np.uint8]:
        """The rendered stimulus of the trial with debug information."""
        if self._stimulus_debug is None:
            self.render_stimulus_debug()
        assert self._stimulus_debug is not None
        return self._stimulus_debug

    @property
    def stimulus_clean(self) -> npt.NDArray[np.uint8]:
        """The clean stimulus of the trial with shape (T, H, W)."""
        if self._stimulus_clean is None:
            self.render_stimulus_clean()
        assert self._stimulus_clean is not None
        return self._stimulus_clean

    @property
    def stimulus_segmentation(self) -> npt.NDArray[np.bool_]:
        """The segmentation of the stimulus with shape (T, H, W)."""
        return np.all(self.stimulus_clean == 0, axis=-1)

    @property
    def distractor_clean(self) -> npt.NDArray[np.uint8]:
        """The clean video of the distractor with shape (T, H, W)."""
        if self._distractor_clean is None:
            self.render_distractor_clean()
        assert self._distractor_clean is not None
        return self._distractor_clean

    @property
    def choices(self) -> list[npt.NDArray[np.uint8]]:
        """The rendered choices of the trial, each with shape (H, W)."""
        if self._choices is None:
            self.render_choices()
        assert self._choices is not None
        return self._choices

    def informative_area(self) -> float:
        """Returns the area of the informative region in the stimulus.
        
        The informative region covers all pixels that are part of exactly one of the
        shapes and therefore allow the observer to discriminate between the target and
        the distractor shapes. The area is measured in unit square.
        """
        if len(self.shapes) != 2:
            raise ValueError("Only two shapes are supported")

        choice0_mask = np.all(self.choices[0] == 0, axis=-1)
        choice1_mask = np.all(self.choices[1] == 0, axis=-1)
        informative_region = np.logical_xor(choice0_mask, choice1_mask)
        return np.sum(informative_region) / choice0_mask.size * 4

    def number_of_informative_dots(self) -> int:
        """Returns the number of informative dots in the stimulus.
        
        This returns the number of dots that allow the observer to discriminate between
        the target and the distractor shapes. This is the number of dots that are
        belong to one of the shapes, but not both. Points are counted on every frame.
        I.e. if a dot fulfills the criteria on multiple frames, it is counted multiple
        times.

        Returns:
            The number of informative dots.
        
        Raises:
            ValueError: If the number of shapes is not 2.
        """
        if len(self.shapes) != 2:
            raise ValueError("Only two shapes are supported")
        
        if self._dot_data is None:
            self.render_stimulus()
        assert self._dot_data is not None

        target = self.stimulus_segmentation

        distractor_shape = self.shapes[1 - self.target]
        distractor = np.all(self._render_clean(distractor_shape) == 0, axis=-1)

        informative = np.logical_xor(target, distractor)

        number_of_dots = 0

        for frame_index in range(informative.shape[0]):
            foreground_xs = self._dot_data["foreground_xs"][:, frame_index]
            foreground_ys = self._dot_data["foreground_ys"][:, frame_index]
            background_xs = self._dot_data["background_xs"][:, frame_index]
            background_ys = self._dot_data["background_ys"][:, frame_index]
            background_visible = self._dot_data["background_visible"][:, frame_index]

            xs = np.concatenate([foreground_xs, background_xs[background_visible]])
            ys = np.concatenate([foreground_ys, background_ys[background_visible]])

            xs = xs.astype(int)
            ys = ys.astype(int)

            number_of_dots += np.count_nonzero(informative[frame_index, ys, xs])

        return number_of_dots

    def save(self, path: str | Path) -> None:
        """Saves all sample data to npz."""
        data = {
            "shapes": self.shapes,
            "shapes_smooth": self.shapes_smooth,
            "orientation": self.orientation,
            "scale": self.scale,
            "background_motion": self.background_motion,
            "foreground_motion": self.foreground_motion,
            "target": self.target,
            "size": self.size,
            "duration": self.duration,
            "random_dot_density": self.random_dot_density,
            "random_dot_lifetime": self.random_dot_lifetime,
            "stimulus": self.stimulus,
            "stimulus_clean": self.stimulus_clean,
            "distractor_clean": self.distractor_clean,
            "choices": self.choices,
            "dot_data": self._dot_data,
            "random_dot_seed": self.random_dot_seed,
            "number_of_informative_dots": self.number_of_informative_dots(),
        }

        torch.save(data, path)

    def render_choices(self) -> None:
        """Renders the choices of the trial.
        
        This method is called automatically when accessing the `choices` property. It
        can be called manually to eagerly render the choices.
        """
        self._choices = []

        for shape in self.shapes:
            choice = self._render_shape(shape, (0, 0))
            self._choices.append(choice)

    def render_stimulus_clean(self) -> None:
        self._stimulus_clean = self._render_clean(self.shapes[self.target])
    
    def render_distractor_clean(self) -> None:
        self._distractor_clean = self._render_clean(self.shapes[1 - self.target])

    def _render_clean(self, shape: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
        frames = []

        for t in np.linspace(-1, 1, self.size[0]):
            position_x = t * self.foreground_motion[0]
            position_y = t * self.foreground_motion[1]

            frames.append(self._render_shape(shape, (position_x, position_y)))

        return np.stack(frames)

    # IDEA: Pass shapes as array (N, 2) instead of an array of shape (2, N)
    # Adapted from sbdzdz/idsprites
    def _render_shape(
        self,
        shape: npt.NDArray[np.float32],
        position: tuple[float, float],
        color: tuple[int, int, int] = (0, 0, 0),
        background_color: tuple[int, int, int] = (128, 128, 128),
    ) -> npt.NDArray[np.uint8]:
        """Renders a shape at a given position.
    
        Parameters:
            shape: A polygon as an array of shape (2, N).
            position: The position of the shape as (x, y) in stimulus coordinates.
            color: The color of the shape as (R, G, B), where R, G, B are in [0, 255].
            background_color: The color of the background as (R, G, B), where R, G, B 
                are in [0, 255].
        """
        H, W = self.size[1], self.size[2]
 
        canvas = np.zeros((H, W, 3), dtype=np.int32)
        canvas[:, :] = background_color

        rotation_matrix = np.array([
            [np.cos(self.orientation), -np.sin(self.orientation)],
            [np.sin(self.orientation), np.cos(self.orientation)],
        ])
        shape = rotation_matrix @ shape
        shape = shape * self.scale + np.array(position)[:, np.newaxis]

        shape = (shape + 1) * np.array([[W], [H]]) / 2
        shape = np.round(shape).astype(np.int32)

        cv2.fillPoly(img=canvas, pts=[shape.T], color=color, lineType=cv2.LINE_AA)

        return canvas.astype(np.uint8)

    def render_stimulus(self) -> None:
        """Renders the stimulus of the trial.
        
        This method is called automatically when accessing the `stimulus` property. It
        can be called manually to eagerly render the stimulus.
        """
        generator = np.random.default_rng(self.random_dot_seed)
        self._stimulus, self._dot_data = self._render_stimulus(generator, debug=False)
    
    def render_stimulus_debug(self) -> None:
        """Renders the stimulus of the trial with debug information."""
        generator = np.random.default_rng(self.random_dot_seed)
        self._stimulus_debug, _ = self._render_stimulus(generator, debug=True)

    def _render_stimulus(
        self, generator: np.random.Generator, debug: bool = False
    ) -> tuple[npt.NDArray[np.uint8], dict[str, npt.NDArray]]:
        T, H, W = self.size

        foreground_motion = np.array(self.foreground_motion) * np.array([W, H]) / T
        background_motion = np.array(self.background_motion) * np.array([W, H]) / T

        random_dot_lifetime = int(round(self.random_dot_lifetime * T / 2))

        foreground_mask = self.stimulus_segmentation

        background_dot_number = int(4 * self.random_dot_density)
        foreground_pixels = np.sum(foreground_mask[0])
        foreground_dot_number = \
            int(4 * self.random_dot_density * foreground_pixels / H / W)

        foreground = np.where(foreground_mask[0])
        indices = generator.choice(
            len(foreground[0]), foreground_dot_number, replace=False
        )
        foreground_xs = foreground[1][indices].astype(np.float32)
        foreground_ys = foreground[0][indices].astype(np.float32)
        foreground_age = generator.integers(
            0, random_dot_lifetime, foreground_dot_number
        )

        background_xs = generator.random(background_dot_number) * W
        background_ys = generator.random(background_dot_number) * H
        background_age = generator.integers(
            0, random_dot_lifetime, background_dot_number
        )

        dot_data: dict[str, list[npt.NDArray]] = {
            "foreground_xs": [],
            "foreground_ys": [],
            "foreground_replace": [],
            "background_xs": [],
            "background_ys": [],
            "background_replace": [],
            "background_visible": [],
        }

        frames = []
        for i, mask in enumerate(foreground_mask):
            foreground_xs += foreground_motion[0]
            foreground_ys += foreground_motion[1]
            foreground_age += 1

            replace = foreground_age >= random_dot_lifetime
            num_replace = int(replace.sum())
            foreground = np.where(mask)
            indices = generator.choice(len(foreground[0]), num_replace, replace=False)
            foreground_xs[replace] = foreground[1][indices].astype(np.float32)
            foreground_ys[replace] = foreground[0][indices].astype(np.float32)
            foreground_age[replace] = 0

            foreground_xs = np.mod(foreground_xs, W)
            foreground_ys = np.mod(foreground_ys, H)

            dot_data["foreground_xs"].append(foreground_xs.copy())
            dot_data["foreground_ys"].append(foreground_ys.copy())
            dot_data["foreground_replace"].append(replace)

            background_xs += background_motion[0]
            background_ys += background_motion[1]
            background_age += 1

            replace = background_age >= random_dot_lifetime
            num_replace = int(replace.sum())
            background_xs[replace] = generator.random(num_replace) * W
            background_ys[replace] = generator.random(num_replace) * H
            background_age[replace] = 0

            background_xs = np.mod(background_xs, W)
            background_ys = np.mod(background_ys, H)

            dot_data["background_xs"].append(background_xs.copy())
            dot_data["background_ys"].append(background_ys.copy())
            dot_data["background_replace"].append(replace)

            background_grid = np.stack([
                background_xs / W * 2 - 1,
                background_ys / H * 2 - 1,
            ], axis=-1).reshape((1, 1, -1, 2))
            interpolated_mask = F.grid_sample(
                torch.from_numpy(1 - mask)[None, None].float(),
                torch.from_numpy(background_grid).float(),
                align_corners=False,
            ).squeeze().numpy()
            visible = interpolated_mask > 0.5

            dot_data["background_visible"].append(visible)

            all_xs = np.concatenate([foreground_xs, background_xs[visible]])
            all_ys = np.concatenate([foreground_ys, background_ys[visible]])

            background = None if not debug else self.stimulus_clean[i]
            frame = self._render_dots(
                mask.shape, all_xs, all_ys, 2.0, background=background
            )
            frames.append(frame)

        return (
            np.stack(frames),
            { key: np.stack(value, axis=1) for key, value in dot_data.items() }
        )

    @staticmethod
    def _render_dots(
        resolution: tuple[int, int],
        xs: torch.Tensor,
        ys: torch.Tensor,
        size: float,
        background: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.uint8]:
        if background is None:
            frame = PIL.Image.new("L", resolution)
        else:
            frame = PIL.Image.fromarray(background)

        draw = PIL.ImageDraw.Draw(frame)

        if background is None:
            draw.rectangle((0.0, 0.0, *resolution), "gray")
    
        for x, y in zip(xs, ys):
            draw.ellipse(
                (x - size / 2, y - size / 2, x + size / 2, y + size / 2), "white"
            )

        return np.array(frame)


class RandomDotShapeIdentificationDataset:
    def __init__(
        self,
        number_of_samples: int | None = None,
        repeat_shapes: int | None = None,
        seed: int = 0,
        size: tuple[int, int, int] = (31, 256, 256),
        duration: float = 1.0,
        num_choices: int = 4,
        foreground_max_speed: float = 0.5,
        background_max_speed: float = 0.5,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        min_random_dot_density: float = 16.0,
        max_random_dot_density: float = 512.0,
        min_random_dot_lifetime: float = 2/3,
        max_random_dot_lifetime: float = 2.0,
    ) -> None:
        if size[0] % 2 == 0:
            raise ValueError("video_length must be odd")

        if repeat_shapes is not None and number_of_samples is None:
            raise ValueError(
                "number_of_samples must be specified when repeat_shapes is set"
            )
    
        if repeat_shapes is not None and num_choices != 2:
            raise ValueError("num_choices must be 2 when repeat_shapes is set")

        self._idsprites = InfiniteDSprites()

        self.number_of_samples = number_of_samples
        self.repeat_shapes = repeat_shapes

        self.rng = np.random.default_rng(seed)

        self._shapes = None
        self._pairs = None
        if self.repeat_shapes is not None:
            assert self.number_of_samples is not None
            self._pairs = sample_balanced_pairs(
                self.number_of_samples,
                repeats=self.repeat_shapes,
                rng=self.rng,
            )
            num_shapes = max(max(pair) for pair in self._pairs) + 1
            self._shapes = [sample_shape(self.rng) for _ in range(num_shapes)]

        self.num_choices = num_choices
        self.size = size
        self.duration = duration
        self.foreground_max_speed = foreground_max_speed
        self.background_max_speed = background_max_speed
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_random_dot_density = min_random_dot_density
        self.max_random_dot_density = max_random_dot_density
        self.min_random_dot_lifetime = min_random_dot_lifetime
        self.max_random_dot_lifetime = max_random_dot_lifetime

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        shapes: list[tuple[npt.NDArray[np.float32], bool]] | None = None,
        target: int | None = None,
    ) -> RandomDotShapeIdentificationSample:
        if shapes is None:
            shapes = self._sample_shapes(self.rng)
        
        return RandomDotShapeIdentificationSample(
            shapes=[shape for shape, _ in shapes],
            shapes_smooth=[smooth for _, smooth in shapes],
            orientation=self._sample_orientation(self.rng),
            scale=self._sample_scale(self.rng),
            background_motion=self._sample_background_motion(self.rng),
            foreground_motion=self._sample_foreground_motion(self.rng),
            target=target or self._sample_target(self.rng),
            size=self.size,
            duration=self.duration,
            random_dot_density=self._sample_random_dot_density(self.rng),
            random_dot_lifetime=self._sample_random_dot_lifetime(self.rng),
            random_dot_seed=self.rng.integers(0, np.iinfo(np.int32).max),
        )

    def __iter__(self) -> Generator[RandomDotShapeIdentificationSample, None, None]:
        if self.repeat_shapes is not None:
            assert self._pairs is not None
            assert self._shapes is not None
            for target, distractor in self._pairs:
                if self.rng.random() < 0.5:
                    yield self.sample(
                        shapes=[self._shapes[target], self._shapes[distractor]],
                        target=0,
                    )
                else:
                    yield self.sample(
                        shapes=[self._shapes[distractor], self._shapes[target]],
                        target=1,
                    )
        else:
            count = 0
            while True:
                yield self.sample()
                count += 1
                if self.number_of_samples is not None and count >= self.number_of_samples:  # noqa: E501
                    break

    def _sample_shapes(
        self,
        rng: np.random.Generator,
    ) -> list[tuple[npt.NDArray[np.float32], bool]]:
        return [sample_shape(rng) for _ in range(self.num_choices)]

    def __len__(self) -> int:
        if self.number_of_samples is None:
            raise ValueError("number_of_samples is not set")
        return self.number_of_samples

    def _sample_orientation(self, rng: np.random.Generator) -> float:
        return rng.uniform(0, 2 * np.pi)

    def _sample_scale(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.min_scale, self.max_scale)

    def _sample_foreground_motion(
        self, rng: np.random.Generator
    ) -> tuple[float, float]:
        return self._sample_motion(rng, self.foreground_max_speed)

    def _sample_background_motion(
        self, rng: np.random.Generator
    ) -> tuple[float, float]:
        return self._sample_motion(rng, self.background_max_speed)

    def _sample_motion(
        self, rng: np.random.Generator, max_speed: float
    ) -> tuple[float, float]:
        speed = rng.uniform(0, max_speed)
        direction = rng.uniform(0, 2 * np.pi)
        dx = speed * np.cos(direction)
        dy = speed * np.sin(direction)
        return dx, dy

    def _sample_target(self, rng: np.random.Generator) -> int:
        return rng.integers(0, self.num_choices)

    def _sample_random_dot_density(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.min_random_dot_density, self.max_random_dot_density)

    def _sample_random_dot_lifetime(self, rng: np.random.Generator) -> float:
        return rng.uniform(self.min_random_dot_lifetime, self.max_random_dot_lifetime)


def sample_balanced_pairs(
    number_of_pairs: int,
    repeats: int = 2,
    rng: np.random.Generator | None = None,
    max_attempts: int = 1_000_000,
) -> list[tuple[int, int]]:
    """Samples balanced pairs of values.

    This function samples pairs of values (a,b) such that:
    - a != b
    - Each value is used exactly `repeats / 2` times as the first element and
      `repeats / 2` times as the second element.

    Example:
        pairs = sample_balanced_pairs(4, 2)
        # pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]

    Args:
        number_of_pairs: The number of pairs to sample.
        repeats: The number of times each value is used in total. Must be even.
        rng: The random number generator to use. If `None`, the default generator is
            used.
        max_attempts: The maximum number of attempts to sample balanced pairs.

    Returns:
        A list of pairs of values.
    """
    if rng is None:
        rng = np.random.default_rng()

    if repeats % 2 != 0:
        raise ValueError("repeats must be even")

    if number_of_pairs % repeats != 0:
        raise ValueError("number_of_pairs must be divisible by repeats")

    number_of_values = (number_of_pairs * 2) // repeats

    if (number_of_values * (number_of_values - 1)) < number_of_pairs:
        raise ValueError("Not enough values to sample distinct pairs")

    values = list(range(number_of_values))

    left = values.copy() * (repeats // 2)
    right = values.copy() * (repeats // 2)

    for _ in range(max_attempts):
        rng.shuffle(left)
        rng.shuffle(right)

        pairs = list(zip(left, right))
        assert len(pairs) == number_of_pairs

        if len(set(pairs)) < number_of_pairs:
            continue
    
        if any(a == b for a, b in pairs):
            continue

        return pairs
    
    raise ValueError(f"Failed to sample balanced pairs (max_attempts={max_attempts})")
