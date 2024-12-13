"""Streams of random dot shape identification samples."""

import abc
import math
from itertools import islice
from typing import Generator

import numpy as np

from dataset import RandomDotShapeIdentificationSample
from random_utils import sample_motion, sample_shape


class Stream(abc.ABC):
    """Stream of random dot shape identification samples."""

    def __init__(
        self, seed: int | None = None, rng: np.random.Generator | None = None
    ) -> None:
        """Initialize the stream."""
        if seed is None and rng is None:
            raise ValueError("Either seed or rng must be provided.")
        if seed is not None and rng is not None:
            raise ValueError("Only one of seed or rng must be provided.")

        if rng is None:
            rng = np.random.default_rng(seed)
        
        self.rng = rng

    @abc.abstractmethod
    def __iter__(self) -> Generator[RandomDotShapeIdentificationSample, None, None]:
        """Return an iterator over samples."""
        pass

    @abc.abstractmethod
    def split(self, n: int) -> list["Stream"]:
        """Split the stream into n streams."""
        pass


class IndependentUniformStream(Stream):
    """Stream of samples with independent and uniformly distributed attributes."""

    def __init__(
        self,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        size: tuple[int, int, int] = (31, 256, 256),
        duration: float = 1.0,
        foreground_max_speed: float = 0.5,
        background_max_speed: float = 0.5,
        min_scale: float = 0.5,
        max_scale: float = 1.0,
        min_random_dot_density: float = 16.0,
        max_random_dot_density: float = 512.0,
        min_random_dot_lifetime: float = 2/3,
        max_random_dot_lifetime: float = 2.0,
    ) -> None:
        """Initializes the stream."""
        super().__init__(seed, rng)
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

    def __iter__(self) -> Generator[RandomDotShapeIdentificationSample, None, None]:
        """Returns an iterator over samples."""
        while True:
            shapes_and_smooths = [
                sample_shape(self.rng) for _ in range(2)
            ]
            shapes = [shape for shape, _ in shapes_and_smooths]
            smooths = [smooth for _, smooth in shapes_and_smooths]

            yield RandomDotShapeIdentificationSample(
                shapes=shapes,
                shapes_smooth=smooths,
                orientation=self.rng.uniform(0, 2 * np.pi),
                scale=self.rng.uniform(self.min_scale, self.max_scale),
                foreground_motion=sample_motion(self.rng, self.foreground_max_speed),
                background_motion=sample_motion(self.rng, self.background_max_speed),
                target=self.rng.integers(0, 2),
                size=self.size,
                duration=self.duration,
                random_dot_density=self.rng.uniform(
                    self.min_random_dot_density, self.max_random_dot_density
                ),
                random_dot_lifetime=self.rng.uniform(
                    self.min_random_dot_lifetime, self.max_random_dot_lifetime
                ),
                random_dot_seed=self.rng.integers(0, 2**32),
            )
    
    def split(self, n: int) -> list["Stream"]:
        """Splits the stream into n streams."""
        rngs = self.rng.spawn(n)
        return [
            IndependentUniformStream(
                rng=rng,
                size=self.size,
                duration=self.duration,
                foreground_max_speed=self.foreground_max_speed,
                background_max_speed=self.background_max_speed,
                min_scale=self.min_scale,
                max_scale=self.max_scale,
                min_random_dot_density=self.min_random_dot_density,
                max_random_dot_density=self.max_random_dot_density,
                min_random_dot_lifetime=self.min_random_dot_lifetime,
                max_random_dot_lifetime=self.max_random_dot_lifetime,
            )
            for rng in rngs
        ]


class UniformNumberOfInformativeDotsStream(Stream):
    """Stream of samples with uniformly distributed number of informative dots.
    
    This stream uses rejection sampling to ensure that the distribution of the
    number of informative dots is approximately uniform. The proposal distribution
    is estimated from the samples in the stream using a histogram. Samples from the
    proposal distribution are then accepted with a probability inversely proportional
    to the proposal distribution.
    """

    def __init__(
        self,
        proposal_stream: Stream,
        min_number_of_informative_dots: int = 0,
        max_number_of_informative_dots: int = 100,
        bins: int = 25,
        estimation_samples: int = 1000,
    ) -> None:
        """Initializes the stream.
        
        Parameters:
            proposal_stream: Stream of samples from which the proposal distribution
                is estimated.
            min_number_of_informative_dots: Minimum number of informative dots.
            max_number_of_informative_dots: Maximum number of informative dots.
            bins: Number of bins for the histogram of the proposal distribution.
            estimation_samples: Number of samples used to estimate the proposal
                distribution.
        """
        super().__init__(rng=proposal_stream.rng)
        self._estimation_stream, self._proposal_stream = proposal_stream.split(2)
        self.min_number_of_informative_dots = min_number_of_informative_dots
        self.max_number_of_informative_dots = max_number_of_informative_dots
        self.bins = bins
        self.estimation_samples = estimation_samples

        self._estimate_proposal_distribution()

    def _estimate_proposal_distribution(self) -> None:
        self._proposal_distribution= np.zeros(self.bins)

        for sample in islice(self._estimation_stream, self.estimation_samples):
            bin_index = self._bin_index(sample.number_of_informative_dots())

            if bin_index is None:
                continue

            self._proposal_distribution[bin_index] += 1
        
        if self._proposal_distribution.min() == 0:
            raise ValueError("Proposal distribution is not defined for some bins.")

        self._proposal_distribution /= self._proposal_distribution.sum()
    
    def __iter__(self) -> Generator[RandomDotShapeIdentificationSample, None, None]:
        thresholds = self._proposal_distribution.min() / self._proposal_distribution

        for proposal in self._proposal_stream:
            bin_index = self._bin_index(proposal.number_of_informative_dots())
            if bin_index is None:
                continue
    
            if self.rng.uniform() <= thresholds[bin_index]:
                yield proposal

    def _bin_index(self, value: int) -> int | None:
        vmin = self.min_number_of_informative_dots
        vmax = self.max_number_of_informative_dots
        if value < vmin or value > vmax:
            return None
        if math.isclose(value, vmax):
            return self.bins - 1
        return int(math.floor((value - vmin) / (vmax - vmin) * self.bins))

    def split(self, n: int) -> list["Stream"]:
        return [
            UniformNumberOfInformativeDotsStream(
                proposal_stream=proposal_stream,
                min_number_of_informative_dots=self.min_number_of_informative_dots,
                max_number_of_informative_dots=self.max_number_of_informative_dots,
                bins=self.bins,
                estimation_samples=self.estimation_samples,
            )
            for proposal_stream in self._proposal_stream.split(n)
        ]


class EasyStream(Stream):
    """Stream of "easy" samples."""

    def __init__(
        self,
        proposal_stream: Stream,
        min_number_of_informative_dots: int = 0,
        min_motion_difference: float = 0.1,
    ) -> None:
        """Initializes the stream.
        
        Parameters:
            proposal_stream: Stream of proposals.
            min_number_of_informative_dots: Minimum number of informative dots.
            min_motion_difference: Minimum motion difference between the shape and the
                background.
        """
        super().__init__(rng=proposal_stream.rng)
        self._proposal_stream = proposal_stream
        self.min_number_of_informative_dots = min_number_of_informative_dots
        self.min_motion_difference = min_motion_difference
    
    def __iter__(self) -> Generator[RandomDotShapeIdentificationSample, None, None]:
        for proposal in self._proposal_stream:
            if (
                proposal.number_of_informative_dots() >= self.min_number_of_informative_dots and  # noqa: E501
                proposal.motion_difference >= self.min_motion_difference
            ):
                yield proposal

    def split(self, n: int) -> list["Stream"]:
        return [
            EasyStream(
                proposal_stream=proposal_stream,
                min_number_of_informative_dots=self.min_number_of_informative_dots,
                min_motion_difference=self.min_motion_difference,
            )
            for proposal_stream in self._proposal_stream.split(n)
        ]
