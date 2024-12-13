"""Random sampling methods."""

import numpy as np
from idsprites import InfiniteDSprites


def sample_motion(rng: np.random.Generator, max_speed: float) -> tuple[float, float]:
    speed = rng.uniform(0, max_speed)
    direction = rng.uniform(0, 2 * np.pi)
    dx = speed * np.cos(direction)
    dy = speed * np.sin(direction)
    return dx, dy


# https://github.com/sbdzdz/idsprites/blob/4609d56bee48a795250f1fa5051c507ae2d207e2/idsprites/infinite_dsprites.py#L186
def sample_shape(rng: np.random.Generator):
    """Return a random shape."""
    idsprites = InfiniteDSprites()
    verts = _sample_vertex_positions(rng)
    smooth = rng.random() < 0.5
    shape = (
        idsprites.interpolate(verts)
        if smooth
        else idsprites.interpolate(verts, k=1)
    )
    shape = idsprites.align(shape)
    shape = idsprites.center_and_scale(shape)

    return shape, smooth


# https://github.com/sbdzdz/idsprites/blob/4609d56bee48a795250f1fa5051c507ae2d207e2/idsprites/infinite_dsprites.py#L204
def _sample_vertex_positions(
    rng: np.random.Generator,
    min_verts: int = 5,
    max_verts: int = 8,
    radius_std: float = 0.4,
    angle_std: float = 0.5,
):
    """Sample the positions of the vertices of a polygon.
    Args:
        min_verts: Minimum number of vertices (inclusive).
        max_verts: Maximum number of vertices (inclusive).
        radius_std: Standard deviation of the polar radius when sampling the
            vertices.
        angle_std: Standard deviation of the polar angle when sampling the vertices.
    Returns:
        An array of shape (2, num_verts).
    """
    num_verts = rng.integers(min_verts, max_verts + 1)
    rs = rng.normal(1.0, radius_std, num_verts)
    rs = np.clip(rs, 0.1, 1.9)

    epsilon = 1e-6
    circle_sector = np.pi / num_verts - epsilon
    intervals = np.linspace(0, 2 * np.pi, num_verts, endpoint=False)
    thetas = rng.normal(0.0, circle_sector * angle_std, num_verts)
    thetas = np.clip(thetas, -circle_sector, circle_sector) + intervals

    verts = [[r * np.cos(theta), r * np.sin(theta)] for r, theta in zip(rs, thetas)]
    return np.array(verts).T
