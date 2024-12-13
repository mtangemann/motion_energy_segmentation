"""Stimulus transforms.

Transforms are modifications of input data (images, videos, ...) and applied before the
stimulus is passed to a model.
"""

import numpy as np
import PIL.Image
import PIL.ImageDraw
import torch
import torch.nn.functional as F


def object_kinematogram(
    flow: torch.Tensor,
    num_dots: int = 2500,
    lifetime: int = 17,
    size: float = 2.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Generates object kinematograms as described by Robert et al. (2023).

    The parameters from the original paper are used as default values. The reported
    values are however relative to the resolution (720x460) and frame rate (60Hz). You
    might need to adapt the parameters for your setting.

    !!! quote "Reference"
        Sophia Robert, Leslie G. Ungerleider and Maryam Vaziri-Pashkam (2023)<br>
        *Disentangling Object Category Representations Driven by Dynamic and Static
        Visual Input*.<br>
        Journal of Neuroscience 43(4), 621-634.<br>
        [HTML](https://www.jneurosci.org/content/43/4/621),
        [PDF](https://www.jneurosci.org/content/jneuro/43/4/621.full.pdf)

    Parameters:
        flow: Optical flow as float Tensor with shape `(T, 2, H, W)`. The channels are
            horizontal and vertical displacement, respectively.
        num_dots: Number of dots. Robert et al. (2023) used 2500 points for videos with
            a resolution of 720x460 pixel.
        lifetime: Lifetime of each dot in frames. Robert et al. (2023) used a lifetime
            of 17 frames for videos with a frame rate of 60Hz.
        size: Diameter of each dot in pixel.
        generator: A random number generator. If set to None, the global random number
            generator of PyTorch will be used.

    Returns:
        The grayscale object kinematogram as float tensor of shape `(T+1, 1, H, W)` and
            values in `[0.0, 1.0]`.
    """       
    _, _, H, W = flow.shape

    xs = torch.rand((num_dots,), generator=generator) * W
    ys = torch.rand((num_dots,), generator=generator) * H
    age = torch.randint(0, lifetime, (num_dots,), generator=generator)

    frames = [_render_dots((W, H), xs, ys, size)]

    for flow_frame in flow:
        grid = torch.stack([
            xs / W * 2 - 1,
            ys / H * 2 - 1,
        ], dim=-1).unsqueeze(0).unsqueeze(0)

        dot_flow = F.grid_sample(
            flow_frame.unsqueeze(0), grid, align_corners=False
        ).squeeze()

        xs = xs + dot_flow[0]
        ys = ys + dot_flow[1]

        age = age + 1

        replace = (age >= lifetime) | (xs < 0) | (xs >= W) | (ys < 0) | (ys >= H)
        num_replace = int(replace.sum().item())

        xs[replace] = torch.rand((num_replace,), generator=generator) * W
        ys[replace] = torch.rand((num_replace,), generator=generator) * H
        age[replace] = 0

        frames.append(_render_dots((W, H), xs, ys, size))

    return torch.stack(frames)


def _render_dots(
    resolution: tuple[int, int],
    xs: torch.Tensor,
    ys: torch.Tensor,
    size: float,
) -> torch.Tensor:
    frame = PIL.Image.new("L", resolution)
    draw = PIL.ImageDraw.Draw(frame)
    draw.rectangle((0.0, 0.0, *resolution), "gray")
 
    for x, y in zip(xs, ys):
        draw.ellipse((x - size / 2, y - size / 2, x + size / 2, y + size / 2), "white")

    return torch.from_numpy(np.array(frame)).unsqueeze(0)
