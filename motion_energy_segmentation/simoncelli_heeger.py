"""PyTorch implementation of the Simoncelli & Heeger (1998) model."""

import math

import torch
import torch.nn.functional as F
from torch import nn

# Values from `pars = shPars` in Matlab implementation.
_DEFAULT_PARAMETERS = {
    "mtAlpha": torch.tensor(0.3171285328532853),
    "mtC50": torch.tensor(0.1),
    "mtPopulationVelocities": torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0471975511965976, 1.0],
        [2.0943951023931957, 1.0],
        [3.141592653589793, 1.0],
        [4.18879020478639, 1.0],
        [5.235987755982989, 1.0],
        [0.26179938779914935, 6.0],
        [0.7853981633974482, 6.0],
        [1.308996938995747, 6.0],
        [1.832595714594046, 6.0],
        [2.356194490192345, 6.0],
        [2.879793265790644, 6.0],
        [3.4033920413889422, 6.0],
        [3.9269908169872414, 6.0],
        [4.45058959258554, 6.0],
        [4.974188368183839, 6.0],
        [5.497787143782138, 6.0],
        [6.021385919380436, 6.0],
    ]),
    "mtSpatialPoolingFilter": torch.tensor([
        [
            0.0014794516787135662,
            0.003804239018665086,
            0.00875346226514214,
            0.018023411141080636,
            0.03320772876259043,
            0.05475028876252328,
            0.0807753247211904,
            0.10663900118033988,
            0.12597909446198644,
            0.1331759960155365,
            0.12597909446198644,
            0.10663900118033988,
            0.0807753247211904,
            0.05475028876252328,
            0.03320772876259043,
            0.018023411141080636,
            0.00875346226514214,
            0.003804239018665086,
            0.0014794516787135662,
        ],
    ]),
    "v1C50": torch.tensor(0.1),
    "v1ComplexFilter": torch.tensor([
        [0.00188981, 0.01096042, 0.04301196, 0.11421021, 0.20519858,
        0.24945803, 0.20519858, 0.11421021, 0.04301196, 0.01096042,
        0.00188981]
    ]),
    "v1PopulationDirections": torch.tensor([
        [5.5475, 0.2162],
        [3.2461, 0.2176],
        [0.6942, 0.2266],
        [4.0698, 0.2326],
        [1.6542, 0.2393],
        [6.043 , 0.2428],
        [5.0491, 0.2528],
        [2.6917, 0.2614],
        [4.5468, 0.2656],
        [2.1323, 0.2685],
        [0.2522, 0.2885],
        [1.1696, 0.3033],
        [3.6457, 0.4343],
        [3.0694, 0.7118],
        [5.5522, 0.7333],
        [2.3645, 0.7374],
        [1.5678, 0.7681],
        [0.7111, 0.7819],
        [4.8972, 0.7988],
        [6.1972, 0.8059],
        [4.1718, 0.8077],
        [3.635 , 1.4046],
        [1.13  , 1.4926],
        [1.9858, 1.5147],
        [2.8434, 1.5681],
        [0.251 , 1.5743],
        [4.5454, 1.6289],
        [5.5369, 1.6311],
    ]),
    "v1SpatialFilters": torch.tensor([
        [ 0.0007,  0.0031,  0.0103,  0.0272],
        [ 0.0155,  0.0352,  0.0618,  0.0659],
        [ 0.0903,  0.1187,  0.0856, -0.0536],
        [ 0.2345,  0.1439, -0.0617, -0.2037],
        [ 0.3179,  0.    , -0.192 ,  0.    ],
        [ 0.2345, -0.1439, -0.0617,  0.2037],
        [ 0.0903, -0.1187,  0.0856,  0.0536],
        [ 0.0155, -0.0352,  0.0618, -0.0659],
        [ 0.0007, -0.0031,  0.0103, -0.0272],
    ]),
    "v1TemporalFilters": torch.tensor([
        [ 0.0007,  0.0031,  0.0103,  0.0272],
        [ 0.0155,  0.0352,  0.0618,  0.0659],
        [ 0.0903,  0.1187,  0.0856, -0.0536],
        [ 0.2345,  0.1439, -0.0617, -0.2037],
        [ 0.3179,  0.    , -0.192 ,  0.    ],
        [ 0.2345, -0.1439, -0.0617,  0.2037],
        [ 0.0903, -0.1187,  0.0856,  0.0536],
        [ 0.0155, -0.0352,  0.0618, -0.0659],
        [ 0.0007, -0.0031,  0.0103, -0.0272],
    ]),
    "scaleFactors.v1Linear": torch.tensor(6.608446302926781),
    "scaleFactors.v1FullWaveRectified": torch.tensor(1.9262519669114633),
    "scaleFactors.v1Blur": torch.tensor(1.0205372750866895),
    "scaleFactors.v1Complex": torch.tensor(0.99),
    "scaleFactors.v1NormalizationPopulationK": torch.tensor(0.24005422696518156),
    "scaleFactors.v1NormalizationStrength": torch.tensor(0.98),
    "scaleFactors.mtLinear": torch.tensor(1.5269596669497718),
    "scaleFactors.mtHalfWaveRectification": torch.tensor(0.5764261625233018),
    "scaleFactors.mtPattern": torch.tensor(0.02129606730294584),
    "scaleFactors.mtNormalizationStrength": torch.tensor(0.002129580606669143),
}


class SimoncelliHeegerModel(nn.Module):
    """Simoncelli & Heeger (1998) model of V1 and MT motion energy."""

    __version__ = "1.0.0"

    def __init__(self, num_scales: int = 1, padding: str = "valid") -> None:
        """Initializes the model.
        
        Parameters:
            num_scales: Number of scales to use for the model. Each scale is downsampled
                by a factor of 2 in space and time compared to the previous scale.
            padding: Padding mode to use for the convolutional layers. Can be either
                "valid" or "same". The original implementation uses "valid" padding.
        """
        super().__init__()
        self.v1_linear = V1Linear(num_scales, padding)
        self.v1_nonlinear = FullWaveRectification()
        self.v1_blur = V1Blur(padding)
        self.v1_normalize = V1Normalization_Tuned()
        self.mt_linear = MTLinear()
        self.mt_blur = MTPreThresholdBlur(padding)
        self.mt_nonlinear = HalfWaveRectification()
        self.mt_normalize = MTNormalization_Tuned()

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Applies the model to the given input.

        Parameters:
            input: Batched grayscale videos with as float tensor with shape
                (N, 1, T, H, W) and values in [0.0, 1.0].
        
        Returns:
            A list of tensors with shape (N, 1, T_i, H_i, W_i) containing the model
                output at different scales.
        """
        if input.ndim != 5 or input.shape[1] != 1:
            raise ValueError(
                "Expected input with shape (N, 1, T, H, W), but got input with shape "
                f"{tuple(input.shape)}."
            )

        # Internally, the model uses the format (N, 1, H, W, T).
        # NOTE Improvement: Change the layers to use the format (N, 1, T, H, W) instead.
        input = input.permute(0, 1, 3, 4, 2)

        outputs = self.v1_linear(input)
        outputs = self.v1_nonlinear(outputs)
        outputs = self.v1_blur(outputs)
        outputs = self.v1_normalize(outputs)
        outputs = self.mt_linear(outputs)
        outputs = self.mt_blur(outputs)
        outputs = self.mt_nonlinear(outputs)
        outputs = self.mt_normalize(outputs)

        return [output.permute(0, 1, 4, 2, 3) for output in outputs]


class V1Linear(nn.Module):
    """Linear filtering stage of the V1 model.
    
    Corresponds to `shModelV1Linear` in the original implementation.
    """

    def __init__(self, num_scales: int = 1, padding: str = "valid") -> None:
        super().__init__()
        self.num_scales = num_scales

        weight_t, weight_x, weight_y = self._create_weights()
        channels = weight_t.shape[0]
        
        self.conv_t = nn.Conv3d(
            1, channels, weight_t.shape[-3:], bias=False,
            padding=padding, padding_mode="replicate",
        )
        self.conv_t.weight.data = weight_t
        self.conv_t.weight.requires_grad = False

        self.conv_x = nn.Conv3d(
            channels, channels, weight_x.shape[-3:], bias=False, groups=channels,
            padding=padding, padding_mode="replicate",
        )
        self.conv_x.weight.data = weight_x
        self.conv_x.weight.requires_grad = False

        self.conv_y = nn.Conv3d(
            channels, channels, weight_y.shape[-3:], bias=False, groups=channels,
            padding=padding, padding_mode="replicate",
        )
        self.conv_y.weight.data = weight_y
        self.conv_y.weight.requires_grad = False

        population_directions = _DEFAULT_PARAMETERS["v1PopulationDirections"]
        self.readout = nn.Conv3d(
            channels, population_directions.shape[0], (1, 1, 1), bias=False,
            padding=padding, padding_mode="replicate",
        )
        self.readout.weight.data = swts(population_directions).reshape(
            population_directions.shape[0], channels, 1, 1, 1,
        )
        self.readout.weight.requires_grad = False

        self.scale_factor = _DEFAULT_PARAMETERS["scaleFactors.v1Linear"]

    def _create_weights(self):
        spatial_filters = _DEFAULT_PARAMETERS["v1SpatialFilters"]
        temporal_filters = _DEFAULT_PARAMETERS["v1TemporalFilters"]
        kernel_size = spatial_filters.shape[0]
        order = 3

        weights_t = []
        weights_x = []
        weights_y = []

        for order_t in range(order + 1):
            weight_t = temporal_filters[:, order_t].flip(0).view(1, 1, kernel_size)

            for order_x in range(order - order_t + 1):
                order_y = order - order_t - order_x
                weight_x = spatial_filters[:, order_x].view(1, kernel_size, 1)
                weight_y = spatial_filters[:, order_y].flip(0).view(kernel_size, 1, 1)

                weights_t.append(weight_t)
                weights_x.append(weight_x)
                weights_y.append(weight_y)
        
        return (
            torch.stack(weights_t).unsqueeze(1),
            torch.stack(weights_x).unsqueeze(1),
            torch.stack(weights_y).unsqueeze(1),
        )

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        outputs = []

        for scale in range(self.num_scales):
            scale_input = blurDn3(input, scale + 1)
            scale_output = self.conv_x(self.conv_y(self.conv_t(scale_input)))
            scale_output = self.readout(scale_output) * self.scale_factor
            outputs.append(scale_output)

        return outputs


class FullWaveRectification(nn.Module):
    """Full wave rectification stage.
    
    Corresponds to `shModelFullWaveRectification` in the original implementation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = _DEFAULT_PARAMETERS["scaleFactors.v1FullWaveRectified"]

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [input ** 2 * self.scale_factor for input in inputs]


class V1Blur(nn.Module):
    """Blurring stage of the V1 model.

    Corresponds to `shModelV1Blur` in the original implementation.
    """

    def __init__(self, padding: str = "valid") -> None:
        super().__init__()
        self.padding = padding
        self.register_buffer("complex_filters", _DEFAULT_PARAMETERS["v1ComplexFilter"])
        self.scale_factor = _DEFAULT_PARAMETERS["scaleFactors.v1Blur"]

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = gaussian_blur(inputs, self.complex_filters, self.padding)  # type: ignore
        return [output * self.scale_factor for output in outputs]


class V1Normalization_Tuned(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor_numerator = _DEFAULT_PARAMETERS["scaleFactors.v1Complex"]
        self.scale_factor_denominator = (
            _DEFAULT_PARAMETERS["scaleFactors.v1NormalizationPopulationK"] *
            _DEFAULT_PARAMETERS["scaleFactors.v1NormalizationStrength"]
        )
        self.sigma = _DEFAULT_PARAMETERS["v1C50"]

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []

        for input in inputs:
            numerator = input * self.scale_factor_numerator
            denominator = input * self.scale_factor_denominator
            denominator = denominator.sum(dim=-4, keepdim=True) + self.sigma ** 2
            output = numerator / denominator
            outputs.append(output)

        return outputs


class MTLinear(nn.Module):
    """Linear filtering stage of the MT model.
    
    Corresponds to `shModelMTLinear` in the original implementation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = _DEFAULT_PARAMETERS["scaleFactors.mtLinear"]
        weight = mt_wts(
            _DEFAULT_PARAMETERS["v1PopulationDirections"],
            _DEFAULT_PARAMETERS["mtPopulationVelocities"],
        )
        out_channels, in_channels = weight.shape
        self.conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.conv.weight.data = weight.view(*weight.shape, 1, 1, 1)
        self.conv.weight.requires_grad = False

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            self.conv(input) * self.scale_factor for input in inputs
        ]


class MTPreThresholdBlur(nn.Module):
    """Blurring stage of the MT model before squaring and normalization.
    
    Corresponds to `shModelMTPreThresholdBlur` in the original implementation.
    """
    def __init__(self, padding: str = "valid") -> None:
        super().__init__()
        self.padding = padding
        self.register_buffer(
            "spatial_pooling_filter",
            _DEFAULT_PARAMETERS["mtSpatialPoolingFilter"],
        )

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return gaussian_blur(inputs, self.spatial_pooling_filter, self.padding)  # type: ignore


class HalfWaveRectification(nn.Module):
    """Half wave rectification stage (MT).
    
    Corresponds to `shModelHalfWaveRectification` in the original implementation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.alpha = _DEFAULT_PARAMETERS["mtAlpha"]
        self.scale_factor = \
             _DEFAULT_PARAMETERS["scaleFactors.mtHalfWaveRectification"]

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.clamp(input + self.alpha, 0.0) ** 2 * self.scale_factor
            for input in inputs
        ]


class MTNormalization_Tuned(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor_numerator = _DEFAULT_PARAMETERS["scaleFactors.mtPattern"]
        self.scale_factor_denominator = \
            _DEFAULT_PARAMETERS["scaleFactors.mtNormalizationStrength"]
        self.sigma = _DEFAULT_PARAMETERS["mtC50"]

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []

        for input in inputs:
            numerator = input * self.scale_factor_numerator
            denominator = input * self.scale_factor_denominator
            denominator = denominator.sum(dim=-4, keepdim=True) + self.sigma ** 2
            output = numerator / denominator
            outputs.append(output)

        return outputs


def atan3(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    theta = torch.atan2(y, x)
    theta[theta < 0] = theta[theta < 0] + 2 * torch.pi
    return theta


def blurDn3(input: torch.Tensor, level: int = 2) -> torch.Tensor:
    """Blurs and downsamples the given input.
    
    Given input dimensions (t, h, w), the output dimensions will be
    ((t-4)/2, (h-4)/2, (w-4)/2) when using level=2 (default).

    Corresponds to `blurDn3` in the original implementation.

    Parameters:
        input: Tensor with shape (T, H, W).
        level: Recursively apply the filter (level-1) times.
    """
    if level <= 1:
        return input

    weight = torch.Tensor([0.0884, 0.3536, 0.5303, 0.3536, 0.0884]).to(input.device)

    output = F.conv3d(input, weight.view(1, 1, 1, 1, -1))
    output = F.conv3d(output, weight.view(1, 1, 1, -1, 1))
    output = F.conv3d(output, weight.view(1, 1, -1, 1, 1))
    output = output[..., ::2, ::2, ::2]

    if level > 2:
        output = blurDn3(output, level - 1)

    return output


def gaussian_blur(
    inputs: list[torch.Tensor],
    weight: torch.Tensor,
    padding: str = "valid",
) -> list[torch.Tensor]:
    """Applies a gaussian filter to the given inputs.

    Warning: The Matlab implementation does not work for multi scale inputs. I assume
    the goal is to upscale the input before applying the filter and downscaling
    afterwards (i.e., effectively using a smaller blur size). The downscaling is howver
    missing in the Matlab implementation, which leads to a memory error. Therefore this
    implementation is not compared to the original implementation.
    """
    weight_x = weight.view([1, 1, 1, -1, 1])
    weight_y = weight.view([1, 1, -1, 1, 1])

    outputs = []

    for scale_index, input in enumerate(inputs):
        num_channels = input.shape[1]

        if scale_index > 0:
            scale_factor = 2 ** scale_index
            input = F.interpolate(input, scale_factor=scale_factor, mode="trilinear")

        if padding == "same":
            pad_front = math.floor(weight_x.numel() / 2)
            pad_back = math.floor(weight_x.numel() / 2)
            input = F.pad(input, (0, 0, pad_front, pad_back, 0, 0), mode="reflect")

        expanded_weight_x = weight_x.expand(num_channels, -1, -1, -1, -1)
        output = F.conv3d(input, expanded_weight_x, groups=num_channels)

        if padding == "same":
            pad_front = math.floor(weight_y.numel() / 2)
            pad_back = math.floor(weight_y.numel() / 2)
            output = F.pad(output, (0, 0, 0, 0, pad_front, pad_back), mode="reflect")

        expanded_weight_y = weight_y.expand(num_channels, -1, -1, -1, -1)
        output = F.conv3d(output, expanded_weight_y, groups=num_channels)

        if scale_index > 0:
            scale_factor = 0.5 ** scale_index
            output = F.interpolate(output, scale_factor=scale_factor, mode="trilinear")

        outputs.append(output)

    return outputs


def mt_v1_components(neuron: torch.Tensor) -> torch.Tensor:
    """Corrsponds to `shMtV1Components` in the original implementation."""
    neuron = torch.stack([
        neuron[1] * neuron[0].sin(),
        neuron[1] * neuron[0].cos(),
    ])

    norm = torch.sum(neuron ** 2)
    if torch.isclose(norm, torch.tensor(0.0)):
        d1 = torch.tensor([1., 0, 0])
        d2 = torch.tensor([0., 1, 0])
    else:
        d1 = torch.tensor([-neuron[0], -neuron[1], norm])
        d1 = d1 / torch.norm(d1, p=2)
        d2 = torch.tensor([-neuron[1], neuron[0], 0])
        d2 = d2 / torch.norm(d2, p=2)
 
    angles = torch.tensor([0, 0.25, 0.5, 0.75]).unsqueeze(-1) * torch.pi
    v1_components = angles.cos() * d1 + angles.sin() * d2
    v1_components = rec2sphere(v1_components)
    v1_components = torch.stack([
        torch.fmod(v1_components[:, 0] + torch.pi, 2*torch.pi),
        torch.tan(v1_components[:, 1])
    ], dim=1)

    return v1_components


def mt_wts(
    v1_population_directions: torch.Tensor,
    mt_population_velocities: torch.Tensor,
) -> torch.Tensor:
    weights = []

    for direction in mt_population_velocities:
        directions = mt_v1_components(direction)
        weight = qwts(directions) @ torch.pinverse(qwts(v1_population_directions))
        weight = weight.sum(dim=0)
        weight = weight - weight.mean()
        weights.append(weight)

    return torch.stack(weights)


# REVIEW This is very similar to `swts` -> unify?
def qwts(dirs: torch.Tensor) -> torch.Tensor:
    """Returns weights for interpolating squared directional derivative filters.
    
    Corresponds to `shQwts` in the original implementation.
    """
    # switch back to rectangular coordinates
    dirs = dirs.clone()
    dirs[:, 1] = atan3(dirs[:, 1], torch.ones_like(dirs[:, 1]))
    dirs = sphere2rec(dirs)

    # normalize the direction vectors
    d = torch.sqrt(torch.sum(dirs ** 2, dim=1, keepdim=True))
    dirs = dirs / d

    # precalculate factorials
    # In this case, we can just use a tensor since it's only the first seven factorials
    fac = torch.tensor([1, 1, 2, 6, 24, 120, 720], dtype=torch.float32)

    # generate the weighting vectors
    res = torch.zeros((dirs.size(0), 28), dtype=torch.float32)
    pt = 0
    for o3 in range(7):
        for o2 in range(7 - o3):
            o1 = 6 - o3 - o2
            pt += 1
            const = fac[6] / (fac[o3] * fac[o2] * fac[o1])
            res[:, pt - 1] = \
                const * dirs[:, 0] ** o1 * dirs[:, 1] ** o2 * dirs[:, 2] ** o3

    return res


def rec2sphere(rectangular_points: torch.Tensor) -> torch.Tensor:
    spherical_points = torch.zeros_like(rectangular_points)

    spherical_points[:, 0] = atan3(rectangular_points[:, 0], rectangular_points[:, 1])
    spherical_points[:, 1] = torch.atan2(
        rectangular_points[:, 2],
        torch.sqrt(torch.sum(rectangular_points[:, :2] ** 2, 1))
    )
    spherical_points[:, 2] = torch.sqrt(torch.sum(rectangular_points ** 2, 1))

    return spherical_points


def sphere2rec(spherical_points: torch.Tensor) -> torch.Tensor:
    """Converts spherical coordinates to rectangular coordinates.
    
    Corresponds to `sphere2rec` in the original implementation.
    """
    if spherical_points.shape[1] == 2:
        spherical_points = torch.cat([
            spherical_points,
            torch.ones_like(spherical_points[:, :1]),
        ], dim=-1)

    rectangular_points = torch.zeros_like(spherical_points)

    rectangular_points[:, 0] = (
        spherical_points[:, 2] *
        torch.cos(spherical_points[:, 1]) *
        torch.sin(spherical_points[:, 0])
    )
    rectangular_points[:, 1] = (
        spherical_points[:, 2] *
        torch.cos(spherical_points[:, 1]) *
        torch.cos(spherical_points[:, 0])
    )
    rectangular_points[:, 2] = (
        spherical_points[:, 2] *
        torch.sin(spherical_points[:, 1])
    )

    return rectangular_points


def swts(dirs: torch.Tensor):
    """Computes spherical weights for interpolating filter responses.
    
    Corresponds to `shSwts` in the original implementation.
    """
    # switch back to rectangular coordinates
    dirs = dirs.clone()
    dirs[:, 1] = atan3(dirs[:, 1], torch.ones_like(dirs[:, 1]))
    dirs = sphere2rec(dirs)

    # normalize the direction vectors
    d = torch.sqrt(torch.sum(dirs ** 2, dim=1, keepdim=True))
    dirs = dirs / d

    # precalculate factorials
    # In this case, we can just use a tensor since it's only the first four factorials
    fac = torch.tensor([1, 1, 2, 6], dtype=torch.float32)

    # generate the weighting vectors
    res = torch.zeros((dirs.size(0), 10), dtype=torch.float32)
    pt = 0
    for o3 in range(4):
        for o2 in range(4 - o3):
            o1 = 3 - o3 - o2
            pt += 1
            const = fac[3] / (fac[o3] * fac[o2] * fac[o1])
            res[:, pt - 1] = \
                const * dirs[:, 0] ** o1 * dirs[:, 1] ** o2 * dirs[:, 2] ** o3

    return res


def validCorrDn3(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Single channel, single filter 3D convolution without padding.

    Corresponds to `validCorrDn3` in the original implementation.
    """
    input = input.unsqueeze(0)  # add channel dimension
    weight = weight.unsqueeze(0).unsqueeze(0)  # add input/output channel dimensions
    return F.conv3d(input, weight).squeeze(0)  # remove channel dimension
