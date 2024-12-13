import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F
from mmflow.apis import inference_model, init_model
from torch import nn

from .simoncelli_heeger import SimoncelliHeegerModel

LOGGER = logging.getLogger(__name__)


class MotionSegmentationModel(nn.Module):
    def __init__(
        self,
        motion_estimator: nn.Module,
        segmenter: nn.Module,
        input_pyramid: nn.Module | None = None,
        output_pyramid: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if (input_pyramid is None) == (output_pyramid is None):
            raise ValueError(
                "Expected exactly one pyramid (input or output).",
            )

        self.motion_estimator = motion_estimator
        self.segmenter = segmenter
        self.input_pyramid = input_pyramid
        self.output_pyramid = output_pyramid

    @property
    def uses_motion_energy(self) -> bool:
        return isinstance(self.motion_estimator, SimoncelliHeegerCNN)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.input_pyramid is not None:
            flow = None
            motion = [
                self.motion_estimator(input).squeeze(2)
                for input in list(reversed(self.input_pyramid(input)))
            ]
        else:
            flow = self.motion_estimator(input)
            motion = list(reversed(self.output_pyramid(flow)))

        return self.segmenter(motion)[-1], flow

    @staticmethod
    def from_config(config: dict[str, Any]) -> "MotionSegmentationModel":
        """Creates a MotionSegmentationNetwork from a config dict."""
        config = deepcopy(config)

        _type = config.pop("_type", "MotionSegmentationModel")
        if _type != "MotionSegmentationModel":
            raise ValueError(
                f"Expected _type to be 'MotionSegmentationModel', but got '{_type}'."
            )

        motion_estimator_type = config["motion_estimator._type"]

        if motion_estimator_type == "SimoncelliHeegerCNN":
            motion_estimator = SimoncelliHeegerCNN.from_config(
                config.pop("motion_estimator")
            )
            input_pyramid = ImagePyramid(config.pop("levels"))
            output_pyramid = None

        elif motion_estimator_type == "MMFlowEstimator":
            motion_estimator = MMFlowEstimator.from_config(
                config.pop("motion_estimator")
            )
            for parameter in motion_estimator.parameters():
                parameter.requires_grad = False
            input_pyramid = None
            output_pyramid = FeaturePyramid(config.pop("levels"))

        elif motion_estimator_type == "GMFlow":
            from .unimatch import GMFlow
            motion_estimator = GMFlow.from_config(config.pop("motion_estimator"))
            for parameter in motion_estimator.parameters():
                parameter.requires_grad = False
            input_pyramid = None
            output_pyramid = FeaturePyramid(config.pop("levels"))
        
        elif motion_estimator_type == "FlowFormerPlusPlus":
            from .flow_former_plus_plus import FlowFormerPlusPlus
            flow_former_config = config.pop("motion_estimator")
            motion_estimator = FlowFormerPlusPlus.from_config(flow_former_config)
            for parameter in motion_estimator.parameters():
                parameter.requires_grad = False
            input_pyramid = None
            output_pyramid = FeaturePyramid(config.pop("levels"))

        else:
            raise ValueError(
                f"Unsupported motion estimator type: {motion_estimator_type}"
            )

        segmenter = RefineNet.from_config(config.pop("segmenter"))

        return MotionSegmentationModel(
            motion_estimator,
            segmenter,
            input_pyramid=input_pyramid,
            output_pyramid=output_pyramid,
        )


class ImagePyramid(nn.Module):
    """Image pyramid based on downsampling by a factor of 2."""

    def __init__(self, levels: Sequence[int]) -> None:
        """Return a list of images downsampled to the specified levels."""
        super().__init__()

        levels = sorted(set(levels))
        if levels[0] < 0:
            raise ValueError(
                f"Expected all levels to be non-negative, but got {levels=}"
            )
        self.levels = levels

        self.blur = nn.Sequential(
            nn.Conv3d(
                1, 1, (1, 1, 5), bias=False, padding="same", padding_mode="replicate"),
            nn.Conv3d(
                1, 1, (1, 5, 1), bias=False, padding="same", padding_mode="replicate"),
        )
        weight = torch.Tensor([0.0884, 0.3536, 0.5303, 0.3536, 0.0884])
        weight_x = weight.view(1, 1, 1, 1, -1).expand_as(self.blur[0].weight)
        self.blur[0].weight.data = weight_x
        weight_y = weight.view(1, 1, 1, -1, 1).expand_as(self.blur[1].weight)
        self.blur[1].weight.data = weight_y
        for parameter in self.blur.parameters():
            parameter.requires_grad = False

        self.downsample = nn.AvgPool3d((1, 2, 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return a list of images downsampled to the specified levels."""
        pyramid = []

        for level in range(self.levels[-1] + 1):
            if level in self.levels:
                pyramid.append(input)

            input = self.downsample(self.blur(input))

        return pyramid


class SimoncelliHeegerCNN(nn.Sequential):
    """CNN based on the Simoncelli & Heeger (1998) model.
    
    This keeps the linear filters, nonlinearities, and normalization from the original
    model. Scale factors are dropped everywhere.
    """

    def __init__(
        self,
        padding: Literal["same", "valid"] = "same",
        padding_time: Literal["same", "valid"] | None = None,
    ) -> None:
        if padding_time is None:
            padding_time = padding

        layers = OrderedDict()
        layers["v1_linear"] = nn.Sequential(OrderedDict([
            ("conv_t", nn.Conv3d(
                1, 10, (9, 1, 1),
                bias=False,
                padding=padding_time,
                padding_mode="replicate",
            )),
            ("conv_y", nn.Conv3d(
                10, 10, (1, 9, 1),
                groups=10,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )),
            ("conv_x", nn.Conv3d(
                10, 10, (1, 1, 9),
                groups=10,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )),
            ("readout", nn.Conv3d(
                10, 28, (1, 1, 1),
                bias=False,
                padding=padding,
                padding_mode="replicate",
            )),
        ]))
        layers["v1_nonlinear"] = Square()
        layers["v1_blur"] = nn.Sequential(
            nn.Conv3d(
                28, 28, (1, 11, 1),
                groups=28,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            ),
            nn.Conv3d(
                28, 28, (1, 1, 11),
                groups=28,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            ),
        )
        layers["v1_normalize"] = ChannelNorm()
        layers["mt_linear"] = nn.Conv3d(28, 19, 1, bias=False)
        layers["mt_blur"] = nn.Sequential(
            nn.Conv3d(
                19, 19, (1, 19, 1),
                groups=19,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            ),
            nn.Conv3d(
                19, 19, (1, 1, 19),
                groups=19,
                bias=False,
                padding=padding,
                padding_mode="replicate",
            ),
        )
        layers["mt_nonlinear"] = RectifiedSquare()
        layers["mt_normalize"] = ChannelNorm()

        super().__init__(layers)
    
    @staticmethod
    def from_original_model(
        padding: Literal["same", "valid"] = "same",
        padding_time: Literal["same", "valid"] | None = None,
        layers: Sequence[str] | None = None,
    ) -> "SimoncelliHeegerCNN":
        LOGGER.info("Initializing SimoncelliHeegerCNN from original model")

        if layers is None:
            layers = ["v1_linear", "v1_blur", "mt_linear", "mt_blur"]

        model = SimoncelliHeegerCNN(padding, padding_time)
        original_model = SimoncelliHeegerModel()

        if "v1_linear" in layers:
            model.v1_linear.conv_t.weight.data = \
                original_model.v1_linear.conv_t.weight.data
            model.v1_linear.conv_y.weight.data = \
                original_model.v1_linear.conv_y.weight.data
            model.v1_linear.conv_x.weight.data = \
                original_model.v1_linear.conv_x.weight.data
            model.v1_linear.readout.weight.data = \
                original_model.v1_linear.readout.weight.data

        if "v1_blur" in layers:
            blur_weight = original_model.v1_blur.complex_filters
            model.v1_blur[0].weight.data = \
                blur_weight.view(1, 1, 1, -1, 1).expand_as(model.v1_blur[0].weight)
            model.v1_blur[1].weight.data = \
                blur_weight.view(1, 1, 1, 1, -1).expand_as(model.v1_blur[1].weight)

        if "mt_linear" in layers:
            model.mt_linear.weight.data = original_model.mt_linear.conv.weight.data

        if "mt_blur" in layers:
            blur_weight = original_model.mt_blur.spatial_pooling_filter
            model.mt_blur[0].weight.data = \
                blur_weight.view(1, 1, 1, -1, 1).expand_as(model.mt_blur[0].weight)
            model.mt_blur[1].weight.data = \
                blur_weight.view(1, 1, 1, 1, -1).expand_as(model.mt_blur[1].weight)

        return model

    @staticmethod
    def from_config(config: dict[str, Any]) -> "SimoncelliHeegerCNN":
        """Creates a SimoncelliHeegerCNN from a config dict."""
        config = deepcopy(config)

        _type = config.pop("_type", "SimoncelliHeegerCNN")
        if _type != "SimoncelliHeegerCNN":
            raise ValueError(
                f"Expected _type to be 'SimoncelliHeegerCNN', but got '{_type}'."
            )

        from_original_model = config.get("from_original_model", False)
        freeze_original_model = config.get("freeze_original_model", False)

        padding = config.get("padding", "same")
        padding_time = config.get("padding_time", "same")

        if from_original_model is not False:
            if isinstance(from_original_model, bool):
                layers = None
            else:
                layers = from_original_model
                
            model = SimoncelliHeegerCNN.from_original_model(
                padding=padding, padding_time=padding_time,layers=layers,
            )

        else:
            model = SimoncelliHeegerCNN(padding=padding, padding_time=padding_time)
        
        if isinstance(freeze_original_model, bool):
            for parameter in model.parameters():
                parameter.requires_grad = not freeze_original_model

        else:
            for name, module in model.named_children():
                if name in freeze_original_model:
                    for parameter in module.parameters():
                        parameter.requires_grad = False

        return model


class Square(nn.Module):
    def forward(self, x):
        return x ** 2


class ChannelNorm(nn.Module):
    def forward(self, x, epsilon: float = 1e-6):
        return x / (torch.sum(x, dim=1, keepdim=True) + epsilon)


class RectifiedSquare(nn.Module):
    def forward(self, x):
        return torch.relu(x) ** 2


class RefineNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        out_channels: int,
        input_cnn: dict[str, Any],
        refine_cnn: dict[str, Any],
        output_cnn: dict[str, Any],
    ) -> None:
        super().__init__()

        input_cnn_depth = input_cnn.pop("depth")
        input_cnn_channels = [in_channels] + [channels] * input_cnn_depth
        self.input_cnn = ConvNet2d(input_cnn_channels, **input_cnn)

        refine_cnn_depth = refine_cnn.pop("depth")
        refine_cnn_channels = [channels * 2] + [channels] * refine_cnn_depth
        self.refine_cnn = ConvNet2d(refine_cnn_channels, **refine_cnn)

        output_cnn_depth = output_cnn.pop("depth")
        output_cnn_channels = [channels] * output_cnn_depth + [out_channels]
        self.output_cnn = ConvNet2d(output_cnn_channels, **output_cnn)

        self.prior = nn.Parameter(torch.randn(1, channels, 16, 16))


    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        xs = [self.input_cnn(x) for x in xs]

        outputs = []

        state = self.prior.expand(xs[0].shape[0], -1, -1, -1)
        for x in xs:
            state = torch.cat([state, x], dim=1)
            state = self.refine_cnn(state)
            outputs.append(self.output_cnn(state))
            state = F.interpolate(
                state, scale_factor=2, mode="bilinear", align_corners=False)
        
        return outputs

    @staticmethod
    def from_config(config: dict[str, Any]) -> "RefineNet":
        """Creates a RefineNet from a config dict."""
        _type = config.pop("_type", "RefineNet")
        if _type != "RefineNet":
            raise ValueError(f"Expected _type to be 'RefineNet', but got '{_type}'.")

        return RefineNet(**config)


class ConvNet2d(nn.Sequential):
    """Generic 2D convolutional network."""

    def __init__(
        self,
        channels: Sequence[int],
        kernel_size: int | Sequence[int],
        dilation: int | Sequence[int] = 1,
        normalization: Literal["BatchNorm", "InstanceNorm"] = "InstanceNorm",
        final_normalization: bool = False,
        activation: Literal["ReLU", "CELU"] = "CELU",
        final_activation: bool = False,
    ) -> None:
        """Initializes the network.
        
        Parameters:
            channels: The number of channels in each layer. The first element is the
                number of input channels, and the last element is the number of output
                channels. So, len(channels) - 1 is the number of convolutional layers.
            kernel_size: The kernel size of each convolutional layer. If an int, the
                same kernel size is used for all layers.
            dilation: The dilation of each convolutional layer. If an int, the same
                dilation is used for all layers.
            normalization: The type of normalization to use after each convolutional
                layer. Either "BatchNorm" or "InstanceNorm".
            final_normalization: Whether to apply normalization after the final conv
                layer.
            activation: The type of activation to use after each convolutional layer.
                Either "ReLU" or "CELU".
            final_activation: Whether to apply activation after the final conv layer.

        Raises:
            ValueError: If the length of `kernel_size` or `dilation` is not equal to
                `len(channels) - 1`.
        """
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (len(channels) - 1)
        elif len(kernel_size) != len(channels) - 1:
            raise ValueError(
                "Expected len(kernel_size) == len(channels) - 1, but got "
                f"{len(kernel_size)=} and {len(channels)=}."
            )

        if isinstance(dilation, int):
            dilation = [dilation] * (len(channels) - 1)
        elif len(dilation) != len(channels) - 1:
            raise ValueError(
                "Expected len(dilation) == len(channels) - 1, but got "
                f"{len(dilation)=} and {len(channels)=}."
            )
    
        layers = []
 
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(
                    channels[i], channels[i+1], kernel_size[i], dilation=dilation[i],
                    padding="same", padding_mode="reflect",
                )
            )

            if i < len(channels) - 2 or final_normalization:
                if normalization == "BatchNorm":
                    layers.append(nn.BatchNorm2d(channels[i+1]))
                elif normalization == "InstanceNorm":
                    layers.append(nn.InstanceNorm2d(channels[i+1]))
                else:
                    raise ValueError(f"Unsupported normalization type: {normalization}")

            if i < len(channels) - 2 or final_activation:
                if activation == "ReLU":
                    layers.append(nn.ReLU())
                elif activation == "CELU":
                    layers.append(nn.CELU())
                else:
                    raise ValueError(f"Unknown activation type: {activation}")

        super().__init__(*layers)


class MMFlowEstimator(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        frame0 = list(input[:, :, 0].movedim(1, -1).numpy())
        frame1 = list(input[:, :, 1].movedim(1, -1).numpy())

        outputs = inference_model(self.model, frame0, frame1, [None] * len(frame0))

        flow = torch.stack([
            torch.tensor(output["flow"]).movedim(-1, 0) for output in outputs
        ], dim=0)
        flow = flow.to(next(iter(self.parameters())).device)

        return flow

    @staticmethod
    def from_config(config: dict[str, Any]) -> "MMFlowEstimator":
        """Creates a MMFlowEstimator from a config dict."""
        _type = config.pop("_type", "MMFlowEstimator")
        if _type != "MMFlowEstimator":
            raise ValueError(
                f"Expected _type to be 'MMFlowEstimator', but got '{_type}'."
            )

        model = init_model(config["config"], config["checkpoint"])
        return MMFlowEstimator(model)


class FeaturePyramid(nn.Module):
    def __init__(self, levels: Sequence[int]) -> None:
        super().__init__()
        self.levels = levels
    
    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        pyramid = []

        for level in range(self.levels[-1] + 1):
            if level in self.levels:
                pyramid.append(input)

            input = F.interpolate(
                input, scale_factor=0.5, mode="bilinear", align_corners=False
            )

        return pyramid
