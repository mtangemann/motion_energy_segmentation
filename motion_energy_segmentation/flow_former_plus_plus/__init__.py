"""FlowFormer++ (Shi et al. 2023).

This module provides a thin wrapper around the original FlowFormer++ implementation to
make it compatible with the standard model interface.

!!! quote "Reference"
    Xiaoyu Shi, Zhaoyang Huang, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See,
    Hongwei Qin, Jifeng Dai, Hongsheng Li.<br>
    *FlowFormer++: Masked cost volume autoencoding for pretraining optical flow
    estimation.*.<br>
    CVPR 2023.

    [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Shi_FlowFormer_Masked_Cost_Volume_Autoencoding_for_Pretraining_Optical_Flow_Estimation_CVPR_2023_paper.html)
    • [Code](https://github.com/XiaoyuShi97/FlowFormerPlusPlus)
"""

import logging
import sys
from importlib import import_module
from typing import Any

import torch
from torch import nn

from ._extern.core.FlowFormer import build_flowformer
from ._extern.core.FlowFormer.PerCostFormer3.transformer import FlowFormer

LOGGER = logging.getLogger(__name__)

_RECOMMENDED_DEPENDENCIES = {
    "timm": "0.4.12",
    "torch": "1.6.0",
    "torchvision": "0.7.0",
}


class FlowFormerPlusPlus(nn.Module):
    """FlowFormer++ model."""

    __version__ = "1.0.0"

    def __init__(self, original_model: FlowFormer) -> None:
        """Initializes the FlowFormer++ model.
        
        Parameters:
            original_model: FlowFormer++ instance from the original implementation.
        """
        super().__init__()
        self.original_model = original_model

    @staticmethod
    def from_config(config: dict[str, Any]) -> "FlowFormerPlusPlus":
        """Creates a FlowFormer++ model from a configuration dictionary.
        
        Parameters:
            config: Configuration dictionary.
        """
        _type = config.pop("_type", "FlowFormerPlusPlus")
        if _type != "FlowFormerPlusPlus":
            raise ValueError(
                f"Expected _type to be 'FlowFormerPlusPlus', but got '{_type}'."
            )
        
        # Warn if different versions of the recommended dependencies are installed.
        for package, version in _RECOMMENDED_DEPENDENCIES.items():
            if package not in sys.modules:
                LOGGER.warning(
                    f"Recommended dependency '{package}' not found. "
                    f"Please install version {version} to ensure compatibility."
                )
            elif sys.modules[package].__version__ != version:
                LOGGER.warning(
                    f"Recommended dependency '{package}' is installed in version "
                    f"{sys.modules[package].__version__}. "
                    f"Please install version {version} to ensure compatibility."
                )
        
        config_module = import_module(
            f"neuroai.models.flow_former_plus_plus._extern.configs.{config['config']}"
        )
        cfg = config_module.get_cfg()

        model = build_flowformer(cfg)

        if "checkpoint" in config:
            checkpoint = torch.load(config["checkpoint"])
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)

        return FlowFormerPlusPlus(model)

    @property
    def device(self) -> torch.device:
        """Returns the device of the model."""
        return next(iter(self.parameters())).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts the optical flow between two frames.
        
        Parameters:
            x: Input clips with shape (B, C, 2, H, W) and values in the range [0, 255].

        Returns:
            Forward flow as tensor with shape (B, 2, H, W).
        """
        _, _, T, _, _ = x.shape
        if T != 2:
            raise ValueError(f"Expected input tensor to have 2 frames, but got {T}.")

        # Normalization is handled by the original model.
        frame0 = x[:, :, 0].to(self.device, dtype=torch.float32)
        frame1 = x[:, :, 1].to(self.device, dtype=torch.float32)

        # The original model returns flow fields obtained for several iterations. We're
        # only using the last one.
        flow = self.original_model(frame0, frame1)[-1]

        return flow
