"""Unimatch (Xu et al. 2023).

This module provides a thin wrapper around the original Unimatch implementation to
make it compatible with the standard model interface.

!!! quote "Reference"
    Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao,
    Andreas Geiger.<br>
    *Unifying Flow, Stereo and Depth Estimation*.<br>
    TPAMI 2023.

    [Paper](https://ieeexplore.ieee.org/abstract/document/10193833/) •
    [Code](https://github.com/autonomousvision/unimatch)
"""
import torch
from torch import nn

from ._extern.main_flow import get_args_parser
from ._extern.unimatch.unimatch import UniMatch


class GMFlow(nn.Module):
    """GMFlow model."""

    __version__ = "1.0.0"

    def __init__(self, original_model: UniMatch, **kwargs) -> None:
        """Initializes the GMFlow model.
        
        Parameters:
            original_model: Unimatch instance from the original codebase.
        """
        super().__init__()
        self.original_model = original_model
        self.kwargs = kwargs

    @staticmethod
    def from_config(config: dict) -> 'GMFlow':
        """Creates a GMFlow model from a configuration dictionary.
        
        Parameters:
            config: Configuration dictionary.
        """
        _type = config.pop("_type", "GMFlow")
        if _type != "GMFlow":
            raise ValueError(f"Expected _type to be 'GMFlow', but got '{_type}'.")

        # Use the default configuration from the original codebase
        parser = get_args_parser()
        args = parser.parse_args([])

        original_model = UniMatch(
            feature_channels=config.get("feature_channels", args.feature_channels),
            num_scales=config.get("num_scales", args.num_scales),
            upsample_factor=config.get("upsample_factor", args.upsample_factor),
            num_head=config.get("num_head", args.num_head),
            ffn_dim_expansion=config.get("ffn_dim_expansion", args.ffn_dim_expansion),
            num_transformer_layers=config.get("num_transformer_layers", args.num_transformer_layers),  # noqa: E501
            reg_refine=config.get("reg_refine", args.reg_refine),
            task="flow",
        )

        kwargs = {
            "attn_type": config.get("attn_type", args.attn_type),
            "attn_splits_list": config.get("attn_splits_list", args.attn_splits_list),
            "corr_radius_list": config.get("corr_radius_list", args.corr_radius_list),
            "prop_radius_list": config.get("prop_radius_list", args.prop_radius_list),
            "num_reg_refine": config.get("num_reg_refine", args.num_reg_refine),
        }

        if "checkpoint" in config:
            checkpoint = torch.load(config["checkpoint"])
            original_model.load_state_dict(checkpoint["model"])

        return GMFlow(original_model, **kwargs)

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
        frame0, frame1 = x.unbind(dim=2)
        frame0 = frame0.to(self.device)
        frame1 = frame1.to(self.device)
        
        output = self.original_model(frame0, frame1, **self.kwargs)

        return output["flow_preds"][-1]
