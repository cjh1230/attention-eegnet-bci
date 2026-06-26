"""
MAA-EEGNet-Pre: MAA applied BEFORE temporal conv on raw EEG.

Unlike MAA-EEGNet (MAA after temporal conv), this variant applies
motor-area attention directly to the raw (B, 8, T) input, then feeds
the reweighted EEG into a standard EEGNet.

This tests whether region-based preprocessing (rather than mid-network
attention) is a better inductive bias.
"""
import torch
import torch.nn as nn

from models.eegnet import EEGNet
from models.motor_area_attention import MotorAreaAttention


class MAAEEGNetPre(nn.Module):
    """EEGNet with MAA applied as preprocessing on raw EEG input.

    Parameters
    ----------
    n_channels, n_classes, F1, D, F2, dropout : EEGNet hyperparameters.
    """

    input_requires_filter_bank: bool = False

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.maa = MotorAreaAttention(n_channels=n_channels)
        self.eegnet = EEGNet(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=F1, D=D, F2=F2,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) — MAA applied first, then standard EEGNet."""
        x = self.maa(x)
        return self.eegnet(x)
