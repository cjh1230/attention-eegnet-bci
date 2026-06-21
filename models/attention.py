"""
Channel Attention module for EEG — learns spatial weights per channel.

Idea: automatically weight C3, Cz, C4 contributions without manual ROI selection.
"""
import torch
import torch.nn as nn


class ChannelAttention1D(nn.Module):
    """Squeeze-and-excitation style attention over EEG channels."""

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(n_channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(n_channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        s = self.squeeze(x).view(B, C)       # (B, C)
        w = self.excitation(s).view(B, C, 1)  # (B, C, 1)
        return x * w
