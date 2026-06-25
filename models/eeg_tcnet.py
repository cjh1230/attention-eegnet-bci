"""
EEG-TCNet: EEGNet backbone + Temporal Convolutional Network for MI-EEG.

Reference:
    Ingolfsson et al., "EEG-TCNet: An Accurate Temporal Convolutional Network
    for Embedded Motor-Imagery Brain-Machine Interfaces" (arXiv:2006.00622)

Architecture:
    EEGNet Block 1 (temporal + spatial conv) → TCN (dilated depthwise convs
    with residual connections) → pointwise conv → avg pool → FC.

Compatible with EEGNet's (B, C, T) input — no special preprocessing needed.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TCN Block
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """Stack of dilated depthwise separable conv1d layers with residual
    connections.

    Parameters
    ----------
    in_channels : int
        Number of input channels (D * F1 from EEGNet Block 1).
    kernel_size : int
        Temporal kernel size for each depthwise conv layer.
    dilations : list[int]
        Dilation rates for each layer (typically [1, 2, 4, 8]).
    dropout : float
        Dropout probability after each layer.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 16,
        dilations: list[int] | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.layers = nn.ModuleList()
        for d in dilations:
            total_pad = (kernel_size - 1) * d
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            self.layers.append(
                nn.Sequential(
                    nn.ConstantPad1d((pad_left, pad_right), 0.0),
                    nn.Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size,
                        dilation=d,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.ELU(),
                    nn.Dropout(dropout),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, C, T) with residual connections."""
        for layer in self.layers:
            residual = x
            x = layer(x)
            if x.shape[-1] == residual.shape[-1]:
                x = x + residual
        return x


# ---------------------------------------------------------------------------
# EEG-TCNet
# ---------------------------------------------------------------------------

class EEGTCNet(nn.Module):
    """EEG-TCNet: EEGNet temporal/spatial conv + TCN for MI decoding.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default 8).
    n_classes : int
        Number of output classes (default 3).
    F1 : int
        Temporal filters in Block 1 (default 8).
    D : int
        Depth multiplier (default 2 → D*F1 = 16 spatial filters).
    F2 : int
        Output channels after pointwise conv (default 16).
    tcn_kernel : int
        TCN depthwise kernel size (default 16).
    tcn_dilations : list[int]
        TCN dilation schedule (default [1, 2, 4, 8]).
    dropout : float
        Dropout probability (default 0.5).
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        tcn_kernel: int = 16,
        tcn_dilations: list[int] | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if tcn_dilations is None:
            tcn_dilations = [1, 2, 4, 8]

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        tcn_in = D * F1

        # ── Block 1: EEGNet temporal + spatial convolution ──
        self.block1 = nn.Sequential(
            # Temporal
            nn.Conv2d(1, F1, (1, 64), bias=False),
            nn.BatchNorm2d(F1),
            # Spatial (depthwise)
            nn.Conv2d(F1, tcn_in, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(tcn_in),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # ── TCN Block ──
        self.tcn = TCNBlock(
            in_channels=tcn_in,
            kernel_size=tcn_kernel,
            dilations=tcn_dilations,
            dropout=dropout,
        )

        # ── Pointwise conv + output ──
        self.output = nn.Sequential(
            nn.Conv1d(tcn_in, F2, 1, bias=False),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(F2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)
            Single-band EEG input (same format as EEGNet).

        Returns
        -------
        logits : torch.Tensor, shape (B, n_classes)
        """
        # Add channel dim for Conv2d: (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1 → (B, D*F1, 1, T')
        x = self.block1(x)

        # Squeeze spatial dim → (B, D*F1, T')
        x = x.squeeze(2)

        # TCN → (B, D*F1, T')
        x = self.tcn(x)

        # Output → (B, n_classes)
        return self.output(x)
