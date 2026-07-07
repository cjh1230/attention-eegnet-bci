"""
EEG-TCNet: EEGNet backbone + Temporal Convolutional Network for MI-EEG.

Reference:
    Ingolfsson et al., "EEG-TCNet: An Accurate Temporal Convolutional Network
    for Embedded Motor-Imagery Brain-Machine Interfaces" (arXiv:2006.00622)
    Official repo: https://github.com/iis-eth-zurich/eeg-tcnet

Architecture (aligned with ETH Zurich official implementation):
    EEGNet Block 1 (temporal + spatial conv) → TCN (double-conv blocks with
    causal dilated convolutions + residual connections) → pointwise conv →
    avg pool → FC.

Key design choices from official:
    - Causal padding: each time step only sees past (critical for real-time).
    - Double-conv TCN blocks: Conv1D→BN→Act→Drop→Conv1D→BN→Act→Drop→+residual.
    - 1×1 conv skip connection when channel dimensions mismatch.
    - Standard (non-depthwise) Conv1D for TCN expressiveness.
    - Exponentially increasing dilation: 1, 2, 4, 8, …

Compatible with EEGNet's (B, C, T) input — no special preprocessing needed.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# TCN Block (double-conv, causal — aligned with official ETH Zurich impl)
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """Stack of double-conv dilated Conv1d layers with residual connections.

    Each layer = two Conv1D sub-layers with the same dilation, matching the
    WaveNet-style design used in the official EEG-TCNet (Keras) implementation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (D * F1 from EEGNet Block 1).
    out_channels : int
        Number of output channels (same as in_channels for residual; use
        a different value to project).
    kernel_size : int
        Temporal kernel size for each Conv1d layer.
    depth : int
        Number of double-conv blocks (dilations = [2^0, 2^1, …, 2^{depth-1}]).
    dropout : float
        Dropout probability after each sub-layer.
    causal : bool
        If True, use causal padding (only past context). Recommended for
        real-time BCI. If False, use symmetric padding (offline only).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 10,
        depth: int = 3,
        dropout: float = 0.0,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.causal = causal
        self.layers = nn.ModuleList()

        for i in range(depth):
            dilation = 2 ** i
            self.layers.append(
                _DoubleConvBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout,
                    causal,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, T) → (B, C_out, T)."""
        for layer in self.layers:
            x = layer(x)
        return x


class _DoubleConvBlock(nn.Module):
    """Two Conv1d sub-layers with residual connection (official WaveNet design).

    Conv1D → BN → Activation → Dropout →
    Conv1D → BN → Activation → Dropout →
    Add(input, projected via 1×1 conv if dims differ) → Activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()

        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size,
                                   dilation, causal)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size,
                                   dilation, causal)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        # 1×1 skip-projection when channel dims mismatch
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        self.out_act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return self.out_act(out + residual)


class _CausalConv1d(nn.Module):
    """1D convolution with optional causal (left-only) padding.

    Causal mode: pads (kernel_size - 1) * dilation zeros on the LEFT only,
    so output[t] depends only on input[≤t]. This is the official EEG-TCNet
    design and is required for real-time BCI.

    Symmetric mode: splits padding evenly (standard for offline training).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size

        if causal:
            self.pad_left = (kernel_size - 1) * dilation
            self.pad_right = 0
            self.padder = nn.ConstantPad1d((self.pad_left, self.pad_right), 0.0)
        else:
            total_pad = (kernel_size - 1) * dilation
            self.pad_left = total_pad // 2
            self.pad_right = total_pad - self.pad_left
            self.padder = nn.ConstantPad1d((self.pad_left, self.pad_right), 0.0)

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.padder(x))


# ---------------------------------------------------------------------------
# EEG-TCNet
# ---------------------------------------------------------------------------

class EEGTCNet(nn.Module):
    """EEG-TCNet: EEGNet temporal/spatial conv + TCN for MI decoding.

    Aligned with the official ETH Zurich implementation (v2020):
    - Causal TCN (default) for real-time compatibility.
    - Double-conv WaveNet-style TCN blocks.
    - Standard (non-depthwise) Conv1d in TCN for expressiveness.
    - Exponentially increasing dilation schedule.

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
        TCN kernel size (default 10, matches official).
    tcn_depth : int
        Number of double-conv TCN blocks (default 3, matches official for
        250 Hz × 3 s ≈ 750 time samples → receptive field ~ 70 samples).
    tcn_filters : int
        Number of filters in each TCN Conv1d (default 10, matches official).
    tcn_dropout : float
        Dropout probability in TCN (default 0.0 as in official).
    causal : bool
        Use causal padding (default True, matches official). Set to False
        for symmetric padding (legacy behaviour).
    dropout : float
        Dropout probability in EEGNet Block 1 (default 0.5).
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        tcn_kernel: int = 10,
        tcn_depth: int = 3,
        tcn_filters: int = 10,
        tcn_dropout: float = 0.0,
        causal: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.causal = causal

        tcn_in = D * F1  # channels out of Block 1, into TCN

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

        # ── TCN Block (double-conv, causal, matching official) ──
        self.tcn = TCNBlock(
            in_channels=tcn_in,
            out_channels=tcn_filters,
            kernel_size=tcn_kernel,
            depth=tcn_depth,
            dropout=tcn_dropout,
            causal=causal,
        )

        # ── Pointwise conv + output ──
        self.output = nn.Sequential(
            nn.Conv1d(tcn_filters, F2, 1, bias=False),
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

        # TCN → (B, tcn_filters, T')
        x = self.tcn(x)

        # Output → (B, n_classes)
        return self.output(x)
