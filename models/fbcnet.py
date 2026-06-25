"""
FBCNet: Filter-Bank Convolutional Network for MI-BCI.

Reference:
    Bakshi et al., "FBCNet: A Multi-view Convolutional Neural Network
    for Brain-Computer Interface" (arXiv:2104.01233)

Architecture:
    Multi-band EEG → Spatial Conv (per band) → Temporal Depthwise Conv
    → Variance Pooling → Log → FC → Output

The model expects input of shape (B, n_bands, C, T).  Use
``apply_filter_bank()`` to convert standard (B, C, T) trials into
multi-band format before feeding to the model.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Filter bank helper (numpy / scipy — runs once per batch on CPU)
# ---------------------------------------------------------------------------

def _bandpass_np(data: np.ndarray, low: float, high: float,
                 fs: int = 250, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass filter along the last axis (numpy array)."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def apply_filter_bank(
    X: np.ndarray,
    fs: int = 250,
    bands: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Convert (N, C, T) trials into (N, n_bands, C, T) multi-band format.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
        Single-band EEG trials (typically already 8–30 Hz filtered).
    fs : int
        Sampling frequency in Hz.
    bands : list of (low, high) or None
        Frequency bands.  If None, uses ``FBCSP_BANDS`` from config.

    Returns
    -------
    X_mb : np.ndarray, shape (N, n_bands, C, T)
    """
    if bands is None:
        from utils.config import FBCSP_BANDS as bands

    bands_out = []
    for low, high in bands:
        bands_out.append(_bandpass_np(X, low, high, fs=fs))
    return np.stack(bands_out, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# FBCNet model
# ---------------------------------------------------------------------------

class FBCNet(nn.Module):
    """Filter-Bank Convolutional Network for motor imagery decoding.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands in the input (default 9).
    n_channels : int
        Number of EEG channels.
    n_classes : int
        Number of output classes.
    m : int
        Number of spatial filters per band (default 32).
    t_kernel : int
        Temporal depthwise convolution kernel size (default 64).
    dropout : float
        Dropout probability after the hidden FC layer (default 0.5).
    hidden : int
        Hidden FC layer size (default 32).
    """

    # Signal to training scripts that this model needs multi-band input.
    input_requires_filter_bank: bool = True

    def __init__(
        self,
        n_bands: int = 9,
        n_channels: int = 8,
        n_classes: int = 3,
        m: int = 32,
        t_kernel: int = 64,
        dropout: float = 0.5,
        hidden: int = 32,
    ) -> None:
        super().__init__()

        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.m = m
        self.t_kernel = t_kernel

        # ── Spatial Convolution Block (SCB) ──
        # (C, 1) kernel collapses all channels into one spatial dimension
        # while learning M different spatial filters.
        self.scb = nn.Sequential(
            nn.Conv2d(1, m, (n_channels, 1), bias=False),
            nn.BatchNorm2d(m),
        )

        # ── Temporal depthwise convolution ──
        # Each spatial filter gets its own temporal kernel (groups=m).
        self.temporal = nn.Sequential(
            nn.Conv2d(m, m, (1, t_kernel), groups=m, bias=False),
            nn.BatchNorm2d(m),
            nn.ELU(),
        )

        # ── Classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(n_bands * m, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, n_bands, C, T)
            Multi-band EEG input.

        Returns
        -------
        logits : torch.Tensor, shape (B, n_classes)
        """
        B, n_bands = x.shape[:2]

        # Merge batch and band dims → (B * n_bands, 1, C, T)
        x = x.reshape(B * n_bands, 1, self.n_channels, -1)

        # Spatial convolution → (B * n_bands, M, 1, T)
        x = self.scb(x)

        # Temporal depthwise conv → (B * n_bands, M, 1, T')
        x = self.temporal(x)

        # Squeeze spatial dim → (B * n_bands, M, T')
        x = x.squeeze(2)

        # Variance pooling along time → (B * n_bands, M)
        x = x.var(dim=-1, unbiased=False)

        # Log transform (mimics CSP log-variance)
        x = torch.log(x + 1e-6)

        # Reshape back → (B, n_bands * M)
        x = x.view(B, -1)

        return self.classifier(x)
