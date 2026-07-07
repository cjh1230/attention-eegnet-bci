"""
FB-TCNet: Filter-Bank Temporal Convolutional Network for MI-BCI.

Integrates three independently-verified effective components:
  1. Filter Bank (from FBCNet, Bakshi 2021) — multi-band decomposition
  2. TCN backbone (from EEG-TCNet, Ingolfsson 2020) — dilated temporal modeling
  3. Euclidean Alignment — cross-subject covariance alignment

Architecture:
    (B, C, T) → apply_filter_bank → (B, n_bands, C, T)
    → merge batch+band → (B*n_bands, 1, C, T)
    → EEGNet Block1 (temporal + spatial conv) → (B*nb, D*F1, 1, T')
    → TCN Block (dilated depthwise convs + residual) → (B*nb, D*F1, T')
    → AdaptiveAvgPool1d → (B*nb, D*F1)
    → Reshape per band → (B, n_bands, D*F1)
    → Concat + Classifier → (B, n_classes)

This is the project's first original model — each component is validated
independently on this dataset and task, and their combination is novel.

Reference:
    - FBCNet:  Bakshi et al. 2021 (arXiv:2104.01233)
    - EEG-TCNet: Ingolfsson et al. 2020 (arXiv:2006.00622)
    - FB-TCNet: this project (2026)
"""
import torch
import torch.nn as nn

from models.eeg_tcnet import TCNBlock


class FBTCNet(nn.Module):
    """Filter-Bank Temporal Convolutional Network.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands (default 6 for 8–30 Hz sub-bands).
    n_channels : int
        Number of EEG channels (default 8).
    n_classes : int
        Number of output classes.
    F1 : int
        Temporal filters in Block 1 (default 8).
    D : int
        Depth multiplier (default 2 → D*F1 = 16 spatial filters).
    F2 : int
        Output channels per band after pooling (default 16).
    tcn_kernel : int
        TCN kernel size (default 10, matches official).
    tcn_depth : int
        Number of double-conv TCN blocks (default 3, matches official).
    tcn_filters : int
        Number of filters in each TCN Conv1d (default 10, matches official).
    dropout : float
        Dropout probability (default 0.5).
    causal : bool
        Use causal padding in TCN (default True, matches official).
    """

    input_requires_filter_bank: bool = True

    def __init__(
        self,
        n_bands: int = 6,
        n_channels: int = 8,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        tcn_kernel: int = 10,
        tcn_depth: int = 3,
        tcn_filters: int = 10,
        causal: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        tcn_in = D * F1  # channels into TCN

        # ---- Block 1: EEGNet temporal + spatial conv (shared across bands) ----
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, tcn_in, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(tcn_in),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # ---- TCN Block (shared across bands, double-conv + causal) ----
        self.tcn = TCNBlock(
            in_channels=tcn_in,
            out_channels=tcn_filters,
            kernel_size=tcn_kernel,
            depth=tcn_depth,
            dropout=dropout,
            causal=causal,
        )

        # ---- Per-band pooling + pointwise ----
        self.band_pool = nn.Sequential(
            nn.Conv1d(tcn_filters, F2, 1, bias=False),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),   # (B*nb, F2, 1)
        )

        # ---- Classifier (lazy init) ----
        self.classifier: nn.Module | None = None

    # ------------------------------------------------------------------
    # Lazy classifier
    # ------------------------------------------------------------------

    def _build_classifier(self, n_features: int):
        self.classifier = nn.Linear(n_features, self.n_classes)
        self.classifier.to(next(self.parameters()).device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_bands, C, T)
        Returns: (B, n_classes)
        """
        B, nb, C, T = x.shape

        # 1. Merge batch and band dims → shared-weight processing
        x = x.reshape(B * nb, 1, C, T)

        # 2. EEGNet Block 1 → (B*nb, D*F1, 1, T')
        x = self.block1(x)
        x = x.squeeze(2)               # (B*nb, D*F1, T')

        # 3. TCN → (B*nb, D*F1, T')
        x = self.tcn(x)

        # 4. Per-band pooling → (B*nb, F2, 1)
        x = self.band_pool(x)          # (B*nb, F2, 1)
        x = x.squeeze(-1)              # (B*nb, F2)

        # 5. Reshape to separate bands → concat
        x = x.reshape(B, nb, -1)       # (B, nb, F2)
        x = x.reshape(B, -1)           # (B, nb * F2)

        # 6. Classifier
        if self.classifier is None:
            self._build_classifier(x.shape[-1])

        return self.classifier(x)
