"""
FB-MAA-EEGNet: Filter-Bank Motor-Area Attention EEGNet for MI-BCI.

Architecture:
    Multi-band EEG (B, n_bands, C, T)
    → merge batch+band → (B*n_bands, 1, C, T)
    → Temporal Conv (EEGNet Block1 conv1)   → (B*nb, F1, C, T')
    → Motor-Area Attention (per-filter)      → (B*nb, F1, C, T')
    → Depthwise Spatial Conv                 → (B*nb, D*F1, 1, T')
    → Pool + Dropout
    → Separable Conv                         → (B*nb, F2, 1, T'')
    → Reshape per-band, concat               → (B, n_bands * F2 * T'')
    → Lazy Classifier                        → (B, n_classes)

The model expects input of shape (B, n_bands, C, T).  Use
``apply_filter_bank()`` from ``models.fbcnet`` to convert standard
(B, C, T) trials into multi-band format, or set the
``input_requires_filter_bank`` flag so the training script does it
automatically.

References:
    - EEGNet:  Lawhern et al. 2018 (doi:10.1088/1741-2552/aace8c)
    - FBCNet:  Bakshi et al. 2021 (arXiv:2104.01233)
    - MAA:     motor-area attention (this project)
"""
import torch
import torch.nn as nn


class FBMAAEEGNet(nn.Module):
    """Filter-Bank Motor-Area Attention EEGNet.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands (default 6 for 8–30 Hz sub-bands).
    n_channels : int
        Number of EEG channels (default 8 for motor montage).
    n_classes : int
        Number of output classes.
    F1 : int
        Temporal filters (default 8).
    D : int
        Depth multiplier for spatial filters (default 2).
    F2 : int
        Pointwise filters in separable conv (default 16).
    dropout : float
        Dropout rate (default 0.5).
    """

    # Signal to training scripts: convert (B,C,T) → (B,n_bands,C,T)
    input_requires_filter_bank: bool = True

    def __init__(
        self,
        n_bands: int = 6,
        n_channels: int = 8,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # ---- Block 1: Temporal Conv (EEGNet-style) ----
        # (B*nb, 1, C, T) → (B*nb, F1, C, T')
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )

        # ---- Motor-Area Attention ----
        from models.motor_area_attention import MotorAreaAttention

        self.maa = MotorAreaAttention(n_channels=n_channels)

        # ---- Block 1 continued: Depthwise Spatial Conv ----
        self.depthwise = nn.Conv2d(
            F1, D * F1, kernel_size=(n_channels, 1),
            groups=F1, bias=False,
        )
        self.bn_depth = nn.BatchNorm2d(D * F1)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # ---- Block 2: Separable Conv ----
        self.separable = nn.Sequential(
            nn.Conv2d(
                D * F1, D * F1,
                kernel_size=(1, 16), padding=(0, 8),
                groups=D * F1, bias=False,
            ),
            nn.Conv2d(D * F1, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # ---- Classifier (lazy init, like EEGNet) ----
        self.classifier: nn.Module | None = None

    # ------------------------------------------------------------------
    # Lazy classifier
    # ------------------------------------------------------------------

    def _build_classifier(self, n_features: int):
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, self.n_classes),
        )
        self.classifier.to(next(self.parameters()).device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, n_bands, C, T)

        Returns
        -------
        torch.Tensor, shape (B, n_classes)
        """
        B, nb, C, T = x.shape

        # 1. Merge batch and band dims for shared-weight processing
        x = x.reshape(B * nb, 1, C, T)          # (B*nb, 1, C, T)

        # 2. Temporal conv (preserves C dim)
        x = self.temporal_conv(x)                 # (B*nb, F1, C, T')

        # 3. Motor-Area Attention
        #    MAA expects (B, C, T).  We have (B*nb, F1, C, T').
        #    Treat each (batch_item, filter) as independent: (B*nb*F1, C, T')
        B_nb, F1_dim, C_out, T_out = x.shape
        x = x.permute(0, 2, 1, 3)                     # (B*nb, C, F1, T')
        x = x.reshape(B_nb * F1_dim, C_out, T_out)    # (B*nb*F1, C, T')
        x = self.maa(x)                                # (B*nb*F1, C, T')
        x = x.reshape(B_nb, C_out, F1_dim, T_out)     # (B*nb, C, F1, T')
        x = x.permute(0, 2, 1, 3)                # (B*nb, F1, C, T')

        # 4. Depthwise spatial conv (collapses C → 1)
        x = self.depthwise(x)                     # (B*nb, D*F1, 1, T')
        x = self.bn_depth(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # 5. Separable conv
        x = self.separable(x)                     # (B*nb, F2, 1, T'')

        # 6. Flatten per band and concatenate
        x = x.reshape(B, nb, -1)                  # (B, nb, F2 * T'')
        x = x.reshape(B, -1)                      # (B, nb * F2 * T'')

        # 7. Classifier
        if self.classifier is None:
            self._build_classifier(x.shape[-1])

        return self.classifier(x)
