"""
EEGNet implementation (Lawhern et al. 2018).

Reference: https://doi.org/10.1088/1741-2552/aace8c

Input shape: (batch, 1, C, T)  — single channel dim for Conv2d
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet for Motor Imagery classification.
    Handles arbitrary time lengths (lazy linear init).

    Reference: Lawhern et al. 2018 (doi:10.1088/1741-2552/aace8c)

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_classes : int
        Number of output classes.
    F1 : int
        Number of temporal filters.
    D : int
        Depth multiplier for spatial filters.
    F2 : int
        Number of pointwise filters.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 16,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: Temporal → Spatial
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Conv2d(
            F1,
            D * F1,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn_depth = nn.BatchNorm2d(D * F1)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable conv
        self.separable = nn.Sequential(
            nn.Conv2d(
                D * F1,
                D * F1,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=D * F1,
                bias=False,
            ),
            nn.Conv2d(D * F1, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Classifier built lazily on first forward pass (to handle variable T)
        self.classifier = None

    def _build_classifier(self, n_features: int):
        """Create the linear classifier once we know the flattened dim."""
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, self.n_classes),
        )
        self.classifier.to(next(self.parameters()).device)

    def forward(self, x):
        # x: (B, C, T) → (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.bn_depth(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable(x)

        # Lazy init
        if self.classifier is None:
            n_features = x.reshape(x.size(0), -1).size(1)
            self._build_classifier(n_features)

        return self.classifier(x)
