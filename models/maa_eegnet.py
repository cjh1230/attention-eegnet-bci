"""
MAA-EEGNet: EEGNet + Motor-Area Attention (no filter bank).

A minimal model to test whether motor-area attention helps on its own.
MAA is inserted between EEGNet's temporal conv and depthwise spatial conv,
where the 8 motor-cortex channels are still preserved.

Architecture:
    (B, C, T)  →  Temporal Conv   →  (B, F1, C, T')
               →  MAA (per-filter) →  (B, F1, C, T')
               →  Depthwise Spatial Conv → (B, D*F1, 1, T')
               →  Separable Conv   →  (B, F2, 1, T'')
               →  Lazy Classifier  →  (B, n_classes)
"""
import torch
import torch.nn as nn


class MAAEEGNet(nn.Module):
    """EEGNet with Motor-Area Attention inserted after temporal conv.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default 8 for motor montage).
    n_classes : int
        Number of output classes.
    F1, D, F2, dropout : EEGNet hyperparameters.
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
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # ---- Block 1: Temporal Conv ----
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

        # ---- Classifier (lazy init) ----
        self.classifier: nn.Module | None = None

    def _build_classifier(self, n_features: int):
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, self.n_classes),
        )
        self.classifier.to(next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) or (B, 1, C, T)"""
        if x.dim() == 3:
            x = x.unsqueeze(1)                     # (B, 1, C, T)

        # 1. Temporal conv
        x = self.temporal_conv(x)                  # (B, F1, C, T')
        B, F1_dim, C_out, T_out = x.shape

        # 2. MAA: reshape (B, F1, C, T') → (B*F1, C, T')
        x = x.permute(0, 2, 1, 3)                 # (B, C, F1, T')
        x = x.reshape(B * F1_dim, C_out, T_out)   # (B*F1, C, T')
        x = self.maa(x)                            # (B*F1, C, T')
        x = x.reshape(B, C_out, F1_dim, T_out)    # (B, C, F1, T')
        x = x.permute(0, 2, 1, 3)                 # (B, F1, C, T')

        # 3. Depthwise spatial conv
        x = self.depthwise(x)                      # (B, D*F1, 1, T')
        x = self.bn_depth(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # 4. Separable conv
        x = self.separable(x)                      # (B, F2, 1, T'')

        # 5. Classifier
        if self.classifier is None:
            n_features = x.reshape(x.size(0), -1).size(1)
            self._build_classifier(n_features)

        return self.classifier(x)
