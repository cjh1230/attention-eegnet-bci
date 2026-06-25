"""
EEGNet with integrated attention for Motor Imagery classification.

Architecture:
    EEGNet Block1 (temporal + spatial conv)
    → Channel/Temporal Attention (inserted between Block1 and Block2)
    → EEGNet Block2 (separable conv)
    → Classifier

This is where attention makes the biggest difference — after spatial filtering
(where C3/Cz/C4 patterns emerge) but before the separable convolution collapses
everything into features.

Supported attention types:
    "se"         — ChannelAttention1D (original SE)
    "mhsa"       — MultiHeadChannelAttention
    "temporal"   — TemporalAttention
    "spatiotemporal" — SpatiotemporalAttention (MHSA + Temporal)
"""
import torch
import torch.nn as nn

from models.eegnet import EEGNet
from models.attention import (
    ChannelAttention1D,
    MultiHeadChannelAttention,
    TemporalAttention,
    SpatiotemporalAttention,
)


class EEGNetWithAttention(nn.Module):
    """
    EEGNet with attention inserted between Block1 and Block2.

    Parameters
    ----------
    n_channels : int
    n_classes : int
    F1, D, F2, dropout : EEGNet hyperparams
    attn_type : str
        One of "se", "mhsa", "temporal", "spatiotemporal".
    n_heads : int
        Number of attention heads (for mhsa / spatiotemporal).
    attn_dropout : float
        Dropout inside attention.
    """

    def __init__(
        self,
        n_channels: int = 16,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
        attn_type: str = "mhsa",
        n_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # ---- Block 1 (temporal → spatial) ----
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Conv2d(
            F1, D * F1, kernel_size=(n_channels, 1), groups=F1, bias=False,
        )
        self.bn_depth = nn.BatchNorm2d(D * F1)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # ---- Attention (inserted here) ----
        # After Block1: shape is (B, D*F1, 1, T') — need to reshape for attention
        self._n_channels_after_block1 = D * F1

        if attn_type == "se":
            self.attention = ChannelAttention1D(self._n_channels_after_block1)
        elif attn_type == "mhsa":
            assert self._n_channels_after_block1 % n_heads == 0, (
                f"D*F1 ({self._n_channels_after_block1}) must be divisible "
                f"by n_heads ({n_heads})"
            )
            self.attention = MultiHeadChannelAttention(
                self._n_channels_after_block1, n_heads=n_heads, dropout=attn_dropout
            )
        elif attn_type == "temporal":
            self.attention = TemporalAttention(n_times=None)  # lazy
        elif attn_type == "spatiotemporal":
            self.attention = SpatiotemporalAttention(
                self._n_channels_after_block1, n_times=None,
                n_heads=n_heads, dropout=attn_dropout,
            )
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        # ---- Block 2 (separable conv) ----
        self.separable = nn.Sequential(
            nn.Conv2d(
                D * F1, D * F1,
                kernel_size=(1, 16), padding=(0, 8), groups=D * F1, bias=False,
            ),
            nn.Conv2d(D * F1, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Classifier built lazily
        self.classifier = None

    def _build_classifier(self, n_features: int):
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, self.n_classes),
        )
        self.classifier.to(next(self.parameters()).device)

    def forward(self, x):
        # x: (B, C, T) → (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.bn_depth(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Squeeze spatial dim (1) for attention: (B, D*F1, 1, T') → (B, D*F1, T')
        x_sq = x.squeeze(2)

        # Attention
        x_sq = self.attention(x_sq)

        # Un-squeeze: (B, D*F1, T') → (B, D*F1, 1, T')
        x = x_sq.unsqueeze(2)

        # Block 2
        x = self.separable(x)

        # Lazy classifier
        if self.classifier is None:
            n_features = x.reshape(x.size(0), -1).size(1)
            self._build_classifier(n_features)

        return self.classifier(x)


def create_model(
    model_type: str = "eegnet",
    n_channels: int = 16,
    n_classes: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function for model creation.

    Parameters
    ----------
    model_type : str
        "eegnet" | "eegnet_se" | "eegnet_mhsa" | "eegnet_temporal" |
        "eegnet_spatiotemporal" | "fbcnet" | "eeg_tcnet"
    n_channels : int
    n_classes : int
    **kwargs : passed to model constructor

    Returns
    -------
    nn.Module
    """
    attn_map = {
        "eegnet_se": "se",
        "eegnet_mhsa": "mhsa",
        "eegnet_temporal": "temporal",
        "eegnet_spatiotemporal": "spatiotemporal",
    }

    if model_type == "eegnet":
        return EEGNet(n_channels=n_channels, n_classes=n_classes, **kwargs)

    if model_type in attn_map:
        return EEGNetWithAttention(
            n_channels=n_channels,
            n_classes=n_classes,
            attn_type=attn_map[model_type],
            **kwargs,
        )

    if model_type == "fbcnet":
        from models.fbcnet import FBCNet

        return FBCNet(n_channels=n_channels, n_classes=n_classes, **kwargs)

    if model_type == "eeg_tcnet":
        from models.eeg_tcnet import EEGTCNet

        return EEGTCNet(n_channels=n_channels, n_classes=n_classes, **kwargs)

    if model_type == "eeg_conformer":
        from models.eeg_conformer import EEGConformer

        return EEGConformer(n_channels=n_channels, n_classes=n_classes, **kwargs)

    raise ValueError(f"Unknown model_type: {model_type}. "
                     f"Choices: eegnet, fbcnet, eeg_tcnet, "
                     f"eeg_conformer, {', '.join(attn_map)}")
