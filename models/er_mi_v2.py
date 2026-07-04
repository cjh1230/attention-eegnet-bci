"""
ER-MI-v2: Evidence Reasoning Network with Explicit Multi-Token Evidence.

Architecture (v2):
    EEG → EEGNet Block1 (shared encoder)
        → Evidence Tokenizer (4 parallel branches)
            ├─ Mu token (large kernel ≈ mu rhythm)
            ├─ Beta token (small kernel ≈ beta rhythm)
            ├─ Spatial token (channel attention)
            └─ Global token (direct pool)
        → Token Interaction (TransformerEncoder)
        → Evidence Aggregation (mean)
        → GRUCell Reasoning × K
        → Step-wise Classifier
        → [logits_1, ..., logits_K] (train) / logits_K (eval)

Key difference from v1:
    v1: single evidence vector from global average pool
    v2: 4 distinct evidence tokens with cross-token attention before GRU reasoning
"""

import torch
import torch.nn as nn

from models.attention import ChannelAttention1D


class EvidenceTokenizer(nn.Module):
    """
    Extract multiple evidence tokens from the shared feature map.

    Parameters
    ----------
    in_channels : int
        Number of channels in the feature map (D * F1 from Block1).
    hidden_dim : int
        Output token dimension.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 64):
        super().__init__()

        # Mu token: large kernel for slow rhythms (~260ms @ 250Hz = kernel 65)
        self.mu_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=65, padding=32,
                      groups=in_channels, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Beta token: smaller kernel for fast rhythms (~130ms @ 250Hz = kernel 33)
        self.beta_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=33, padding=16,
                      groups=in_channels, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Spatial token: channel attention → weighted sum → project
        self.channel_attn = ChannelAttention1D(in_channels)
        self.spatial_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Global token: direct pooling (equivalent to v1's evidence)
        self.global_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_feat, T') — feature map from Block1, spatial dim squeezed.

        Returns:
            tokens: (B, 4, hidden_dim)
        """
        # Mu token: (B, C, T') → (B, hidden_dim, 1) → (B, hidden_dim)
        mu = self.mu_conv(x).squeeze(-1)

        # Beta token
        beta = self.beta_conv(x).squeeze(-1)

        # Spatial token
        x_attn = self.channel_attn(x)       # (B, C, T') with channel weights applied
        spatial = self.spatial_proj(x_attn).squeeze(-1)

        # Global token
        glob = self.global_proj(x)

        # Stack: (B, 4, hidden_dim)
        return torch.stack([mu, beta, spatial, glob], dim=1)


class ERMIv2(nn.Module):
    """
    Evidence Reasoning Network v2 — Multi-Token Evidence.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels. Default 8.
    n_classes : int
        Number of output classes. Default 2.
    F1 : int
        Temporal filters in Block1.
    D : int
        Depth multiplier for spatial filters.
    hidden_dim : int
        Dimension of evidence tokens and GRU hidden state.
    steps : int
        Number of reasoning steps (K). Default 3.
    n_tokens : int
        Number of evidence tokens (fixed at 4).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        hidden_dim: int = 64,
        steps: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.hidden_dim = hidden_dim
        self.steps = steps

        feat_channels = D * F1  # = 16 with defaults

        # ── Shared Encoder: EEGNet Block1 ──────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Conv2d(
            F1, feat_channels, kernel_size=(n_channels, 1), groups=F1, bias=False,
        )
        self.bn_depth = nn.BatchNorm2d(feat_channels)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # ── Evidence Tokenizer ─────────────────────────────────────
        self.tokenizer = EvidenceTokenizer(feat_channels, hidden_dim)

        # ── Token Interaction: 1-layer Transformer ─────────────────
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # Positional encoding (learned) for tokens
        self.token_pos = nn.Parameter(torch.zeros(1, 4, hidden_dim))

        # ── Evidence Aggregation ───────────────────────────────────
        self.evidence_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # ── Reasoning cell ─────────────────────────────────────────
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # ── Step classifier ────────────────────────────────────────
        self.step_classifier = nn.Linear(hidden_dim, n_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_all_steps: bool = False):
        """
        Args:
            x: (B, C, T) or (B, 1, C, T).
            return_all_steps: If True, always return list of all step logits.

        Returns:
            training mode or return_all_steps: list[Tensor], each (B, n_classes).
            eval mode (default): Tensor (B, n_classes).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # ── Shared Encoder ─────────────────────────────────────────
        x = self.conv1(x)                          # (B, F1, C, T)
        x = self.depthwise(x)                      # (B, D*F1, 1, T)
        x = self.bn_depth(x)
        x = self.act1(x)
        x = self.pool1(x)                          # (B, D*F1, 1, T//4)
        x = self.drop1(x)
        x = x.squeeze(2)                           # (B, D*F1, T//4)

        # ── Evidence Tokenizer ─────────────────────────────────────
        tokens = self.tokenizer(x)                 # (B, 4, hidden_dim)

        # ── Token Interaction ──────────────────────────────────────
        tokens = self.token_norm(tokens + self.token_pos)
        tokens = self.token_transformer(tokens)    # (B, 4, hidden_dim)

        # ── Evidence Aggregation ───────────────────────────────────
        evidence = tokens.mean(dim=1)               # (B, hidden_dim)
        evidence = self.evidence_proj(evidence)

        # ── Multi-step Reasoning ───────────────────────────────────
        logits_list = []
        h = evidence

        for _ in range(self.steps):
            h = self.gru_cell(evidence, h)
            logits_t = self.step_classifier(h)
            logits_list.append(logits_t)

        if self.training or return_all_steps:
            return logits_list
        else:
            return logits_list[-1]
