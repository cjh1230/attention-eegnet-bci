"""
EEG Conformer: CNN backbone + lightweight Transformer encoder for MI-EEG.

Reference:
    Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding"
    (arXiv:2305.10807)

Architecture:
    EEGNet-style CNN → Transformer Encoder (×N) → GlobalPool → FC

Designed for small-sample, few-channel MI with aggressive dropout to
combat overfitting.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Learned positional encoding for 1D sequences."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D) with added positional encoding."""
        T = x.shape[1]
        if T > self.pe.shape[1]:
            raise ValueError(
                f"Input sequence length {T} exceeds max_len={self.pe.shape[1]}. "
                f"Increase max_len or reduce time dimension."
            )
        return x + self.pe[:, :T, :]


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int = 4, d_ff: int = 64,
                 dropout: float = 0.5) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# EEG Conformer
# ---------------------------------------------------------------------------

class EEGConformer(nn.Module):
    """CNN + Transformer encoder for motor imagery EEG.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default 8).
    n_classes : int
        Number of output classes (default 3).
    F1 : int
        Temporal filters in CNN backbone (default 8).
    D : int
        Depth multiplier (default 2).
    d_model : int
        Transformer hidden dimension (default 32).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of Transformer encoder layers (default 2).
    d_ff : int
        FFN hidden dimension (default 64).
    dropout : float
        Dropout probability (default 0.5).
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D

        cnn_out = D * F1  # channels after spatial conv

        # ── CNN backbone ─────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, cnn_out, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(cnn_out),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Project CNN channels to d_model
        self.proj = nn.Linear(cnn_out, d_model)

        # ── Transformer ──────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=256)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)

        # ── Output head ──────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)
            Single-band EEG input.

        Returns
        -------
        logits : torch.Tensor, shape (B, n_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        # CNN → (B, D*F1, 1, T')
        x = self.cnn(x)
        x = x.squeeze(2)                  # (B, D*F1, T')
        x = x.transpose(1, 2)             # (B, T', D*F1)

        # Project to d_model
        x = self.proj(x)                  # (B, T', d_model)

        # Transformer
        x = self.pos_enc(x)
        for blk in self.blocks:
            x = blk(x)                     # (B, T', d_model)
        x = self.ln_final(x)

        # Pool + classify
        x = x.transpose(1, 2)             # (B, d_model, T')
        return self.head(x)               # (B, n_classes)
