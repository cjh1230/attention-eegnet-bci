"""
Upgraded attention modules for EEG Motor Imagery.

Modules:
- ChannelAttention1D      — original SE-style (backward compat)
- MultiHeadChannelAttention  — multi-head self-attention over channels
- TemporalAttention       — learned time-point weighting per channel
- SpatiotemporalAttention — combined channel + temporal attention
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Original (backward compatible)
# ---------------------------------------------------------------------------

class ChannelAttention1D(nn.Module):
    """Squeeze-and-excitation style attention over EEG channels."""

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(n_channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(n_channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        s = self.squeeze(x).view(B, C)       # (B, C)
        w = self.excitation(s).view(B, C, 1)  # (B, C, 1)
        return x * w


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention over channels
# ---------------------------------------------------------------------------

class MultiHeadChannelAttention(nn.Module):
    """
    Multi-head self-attention over EEG channels.

    Treats channels as tokens, time as feature dim (pooled per head).
    Learns which channels attend to which — crucial for C3/Cz/C4 motor area weighting.
    """

    def __init__(self, n_channels: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert n_channels % n_heads == 0, f"n_channels ({n_channels}) must be divisible by n_heads ({n_heads})"
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.head_dim = n_channels // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(n_channels, 3 * n_channels)
        self.out_proj = nn.Linear(n_channels, n_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        # Pool time dimension → (B, C)
        x_pooled = x.mean(dim=-1)  # simple mean pool over time
        # QKV projection
        qkv = self.qkv(x_pooled)  # (B, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, C)

        # Reshape to (B, n_heads, 1, head_dim)
        q = q.view(B, self.n_heads, 1, self.head_dim)
        k = k.view(B, self.n_heads, 1, self.head_dim)
        v = v.view(B, self.n_heads, 1, self.head_dim)

        # Attention: each channel is a token, we want channel×channel attention
        # Reshape to (B, n_heads, C, head_dim) for full channel attention
        q = q.expand(-1, -1, C, -1)
        k = k.expand(-1, -1, C, -1)
        v = v.expand(-1, -1, C, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, C, C)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, n_heads, C, head_dim)
        out = out.transpose(1, 2).reshape(B, C, self.n_channels)  # (B, C, C)
        out = out.mean(dim=1)  # (B, C) — aggregate over attended channels
        out = self.out_proj(out)  # (B, C)
        out = out.sigmoid().unsqueeze(-1)  # (B, C, 1)

        return x * out


# ---------------------------------------------------------------------------
# Temporal attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Learned weighting over time points.

    Learns which time segments (early/late in trial) carry the most MI information.
    Supports lazy init: pass n_times=None to build on first forward.
    """

    def __init__(self, n_times: int = None, reduction: int = 8):
        super().__init__()
        self.excitation = None  # built lazily if n_times is None
        self.reduction = reduction
        if n_times is not None:
            self._build(n_times)

    def _build(self, n_times: int):
        bottleneck = max(n_times // self.reduction, 1)
        self.excitation = nn.Sequential(
            nn.Linear(n_times, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, n_times),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        if self.excitation is None:
            self._build(T)
            self.excitation = self.excitation.to(x.device)
        s = x.mean(dim=1)  # (B, T) — pool channels
        w = self.excitation(s).unsqueeze(1)  # (B, 1, T)
        return x * w


# ---------------------------------------------------------------------------
# Combined spatiotemporal attention
# ---------------------------------------------------------------------------

class SpatiotemporalAttention(nn.Module):
    """
    Sequential channel → temporal attention.

    First learns which channels to attend to, then which time points matter.
    """

    def __init__(self, n_channels: int, n_times: int = None,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.channel_attn = MultiHeadChannelAttention(
            n_channels, n_heads=n_heads, dropout=dropout
        )
        self.temporal_attn = None  # built lazily if n_times is known
        self._n_times = n_times
        if n_times is not None:
            self.temporal_attn = TemporalAttention(n_times)

    def forward(self, x):
        # x: (B, C, T)
        x = self.channel_attn(x)
        if self.temporal_attn is None:
            # Build lazily on first forward
            self.temporal_attn = TemporalAttention(x.shape[-1]).to(x.device)
        x = self.temporal_attn(x)
        return x
