"""
Motor-Area Attention (MAA) for 8-channel motor-cortex EEG.

Groups channels into three physiologically meaningful regions:
  - Premotor:     FC3, FC4          (indices 0, 4)
  - Motor:        C3, Cz, C4        (indices 1, 2, 3)
  - Sensorimotor: CP3, CPz, CP4     (indices 5, 6, 7)

Each group gets a learned attention weight computed from the full
spatiotemporal input.  Channels within a group share the same weight,
reflecting the hypothesis that motor imagery primarily activates
region-level rather than channel-level patterns.
"""
import torch
import torch.nn as nn


class MotorAreaAttention(nn.Module):
    """Motor-area attention for 8-channel motor-cortex EEG.

    Input:  (B, C, T)
    Output: (B, C, T)  — same shape, channels reweighted by group attention.

    Parameters
    ----------
    n_channels : int
        Must be 8 (motor cortex montage).
    reduction : int
        Bottleneck ratio for the group-weight MLP (default 4).
    """

    # Channel indices for each motor region.
    # Maps to MOTOR_CHANNELS = [Fc3., C3.., Cz.., C4.., Fc4., Cp3., Cpz., Cp4.]
    GROUP_INDICES = {
        "premotor":     [0, 4],          # FC3, FC4
        "motor":        [1, 2, 3],       # C3, Cz, C4
        "sensorimotor": [5, 6, 7],       # CP3, CPz, CP4
    }
    N_GROUPS = 3

    def __init__(self, n_channels: int = 8, reduction: int = 4):
        super().__init__()
        if n_channels != 8:
            raise ValueError(
                f"MotorAreaAttention expects 8 channels, got {n_channels}"
            )
        self.n_channels = n_channels

        # Binary mask: (N_GROUPS, C) — 1.0 where channel belongs to group.
        self.register_buffer("group_mask", self._build_group_mask())

        # Per-group weight MLP: pools full (B, C, T) → (B, N_GROUPS) weights.
        bottleneck = max(n_channels // reduction, 1)
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),       # (B, C, 1)
            nn.Flatten(),                   # (B, C)
            nn.Linear(n_channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, self.N_GROUPS),
            nn.Sigmoid(),                   # weights ∈ [0, 1]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_group_mask(self) -> torch.Tensor:
        """Build (N_GROUPS, C) binary mask for channel-to-group assignment."""
        mask = torch.zeros(self.N_GROUPS, self.n_channels)
        for g_idx, indices in enumerate(self.GROUP_INDICES.values()):
            for ch in indices:
                mask[g_idx, ch] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply motor-area attention.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, T)

        Returns
        -------
        torch.Tensor, shape (B, C, T) — reweighted input.
        """
        B, C, T = x.shape

        # Compute group weights from full spatiotemporal context
        group_weights = self.weight_net(x)                    # (B, N_GROUPS)

        # Expand to per-channel weights via group mask
        # group_mask: (N_GROUPS, C) → (1, N_GROUPS, C)
        # group_weights: (B, N_GROUPS) → (B, N_GROUPS, 1)
        channel_weights = (
            group_weights.unsqueeze(-1) * self.group_mask.unsqueeze(0)
        ).sum(dim=1).unsqueeze(-1)                             # (B, C, 1)

        return x * channel_weights
