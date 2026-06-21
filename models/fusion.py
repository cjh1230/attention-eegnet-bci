"""
Multi-band fusion model.

Architecture:
    8–13 Hz branch → temporal conv → feature vector
    13–30 Hz branch → temporal conv → feature vector
    full band branch → temporal conv → feature vector
    → concatenate → classifier
"""
import torch
import torch.nn as nn


class MultiBandFusion(nn.Module):
    """Three-branch model that fuses mu, beta, and full-band features."""

    def __init__(
        self,
        n_channels: int = 16,
        n_times: int = 750,
        n_classes: int = 3,
        hidden: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes

        # Shared backbone per band
        def band_branch():
            return nn.Sequential(
                nn.Conv1d(n_channels, hidden, kernel_size=25, padding=12),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )

        self.mu_branch = band_branch()    # 8-13 Hz
        self.beta_branch = band_branch()  # 13-30 Hz
        self.full_branch = band_branch()  # 8-30 Hz

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, mu, beta, full):
        """All three bands must be provided as separate tensors."""
        f_mu = self.mu_branch(mu)
        f_beta = self.beta_branch(beta)
        f_full = self.full_branch(full)
        fused = torch.cat([f_mu, f_beta, f_full], dim=1)
        return self.classifier(fused)
