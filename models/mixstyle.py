"""
MixStyle: Domain generalization via feature-statistics mixing.

Reference:
    Zhou et al., "Domain Generalization with MixStyle" (ICLR 2021)

MixStyle randomly mixes the instance-level mean and standard deviation
between two samples, simulating domain shift without requiring domain
labels.  Insert into any model after BatchNorm / activation layers.

Usage::

    from models.mixstyle import MixStyle1d, MixStyle2d

    self.mixstyle = MixStyle2d(p=0.5, alpha=0.1)
    x = self.conv(x)
    x = self.mixstyle(x)     # mixed during training, identity during eval
"""

import torch
import torch.nn as nn


class MixStyle1d(nn.Module):
    """MixStyle for 1D feature maps (B, C, T).

    Parameters
    ----------
    p : float
        Probability of applying MixStyle to a given batch (default 0.5).
    alpha : float
        Beta distribution shape parameter controlling mix strength.
        Smaller alpha → closer to original (default 0.1).
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x  # no mixing during eval

        B = x.shape[0]
        if B < 2:
            return x  # need at least 2 samples

        # Instance statistics: (B, C)
        mu = x.mean(dim=-1)
        sigma = x.std(dim=-1, unbiased=False)

        # Shuffle + mix (PyTorch RNG for reproducibility)
        perm = torch.randperm(B, device=x.device)
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0
        lam = max(lam, 1.0 - lam)

        mu_mixed = lam * mu + (1.0 - lam) * mu[perm]
        sigma_mixed = lam * sigma + (1.0 - lam) * sigma[perm]

        # Normalise + rescale
        x_norm = (x - mu.unsqueeze(-1)) / (sigma.unsqueeze(-1) + 1e-8)
        return x_norm * sigma_mixed.unsqueeze(-1) + mu_mixed.unsqueeze(-1)


class MixStyle2d(nn.Module):
    """MixStyle for 2D feature maps (B, C, H, W).

    Parameters
    ----------
    p : float
        Probability of applying MixStyle to a given batch.
    alpha : float
        Beta distribution shape parameter.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B = x.shape[0]
        if B < 2:
            return x

        # Per-instance, per-channel mean/std over spatial dims
        mu = x.mean(dim=[-2, -1])
        sigma = x.std(dim=[-2, -1], unbiased=False)

        perm = torch.randperm(B, device=x.device)
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0
        lam = max(lam, 1.0 - lam)

        mu_mixed = lam * mu + (1.0 - lam) * mu[perm]
        sigma_mixed = lam * sigma + (1.0 - lam) * sigma[perm]

        # Broadcast back: (B, C) → (B, C, H, W)
        shape = [-1, mu.shape[1]] + [1] * (x.ndim - 2)
        x_norm = (x - mu.view(shape)) / (sigma.view(shape) + 1e-8)
        return x_norm * sigma_mixed.view(shape) + mu_mixed.view(shape)
