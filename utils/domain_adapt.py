"""
Domain adaptation losses for cross-subject EEG.

- Center Loss (Wen et al. 2016): pulls features of the same class toward
  a shared class center, implicitly reducing inter-subject scatter.
- MMD Loss (Gretton et al. 2012): minimises the Maximum Mean Discrepancy
  between two feature distributions via multi-kernel RBF.

Typical usage (with a PyTorch model that exposes intermediate features)::

    from utils.domain_adapt import center_loss, multi_kernel_mmd

    # During training:
    features = model.intermediate_features(x)   # (B, D)
    logits = model.classifier(features)          # (B, n_classes)
    ce = criterion(logits, y)
    ct = center_loss(features, y, n_classes=3)
    loss = ce + 0.01 * ct
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Center Loss
# ---------------------------------------------------------------------------

def center_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
    centers: torch.Tensor | None = None,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute center loss and update class centers via EMA.

    L_center = (1/N) * sum_i || f_i - c_{y_i} ||²

    Parameters
    ----------
    features : torch.Tensor, shape (B, D)
        Feature vectors from the penultimate layer.
    labels : torch.Tensor, shape (B,)
        Integer class labels.
    n_classes : int
        Number of classes.
    centers : torch.Tensor or None, shape (n_classes, D)
        Current class centers.  If None, initialised from the first batch.
    alpha : float
        EMA update rate for centers (0 < alpha <= 1).

    Returns
    -------
    loss : torch.Tensor, scalar
    centers : torch.Tensor, shape (n_classes, D) — updated centers
    """
    device = features.device
    D = features.shape[1]

    if centers is None:
        centers = torch.zeros(n_classes, D, device=device)

    # Detach centers so gradient only flows through features
    centers_batch = centers.index_select(0, labels)
    diff = features - centers_batch.detach()
    loss = (diff ** 2).sum() / features.shape[0]

    # Update centers with EMA (no grad)
    with torch.no_grad():
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                c_update = features[mask].mean(dim=0)
                centers[c] = alpha * c_update + (1 - alpha) * centers[c]

    return loss, centers


# ---------------------------------------------------------------------------
# MMD Loss
# ---------------------------------------------------------------------------

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian RBF kernel between two sets of vectors."""
    xx = (x ** 2).sum(dim=1, keepdim=True)        # (Bx, 1)
    yy = (y ** 2).sum(dim=1, keepdim=True)        # (By, 1)
    xy = x @ y.T                                    # (Bx, By)
    dist = xx + yy.T - 2 * xy
    gamma = 1.0 / (2.0 * sigma ** 2)
    return torch.exp(-gamma * dist)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Single RBF-kernel MMD² between two feature sets.

    MMD² = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    """
    Bx, By = x.shape[0], y.shape[0]
    if Bx == 0 or By == 0:
        return torch.tensor(0.0, device=x.device)
    k_xx = _rbf_kernel(x, x, sigma)
    k_yy = _rbf_kernel(y, y, sigma)
    k_xy = _rbf_kernel(x, y, sigma)
    # Remove diagonal for unbiased estimate
    k_xx_no_diag = (k_xx.sum() - k_xx.trace()) / (Bx * (Bx - 1) + 1e-8)
    k_yy_no_diag = (k_yy.sum() - k_yy.trace()) / (By * (By - 1) + 1e-8)
    k_xy_mean = k_xy.mean()
    return k_xx_no_diag + k_yy_no_diag - 2 * k_xy_mean


def multi_kernel_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: list[float] | None = None,
) -> torch.Tensor:
    """Multi-kernel MMD² (averaged over a bank of RBF bandwidths).

    Parameters
    ----------
    x : torch.Tensor, shape (Bx, D)
    y : torch.Tensor, shape (By, D)
    sigmas : list[float] or None
        RBF bandwidths.  Default: [1, 2, 4, 8, 16].

    Returns
    -------
    mmd2 : torch.Tensor, scalar
    """
    if sigmas is None:
        sigmas = [1.0, 2.0, 4.0, 8.0, 16.0]
    vals = []
    for s in sigmas:
        vals.append(mmd_rbf(x, y, sigma=s))
    return torch.stack(vals).mean()
