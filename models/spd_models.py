"""
SPDNet: A Riemannian Neural Network for SPD Matrix Learning.

Reference:
    Huang & Van Gool, "A Riemannian Network for SPD Matrix Learning"
    Proceedings of the AAAI Conference on Artificial Intelligence, 2017.

The core layers operate on SPD (Symmetric Positive Definite) matrices:

    BiMap  – bilinear mapping (W @ C @ W.T), reduces SPD dimension
    ReEig  – eigenvalue rectification (nonlinear activation on SPD manifold)
    LogEig – matrix logarithm (maps SPD manifold → tangent / Euclidean space)

After LogEig, the upper-triangular elements are flattened and fed to a
standard linear classifier.

Usage::

    from models.spd_models import SPDNetModel

    model = SPDNetModel(n_classes=2, bimap_dims=[8, 6, 4])
    # Input: (B, 8, 8) SPD covariance matrices
    # Output: (B, 2) logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core SPD manifold layers
# ---------------------------------------------------------------------------


class BiMap(nn.Module):
    """Bilinear mapping layer for SPD matrices.

    Transforms an SPD matrix C (d_in × d_in) to a lower-dimensional
    SPD matrix (d_out × d_out) via:

        C_out = W @ C_in @ W.T

    where W is a learnable (d_out × d_in) weight matrix.

    In the original SPDNet, W is constrained to be semi-orthogonal
    (on the compact Stiefel manifold).  In practice, an unconstrained
    linear transform works well and is much simpler to optimise.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        # Weight shape: (d_out, d_in) — each row maps d_in → 1
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self._init_weight()

    def _init_weight(self):
        # Initialise close to a semi-orthogonal sub-matrix (random rotation
        # rows) so early forward passes stay numerically stable.
        nn.init.orthogonal_(self.weight)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        C : (B, d_in, d_in)  SPD matrices.

        Returns
        -------
        C_out : (B, d_out, d_out)  SPD matrices.
        """
        # C_out = W @ C @ W.T
        WC = torch.matmul(self.weight, C)  # (B, d_out, d_in)
        return torch.matmul(WC, self.weight.T)  # (B, d_out, d_out)


class ReEig(nn.Module):
    """Eigenvalue rectification layer.

    Nonlinear activation on the SPD manifold:

        C_out = U @ max(eps * I, Sigma) @ U.T

    where C = U @ Sigma @ U.T is the eigendecomposition.

    This is the SPD analogue of ReLU — it ensures eigenvalues stay
    bounded away from zero, preserving positive definiteness.
    """

    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        C : (B, d, d)  SPD matrices.

        Returns
        -------
        C_out : (B, d, d)  SPD matrices with rectified eigenvalues.
        """
        # Use float64 for eigendecomposition for numerical stability
        dtype_in = C.dtype
        C64 = C.to(torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(C64)
        eigvals = torch.clamp(eigvals, min=self.eps)
        eigvals = eigvals.to(dtype_in)
        eigvecs = eigvecs.to(dtype_in)
        # Reconstruct: C_out = U @ diag(clamped_eigvals) @ U.T
        C_out = torch.matmul(
            eigvecs,
            torch.matmul(torch.diag_embed(eigvals), eigvecs.transpose(-1, -2)),
        )
        return C_out


class LogEig(nn.Module):
    """Matrix logarithm layer.

    Maps SPD matrices from the Riemannian manifold to the tangent
    (Euclidean) space at the identity:

        C_log = U @ log(Sigma) @ U.T

    After this layer, standard Euclidean operations (flatten, linear,
    softmax) can be applied.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        C : (B, d, d)  SPD matrices.

        Returns
        -------
        C_log : (B, d, d)  Matrix logarithms (symmetric, in tangent space).
        """
        # Use float64 for eigendecomposition for numerical stability
        dtype_in = C.dtype
        C64 = C.to(torch.float64)
        eigvals, eigvecs = torch.linalg.eigh(C64)
        eigvals = torch.clamp(eigvals, min=self.eps)
        eigvals = eigvals.to(dtype_in)
        eigvecs = eigvecs.to(dtype_in)
        log_eigvals = torch.log(eigvals)
        C_log = torch.matmul(
            eigvecs,
            torch.matmul(torch.diag_embed(log_eigvals), eigvecs.transpose(-1, -2)),
        )
        return C_log


# ---------------------------------------------------------------------------
# SPDNet model
# ---------------------------------------------------------------------------


class SPDNetModel(nn.Module):
    """SPDNet for MI-EEG covariance classification.

    Architecture::

        Input: (B, C, C)  SPD covariance matrices (C=8 for DeepBCI)
          BiMap(C → d1)     → (B, d1, d1)  bilinear dimension reduction
          ReEig              → (B, d1, d1)  eigenvalue rectification
          BiMap(d1 → d2)    → (B, d2, d2)
          ReEig              → (B, d2, d2)
          LogEig             → (B, d2, d2)  map to tangent space
          upper_tri + flat   → (B, d2*(d2+1)/2)
          Linear             → (B, n_classes)

    Total parameters: ~2K for default dims [8, 6, 4] — extremely lightweight.

    Parameters
    ----------
    n_classes : int
        Number of output classes (2 for binary, 4 for BCI IV 2a).
    bimap_dims : list[int]
        Dimensions for successive BiMap layers.
        Default [8, 6, 4] for 8-channel input.
    dropout : float
        Dropout rate applied before the final linear layer.
    """

    def __init__(
        self,
        n_classes: int = 2,
        bimap_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if bimap_dims is None:
            bimap_dims = [8, 6, 4]

        self.bimap_dims = bimap_dims
        self.n_classes = n_classes

        # Build BiMap → ReEig blocks
        layers: list[nn.Module] = []
        for i in range(len(bimap_dims) - 1):
            layers.append(BiMap(bimap_dims[i], bimap_dims[i + 1]))
            layers.append(ReEig())
        self.spd_blocks = nn.Sequential(*layers)

        # LogEig → tangent space
        self.log_eig = LogEig()

        # After LogEig we have a (B, d_last, d_last) symmetric matrix.
        # Use upper-triangular elements (including diagonal) as features.
        d_last = bimap_dims[-1]
        self.feat_dim = d_last * (d_last + 1) // 2  # upper-tri count

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feat_dim, n_classes)

        # Cache upper-tri indices
        self._triu_idx: torch.Tensor | None = None

    def _get_triu_idx(self, d: int, device: torch.device) -> torch.Tensor:
        """Return (row, col) indices for the upper triangle (incl. diagonal)."""
        if self._triu_idx is None or self._triu_idx.device != device:
            rows, cols = torch.triu_indices(d, d, device=device)
            self._triu_idx = torch.stack([rows, cols])
        return self._triu_idx

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        C : (B, d_in, d_in)  SPD covariance matrices.

        Returns
        -------
        logits : (B, n_classes)  Class logits.
        """
        # SPD manifold layers
        out = self.spd_blocks(C)  # (B, d_last, d_last)

        # Map to tangent space
        out = self.log_eig(out)  # (B, d_last, d_last)

        # Extract upper-triangular elements
        d_last = out.shape[-1]
        idx = self._get_triu_idx(d_last, out.device)
        feats = out[:, idx[0], idx[1]]  # (B, feat_dim)

        feats = self.dropout(feats)
        return self.classifier(feats)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def create_spdnet(
    n_channels: int = 8,
    n_classes: int = 2,
    bimap_dims: list[int] | None = None,
    dropout: float = 0.3,
) -> SPDNetModel:
    """Create an SPDNet model for MI-EEG classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (determines input covariance size).
    n_classes : int
        Number of output classes.
    bimap_dims : list[int] | None
        BiMap dimensions.  If None, defaults to [n_channels, n_channels-2,
        max(n_channels-4, 2)].
    dropout : float
        Dropout rate.

    Returns
    -------
    SPDNetModel instance.
    """
    if bimap_dims is None:
        d1 = max(n_channels - 2, 4)
        d2 = max(n_channels - 4, 2)
        bimap_dims = [n_channels, d1, d2]

    return SPDNetModel(
        n_classes=n_classes,
        bimap_dims=bimap_dims,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# SPD Decoder for masked reconstruction
# ---------------------------------------------------------------------------


class SPDDecoder(nn.Module):
    """Decoder that reconstructs log-covariance from LogEig features.

    Takes upper-triangular elements from LogEig output of a masked
    covariance matrix and predicts the upper-triangular elements of
    the original (unmasked) log-covariance matrix.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct log-covariance upper-tri from encoded features.

        Parameters
        ----------
        z : (B, feat_dim)  LogEig features from masked input.

        Returns
        -------
        recon : (B, feat_dim)  Reconstructed log-cov upper-tri.
        """
        return self.net(z)


# ---------------------------------------------------------------------------
# Multi-band SPDNet
# ---------------------------------------------------------------------------


class MultiBandSPDNet(nn.Module):
    """Multi-band SPDNet for MI-EEG with separate per-band SPDNet branches.

    Each frequency band gets its own SPDNet branch (shared or separate weights).
    Features from all bands are concatenated after LogEig and fed to a single
    classifier.

    Parameters
    ----------
    n_bands : int  Number of frequency bands.
    n_channels : int  Number of EEG channels.
    n_classes : int  Number of output classes.
    bimap_dims : list[int]  BiMap dims for each branch.
    dropout : float
    share_branches : bool  If True, all bands share the same SPDNet weights.
    """

    def __init__(
        self,
        n_bands: int = 2,
        n_channels: int = 8,
        n_classes: int = 2,
        bimap_dims: list[int] | None = None,
        dropout: float = 0.3,
        share_branches: bool = True,
    ):
        super().__init__()

        if bimap_dims is None:
            bimap_dims = [n_channels, n_channels]

        self.n_bands = n_bands
        self.share_branches = share_branches
        d_last = bimap_dims[-1]

        if share_branches:
            # Single branch applied to each band
            self.branch = SPDNetModel(
                n_classes=n_classes,
                bimap_dims=bimap_dims,
                dropout=0.0,  # dropout applied after concat
            )
            self.branches = None
        else:
            # Separate branch per band
            self.branch = None
            self.branches = nn.ModuleList([
                SPDNetModel(
                    n_classes=n_classes,
                    bimap_dims=bimap_dims,
                    dropout=0.0,
                )
                for _ in range(n_bands)
            ])

        # Feature dim: n_bands × d_last × (d_last+1) / 2
        feat_dim = n_bands * d_last * (d_last + 1) // 2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(
        self, C_bands: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        C_bands : dict[str, (B, C, C)]  Per-band SPD covariance matrices.

        Returns
        -------
        logits : (B, n_classes)
        """
        features = []
        for i, (band_name, C) in enumerate(C_bands.items()):
            if self.share_branches:
                # Use shared branch, but only pass through spd_blocks + log_eig
                out = self.branch.spd_blocks(C)
                out = self.branch.log_eig(out)
            else:
                out = self.branches[i].spd_blocks(C)
                out = self.branches[i].log_eig(out)

            d = out.shape[-1]
            idx = self.branch._get_triu_idx(d, out.device) if self.share_branches else \
                  self.branches[i]._get_triu_idx(d, out.device)
            feats = out[:, idx[0], idx[1]]
            features.append(feats)

        all_feats = torch.cat(features, dim=-1)
        all_feats = self.dropout(all_feats)
        return self.classifier(all_feats)
