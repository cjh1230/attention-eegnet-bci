"""
Per-trial SPD covariance matrix computation from raw EEG trials.

Provides utilities to compute sample covariance matrices (SCM) or
Ledoit-Wolf shrinkage estimates for use as input to SPD manifold
deep learning models (SPDNet, etc.).

All functions return (N, C, C) float32 arrays of symmetric positive
definite (SPD) matrices.
"""

import numpy as np

try:
    from pyriemann.estimation import Covariances as PyRiemannCov

    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False


def compute_covariance_scm(X: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """Compute per-trial sample covariance matrices (SCM).

    C_i = X_i @ X_i.T / T + reg * trace / C * I

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
        Raw EEG trials.
    reg : float
        Regularisation strength (default 1e-4).  Added as
        ``reg * trace(C)/C * I`` to each matrix before return.

    Returns
    -------
    covs : np.ndarray, shape (N, C, C), dtype float32
        Per-trial regularised SPD covariance matrices.
    """
    N, C, T = X.shape
    covs = np.empty((N, C, C), dtype=np.float32)
    for i in range(N):
        x = X[i]  # (C, T)
        C_i = (x @ x.T) / T
        # Regularisation proportional to trace for numerical stability
        if reg > 0:
            tr = np.trace(C_i)
            ridge = reg * tr / C if tr > 0 else reg / C
            C_i = C_i + ridge * np.eye(C, dtype=C_i.dtype)
        covs[i] = C_i
    return covs


def compute_covariance_lwf(X: np.ndarray) -> np.ndarray:
    """Compute per-trial Ledoit-Wolf shrinkage covariance estimates.

    Requires pyriemann >= 0.5.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
        Raw EEG trials.

    Returns
    -------
    covs : np.ndarray, shape (N, C, C), dtype float32
        Per-trial SPD covariance matrices (shrunk).
    """
    if not HAS_PYRIEMANN:
        raise ImportError(
            "pyriemann is required for Ledoit-Wolf covariance estimation. "
            "Install it with: pip install pyriemann"
        )
    # pyriemann expects (N, C, T) input
    cov_estimator = PyRiemannCov(estimator="lwf")
    return cov_estimator.fit_transform(X).astype(np.float32)


def compute_covariance(
    X: np.ndarray,
    estimator: str = "scm",
    reg: float = 1e-4,
) -> np.ndarray:
    """Compute per-trial SPD covariance matrices.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
        Raw EEG trials.
    estimator : str
        Covariance estimator: "scm" (sample) or "lwf" (Ledoit-Wolf).
    reg : float
        Regularisation strength for SCM estimator (default 1e-4).

    Returns
    -------
    covs : np.ndarray, shape (N, C, C), dtype float32
        Per-trial SPD covariance matrices.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N, C, T), got shape {X.shape}")

    if estimator == "scm":
        return compute_covariance_scm(X, reg=reg)
    elif estimator == "lwf":
        return compute_covariance_lwf(X)
    else:
        raise ValueError(
            f"Unknown estimator '{estimator}'. Choose 'scm' or 'lwf'."
        )


def is_spd(C: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Check whether matrices are symmetric positive definite.

    Parameters
    ----------
    C : np.ndarray, shape (..., C, C)
        Covariance matrices.
    tol : float
        Tolerance for eigenvalue positivity check.

    Returns
    -------
    ok : np.ndarray, shape (...,), dtype bool
        True for each matrix that is SPD.
    """
    # Symmetry check (within tolerance)
    sym_err = np.abs(C - C.swapaxes(-1, -2)).max(axis=(-1, -2))
    symmetric = sym_err < 1e-5

    # Positive definiteness check
    eigvals = np.linalg.eigvalsh(C)
    positive = np.all(eigvals > tol, axis=-1)

    return symmetric & positive


# ---------------------------------------------------------------------------
# Multi-band covariance
# ---------------------------------------------------------------------------


def _bandpass_filter(
    X: np.ndarray,
    low: float,
    high: float,
    fs: float = 250.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    X : (N, C, T)  Raw EEG trials.
    low, high : float  Cutoff frequencies (Hz).
    fs : float  Sampling rate.
    order : int  Filter order.

    Returns
    -------
    X_filt : (N, C, T)  Filtered EEG.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    # Use second-order sections for numerical stability
    sos = butter(order, [low_n, high_n], btype="band", output="sos")
    # Apply along the time axis
    X_filt = sosfiltfilt(sos, X, axis=-1)
    return X_filt.astype(np.float32)


# Standard sub-bands within the motor-imagery 8–30 Hz range
MI_SUB_BANDS: list[tuple[str, float, float]] = [
    ("mu", 8.0, 13.0),
    ("beta", 13.0, 30.0),
]


def compute_multiband_covariance(
    X: np.ndarray,
    estimator: str = "scm",
    reg: float = 1e-4,
    bands: list[tuple[str, float, float]] | None = None,
    fs: float = 250.0,
) -> dict[str, np.ndarray]:
    """Compute per-band SPD covariance matrices.

    Parameters
    ----------
    X : (N, C, T)  Raw EEG trials (should already be 8–30 Hz bandpassed).
    estimator : str  "scm" or "lwf".
    reg : float  Regularisation for SCM.
    bands : list of (name, low_hz, high_hz) | None.
    fs : float  Sampling rate.

    Returns
    -------
    covs : dict mapping band name → (N, C, C) SPD covariance matrix.
    """
    if bands is None:
        bands = MI_SUB_BANDS

    result = {}
    for name, low, high in bands:
        X_filt = _bandpass_filter(X, low, high, fs=fs)
        result[name] = compute_covariance(X_filt, estimator=estimator, reg=reg)

    return result


# ---------------------------------------------------------------------------
# SPD-aware data augmentation (for SSL pre-training)
# ---------------------------------------------------------------------------


def augment_covariance_channel_dropout(
    C: np.ndarray, n_drop: int = 1, fill_value: float = 0.0
) -> np.ndarray:
    """Randomly drop channels from SPD covariance matrices.

    Drops n_drop random channels, then pads back to original size
    with identity-like fill on the diagonal.

    Parameters
    ----------
    C : (..., C, C)  SPD covariance matrices.
    n_drop : int  Number of channels to drop.
    fill_value : float  Value for padded diagonal elements.

    Returns
    -------
    C_aug : (..., C, C)  Augmented SPD matrices.
    """
    *batch, C_in, _ = C.shape
    keep = C_in - n_drop
    kept_idx = np.sort(
        np.random.choice(C_in, size=keep, replace=False)
    )

    # Extract sub-matrix for kept channels
    C_sub = C[..., kept_idx[:, None], kept_idx]

    # Pad back to original size
    C_aug = np.zeros_like(C)
    for i, ki in enumerate(kept_idx):
        for j, kj in enumerate(kept_idx):
            C_aug[..., ki, kj] = C_sub[..., i, j]

    # Fill diagonal of dropped channels
    dropped = list(set(range(C_in)) - set(kept_idx.tolist()))
    for d in dropped:
        # Use the mean of kept diagonal as fill
        diag_fill = np.mean(np.diagonal(C_sub, axis1=-2, axis2=-1), axis=-1)
        C_aug[..., d, d] = diag_fill + 1e-6

    return C_aug


def augment_covariance_perturb(
    C: np.ndarray, scale: float = 0.05
) -> np.ndarray:
    """Add small symmetric perturbation, preserving SPD property.

    C_aug = C + eps  where eps is symmetric and C_aug stays SPD.

    Parameters
    ----------
    C : (..., C, C)  SPD covariance matrices.
    scale : float  Perturbation scale relative to trace.

    Returns
    -------
    C_aug : (..., C, C)  Augmented SPD matrices.
    """
    *batch, C_dim, _ = C.shape
    # Symmetric noise scaled by trace
    eps = np.random.randn(*batch, C_dim, C_dim).astype(np.float32)
    eps = (eps + eps.swapaxes(-1, -2)) / 2  # symmetrize

    # Scale by trace for each matrix
    traces = np.trace(C, axis1=-2, axis2=-1)
    eps = eps * (scale * traces[..., None, None] / C_dim)

    C_aug = C + eps

    # Ensure SPD: clamp eigenvalues
    eigvals, eigvecs = np.linalg.eigh(C_aug)
    eigvals = np.maximum(eigvals, 1e-10)
    # Reconstruct: U @ diag(eigvals) @ U.T via einsum
    C_aug = np.einsum("...ij,...j,...kj->...ik", eigvecs, eigvals, eigvecs)

    return C_aug.astype(np.float32)


def augment_temporal_crop(
    X: np.ndarray, crop_ratio: float = 0.8
) -> np.ndarray:
    """Randomly crop temporal window from raw EEG trials.

    Parameters
    ----------
    X : (N, C, T)  Raw EEG trials.
    crop_ratio : float  Fraction of time window to keep.

    Returns
    -------
    X_crop : (N, C, T_crop)  Cropped EEG.
    """
    N, C, T = X.shape
    new_T = int(T * crop_ratio)
    starts = np.random.randint(0, T - new_T + 1, size=N)
    X_crop = np.empty((N, C, new_T), dtype=X.dtype)
    for i in range(N):
        X_crop[i] = X[i, :, starts[i]:starts[i] + new_T]
    return X_crop


def paired_spd_augment(
    C: np.ndarray,
    aug_types: tuple[str, str] = ("channel_dropout", "perturb"),
    n_drop: int = 1,
    perturb_scale: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two augmented views of the same SPD matrices.

    Used for contrastive learning: same trial → two augmentations → positive pair.

    Parameters
    ----------
    C : (N, C, C)  Original SPD covariance matrices.
    aug_types : (str, str)  Two augmentation types.
    n_drop : int  Channels to drop for channel_dropout.
    perturb_scale : float  Noise scale for perturb.

    Returns
    -------
    (C1, C2) : (N, C, C), (N, C, C)  Two augmented views.
    """
    C1 = _apply_spd_aug(C, aug_types[0], n_drop, perturb_scale)
    C2 = _apply_spd_aug(C, aug_types[1], n_drop, perturb_scale)
    return C1, C2


def mask_covariance_channels(
    C: np.ndarray, n_mask: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly mask channels in SPD covariance matrices.

    Masked channels have their rows/columns zeroed and diagonal set to 1.
    The resulting matrix remains SPD (identity on masked subspace).

    Parameters
    ----------
    C : (N, C_in, C_in)  SPD covariance matrices.
    n_mask : int  Number of channels to mask.

    Returns
    -------
    C_masked : (N, C_in, C_in)  Masked SPD matrices.
    mask : (N, C_in)  Boolean mask (True = masked channel).
    """
    N, C_in, _ = C.shape
    C_masked = C.copy()
    mask = np.zeros((N, C_in), dtype=bool)

    for i in range(N):
        masked_ch = np.random.choice(C_in, size=n_mask, replace=False)
        mask[i, masked_ch] = True
        # Zero out rows and columns
        C_masked[i, masked_ch, :] = 0.0
        C_masked[i, :, masked_ch] = 0.0
        # Set diagonal to 1 for masked channels (identity-like)
        C_masked[i, masked_ch, masked_ch] = 1.0

    return C_masked, mask


def _apply_spd_aug(
    C: np.ndarray, aug_type: str, n_drop: int, perturb_scale: float
) -> np.ndarray:
    if aug_type == "channel_dropout":
        return augment_covariance_channel_dropout(C, n_drop=n_drop)
    elif aug_type == "perturb":
        return augment_covariance_perturb(C, scale=perturb_scale)
    elif aug_type == "none":
        return C.copy()
    else:
        raise ValueError(f"Unknown augmentation: {aug_type}")
