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


def compute_covariance_scm(X: np.ndarray) -> np.ndarray:
    """Compute per-trial sample covariance matrices (SCM).

    C_i = X_i @ X_i.T / T

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
        Raw EEG trials.

    Returns
    -------
    covs : np.ndarray, shape (N, C, C), dtype float32
        Per-trial SPD covariance matrices.
    """
    N, C, T = X.shape
    covs = np.empty((N, C, C), dtype=np.float32)
    for i in range(N):
        x = X[i]  # (C, T)
        covs[i] = (x @ x.T) / T
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


def compute_covariance(X: np.ndarray, estimator: str = "scm") -> np.ndarray:
    """Compute per-trial SPD covariance matrices.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
        Raw EEG trials.
    estimator : str
        Covariance estimator: "scm" (sample) or "lwf" (Ledoit-Wolf).

    Returns
    -------
    covs : np.ndarray, shape (N, C, C), dtype float32
        Per-trial SPD covariance matrices.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array (N, C, T), got shape {X.shape}")

    if estimator == "scm":
        return compute_covariance_scm(X)
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
