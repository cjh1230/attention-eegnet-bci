"""
Euclidean Alignment (EA) for cross-subject EEG covariance alignment.

Reference:
    He & Wu, "Transfer Learning for Brain-Computer Interfaces:
    A Euclidean Space Data Alignment Approach" (arXiv:1808.05464)

EA reduces the distribution shift across subjects by aligning the covariance
structure of EEG trials in Euclidean space.  It is unsupervised (requires no
labels) and computationally cheap — a single (C×C) matrix square-root inverse
per fold.

Typical usage (LOSO)::

    from preprocessing.alignment import EuclideanAlignment

    ea = EuclideanAlignment()
    ea.fit([train_subj_1_X, train_subj_2_X, ...])   # R_bar from train subjects
    for s in train_subjs:
        s["X"] = ea.transform(s["X"])
    test_subj["X"] = ea.transform(test_subj["X"])
"""

import numpy as np


def _matrix_sqrt_inv(R: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Compute R^(-1/2) via symmetric eigendecomposition with regularisation.

    Parameters
    ----------
    R : np.ndarray, shape (C, C)
        Symmetric positive semi-definite covariance matrix.
    reg : float
        Regularisation added as ``reg * trace(R)/C * I`` before inversion.

    Returns
    -------
    R_inv_sqrt : np.ndarray, shape (C, C)
        Matrix square-root inverse of the regularised R.
    """
    C = R.shape[0]
    trace_R = np.trace(R)
    if trace_R > 0:
        ridge = reg * (trace_R / C)
    else:
        ridge = reg
    R_reg = R + ridge * np.eye(C, dtype=R.dtype)

    eigvals, eigvecs = np.linalg.eigh(R_reg)

    # Clamp tiny / negative eigenvalues for numerical safety
    eigvals = np.maximum(eigvals, 1e-15)
    eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)

    return eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T


class EuclideanAlignment:
    """Euclidean Alignment for cross-subject EEG.

    Computes a reference covariance matrix ``R_bar`` from a list of subject
    arrays and aligns each trial via ``X_aligned = R_bar^(-1/2) @ X``.

    Parameters
    ----------
    reg : float
        Regularisation strength (default 1e-6).
    """

    def __init__(self, reg: float = 1e-6) -> None:
        self.reg = reg
        self.R_ref: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_list: list[np.ndarray]) -> "EuclideanAlignment":
        """Compute the reference covariance matrix from a list of subject arrays.

        Parameters
        ----------
        X_list : list of np.ndarray
            Each element is one subject's data with shape ``(N_i, C, T)``.
            All subjects must share the same channel count ``C``.

        Returns
        -------
        self : EuclideanAlignment
        """
        if not X_list:
            raise ValueError("X_list must contain at least one subject array.")

        C = X_list[0].shape[1]
        dtype = X_list[0].dtype

        # Accumulate per-trial covariance (equal weight per trial)
        cov_sum = np.zeros((C, C), dtype=np.float64)
        n_total = 0

        for X in X_list:
            if X.ndim != 3:
                raise ValueError(
                    f"Expected 3D array (N, C, T), got shape {X.shape}"
                )
            if X.shape[1] != C:
                raise ValueError(
                    f"Channel mismatch: expected {C}, got {X.shape[1]}"
                )
            # (N, C, T) @ (N, T, C) → manual loop for memory efficiency
            for i in range(X.shape[0]):
                trial = X[i]  # (C, T)
                cov_sum += trial @ trial.T  # (C, C)
                n_total += 1

        if n_total == 0:
            raise ValueError("No trials found in X_list.")

        self.R_ref = (cov_sum / n_total).astype(dtype)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Align trials using the stored reference covariance.

        Parameters
        ----------
        X : np.ndarray, shape (N, C, T)
            EEG trials to align.

        Returns
        -------
        X_aligned : np.ndarray, shape (N, C, T)
            Aligned trials (new array — input is NOT modified).
        """
        if not self._fitted or self.R_ref is None:
            raise RuntimeError("Must call fit() before transform().")

        R_inv_sqrt = _matrix_sqrt_inv(self.R_ref, reg=self.reg)

        X_aligned = np.empty_like(X)
        for i in range(X.shape[0]):
            X_aligned[i] = R_inv_sqrt @ X[i]
        return X_aligned

    def fit_transform(self, X_list: list[np.ndarray]) -> list[np.ndarray]:
        """Fit on *X_list* and return the aligned copies.

        Equivalent to ``ea.fit(X_list); [ea.transform(X) for X in X_list]``.
        """
        self.fit(X_list)
        return [self.transform(X) for X in X_list]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        """Whether ``fit()`` has been called."""
        return self._fitted

    def __repr__(self) -> str:
        c = self.R_ref.shape[0] if self.R_ref is not None else "?"
        return f"EuclideanAlignment(C={c}, fitted={self._fitted})"
