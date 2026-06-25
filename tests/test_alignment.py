"""Tests for EuclideanAlignment in preprocessing/alignment.py."""

import numpy as np
import pytest

from preprocessing.alignment import EuclideanAlignment, _matrix_sqrt_inv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subject(rng, n_trials: int, n_channels: int, n_times: int,
                  scale: float = 1.0, shift: float = 0.0) -> np.ndarray:
    """Generate synthetic EEG trials with a controlled covariance structure."""
    # Create a random spatial mixing matrix to give structured covariance
    mixing = rng.randn(n_channels, n_channels) * scale + shift
    cov = mixing @ mixing.T
    L = np.linalg.cholesky(cov + np.eye(n_channels) * 0.1)
    white = rng.randn(n_trials, n_channels, n_times)
    return (L @ white).astype(np.float32)


# ---------------------------------------------------------------------------
# _matrix_sqrt_inv
# ---------------------------------------------------------------------------

def test_matrix_sqrt_inv_identity():
    """R^(-1/2) of identity ≈ identity."""
    I = np.eye(5, dtype=np.float64)
    result = _matrix_sqrt_inv(I, reg=0.0)
    np.testing.assert_allclose(result, I, atol=1e-10)


def test_matrix_sqrt_inv_inverts():
    """R^(-1/2) @ R @ R^(-1/2) ≈ I."""
    rng = np.random.RandomState(123)
    A = rng.randn(6, 6)
    R = A @ A.T  # PSD
    R_inv_sqrt = _matrix_sqrt_inv(R, reg=1e-8)
    sandwich = R_inv_sqrt @ R @ R_inv_sqrt
    np.testing.assert_allclose(sandwich, np.eye(6), atol=1e-6)


# ---------------------------------------------------------------------------
# EuclideanAlignment
# ---------------------------------------------------------------------------

class TestEuclideanAlignment:
    """Unit tests for the EuclideanAlignment class."""

    def test_output_shape(self, rng):
        """Aligned output keeps the same (N, C, T) shape."""
        X1 = rng.randn(20, 8, 500).astype(np.float32)
        X2 = rng.randn(15, 8, 500).astype(np.float32)

        ea = EuclideanAlignment()
        aligned = ea.fit_transform([X1, X2])

        assert aligned[0].shape == X1.shape
        assert aligned[1].shape == X2.shape
        assert aligned[0].dtype == X1.dtype

    def test_reduces_covariance_difference(self, rng):
        """EA should reduce the Frobenius distance between subject covariances."""
        # Two subjects with very different scales
        X1 = _make_subject(rng, 50, 8, 300, scale=3.0)
        X2 = _make_subject(rng, 50, 8, 300, scale=0.3)

        def mean_cov(X):
            return np.mean([t @ t.T for t in X], axis=0)

        cov1_before = mean_cov(X1)
        cov2_before = mean_cov(X2)
        dist_before = float(np.linalg.norm(cov1_before - cov2_before, ord="fro"))

        ea = EuclideanAlignment()
        X1_a, X2_a = ea.fit_transform([X1, X2])

        cov1_after = mean_cov(X1_a)
        cov2_after = mean_cov(X2_a)
        dist_after = float(np.linalg.norm(cov1_after - cov2_after, ord="fro"))

        assert dist_after < dist_before, (
            f"EA should reduce covariance distance: "
            f"{dist_before:.4f} → {dist_after:.4f}"
        )

    def test_numerical_stability_zero_variance(self):
        """Zero-variance channels should not produce NaN."""
        X = np.zeros((10, 8, 300), dtype=np.float32)
        X[:, 0, :] = 1e-8  # near-zero on one channel
        X[:, 1:, :] = np.random.RandomState(42).randn(10, 7, 300).astype(np.float32)

        ea = EuclideanAlignment(reg=1e-4)
        aligned = ea.fit_transform([X])

        assert not np.any(np.isnan(aligned[0]))
        assert not np.any(np.isinf(aligned[0]))

    def test_numerical_stability_extreme_values(self, rng):
        """Extreme-valued channels should not produce NaN."""
        X = rng.randn(20, 8, 300).astype(np.float32)
        X[:, 0, :] *= 1e6  # one channel with very large values
        X[:, 1, :] *= 1e-6  # one channel with very small values

        ea = EuclideanAlignment(reg=1e-4)
        aligned = ea.fit_transform([X])

        assert not np.any(np.isnan(aligned[0]))
        assert not np.any(np.isinf(aligned[0]))

    def test_idempotent(self, rng):
        """Applying EA twice: second application changes very little."""
        X1 = _make_subject(rng, 30, 8, 300, scale=2.0)
        X2 = _make_subject(rng, 25, 8, 300, scale=0.8)

        ea = EuclideanAlignment()
        aligned_once = ea.fit_transform([X1, X2])

        # Second EA on already-aligned data
        ea2 = EuclideanAlignment()
        aligned_twice = ea2.fit_transform(aligned_once)

        # After first EA, covariances are already close to identity,
        # so second EA should barely change things
        diff = float(np.mean([
            np.linalg.norm(aligned_once[i] - aligned_twice[i])
            for i in range(2)
        ]))
        original_norm = float(np.mean([
            np.linalg.norm(aligned_once[i]) for i in range(2)
        ]))

        assert diff / original_norm < 0.1, (
            f"Second EA pass should have minimal effect, got relative diff "
            f"{diff / original_norm:.6f}"
        )

    def test_fit_transform_equivalent(self, rng):
        """fit() + transform() == fit_transform()."""
        X1 = rng.randn(20, 8, 500).astype(np.float32)
        X2 = rng.randn(15, 8, 500).astype(np.float32)

        # fit_transform path
        ea1 = EuclideanAlignment()
        result1 = ea1.fit_transform([X1, X2])

        # fit + transform path (fresh copies)
        X1b = X1.copy()
        X2b = X2.copy()
        ea2 = EuclideanAlignment()
        ea2.fit([X1b, X2b])
        result2 = [ea2.transform(X1b), ea2.transform(X2b)]

        for r1, r2 in zip(result1, result2):
            np.testing.assert_allclose(r1, r2, atol=1e-5)

    def test_single_subject(self, rng):
        """Single subject should work as a degenerate case."""
        X = rng.randn(20, 8, 500).astype(np.float32)

        ea = EuclideanAlignment()
        aligned = ea.fit_transform([X])

        assert aligned[0].shape == X.shape
        assert ea.fitted

    def test_regularization(self, rng):
        """reg alters the transform — heavy reg causes less whitening."""
        X1 = _make_subject(rng, 30, 8, 300, scale=3.0)

        ea_none = EuclideanAlignment(reg=0.0)
        ea_heavy = EuclideanAlignment(reg=100.0)

        aligned_none = ea_none.fit_transform([X1.copy()])[0]
        aligned_heavy = ea_heavy.fit_transform([X1.copy()])[0]

        # reg=0 and reg=100 should produce different results
        assert not np.allclose(aligned_none, aligned_heavy), (
            "Different reg values should produce different aligned outputs"
        )

        # With reg=0, the aligned covariance should be close to identity
        # (strong whitening).  With reg=100, the covariance should be
        # closer to the original (less whitening).
        def cov_dist_to_identity(X):
            covs = np.array([t @ t.T for t in X])
            mean_cov = covs.mean(axis=0)
            C = mean_cov.shape[0]
            mean_cov_norm = mean_cov / (np.trace(mean_cov) / C)
            return float(np.linalg.norm(mean_cov_norm - np.eye(C), ord="fro"))

        dist_none = cov_dist_to_identity(aligned_none)
        dist_heavy = cov_dist_to_identity(aligned_heavy)

        assert dist_none < dist_heavy, (
            f"reg=0 (whitening) should produce covariance closer to identity: "
            f"reg=0 dist={dist_none:.4f}, reg=100 dist={dist_heavy:.4f}"
        )

    def test_immutability(self, rng):
        """transform() must not modify the input array."""
        X = rng.randn(20, 8, 500).astype(np.float32)
        X_original = X.copy()

        ea = EuclideanAlignment()
        ea.fit([X])
        _ = ea.transform(X)

        np.testing.assert_array_equal(X, X_original)

    def test_empty_list_raises(self):
        """Empty X_list should raise ValueError."""
        ea = EuclideanAlignment()
        with pytest.raises(ValueError, match="at least one"):
            ea.fit([])

    def test_unfitted_transform_raises(self, rng):
        """Calling transform() before fit() should raise RuntimeError."""
        ea = EuclideanAlignment()
        X = rng.randn(5, 8, 300).astype(np.float32)
        with pytest.raises(RuntimeError, match="Must call fit"):
            ea.transform(X)

    def test_channel_mismatch_raises(self, rng):
        """Different channel counts in X_list should raise ValueError."""
        X1 = rng.randn(10, 8, 300).astype(np.float32)
        X2 = rng.randn(10, 6, 300).astype(np.float32)

        ea = EuclideanAlignment()
        with pytest.raises(ValueError, match="Channel mismatch"):
            ea.fit([X1, X2])

    def test_2d_input_raises(self, rng):
        """2D input should raise ValueError."""
        X = rng.randn(8, 300).astype(np.float32)

        ea = EuclideanAlignment()
        with pytest.raises(ValueError, match="Expected 3D"):
            ea.fit([X])

    def test_repr(self):
        """__repr__ should show state."""
        ea = EuclideanAlignment(reg=1e-4)
        assert "fitted=False" in repr(ea)
        assert "EuclideanAlignment" in repr(ea)

    def test_correct_alignment_known_covariance(self):
        """With known covariance structure, verify EA produces near-white output."""
        rng = np.random.RandomState(777)
        C, T = 4, 200

        # Create a fixed spatial covariance
        L = rng.randn(C, C)
        true_cov = L @ L.T + np.eye(C) * 0.5

        # Generate trials with this exact covariance
        X = (np.linalg.cholesky(true_cov) @ rng.randn(100, C, T)).astype(np.float32)

        ea = EuclideanAlignment(reg=1e-8)
        X_aligned = ea.fit_transform([X])[0]

        # After EA, the average covariance should be close to identity
        # (up to a scale factor — EA doesn't enforce unit trace)
        aligned_covs = np.array([t @ t.T for t in X_aligned])
        mean_aligned_cov = aligned_covs.mean(axis=0)

        # Normalise both to unit trace for comparison
        mean_aligned_cov /= np.trace(mean_aligned_cov) / C

        np.testing.assert_allclose(
            mean_aligned_cov, np.eye(C), atol=0.05,
            err_msg="EA should whiten the data covariance"
        )
