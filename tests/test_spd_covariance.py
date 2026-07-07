"""Tests for features/spd_covariance.py — SPD covariance computation."""
import numpy as np
import pytest

from features.spd_covariance import (
    compute_covariance,
    compute_covariance_scm,
    compute_covariance_shrinkage,
    compute_multiband_covariance,
    is_spd,
    geodesic_mixup,
    batch_geodesic_mixup,
    paired_spd_augment,
    augment_covariance_channel_dropout,
    augment_covariance_perturb,
    augment_temporal_crop,
    mask_covariance_channels,
    HAS_PYRIEMANN,
    MI_SUB_BANDS,
)


@pytest.fixture
def X():
    """Simulated 8ch EEG trials: (N, C, T)."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 8, 250).astype(np.float32)
    # Add structured signal to avoid degenerate covariances
    t = np.linspace(0, 1, 250)
    for i in range(20):
        X[i, 1, :] += 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz on C3
        X[i, 3, :] += 0.5 * np.sin(2 * np.pi * 12 * t)  # 12 Hz on C4
    return X


@pytest.fixture
def C(X):
    """Pre-computed SCM covariances."""
    return compute_covariance_scm(X)


# ── SCM ──────────────────────────────────────────────────────────────────────

class TestSCM:
    def test_shape(self, X):
        C = compute_covariance_scm(X)
        assert C.shape == (20, 8, 8)
        assert C.dtype == np.float32

    def test_symmetry(self, X):
        C = compute_covariance_scm(X)
        for i in range(len(C)):
            assert np.allclose(C[i], C[i].T, atol=1e-5), f"Matrix {i} not symmetric"

    def test_psd(self, X):
        C = compute_covariance_scm(X)
        for i in range(len(C)):
            eigvals = np.linalg.eigvalsh(C[i])
            assert np.all(eigvals >= -1e-7), f"Matrix {i} has negative eigenvalues: {eigvals[:3]}"

    def test_custom_regularisation(self, X):
        C = compute_covariance_scm(X, reg=0.01)
        assert C.shape == (20, 8, 8)

    def test_no_regularisation(self, X):
        C = compute_covariance_scm(X, reg=0.0)
        assert C.shape == (20, 8, 8)


# ── compute_covariance dispatcher ────────────────────────────────────────────

class TestComputeCovariance:
    def test_scm(self, X):
        C = compute_covariance(X, estimator="scm")
        assert C.shape == (20, 8, 8)

    @pytest.mark.skipif(not HAS_PYRIEMANN, reason="pyriemann not installed")
    def test_lwf(self, X):
        C = compute_covariance(X, estimator="lwf")
        assert C.shape == (20, 8, 8)
        assert np.all(np.isfinite(C))

    def test_unknown_estimator(self, X):
        with pytest.raises(ValueError, match="Unknown estimator"):
            compute_covariance(X, estimator="fake")

    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="Expected 3D"):
            compute_covariance(np.random.randn(8, 250))


# ── Shrinkage ────────────────────────────────────────────────────────────────

class TestShrinkage:
    def test_shape(self, X):
        C = compute_covariance_shrinkage(X)
        assert C.shape == (20, 8, 8)

    def test_oas_auto(self, X):
        C = compute_covariance_shrinkage(X, alpha=None)
        assert C.shape == (20, 8, 8)
        assert is_spd(C).all()

    def test_manual_alpha(self, X):
        C = compute_covariance_shrinkage(X, alpha=0.5)
        assert C.shape == (20, 8, 8)

    def test_is_spd(self, X):
        C = compute_covariance_shrinkage(X, reg=0.001)
        assert is_spd(C).all()


# ── is_spd ───────────────────────────────────────────────────────────────────

class TestIsSPD:
    def test_spd_matrix(self, C):
        ok = is_spd(C)
        assert ok.all()

    def test_non_symmetric(self):
        A = np.random.randn(10, 8, 8).astype(np.float32)
        ok = is_spd(A)
        assert not ok.all()

    def test_negative_definite(self):
        # Create a negative-definite matrix
        A = -np.eye(8, dtype=np.float32) - 1.0
        ok = is_spd(A)
        assert not ok


# ── Multi-band ───────────────────────────────────────────────────────────────

class TestMultibandCovariance:
    def test_default_bands(self, X):
        result = compute_multiband_covariance(X, fs=250)
        assert isinstance(result, dict)
        assert "mu" in result
        assert "beta" in result
        for name, C in result.items():
            assert C.shape == (20, 8, 8), f"{name}: {C.shape}"
            assert is_spd(C).all(), f"{name}: not all SPD"

    def test_custom_bands(self, X):
        bands = [("low", 8, 12), ("high", 12, 16)]
        result = compute_multiband_covariance(X, bands=bands, fs=250)
        assert "low" in result
        assert "high" in result


# ── Augmentations ────────────────────────────────────────────────────────────

class TestAugmentation:
    def test_channel_dropout(self, C):
        C_aug = augment_covariance_channel_dropout(C, n_drop=2)
        assert C_aug.shape == C.shape
        assert is_spd(C_aug).all()

    def test_perturb(self, C):
        C_aug = augment_covariance_perturb(C, scale=0.05)
        assert C_aug.shape == C.shape
        assert is_spd(C_aug).all()

    def test_temporal_crop(self, X):
        X_crop = augment_temporal_crop(X, crop_ratio=0.8)
        assert X_crop.shape[0] == X.shape[0]
        assert X_crop.shape[1] == X.shape[1]
        assert X_crop.shape[2] == int(X.shape[2] * 0.8)

    def test_paired_augment(self, C):
        C1, C2 = paired_spd_augment(C)
        assert C1.shape == C.shape
        assert C2.shape == C.shape
        assert not np.allclose(C1, C2)  # should differ

    def test_mask_channels(self, C):
        C_masked, mask = mask_covariance_channels(C, n_mask=1)
        assert C_masked.shape == C.shape
        assert mask.shape == (len(C), C.shape[-1])
        assert mask.sum() == len(C)  # 1 masked channel per sample

    def test_channel_dropout_keeps_size(self, C):
        C_aug = augment_covariance_channel_dropout(C, n_drop=1)
        assert C_aug.shape == C.shape


# ── Geodesic mixup ───────────────────────────────────────────────────────────

class TestGeodesicMixup:
    def test_two_matrices(self):
        rng = np.random.RandomState(42)
        A = np.random.randn(8, 250).astype(np.float32)
        B = np.random.randn(8, 250).astype(np.float32)
        C1 = compute_covariance_scm(A[np.newaxis])[0]
        C2 = compute_covariance_scm(B[np.newaxis])[0]
        C_mix = geodesic_mixup(C1, C2, lam=0.5)
        assert C_mix.shape == (8, 8)
        assert is_spd(C_mix[np.newaxis]).all()

    def test_batch(self, C):
        C_mix = geodesic_mixup(C[:5], C[5:10], lam=0.5)
        assert C_mix.shape == (5, 8, 8)
        assert is_spd(C_mix).all()

    def test_random_lambda(self, C):
        """lam=None should use random mixing ratio."""
        C_mix = geodesic_mixup(C[:3], C[3:6], lam=None)
        assert C_mix.shape == (3, 8, 8)
        assert is_spd(C_mix).all()

    def test_mix_is_between(self, C):
        """Geodesic mix at lam=0.5 should differ from both endpoints."""
        C_mix = geodesic_mixup(C[:5], C[5:10], lam=0.5)
        assert not np.allclose(C_mix, C[:5])
        assert not np.allclose(C_mix, C[5:10])


class TestBatchGeodesicMixup:
    def test_output_shapes(self, C):
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2)
        C_mixed, y_mixed = batch_geodesic_mixup(C, y, alpha=0.5)
        assert C_mixed.shape == C.shape
        assert y_mixed.shape == (len(C), y.max() + 1)
        # Soft labels should sum to ~1
        assert np.allclose(y_mixed.sum(axis=-1), 1.0)
