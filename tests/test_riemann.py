"""Tests for features/riemann.py — Riemannian Geometry baselines."""

import numpy as np
import pytest

try:
    from features.riemann import (
        RiemannCovariances,
        FilterBankRiemann,
        riemann_tangent_classify,
        riemann_mdm_classify,
        fgmdm_classify,
        riemann_classify,
    )

    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _make_dummy_data(
    n_trials: int = 40,
    n_channels: int = 8,
    n_times: int = 200,
    n_classes: int = 2,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic EEG with class-conditional covariance structure.

    Each class gets a different random spatial covariance matrix (via Cholesky
    decomposition), ensuring that covariance-based methods can separate the
    classes above chance.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # One random spatial covariance per class
    cov_mats = []
    for c in range(n_classes):
        A = rng.randn(n_channels, n_channels)
        cov = A @ A.T  # PSD
        cov += np.eye(n_channels) * 0.1  # regularization
        cov_mats.append(cov)

    X = np.zeros((n_trials, n_channels, n_times), dtype=np.float32)
    y = np.zeros(n_trials, dtype=np.int64)

    for i in range(n_trials):
        cls = i % n_classes
        L = np.linalg.cholesky(cov_mats[cls])
        # Generate trial as multivariate white noise shaped by L
        noise = rng.randn(n_channels, n_times)
        X[i] = (L @ noise).astype(np.float32)
        y[i] = cls

    return X, y


_TEST_BANDS = [(4, 8), (8, 12), (12, 16)]


# ---------------------------------------------------------------------------
# RiemannCovariances
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYRIEMANN, reason="pyriemann not installed")
class TestRiemannCovariances:
    """Tests for RiemannCovariances transformer."""

    def test_output_shape(self):
        X, y = _make_dummy_data(n_trials=40, n_channels=8)
        rc = RiemannCovariances(estimator="scm")
        covs = rc.fit_transform(X, y)
        assert covs.shape == (40, 8, 8)

    def test_symmetric(self):
        X, y = _make_dummy_data(n_trials=40, n_channels=8)
        rc = RiemannCovariances(estimator="scm")
        covs = rc.fit_transform(X, y)
        for i in range(covs.shape[0]):
            assert np.allclose(covs[i], covs[i].T)

    def test_positive_diagonal(self):
        X, y = _make_dummy_data(n_trials=40, n_channels=8)
        rc = RiemannCovariances(estimator="scm")
        covs = rc.fit_transform(X, y)
        for i in range(covs.shape[0]):
            assert np.all(np.diag(covs[i]) > 0)

    def test_different_estimators(self):
        X, y = _make_dummy_data(n_trials=30, n_channels=8)
        for est in ["scm", "lwf", "oas"]:
            rc = RiemannCovariances(estimator=est)
            covs = rc.fit_transform(X, y)
            assert covs.shape == (30, 8, 8)
            assert not np.any(np.isnan(covs))

    def test_invalid_estimator_raises(self):
        X, _ = _make_dummy_data()
        rc = RiemannCovariances(estimator="invalid")
        with pytest.raises(ValueError, match="Unknown cov_estimator"):
            rc.fit(X)

    def test_transform_before_fit_raises(self):
        X, _ = _make_dummy_data()
        rc = RiemannCovariances(estimator="scm")
        with pytest.raises(RuntimeError, match="Must call fit"):
            rc.transform(X)


# ---------------------------------------------------------------------------
# FilterBankRiemann
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYRIEMANN, reason="pyriemann not installed")
class TestFilterBankRiemann:
    """Tests for FilterBankRiemann transformer."""

    def test_features_shape(self):
        X, y = _make_dummy_data(n_trials=40, n_channels=8)
        fb = FilterBankRiemann(
            freq_bands=_TEST_BANDS, cov_estimator="scm", metric="riemann",
        )
        feats = fb.fit_transform(X, y)
        # Tangent space dimension for C channels is C*(C+1)//2
        expected_per_band = 8 * 9 // 2  # 36
        expected_total = len(_TEST_BANDS) * expected_per_band  # 3 * 36 = 108
        assert feats.shape == (40, expected_total)

    def test_no_nan(self):
        X, y = _make_dummy_data(n_trials=40, n_channels=8)
        fb = FilterBankRiemann(
            freq_bands=_TEST_BANDS, cov_estimator="scm", metric="riemann",
        )
        feats = fb.fit_transform(X, y)
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))

    def test_default_bands(self):
        """When freq_bands is None, uses FBCSP_BANDS from config."""
        X, y = _make_dummy_data(n_trials=40, n_channels=8)
        fb = FilterBankRiemann(cov_estimator="scm", metric="riemann")
        feats = fb.fit_transform(X, y)
        assert feats.shape[0] == 40
        assert feats.shape[1] > 0
        assert not np.any(np.isnan(feats))

    def test_different_metrics(self):
        X, y = _make_dummy_data(n_trials=30, n_channels=8)
        for metric in ["riemann", "euclid", "logeuclid"]:
            fb = FilterBankRiemann(
                freq_bands=_TEST_BANDS, cov_estimator="scm", metric=metric,
            )
            feats = fb.fit_transform(X, y)
            assert not np.any(np.isnan(feats))

    def test_transform_before_fit_raises(self):
        X, _ = _make_dummy_data()
        fb = FilterBankRiemann(freq_bands=_TEST_BANDS)
        with pytest.raises(RuntimeError, match="must be fitted"):
            fb.transform(X)


# ---------------------------------------------------------------------------
# riemann_tangent_classify
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYRIEMANN, reason="pyriemann not installed")
class TestRiemannTangentClassify:
    """Tests for riemann_tangent_classify()."""

    def test_returns_dict_keys(self):
        X, y = _make_dummy_data()
        result = riemann_tangent_classify(X, y, cv=3)
        for key in ["accuracy", "accuracy_std", "scores"]:
            assert key in result
        assert result["method"] == "tangent"

    def test_accuracy_in_range(self):
        X, y = _make_dummy_data()
        result = riemann_tangent_classify(X, y, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_scores_length_matches_cv(self):
        X, y = _make_dummy_data()
        cv = 3
        result = riemann_tangent_classify(X, y, cv=cv)
        assert len(result["scores"]) == cv

    def test_3class(self):
        X, y = _make_dummy_data(n_classes=3)
        result = riemann_tangent_classify(X, y, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_different_classifiers(self):
        X, y = _make_dummy_data()
        for clf in ["lda", "svm"]:
            result = riemann_tangent_classify(X, y, classifier=clf, cv=3)
            assert result["classifier"] == clf
            assert 0.0 <= result["accuracy"] <= 1.0

    def test_different_cov_estimators(self):
        X, y = _make_dummy_data()
        for est in ["scm", "lwf", "oas"]:
            result = riemann_tangent_classify(X, y, cov_estimator=est, cv=3)
            assert result["cov_estimator"] == est
            assert 0.0 <= result["accuracy"] <= 1.0

    def test_above_chance_on_structured_data(self):
        """Covariance-structured data should yield above-chance accuracy."""
        X, y = _make_dummy_data(n_trials=60, n_channels=8, n_classes=2)
        result = riemann_tangent_classify(X, y, cv=5)
        # With class-conditional covariances, accuracy should be well above 0.5
        assert result["accuracy"] > 0.5


# ---------------------------------------------------------------------------
# riemann_mdm_classify
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYRIEMANN, reason="pyriemann not installed")
class TestRiemannMDMClassify:
    """Tests for riemann_mdm_classify()."""

    def test_returns_dict_keys(self):
        X, y = _make_dummy_data()
        result = riemann_mdm_classify(X, y, cv=3)
        for key in ["accuracy", "accuracy_std", "scores"]:
            assert key in result
        assert result["method"] == "mdm"
        assert result["classifier"] == "mdm"

    def test_accuracy_in_range(self):
        X, y = _make_dummy_data()
        result = riemann_mdm_classify(X, y, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_scores_length_matches_cv(self):
        X, y = _make_dummy_data()
        cv = 3
        result = riemann_mdm_classify(X, y, cv=cv)
        assert len(result["scores"]) == cv

    def test_different_metrics(self):
        X, y = _make_dummy_data()
        for metric in ["riemann", "euclid"]:
            result = riemann_mdm_classify(X, y, metric=metric, cv=3)
            assert result["metric"] == metric
            assert 0.0 <= result["accuracy"] <= 1.0

    def test_3class(self):
        X, y = _make_dummy_data(n_classes=3)
        result = riemann_mdm_classify(X, y, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# fgmdm_classify
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYRIEMANN, reason="pyriemann not installed")
class TestFgMDMClassify:
    """Tests for fgmdm_classify()."""

    def test_returns_dict_keys(self):
        X, y = _make_dummy_data()
        result = fgmdm_classify(X, y, freq_bands=_TEST_BANDS, cv=3)
        for key in ["accuracy", "accuracy_std", "scores"]:
            assert key in result
        assert result["method"] == "fgmdm"

    def test_accuracy_in_range(self):
        X, y = _make_dummy_data()
        result = fgmdm_classify(X, y, freq_bands=_TEST_BANDS, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_scores_length_matches_cv(self):
        X, y = _make_dummy_data()
        cv = 3
        result = fgmdm_classify(X, y, freq_bands=_TEST_BANDS, cv=cv)
        assert len(result["scores"]) == cv

    def test_no_nan(self):
        X, y = _make_dummy_data()
        result = fgmdm_classify(X, y, freq_bands=_TEST_BANDS, cv=3)
        assert not np.any(np.isnan(result["scores"]))
        assert not np.isnan(result["accuracy"])

    def test_no_nan_with_default_bands(self):
        X, y = _make_dummy_data()
        result = fgmdm_classify(X, y, cv=3)
        assert not np.isnan(result["accuracy"])

    def test_3class(self):
        X, y = _make_dummy_data(n_classes=3)
        result = fgmdm_classify(X, y, freq_bands=_TEST_BANDS, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# riemann_classify (unified entry point)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYRIEMANN, reason="pyriemann not installed")
class TestRiemannClassify:
    """Tests for the unified riemann_classify() entry point."""

    def test_tangent_dispatches(self):
        X, y = _make_dummy_data()
        a = riemann_classify(X, y, method="tangent", cv=3)
        b = riemann_tangent_classify(X, y, cv=3)
        assert a["method"] == b["method"]

    def test_mdm_dispatches(self):
        X, y = _make_dummy_data()
        a = riemann_classify(X, y, method="mdm", cv=3)
        b = riemann_mdm_classify(X, y, cv=3)
        assert a["method"] == b["method"]

    def test_fgmdm_dispatches(self):
        X, y = _make_dummy_data()
        a = riemann_classify(X, y, method="fgmdm", freq_bands=_TEST_BANDS, cv=3)
        b = fgmdm_classify(X, y, freq_bands=_TEST_BANDS, cv=3)
        assert a["method"] == b["method"]

    def test_invalid_method_raises(self):
        X, y = _make_dummy_data()
        with pytest.raises(ValueError, match="Unknown method"):
            riemann_classify(X, y, method="nonexistent")

    def test_returns_consistent_structure(self):
        X, y = _make_dummy_data()
        for method in ["tangent", "mdm", "fgmdm"]:
            result = riemann_classify(
                X, y, method=method, freq_bands=_TEST_BANDS, cv=3,
            )
            assert result["method"] == method
            assert "accuracy" in result
            assert "accuracy_std" in result
            assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Numerical stability / edge cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYRIEMANN, reason="pyriemann not installed")
class TestNumericalStability:
    """Edge case and numerical stability tests."""

    def test_low_trials_with_shrinkage(self):
        """Ledoit-Wolf shrinkage should handle n_trials < n_channels."""
        X, y = _make_dummy_data(n_trials=6, n_channels=8)
        # LWF shrinkage helps when there are too few trials per class
        result = riemann_tangent_classify(
            X, y, cov_estimator="lwf", cv=3,
        )
        assert not np.isnan(result["accuracy"])

    def test_constant_signal_no_crash(self):
        """Constant input should not produce NaN (shrinkage helps)."""
        rng = np.random.RandomState(42)
        X = np.ones((30, 8, 200), dtype=np.float32) + 0.01 * rng.randn(30, 8, 200).astype(np.float32)
        y = np.array([0, 1] * 15, dtype=np.int64)
        result = riemann_tangent_classify(
            X, y, cov_estimator="lwf", cv=3,
        )
        assert not np.isnan(result["accuracy"])

    def test_few_channels(self):
        """Works with only 3 channels."""
        X, y = _make_dummy_data(n_trials=40, n_channels=3)
        result = riemann_tangent_classify(X, y, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Module-level guards
# ---------------------------------------------------------------------------

@pytest.mark.skipif(HAS_PYRIEMANN, reason="pyriemann IS installed — skip guard test")
def test_import_guard():
    """When pyriemann is absent, the module should set HAS_PYRIEMANN=False."""
    from features import riemann as rmod
    assert rmod.HAS_PYRIEMANN is False
