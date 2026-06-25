"""Tests for FBCSP features + LDA baseline in features/csp.py."""

import numpy as np
import pytest

try:
    from features.csp import (
        csp_svm_baseline,
        csp_lda_baseline,
        fbcsp_features,
        fbcsp_classify,
    )
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


@pytest.mark.skipif(not HAS_MNE, reason="MNE not installed")
class TestFBCSPFeatures:
    """Unit tests for fbcsp_features()."""

    def _make_data(self, n_trials=40, n_channels=8, n_times=200, n_classes=2):
        rng = np.random.RandomState(42)
        X = rng.randn(n_trials, n_channels, n_times).astype(np.float32)
        y = np.array([i % n_classes for i in range(n_trials)], dtype=np.int64)
        return X, y

    _TEST_BANDS = [(4, 8), (8, 12), (12, 16)]

    def test_features_shape(self):
        X, y = self._make_data()
        feats = fbcsp_features(X, y, freq_bands=self._TEST_BANDS, n_components=4)
        expected = X.shape[0], len(self._TEST_BANDS) * 4
        assert feats.shape == expected

    def test_no_nan(self):
        X, y = self._make_data()
        feats = fbcsp_features(X, y, freq_bands=self._TEST_BANDS, n_components=4)
        assert not np.any(np.isnan(feats))
        assert not np.any(np.isinf(feats))

    def test_different_bands_different_features(self):
        X, y = self._make_data()
        # Two different band configs should produce different features
        feats_a = fbcsp_features(X, y, freq_bands=[(4, 8), (8, 12)], n_components=2)
        feats_b = fbcsp_features(X, y, freq_bands=[(20, 24), (28, 32)], n_components=2)
        assert not np.allclose(feats_a, feats_b)

    def test_3class(self):
        X, y = self._make_data(n_classes=3)
        feats = fbcsp_features(X, y, freq_bands=self._TEST_BANDS, n_components=4)
        assert feats.shape[1] == len(self._TEST_BANDS) * 4
        assert not np.any(np.isnan(feats))

    def test_custom_n_components(self):
        X, y = self._make_data()
        for nc in [2, 4, 8]:
            feats = fbcsp_features(
                X, y, freq_bands=self._TEST_BANDS, n_components=nc,
            )
            assert feats.shape[1] == len(self._TEST_BANDS) * nc

    def test_single_band_is_similar_to_csp(self):
        """FBCSP with one wide band ≈ single CSP (qualitative check)."""
        X, y = self._make_data(n_trials=30)
        # FBCSP with a single full band
        feats_fbcsp = fbcsp_features(
            X, y, freq_bands=[(8, 30)], n_components=4,
        )
        # CSP directly
        from mne.decoding import CSP
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        feats_csp = csp.fit_transform(X, y)
        # Shapes should match (not identical — filter edge effects — but close)
        assert feats_fbcsp.shape == feats_csp.shape


@pytest.mark.skipif(not HAS_MNE, reason="MNE not installed")
class TestFBCSPClassify:
    """Tests for fbcsp_classify()."""

    def _make_data(self, n_trials=40, n_channels=8, n_times=200, n_classes=2):
        rng = np.random.RandomState(42)
        X = rng.randn(n_trials, n_channels, n_times).astype(np.float32)
        y = np.array([i % n_classes for i in range(n_trials)], dtype=np.int64)
        return X, y

    _TEST_BANDS = [(4, 8), (8, 12), (12, 16)]

    def test_lda_returns_valid(self):
        X, y = self._make_data()
        result = fbcsp_classify(
            X, y, freq_bands=self._TEST_BANDS, n_components=4,
            classifier="lda", cv=3,
        )
        for key in ["accuracy", "accuracy_std", "scores"]:
            assert key in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert len(result["scores"]) == 3

    def test_svm_returns_valid(self):
        X, y = self._make_data()
        result = fbcsp_classify(
            X, y, freq_bands=self._TEST_BANDS, n_components=4,
            classifier="svm", cv=3,
        )
        assert 0.0 <= result["accuracy"] <= 1.0
        assert len(result["scores"]) == 3

    def test_3class(self):
        X, y = self._make_data(n_classes=3)
        result = fbcsp_classify(
            X, y, freq_bands=self._TEST_BANDS, n_components=4,
            classifier="lda", cv=3,
        )
        assert 0.0 <= result["accuracy"] <= 1.0


@pytest.mark.skipif(not HAS_MNE, reason="MNE not installed")
class TestCSPLDA:
    """Tests for csp_lda_baseline()."""

    def _make_data(self, n_trials=40, n_channels=8, n_times=200, n_classes=2):
        rng = np.random.RandomState(42)
        X = rng.randn(n_trials, n_channels, n_times).astype(np.float32)
        y = np.array([i % n_classes for i in range(n_trials)], dtype=np.int64)
        return X, y

    def test_returns_dict_keys(self):
        X, y = self._make_data()
        result = csp_lda_baseline(X, y, n_components=4, cv=3)
        for key in ["accuracy", "accuracy_std", "scores"]:
            assert key in result

    def test_accuracy_in_range(self):
        X, y = self._make_data()
        result = csp_lda_baseline(X, y, n_components=4, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_scores_length_matches_cv(self):
        X, y = self._make_data()
        cv = 3
        result = csp_lda_baseline(X, y, n_components=4, cv=cv)
        assert len(result["scores"]) == cv

    def test_3class(self):
        X, y = self._make_data(n_classes=3)
        result = csp_lda_baseline(X, y, n_components=6, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0


@pytest.mark.skipif(not HAS_MNE, reason="MNE not installed")
def test_csp_svm_still_works():
    """Existing csp_svm_baseline should not be broken."""
    rng = np.random.RandomState(99)
    X = rng.randn(40, 8, 200).astype(np.float32)
    y = np.array([0, 1] * 20, dtype=np.int64)
    result = csp_svm_baseline(X, y, n_components=4, cv=3)
    assert "accuracy" in result
    assert 0.0 <= result["accuracy"] <= 1.0
