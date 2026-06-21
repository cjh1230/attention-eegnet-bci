"""Tests for features/csp.py — CSP + SVM baseline."""
import numpy as np
import pytest

try:
    from features.csp import csp_svm_baseline
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


@pytest.mark.skipif(not HAS_MNE, reason="MNE not installed")
class TestCSPSVM:
    def _make_dummy_data(self, n_trials=40, n_channels=8, n_times=200, n_classes=2, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        X = rng.randn(n_trials, n_channels, n_times).astype(np.float32)
        y = np.array([i % n_classes for i in range(n_trials)], dtype=np.int64)
        return X, y

    def test_returns_dict_with_keys(self):
        X, y = self._make_dummy_data()
        result = csp_svm_baseline(X, y, n_components=4, cv=3)
        for key in ["accuracy", "accuracy_std", "scores"]:
            assert key in result

    def test_accuracy_in_range(self):
        X, y = self._make_dummy_data()
        result = csp_svm_baseline(X, y, n_components=4, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_scores_length_matches_cv(self):
        X, y = self._make_dummy_data()
        cv = 3
        result = csp_svm_baseline(X, y, n_components=4, cv=cv)
        assert len(result["scores"]) == cv

    def test_3class(self):
        """CSP + SVM should work on 3-class data (OvR)."""
        rng = np.random.RandomState(42)
        X = rng.randn(60, 8, 200).astype(np.float32)
        y = np.array([i % 3 for i in range(60)], dtype=np.int64)
        result = csp_svm_baseline(X, y, n_components=6, cv=3)
        assert 0.0 <= result["accuracy"] <= 1.0
