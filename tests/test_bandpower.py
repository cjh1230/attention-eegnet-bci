"""Tests for features/bandpower.py — bandpower ratio feature extraction."""
import numpy as np
import pytest

from features.bandpower import bandpower_ratio


@pytest.fixture
def X():
    """Simulated 8ch EEG: (N, C, T)."""
    rng = np.random.RandomState(42)
    N, C, T = 10, 8, 250
    X = rng.randn(N, C, T).astype(np.float32)
    # Add alpha-band (10 Hz) oscillation on C3 for half the trials
    t = np.linspace(0, 1, T)
    for i in range(N // 2):
        X[i, 1, :] += 2.0 * np.sin(2 * np.pi * 10 * t)
    return X


class TestBandpowerRatio:
    def test_output_shape(self, X):
        ratios = bandpower_ratio(X, fs=250)
        assert ratios.shape == (X.shape[0], X.shape[1])

    def test_no_nan(self, X):
        ratios = bandpower_ratio(X, fs=250)
        assert not np.isnan(ratios).any()
        assert not np.isinf(ratios).any()

    def test_no_negative(self, X):
        """Bandpower (variance) should always be non-negative."""
        ratios = bandpower_ratio(X, fs=250)
        assert (ratios >= 0).all()

    def test_alpha_enhanced_trials_higher(self, X):
        """Trials with added alpha oscillation should have higher mu/beta ratio."""
        ratios = bandpower_ratio(X, fs=250)
        half = len(X) // 2
        # Alpha-enhanced trials (0..half-1) should have higher C3 ratio
        mean_alpha = ratios[:half, 1].mean()   # C3 with alpha
        mean_baseline = ratios[half:, 1].mean()  # C3 without alpha
        assert mean_alpha > mean_baseline

    def test_single_trial(self):
        x = np.random.RandomState(1).randn(1, 8, 250).astype(np.float32)
        ratios = bandpower_ratio(x, fs=250)
        assert ratios.shape == (1, 8)

    def test_custom_fs(self, X):
        ratios_250 = bandpower_ratio(X, fs=250)
        ratios_500 = bandpower_ratio(X, fs=500)
        # Different fs should produce different filter designs → different ratios
        assert not np.allclose(ratios_250, ratios_500)
