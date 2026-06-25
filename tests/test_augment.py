"""Tests for preprocessing/augment.py."""

import numpy as np
import pytest

from preprocessing.augment import (
    gaussian_noise,
    channel_dropout,
    time_shift,
    amplitude_scale,
    frequency_mask,
    crop_augment,
    mixup_batch,
    mixup_criterion,
    eeg_augment,
    augment_dataset,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def X(rng):
    return rng.randn(20, 8, 300).astype(np.float32)


@pytest.fixture
def y():
    return np.array([0, 1] * 10, dtype=np.int64)


class TestGaussianNoise:
    def test_shape(self, X):
        assert gaussian_noise(X).shape == X.shape

    def test_not_identical(self, X):
        assert not np.allclose(gaussian_noise(X, sigma=0.1), X)


class TestChannelDropout:
    def test_shape(self, X):
        assert channel_dropout(X, p=0.3).shape == X.shape

    def test_some_channels_zeroed(self, X):
        result = channel_dropout(X, p=0.99)
        assert np.any(result == 0.0)

    def test_p0_no_dropout(self, X):
        result = channel_dropout(X, p=0.0)
        np.testing.assert_array_equal(result, X)


class TestTimeShift:
    def test_shape(self, X):
        assert time_shift(X, max_shift=10).shape == X.shape


class TestAmplitudeScale:
    def test_shape(self, X):
        assert amplitude_scale(X).shape == X.shape

    def test_not_identical(self, X):
        scaled = amplitude_scale(X, scale_range=(0.5, 0.5))
        assert not np.allclose(scaled, X)


class TestFrequencyMask:
    def test_shape(self, X):
        result = frequency_mask(X, max_width_hz=4)
        assert result.shape == X.shape

    def test_no_nan(self, X):
        result = frequency_mask(X, max_width_hz=4)
        assert not np.any(np.isnan(result))

    def test_different_from_input(self, X):
        result = frequency_mask(X, max_width_hz=10)
        assert not np.allclose(result, X)


class TestCropAugment:
    def test_shape(self, X, y):
        Xc, yc = crop_augment(X, y, window=200, stride=100)
        assert Xc.shape[1:] == (X.shape[1], 200)
        assert Xc.shape[0] == yc.shape[0]
        assert Xc.shape[0] >= X.shape[0]

    def test_labels_preserved(self, X, y):
        _, yc = crop_augment(X, y, window=200, stride=100)
        assert set(np.unique(yc)) == set(np.unique(y))

    def test_no_crop_when_window_exceeds_t(self, X, y):
        Xc, yc = crop_augment(X, y, window=500)
        np.testing.assert_array_equal(Xc, X)
        np.testing.assert_array_equal(yc, y)


class TestMixupBatch:
    def test_shape(self, X, y):
        rng_state = np.random.RandomState(42)
        Xm, ya, yb, lam = mixup_batch(X, y, alpha=0.2, rng=rng_state)
        assert Xm.shape == X.shape
        assert ya.shape == y.shape
        assert yb.shape == y.shape
        assert 0.5 <= lam <= 1.0

    def test_mixed_is_interpolation(self, X, y):
        rng_state = np.random.RandomState(0)
        Xm, _, _, _ = mixup_batch(X, y, alpha=0.2, rng=rng_state)
        # Mixed should not equal original (unless lam=1 which is rare)
        assert not np.allclose(Xm, X)


class TestMixupCriterion:
    def test_computes_loss(self):
        import torch
        criterion = torch.nn.CrossEntropyLoss()
        pred = torch.randn(4, 3)
        y_a = torch.tensor([0, 1, 2, 0])
        y_b = torch.tensor([1, 0, 1, 2])
        loss = mixup_criterion(criterion, pred, y_a, y_b, 0.7)
        assert loss.item() > 0


class TestEEGAugment:
    def test_shape_with_labels(self, X, y):
        Xa, ya = eeg_augment(X, y, seed=1)
        assert Xa.shape == X.shape
        assert ya.shape == y.shape

    def test_shape_without_labels(self, X):
        Xa = eeg_augment(X, seed=1)
        assert Xa.shape == X.shape


class TestAugmentDataset:
    def test_factor_2(self, X, y):
        Xa, ya = augment_dataset(X, y, factor=2, seed=42)
        assert Xa.shape[0] == X.shape[0] * 2
        assert ya.shape[0] == y.shape[0] * 2
