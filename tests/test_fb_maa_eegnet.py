"""Tests for FB-MAA-EEGNet model."""
import numpy as np
import torch
import pytest

from models.fb_maa_eegnet import FBMAAEEGNet
from models.fbcnet import apply_filter_bank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return FBMAAEEGNet(n_bands=6, n_channels=8, n_classes=2)


@pytest.fixture
def input_batch():
    """(B, n_bands, C, T) multi-band input."""
    return torch.randn(4, 6, 8, 200)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestFBMAAEEGNetInit:
    def test_default_constructor(self):
        model = FBMAAEEGNet()
        assert model.n_bands == 6
        assert model.n_channels == 8
        assert model.n_classes == 2
        assert model.F1 == 8
        assert model.D == 2
        assert model.F2 == 16

    def test_custom_params(self):
        model = FBMAAEEGNet(
            n_bands=4, n_channels=8, n_classes=4,
            F1=16, D=4, F2=32, dropout=0.3,
        )
        assert model.n_bands == 4
        assert model.n_classes == 4
        assert model.F1 == 16

    def test_requires_filter_bank_flag(self):
        model = FBMAAEEGNet()
        assert model.input_requires_filter_bank is True


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestFBMAAEEGNetForward:
    def test_output_shape(self, model, input_batch):
        out = model(input_batch)
        assert out.shape == (4, 2)

    def test_single_sample(self, model):
        x = torch.randn(1, 6, 8, 200)
        out = model(x)
        assert out.shape == (1, 2)

    def test_deterministic_in_eval(self, model, input_batch):
        model.eval()
        with torch.no_grad():
            out1 = model(input_batch)
            out2 = model(input_batch)
        assert torch.equal(out1, out2)

    def test_different_bands_different_output(self, model):
        """Output should differ when bands are shuffled."""
        x = torch.randn(1, 6, 8, 200)
        model.eval()
        with torch.no_grad():
            out1 = model(x)
            # Shuffle band order
            x_shuffled = x[:, [2, 0, 1, 5, 3, 4], :, :]
            out2 = model(x_shuffled)
        assert not torch.allclose(out1, out2)

    def test_single_band_works(self):
        model = FBMAAEEGNet(n_bands=1, n_channels=8, n_classes=2)
        x = torch.randn(2, 1, 8, 200)
        out = model(x)
        assert out.shape == (2, 2)

    def test_custom_n_classes(self):
        model = FBMAAEEGNet(n_bands=6, n_channels=8, n_classes=4)
        x = torch.randn(2, 6, 8, 200)
        out = model(x)
        assert out.shape == (2, 4)

    def test_different_time_lengths(self):
        """Different T values should work with fresh models (lazy classifier)."""
        for T in [100, 200, 400, 750]:
            model = FBMAAEEGNet(n_bands=6, n_channels=8, n_classes=2)
            x = torch.randn(2, 6, 8, T)
            out = model(x)
            assert out.shape == (2, 2)

    def test_no_nan(self, model, input_batch):
        out = model(input_batch)
        assert not torch.isnan(out).any()

    def test_softmax_yields_probabilities(self, model, input_batch):
        model.eval()
        with torch.no_grad():
            out = model(input_batch)
            probs = torch.softmax(out, dim=1)
            assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestFBMAAEEGNetGradient:
    def test_gradient_flows(self, model, input_batch):
        x = input_batch.clone().requires_grad_(True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_gradient_updates_parameters(self, model, input_batch):
        trainable = [p for p in model.parameters() if p.requires_grad]
        params_before = [p.clone() for p in trainable]
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        out = model(input_batch)
        loss = out.sum()
        loss.backward()
        opt.step()
        changed = sum(
            1 for b, a in zip(params_before, trainable)
            if not torch.equal(b, a)
        )
        assert changed >= 1, "No trainable parameters changed"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestFBMAAEEGNetCheckpoint:
    def test_save_load_roundtrip(self, model, input_batch, tmp_path):
        model.eval()
        with torch.no_grad():
            out_before = model(input_batch)

        path = tmp_path / "fb_maa_eegnet.pt"
        torch.save(model.state_dict(), path)

        model2 = FBMAAEEGNet(n_bands=6, n_channels=8, n_classes=2)
        # Forward pass to build lazy classifier on model2
        with torch.no_grad():
            _ = model2(input_batch)
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()

        with torch.no_grad():
            out_after = model2(input_batch)
        assert torch.allclose(out_before, out_after, atol=1e-6)


# ---------------------------------------------------------------------------
# Integration: apply_filter_bank + FBMAAEEGNet
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_with_apply_filter_bank(self, model, rng):
        """End-to-end: raw EEG → filter bank → FB-MAA-EEGNet."""
        X = rng.randn(8, 8, 250).astype(np.float32)
        X_mb = apply_filter_bank(X, fs=250)
        assert X_mb.shape == (8, 6, 8, 250)

        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(X_mb))
        assert out.shape == (8, 2)
        assert not torch.isnan(out).any()

    def test_large_input_stable(self, model, rng):
        """Larger batch and time dims should not cause OOM or NaN."""
        X = rng.randn(16, 8, 500).astype(np.float32)
        X_mb = apply_filter_bank(X, fs=250)
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(X_mb))
        assert out.shape == (16, 2)
        assert not torch.isnan(out).any()
