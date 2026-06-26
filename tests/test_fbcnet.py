"""Tests for models/fbcnet.py — FBCNet model + apply_filter_bank."""

import numpy as np
import pytest
import torch

from models.fbcnet import FBCNet, apply_filter_bank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def small_fbcnet():
    """FBCNet with small config for fast tests."""
    return FBCNet(n_bands=3, n_channels=8, n_classes=3, m=8, t_kernel=16, hidden=16)


# ---------------------------------------------------------------------------
# apply_filter_bank tests
# ---------------------------------------------------------------------------

class TestApplyFilterBank:
    """Tests for the apply_filter_bank() helper."""

    def test_output_shape(self, rng):
        X = rng.randn(10, 8, 300).astype(np.float32)
        X_mb = apply_filter_bank(X, bands=[(4, 8), (8, 12)])
        assert X_mb.shape == (10, 2, 8, 300)

    def test_no_nan(self, rng):
        X = rng.randn(10, 8, 300).astype(np.float32)
        X_mb = apply_filter_bank(X, bands=[(4, 8), (8, 12)])
        assert not np.any(np.isnan(X_mb))

    def test_different_bands_different_output(self, rng):
        X = rng.randn(10, 8, 300).astype(np.float32)
        X_mb = apply_filter_bank(X, bands=[(4, 8), (8, 12), (12, 16)])
        # Different bands should produce different filtered signals
        assert not np.allclose(X_mb[:, 0], X_mb[:, 1])

    def test_default_bands(self, rng):
        """Should work with default FBCSP_BANDS from config."""
        X = rng.randn(5, 8, 200).astype(np.float32)
        X_mb = apply_filter_bank(X)
        assert X_mb.shape[1] == 6  # FBCSP_BANDS has 6 bands (8–30 Hz)


# ---------------------------------------------------------------------------
# FBCNet model tests
# ---------------------------------------------------------------------------

class TestFBCNetInit:
    """Constructor tests."""

    def test_default_constructor(self):
        model = FBCNet()
        assert model.n_bands == 6
        assert model.n_channels == 8
        assert model.n_classes == 3
        assert model.m == 32

    def test_custom_params(self):
        model = FBCNet(n_bands=5, n_channels=16, n_classes=4, m=16,
                       t_kernel=32, dropout=0.3, hidden=64)
        assert model.n_bands == 5
        assert model.n_classes == 4
        assert model.m == 16

    def test_requires_filter_bank_flag(self):
        model = FBCNet()
        assert model.input_requires_filter_bank is True


class TestFBCNetForward:
    """Forward pass tests."""

    def test_output_shape(self, small_fbcnet):
        x = torch.randn(4, 3, 8, 300)  # (B, n_bands, C, T)
        out = small_fbcnet(x)
        assert out.shape == (4, 3)  # (B, n_classes)

    def test_single_sample(self, small_fbcnet):
        x = torch.randn(1, 3, 8, 300)
        out = small_fbcnet(x)
        assert out.shape == (1, 3)

    def test_deterministic_in_eval(self, small_fbcnet):
        small_fbcnet.eval()
        x = torch.randn(2, 3, 8, 300)
        with torch.no_grad():
            out1 = small_fbcnet(x)
            out2 = small_fbcnet(x)
        torch.testing.assert_close(out1, out2)

    def test_different_bands_different_output(self, small_fbcnet):
        small_fbcnet.eval()
        x1 = torch.randn(2, 3, 8, 300)
        x2 = x1.clone()
        x2[:, 0] = torch.randn(2, 8, 300)  # change first band
        with torch.no_grad():
            out1 = small_fbcnet(x1)
            out2 = small_fbcnet(x2)
        assert not torch.allclose(out1, out2)

    def test_single_band_works(self):
        model = FBCNet(n_bands=1, n_channels=8, n_classes=2, m=4, hidden=8)
        x = torch.randn(2, 1, 8, 300)
        out = model(x)
        assert out.shape == (2, 2)

    def test_custom_n_classes(self):
        for nc in [2, 4, 5]:
            model = FBCNet(n_bands=3, n_channels=8, n_classes=nc, m=8, hidden=16)
            x = torch.randn(2, 3, 8, 300)
            assert model(x).shape == (2, nc)


class TestFBCNetGradient:
    """Training mode tests."""

    def test_gradient_flows(self, small_fbcnet):
        small_fbcnet.train()
        x = torch.randn(4, 3, 8, 300)
        out = small_fbcnet(x)
        loss = out.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in small_fbcnet.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_gradient_updates_parameters(self, small_fbcnet):
        small_fbcnet.train()
        # Snapshot parameters
        before = {n: p.clone() for n, p in small_fbcnet.named_parameters()}

        x = torch.randn(4, 3, 8, 300)
        opt = torch.optim.SGD(small_fbcnet.parameters(), lr=0.1)
        loss = small_fbcnet(x).sum()
        loss.backward()
        opt.step()

        for name, param in small_fbcnet.named_parameters():
            assert not torch.equal(before[name], param), (
                f"Parameter {name} did not update"
            )


class TestFBCNetCheckpoint:
    """Save/load roundtrip tests."""

    def test_save_load_roundtrip(self, small_fbcnet, tmp_path):
        small_fbcnet.eval()
        x = torch.randn(4, 3, 8, 300)
        with torch.no_grad():
            out_before = small_fbcnet(x)

        # Save
        ckpt = {
            "model_type": "fbcnet",
            "state_dict": small_fbcnet.state_dict(),
            "config": {"n_channels": 8, "n_classes": 3},
        }
        path = tmp_path / "fbcnet_test.pt"
        torch.save(ckpt, path)

        # Load into fresh model
        loaded = FBCNet(n_bands=3, n_channels=8, n_classes=3, m=8, t_kernel=16, hidden=16)
        loaded.eval()
        loaded.load_state_dict(torch.load(path, weights_only=True)["state_dict"])

        with torch.no_grad():
            out_after = loaded(x)

        torch.testing.assert_close(out_before, out_after)
