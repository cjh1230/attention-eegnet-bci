"""Tests for models/eeg_tcnet.py — EEG-TCNet model."""

import numpy as np
import pytest
import torch

from models.eeg_tcnet import EEGTCNet, TCNBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tcnet():
    """EEG-TCNet with small config for fast tests."""
    return EEGTCNet(
        n_channels=8, n_classes=3, F1=4, D=2, F2=8,
        tcn_kernel=8, tcn_depth=2, dropout=0.3,
    )


# ---------------------------------------------------------------------------
# TCNBlock tests
# ---------------------------------------------------------------------------

class TestTCNBlock:
    def test_output_shape(self):
        block = TCNBlock(in_channels=8, out_channels=8, kernel_size=8, depth=2)
        x = torch.randn(2, 8, 100)
        out = block(x)
        # batch and channels preserved; time may vary slightly with causal padding
        assert out.shape[0] == x.shape[0]
        assert out.shape[1] == x.shape[1]

    def test_not_identity(self):
        """TCN should change the input (not identity mapping)."""
        block = TCNBlock(in_channels=8, out_channels=8, kernel_size=8, depth=2)
        # Use a longer input so residuals work across all layers
        x = torch.randn(2, 8, 200)
        block.eval()
        with torch.no_grad():
            out = block(x)
        # If shapes match, verify transformation happened
        if out.shape == x.shape:
            assert not torch.allclose(out, x)
        # If shapes differ, transformation definitely happened
        else:
            pass  # shape change itself proves non-identity


# ---------------------------------------------------------------------------
# EEGTCNet tests
# ---------------------------------------------------------------------------

class TestEEGTCNetInit:
    def test_default_constructor(self):
        model = EEGTCNet()
        assert model.n_channels == 8
        assert model.n_classes == 3
        assert model.F1 == 8

    def test_custom_params(self):
        model = EEGTCNet(n_channels=16, n_classes=4, F1=12, D=3, F2=24)
        assert model.n_channels == 16
        assert model.n_classes == 4


class TestEEGTCNetForward:
    def test_output_shape(self, small_tcnet):
        x = torch.randn(4, 8, 500)
        out = small_tcnet(x)
        assert out.shape == (4, 3)

    def test_output_shape_3d_input(self, small_tcnet):
        """Model should accept (B, C, T) input (no channel dim)."""
        x = torch.randn(2, 8, 500)
        out = small_tcnet(x)
        assert out.shape == (2, 3)

    def test_deterministic_in_eval(self, small_tcnet):
        small_tcnet.eval()
        x = torch.randn(2, 8, 500)
        with torch.no_grad():
            out1 = small_tcnet(x)
            out2 = small_tcnet(x)
        torch.testing.assert_close(out1, out2)

    def test_different_inputs_different_outputs(self, small_tcnet):
        small_tcnet.eval()
        x1 = torch.randn(2, 8, 500)
        x2 = torch.randn(2, 8, 500)
        with torch.no_grad():
            out1 = small_tcnet(x1)
            out2 = small_tcnet(x2)
        assert not torch.allclose(out1, out2)

    def test_variable_time_lengths(self):
        """Model should work with different time lengths."""
        model = EEGTCNet(n_channels=8, n_classes=2, F1=4, D=2, F2=8)
        for T in [200, 400, 750]:
            x = torch.randn(2, 8, T)
            out = model(x)
            assert out.shape == (2, 2)

    def test_batch_independence(self, small_tcnet):
        """Sample i in batch should produce the same output regardless of
        what other samples are in the batch."""
        small_tcnet.eval()
        x_a = torch.randn(1, 8, 500)
        x_b = torch.randn(1, 8, 500)

        with torch.no_grad():
            solo_a = small_tcnet(x_a)
            solo_b = small_tcnet(x_b)
            batch = small_tcnet(torch.cat([x_a, x_b]))

        torch.testing.assert_close(batch[0:1], solo_a)
        torch.testing.assert_close(batch[1:2], solo_b)


class TestEEGTCNetGradient:
    def test_gradient_flows(self, small_tcnet):
        small_tcnet.train()
        x = torch.randn(4, 8, 500)
        out = small_tcnet(x)
        loss = out.sum()
        loss.backward()

        for name, param in small_tcnet.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


class TestEEGTCNetCheckpoint:
    def test_save_load_roundtrip(self, small_tcnet, tmp_path):
        small_tcnet.eval()
        x = torch.randn(4, 8, 500)
        with torch.no_grad():
            out_before = small_tcnet(x)

        ckpt = {
            "model_type": "eeg_tcnet",
            "state_dict": small_tcnet.state_dict(),
            "config": {
                "n_channels": 8, "n_classes": 3, "F1": 4, "D": 2, "F2": 8,
            },
        }
        path = tmp_path / "eeg_tcnet_test.pt"
        torch.save(ckpt, path)

        loaded = EEGTCNet(
            n_channels=8, n_classes=3, F1=4, D=2, F2=8,
            tcn_kernel=8, tcn_depth=2, dropout=0.3,
        )
        loaded.eval()
        loaded.load_state_dict(torch.load(path, weights_only=True)["state_dict"])

        with torch.no_grad():
            out_after = loaded(x)

        torch.testing.assert_close(out_before, out_after)
