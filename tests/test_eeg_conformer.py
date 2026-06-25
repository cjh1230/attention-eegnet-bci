"""Tests for models/eeg_conformer.py — EEG Conformer."""

import torch
import pytest

from models.eeg_conformer import EEGConformer, PositionalEncoding, TransformerBlock


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(d_model=32, max_len=100)
        x = torch.randn(4, 50, 32)
        out = pe(x)
        assert out.shape == x.shape

    def test_adds_position_info(self):
        pe = PositionalEncoding(d_model=16)
        x = torch.randn(2, 20, 16)
        out = pe(x)
        assert not torch.allclose(out, x)


class TestTransformerBlock:
    def test_output_shape(self):
        blk = TransformerBlock(d_model=32, n_heads=4, d_ff=64)
        x = torch.randn(4, 50, 32)
        out = blk(x)
        assert out.shape == x.shape

    def test_gradient_flows(self):
        blk = TransformerBlock(d_model=32)
        x = torch.randn(4, 50, 32, requires_grad=True)
        out = blk(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestEEGConformer:
    @pytest.fixture
    def small_model(self):
        return EEGConformer(n_channels=8, n_classes=3, F1=4, D=2,
                            d_model=16, n_heads=2, n_layers=1, d_ff=32)

    def test_output_shape(self, small_model):
        x = torch.randn(4, 8, 500)
        out = small_model(x)
        assert out.shape == (4, 3)

    def test_output_shape_3d_input(self, small_model):
        x = torch.randn(2, 8, 500)
        out = small_model(x)
        assert out.shape == (2, 3)

    def test_deterministic_in_eval(self, small_model):
        small_model.eval()
        x = torch.randn(2, 8, 500)
        with torch.no_grad():
            o1 = small_model(x)
            o2 = small_model(x)
        torch.testing.assert_close(o1, o2)

    def test_variable_time_lengths(self):
        model = EEGConformer(n_channels=8, n_classes=2, F1=4, D=2)
        for T in [200, 400, 750]:
            x = torch.randn(2, 8, T)
            out = model(x)
            assert out.shape == (2, 2)

    def test_gradient_flows(self, small_model):
        small_model.train()
        x = torch.randn(4, 8, 500)
        out = small_model(x)
        loss = out.sum()
        loss.backward()
        for name, p in small_model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_checkpoint_roundtrip(self, small_model, tmp_path):
        small_model.eval()
        x = torch.randn(4, 8, 500)
        with torch.no_grad():
            out_before = small_model(x)

        ckpt = {
            "model_type": "eeg_conformer",
            "state_dict": small_model.state_dict(),
            "config": {"n_channels": 8, "n_classes": 3},
        }
        path = tmp_path / "conformer_test.pt"
        torch.save(ckpt, path)

        loaded = EEGConformer(n_channels=8, n_classes=3, F1=4, D=2,
                              d_model=16, n_heads=2, n_layers=1, d_ff=32)
        loaded.eval()
        loaded.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        with torch.no_grad():
            out_after = loaded(x)
        torch.testing.assert_close(out_before, out_after)

    def test_batch_independence(self, small_model):
        small_model.eval()
        x_a = torch.randn(1, 8, 500)
        x_b = torch.randn(1, 8, 500)
        with torch.no_grad():
            solo_a = small_model(x_a)
            solo_b = small_model(x_b)
            batch = small_model(torch.cat([x_a, x_b]))
        torch.testing.assert_close(batch[0:1], solo_a)
        torch.testing.assert_close(batch[1:2], solo_b)
