"""Tests for MAA-EEGNet and MAA-EEGNet-Pre models."""
import pytest
import torch

from models.maa_eegnet import MAAEEGNet
from models.maa_eegnet_pre import MAAEEGNetPre


@pytest.fixture
def input_batch():
    return torch.randn(4, 8, 500)


class TestMAAEEGNet:
    def test_default_init(self):
        m = MAAEEGNet()
        assert m.n_channels == 8
        assert m.n_classes == 2
        assert m.input_requires_filter_bank is False

    def test_output_shape(self, input_batch):
        m = MAAEEGNet(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_3d_input(self):
        """Should accept (B, C, T) without explicit channel dim."""
        m = MAAEEGNet(n_channels=8, n_classes=2)
        m.eval()
        x = torch.randn(2, 8, 500)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 2)

    def test_deterministic_in_eval(self, input_batch):
        m = MAAEEGNet(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            o1 = m(input_batch)
            o2 = m(input_batch)
        torch.testing.assert_close(o1, o2)

    def test_different_inputs_different_outputs(self):
        m = MAAEEGNet(n_channels=8, n_classes=2)
        m.eval()
        x1 = torch.randn(2, 8, 500)
        x2 = torch.randn(2, 8, 500)
        with torch.no_grad():
            o1 = m(x1)
            o2 = m(x2)
        assert not torch.allclose(o1, o2)

    def test_variable_time_lengths(self):
        """Lazy classifier is fixed once built — recreate model per T."""
        for T in [200, 400, 750]:
            m = MAAEEGNet(n_channels=8, n_classes=2)
            x = torch.randn(2, 8, T)
            out = m(x)
            assert out.shape == (2, 2)

    def test_no_nan(self, input_batch):
        m = MAAEEGNet(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert not torch.isnan(out).any()

    def test_gradient_flows(self, input_batch):
        m = MAAEEGNet(n_channels=8, n_classes=2)
        m.train()
        out = m(input_batch)
        loss = out.sum()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No grad: {name}"

    def test_save_load_roundtrip(self, input_batch, tmp_path):
        m = MAAEEGNet(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            out_before = m(input_batch)

        path = tmp_path / "maa_eegnet.pt"
        torch.save(m.state_dict(), path)

        m2 = MAAEEGNet(n_channels=8, n_classes=2)
        m2.eval()
        with torch.no_grad():
            _ = m2(torch.randn(1, 8, 500))  # warm-up
        m2.load_state_dict(torch.load(path, weights_only=True))
        with torch.no_grad():
            out_after = m2(input_batch)
        torch.testing.assert_close(out_before, out_after)

    def test_3_class(self):
        m = MAAEEGNet(n_channels=8, n_classes=3)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 8, 500))
        assert out.shape == (2, 3)


class TestMAAEEGNetPre:
    def test_default_init(self):
        m = MAAEEGNetPre()
        assert hasattr(m, "maa")
        assert hasattr(m, "eegnet")
        assert m.input_requires_filter_bank is False

    def test_output_shape(self, input_batch):
        m = MAAEEGNetPre(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_3d_input(self):
        m = MAAEEGNetPre(n_channels=8, n_classes=2)
        m.eval()
        x = torch.randn(2, 8, 500)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 2)

    def test_deterministic(self, input_batch):
        m = MAAEEGNetPre(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            o1 = m(input_batch)
            o2 = m(input_batch)
        torch.testing.assert_close(o1, o2)

    def test_gradient_flows(self, input_batch):
        m = MAAEEGNetPre(n_channels=8, n_classes=2)
        m.train()
        out = m(input_batch)
        loss = out.sum()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, f"No grad: {name}"

    def test_no_nan(self, input_batch):
        m = MAAEEGNetPre(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert not torch.isnan(out).any()

    def test_maa_not_identity(self, input_batch):
        """MAA should alter the input before EEGNet sees it."""
        m = MAAEEGNetPre(n_channels=8, n_classes=2)
        m.eval()
        with torch.no_grad():
            maa_out = m.maa(input_batch)
        # MAA output should differ from input
        assert not torch.allclose(maa_out, input_batch)
