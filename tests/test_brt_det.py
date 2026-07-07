"""Tests for models/brt_det.py — BRT-Det evidence detector."""
import pytest
import torch
import numpy as np

from models.brt_det import BRTDet


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    """Default BRT-Det v8 — region pool, band gate, objectness agg."""
    return BRTDet(n_channels=8, n_classes=2)


@pytest.fixture
def model_no_region():
    """Channel-mode — keeps 8 channel spatial dim."""
    return BRTDet(n_channels=8, n_classes=2, use_region_pool=False)


@pytest.fixture
def input_batch():
    """Standard filter-bank input: (B, n_bands, C, T)."""
    return torch.randn(4, 6, 8, 750)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


# ── Init ─────────────────────────────────────────────────────────────────────

class TestBRTDetInit:
    def test_default_constructor(self, model):
        assert model.n_channels == 8
        assert model.n_classes == 2
        assert model.n_bands == 6
        assert model.use_region_pool is True
        assert model.use_objectness is True
        assert model.agg_mode == "objectness"
        assert model.input_requires_filter_bank is True

    def test_region_pool_false(self, model_no_region):
        assert model_no_region.use_region_pool is False

    def test_custom_n_classes(self):
        m = BRTDet(n_channels=8, n_classes=4)
        assert m.n_classes == 4

    def test_diff_channels(self):
        m = BRTDet(n_channels=8, n_classes=2, use_diff_channels=True,
                   use_region_pool=False)
        assert m.use_diff_channels is True
        # spatial dim should be 8 + 3 diff pairs = 11
        assert m._spatial_dim == 11

    def test_band_gate(self):
        m = BRTDet(n_channels=8, n_classes=2, use_band_gate=True)
        assert m.use_band_gate is True
        assert hasattr(m, "band_gate_proj")

    def test_temporal_gate(self):
        m = BRTDet(n_channels=8, n_classes=2, use_temporal_gate=True)
        assert m.use_temporal_gate is True
        assert hasattr(m, "temporal_gate_proj")

    def test_topk_config(self):
        m = BRTDet(n_channels=8, n_classes=2, topk=10)
        assert m.topk == 10

    def test_multi_scale(self):
        m = BRTDet(n_channels=8, n_classes=2, multi_scale=True)
        assert m.multi_scale is True


# ── Forward ──────────────────────────────────────────────────────────────────

class TestBRTDetForward:
    def test_output_shape(self, model, input_batch):
        out = model(input_batch)
        assert out.shape == (4, 2)

    def test_single_sample(self, model):
        out = model(torch.randn(1, 6, 8, 750))
        assert out.shape == (1, 2)

    def test_no_region_pool(self, model_no_region, input_batch):
        out = model_no_region(input_batch)
        assert out.shape == (4, 2)

    def test_return_objectness(self, model, input_batch):
        logits, obj = model(input_batch, return_objectness=True)
        assert logits.shape == (4, 2)
        # objectness map: (B, nb, S, T_cells)
        assert obj.shape[0] == 4
        assert obj.shape[1] == 6         # n_bands
        assert obj.shape[2] == 3         # n_regions

    def test_no_region_return_objectness(self, model_no_region, input_batch):
        logits, obj = model_no_region(input_batch, return_objectness=True)
        assert logits.shape == (4, 2)
        assert obj.shape[2] == 8         # n_channels (no region pool)

    def test_deterministic_in_eval(self, model, input_batch):
        model.eval()
        with torch.no_grad():
            o1 = model(input_batch)
            o2 = model(input_batch)
        torch.testing.assert_close(o1, o2)

    def test_different_inputs_different_outputs(self, model):
        model.eval()
        x1 = torch.randn(2, 6, 8, 750)
        x2 = torch.randn(2, 6, 8, 750)
        with torch.no_grad():
            o1 = model(x1)
            o2 = model(x2)
        assert not torch.allclose(o1, o2)

    def test_variable_time_lengths(self, model):
        for T in [200, 400, 750, 1000]:
            x = torch.randn(2, 6, 8, T)
            out = model(x)
            assert out.shape == (2, 2)

    def test_no_nan(self, model, input_batch):
        out = model(input_batch)
        assert not torch.isnan(out).any()

    def test_batch_independence(self, model):
        model.eval()
        x_a = torch.randn(1, 6, 8, 750)
        x_b = torch.randn(1, 6, 8, 750)
        with torch.no_grad():
            solo_a = model(x_a)
            solo_b = model(x_b)
            batch = model(torch.cat([x_a, x_b]))
        torch.testing.assert_close(batch[0:1], solo_a)
        torch.testing.assert_close(batch[1:2], solo_b)

    def test_diff_channels(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, use_diff_channels=True,
                   use_region_pool=False)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)
        assert not torch.isnan(out).any()


# ── Aggregation modes ────────────────────────────────────────────────────────

class TestBRTDetAggModes:
    def test_agg_mean(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, agg_mode="mean")
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_agg_topk(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, agg_mode="topk",
                   agg_topk=10)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_agg_logsumexp(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, agg_mode="logsumexp",
                   agg_tau=0.5)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_agg_softmax_weight(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, agg_mode="softmax_weight")
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_agg_objectness(self, model, input_batch):
        model.eval()
        with torch.no_grad():
            out = model(input_batch)
        assert out.shape == (4, 2)
        assert not torch.isnan(out).any()


# ── Gates ────────────────────────────────────────────────────────────────────

class TestBRTDetGates:
    def test_band_gate_forward(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, use_band_gate=True)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_temporal_gate_forward(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, use_temporal_gate=True)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_both_gates_forward(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, use_band_gate=True,
                   use_temporal_gate=True)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)


# ── Gradient ─────────────────────────────────────────────────────────────────

class TestBRTDetGradient:
    def test_gradient_flows(self, model, input_batch):
        model.train()
        out = model(input_batch)
        loss = out.sum()
        loss.backward()
        no_grad = []
        for name, p in model.named_parameters():
            if p.grad is None:
                no_grad.append(name)
            elif torch.all(p.grad == 0):
                no_grad.append(f"{name} (zero)")
        assert len(no_grad) == 0, f"Params without gradient: {no_grad}"

    def test_gradient_with_objectness(self, model, input_batch):
        model.train()
        logits, obj = model(input_batch, return_objectness=True)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"


# ── Checkpoint ───────────────────────────────────────────────────────────────

class TestBRTDetCheckpoint:
    def test_save_load_roundtrip(self, model, input_batch, tmp_path):
        model.eval()
        with torch.no_grad():
            out_before = model(input_batch)

        ckpt = {"state_dict": model.state_dict()}
        path = tmp_path / "brt_det.pt"
        torch.save(ckpt, path)

        model2 = BRTDet(n_channels=8, n_classes=2)
        model2.eval()
        # Warm-up required: BRT-Det doesn't have lazy FC, but do dummy pass
        # to build any internal buffers
        with torch.no_grad():
            _ = model2(torch.randn(1, 6, 8, 750))
        model2.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        model2.eval()

        with torch.no_grad():
            out_after = model2(input_batch)
        torch.testing.assert_close(out_before, out_after)


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestBRTDetEdgeCases:
    def test_single_band(self):
        m = BRTDet(n_channels=8, n_classes=2, n_bands=1)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 1, 8, 750))
        assert out.shape == (2, 2)

    def test_multi_scale_forward(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, multi_scale=True)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)
        assert not torch.isnan(out).any()

    def test_band_mixer_forward(self, input_batch):
        m = BRTDet(n_channels=8, n_classes=2, use_band_mixer=True)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_custom_time_cells(self):
        m = BRTDet(n_channels=8, n_classes=2, n_time_cells=24)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 6, 8, 750))
        assert out.shape == (2, 2)

    def test_3_class(self):
        m = BRTDet(n_channels=8, n_classes=3)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 6, 8, 750))
        assert out.shape == (2, 3)

    def test_16_channel(self):
        m = BRTDet(n_channels=16, n_classes=2, use_region_pool=False)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 6, 16, 750))
        assert out.shape == (2, 2)
