"""Tests for models/eegnet_attn.py — EEGNetWithAttention and create_model factory."""
import pytest
import torch

from models.eegnet import EEGNet
from models.eegnet_attn import EEGNetWithAttention, create_model


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def input_batch():
    return torch.randn(4, 8, 750)


@pytest.fixture
def input_batch_16ch():
    return torch.randn(4, 16, 750)


# ── EEGNetWithAttention ─────────────────────────────────────────────────────

class TestEEGNetWithAttentionInit:
    def test_default_init_se(self):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="se")
        assert m.attn_type == "se"
        assert m.n_channels == 8
        assert m.n_classes == 2

    def test_default_init_mhsa(self):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="mhsa", n_heads=4)
        assert m.attn_type == "mhsa"
        assert m._n_channels_after_block1 % 4 == 0

    def test_default_init_temporal(self):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="temporal")
        assert m.attn_type == "temporal"

    def test_default_init_spatiotemporal(self):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="spatiotemporal", n_heads=4)
        assert m.attn_type == "spatiotemporal"

    def test_unknown_attn_type_raises(self):
        with pytest.raises(ValueError, match="Unknown attn_type"):
            EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="nonexistent")

    def test_mhsa_head_divisibility_check(self):
        """D*F1 must be divisible by n_heads."""
        # Default: D=2, F1=8 → D*F1=16, n_heads=3 → 16%3 != 0
        with pytest.raises(AssertionError):
            EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="mhsa", n_heads=3)

    def test_custom_params(self):
        m = EEGNetWithAttention(
            n_channels=16, n_classes=4, F1=12, D=2, F2=24,
            attn_type="se", dropout=0.3, attn_dropout=0.2,
        )
        assert m.n_channels == 16
        assert m.n_classes == 4
        assert m.F1 == 12
        assert m.F2 == 24


class TestEEGNetWithAttentionForward:
    def test_output_shape_se(self, input_batch):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="se")
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_output_shape_mhsa(self, input_batch):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="mhsa", n_heads=4)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_output_shape_temporal(self, input_batch):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="temporal")
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_output_shape_spatiotemporal(self, input_batch):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="spatiotemporal", n_heads=4)
        m.eval()
        with torch.no_grad():
            out = m(input_batch)
        assert out.shape == (4, 2)

    def test_single_sample(self):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="se")
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, 8, 750))
        assert out.shape == (1, 2)

    def test_deterministic_in_eval(self, input_batch):
        for attn_type in ["se", "mhsa", "temporal", "spatiotemporal"]:
            kwargs = {}
            if attn_type in ("mhsa", "spatiotemporal"):
                kwargs["n_heads"] = 4
            m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type=attn_type, **kwargs)
            m.eval()
            with torch.no_grad():
                o1 = m(input_batch)
                o2 = m(input_batch)
            torch.testing.assert_close(o1, o2)

    def test_variable_time_lengths_se(self):
        for T in [200, 400, 750, 1000]:
            m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="se")
            x = torch.randn(2, 8, T)
            out = m(x)
            assert out.shape == (2, 2)

    def test_no_nan(self, input_batch):
        for attn_type in ["se", "mhsa", "temporal", "spatiotemporal"]:
            kwargs = {}
            if attn_type in ("mhsa", "spatiotemporal"):
                kwargs["n_heads"] = 4
            m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type=attn_type, **kwargs)
            m.eval()
            with torch.no_grad():
                out = m(input_batch)
            assert not torch.isnan(out).any(), f"NaN in {attn_type}"

    def test_16_channel_input(self, input_batch_16ch):
        m = EEGNetWithAttention(n_channels=16, n_classes=2, attn_type="se")
        m.eval()
        with torch.no_grad():
            out = m(input_batch_16ch)
        assert out.shape == (4, 2)

    def test_3_class(self):
        m = EEGNetWithAttention(n_channels=8, n_classes=3, attn_type="se")
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(2, 8, 750))
        assert out.shape == (2, 3)


class TestEEGNetWithAttentionGradient:
    def test_gradient_flows_se(self, input_batch):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="se")
        m.train()
        out = m(input_batch)
        loss = out.sum()
        loss.backward()
        no_grad = [n for n, p in m.named_parameters() if p.grad is None]
        assert len(no_grad) == 0, f"Params without gradient: {no_grad}"

    def test_gradient_flows_mhsa(self, input_batch):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="mhsa", n_heads=4)
        m.train()
        out = m(input_batch)
        loss = out.sum()
        loss.backward()
        no_grad = [n for n, p in m.named_parameters() if p.grad is None]
        assert len(no_grad) == 0, f"Params without gradient: {no_grad}"

    def test_gradient_flows_spatiotemporal(self, input_batch):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="spatiotemporal", n_heads=4)
        m.train()
        out = m(input_batch)
        loss = out.sum()
        loss.backward()
        no_grad = [n for n, p in m.named_parameters() if p.grad is None]
        assert len(no_grad) == 0, f"Params without gradient: {no_grad}"


class TestEEGNetWithAttentionCheckpoint:
    def test_save_load_roundtrip(self, input_batch, tmp_path):
        m = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="se")
        m.eval()
        with torch.no_grad():
            out_before = m(input_batch)

        path = tmp_path / "eegnet_attn.pt"
        torch.save(m.state_dict(), path)

        m2 = EEGNetWithAttention(n_channels=8, n_classes=2, attn_type="se")
        m2.eval()
        with torch.no_grad():
            _ = m2(torch.randn(1, 8, 750))  # warm-up
        m2.load_state_dict(torch.load(path, weights_only=True))
        with torch.no_grad():
            out_after = m2(input_batch)
        torch.testing.assert_close(out_before, out_after)


# ── create_model factory ────────────────────────────────────────────────────

class TestCreateModel:
    def test_eegnet(self):
        m = create_model("eegnet", n_channels=8, n_classes=3)
        assert isinstance(m, EEGNet)
        assert m.n_classes == 3

    def test_eegnet_se(self):
        m = create_model("eegnet_se", n_channels=8, n_classes=2)
        assert isinstance(m, EEGNetWithAttention)
        assert m.attn_type == "se"

    def test_eegnet_mhsa(self):
        m = create_model("eegnet_mhsa", n_channels=8, n_classes=2, n_heads=4)
        assert isinstance(m, EEGNetWithAttention)
        assert m.attn_type == "mhsa"

    def test_eegnet_temporal(self):
        m = create_model("eegnet_temporal", n_channels=8, n_classes=2)
        assert isinstance(m, EEGNetWithAttention)
        assert m.attn_type == "temporal"

    def test_eegnet_spatiotemporal(self):
        m = create_model("eegnet_spatiotemporal", n_channels=8, n_classes=2, n_heads=4)
        assert isinstance(m, EEGNetWithAttention)
        assert m.attn_type == "spatiotemporal"

    def test_fbcnet(self):
        from models.fbcnet import FBCNet
        m = create_model("fbcnet", n_channels=8, n_classes=2)
        assert isinstance(m, FBCNet)

    def test_eeg_tcnet(self):
        from models.eeg_tcnet import EEGTCNet
        m = create_model("eeg_tcnet", n_channels=8, n_classes=2)
        assert isinstance(m, EEGTCNet)

    def test_eeg_conformer(self):
        from models.eeg_conformer import EEGConformer
        m = create_model("eeg_conformer", n_channels=8, n_classes=2)
        assert isinstance(m, EEGConformer)

    def test_fb_maa_eegnet(self):
        from models.fb_maa_eegnet import FBMAAEEGNet
        m = create_model("fb_maa_eegnet", n_channels=8, n_classes=2)
        assert isinstance(m, FBMAAEEGNet)

    def test_maa_eegnet(self):
        from models.maa_eegnet import MAAEEGNet
        m = create_model("maa_eegnet", n_channels=8, n_classes=2)
        assert isinstance(m, MAAEEGNet)

    def test_maa_eegnet_pre(self):
        from models.maa_eegnet_pre import MAAEEGNetPre
        m = create_model("maa_eegnet_pre", n_channels=8, n_classes=2)
        assert isinstance(m, MAAEEGNetPre)

    def test_fb_tcnet(self):
        from models.fb_tcnet import FBTCNet
        m = create_model("fb_tcnet", n_channels=8, n_classes=2)
        assert isinstance(m, FBTCNet)

    def test_spdnet(self):
        from models.spd_models import SPDNetModel
        m = create_model("spdnet", n_channels=8, n_classes=2)
        assert isinstance(m, SPDNetModel)

    def test_er_mi(self):
        from models.er_mi import ERMI
        m = create_model("er_mi", n_channels=8, n_classes=2)
        assert isinstance(m, ERMI)

    def test_er_mi_v2(self):
        from models.er_mi_v2 import ERMIv2
        m = create_model("er_mi_v2", n_channels=8, n_classes=2)
        assert isinstance(m, ERMIv2)

    def test_brt_det(self):
        from models.brt_det import BRTDet
        m = create_model("brt_det", n_channels=8, n_classes=2)
        assert isinstance(m, BRTDet)

    def test_forward_all_models(self):
        """Every model created by the factory should produce the right output shape."""
        x = torch.randn(2, 8, 750)
        x_bands = torch.randn(2, 6, 8, 750)  # for filter-bank models
        x_3ch = torch.randn(2, 1, 8, 750)    # EEGNetWithAttention needs 4D internally

        models_3d = {
            "eegnet": (2, 3, False),
            "maa_eegnet": (2, 2, False),
            "maa_eegnet_pre": (2, 2, False),
            "eeg_tcnet": (2, 2, False),
            "eeg_conformer": (2, 2, False),
            "er_mi": (2, 2, False),
            "er_mi_v2": (2, 2, False),
        }
        models_4d_attn = {
            "eegnet_se": (2, 2, False),
            "eegnet_mhsa": (2, 2, False),
            "eegnet_temporal": (2, 2, False),
            "eegnet_spatiotemporal": (2, 2, False),
        }
        models_fb = {
            "fbcnet": False,
            "fb_maa_eegnet": False,
            "fb_tcnet": False,
            "brt_det": False,
        }

        for name, (B, n_cls, _) in models_3d.items():
            m = create_model(name, n_channels=8, n_classes=n_cls)
            m.eval()
            with torch.no_grad():
                out = m(x)
            assert out.shape == (B, n_cls), f"{name}: {out.shape} != ({B}, {n_cls})"

        for name, (B, n_cls, _) in models_4d_attn.items():
            # EEGNetWithAttention handles 3D→4D internally
            m = create_model(name, n_channels=8, n_classes=n_cls)
            m.eval()
            with torch.no_grad():
                out = m(x)
            assert out.shape == (B, n_cls), f"{name}: {out.shape} != ({B}, {n_cls})"

        for name, _ in models_fb.items():
            m = create_model(name, n_channels=8, n_classes=2)
            m.eval()
            with torch.no_grad():
                out = m(x_bands)
            assert out.shape == (2, 2), f"{name}: {out.shape} != (2, 2)"

    def test_eegnet_passes_kwargs(self):
        m = create_model("eegnet", n_channels=8, n_classes=3, F1=12, D=3, dropout=0.3)
        assert m.F1 == 12
        assert m.D == 3

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model("nonexistent_model", n_channels=8, n_classes=2)
