"""Tests for models/er_mi_v2.py — ER-MI-v2 with multi-token evidence."""
import torch

from models.er_mi_v2 import ERMIv2, EvidenceTokenizer


def _make_batch(batch=4, channels=8, times=750):
    return torch.randn(batch, channels, times)


# ---------------------------------------------------------------------------
# EvidenceTokenizer
# ---------------------------------------------------------------------------


class TestEvidenceTokenizer:
    def test_output_shape(self):
        tokenizer = EvidenceTokenizer(in_channels=16, hidden_dim=64)
        x = torch.randn(4, 16, 187)  # typical post-Block1 shape
        tokens = tokenizer(x)
        assert tokens.shape == (4, 4, 64), f"Expected (4,4,64), got {tokens.shape}"

    def test_tokens_differ(self):
        """Different token branches should produce different outputs."""
        tokenizer = EvidenceTokenizer(in_channels=16, hidden_dim=64)
        x = torch.randn(4, 16, 187)
        tokens = tokenizer(x)
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(tokens[:, i, :], tokens[:, j, :], atol=1e-4), \
                    f"Token {i} and {j} should differ"

    def test_deterministic(self):
        tokenizer = EvidenceTokenizer(in_channels=16, hidden_dim=64)
        tokenizer.eval()
        x = torch.randn(4, 16, 187)
        with torch.no_grad():
            t1 = tokenizer(x)
            t2 = tokenizer(x)
        torch.testing.assert_close(t1, t2)


# ---------------------------------------------------------------------------
# ERMIv2 Init
# ---------------------------------------------------------------------------


class TestERMIv2Init:
    def test_default_constructor(self):
        model = ERMIv2()
        assert model.n_channels == 8
        assert model.n_classes == 2
        assert model.steps == 3
        assert model.hidden_dim == 64

    def test_custom_params(self):
        model = ERMIv2(
            n_channels=4, n_classes=3, F1=4, D=1,
            hidden_dim=32, steps=5, dropout=0.3,
        )
        assert model.F1 == 4
        assert model.hidden_dim == 32
        assert model.steps == 5


# ---------------------------------------------------------------------------
# ERMIv2 Forward
# ---------------------------------------------------------------------------


class TestERMIv2Forward:
    def test_eval_mode_returns_tensor(self):
        model = ERMIv2(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 2)

    def test_train_mode_returns_list(self):
        model = ERMIv2(n_channels=8, n_classes=2, steps=3)
        model.train()
        x = _make_batch(4)
        out = model(x)
        assert isinstance(out, list)
        assert len(out) == 3
        for logits in out:
            assert logits.shape == (4, 2)

    def test_return_all_steps_in_eval(self):
        model = ERMIv2(n_channels=8, n_classes=2, steps=3)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x, return_all_steps=True)
        assert isinstance(out, list)
        assert len(out) == 3

    def test_forward_with_channel_dim(self):
        model = ERMIv2(n_channels=8, n_classes=2)
        model.eval()
        x = torch.randn(4, 1, 8, 750)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_no_nan(self):
        model = ERMIv2(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_batch_independence(self):
        model = ERMIv2(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out_full = model(x)
            out_single = model(x[0:1])
        assert torch.allclose(out_full[0:1], out_single, atol=1e-5)

    def test_deterministic_eval(self):
        model = ERMIv2(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        torch.testing.assert_close(o1, o2)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


class TestERMIv2Steps:
    def test_steps_1(self):
        model = ERMIv2(n_channels=8, n_classes=2, steps=1)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(4))
        assert out.shape == (4, 2)

    def test_steps_5(self):
        model = ERMIv2(n_channels=8, n_classes=2, steps=5)
        model.train()
        out = model(_make_batch(4))
        assert len(out) == 5


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestERMIv2Gradient:
    def test_gradient_flows(self):
        model = ERMIv2(n_channels=8, n_classes=2, steps=3)
        x = _make_batch(4)
        y = torch.randint(0, 2, (4,))
        model.train()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out[-1], y)
        for step_logits in out[:-1]:
            loss = loss + 0.3 * torch.nn.functional.cross_entropy(step_logits, y)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters()
                      if p.grad is not None]
        assert any(g > 0 for g in grad_norms), "No gradients flowing"

    def test_gradient_updates_parameters(self):
        model = ERMIv2(n_channels=8, n_classes=2, steps=3)
        x = _make_batch(4)
        y = torch.randint(0, 2, (4,))
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        params_before = [p.clone().detach() for p in model.parameters()]
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out[-1], y)
        for step_logits in out[:-1]:
            loss = loss + 0.3 * torch.nn.functional.cross_entropy(step_logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        changed = sum(
            1 for pb, pa in zip(params_before, model.parameters())
            if pb.shape == pa.shape and pa.grad is not None
            and not torch.allclose(pb, pa, atol=1e-5)
        )
        assert changed > 0, "No parameters changed"


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestERMIv2Checkpoint:
    def test_save_load_roundtrip(self, tmp_path):
        model = ERMIv2(n_channels=8, n_classes=2, steps=3)
        model.eval()
        dummy = torch.randn(1, 8, 750)
        with torch.no_grad():
            ref_out = model(dummy)

        ckpt = {
            "epoch": 10,
            "state_dict": model.state_dict(),
            "opt": torch.optim.Adam(model.parameters()).state_dict(),
            "acc": 0.65,
            "model_type": "er_mi_v2",
            "config": {
                "n_channels": 8, "n_classes": 2, "n_times": 750,
                "F1": model.F1, "D": model.D,
                "hidden_dim": model.hidden_dim, "steps": model.steps,
                "dropout": 0.5,
            },
        }
        ckpt_path = tmp_path / "ermi_v2.pt"
        torch.save(ckpt, ckpt_path)

        from models.eegnet_attn import create_model
        loaded = create_model("er_mi_v2", n_channels=8, n_classes=2, steps=3)
        loaded.eval()
        with torch.no_grad():
            loaded(torch.zeros(1, 8, 750))
        loaded.load_state_dict(ckpt["state_dict"])

        loaded.eval()
        with torch.no_grad():
            new_out = loaded(dummy)
        assert torch.allclose(ref_out, new_out, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestERMIv2Integration:
    def test_create_model_factory(self):
        from models.eegnet_attn import create_model
        model = create_model("er_mi_v2", n_channels=8, n_classes=2)
        assert isinstance(model, ERMIv2)
        assert model.steps == 3

    def test_train_eval_modes(self):
        from models.eegnet_attn import create_model
        model = create_model("er_mi_v2", n_channels=8, n_classes=2, steps=3)
        model.train()
        out = model(_make_batch(4))
        assert isinstance(out, list)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(4))
        assert isinstance(out, torch.Tensor)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestERMIv2EdgeCases:
    def test_single_sample(self):
        model = ERMIv2(n_channels=8, n_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(1))
        assert out.shape == (1, 2)

    def test_large_batch(self):
        model = ERMIv2(n_channels=8, n_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(32))
        assert out.shape == (32, 2)

    def test_binary(self):
        model = ERMIv2(n_channels=8, n_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(4))
        assert out.shape == (4, 2)

    def test_multiclass(self):
        model = ERMIv2(n_channels=8, n_classes=4)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(4))
        assert out.shape == (4, 4)
