"""Tests for models/er_mi.py — ER-MI architecture and multi-step reasoning."""
import numpy as np
import torch

from models.er_mi import ERMI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(batch=4, channels=8, times=750):
    """Create a random EEG batch with 8 channels (motor8 default)."""
    return torch.randn(batch, channels, times)


def _make_labels(batch=4, n_classes=2):
    """Create random integer labels."""
    return torch.randint(0, n_classes, (batch,))


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestERMIInit:
    def test_default_constructor(self):
        model = ERMI()
        assert model.n_channels == 8
        assert model.n_classes == 2
        assert model.F1 == 8
        assert model.D == 2
        assert model.F2 == 16
        assert model.hidden_dim == 64
        assert model.steps == 3
        assert model.evidence_proj is None  # lazy

    def test_custom_params(self):
        model = ERMI(
            n_channels=4, n_classes=3, F1=4, D=1, F2=8,
            hidden_dim=32, steps=5, dropout=0.3,
        )
        assert model.F1 == 4
        assert model.D == 1
        assert model.F2 == 8
        assert model.hidden_dim == 32
        assert model.steps == 5
        assert model.n_classes == 3

    def test_evidence_proj_none_before_forward(self):
        model = ERMI()
        assert model.evidence_proj is None

    def test_gru_cell_exists(self):
        model = ERMI()
        assert isinstance(model.gru_cell, torch.nn.GRUCell)
        assert model.gru_cell.input_size == 64
        assert model.gru_cell.hidden_size == 64

    def test_step_classifier_exists(self):
        model = ERMI()
        assert isinstance(model.step_classifier, torch.nn.Linear)
        assert model.step_classifier.out_features == 2


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


class TestERMIForward:
    def test_eval_mode_returns_tensor(self):
        """In eval mode, forward returns a single tensor (final logits)."""
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, torch.Tensor), \
            f"Expected Tensor in eval mode, got {type(out)}"
        assert out.shape == (4, 2)

    def test_train_mode_returns_list(self):
        """In training mode, forward returns a list of logits per step."""
        model = ERMI(n_channels=8, n_classes=2, steps=3)
        model.train()
        x = _make_batch(4)
        out = model(x)
        assert isinstance(out, list), \
            f"Expected list in train mode, got {type(out)}"
        assert len(out) == 3, f"Expected 3 steps, got {len(out)}"
        for i, logits in enumerate(out):
            assert isinstance(logits, torch.Tensor), \
                f"Step {i}: expected Tensor, got {type(logits)}"
            assert logits.shape == (4, 2), \
                f"Step {i}: expected (4, 2), got {logits.shape}"

    def test_eval_forward_builds_evidence_proj(self):
        model = ERMI(n_channels=8, n_classes=2)
        assert model.evidence_proj is None
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            model(x)
        assert model.evidence_proj is not None

    def test_forward_with_channel_dim(self):
        """Input already has channel dim: (B, 1, C, T)."""
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = torch.randn(4, 1, 8, 750)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_output_not_zero(self):
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(1)
        with torch.no_grad():
            out = model(x)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_different_inputs_different_outputs(self):
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x1 = _make_batch(1)
        x2 = _make_batch(1)
        with torch.no_grad():
            o1 = model(x1)
            o2 = model(x2)
        assert not torch.allclose(o1, o2)

    def test_batch_independence(self):
        """Output for sample i should not depend on other samples in batch."""
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out_full = model(x)
            out_single = model(x[0:1])
        assert torch.allclose(out_full[0:1], out_single, atol=1e-5)

    def test_deterministic_eval(self):
        """Eval mode should produce identical outputs for identical inputs."""
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            o1 = model(x)
            o2 = model(x)
        torch.testing.assert_close(o1, o2)

    def test_no_nan(self):
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_no_nan_constant_input(self):
        """Constant input should not produce NaN."""
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = torch.ones(4, 8, 750)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_softmax_yields_probabilities(self):
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Steps ablation
# ---------------------------------------------------------------------------


class TestERMISteps:
    def test_steps_1_eval(self):
        model = ERMI(n_channels=8, n_classes=2, steps=1)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_steps_1_train(self):
        model = ERMI(n_channels=8, n_classes=2, steps=1)
        model.train()
        x = _make_batch(4)
        out = model(x)
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0].shape == (4, 2)

    def test_steps_5_train(self):
        model = ERMI(n_channels=8, n_classes=2, steps=5)
        model.train()
        x = _make_batch(4)
        out = model(x)
        assert len(out) == 5
        for logits in out:
            assert logits.shape == (4, 2)

    def test_steps_5_eval(self):
        model = ERMI(n_channels=8, n_classes=2, steps=5)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_steps_produce_different_outputs(self):
        """Each reasoning step should produce different logits."""
        model = ERMI(n_channels=8, n_classes=2, steps=3)
        model.train()
        x = _make_batch(4)
        out = model(x)
        # Steps should differ from each other (evidence is being refined)
        assert not torch.allclose(out[0], out[1], atol=1e-4), \
            "Step 1 and Step 2 should produce different logits"
        assert not torch.allclose(out[1], out[2], atol=1e-4), \
            "Step 2 and Step 3 should produce different logits"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestERMIGradient:
    def test_gradient_flows(self):
        """One forward-backward pass should produce gradients on all params."""
        model = ERMI(n_channels=8, n_classes=2, steps=3)
        x = _make_batch(4)
        y = _make_labels(4, n_classes=2)
        model.train()
        out = model(x)
        # Accumulate loss across all steps (as in training)
        loss = torch.nn.functional.cross_entropy(out[-1], y)
        for step_logits in out[:-1]:
            loss = loss + 0.3 * torch.nn.functional.cross_entropy(step_logits, y)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters()
                      if p.grad is not None]
        assert any(g > 0 for g in grad_norms), "No gradients flowing"

    def test_gradient_updates_parameters(self):
        """Optimizer step should change at least some parameters."""
        model = ERMI(n_channels=8, n_classes=2, steps=3)
        x = _make_batch(4)
        y = _make_labels(4, n_classes=2)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)

        # Snapshot parameters
        params_before = [p.clone().detach() for p in model.parameters()]

        out = model(x)
        loss = torch.nn.functional.cross_entropy(out[-1], y)
        for step_logits in out[:-1]:
            loss = loss + 0.3 * torch.nn.functional.cross_entropy(step_logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # At least some parameters should change after an optimizer step
        changed = 0
        for i, (p_before, p_after) in enumerate(
            zip(params_before, model.parameters())
        ):
            if p_before.shape != p_after.shape:
                continue
            if p_after.grad is None:
                continue
            if not torch.allclose(p_before, p_after, atol=1e-5):
                changed += 1
        assert changed > 0, f"No parameters changed after optimizer step"


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestERMICheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        """Save checkpoint → load → same predictions."""
        model = ERMI(n_channels=8, n_classes=2, steps=3)
        model.eval()
        dummy = torch.randn(1, 8, 750)
        with torch.no_grad():
            ref_out = model(dummy)

        # Save
        ckpt = {
            "epoch": 10,
            "state_dict": model.state_dict(),
            "opt": torch.optim.Adam(model.parameters()).state_dict(),
            "acc": 0.65,
            "model_type": "er_mi",
            "config": {
                "n_channels": 8, "n_classes": 2, "n_times": 750,
                "F1": model.F1, "D": model.D, "F2": model.F2,
                "hidden_dim": model.hidden_dim, "steps": model.steps,
                "dropout": 0.5,
            },
        }
        ckpt_path = tmp_path / "ermi_test.pt"
        torch.save(ckpt, ckpt_path)

        # Load via factory
        from models.eegnet_attn import create_model
        loaded = create_model("er_mi", n_channels=8, n_classes=2, steps=3)
        loaded.eval()
        # Warm up: build lazy layers
        with torch.no_grad():
            loaded(torch.zeros(1, 8, 750))
        loaded.load_state_dict(ckpt["state_dict"])

        loaded.eval()
        with torch.no_grad():
            new_out = loaded(dummy)
        assert torch.allclose(ref_out, new_out, atol=1e-5), \
            "Checkpoint round-trip produced different outputs"


# ---------------------------------------------------------------------------
# Integration via create_model
# ---------------------------------------------------------------------------


class TestERMIIntegration:
    def test_create_model_factory(self):
        from models.eegnet_attn import create_model
        model = create_model(
            "er_mi", n_channels=8, n_classes=2,
            F1=8, D=2, F2=16, hidden_dim=64, steps=3,
        )
        assert isinstance(model, ERMI)
        assert model.n_channels == 8
        assert model.steps == 3

    def test_create_model_train_eval_modes(self):
        from models.eegnet_attn import create_model
        model = create_model("er_mi", n_channels=8, n_classes=2, steps=3)

        # Train mode
        model.train()
        x = _make_batch(4)
        out_train = model(x)
        assert isinstance(out_train, list)
        assert len(out_train) == 3

        # Eval mode
        model.eval()
        with torch.no_grad():
            out_eval = model(x)
        assert isinstance(out_eval, torch.Tensor)
        assert out_eval.shape == (4, 2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestERMIEdgeCases:
    def test_single_sample(self):
        """Model should handle batch_size=1."""
        model = ERMI(n_channels=8, n_classes=2, steps=3)
        model.eval()
        x = _make_batch(1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_large_batch(self):
        """Model should handle larger batches."""
        model = ERMI(n_channels=8, n_classes=2, steps=3)
        model.eval()
        x = _make_batch(32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, 2)

    def test_different_time_length(self):
        """Each instance adapts to its first-seen T (lazy evidence_proj)."""
        model = ERMI(n_channels=8, n_classes=2, steps=3)
        model.eval()
        x = torch.randn(4, 8, 500)  # shorter than default 750
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_binary_classification(self):
        model = ERMI(n_channels=8, n_classes=2)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_multiclass_classification(self):
        model = ERMI(n_channels=8, n_classes=4)
        model.eval()
        x = _make_batch(4)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 4)
