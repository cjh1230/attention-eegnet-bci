"""Tests for models/eegnet.py — EEGNet architecture and lazy classifier."""
import numpy as np
import torch

from models.eegnet import EEGNet


class TestEEGNetInit:
    def test_default_constructor(self):
        model = EEGNet()
        assert model.n_channels == 16
        assert model.n_classes == 3
        assert model.classifier is None  # lazy

    def test_custom_params(self):
        model = EEGNet(n_channels=8, n_classes=2, F1=4, D=1, F2=8, dropout=0.3)
        assert model.F1 == 4
        assert model.D == 1
        assert model.F2 == 8

    def test_classifier_none_before_forward(self):
        model = EEGNet()
        assert model.classifier is None


class TestEEGNetForward:
    def test_forward_builds_classifier(self, small_eegnet):
        x = torch.randn(2, 8, 500)
        small_eegnet.eval()
        with torch.no_grad():
            out = small_eegnet(x)
        assert small_eegnet.classifier is not None
        assert out.shape == (2, 3)  # (batch, n_classes)

    def test_forward_with_channel_dim(self):
        """Input already has channel dim: (B, 1, C, T)."""
        model = EEGNet(n_channels=16, n_classes=3)
        model.eval()
        x = torch.randn(2, 1, 16, 500)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)

    def test_output_sum_is_not_zero(self, small_eegnet):
        """Sanity: logits shouldn't be all zeros."""
        x = torch.randn(1, 8, 500)
        small_eegnet.eval()
        with torch.no_grad():
            out = small_eegnet(x)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_different_inputs_different_outputs(self, small_eegnet):
        x1 = torch.randn(1, 8, 500)
        x2 = torch.randn(1, 8, 500)
        small_eegnet.eval()
        with torch.no_grad():
            o1 = small_eegnet(x1)
            o2 = small_eegnet(x2)
        assert not torch.allclose(o1, o2)

    def test_different_time_lengths_separate_models(self):
        """Each EEGNet instance adapts to its first-seen T (lazy classifier).
        Different T lengths require separate model instances."""
        # Short model
        model_short = EEGNet(n_channels=4, n_classes=2, F1=4, D=2, F2=8)
        model_short.eval()
        with torch.no_grad():
            out_short = model_short(torch.randn(1, 4, 250))
        assert out_short.shape == (1, 2)

        # Long model — fresh instance
        model_long = EEGNet(n_channels=4, n_classes=2, F1=4, D=2, F2=8)
        model_long.eval()
        with torch.no_grad():
            out_long = model_long(torch.randn(1, 4, 750))
        assert out_long.shape == (1, 2)

    def test_same_t_consistent(self):
        """Multiple forwards with same T should work once classifier is built."""
        model = EEGNet(n_channels=4, n_classes=2, F1=4, D=2, F2=8)
        model.eval()
        with torch.no_grad():
            o1 = model(torch.randn(1, 4, 500))
            o2 = model(torch.randn(1, 4, 500))
        assert o1.shape == o2.shape == (1, 2)

    def test_batch_independence(self, small_eegnet):
        """Output for sample i should not depend on other samples in batch."""
        x = torch.randn(4, 8, 500)
        small_eegnet.eval()
        with torch.no_grad():
            out_full = small_eegnet(x)
            out_single = small_eegnet(x[0:1])
        assert torch.allclose(out_full[0:1], out_single, atol=1e-5)


class TestCheckpointRoundTrip:
    def test_save_load_roundtrip(self, small_eegnet, tmp_path):
        """Save checkpoint → load → same predictions."""
        # Build classifier
        dummy = torch.zeros(1, 8, 500)
        small_eegnet.eval()
        with torch.no_grad():
            ref_out = small_eegnet(dummy)

        # Save
        ckpt = {
            "epoch": 10,
            "state_dict": small_eegnet.state_dict(),
            "opt": torch.optim.Adam(small_eegnet.parameters()).state_dict(),
            "acc": 0.75,
            "config": {"n_channels": 8, "n_classes": 3, "n_times": 500,
                        "F1": small_eegnet.F1, "D": small_eegnet.D,
                        "F2": small_eegnet.F2, "dropout": 0.5},
        }
        ckpt_path = tmp_path / "test.pt"
        torch.save(ckpt, ckpt_path)

        # Load
        from training.train_eegnet import load_checkpoint
        loaded = load_checkpoint(str(ckpt_path), device="cpu")
        loaded.eval()
        with torch.no_grad():
            new_out = loaded(dummy)
        assert torch.allclose(ref_out, new_out, atol=1e-5)


class TestEEGNetTrainingMode:
    def test_gradient_flows(self, small_eegnet):
        """One forward-backward pass should produce gradients."""
        x = torch.randn(2, 8, 500)
        y = torch.tensor([0, 1], dtype=torch.long)
        opt = torch.optim.SGD(small_eegnet.parameters(), lr=0.01)
        small_eegnet.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(small_eegnet(x), y)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in small_eegnet.parameters()
                       if p.grad is not None]
        assert any(g > 0 for g in grad_norms), "No gradients flowing"
