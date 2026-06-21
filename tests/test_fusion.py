"""Tests for models/fusion.py — MultiBandFusion."""
import torch

from models.fusion import MultiBandFusion


class TestMultiBandFusion:
    def test_output_shape(self, fusion_model):
        B, C, T = 4, 8, 500
        mu = torch.randn(B, C, T)
        beta = torch.randn(B, C, T)
        full = torch.randn(B, C, T)
        out = fusion_model(mu, beta, full)
        assert out.shape == (B, 3)  # (batch, n_classes)

    def test_different_bands_different_features(self, fusion_model):
        """Feeding different data to each band should matter."""
        B, C, T = 2, 8, 500
        mu1 = torch.randn(B, C, T)
        beta1 = torch.randn(B, C, T)
        full1 = torch.randn(B, C, T)

        mu2 = torch.randn(B, C, T)
        beta2 = torch.randn(B, C, T)
        full2 = torch.randn(B, C, T)

        with torch.no_grad():
            out1 = fusion_model(mu1, beta1, full1)
            out2 = fusion_model(mu2, beta2, full2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows(self, fusion_model):
        B, C, T = 2, 8, 500
        mu = torch.randn(B, C, T)
        beta = torch.randn(B, C, T)
        full = torch.randn(B, C, T)
        y = torch.tensor([0, 1], dtype=torch.long)

        opt = torch.optim.SGD(fusion_model.parameters(), lr=0.01)
        fusion_model.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(fusion_model(mu, beta, full), y)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in fusion_model.parameters()
                       if p.grad is not None]
        assert any(g > 0 for g in grad_norms)

    def test_custom_classes(self):
        """Test with different number of output classes."""
        model = MultiBandFusion(n_channels=4, n_classes=5, hidden=16)
        mu = torch.randn(2, 4, 300)
        out = model(mu, mu, mu)  # same input for all bands
        assert out.shape == (2, 5)
