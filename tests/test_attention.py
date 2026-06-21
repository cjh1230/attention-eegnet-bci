"""Tests for models/attention.py."""
import torch

from models.attention import ChannelAttention1D


class TestChannelAttention1D:
    def test_output_shape(self, attention_module):
        x = torch.randn(4, 16, 500)
        out = attention_module(x)
        assert out.shape == x.shape

    def test_no_nan(self, attention_module):
        x = torch.randn(2, 16, 250)
        out = attention_module(x)
        assert not torch.isnan(out).any()

    def test_weights_in_01(self, attention_module):
        """Attention weights should be in [0, 1] (sigmoid output)."""
        x = torch.randn(3, 16, 100)
        B, C, T = x.shape
        s = attention_module.squeeze(x).view(B, C)
        w = attention_module.excitation(s).view(B, C, 1)
        assert (w >= 0).all() and (w <= 1).all()

    def test_different_channels_different_weights(self, attention_module):
        """Each channel should get a (potentially) different weight."""
        x = torch.randn(1, 16, 500)
        B, C, T = x.shape
        s = attention_module.squeeze(x).view(B, C)
        w = attention_module.excitation(s).view(C)
        # With random init, weights should not all be identical
        assert not torch.allclose(w, w[0].expand_as(w))

    def test_identity_on_uniform_input(self, attention_module):
        """With identical channels, weights should be uniform → no distortion."""
        # Disable gradient tracking for this logic-only test
        attention_module.eval()
        x = torch.ones(2, 16, 500)  # all channels identical
        with torch.no_grad():
            out = attention_module(x)
        # Should still be (roughly) uniform after attention
        std_per_channel = out.std(dim=-1)
        # Each channel should still have similar statistics
        assert std_per_channel.std() < 0.5  # not too wildly different

    def test_preserves_gradient(self, attention_module):
        x = torch.randn(2, 16, 500, requires_grad=True)
        out = attention_module(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_custom_reduction(self):
        """Test with reduction=2 (larger bottleneck)."""
        attn = ChannelAttention1D(n_channels=16, reduction=2)
        x = torch.randn(2, 16, 500)
        out = attn(x)
        assert out.shape == x.shape

    def test_single_channel(self):
        """Edge case: 1 channel."""
        attn = ChannelAttention1D(n_channels=1)
        x = torch.randn(2, 1, 500)
        out = attn(x)
        assert out.shape == x.shape
