"""Tests for models/mixstyle.py — MixStyle1d / MixStyle2d."""

import torch
import pytest

from models.mixstyle import MixStyle1d, MixStyle2d


class TestMixStyle1d:
    def test_output_shape(self):
        ms = MixStyle1d(p=1.0, alpha=0.2)
        ms.train()
        x = torch.randn(8, 16, 100)
        out = ms(x)
        assert out.shape == x.shape

    def test_identity_in_eval(self):
        ms = MixStyle1d(p=1.0, alpha=0.2)
        ms.eval()
        x = torch.randn(8, 16, 100)
        out = ms(x)
        torch.testing.assert_close(out, x)

    def test_changes_in_train(self):
        ms = MixStyle1d(p=1.0, alpha=0.5)
        ms.train()
        x = torch.randn(8, 16, 100)
        out = ms(x)
        assert not torch.allclose(out, x)

    def test_batch_size_one(self):
        ms = MixStyle1d(p=1.0, alpha=0.2)
        ms.train()
        x = torch.randn(1, 16, 100)
        out = ms(x)
        torch.testing.assert_close(out, x)  # should be no-op

    def test_gradient_flows(self):
        ms = MixStyle1d(p=1.0, alpha=0.2)
        ms.train()
        x = torch.randn(8, 16, 100, requires_grad=True)
        out = ms(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_preserves_no_nan(self):
        ms = MixStyle1d(p=1.0, alpha=0.2)
        ms.train()
        x = torch.randn(8, 16, 100)
        out = ms(x)
        assert not torch.any(torch.isnan(out))


class TestMixStyle2d:
    def test_output_shape(self):
        ms = MixStyle2d(p=1.0, alpha=0.2)
        ms.train()
        x = torch.randn(8, 16, 4, 50)
        out = ms(x)
        assert out.shape == x.shape

    def test_identity_in_eval(self):
        ms = MixStyle2d(p=1.0, alpha=0.2)
        ms.eval()
        x = torch.randn(8, 16, 4, 50)
        out = ms(x)
        torch.testing.assert_close(out, x)

    def test_changes_in_train(self):
        ms = MixStyle2d(p=1.0, alpha=0.5)
        ms.train()
        x = torch.randn(8, 16, 4, 50)
        out = ms(x)
        assert not torch.allclose(out, x)

    def test_batch_size_one(self):
        ms = MixStyle2d(p=1.0, alpha=0.2)
        ms.train()
        x = torch.randn(1, 16, 4, 50)
        out = ms(x)
        torch.testing.assert_close(out, x)
