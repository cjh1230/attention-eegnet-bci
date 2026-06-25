"""Tests for utils/domain_adapt.py — Center Loss + MMD."""

import numpy as np
import pytest
import torch

from utils.domain_adapt import center_loss, mmd_rbf, multi_kernel_mmd


class TestCenterLoss:
    def test_loss_positive(self):
        feats = torch.randn(20, 16)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                               1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        loss, _ = center_loss(feats, labels, n_classes=3)
        assert loss.item() > 0

    def test_loss_decreases_with_identical_features(self):
        """Center loss should decrease as centers converge to features."""
        D = 8
        feats = torch.zeros(12, D)
        feats[0::3] = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0])
        feats[1::3] = torch.tensor([0, 1.0, 0, 0, 0, 0, 0, 0])
        feats[2::3] = torch.tensor([0, 0, 1.0, 0, 0, 0, 0, 0])
        labels = torch.tensor([0, 1, 2] * 4)
        # First call: centers start at zero, loss is high
        loss1, centers = center_loss(feats, labels, n_classes=3)
        # Second call: centers have moved toward features, loss should drop
        loss2, _ = center_loss(feats, labels, n_classes=3, centers=centers)
        assert loss2.item() < loss1.item()

    def test_centers_update(self):
        feats = torch.randn(16, 8)
        labels = torch.tensor([0, 1, 0, 1] * 4)
        _, centers = center_loss(feats, labels, n_classes=2)
        assert centers.shape == (2, 8)
        # Centers should not be all zeros after update
        assert not torch.allclose(centers, torch.zeros_like(centers))

    def test_n_classes_dynamic(self):
        for nc in [2, 3, 5]:
            feats = torch.randn(10 * nc, 8)
            labels = torch.arange(nc).repeat(10)
            loss, centers = center_loss(feats, labels, n_classes=nc)
            assert centers.shape == (nc, 8)
            assert loss.item() > 0

    def test_gradient_flows(self):
        feats = torch.randn(10, 8, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        loss, _ = center_loss(feats, labels, n_classes=2)
        loss.backward()
        assert feats.grad is not None
        assert not torch.all(feats.grad == 0)


class TestMMD:
    def test_mmd_same_distribution_near_zero(self):
        """MMD between two samples from the same distribution ≈ 0."""
        rng = torch.Generator().manual_seed(42)
        x = torch.randn(50, 16, generator=rng)
        y = torch.randn(50, 16, generator=rng)
        val = mmd_rbf(x, y, sigma=2.0)
        # Should be close to zero for identical distributions
        assert abs(val.item()) < 0.1

    def test_mmd_different_distribution_positive(self):
        """MMD between different distributions should be > 0."""
        x = torch.randn(50, 16) * 0.5  # narrow
        y = torch.randn(50, 16) * 3.0  # wide
        val = mmd_rbf(x, y, sigma=2.0)
        assert val.item() > 0

    def test_multi_kernel_mmd(self):
        x = torch.randn(30, 8)
        y = torch.randn(30, 8) * 2.0
        val = multi_kernel_mmd(x, y, sigmas=[1.0, 2.0, 4.0])
        # Should be a positive scalar
        assert val.item() > 0
        assert val.ndim == 0

    def test_mmd_gradient(self):
        x = torch.randn(20, 8, requires_grad=True)
        y = torch.randn(20, 8)
        val = mmd_rbf(x, y, sigma=2.0)
        val.backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
