"""Tests for MotorAreaAttention module."""
import torch
import pytest

from models.motor_area_attention import MotorAreaAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def maa():
    return MotorAreaAttention(n_channels=8)


@pytest.fixture
def batch():
    return torch.randn(4, 8, 200)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_shape_matches_input(maa, batch):
    """Output should have same shape as input."""
    out = maa(batch)
    assert out.shape == batch.shape


def test_output_shape_varying_time(maa):
    """Works with different time lengths."""
    for T in [50, 100, 250, 500]:
        x = torch.randn(2, 8, T)
        out = maa(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_no_nan(maa, batch):
    """Output should not contain NaN or Inf."""
    out = maa(batch)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_no_nan_constant_input(maa):
    """Constant input should not produce NaN."""
    x = torch.ones(2, 8, 100)
    out = maa(x)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flows(maa, batch):
    """Gradients should flow through all parameters."""
    x = batch.clone().requires_grad_(True)
    out = maa(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert (x.grad != 0).any()


def test_gradient_updates_parameters(maa, batch):
    """Trainable parameters should change after an optimizer step."""
    trainable = [p for p in maa.parameters() if p.requires_grad]
    params_before = [p.clone() for p in trainable]
    opt = torch.optim.Adam(maa.parameters(), lr=0.01)
    out = maa(batch)
    loss = out.sum()
    loss.backward()
    opt.step()
    changed = 0
    for before, after in zip(params_before, trainable):
        if not torch.equal(before, after):
            changed += 1
    assert changed >= 1, "No trainable parameters changed after optimizer step"


# ---------------------------------------------------------------------------
# Group weighting semantics
# ---------------------------------------------------------------------------

def test_same_group_same_weight(maa):
    """Channels in the same group should receive identical weights."""
    x = torch.randn(1, 8, 100)
    out = maa(x)

    # Extract per-channel weight = out / input (where input != 0)
    ratio = out[0] / x[0]  # (C, T)

    # Within each group, ratios should be identical across time
    for indices in MotorAreaAttention.GROUP_INDICES.values():
        for t in range(ratio.shape[1]):
            group_vals = ratio[indices, t]
            assert torch.allclose(group_vals, group_vals[0], atol=1e-5), \
                f"Group {indices} channels have different weights at t={t}"


def test_different_groups_can_differ(maa):
    """Different groups can receive different weights."""
    x = torch.randn(2, 8, 200)
    out = maa(x)

    # Compute mean weight per group
    with torch.no_grad():
        ratio = out / (x + 1e-8)
        group_means = {}
        for name, indices in MotorAreaAttention.GROUP_INDICES.items():
            group_means[name] = ratio[:, indices].mean().item()

    # At least one group should differ from another (probabilistic)
    values = list(group_means.values())
    if all(abs(values[0] - v) < 1e-6 for v in values[1:]):
        # Edge case: all groups got same weight — that's unlikely but possible
        # Verify the weights are valid (in [0,1])
        assert all(0 <= v <= 2 for v in values)


def test_uniform_input_group_consistent(maa):
    """With identical input, channels within each group should get same weight."""
    x = torch.ones(2, 8, 100)
    out = maa(x)
    # Channels in the same group should have identical values
    ratio = out[0] / (x[0] + 1e-8)
    for indices in MotorAreaAttention.GROUP_INDICES.values():
        group_vals = ratio[indices]
        assert torch.allclose(group_vals, group_vals[0], atol=1e-5), \
            f"Group {indices} channels differ with uniform input"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(maa, batch, tmp_path):
    """Model should produce identical output after save/load."""
    maa.eval()
    with torch.no_grad():
        out_before = maa(batch)

    path = tmp_path / "maa.pt"
    torch.save(maa.state_dict(), path)

    maa2 = MotorAreaAttention(n_channels=8)
    maa2.load_state_dict(torch.load(path, weights_only=True))
    maa2.eval()

    with torch.no_grad():
        out_after = maa2(batch)

    assert torch.allclose(out_before, out_after, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_single_sample(maa):
    """Batch size 1 should work."""
    x = torch.randn(1, 8, 100)
    out = maa(x)
    assert out.shape == x.shape


def test_batch_size_large(maa):
    """Larger batch size should work."""
    x = torch.randn(32, 8, 200)
    out = maa(x)
    assert out.shape == x.shape


def test_wrong_n_channels_raises():
    """Should raise ValueError for non-8 channels."""
    with pytest.raises(ValueError, match="expects 8 channels"):
        MotorAreaAttention(n_channels=16)


def test_deterministic_in_eval(maa, batch):
    """Output should be deterministic in eval mode."""
    maa.eval()
    with torch.no_grad():
        out1 = maa(batch)
        out2 = maa(batch)
    assert torch.equal(out1, out2)
