"""
Shared test fixtures for BCI project tests.

All fixtures use synthetic data — no real EEG files required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def dummy_eeg_2d(rng):
    """Return a single 2-class EEG trial: (16, 500) float32."""
    return rng.randn(16, 500).astype(np.float32)


@pytest.fixture
def dummy_eeg_batch(rng):
    """Return a batch of 3-class EEG: (8, 16, 750) float32."""
    return rng.randn(8, 16, 750).astype(np.float32)


@pytest.fixture
def dummy_labels_3class():
    """Return (8,) int labels with all 3 classes represented."""
    return np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)


@pytest.fixture
def dummy_labels_2class():
    """Return (8,) int labels for 2-class."""
    return np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)


@pytest.fixture
def dummy_eeg_4d(rng):
    """Return EEG as (4, 1, 16, 750) float32 (with channel dim for Conv2d)."""
    return rng.randn(4, 1, 16, 750).astype(np.float32)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_eegnet():
    """EEGNet with small config for fast tests."""
    from models.eegnet import EEGNet

    return EEGNet(n_channels=8, n_classes=3, F1=4, D=2, F2=8)


@pytest.fixture
def warm_eegnet(small_eegnet):
    """EEGNet that has had its lazy classifier built."""
    small_eegnet.eval()
    with torch.no_grad():
        small_eegnet(torch.zeros(1, 8, 500))
    return small_eegnet


@pytest.fixture
def attention_module():
    from models.attention import ChannelAttention1D

    return ChannelAttention1D(n_channels=16)


@pytest.fixture
def fusion_model():
    from models.fusion import MultiBandFusion

    return MultiBandFusion(n_channels=8, n_classes=3, hidden=32)


# ---------------------------------------------------------------------------
# Buffer / stream fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ring_buffer():
    from realtime.buffer import RingBuffer

    return RingBuffer(n_channels=8, window_s=1.0, s_freq=250)


@pytest.fixture
def filled_buffer(ring_buffer, rng):
    """Buffer pre-filled with 1 second of data."""
    data = rng.randn(8, 100).astype(np.float32)
    ring_buffer.push(data)
    return ring_buffer
