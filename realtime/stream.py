"""
DeepBCI / LSL stream reader.

Supports:
  - LSL (Lab Streaming Layer) — most common for research devices
  - DeepBCI WebSocket — if available
  - CSV/file simulation — for offline testing
"""
import time
import numpy as np

from utils.config import N_CHANNELS, SFREQ


class DummyStream:
    """
    Simulated EEG stream for testing the real-time pipeline.
    Generates random EEG-like noise.
    """

    def __init__(self, n_channels: int = N_CHANNELS, s_freq: int = SFREQ):
        self.n_channels = n_channels
        self.s_freq = s_freq
        self.chunk_size = int(s_freq * 0.125)  # 125 ms chunks @ 250 Hz → ~31 samples

    def open(self):
        pass

    def read_chunk(self) -> np.ndarray:
        """Return (n_channels, chunk_size) array of synthetic EEG."""
        time.sleep(0.125)  # simulate real-time pacing
        return np.random.randn(self.n_channels, self.chunk_size).astype(np.float32)

    def close(self):
        pass

