"""
Ring buffer for real-time EEG streaming.

Maintains a sliding window of recent EEG data for MI inference.
"""
import threading
import numpy as np

from utils.config import N_CHANNELS, SFREQ, BUFFER_WINDOW


class RingBuffer:
    """
    Thread-safe ring buffer for EEG data.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    window_s : float
        Buffer window in seconds.
    s_freq : int
        Sampling frequency.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        window_s: float = BUFFER_WINDOW,
        s_freq: int = SFREQ,
    ):
        self.n_channels = n_channels
        self.s_freq = s_freq
        self.capacity = int(window_s * s_freq)
        self.buffer = np.zeros((n_channels, self.capacity), dtype=np.float32)
        self._write_pos = 0
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray):
        """
        Push new samples into the buffer.

        Parameters
        ----------
        samples : np.ndarray, shape (n_channels, n_samples)
        """
        n_samples = samples.shape[-1]
        with self._lock:
            for i in range(n_samples):
                self.buffer[:, self._write_pos] = samples[:, i]
                self._write_pos = (self._write_pos + 1) % self.capacity

    def read(self) -> np.ndarray:
        """Return a copy of the current buffer (n_channels, capacity)."""
        with self._lock:
            return self.buffer.copy()

    def reset(self):
        with self._lock:
            self.buffer.fill(0)
            self._write_pos = 0
