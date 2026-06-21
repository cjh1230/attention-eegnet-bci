"""
Real LSL stream reader for DeepBCI and other LSL-compatible EEG devices.

Usage (when pylsl is installed):
    stream = LSLStream(name="DeepBCI")
    stream.open()
    while True:
        chunk = stream.read_chunk()  # (n_channels, n_samples)
"""
import sys
import time
from typing import Optional

import numpy as np

try:
    from pylsl import StreamInlet, resolve_byprop, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

from utils.config import N_CHANNELS, SFREQ


class LSLStream:
    """
    Connect to an LSL EEG stream by name or type.

    Parameters
    ----------
    name : str
        LSL stream name (e.g., "DeepBCI", "BrainVision").
    stream_type : str
        LSL stream type (e.g., "EEG").
    n_channels : int
        Expected number of channels.
    s_freq : int
        Expected sampling frequency.
    timeout : float
        Seconds to wait for stream on open().
    """

    def __init__(
        self,
        name: str = "DeepBCI",
        stream_type: str = "EEG",
        n_channels: int = N_CHANNELS,
        s_freq: int = SFREQ,
        timeout: float = 5.0,
    ):
        if not HAS_LSL:
            raise ImportError(
                "pylsl not installed. Run: pip install pylsl\n"
                "Or use DummyStream for testing."
            )
        self.name = name
        self.stream_type = stream_type
        self.n_channels = n_channels
        self.s_freq = s_freq
        self.timeout = timeout
        self.inlet: Optional[StreamInlet] = None
        self._chunk_size = max(1, int(s_freq * 0.125))

    def open(self):
        """Resolve and connect to the LSL stream."""
        print(f"Looking for LSL stream: name='{self.name}', type='{self.stream_type}'")
        streams = resolve_byprop("name", self.name, timeout=self.timeout)
        if not streams:
            # Try finding any EEG stream
            print(f"  No exact match — scanning for any '{self.stream_type}' stream...")
            streams = resolve_byprop("type", self.stream_type, timeout=self.timeout)
            if not streams:
                available = resolve_streams(timeout=2.0)
                names = [s.name() for s in available]
                raise RuntimeError(
                    f"No LSL stream found. Available streams: {names or 'none'}"
                )
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        print(f"  Connected: name='{info.name()}', "
              f"channels={info.channel_count()}, sfreq={info.nominal_srate()}")

    def read_chunk(self) -> np.ndarray:
        """
        Read a chunk of samples and return (n_channels, n_samples).
        Blocks until enough samples arrive.
        """
        if self.inlet is None:
            raise RuntimeError("Stream not open. Call open() first.")

        samples, timestamps = self.inlet.pull_chunk(
            max_samples=self._chunk_size
        )
        if not samples:
            return np.zeros((self.n_channels, self._chunk_size), dtype=np.float32)

        arr = np.array(samples, dtype=np.float32)
        # LSL returns (time, channels) → transpose to (channels, time)
        return arr.T

    def close(self):
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
            print("Stream closed.")
