"""
Abstract interface for all EEG stream data sources.

Defines the EEGSource Protocol that all stream sources must satisfy
via structural subtyping (duck typing) — no inheritance required.

Implementations:
    - DummySource (stream.py)          — synthetic noise
    - FileReplaySource (file_replay.py) — offline .npy replay
    - LSLStream (stream_lsl.py)        — Lab Streaming Layer
    - DeepBCISource (future)           — real hardware
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EEGSource(Protocol):
    """
    Protocol for EEG stream data sources.

    Any class implementing open() / read_chunk() / close() satisfies
    this protocol automatically — no explicit inheritance needed.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    s_freq : int
        Sampling frequency in Hz.
    """

    n_channels: int
    s_freq: int

    def open(self) -> None:
        """
        Open the data source.

        Called once before the first read_chunk().
        Should initialise hardware, open files, or resolve LSL streams.
        """
        ...

    def read_chunk(self) -> np.ndarray:
        """
        Read the next chunk of EEG data.

        Returns
        -------
        chunk : np.ndarray, shape (n_channels, n_samples), dtype float32
        """
        ...

    def close(self) -> None:
        """
        Close the data source.

        Called once after the last read_chunk().
        Should release hardware, close files, or disconnect LSL streams.
        """
        ...
