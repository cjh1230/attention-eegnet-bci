"""
Placeholder for DeepBCI hardware EEG source.

Implements the EEGSource protocol (open/read_chunk/close) as a skeleton.
Replace the NotImplementedError bodies with real device integration when
the DeepBCI 8-channel amplifier becomes available.

Usage (after hardware integration):
    from realtime.deepbci_source import DeepBCISource
    source = DeepBCISource()
    source.open()
    chunk = source.read_chunk()  # (8, n_samples) float32
    source.close()
"""

import numpy as np

from utils.config import N_CHANNELS, SFREQ


class DeepBCISource:
    """
    EEGSource for DeepBCI 8-channel hardware (PLACEHOLDER).

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default 8).
    s_freq : int
        Sampling frequency in Hz (default 250).
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        s_freq: int = SFREQ,
    ):
        self.n_channels = n_channels
        self.s_freq = s_freq

    def open(self) -> None:
        """
        Connect to DeepBCI hardware and start streaming.

        Raises
        ------
        NotImplementedError
            Hardware not yet available.  Implement serial / Bluetooth / USB
            connection logic here.
        """
        raise NotImplementedError(
            "DeepBCI hardware source is not yet implemented.\n"
            "Expected: connect to 8-channel amplifier, start 250 Hz stream.\n"
            "Implement serial/Bluetooth/USB connection and streaming loop here."
        )

    def read_chunk(self) -> np.ndarray:
        """
        Read the next EEG chunk from the device.

        Returns
        -------
        chunk : np.ndarray, shape (n_channels, n_samples), dtype float32

        Raises
        ------
        NotImplementedError
            Hardware not yet available.
        """
        raise NotImplementedError(
            "DeepBCI read_chunk() is not yet implemented.\n"
            "Expected: read from device buffer, return (8, ~31) float32 array."
        )

    def close(self) -> None:
        """
        Disconnect from DeepBCI hardware and stop streaming.

        Raises
        ------
        NotImplementedError
            Hardware not yet available.
        """
        raise NotImplementedError(
            "DeepBCI close() is not yet implemented.\n"
            "Expected: close serial/USB connection, stop streaming."
        )
