"""
DeepBCI stream reader — connects to DeepBCI hardware or replays recorded data.

Follows the same open() / read_chunk() / close() interface as DummyStream
and LSLStream, so it's a drop-in replacement in the real-time pipeline.

Modes:
  1. File replay — reads from a pre-recorded CSV (offline testing)
  2. LSL — connects to DeepBCI LSL stream (requires hardware + pylsl)
  3. Dummy — fallback noise generator (wraps DummyStream)

Usage:
    from realtime.deepbci_stream import DeepBCIStream

    stream = DeepBCIStream(mode="file", file_path="data/subjects/sub_001/raw.csv")
    # or: stream = DeepBCIStream(mode="lsl", stream_name="DeepBCI")
    # or: stream = DeepBCIStream(mode="dummy")

    stream.open()
    while True:
        chunk = stream.read_chunk()  # shape: (n_channels, n_samples)
        ...
    stream.close()
"""

import time
from pathlib import Path

import numpy as np

from utils.config import N_CHANNELS, SFREQ

# ── Optional LSL ──────────────────────────────────────────────────────
try:
    from pylsl import StreamInlet, resolve_byprop

    HAS_LSL = True
except ImportError:
    HAS_LSL = False


class DeepBCIStream:
    """
    Unified DeepBCI stream reader with file replay / LSL / dummy modes.

    Parameters
    ----------
    mode : str
        "file" — replay from recorded CSV
        "lsl"  — live LSL stream (requires pylsl + hardware)
        "dummy" — synthetic noise (wraps DummyStream)
    file_path : str or Path, optional
        Path to raw.csv for "file" mode.
    stream_name : str
        LSL stream name (default "DeepBCI").
    n_channels : int
    s_freq : int
    chunk_duration_s : float
        Duration of each read_chunk in seconds.
    """

    def __init__(
        self,
        mode: str = "dummy",
        file_path: str | Path | None = None,
        stream_name: str = "DeepBCI",
        n_channels: int = N_CHANNELS,
        s_freq: int = SFREQ,
        chunk_duration_s: float = 0.125,
    ):
        if mode not in ("file", "lsl", "dummy"):
            raise ValueError(f"Unknown mode '{mode}'. Choose: file, lsl, dummy")

        self.mode = mode
        self.file_path = Path(file_path) if file_path else None
        self.stream_name = stream_name
        self.n_channels = n_channels
        self.s_freq = s_freq
        self.chunk_duration_s = chunk_duration_s
        self.chunk_samples = int(s_freq * chunk_duration_s)

        self._inlet = None
        self._data: np.ndarray | None = None  # preloaded CSV data (C, T)
        self._cursor = 0
        self._is_open = False

    # ── Public API ─────────────────────────────────────────────────

    def open(self) -> None:
        """Open the stream connection."""
        if self._is_open:
            return

        if self.mode == "file":
            self._open_file()
        elif self.mode == "lsl":
            self._open_lsl()
        elif self.mode == "dummy":
            self._open_dummy()

        self._is_open = True

    def read_chunk(self) -> np.ndarray:
        """
        Read one chunk of EEG data.

        Returns
        -------
        chunk : np.ndarray, shape (n_channels, chunk_samples), float32
        """
        if not self._is_open:
            raise RuntimeError("Stream not open. Call open() first.")

        if self.mode == "file":
            return self._read_file_chunk()
        elif self.mode == "lsl":
            return self._read_lsl_chunk()
        else:
            return self._read_dummy_chunk()

    def close(self) -> None:
        """Close the stream connection."""
        if self.mode == "lsl" and self._inlet is not None:
            self._inlet.close_stream()
        self._is_open = False
        self._cursor = 0

    # ── Private: file replay ───────────────────────────────────────

    def _open_file(self) -> None:
        if self.file_path is None or not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        data = np.loadtxt(self.file_path, delimiter=",", dtype=np.float32)
        if data.ndim == 2:
            self._data = data.T  # (n_channels, n_samples)
        else:
            self._data = data.reshape(1, -1)

        if self._data.shape[0] != self.n_channels:
            print(
                f"  WARNING: file has {self._data.shape[0]} channels, "
                f"expected {self.n_channels}"
            )
        self._cursor = 0
        print(f"  File replay: {self.file_path} ({self._data.shape[1]} samples)")

    def _read_file_chunk(self) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("File data not loaded.")

        end = self._cursor + self.chunk_samples
        if end > self._data.shape[1]:
            # Wrap around for continuous replay
            chunk = self._data[:, self._cursor:]
            remaining = end - self._data.shape[1]
            chunk = np.concatenate([chunk, self._data[:, :remaining]], axis=1)
            self._cursor = remaining
        else:
            chunk = self._data[:, self._cursor:end]
            self._cursor = end

        time.sleep(self.chunk_duration_s)  # real-time pacing
        return chunk.astype(np.float32)

    # ── Private: LSL ────────────────────────────────────────────────

    def _open_lsl(self) -> None:
        if not HAS_LSL:
            raise ImportError(
                "pylsl not installed. Run: pip install pylsl\n"
                "Or use mode='file' or mode='dummy' for offline testing."
            )
        streams = resolve_byprop("name", self.stream_name, timeout=5)
        if not streams:
            raise RuntimeError(
                f"LSL stream '{self.stream_name}' not found. "
                f"Is the DeepBCI device connected and streaming?"
            )
        self._inlet = StreamInlet(streams[0])
        print(f"  LSL connected: {self.stream_name}")

    def _read_lsl_chunk(self) -> np.ndarray:
        if self._inlet is None:
            raise RuntimeError("LSL inlet not initialized.")

        samples, _ = self._inlet.pull_chunk(max_samples=self.chunk_samples)
        if not samples:
            return np.zeros((self.n_channels, self.chunk_samples), dtype=np.float32)
        return np.array(samples).T.astype(np.float32)

    # ── Private: dummy ──────────────────────────────────────────────

    def _open_dummy(self) -> None:
        print("  Dummy stream (synthetic EEG noise)")

    def _read_dummy_chunk(self) -> np.ndarray:
        time.sleep(self.chunk_duration_s)
        return np.random.randn(self.n_channels, self.chunk_samples).astype(np.float32)


# ── Convenience: file recorder player ──────────────────────────────────
def replay_session(subject_dir: str | Path) -> DeepBCIStream:
    """
    Open a DeepBCIStream in file mode for a recorded subject directory.

    Parameters
    ----------
    subject_dir : str or Path
        Path to data/subjects/sub_XXX/ containing raw.csv

    Returns
    -------
    DeepBCIStream in "file" mode.
    """
    file_path = Path(subject_dir) / "raw.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"No raw.csv found in {subject_dir}")
    return DeepBCIStream(mode="file", file_path=file_path)
