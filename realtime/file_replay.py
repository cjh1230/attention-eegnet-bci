"""
Offline file-replay source — streams preprocessed .npy data as a simulated
real-time EEG stream for closed-loop pipeline testing without hardware.

Reads X.npy (shape N x C x T) and optionally y.npy (shape N), then replays
the data in 125 ms chunks at a simulated 250 Hz pace.

Usage:
    from realtime.file_replay import FileReplaySource

    source = FileReplaySource("data/loso_binary/subj_01/X.npy",
                              labels_path="data/loso_binary/subj_01/y.npy")
    source.open()
    while not source.exhausted:
        chunk = source.read_chunk()  # (8, 31) float32
    source.close()
"""

import time
import warnings
from pathlib import Path

import numpy as np

from utils.config import N_CHANNELS, SFREQ


class FileReplaySource:
    """
    Replay preprocessed EEG trials as a simulated streaming source.

    Reads preprocessed .npy files (N_trials × C × T) and emits fixed-size
    temporal chunks at simulated real-time pacing.  Handles cross-trial
    boundaries by seamlessly splicing consecutive trials into the output
    stream.

    Parameters
    ----------
    data_path : str or Path
        Path to X.npy — shape (N, C, T), dtype float32.
    labels_path : str or Path, optional
        Path to y.npy — shape (N,), dtype int.  If omitted, labels are
        reported as -1.
    chunk_duration_s : float
        Duration of each read_chunk() output in seconds (default 0.125).
    s_freq : int
        Sampling frequency in Hz (default 250).
    loop : bool
        If True, restart from the first trial after exhaustion instead of
        returning zeros (default False).
    n_channels : int
        Expected channel count (default 8).  If the data has a different
        count, a warning is issued but playback continues.
    """

    def __init__(
        self,
        data_path: str | Path,
        labels_path: str | Path | None = None,
        chunk_duration_s: float = 0.125,
        s_freq: int = SFREQ,
        loop: bool = False,
        n_channels: int = N_CHANNELS,
        trial_mode: bool = False,
    ):
        self.data_path = Path(data_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.chunk_duration_s = chunk_duration_s
        self.s_freq = s_freq
        self.loop = loop
        self.n_channels = n_channels
        self.trial_mode = trial_mode

        self.chunk_samples: int = int(s_freq * chunk_duration_s)

        # ── Runtime state (set in open, reset in close) ──────────
        self._data: np.ndarray | None = None       # (N, C, T)
        self._labels: np.ndarray | None = None     # (N,)
        self._n_trials: int = 0
        self._trial_index: int = 0
        self._sample_offset: int = 0
        self._exhausted: bool = False
        self.current_trial: int = 0
        self.current_label: int = -1

    # ── Public properties ───────────────────────────────────────

    @property
    def exhausted(self) -> bool:
        """True when all trials have been consumed (and loop is off)."""
        return self._exhausted

    # ── Public API ──────────────────────────────────────────────

    def open(self) -> None:
        """Load .npy data and validate shapes."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = np.load(self.data_path).astype(np.float32)

        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D data (N, C, T), got {data.ndim}D: {data.shape}"
            )
        if data.shape[0] == 0:
            raise ValueError("Data file contains 0 trials")

        actual_channels = data.shape[1]
        if actual_channels != self.n_channels:
            warnings.warn(
                f"Channel mismatch: data has {actual_channels} channels, "
                f"but n_channels={self.n_channels} was specified. "
                f"Proceeding with {actual_channels} channels."
            )
            self.n_channels = actual_channels
            self.chunk_samples = int(self.s_freq * self.chunk_duration_s)

        self._data = data
        self._n_trials = data.shape[0]

        # Labels
        if self.labels_path and self.labels_path.exists():
            labels = np.load(self.labels_path).astype(np.int64)
            if len(labels) != self._n_trials:
                raise ValueError(
                    f"Label count ({len(labels)}) != trial count ({self._n_trials})"
                )
            self._labels = labels
        else:
            self._labels = np.full(self._n_trials, -1, dtype=np.int64)

        # Reset runtime state
        self._trial_index = 0
        self._sample_offset = 0
        self._exhausted = False
        self.current_trial = 0
        self.current_label = -1

    def read_chunk(self) -> np.ndarray:
        """
        Read the next chunk of EEG data.

        In streaming mode (trial_mode=False):
            Returns (n_channels, chunk_samples) — a temporal slice that may
            span trial boundaries.

        In trial mode (trial_mode=True):
            Returns (n_channels, trial_length) — one complete trial.

        Returns
        -------
        chunk : np.ndarray, shape varies, dtype float32
        """
        if self._data is None:
            raise RuntimeError("Source not opened. Call open() first.")

        # ── Trial mode: return full trials ───────────────────────
        if self.trial_mode:
            if self._exhausted:
                return np.zeros(
                    (self.n_channels, 1), dtype=np.float32
                )
            if self._trial_index >= self._n_trials:
                if self.loop:
                    self._trial_index = 0
                else:
                    self._exhausted = True
                    return np.zeros(
                        (self.n_channels, 1), dtype=np.float32
                    )

            trial = self._data[self._trial_index].copy()
            self.current_trial = self._trial_index
            self.current_label = int(self._labels[self._trial_index])
            self._trial_index += 1
            time.sleep(self.chunk_duration_s)
            return trial

        # ── Streaming mode: temporal slices ──────────────────────
        if self._exhausted:
            return np.zeros(
                (self.n_channels, self.chunk_samples), dtype=np.float32
            )

        chunk = np.zeros(
            (self.n_channels, self.chunk_samples), dtype=np.float32
        )
        samples_written = 0

        while samples_written < self.chunk_samples:
            if self._trial_index >= self._n_trials:
                if self.loop:
                    self._trial_index = 0
                    self._sample_offset = 0
                else:
                    self._exhausted = True
                    break

            trial = self._data[self._trial_index]        # (C, T)
            trial_length = trial.shape[-1]
            remaining_in_trial = trial_length - self._sample_offset
            to_copy = min(
                self.chunk_samples - samples_written,
                remaining_in_trial,
            )

            chunk[:, samples_written : samples_written + to_copy] = (
                trial[:, self._sample_offset : self._sample_offset + to_copy]
            )
            samples_written += to_copy
            self._sample_offset += to_copy

            if self._sample_offset >= trial_length:
                self._trial_index += 1
                self._sample_offset = 0

        # Update public tracking
        if self._trial_index < self._n_trials:
            self.current_trial = self._trial_index
            self.current_label = int(self._labels[self._trial_index])
        elif not self._exhausted:
            self.current_trial = self._n_trials - 1
            self.current_label = int(self._labels[self._n_trials - 1])

        time.sleep(self.chunk_duration_s)
        return chunk

    def close(self) -> None:
        """Release loaded data and reset state so the instance is reusable."""
        self._data = None
        self._labels = None
        self._n_trials = 0
        self._trial_index = 0
        self._sample_offset = 0
        self._exhausted = False
        self.current_trial = 0
        self.current_label = -1
