"""Tests for realtime/file_replay.py — FileReplaySource.

Tests both streaming mode (temporal slices) and trial mode (full trials).
"""

import numpy as np
import pytest

from realtime.file_replay import FileReplaySource


class TestFileReplayInit:
    """Constructor and parameter handling."""

    def test_default_parameters(self, tmp_path):
        """Constructor populates defaults from config."""
        data = np.random.randn(2, 8, 100).astype(np.float32)
        path = tmp_path / "X.npy"
        np.save(path, data)
        source = FileReplaySource(data_path=str(path))
        assert source.n_channels == 8
        assert source.s_freq == 250
        assert source.chunk_duration_s == 0.125
        assert source.loop is False
        assert source.trial_mode is False

    def test_custom_parameters(self, tmp_path):
        """Constructor accepts custom values."""
        data = np.random.randn(2, 8, 100).astype(np.float32)
        path = tmp_path / "X.npy"
        np.save(path, data)
        source = FileReplaySource(
            data_path=str(path),
            chunk_duration_s=0.5,
            s_freq=500,
            loop=True,
            n_channels=8,
            trial_mode=True,
        )
        assert source.chunk_duration_s == 0.5
        assert source.chunk_samples == 250  # 500 * 0.5
        assert source.loop is True
        assert source.trial_mode is True


class TestFileReplayOpen:
    """File loading and validation during open()."""

    def test_open_loads_data(self, tmp_path):
        data = np.random.randn(5, 8, 250).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(data_path=str(tmp_path / "X.npy"))
        source.open()
        assert source._data.shape == (5, 8, 250)
        assert source._n_trials == 5

    def test_open_loads_labels(self, tmp_path):
        data = np.random.randn(3, 8, 100).astype(np.float32)
        labels = np.array([0, 1, 2], dtype=np.int64)
        np.save(tmp_path / "X.npy", data)
        np.save(tmp_path / "y.npy", labels)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            labels_path=str(tmp_path / "y.npy"),
        )
        source.open()
        assert source._labels.shape == (3,)
        assert list(source._labels) == [0, 1, 2]

    def test_open_file_not_found(self, tmp_path):
        source = FileReplaySource(data_path=str(tmp_path / "missing.npy"))
        with pytest.raises(FileNotFoundError):
            source.open()

    def test_open_invalid_ndim(self, tmp_path):
        """2D data should raise ValueError."""
        data = np.random.randn(8, 250).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(data_path=str(tmp_path / "X.npy"))
        with pytest.raises(ValueError, match="Expected 3D"):
            source.open()

    def test_open_empty_trials(self, tmp_path):
        """0-trial data should raise ValueError."""
        data = np.zeros((0, 8, 250), dtype=np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(data_path=str(tmp_path / "X.npy"))
        with pytest.raises(ValueError, match="0 trials"):
            source.open()

    def test_open_label_count_mismatch(self, tmp_path):
        """Label count != trial count should raise ValueError."""
        data = np.random.randn(5, 8, 100).astype(np.float32)
        labels = np.array([0, 1, 2], dtype=np.int64)  # 3 labels, 5 trials
        np.save(tmp_path / "X.npy", data)
        np.save(tmp_path / "y.npy", labels)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            labels_path=str(tmp_path / "y.npy"),
        )
        with pytest.raises(ValueError, match="Label count"):
            source.open()

    def test_open_channel_mismatch_warns(self, tmp_path):
        """Data with different n_channels should warn and adapt."""
        data = np.random.randn(3, 16, 250).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            n_channels=8,  # mismatch
        )
        with pytest.warns(UserWarning, match="Channel mismatch"):
            source.open()
        assert source.n_channels == 16  # adapted to data

    def test_labels_path_missing(self, tmp_path):
        """Missing labels file → all labels = -1."""
        data = np.random.randn(2, 8, 100).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            labels_path=str(tmp_path / "nonexistent.npy"),
        )
        source.open()
        assert list(source._labels) == [-1, -1]

    def test_labels_path_none(self, tmp_path):
        """No labels path → all labels = -1."""
        data = np.random.randn(2, 8, 100).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(data_path=str(tmp_path / "X.npy"))
        source.open()
        assert list(source._labels) == [-1, -1]

    def test_open_resets_runtime_state(self, tmp_path):
        """Calling open() twice resets trial_index and exhausted."""
        data = np.random.randn(2, 8, 100).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(data_path=str(tmp_path / "X.npy"), trial_mode=True)
        source.open()
        source.read_chunk()  # consume one trial
        assert source._trial_index == 1
        source.open()  # re-open
        assert source._trial_index == 0
        assert source._exhausted is False


class TestFileReplayStreaming:
    """Streaming mode (temporal slices) tests."""

    def test_read_chunk_shape(self, tmp_path):
        """Each chunk has shape (n_channels, chunk_samples)."""
        data = np.random.randn(1, 8, 250).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            chunk_duration_s=0.125,  # 31 samples @ 250 Hz
        )
        source.open()
        chunk = source.read_chunk()
        assert chunk.shape == (8, 31)
        assert chunk.dtype == np.float32

    def test_read_chunk_exhaustion(self, tmp_path):
        """After all data consumed, exhausted=True and returns zeros."""
        data = np.random.randn(1, 8, 100).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            chunk_duration_s=1.0,  # 250 samples chunk, data has only 100
        )
        source.open()
        # First chunk: 100 data + 150 zeros; exhausted after this
        chunk = source.read_chunk()
        assert chunk.shape == (8, 250)
        assert source.exhausted
        # Second chunk: all zeros
        chunk2 = source.read_chunk()
        assert np.all(chunk2 == 0)

    def test_read_chunk_without_open_raises(self, tmp_path):
        """Calling read_chunk before open() raises RuntimeError."""
        source = FileReplaySource(data_path=str(tmp_path / "X.npy"))
        with pytest.raises(RuntimeError, match="not opened"):
            source.read_chunk()

    def test_loop_mode_does_not_exhaust(self, tmp_path):
        """With loop=True, data cycles endlessly."""
        data = np.random.randn(1, 8, 60).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            chunk_duration_s=0.5,  # 125 samples > 60 → exhausts quickly
            loop=True,
        )
        source.open()
        chunks = [source.read_chunk() for _ in range(5)]
        assert not source.exhausted
        # None of the chunks should be all-zero (data keeps cycling)
        assert not any(np.all(c == 0) for c in chunks)

    def test_close_resets_state(self, tmp_path):
        """close() releases data and resets all state fields."""
        data = np.random.randn(2, 8, 100).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(data_path=str(tmp_path / "X.npy"))
        source.open()
        source.read_chunk()
        source.close()
        assert source._data is None
        assert source._n_trials == 0
        assert source._exhausted is False


class TestFileReplayTrialMode:
    """Trial mode (full trial per read_chunk()) tests."""

    def test_trial_mode_returns_full_trial(self, tmp_path):
        """read_chunk() returns a complete (C, T) trial."""
        data = np.random.randn(3, 8, 250).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            trial_mode=True,
        )
        source.open()
        chunk = source.read_chunk()
        assert chunk.shape == (8, 250)
        assert source.current_trial == 0

    def test_trial_mode_label_tracking(self, tmp_path):
        """current_trial and current_label are updated per trial."""
        data = np.random.randn(3, 8, 100).astype(np.float32)
        labels = np.array([10, 20, 30], dtype=np.int64)
        np.save(tmp_path / "X.npy", data)
        np.save(tmp_path / "y.npy", labels)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            labels_path=str(tmp_path / "y.npy"),
            trial_mode=True,
        )
        source.open()
        source.read_chunk()
        assert source.current_trial == 0
        assert source.current_label == 10
        source.read_chunk()
        assert source.current_trial == 1
        assert source.current_label == 20

    def test_trial_mode_exhaustion(self, tmp_path):
        """After all trials, third read_chunk triggers exhausted + zero return."""
        data = np.random.randn(2, 8, 100).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            trial_mode=True,
        )
        source.open()
        source.read_chunk()  # trial 0
        source.read_chunk()  # trial 1 → _trial_index = 2
        # Third call: _trial_index(2) >= _n_trials(2) → exhausted, returns zeros
        chunk = source.read_chunk()
        assert source.exhausted
        assert chunk.shape[0] == 8  # n_channels preserved
        assert chunk.shape[1] == 1
        assert np.all(chunk == 0)

    def test_trial_mode_loop(self, tmp_path):
        """With loop=True and trial_mode, trials repeat."""
        data = np.random.randn(1, 8, 100).astype(np.float32)
        np.save(tmp_path / "X.npy", data)
        source = FileReplaySource(
            data_path=str(tmp_path / "X.npy"),
            trial_mode=True,
            loop=True,
        )
        source.open()
        chunk1 = source.read_chunk()
        chunk2 = source.read_chunk()
        # Both should return the same trial
        assert np.array_equal(chunk1, chunk2)
        assert not source.exhausted
