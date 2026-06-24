"""Tests for datasets/deepbci_loader.py — DeepBCILoader."""

import json
from pathlib import Path

import numpy as np
import pytest

from datasets.deepbci_loader import DeepBCILoader, process_session


# ── Mock session helpers ──────────────────────────────────────


def _create_mock_session(
    tmp_path: Path,
    n_chunks: int = 200,
    n_events: int = 3,
    n_channels: int = 8,
    sfreq: int = 250,
    chunk_duration_s: float = 0.125,
) -> Path:
    """Create a mock DeepBCI session directory with synthetic data.

    200 chunks × 0.125s = 25s of continuous data — enough for
    3s epochs with margin at session boundaries.
    """
    session_dir = tmp_path / "sub_001" / "session_20240101_120000"
    session_dir.mkdir(parents=True)

    _write_metadata(session_dir, n_channels=n_channels, sfreq=sfreq)

    chunk_size = int(sfreq * chunk_duration_s)
    rng = np.random.RandomState(42)
    total_duration = 0.0
    with open(session_dir / "raw.csv", "w") as f:
        for i in range(n_chunks):
            ts = i * chunk_size / sfreq
            chunk = rng.randn(n_channels, chunk_size).astype(np.float32)
            row = [f"{ts:.4f}"] + [f"{v:.6f}" for v in chunk.flatten(order="F")]
            f.write(",".join(row) + "\n")
            total_duration = ts + chunk_size / sfreq

    _write_events(session_dir, n_events, total_duration)
    return session_dir


def _write_metadata(
    session_dir: Path, n_channels: int = 8, sfreq: int = 250
) -> None:
    metadata = {
        "session_id": "sub_001/session_20240101_120000",
        "subject_id": 1,
        "date": "2024-01-01T12:00:00",
        "device": "DeepBCI",
        "protocol": "motor_imagery",
        "n_channels": n_channels,
        "sfreq": sfreq,
        "n_trials": 3,
        "notes": "Mock session for testing",
    }
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _write_events(
    session_dir: Path,
    n_events: int,
    total_duration: float,
) -> None:
    """Write events.csv with events spread across the time range."""
    event_labels = ["idle", "left_hand", "right_hand"]
    from datasets.label_mapping import LABEL_MAPS

    margin = 3.0  # seconds from start/end to stay within valid data
    available = total_duration - 2 * margin
    if available <= 0:
        # Very short session — place events at the middle
        margin = total_duration * 0.25
        available = total_duration * 0.5
    with open(session_dir / "events.csv", "w") as f:
        f.write("timestamp_s,event_label,class_id\n")
        for i in range(n_events):
            ts = margin + (i + 1) * available / (n_events + 1)
            label = event_labels[i % len(event_labels)]
            class_id = LABEL_MAPS["deepbci"].get(label, -1)
            f.write(f"{ts:.4f},{label},{class_id}\n")


# ── Tests ─────────────────────────────────────────────────────


class TestDeepBCILoaderInit:
    """Constructor parameter handling."""

    def test_default_parameters(self, tmp_path):
        session = _create_mock_session(tmp_path)
        loader = DeepBCILoader(session)
        assert loader.tmin == -0.5
        assert loader.tmax == 2.5
        assert loader.n_channels == 8
        assert loader.sfreq == 250
        assert loader._epoch_samples == 750

    def test_custom_parameters(self, tmp_path):
        session = _create_mock_session(tmp_path)
        loader = DeepBCILoader(
            session, tmin=-1.0, tmax=3.0, n_channels=16, sfreq=500
        )
        assert loader.tmin == -1.0
        assert loader.tmax == 3.0
        assert loader.n_channels == 16
        assert loader.sfreq == 500
        assert loader._epoch_samples == 2000  # (3.0 - (-1.0)) * 500


class TestDeepBCILoaderLoad:
    """Full load() pipeline."""

    def test_load_returns_correct_shapes(self, tmp_path):
        session = _create_mock_session(tmp_path, n_chunks=200, n_events=3)
        loader = DeepBCILoader(session)
        X, y = loader.load()
        assert X.ndim == 3
        assert X.shape[0] == 3  # 3 events
        assert X.shape[1] == 8  # n_channels
        assert X.shape[2] == 750  # epoch_samples
        assert X.dtype == np.float32
        assert y.shape == (3,)
        assert y.dtype == np.int64

    def test_load_preserves_class_ids(self, tmp_path):
        session = _create_mock_session(tmp_path, n_chunks=200, n_events=4)
        loader = DeepBCILoader(session)
        _X, y = loader.load()
        # idle=0, left_hand=1, right_hand=2, idle=0
        assert list(y) == [0, 1, 2, 0]

    def test_load_epochs_are_not_all_zero(self, tmp_path):
        session = _create_mock_session(tmp_path, n_chunks=200, n_events=2)
        loader = DeepBCILoader(session)
        X, _y = loader.load()
        assert not np.allclose(X[0], 0)
        assert not np.allclose(X[1], 0)

    def test_load_epochs_differ(self, tmp_path):
        """Different events produce different epoch data."""
        session = _create_mock_session(tmp_path, n_chunks=200, n_events=2)
        loader = DeepBCILoader(session)
        X, _y = loader.load()
        assert not np.allclose(X[0], X[1])

    def test_missing_raw_csv(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        _write_metadata(session)
        from datasets.label_mapping import LABEL_MAPS

        # Write events file
        with open(session / "events.csv", "w") as f:
            f.write("timestamp_s,event_label,class_id\n")
            f.write("2.0,left_hand,1\n")
        loader = DeepBCILoader(session)
        with pytest.raises(FileNotFoundError, match="raw.csv"):
            loader.load()

    def test_missing_events_csv(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        _write_metadata(session)
        # Write only raw.csv
        rng = np.random.RandomState(0)
        chunk = rng.randn(8, 31).astype(np.float32)
        with open(session / "raw.csv", "w") as f:
            row = ["0.0000"] + [f"{v:.6f}" for v in chunk.flatten(order="F")]
            f.write(",".join(row) + "\n")
        loader = DeepBCILoader(session)
        with pytest.raises(FileNotFoundError, match="events.csv"):
            loader.load()

    def test_no_events_raises(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        _write_metadata(session)
        rng = np.random.RandomState(0)
        chunk = rng.randn(8, 31).astype(np.float32)
        with open(session / "raw.csv", "w") as f:
            row = ["0.0000"] + [f"{v:.6f}" for v in chunk.flatten(order="F")]
            f.write(",".join(row) + "\n")
        with open(session / "events.csv", "w") as f:
            f.write("timestamp_s,event_label,class_id\n")  # header only, no events
        loader = DeepBCILoader(session)
        with pytest.raises(ValueError, match="No events found"):
            loader.load()

    def test_empty_raw_raises(self, tmp_path):
        session = tmp_path / "session"
        session.mkdir()
        _write_metadata(session)
        (session / "raw.csv").write_text("")
        with open(session / "events.csv", "w") as f:
            f.write("timestamp_s,event_label,class_id\n")
            f.write("2.0,left_hand,1\n")
        loader = DeepBCILoader(session)
        with pytest.raises(ValueError, match="No data rows"):
            loader.load()


class TestDeepBCILoaderRaw:
    """raw.csv loading and Fortran-order reversal."""

    def test_fortran_reversal_is_correct(self, tmp_path):
        """Verify that Fortran-flattened data is correctly reshaped."""
        session = tmp_path / "sub_001" / "session_test"
        session.mkdir(parents=True)
        _write_metadata(session)

        # Create known signal: channel c has all values = c
        n_channels = 4
        chunk_size = 10
        chunk = np.zeros((n_channels, chunk_size), dtype=np.float32)
        for c in range(n_channels):
            chunk[c, :] = float(c)

        with open(session / "raw.csv", "w") as f:
            row = ["0.0000"] + [f"{v:.6f}" for v in chunk.flatten(order="F")]
            f.write(",".join(row) + "\n")

        with open(session / "events.csv", "w") as f:
            f.write("timestamp_s,event_label,class_id\n")
            f.write("0.005,right_hand,2\n")

        loader = DeepBCILoader(session, n_channels=n_channels)
        _timestamps, continuous = loader._load_raw(n_channels)
        assert continuous.shape[0] == n_channels
        # Each channel should have its index as constant value
        for c in range(n_channels):
            assert np.allclose(continuous[c], float(c))


class TestDeepBCILoaderEpoch:
    """Epoch extraction."""

    def test_epoch_boundary_event_skipped(self, tmp_path):
        """Event at t=0.1s should be skipped (window starts at -0.4s)."""
        session = tmp_path / "sub_001" / "session_test"
        session.mkdir(parents=True)
        _write_metadata(session)

        chunk_size = 31
        n_chunks = 200  # enough for at least one valid epoch (25s)
        rng = np.random.RandomState(0)
        with open(session / "raw.csv", "w") as f:
            for i in range(n_chunks):
                ts = i * chunk_size / 250
                chunk = rng.randn(8, chunk_size).astype(np.float32)
                row = [f"{ts:.4f}"] + [
                    f"{v:.6f}" for v in chunk.flatten(order="F")
                ]
                f.write(",".join(row) + "\n")

        # Event at 0.05s — too close to start (epoch needs -0.5s pre)
        # Also add a valid event so load() doesn't raise "no valid epochs"
        with open(session / "events.csv", "w") as f:
            f.write("timestamp_s,event_label,class_id\n")
            f.write("0.05,left_hand,1\n")
            f.write("5.00,right_hand,2\n")

        loader = DeepBCILoader(session)
        with pytest.warns(UserWarning, match="Skipped 1"):
            X, y = loader.load()
        assert X.shape[0] == 1  # only the valid event
        assert y[0] == 2  # right_hand

    def test_valid_event_is_extracted(self, tmp_path):
        """Event well within data range produces a valid epoch."""
        session = _create_mock_session(tmp_path, n_chunks=200, n_events=1)
        loader = DeepBCILoader(session)
        X, y = loader.load()
        assert X.shape[0] == 1
        assert X.shape[2] == 750
        assert y[0] == 0  # first event label = idle → 0


class TestProcessSession:
    """End-to-end process_session function."""

    def test_process_session_saves_files(self, tmp_path):
        session = _create_mock_session(tmp_path, n_chunks=200, n_events=3)
        output = tmp_path / "processed"
        process_session(session, output)
        X_path = output / "sub_001" / "X.npy"
        y_path = output / "sub_001" / "y.npy"
        assert X_path.exists()
        assert y_path.exists()

    def test_process_session_output_shapes(self, tmp_path):
        session = _create_mock_session(tmp_path, n_chunks=200, n_events=3)
        output = tmp_path / "processed"
        process_session(session, output)
        X = np.load(output / "sub_001" / "X.npy")
        y = np.load(output / "sub_001" / "y.npy")
        assert X.ndim == 3
        assert X.shape[1] == 8
        assert X.shape[2] == 750
        assert y.ndim == 1
        assert y.shape[0] == 3
