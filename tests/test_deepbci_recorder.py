"""Tests for realtime/deepbci_recorder.py — DeepBCIRecorder."""

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

from realtime.deepbci_recorder import DeepBCIRecorder


class TestDeepBCIRecorderSession:
    """Session creation and directory structure."""

    def test_start_session_creates_subdirectory(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(subject_id=1)
        assert p.exists()
        assert p.is_dir()
        assert "session_" in p.name
        assert "sub_001" in str(p.parent)

    def test_session_id_has_correct_format(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        recorder.start_session(subject_id=5)
        assert re.match(r"sub_005/session_\d{8}_\d{6}", recorder._session_id)

    def test_start_session_creates_notes_md(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(subject_id=2, notes="pilot test")
        notes = p / "notes.md"
        assert notes.exists()
        content = notes.read_text(encoding="utf-8")
        assert "Subject 002" in content
        assert "pilot test" in content

    def test_start_session_returns_path(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(subject_id=1)
        assert isinstance(p, Path)
        assert p == recorder._session_dir

    def test_multiple_sessions_create_different_dirs(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p1 = recorder.start_session(subject_id=1)
        recorder.end_session()
        time.sleep(1.1)  # ensure different timestamp
        p2 = recorder.start_session(subject_id=1)
        recorder.end_session()
        assert p1 != p2
        assert p1.parent == p2.parent  # same sub_001 parent

    def test_device_protocol_operator_params(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(
            subject_id=1,
            device="TestDevice",
            protocol="custom_protocol",
            operator="Alice",
        )
        recorder.end_session()
        md = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
        assert md["device"] == "TestDevice"
        assert md["protocol"] == "custom_protocol"
        assert md["operator"] == "Alice"


class TestDeepBCIRecorderRecord:
    """Recording chunks and events."""

    def test_record_chunk_writes_raw_csv(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(subject_id=1)
        chunk = np.random.randn(8, 31).astype(np.float32)
        recorder.record_chunk(chunk, timestamp_s=1.5)
        recorder.end_session()

        raw = p / "raw.csv"
        assert raw.exists()
        content = raw.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) >= 1
        assert "1.5000" in lines[0]

    def test_record_chunk_with_event(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(subject_id=1)
        recorder.record_chunk(
            np.random.randn(8, 31).astype(np.float32),
            timestamp_s=0.0,
            event_label="left_hand",
        )
        recorder.end_session()

        events = p / "events.csv"
        assert events.exists()
        content = events.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 2  # header + 1 event
        assert "left_hand" in lines[1]

    def test_record_chunk_unknown_event_label(self, tmp_path):
        """Unknown event label → class_id=-1."""
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(subject_id=1)
        recorder.record_chunk(
            np.random.randn(8, 31).astype(np.float32),
            timestamp_s=0.0,
            event_label="nonexistent_event_xyz",
        )
        recorder.end_session()
        events = p / "events.csv"
        content = events.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert "-1" in lines[1]

    def test_record_chunk_without_session_raises(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        with pytest.raises(RuntimeError, match="not started"):
            recorder.record_chunk(np.random.randn(8, 31).astype(np.float32), 0.0)


class TestDeepBCIRecorderEndSession:
    """Metadata writing and cleanup."""

    def test_end_session_writes_metadata(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(subject_id=3, notes="test session")
        recorder.record_chunk(np.random.randn(8, 31).astype(np.float32), 0.0)
        recorder.record_chunk(np.random.randn(8, 31).astype(np.float32), 1.0)
        recorder.end_session()

        meta = p / "metadata.json"
        assert meta.exists()
        md = json.loads(meta.read_text(encoding="utf-8"))
        assert md["subject_id"] == 3
        assert md["n_channels"] == 8
        assert md["sfreq"] == 250
        assert md["n_trials"] == 0  # no events recorded
        assert md["notes"] == "test session"
        assert isinstance(md["date"], str)
        assert isinstance(md["channels"], list)
        assert len(md["channels"]) == 8

    def test_end_session_n_trials_count(self, tmp_path):
        """n_trials counts only chunks with event_label."""
        recorder = DeepBCIRecorder(data_root=tmp_path)
        p = recorder.start_session(subject_id=1)
        # 2 without event, 3 with event → n_trials=3
        for i in range(5):
            label = "rest" if i < 3 else None
            recorder.record_chunk(
                np.random.randn(8, 31).astype(np.float32),
                timestamp_s=float(i),
                event_label=label,
            )
        recorder.end_session()
        md = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
        assert md["n_trials"] == 3

    def test_end_session_closes_writers(self, tmp_path):
        recorder = DeepBCIRecorder(data_root=tmp_path)
        recorder.start_session(subject_id=1)
        recorder.end_session()
        assert recorder._raw_writer is None
        assert recorder._events_writer is None

    def test_end_session_without_start_noop(self, tmp_path):
        """Calling end_session() without start_session() should not crash."""
        recorder = DeepBCIRecorder(data_root=tmp_path)
        recorder.end_session()  # should not raise
