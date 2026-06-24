"""Tests for realtime/sources.py — EEGSource Protocol."""

import numpy as np
import pytest

from realtime.sources import EEGSource
from realtime.stream import DummyStream
from realtime.file_replay import FileReplaySource
from realtime.deepbci_source import DeepBCISource


class TestEEGSourceProtocol:
    """Structural subtyping (duck typing) checks."""

    def test_dummystream_satisfies_protocol(self):
        """DummyStream has open/read_chunk/close → satisfies EEGSource."""
        stream = DummyStream()
        assert isinstance(stream, EEGSource)

    def test_deepbci_source_satisfies_protocol(self):
        """DeepBCISource has the right method signatures → satisfies EEGSource."""
        source = DeepBCISource()
        assert isinstance(source, EEGSource)

    def test_filereplay_source_satisfies_protocol(self, tmp_path):
        """FileReplaySource (unopened) satisfies EEGSource via structural match."""
        data = np.random.randn(2, 8, 100).astype(np.float32)
        path = tmp_path / "X.npy"
        np.save(path, data)
        source = FileReplaySource(data_path=str(path))
        assert isinstance(source, EEGSource)

    def test_object_without_read_chunk_fails(self):
        """An object missing read_chunk does NOT satisfy the protocol."""

        class BadStream:
            n_channels = 8
            s_freq = 250

            def open(self):
                pass

            def close(self):
                pass

        assert not isinstance(BadStream(), EEGSource)

    def test_object_missing_attributes_fails(self):
        """Plain object without n_channels/s_freq fails."""
        assert not isinstance(object(), EEGSource)

    def test_runtime_checkable_works(self):
        """isinstance checks work at runtime (not just static analysis)."""
        stream = DummyStream()
        # Should not raise TypeError
        result = isinstance(stream, EEGSource)
        assert result is True
