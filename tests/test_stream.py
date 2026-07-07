"""Tests for realtime/stream.py — DummyStream (synthetic EEG stream)."""
import numpy as np
import pytest

from realtime.stream import DummyStream


class TestDummyStream:
    def test_default_init(self):
        ds = DummyStream()
        assert ds.n_channels == 8
        assert ds.s_freq == 250
        assert ds.chunk_size == int(250 * 0.125)  # 31 samples @ 125ms

    def test_custom_params(self):
        ds = DummyStream(n_channels=16, s_freq=500)
        assert ds.n_channels == 16
        assert ds.s_freq == 500
        assert ds.chunk_size == int(500 * 0.125)

    def test_open_close_noop(self):
        ds = DummyStream()
        ds.open()   # should not raise
        ds.close()  # should not raise

    def test_read_chunk_shape(self):
        ds = DummyStream(n_channels=8, s_freq=250)
        ds.open()
        chunk = ds.read_chunk()
        assert chunk.shape == (8, ds.chunk_size)
        assert chunk.dtype == np.float32

    def test_read_chunk_varying_sizes(self):
        for n_ch, sfreq in [(4, 128), (8, 250), (16, 500)]:
            ds = DummyStream(n_channels=n_ch, s_freq=sfreq)
            ds.open()
            chunk = ds.read_chunk()
            assert chunk.shape == (n_ch, ds.chunk_size)
            assert chunk.dtype == np.float32

    def test_multiple_chunks_differ(self):
        ds = DummyStream(n_channels=8, s_freq=250)
        ds.open()
        chunks = [ds.read_chunk() for _ in range(5)]
        ds.close()
        # All chunks should be different (random noise)
        for i in range(len(chunks) - 1):
            assert not np.allclose(chunks[i], chunks[i + 1])

    def test_chunk_values_are_finite(self):
        ds = DummyStream(n_channels=8, s_freq=250)
        ds.open()
        chunk = ds.read_chunk()
        assert np.isfinite(chunk).all()

    def test_close_then_read_does_not_raise(self):
        """DummyStream.close() is a no-op — reading after close should work."""
        ds = DummyStream()
        ds.close()
        chunk = ds.read_chunk()  # should not raise
        assert chunk.shape == (8, ds.chunk_size)
