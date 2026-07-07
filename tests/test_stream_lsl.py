"""Tests for realtime/stream_lsl.py — LSL stream reader.

Tests the LSLStream class behavior when pylsl is NOT installed.
The class raises ImportError in __init__ when pylsl is missing,
so we test that path and the module-level constant.
"""
import pytest


class TestLSLStreamNoPylsl:
    def test_init_raises_when_pylsl_missing(self, monkeypatch):
        """LSLStream.__init__ should raise ImportError when pylsl is not importable."""
        import realtime.stream_lsl as sl
        monkeypatch.setattr(sl, "HAS_LSL", False)
        with pytest.raises(ImportError, match="pylsl"):
            sl.LSLStream(name="Test")

    def test_has_lsl_constant(self):
        """HAS_LSL should be a boolean."""
        import realtime.stream_lsl as sl
        assert isinstance(sl.HAS_LSL, bool)

    def test_class_exists(self):
        """LSLStream class should be importable."""
        from realtime.stream_lsl import LSLStream
        assert LSLStream is not None


class TestLSLStreamInitWhenAvailable:
    def test_init_with_defaults(self, monkeypatch):
        """When pylsl IS available, LSLStream should init without error."""
        import realtime.stream_lsl as sl
        # Mock HAS_LSL = True to bypass the ImportError check
        monkeypatch.setattr(sl, "HAS_LSL", True)
        ds = sl.LSLStream()
        assert ds.name == "DeepBCI"
        assert ds.stream_type == "EEG"
        assert ds.n_channels == 8
        assert ds.s_freq == 250
        assert ds.timeout == 5.0
        assert ds.inlet is None

    def test_init_with_custom_params(self, monkeypatch):
        import realtime.stream_lsl as sl
        monkeypatch.setattr(sl, "HAS_LSL", True)
        ds = sl.LSLStream(
            name="CustomStream",
            stream_type="MEG",
            n_channels=32,
            s_freq=1000,
            timeout=10.0,
        )
        assert ds.name == "CustomStream"
        assert ds.stream_type == "MEG"
        assert ds.n_channels == 32
        assert ds.s_freq == 1000
        assert ds.timeout == 10.0

    def test_read_chunk_before_open_raises(self, monkeypatch):
        import realtime.stream_lsl as sl
        monkeypatch.setattr(sl, "HAS_LSL", True)
        ds = sl.LSLStream()
        with pytest.raises(RuntimeError, match="not open"):
            ds.read_chunk()

    def test_close_before_open_noop(self, monkeypatch):
        import realtime.stream_lsl as sl
        monkeypatch.setattr(sl, "HAS_LSL", True)
        ds = sl.LSLStream()
        ds.close()  # should not raise
        assert ds.inlet is None

    def test_close_after_open_no_inlet(self, monkeypatch):
        """close() should handle None inlet gracefully."""
        import realtime.stream_lsl as sl
        monkeypatch.setattr(sl, "HAS_LSL", True)
        ds = sl.LSLStream()
        ds.inlet = None
        ds.close()  # should not raise
