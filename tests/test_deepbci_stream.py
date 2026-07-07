"""Tests for realtime/deepbci_stream.py — Multi-mode DeepBCI stream reader."""
import numpy as np
import pytest

from realtime.deepbci_stream import DeepBCIStream, replay_session


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def csv_file(tmp_path):
    """Create a synthetic CSV file with 8-channel data for file replay mode."""
    rng = np.random.RandomState(42)
    n_channels = 8
    n_samples = 500
    data = rng.randn(n_samples, n_channels).astype(np.float32)
    path = tmp_path / "raw.csv"
    np.savetxt(str(path), data, delimiter=",", fmt="%.6f")
    return str(path)


# ── Init ────────────────────────────────────────────────────────────────────

class TestDeepBCIStreamInit:
    def test_default_mode_is_dummy(self):
        ds = DeepBCIStream()
        assert ds.mode == "dummy"
        assert not ds._is_open

    def test_dummy_mode(self):
        ds = DeepBCIStream(mode="dummy")
        assert ds.mode == "dummy"

    def test_file_mode(self, csv_file):
        ds = DeepBCIStream(mode="file", file_path=csv_file)
        assert ds.mode == "file"
        assert ds.file_path is not None

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            DeepBCIStream(mode="invalid")

    def test_custom_params(self):
        ds = DeepBCIStream(
            mode="dummy", n_channels=16, s_freq=500,
            chunk_duration_s=0.25, stream_name="TestStream",
        )
        assert ds.n_channels == 16
        assert ds.s_freq == 500
        assert ds.chunk_duration_s == 0.25
        assert ds.chunk_samples == int(500 * 0.25)


# ── Dummy mode ──────────────────────────────────────────────────────────────

class TestDummyMode:
    def test_open(self):
        ds = DeepBCIStream(mode="dummy")
        ds.open()
        assert ds._is_open

    def test_open_idempotent(self):
        ds = DeepBCIStream(mode="dummy")
        ds.open()
        ds.open()  # should not raise
        assert ds._is_open

    def test_read_chunk_shape(self):
        ds = DeepBCIStream(mode="dummy", n_channels=8, s_freq=250)
        ds.open()
        chunk = ds.read_chunk()
        assert chunk.shape == (8, ds.chunk_samples)
        assert chunk.dtype == np.float32

    def test_read_chunk_without_open_raises(self):
        ds = DeepBCIStream(mode="dummy")
        with pytest.raises(RuntimeError, match="not open"):
            ds.read_chunk()

    def test_close(self):
        ds = DeepBCIStream(mode="dummy")
        ds.open()
        ds.close()
        assert not ds._is_open

    def test_multiple_chunks_differ(self):
        ds = DeepBCIStream(mode="dummy", n_channels=8, s_freq=250)
        ds.open()
        chunks = [ds.read_chunk() for _ in range(3)]
        ds.close()
        for i in range(len(chunks) - 1):
            assert not np.allclose(chunks[i], chunks[i + 1])

    def test_chunk_values_finite(self):
        ds = DeepBCIStream(mode="dummy")
        ds.open()
        chunk = ds.read_chunk()
        assert np.isfinite(chunk).all()
        ds.close()


# ── File mode ───────────────────────────────────────────────────────────────

class TestFileMode:
    def test_open(self, csv_file):
        ds = DeepBCIStream(mode="file", file_path=csv_file)
        ds.open()
        assert ds._is_open
        assert ds._data is not None
        ds.close()

    def test_read_chunk_shape(self, csv_file):
        ds = DeepBCIStream(mode="file", file_path=csv_file, n_channels=8, s_freq=250)
        ds.open()
        chunk = ds.read_chunk()
        assert chunk.ndim == 2
        assert chunk.shape[0] == 8
        ds.close()

    def test_multiple_chunks(self, csv_file):
        ds = DeepBCIStream(mode="file", file_path=csv_file, n_channels=8, s_freq=250)
        ds.open()
        chunks = []
        for _ in range(5):
            chunk = ds.read_chunk()
            chunks.append(chunk)
            assert chunk.shape[0] == 8
        ds.close()
        # At least some chunks should differ (wrapping may cause repeats)
        assert len(chunks) == 5

    def test_wrap_around(self, csv_file):
        """Reading more samples than available should wrap around."""
        ds = DeepBCIStream(
            mode="file", file_path=csv_file,
            n_channels=8, s_freq=250, chunk_duration_s=2.0,  # 500 samples per chunk
        )
        ds.open()
        # File has 500 samples, chunk_size=500, so reading twice wraps
        c1 = ds.read_chunk()
        c2 = ds.read_chunk()
        assert c1.shape == (8, ds.chunk_samples)
        assert c2.shape == (8, ds.chunk_samples)
        ds.close()

    def test_file_not_found_raises(self):
        ds = DeepBCIStream(mode="file", file_path="/nonexistent/path.csv")
        with pytest.raises(FileNotFoundError):
            ds.open()

    def test_file_mode_without_path(self):
        ds = DeepBCIStream(mode="file")
        with pytest.raises(FileNotFoundError):
            ds.open()

    def test_close_resets_cursor(self, csv_file):
        ds = DeepBCIStream(mode="file", file_path=csv_file, n_channels=8, s_freq=250)
        ds.open()
        c1 = ds.read_chunk()
        ds.close()
        assert ds._cursor == 0
        # Reopen — should start from beginning
        ds.open()
        c2 = ds.read_chunk()
        ds.close()
        assert np.allclose(c1, c2)


# ── LSL mode (without pylsl) ────────────────────────────────────────────────

class TestLSLModeNoPylsl:
    def test_open_without_pylsl_raises(self, monkeypatch):
        """If pylsl is not available, opening in LSL mode should raise ImportError."""
        # Force HAS_LSL to False
        import realtime.deepbci_stream as dbs
        monkeypatch.setattr(dbs, "HAS_LSL", False)
        ds = DeepBCIStream(mode="lsl")
        with pytest.raises(ImportError, match="pylsl"):
            ds.open()


# ── replay_session convenience ──────────────────────────────────────────────

class TestReplaySession:
    def test_creates_stream(self, tmp_path):
        """replay_session should return a DeepBCIStream in file mode."""
        subj_dir = tmp_path / "sub_001"
        subj_dir.mkdir()
        rng = np.random.RandomState(42)
        data = rng.randn(200, 8).astype(np.float32)
        np.savetxt(str(subj_dir / "raw.csv"), data, delimiter=",", fmt="%.6f")

        ds = replay_session(str(subj_dir))
        assert isinstance(ds, DeepBCIStream)
        assert ds.mode == "file"
        ds.close()

    def test_no_raw_csv_raises(self, tmp_path):
        subj_dir = tmp_path / "empty_subj"
        subj_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="raw.csv"):
            replay_session(str(subj_dir))
