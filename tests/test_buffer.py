"""Tests for realtime/buffer.py — RingBuffer."""
import numpy as np

from realtime.buffer import RingBuffer


class TestRingBuffer:
    def test_init_shape(self):
        buf = RingBuffer(n_channels=8, window_s=1.0, s_freq=250)
        assert buf.buffer.shape == (8, 250)

    def test_read_returns_copy(self, ring_buffer):
        data = ring_buffer.read()
        data[0, 0] = 999.0
        # Original buffer should be unchanged
        assert ring_buffer.buffer[0, 0] != 999.0

    def test_push_and_read(self, ring_buffer, rng):
        samples = rng.randn(8, 50).astype(np.float32)
        ring_buffer.push(samples)
        buf = ring_buffer.read()
        # write_pos starts at 0 — pushed 50 samples occupy buf[:, :50]
        assert np.allclose(buf[:, :50], samples)
        # Unfilled portion should still be zeros
        assert np.all(buf[:, 50:] == 0)

    def test_overwrite_wraps_correctly(self, rng):
        buf = RingBuffer(n_channels=2, window_s=1.0, s_freq=10)  # 10 samples
        # Fill completely
        first = rng.randn(2, 10).astype(np.float32)
        buf.push(first)
        # Push more — should overwrite from start
        second = rng.randn(2, 5).astype(np.float32)
        buf.push(second)
        result = buf.read()
        # First 5 should be overwritten
        assert np.allclose(result[:, :5], second[:, -5:])
        # Last 5 should still be from first batch
        assert np.allclose(result[:, 5:], first[:, 5:])

    def test_reset_zeros_buffer(self, ring_buffer, rng):
        ring_buffer.push(rng.randn(8, 100).astype(np.float32))
        ring_buffer.reset()
        assert np.all(ring_buffer.read() == 0)

    def test_thread_safety_smoke(self, ring_buffer, rng):
        """Basic smoke test: push/read from multiple virtual 'threads'."""
        import threading
        errors = []

        def worker():
            try:
                for _ in range(10):
                    data = rng.randn(8, 10).astype(np.float32)
                    ring_buffer.push(data)
                    _ = ring_buffer.read()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_read_shape_constant(self, ring_buffer, rng):
        """Buffer shape should not change after pushes."""
        shape_before = ring_buffer.read().shape
        ring_buffer.push(rng.randn(8, 30).astype(np.float32))
        shape_after = ring_buffer.read().shape
        assert shape_before == shape_after
