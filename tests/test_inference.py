"""Tests for realtime/inference.py — MIInference."""
import numpy as np
import torch

from realtime.inference import MIInference


class TestMIInference:
    @staticmethod
    def _make_buffer(rng, n_channels=8, window_s=1.0, s_freq=250):
        from realtime.buffer import RingBuffer
        buf = RingBuffer(n_channels=n_channels, window_s=window_s, s_freq=s_freq)
        buf.push(rng.randn(n_channels, int(window_s * s_freq)).astype(np.float32))
        return buf

    def test_predict_returns_tuple(self):
        from models.eegnet import EEGNet
        import numpy as np
        rng = np.random.RandomState(0)
        model = EEGNet(n_channels=8, n_classes=3)
        model.eval()
        buf = self._make_buffer(rng, n_channels=8)
        infer = MIInference(model, buf, device="cpu")
        class_id, conf = infer.predict()
        assert isinstance(class_id, int)
        assert isinstance(conf, float)

    def test_class_id_in_range(self):
        from models.eegnet import EEGNet
        rng = np.random.RandomState(0)
        model = EEGNet(n_channels=8, n_classes=3)
        model.eval()
        buf = self._make_buffer(rng, n_channels=8)
        infer = MIInference(model, buf, device="cpu")
        class_id, _ = infer.predict()
        assert class_id in {0, 1, 2}

    def test_confidence_between_0_and_1(self):
        from models.eegnet import EEGNet
        rng = np.random.RandomState(0)
        model = EEGNet(n_channels=8, n_classes=3)
        model.eval()
        buf = self._make_buffer(rng, n_channels=8)
        infer = MIInference(model, buf, device="cpu")
        _, conf = infer.predict()
        assert 0.0 <= conf <= 1.0

    def test_deterministic_in_eval(self):
        """Same buffer → same prediction (no randomness in eval mode)."""
        from models.eegnet import EEGNet
        rng = np.random.RandomState(0)
        model = EEGNet(n_channels=8, n_classes=3)
        model.eval()
        buf = self._make_buffer(rng, n_channels=8)
        infer = MIInference(model, buf, device="cpu")
        c1, cf1 = infer.predict()
        c2, cf2 = infer.predict()
        assert c1 == c2
        assert abs(cf1 - cf2) < 1e-6

    def test_device_cpu(self):
        from models.eegnet import EEGNet
        rng = np.random.RandomState(0)
        model = EEGNet(n_channels=4, n_classes=2)
        model.eval()
        buf = self._make_buffer(rng, n_channels=4)
        infer = MIInference(model, buf, device="cpu")
        assert infer.device == "cpu"
        class_id, _ = infer.predict()
        assert class_id in {0, 1}
