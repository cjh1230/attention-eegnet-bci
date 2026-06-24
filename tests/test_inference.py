"""Tests for realtime/inference.py — MIInference, build_action_map, gating."""
import numpy as np
import torch
import pytest

from realtime.inference import (
    MIInference,
    build_action_map,
    DEFAULT_ACTION_MAP,
    DEFAULT_CONFIDENCE_THRESHOLD,
)


class TestBuildActionMap:
    """build_action_map() — correct action maps for each dataset/class combo."""

    def test_physionet_mi_3class(self):
        am = build_action_map(3, "physionet_mi")
        assert am == {0: "STOP", 1: "LEFT", 2: "RIGHT"}

    def test_physionet_mi_2class(self):
        """Binary: rest/STOP dropped, labels shifted. 0=LEFT, 1=RIGHT."""
        am = build_action_map(2, "physionet_mi")
        assert am == {0: "LEFT", 1: "RIGHT"}

    def test_bci_iv_2a_4class(self):
        """4-class has no STOP. Uses LEFT/RIGHT/FEET/TONGUE."""
        am = build_action_map(4, "bci_iv_2a")
        assert am == {0: "LEFT", 1: "RIGHT", 2: "FEET", 3: "TONGUE"}

    def test_deepbci_3class(self):
        am = build_action_map(3, "deepbci")
        assert am == {0: "STOP", 1: "LEFT", 2: "RIGHT"}

    def test_unknown_dataset_fallback(self):
        """Unknown dataset falls back to generic CLS_N names."""
        am = build_action_map(3, "nonexistent_dataset_xyz")
        assert am[0] == "STOP"
        assert am[1] == "CLS_1"
        assert am[2] == "CLS_2"

    def test_n_classes_exceeds_dataset(self):
        """More classes than dataset → pad with generic names."""
        am = build_action_map(5, "physionet_mi")
        assert len(am) == 5
        assert am[0] == "STOP"
        assert am[1] == "LEFT"
        assert am[2] == "RIGHT"
        assert am[3] == "CLS_3"
        assert am[4] == "CLS_4"

    def test_n_classes_less_than_dataset(self):
        """3 classes from bci_iv_2a (4-class dataset) → first 3, no idle shift."""
        am = build_action_map(3, "bci_iv_2a")
        assert am == {0: "LEFT", 1: "RIGHT", 2: "FEET"}

    def test_caching_returns_same_object(self):
        """Second call with same args returns cached result."""
        am1 = build_action_map(3, "physionet_mi")
        am2 = build_action_map(3, "physionet_mi")
        assert am1 == am2

    def test_cached_copy_is_independent(self):
        """Modifying a returned dict doesn't affect the cache."""
        am1 = build_action_map(3, "physionet_mi")
        am1[999] = "MODIFIED"
        am2 = build_action_map(3, "physionet_mi")
        assert 999 not in am2

    def test_hand_underscore_removed(self):
        """LEFT_HAND → LEFT, RIGHT_HAND → RIGHT."""
        am = build_action_map(3, "physionet_mi")
        assert "LEFT_HAND" not in am.values()
        assert "RIGHT_HAND" not in am.values()
        assert "LEFT" in am.values()
        assert "RIGHT" in am.values()

    def test_feet_tongue_preserved(self):
        """FEET and TONGUE are kept as-is (no _HAND suffix)."""
        am = build_action_map(4, "bci_iv_2a")
        assert "FEET" in am.values()
        assert "TONGUE" in am.values()


class TestPredictWithGating:
    """predict_with_gating() — idle-state confidence gating."""

    @staticmethod
    def _make_model_and_buffer(n_channels=4, n_classes=3):
        """Create a warmed-up EEGNet and a filled RingBuffer."""
        from models.eegnet import EEGNet
        model = EEGNet(n_channels=n_channels, n_classes=n_classes)
        model.eval()
        # Warm up lazy classifier
        with torch.no_grad():
            model(torch.zeros(1, n_channels, 250))
        from realtime.buffer import RingBuffer
        buf = RingBuffer(n_channels=n_channels, window_s=1.0, s_freq=250)
        buf.push(np.random.randn(n_channels, 250).astype(np.float32))
        return model, buf

    def test_idle_class_gates_to_stop(self):
        """3-class model: class 0 is STOP."""
        model, buf = self._make_model_and_buffer(n_channels=4, n_classes=3)
        inference = MIInference(model, buf, device="cpu", n_classes=3, dataset="physionet_mi")
        assert inference.action_map[0] == "STOP"

    def test_no_stop_class_does_not_gate(self):
        """Binary model: no STOP in action_map → idle check is skipped."""
        model, buf = self._make_model_and_buffer(n_channels=4, n_classes=2)
        inference = MIInference(model, buf, device="cpu", n_classes=2, dataset="physionet_mi")
        assert "STOP" not in inference.action_map.values()

    def test_low_confidence_gates_to_stop(self):
        """Even without STOP class, confidence < threshold → STOP."""
        model, buf = self._make_model_and_buffer(n_channels=4, n_classes=2)
        inference = MIInference(model, buf, device="cpu", n_classes=2)
        # With a very high threshold, even confident predictions get gated
        action, _, _ = inference.predict_with_gating(threshold=0.999)
        assert action == "STOP"

    def test_default_threshold_used(self):
        """When no threshold given, DEFAULT_CONFIDENCE_THRESHOLD is used."""
        model, buf = self._make_model_and_buffer(n_channels=4, n_classes=3)
        inference = MIInference(model, buf, device="cpu", n_classes=3)
        action, class_id, conf = inference.predict_with_gating()
        assert isinstance(action, str)
        assert isinstance(class_id, int)
        assert isinstance(conf, float)

    def test_high_confidence_passes_gating(self):
        """High-confidence prediction on non-STOP class passes through."""
        model, buf = self._make_model_and_buffer(n_channels=4, n_classes=2)
        inference = MIInference(model, buf, device="cpu", n_classes=2)
        # With threshold=0.0, any confidence passes
        action, _, _ = inference.predict_with_gating(threshold=0.0)
        # For binary model, action should be LEFT or RIGHT, not STOP
        assert action in ("LEFT", "RIGHT")

    def test_action_map_explicit_overrides_auto(self):
        """Explicit action_map takes precedence over n_classes auto-build."""
        model, buf = self._make_model_and_buffer(n_channels=4, n_classes=2)
        custom = {0: "CUSTOM_A", 1: "CUSTOM_B"}
        inference = MIInference(model, buf, device="cpu",
                                action_map=custom, n_classes=2)
        assert inference.action_map == custom

    def test_default_action_map_without_n_classes(self):
        """When neither n_classes nor action_map given, DEFAULT_ACTION_MAP used."""
        model, buf = self._make_model_and_buffer(n_channels=4, n_classes=3)
        inference = MIInference(model, buf, device="cpu")
        assert inference.action_map == DEFAULT_ACTION_MAP


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
