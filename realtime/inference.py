"""
Real-time MI inference loop.

Connects stream → buffer → model → action output.

Supports idle-state confidence gating to prevent false triggers
during online demonstrations.  See MIInference.predict_with_gating().
"""
import numpy as np
import torch

# ── Idle gating constants ─────────────────────────────────────────────
IDLE_CLASS = 0
DEFAULT_CONFIDENCE_THRESHOLD = 0.65

# Canonical action names (index = canonical label)
DEFAULT_ACTION_MAP: dict[int, str] = {
    0: "STOP",
    1: "LEFT",
    2: "RIGHT",
    3: "FEET",
    4: "TONGUE",
}


class MIInference:
    """
    Real-time inference wrapper for a trained PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained EEGNet (or similar) model.
    buffer : RingBuffer
        Sliding-window buffer.
    device : str
        Torch device.
    action_map : dict[int, str], optional
        Mapping from class_id → action name.
        Default: {0: "STOP", 1: "LEFT", 2: "RIGHT", ...}
    """

    def __init__(
        self,
        model: torch.nn.Module,
        buffer,
        device: str = "cpu",
        action_map: dict[int, str] | None = None,
    ):
        self.model = model.to(device).eval()
        self.buffer = buffer
        self.device = device
        self.action_map = action_map or dict(DEFAULT_ACTION_MAP)

    def predict(self) -> tuple[int, float]:
        """
        Read buffer, run inference, return (class_id, confidence).

        Returns
        -------
        class_id : int
            0=Idle, 1=Left, 2=Right
        confidence : float
            Softmax probability of the predicted class.
        """
        data = self.buffer.read()  # (C, T)
        tensor = torch.from_numpy(data).unsqueeze(0).to(self.device)  # (1, C, T)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)
            class_id = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs.max().item())

        return class_id, confidence

    def predict_with_gating(
        self,
        threshold: float | None = None,
    ) -> tuple[str, int, float]:
        """
        Predict with idle-state confidence gating.

        Gate logic:
          1. If predicted class is IDLE → "STOP" (no action)
          2. If confidence < threshold → "STOP" (unsure → no action)
          3. Otherwise → mapped action

        This prevents false triggers during online demos by only
        issuing commands when the model is confident AND the class
        is a valid MI class.

        Parameters
        ----------
        threshold : float or None
            Confidence threshold (0–1). Uses DEFAULT_CONFIDENCE_THRESHOLD if None.

        Returns
        -------
        action : str
            Action name (e.g. "STOP", "LEFT", "RIGHT").
        class_id : int
            Raw predicted class.
        confidence : float
            Softmax probability of the predicted class.
        """
        class_id, confidence = self.predict()
        thresh = threshold if threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD

        if class_id == IDLE_CLASS:
            action = "STOP"
        elif confidence < thresh:
            action = "STOP"
        else:
            action = self.action_map.get(class_id, "STOP")

        return action, class_id, confidence
