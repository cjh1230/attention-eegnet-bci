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
# NOTE: Prefer build_action_map() which respects n_classes and dataset semantics.
DEFAULT_ACTION_MAP: dict[int, str] = {
    0: "STOP",
    1: "LEFT",
    2: "RIGHT",
    3: "FEET",
    4: "TONGUE",
}

# ── Per-dataset action maps (built from label_mapping.py class_names) ──
_ACTION_MAP_CACHE: dict[tuple[int, str], dict[int, str]] = {}


def build_action_map(
    n_classes: int,
    dataset: str = "physionet_mi",
    *,
    idle_label: int = 0,
) -> dict[int, str]:
    """
    Build a class-id → action-name mapping for *n_classes*.

    Handles binary-vs-multiclass label shifting:
      - physionet_mi 3-class: 0=STOP, 1=LEFT, 2=RIGHT
      - physionet_mi 2-class: 0=LEFT, 1=RIGHT (rest dropped, labels shifted)
      - bci_iv_2a 4-class: 0=LEFT, 1=RIGHT, 2=FEET, 3=TONGUE
      - deepbci 3-class: 0=STOP, 1=LEFT, 2=RIGHT

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    dataset : str
        Dataset name.
    idle_label : int
        Fallback class ID for STOP/IDLE when dataset is unknown.

    Returns
    -------
    action_map : dict[int, str]
    """
    cache_key = (n_classes, dataset)
    if cache_key in _ACTION_MAP_CACHE:
        return dict(_ACTION_MAP_CACHE[cache_key])

    try:
        from datasets.label_mapping import LABEL_MAPS

        semantic = LABEL_MAPS.get(dataset, {})
        # Ordered names by canonical label value
        full_names = [n for n, _ in sorted(semantic.items(), key=lambda kv: kv[1])]
        full_n = len(full_names)

        if full_n == 0:
            # Unknown dataset — fall through to generic names
            selected = []
        elif n_classes < full_n and full_names[0].lower() in ("rest", "idle"):
            # Binary / reduced-class case: drop rest/idle, shift remaining down
            selected = full_names[1:n_classes + 1]
        elif n_classes <= full_n:
            selected = full_names[:n_classes]
        else:
            selected = full_names
            # Pad with generic names for extra classes
            selected += [f"CLS_{i}" for i in range(full_n, n_classes)]

        # If no names from dataset, fall back to generic
        if not selected:
            selected = [f"CLS_{i}" for i in range(n_classes)]
            if idle_label < n_classes:
                selected[idle_label] = "REST"  # will map to STOP below

        action_map = {}
        for i, name in enumerate(selected):
            upper = name.upper()
            if upper in ("REST", "IDLE"):
                action_map[i] = "STOP"
            elif upper.startswith("CLS_"):
                action_map[i] = upper  # keep CLS_N as-is
            else:
                # Shorten: LEFT_HAND→LEFT, RIGHT_HAND→RIGHT
                action_map[i] = upper.replace("_HAND", "")
    except (ValueError, ImportError):
        action_map = {i: f"CLS_{i}" for i in range(n_classes)}
        if idle_label < n_classes:
            action_map[idle_label] = "STOP"

    _ACTION_MAP_CACHE[cache_key] = action_map
    return dict(action_map)


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
    n_classes : int, optional
        Number of output classes. Used to auto-build action_map if not provided.
    action_map : dict[int, str], optional
        Mapping from class_id → action name.  If None, auto-built from n_classes.
    dataset : str
        Dataset name for semantic class labels (default "physionet_mi").
    """

    def __init__(
        self,
        model: torch.nn.Module,
        buffer,
        device: str = "cpu",
        n_classes: int | None = None,
        action_map: dict[int, str] | None = None,
        dataset: str = "physionet_mi",
    ):
        self.model = model.to(device).eval()
        self.buffer = buffer
        self.device = device
        if action_map is not None:
            self.action_map = action_map
        elif n_classes is not None:
            self.action_map = build_action_map(n_classes, dataset)
        else:
            self.action_map = dict(DEFAULT_ACTION_MAP)

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
          1. If action_map has a "STOP" class AND the prediction matches it → STOP
          2. If confidence < threshold → STOP (unsure → no action)
          3. Otherwise → mapped action

        The idle class is auto-detected from action_map — works correctly
        for binary (LEFT/RIGHT), 3-class (STOP/LEFT/RIGHT), and 4-class models.

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

        # Auto-detect idle class from action_map (binary models have no STOP)
        idle_class = next(
            (k for k, v in self.action_map.items() if v == "STOP"),
            None,
        )

        if idle_class is not None and class_id == idle_class:
            action = "STOP"
        elif confidence < thresh:
            action = "STOP"
        else:
            action = self.action_map.get(class_id, "STOP")

        return action, class_id, confidence
