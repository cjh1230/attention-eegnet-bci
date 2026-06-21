"""
Epoch creation from MNE Raw objects.
"""
import mne
import numpy as np

from utils.config import EVENT_IDS, T_MAX, T_MIN


def epoch(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict = None,
    tmin: float = T_MIN,
    tmax: float = T_MAX,
) -> mne.Epochs:
    """Create MNE Epochs from raw data and events."""
    if event_id is None:
        event_id = EVENT_IDS
    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(tmin, 0),
        preload=True,
    )


def to_array(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """Convert MNE Epochs to [N, C, T] data and [N] labels."""
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y
