"""
End-to-end MNE preprocessing pipeline.

Usage:
    from preprocessing.mne_pipeline import run_pipeline
    X, y = run_pipeline("data/raw/subject01.edf", events_array)
"""
import numpy as np

from preprocessing.filtering import bandpass, notch, load_raw
from preprocessing.artifact import apply_ica
from preprocessing.epoching import epoch, to_array


def run_pipeline(
    file_path: str,
    events: np.ndarray,
    apply_ica_flag: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Execute the full MNE pipeline and return (X, y) arrays.

    Parameters
    ----------
    file_path : str
        Path to raw EEG file.
    events : np.ndarray
        MNE events array (n_events, 3).
    apply_ica_flag : bool
        Whether to apply ICA artifact removal.

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)
    """
    raw = load_raw(file_path)
    raw = notch(raw)
    raw = bandpass(raw, l_freq=8.0, h_freq=30.0)

    if apply_ica_flag:
        raw = apply_ica(raw)

    epochs_obj = epoch(raw, events)
    X, y = to_array(epochs_obj)
    return X, y
