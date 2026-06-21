"""
MNE-based filtering functions for MI-EEG data.

Steps:
  1. Load raw EEG file
  2. Bandpass filter (8–30 Hz)
  3. Notch filter (50 Hz)
"""
from pathlib import Path

import mne


def load_raw(file_path: str | Path) -> mne.io.Raw:
    """Load raw EEG file. Supports .fif, .edf, .gdf, .set."""
    return mne.io.read_raw(file_path, preload=True)


def bandpass(raw: mne.io.Raw, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.io.Raw:
    """Apply zero-phase bandpass filter."""
    return raw.copy().filter(l_freq, h_freq, fir_design="firwin")


def notch(raw: mne.io.Raw, freq: float = 50.0) -> mne.io.Raw:
    """Apply notch filter for power-line noise."""
    return raw.copy().notch_filter(freq, fir_design="firwin")
