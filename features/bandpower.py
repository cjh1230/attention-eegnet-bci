"""
Band power feature extraction (mu / beta ratios).
"""
import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(data, low, high, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def bandpower_ratio(X, fs=250):
    """
    Compute mu/beta bandpower ratio per trial.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    fs : int
        Sampling frequency.

    Returns
    -------
    ratios : np.ndarray, shape (n_trials, n_channels)
    """
    mu = _bandpass(X, 8, 13, fs)
    beta = _bandpass(X, 13, 30, fs)
    mu_power = np.var(mu, axis=-1)
    beta_power = np.var(beta, axis=-1)
    # Avoid division by zero
    ratios = mu_power / (beta_power + 1e-8)
    return ratios
