"""
On-the-fly EEG data augmentation for Motor Imagery.

All transforms operate on (C, T) or (N, C, T) tensors and are designed
to preserve the class label while introducing realistic variability.

Reference patterns:
- Gaussian noise: simulates sensor noise / ambient interference
- Channel dropout: simulates electrode disconnection
- Time shift: simulates slight trial alignment jitter
- Amplitude scaling: simulates impedance / skin conductivity variation
"""
import numpy as np


def gaussian_noise(X: np.ndarray, sigma: float = 0.05, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Add Gaussian noise scaled to per-channel std.

    Parameters
    ----------
    X : np.ndarray, shape (..., C, T)
    sigma : float
        Noise standard deviation relative to per-channel std.
    rng : RandomState or None

    Returns
    -------
    np.ndarray, same shape as X
    """
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.randn(*X.shape).astype(X.dtype)
    scale = sigma * X.std(axis=-1, keepdims=True)
    return X + scale * noise


def channel_dropout(X: np.ndarray, p: float = 0.1, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Randomly zero out channels with probability p.

    Parameters
    ----------
    X : np.ndarray, shape (..., C, T)
    p : float
        Probability of dropping each channel.
    rng : RandomState or None

    Returns
    -------
    np.ndarray, same shape as X
    """
    if rng is None:
        rng = np.random.RandomState()
    result = X.copy()
    C = X.shape[-2]  # channel dim
    mask = rng.rand(C) > p
    # Expand mask to broadcast over non-channel dims
    ndim = X.ndim
    # Build slicer: select channel axis for mask
    idx = [slice(None)] * ndim
    idx[-2] = mask
    result[..., ~mask, :] = 0.0
    return result


def time_shift(X: np.ndarray, max_shift: int = 13, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Randomly shift the signal along the time axis (circular).

    Parameters
    ----------
    X : np.ndarray, shape (..., C, T)
    max_shift : int
        Maximum shift in samples (13 ≈ 50ms @ 250 Hz).
    rng : RandomState or None

    Returns
    -------
    np.ndarray, same shape as X
    """
    if rng is None:
        rng = np.random.RandomState()
    shift = rng.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return X.copy()
    return np.roll(X, shift, axis=-1)


def amplitude_scale(X: np.ndarray, scale_range: tuple = (0.8, 1.2), rng: np.random.RandomState = None) -> np.ndarray:
    """
    Randomly scale amplitude per trial.

    Parameters
    ----------
    X : np.ndarray, shape (..., C, T)
    scale_range : tuple (lo, hi)
    rng : RandomState or None

    Returns
    -------
    np.ndarray, same shape as X
    """
    if rng is None:
        rng = np.random.RandomState()
    lo, hi = scale_range
    factor = rng.uniform(lo, hi)
    return X * factor


def eeg_augment(
    X: np.ndarray,
    y: np.ndarray = None,
    noise_sigma: float = 0.05,
    dropout_p: float = 0.1,
    max_shift: int = 13,
    scale_range: tuple = (0.8, 1.2),
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Apply a random augmentation pipeline to EEG trials.

    Each augmentation is applied independently with probability ~0.5.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray or None, shape (N,)
    noise_sigma : float
    dropout_p : float
    max_shift : int
    scale_range : tuple (lo, hi)
    seed : int or None

    Returns
    -------
    X_aug : np.ndarray
    y_aug : np.ndarray (only if y is provided)
    """
    rng = np.random.RandomState(seed)
    X_aug = X.copy()

    for i in range(len(X_aug)):
        trial = X_aug[i]
        if rng.rand() < 0.5:
            trial = gaussian_noise(trial, sigma=noise_sigma, rng=rng)
        if rng.rand() < 0.5:
            trial = channel_dropout(trial, p=dropout_p, rng=rng)
        if rng.rand() < 0.5:
            trial = time_shift(trial, max_shift=max_shift, rng=rng)
        if rng.rand() < 0.5:
            trial = amplitude_scale(trial, scale_range=scale_range, rng=rng)
        X_aug[i] = trial

    if y is not None:
        return X_aug, y
    return X_aug


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    factor: int = 2,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment a full dataset by generating `factor` augmented copies.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)
    factor : int
        Number of augmented copies per original trial.
    seed : int or None

    Returns
    -------
    X_out : np.ndarray, shape (N * factor, C, T)
    y_out : np.ndarray, shape (N * factor,)
    """
    parts = [X]
    parts_y = [y]
    for k in range(factor - 1):
        X_aug, y_aug = eeg_augment(X, y, seed=None if seed is None else seed + k)
        parts.append(X_aug)
        parts_y.append(y_aug)
    return np.concatenate(parts, axis=0), np.concatenate(parts_y, axis=0)
