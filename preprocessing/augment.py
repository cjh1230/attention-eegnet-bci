"""
On-the-fly EEG data augmentation for Motor Imagery.

All transforms operate on (C, T) or (N, C, T) tensors and are designed
to preserve the class label while introducing realistic variability.

Reference patterns:
- Gaussian noise: simulates sensor noise / ambient interference
- Channel dropout: simulates electrode disconnection
- Time shift: simulates slight trial alignment jitter
- Amplitude scaling: simulates impedance / skin conductivity variation
- Frequency masking: simulates narrowband interference
- Cropped training: overlapping sliding windows for temporal robustness
- Mixup: linear interpolation between trials for regularisation
"""
import numpy as np


def gaussian_noise(X: np.ndarray, sigma: float = 0.05, rng: np.random.RandomState = None) -> np.ndarray:
    """Add Gaussian noise scaled to per-channel std."""
    if rng is None:
        rng = np.random.RandomState()
    noise = rng.randn(*X.shape).astype(X.dtype)
    scale = sigma * X.std(axis=-1, keepdims=True)
    return X + scale * noise


def channel_dropout(X: np.ndarray, p: float = 0.1, rng: np.random.RandomState = None) -> np.ndarray:
    """Randomly zero out channels with probability p."""
    if rng is None:
        rng = np.random.RandomState()
    result = X.copy()
    C = X.shape[-2]
    mask = rng.rand(C) > p
    result[..., ~mask, :] = 0.0
    return result


def time_shift(X: np.ndarray, max_shift: int = 13, rng: np.random.RandomState = None) -> np.ndarray:
    """Randomly shift the signal along the time axis (circular)."""
    if rng is None:
        rng = np.random.RandomState()
    shift = rng.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return X.copy()
    return np.roll(X, shift, axis=-1)


def amplitude_scale(X: np.ndarray, scale_range: tuple = (0.8, 1.2), rng: np.random.RandomState = None) -> np.ndarray:
    """Randomly scale amplitude per trial."""
    if rng is None:
        rng = np.random.RandomState()
    lo, hi = scale_range
    factor = rng.uniform(lo, hi)
    return X * factor


# ---------------------------------------------------------------------------
# New augmentations
# ---------------------------------------------------------------------------

def frequency_mask(
    X: np.ndarray,
    max_width_hz: float = 4.0,
    fs: int = 250,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """Mask a random narrow frequency band in each trial (simulates interference).

    Parameters
    ----------
    X : np.ndarray, shape (..., C, T)
    max_width_hz : float
        Maximum width of the masked band in Hz.
    fs : int
        Sampling frequency.
    rng : RandomState or None

    Returns
    -------
    np.ndarray, same shape as X
    """
    if rng is None:
        rng = np.random.RandomState()
    T = X.shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    max_bins = int(max_width_hz / (fs / T))

    result = X.copy()
    for idx in np.ndindex(X.shape[:-2]):
        trial = result[idx]
        fft = np.fft.rfft(trial, axis=-1)
        hi_limit = max(3, fs / 2 - max_width_hz - 1)
        lo_hz = rng.uniform(2, hi_limit)
        width = rng.randint(1, max(max_bins, 1) + 1)
        lo_bin = int(lo_hz / (fs / T))
        hi_bin = min(lo_bin + width, len(freqs))
        fft[..., lo_bin:hi_bin] = 0.0
        result[idx] = np.fft.irfft(fft, n=T, axis=-1)
    return result


def crop_augment(
    X: np.ndarray,
    y: np.ndarray,
    window: int = 500,
    stride: int = 125,
) -> tuple[np.ndarray, np.ndarray]:
    """Create overlapping sliding-window crops from each trial.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray, shape (N,)
    window : int
        Crop window length in samples (500 = 2s @ 250 Hz).
    stride : int
        Stride between crops in samples.

    Returns
    -------
    X_crops : np.ndarray, shape (N * k, C, window)
    y_crops : np.ndarray, shape (N * k,)
    """
    N, C, T = X.shape
    if window >= T:
        return X, y

    crops_X, crops_y = [], []
    n_crops = max(1, (T - window) // stride + 1)

    for i in range(N):
        for k in range(n_crops):
            start = k * stride
            end = start + window
            if end > T:
                break
            crops_X.append(X[i, :, start:end])
            crops_y.append(y[i])

    return np.array(crops_X, dtype=X.dtype), np.array(crops_y, dtype=y.dtype)


def mixup_batch(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
    rng: np.random.RandomState = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Apply mixup to a batch.

    X_mix = lam * X_i + (1-lam) * X_j

    Parameters
    ----------
    X : np.ndarray, shape (B, C, T)
    y : np.ndarray, shape (B,)
    alpha : float
        Beta distribution shape parameter.
    rng : RandomState or None

    Returns
    -------
    X_mixed : np.ndarray
    y_a : np.ndarray   -- first labels
    y_b : np.ndarray   -- second labels (shuffled)
    lam : float        -- mix ratio
    """
    if rng is None:
        rng = np.random.RandomState()
    B = X.shape[0]
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    lam = max(lam, 1.0 - lam)
    indices = rng.permutation(B)
    X_mixed = lam * X + (1.0 - lam) * X[indices]
    return X_mixed, y, y[indices], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup-aware loss: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def eeg_augment(
    X: np.ndarray,
    y: np.ndarray = None,
    noise_sigma: float = 0.05,
    dropout_p: float = 0.1,
    max_shift: int = 13,
    scale_range: tuple = (0.8, 1.2),
    freq_mask_hz: float = 0.0,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Apply a random augmentation pipeline to EEG trials.

    Each augmentation is applied independently with probability ~0.5.

    Parameters
    ----------
    X : np.ndarray, shape (N, C, T)
    y : np.ndarray or None, shape (N,)
    noise_sigma : float
    dropout_p : float
    max_shift : int
    scale_range : tuple (lo, hi)
    freq_mask_hz : float
        If > 0, enable frequency masking with this max width in Hz.
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
        if freq_mask_hz > 0 and rng.rand() < 0.5:
            trial = frequency_mask(trial, max_width_hz=freq_mask_hz, rng=rng)
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
    """Augment a full dataset by generating `factor` augmented copies.

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
