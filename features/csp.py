"""
Common Spatial Patterns (CSP) feature extraction — MI baseline.

Supports single-band CSP and Filter Bank CSP (FBCSP, Ang et al. 2008)
with LDA or SVM classifiers.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
from scipy.signal import butter, filtfilt


# ---------------------------------------------------------------------------
# Bandpass helper (scipy — no MNE dependency for filtering)
# ---------------------------------------------------------------------------

def _bandpass(data: np.ndarray, low: float, high: float,
              fs: int = 250, order: int = 4) -> np.ndarray:
    """Zero-phase bandpass filter along the last axis."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def _make_classifier(name: str):
    """Return an sklearn-compatible classifier instance."""
    if name == "svm":
        return SVC(kernel="linear", C=1.0, class_weight="balanced")
    elif name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr")
    else:
        raise ValueError(f"Unknown classifier: {name}. Choose 'svm' or 'lda'.")


# ---------------------------------------------------------------------------
# Single-band CSP baselines (kept for backward compatibility)
# ---------------------------------------------------------------------------

def csp_svm_baseline(X, y, n_components=6, cv=5) -> dict:
    """
    Train CSP + linear SVM and return cross-validated results.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)
    n_components : int
        Number of CSP components.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict with keys: accuracy, accuracy_std, scores
    """
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    svm = SVC(kernel="linear", C=1.0, class_weight="balanced")
    pipeline = make_pipeline(csp, svm)

    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return {
        "accuracy": scores.mean(),
        "accuracy_std": scores.std(),
        "scores": scores.tolist(),
    }


def csp_lda_baseline(X, y, n_components=6, cv=5) -> dict:
    """
    Train CSP + LDA and return cross-validated results.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)
    n_components : int
        Number of CSP components.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict with keys: accuracy, accuracy_std, scores
    """
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr")
    pipeline = make_pipeline(csp, lda)

    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return {
        "accuracy": scores.mean(),
        "accuracy_std": scores.std(),
        "scores": scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Filter Bank CSP (FBCSP)
# ---------------------------------------------------------------------------

class FilterBankCSP(BaseEstimator, TransformerMixin):
    """sklearn-compatible FBCSP feature extractor (per-band CSP + log-var).

    CSP is fitted during ``fit()`` and applied during ``transform()``.
    When used inside a sklearn ``Pipeline`` with ``cross_val_score``, CSP
    spatial filters are fitted only on each fold's training data, preventing
    data leakage.

    Parameters
    ----------
    freq_bands : list of (low, high) or None
    n_components : int
        CSP components per band.
    fs : int
        Sampling frequency.
    """

    def __init__(
        self,
        freq_bands: list[tuple[float, float]] | None = None,
        n_components: int = 4,
        fs: int = 250,
    ) -> None:
        self.freq_bands = freq_bands
        self.n_components = n_components
        self.fs = fs
        self._csps: list[CSP] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FilterBankCSP":
        bands = self.freq_bands
        if bands is None:
            from utils.config import FBCSP_BANDS
            bands = FBCSP_BANDS

        self._csps = []
        for low, high in bands:
            X_filt = _bandpass(X, low, high, fs=self.fs)
            csp = CSP(n_components=self.n_components, reg=None,
                      log=True, norm_trace=False)
            csp.fit(X_filt.copy(), y)
            self._csps.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._csps:
            raise RuntimeError("FilterBankCSP must be fitted before transform().")
        bands = list(self._get_bands())
        all_features = []
        for (low, high), csp in zip(bands, self._csps):
            X_filt = _bandpass(X, low, high, fs=self.fs)
            feats = csp.transform(X_filt.copy())
            all_features.append(feats.astype(np.float32))
        return np.concatenate(all_features, axis=1)

    def _get_bands(self):
        bands = self.freq_bands
        if bands is None:
            from utils.config import FBCSP_BANDS
            bands = FBCSP_BANDS
        return bands


def fbcsp_features(
    X: np.ndarray,
    y: np.ndarray,
    freq_bands: list[tuple[float, float]] | None = None,
    n_components: int = 4,
    fs: int = 250,
) -> np.ndarray:
    """Extract FBCSP features — convenience wrapper around FilterBankCSP.

    **WARNING**: This fits CSP on the full dataset.  For cross-validation,
    use ``FilterBankCSP`` inside a sklearn ``Pipeline`` instead, to ensure
    CSP is fitted per-fold.
    """
    fb = FilterBankCSP(freq_bands=freq_bands, n_components=n_components, fs=fs)
    fb.fit(X, y)
    return fb.transform(X)


def fbcsp_classify(
    X: np.ndarray,
    y: np.ndarray,
    freq_bands: list[tuple[float, float]] | None = None,
    n_components: int = 4,
    classifier: str = "lda",
    cv: int = 5,
    fs: int = 250,
) -> dict:
    """FBCSP + classifier with **per-fold CSP fitting** (no data leakage).

    Uses a sklearn ``Pipeline(FilterBankCSP(), classifier)`` inside
    ``cross_val_score`` so that CSP spatial filters are fitted only on
    each fold's training data.
    """
    if freq_bands is None:
        from utils.config import FBCSP_BANDS
        freq_bands = FBCSP_BANDS

    fb = FilterBankCSP(freq_bands=freq_bands, n_components=n_components, fs=fs)
    clf = _make_classifier(classifier)
    pipeline = make_pipeline(fb, clf)

    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    return {
        "accuracy": scores.mean(),
        "accuracy_std": scores.std(),
        "scores": scores.tolist(),
    }
