"""
Riemannian Geometry baseline for MI-BCI classification.

Provides sklearn-compatible pipelines:
  - Tangent Space + LDA/SVM  (strongest baseline)
  - Minimum Distance to Mean (simple baseline)
  - Filter-bank Riemannian   (FgMDM analog)

Dependencies
------------
pyriemann : for SPD manifold operations (covariance estimation, tangent space,
            MDM, FgMDM)
scikit-learn : for LDA, SVM, Pipeline, cross_val_score

References
----------
- Barachant et al., "Multiclass Brain-Computer Interface Classification
  by Riemannian Geometry" (IEEE TBME, 2012)
- Congedo et al., "Riemannian geometry for EEG-based brain-computer
  interfaces; a primer and a review" (Brain-Computer Interfaces, 2017)
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from scipy.signal import butter, filtfilt

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM

    HAS_PYRIEMANN = True
except ImportError:
    HAS_PYRIEMANN = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COV_ESTIMATORS = {"scm", "lwf", "oas", "mcd"}
RIEMANN_METRICS = {"riemann", "euclid", "logchol", "logeuclid", "wasserstein"}
VALID_METHODS = {"tangent", "mdm", "fgmdm"}

# MI-adapted filter bank: 6 sub-bands within 8–30 Hz (mu + beta rhythms)
_DEFAULT_BANDS = [
    (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 30),
]


# ---------------------------------------------------------------------------
# Bandpass helper (shared logic with features/csp.py)
# ---------------------------------------------------------------------------

def _bandpass(
    data: np.ndarray, low: float, high: float,
    fs: int = 250, order: int = 4,
) -> np.ndarray:
    """Zero-phase bandpass filter along the last axis."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_cov_estimator(estimator: str) -> None:
    if estimator not in COV_ESTIMATORS:
        raise ValueError(
            f"Unknown cov_estimator '{estimator}'. "
            f"Choose from {sorted(COV_ESTIMATORS)}."
        )


def _validate_metric(metric: str) -> None:
    if metric not in RIEMANN_METRICS:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose from {sorted(RIEMANN_METRICS)}."
        )


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------

def _make_classifier(name: str):
    """Return an sklearn-compatible classifier for Euclidean features."""
    if name == "svm":
        return SVC(kernel="linear", C=1.0, class_weight="balanced")
    elif name == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr")
    else:
        raise ValueError(f"Unknown classifier: {name}. Choose 'svm' or 'lda'.")


# ---------------------------------------------------------------------------
# sklearn-compatible Covariances wrapper
# ---------------------------------------------------------------------------

class RiemannCovariances(BaseEstimator, TransformerMixin):
    """Estimate SPD covariance matrices from EEG epochs.

    Thin wrapper around pyriemann.estimation.Covariances.

    Parameters
    ----------
    estimator : str
        Covariance estimator: 'scm', 'lwf', 'oas', or 'mcd'.
    """

    def __init__(self, estimator: str = "scm") -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, y=None) -> "RiemannCovariances":
        if not HAS_PYRIEMANN:
            raise ImportError("pyriemann is required for RiemannCovariances.")
        _validate_cov_estimator(self.estimator)
        self._cov = Covariances(estimator=self.estimator)
        self._cov.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_cov"):
            raise RuntimeError("Must call fit() before transform().")
        return self._cov.transform(X)


# ---------------------------------------------------------------------------
# Filter Bank Riemannian (FgMDM-style, sklearn-compatible)
# ---------------------------------------------------------------------------

class FilterBankRiemann(BaseEstimator, TransformerMixin):
    """Filter-bank Riemannian feature extractor.

    For each frequency band:
      1. Bandpass filter epochs
      2. Estimate covariance matrices
      3. Project to tangent space
      4. Collect tangent-space vectors

    All per-band tangent vectors are concatenated along axis=1.

    Parameters
    ----------
    freq_bands : list of (low, high) or None
        Defaults to FBCSP_BANDS from config.
    cov_estimator : str
    metric : str
    fs : int
    """

    def __init__(
        self,
        freq_bands: list[tuple[float, float]] | None = None,
        cov_estimator: str = "scm",
        metric: str = "riemann",
        fs: int = 250,
    ) -> None:
        self.freq_bands = freq_bands
        self.cov_estimator = cov_estimator
        self.metric = metric
        self.fs = fs
        self._tangent_spaces: list = []

    def fit(self, X: np.ndarray, y=None) -> "FilterBankRiemann":
        if not HAS_PYRIEMANN:
            raise ImportError("pyriemann is required for FilterBankRiemann.")
        _validate_cov_estimator(self.cov_estimator)
        _validate_metric(self.metric)

        bands = self._get_bands()

        self._tangent_spaces = []
        for low, high in bands:
            X_filt = _bandpass(X, low, high, fs=self.fs)
            cov = Covariances(estimator=self.cov_estimator).fit(X_filt)
            ts = TangentSpace(metric=self.metric).fit(cov.transform(X_filt))
            self._tangent_spaces.append(ts)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._tangent_spaces:
            raise RuntimeError(
                "FilterBankRiemann must be fitted before transform()."
            )
        bands = self._get_bands()
        all_features = []
        for (low, high), ts in zip(bands, self._tangent_spaces):
            X_filt = _bandpass(X, low, high, fs=self.fs)
            cov = Covariances(estimator=self.cov_estimator).fit_transform(X_filt)
            feats = ts.transform(cov)
            all_features.append(feats.astype(np.float32))
        return np.concatenate(all_features, axis=1)

    def _get_bands(self) -> list[tuple[float, float]]:
        if self.freq_bands is not None:
            return self.freq_bands
        from utils.config import FBCSP_BANDS
        return FBCSP_BANDS


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

def riemann_tangent_classify(
    X: np.ndarray,
    y: np.ndarray,
    cov_estimator: str = "scm",
    metric: str = "riemann",
    classifier: str = "lda",
    cv: int = 5,
) -> dict:
    """Covariance → Tangent Space → LDA/SVM with per-fold fitting.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)
    cov_estimator : str
        Covariance estimator: 'scm', 'lwf', 'oas', 'mcd'.
    metric : str
        Riemannian metric: 'riemann', 'euclid', 'logdet'.
    classifier : str
        'lda' or 'svm'.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict with keys: accuracy, accuracy_std, scores, method, classifier,
                    cov_estimator, metric
    """
    if not HAS_PYRIEMANN:
        raise ImportError("pyriemann is required for riemann_tangent_classify.")
    _validate_cov_estimator(cov_estimator)
    _validate_metric(metric)

    pipeline = make_pipeline(
        Covariances(estimator=cov_estimator),
        TangentSpace(metric=metric),
        _make_classifier(classifier),
    )

    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    return {
        "accuracy": scores.mean(),
        "accuracy_std": scores.std(),
        "scores": scores.tolist(),
        "method": "tangent",
        "classifier": classifier,
        "cov_estimator": cov_estimator,
        "metric": metric,
    }


def riemann_mdm_classify(
    X: np.ndarray,
    y: np.ndarray,
    cov_estimator: str = "scm",
    metric: str = "riemann",
    cv: int = 5,
) -> dict:
    """Covariance → MDM (Minimum Distance to Mean) with per-fold fitting.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)
    cov_estimator : str
    metric : str
    cv : int

    Returns
    -------
    dict with keys: accuracy, accuracy_std, scores, method, cov_estimator,
                    metric
    """
    if not HAS_PYRIEMANN:
        raise ImportError("pyriemann is required for riemann_mdm_classify.")
    _validate_cov_estimator(cov_estimator)
    _validate_metric(metric)

    pipeline = make_pipeline(
        Covariances(estimator=cov_estimator),
        MDM(metric=metric),
    )

    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    return {
        "accuracy": scores.mean(),
        "accuracy_std": scores.std(),
        "scores": scores.tolist(),
        "method": "mdm",
        "classifier": "mdm",
        "cov_estimator": cov_estimator,
        "metric": metric,
    }


def fgmdm_classify(
    X: np.ndarray,
    y: np.ndarray,
    freq_bands: list[tuple[float, float]] | None = None,
    cov_estimator: str = "scm",
    metric: str = "riemann",
    classifier: str = "lda",
    cv: int = 5,
    fs: int = 250,
) -> dict:
    """Filter-bank Riemannian → LDA/SVM (FgMDM-style).

    Uses FilterBankRiemann to extract per-band tangent-space features,
    then applies a Euclidean classifier.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)
    freq_bands : list of (low, high) or None
    cov_estimator : str
    metric : str
    classifier : str
    cv : int
    fs : int

    Returns
    -------
    dict with keys: accuracy, accuracy_std, scores, method, classifier,
                    cov_estimator, metric
    """
    if not HAS_PYRIEMANN:
        raise ImportError("pyriemann is required for fgmdm_classify.")
    _validate_cov_estimator(cov_estimator)
    _validate_metric(metric)

    if freq_bands is None:
        from utils.config import FBCSP_BANDS
        freq_bands = FBCSP_BANDS

    pipeline = make_pipeline(
        FilterBankRiemann(
            freq_bands=freq_bands,
            cov_estimator=cov_estimator,
            metric=metric,
            fs=fs,
        ),
        _make_classifier(classifier),
    )

    scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    return {
        "accuracy": scores.mean(),
        "accuracy_std": scores.std(),
        "scores": scores.tolist(),
        "method": "fgmdm",
        "classifier": classifier,
        "cov_estimator": cov_estimator,
        "metric": metric,
    }


def riemann_classify(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "tangent",
    cov_estimator: str = "scm",
    metric: str = "riemann",
    classifier: str = "lda",
    cv: int = 5,
    freq_bands: list[tuple[float, float]] | None = None,
    fs: int = 250,
) -> dict:
    """Unified entry point dispatching to method-specific classifiers.

    Parameters
    ----------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    y : np.ndarray, shape (n_trials,)
    method : str
        'tangent', 'mdm', or 'fgmdm'.
    cov_estimator : str
    metric : str
    classifier : str
    cv : int
    freq_bands : list of (low, high) or None
    fs : int

    Returns
    -------
    dict — see method-specific functions for exact keys.

    Raises
    ------
    ValueError
        If method is unknown.
    """
    if method == "tangent":
        return riemann_tangent_classify(
            X, y,
            cov_estimator=cov_estimator, metric=metric,
            classifier=classifier, cv=cv,
        )
    elif method == "mdm":
        return riemann_mdm_classify(
            X, y,
            cov_estimator=cov_estimator, metric=metric, cv=cv,
        )
    elif method == "fgmdm":
        return fgmdm_classify(
            X, y,
            freq_bands=freq_bands, cov_estimator=cov_estimator,
            metric=metric, classifier=classifier, cv=cv, fs=fs,
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from {sorted(VALID_METHODS)}."
        )
