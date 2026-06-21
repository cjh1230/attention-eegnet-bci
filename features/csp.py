"""
Common Spatial Patterns (CSP) feature extraction — MI baseline.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP


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
    dict with keys: accuracy, kappa, scores
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
