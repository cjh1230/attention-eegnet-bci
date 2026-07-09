"""
Evaluation metrics for multi-class MI classification.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _all_labels(y_true, y_pred) -> np.ndarray:
    """Fixed label set 0..K-1 so confusion-matrix rows/cols map to class ids.

    Without an explicit ``labels`` argument, ``confusion_matrix`` spans only the
    classes actually present in the data. When a fold is missing a class (in the
    ground truth or the predictions), the matrix shrinks and the row/column
    indices silently shift, misaligning per-class metrics with their semantic
    class names. Pinning ``labels`` to a contiguous 0..K-1 range keeps
    ``enumerate`` indices equal to the true class id.

    Only the confusion-matrix-based per-class helpers use this; the macro
    accuracy/kappa/F1 scores are intentionally left as-is so their reported
    values are unchanged.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    hi = int(max(y_true.max(initial=0), y_pred.max(initial=0)))
    return np.arange(hi + 1)


def classification_report(y_true, y_pred):
    """Return dict of common metrics for BCI classification."""

    cm = confusion_matrix(y_true, y_pred, labels=_all_labels(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "kappa": round(kappa, 4),
        "f1_macro": round(f1, 4),
        "precision_macro": round(precision, 4),
        "recall_macro": round(recall, 4),
        "confusion_matrix": cm.tolist(),
    }


def per_class_accuracy(y_true, y_pred):
    """Return per-class accuracy given integer labels."""
    cm = confusion_matrix(y_true, y_pred, labels=_all_labels(y_true, y_pred))
    row_sums = cm.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return np.diag(cm).astype(float) / row_sums


def per_class_recall(y_true, y_pred) -> dict[int, float]:
    """Return per-class recall (sensitivity) as dict[class_label, value].

    Recall = TP / (TP + FN)
    """
    cm = confusion_matrix(y_true, y_pred, labels=_all_labels(y_true, y_pred))
    row_sums = cm.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    recalls = np.diag(cm).astype(float) / row_sums
    return {i: round(float(r), 4) for i, r in enumerate(recalls)}


def per_class_specificity(y_true, y_pred) -> dict[int, float]:
    """Return per-class specificity as dict[class_label, value].

    Specificity = TN / (TN + FP)  — true negative rate.
    """
    cm = confusion_matrix(y_true, y_pred, labels=_all_labels(y_true, y_pred))
    n_classes = cm.shape[0]
    total = cm.sum()
    specs = {}
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specs[c] = round(float(tn / denom), 4) if denom > 0 else 0.0
    return specs


def per_class_precision(y_true, y_pred) -> dict[int, float]:
    """Return per-class precision as dict[class_label, value].

    Precision = TP / (TP + FP)
    """
    cm = confusion_matrix(y_true, y_pred, labels=_all_labels(y_true, y_pred))
    col_sums = cm.sum(axis=0)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    precisions = np.diag(cm).astype(float) / col_sums
    return {i: round(float(p), 4) for i, p in enumerate(precisions)}


def per_class_f1(y_true, y_pred) -> dict[int, float]:
    """Return per-class F1 score as dict[class_label, value]."""
    prec = per_class_precision(y_true, y_pred)
    rec = per_class_recall(y_true, y_pred)
    f1s = {}
    for c in prec:
        p, r = prec[c], rec[c]
        denom = p + r
        f1s[c] = round(float(2 * p * r / denom), 4) if denom > 0 else 0.0
    return f1s
