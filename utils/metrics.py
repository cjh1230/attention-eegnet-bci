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


def classification_report(y_true, y_pred):
    """Return dict of common metrics for BCI classification."""

    cm = confusion_matrix(y_true, y_pred)
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
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div by zero
    return np.diag(cm).astype(float) / row_sums
