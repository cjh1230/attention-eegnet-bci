"""Tests for utils/metrics.py."""
import numpy as np

from utils.metrics import classification_report, per_class_accuracy


class TestClassificationReport:
    def test_perfect_prediction(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        r = classification_report(y_true, y_pred)
        assert r["accuracy"] == 1.0
        assert r["kappa"] == 1.0
        assert r["f1_macro"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        r = classification_report(y_true, y_pred)
        assert r["accuracy"] == 0.0

    def test_returns_required_keys(self):
        r = classification_report(
            np.array([0, 1, 2]), np.array([0, 1, 2])
        )
        for key in ["accuracy", "kappa", "f1_macro", "precision_macro",
                     "recall_macro", "confusion_matrix"]:
            assert key in r

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        r = classification_report(y_true, y_pred)
        cm = np.array(r["confusion_matrix"])
        assert cm.shape == (3, 3)

    def test_zero_division_handled(self):
        """When a class never appears in preds, macro metrics shouldn't crash."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # only class 0 predicted
        r = classification_report(y_true, y_pred)
        assert r["accuracy"] >= 0.0


class TestPerClassAccuracy:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        pca = per_class_accuracy(y_true, y_pred)
        assert np.allclose(pca, [1.0, 1.0])

    def test_shape_matches_n_classes(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 1])
        pca = per_class_accuracy(y_true, y_pred)
        assert len(pca) == 3
