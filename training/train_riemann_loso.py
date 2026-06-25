"""
Leave-One-Subject-Out evaluation for Riemannian Geometry baselines.

Unlike the deep-learning LOSO in train_loso.py, this script uses sklearn
pipelines (no PyTorch dependency).  It reuses the same per-subject data loader
and exports results in the identical CSV + JSON format for easy comparison.

Usage:
    python training/train_riemann_loso.py --data_dir data/loso_binary --method tangent
    python training/train_riemann_loso.py --data_dir data/loso_binary --method mdm --align
    python training/train_riemann_loso.py --data_dir data/loso_binary --method fgmdm

References
----------
- Barachant et al., "Multiclass Brain-Computer Interface Classification
  by Riemannian Geometry" (IEEE TBME, 2012)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.riemann import (  # noqa: E402
    FilterBankRiemann,
    _make_classifier,
    _validate_cov_estimator,
    _validate_metric,
    HAS_PYRIEMANN,
)

if HAS_PYRIEMANN:
    from pyriemann.estimation import Covariances  # noqa: E402
    from pyriemann.tangentspace import TangentSpace  # noqa: E402
    from pyriemann.classification import MDM  # noqa: E402

from preprocessing.alignment import EuclideanAlignment  # noqa: E402
from utils.metrics import (  # noqa: E402
    classification_report,
    per_class_recall,
    per_class_specificity,
    per_class_f1,
)
from datasets.label_mapping import class_names as get_class_names  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading — reuse train_loso's loader
# ---------------------------------------------------------------------------

def _load_per_subject_data(data_dir: str, n_subjects: int) -> list[dict]:
    """Load per-subject .npy files into a list of dicts."""
    subjects = []
    for i in range(1, n_subjects + 1):
        subj_dir = Path(data_dir) / f"subj_{i:02d}"
        X_path = subj_dir / "X.npy"
        y_path = subj_dir / "y.npy"
        if not X_path.exists():
            print(f"  Subject {i:02d}: missing, skipping")
            continue
        X = np.load(X_path).astype(np.float32)
        y = np.load(y_path).astype(np.int64)
        subjects.append({"id": i, "X": X, "y": y})
    print(f"Loaded {len(subjects)} subjects")
    return subjects


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _build_pipeline(
    method: str,
    cov_estimator: str,
    metric: str,
    classifier: str,
    freq_bands: list[tuple[float, float]] | None = None,
    fs: int = 250,
):
    """Build an sklearn Pipeline for the requested Riemannian method.

    Returns (pipeline, needs_fit_on_train).
    """
    _validate_cov_estimator(cov_estimator)

    if method == "tangent":
        _validate_metric(metric)
        pipeline = make_pipeline(
            Covariances(estimator=cov_estimator),
            TangentSpace(metric=metric),
            _make_classifier(classifier),
        )
    elif method == "mdm":
        _validate_metric(metric)
        pipeline = make_pipeline(
            Covariances(estimator=cov_estimator),
            MDM(metric=metric),
        )
    elif method == "fgmdm":
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
    else:
        raise ValueError(f"Unknown method: {method}")

    return pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LOSO evaluation for Riemannian Geometry baselines",
    )
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--n_subjects", type=int, default=30)
    parser.add_argument(
        "--method", default="tangent",
        choices=["tangent", "mdm", "fgmdm"],
        help="Riemannian classification method",
    )
    parser.add_argument(
        "--cov_estimator", default="scm",
        choices=["scm", "lwf", "oas", "mcd"],
        help="Covariance estimator (default: scm)",
    )
    parser.add_argument(
        "--metric", default="riemann",
        choices=["riemann", "euclid", "logchol", "logeuclid", "wasserstein"],
        help="Riemannian metric (default: riemann)",
    )
    parser.add_argument(
        "--classifier", default="lda",
        choices=["lda", "svm"],
        help="Euclidean classifier for tangent/fgmdm (default: lda)",
    )
    parser.add_argument(
        "--align", action="store_true",
        help="Apply Euclidean Alignment (EA) inside each LOSO fold "
             "(R_bar computed from training subjects only)",
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Directory for per-subject CSV and summary JSON",
    )
    parser.add_argument(
        "--dataset", default="physionet_mi",
        choices=["physionet_mi", "bci_iv_2a", "deepbci"],
        help="Dataset name for semantic class labels in CSV header",
    )
    parser.add_argument(
        "--bands", type=int, nargs="*", default=None,
        help="Custom filter bank: low1 high1 low2 high2 ... "
             "(e.g., --bands 4 8 8 12 12 16). Only for fgmdm.",
    )
    args = parser.parse_args()

    if not HAS_PYRIEMANN:
        print("ERROR: pyriemann is required. Install with: pip install pyriemann")
        sys.exit(1)

    # Parse custom filter bank
    freq_bands = None
    if args.bands:
        if len(args.bands) % 2 != 0:
            parser.error("--bands requires an even number of values (low high pairs)")
        freq_bands = [
            (args.bands[i], args.bands[i + 1]) for i in range(0, len(args.bands), 2)
        ]

    print(f"Method: {args.method}  Estimator: {args.cov_estimator}  "
          f"Metric: {args.metric}  Classifier: {args.classifier}  "
          f"Align: {args.align}")

    # Load per-subject data
    subjects = _load_per_subject_data(args.data_dir, args.n_subjects)
    if len(subjects) < 2:
        print("Need at least 2 subjects for LOSO")
        return

    # ── Label validation ──────────────────────────────────────────────
    semantic_names = get_class_names(args.dataset)

    per_subject_results = []

    for i, test_subj in enumerate(subjects):
        test_id = test_subj["id"]
        train_subjs = [s for s in subjects if s["id"] != test_id]

        print(f"\n{'='*50}")
        print(f"Fold {i+1}/{len(subjects)}: Test=S{test_id:02d}, "
              f"Train={len(train_subjs)} subjects")
        print(f"{'='*50}")

        # ── Euclidean Alignment (per-fold, no data leakage) ───────────
        if args.align:
            ea = EuclideanAlignment()
            ea.fit([s["X"] for s in train_subjs])
            _train = []
            for s in train_subjs:
                _train.append({
                    "id": s["id"],
                    "X": ea.transform(s["X"]),
                    "y": s["y"],
                })
            train_subjs = _train
            test_subj = {
                "id": test_subj["id"],
                "X": ea.transform(test_subj["X"]),
                "y": test_subj["y"],
            }

        # Concatenate training subjects
        X_train = np.concatenate([s["X"] for s in train_subjs], axis=0)
        y_train = np.concatenate([s["y"] for s in train_subjs], axis=0)
        X_test = test_subj["X"]
        y_test = test_subj["y"]

        # Build pipeline
        pipeline = _build_pipeline(
            method=args.method,
            cov_estimator=args.cov_estimator,
            metric=args.metric,
            classifier=args.classifier,
            freq_bands=freq_bands,
        )

        # Fit on train, evaluate on test
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = classification_report(y_test, y_pred)
        metrics["per_class_recall"] = per_class_recall(y_test, y_pred)
        metrics["per_class_specificity"] = per_class_specificity(y_test, y_pred)
        metrics["per_class_f1"] = per_class_f1(y_test, y_pred)
        metrics["n_trials"] = len(y_test)

        print(f"  Test:    acc={metrics['accuracy']:.4f}  "
              f"kappa={metrics['kappa']:.4f}")

        row = {
            "subject": f"S{test_id:02d}",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["f1_macro"],
            "kappa": metrics["kappa"],
            "n_trials": metrics["n_trials"],
        }
        for cls_id, rec in metrics["per_class_recall"].items():
            cls_name = (
                semantic_names[cls_id]
                if cls_id < len(semantic_names)
                else f"cls_{cls_id}"
            )
            row[f"recall_{cls_name}"] = rec
        for cls_id, spec in metrics["per_class_specificity"].items():
            cls_name = (
                semantic_names[cls_id]
                if cls_id < len(semantic_names)
                else f"cls_{cls_id}"
            )
            row[f"specificity_{cls_name}"] = spec
        per_subject_results.append(row)

    # ── Export ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_tag = f"_{args.dataset}" if args.dataset != "physionet_mi" else ""
    ea_tag = "_ea" if args.align else ""

    csv_path = output_dir / f"loso_riemann_{args.method}{ds_tag}{ea_tag}.csv"
    if per_subject_results:
        fieldnames = list(per_subject_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_subject_results)

    accs = np.array([r["accuracy"] for r in per_subject_results])
    kappas = np.array([r["kappa"] for r in per_subject_results])
    summary = {
        "method": f"riemann_{args.method}",
        "dataset": args.dataset,
        "n_subjects": len(subjects),
        "cov_estimator": args.cov_estimator,
        "metric": args.metric,
        "classifier": args.classifier,
        "align": args.align,
        "accuracy_mean": round(float(accs.mean()), 4),
        "accuracy_std": round(float(accs.std()), 4),
        "kappa_mean": round(float(kappas.mean()), 4),
        "kappa_std": round(float(kappas.std()), 4),
        "per_subject": per_subject_results,
    }

    print("\n" + "=" * 60)
    print("LOSO Riemannian Summary")
    print("=" * 60)
    print(f"Method:     {args.method}")
    print(f"Metric:     {args.metric}    Estimator: {args.cov_estimator}")
    print(f"Classifier: {args.classifier}    Align: {args.align}")
    print(f"Accuracy:   mean={accs.mean():.4f}  std={accs.std():.4f}")
    print(f"Kappa:      mean={kappas.mean():.4f}  std={kappas.std():.4f}")
    print(f"Per-subject: {[f'{a:.3f}' for a in accs]}")
    print(f"Best/Worst: {accs.max():.4f} / {accs.min():.4f}")

    json_path = output_dir / f"loso_riemann_{args.method}{ds_tag}{ea_tag}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCSV saved to {csv_path}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
