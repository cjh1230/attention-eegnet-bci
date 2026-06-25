"""
CSP / FBCSP baseline training with LDA or SVM.

Usage:
    python training/train_baseline.py --data_dir data/processed
    python training/train_baseline.py --method fbcsp --classifier lda
    python training/train_baseline.py --method csp --classifier svm
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.csp import csp_svm_baseline, csp_lda_baseline, fbcsp_classify
from utils.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(
        description="CSP / FBCSP baseline training"
    )
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--method", default="csp",
                        choices=["csp", "fbcsp"],
                        help="CSP method: single-band or filter bank")
    parser.add_argument("--classifier", default="svm",
                        choices=["svm", "lda"],
                        help="Classifier: SVM (linear) or LDA")
    parser.add_argument("--n_components", type=int, default=6,
                        help="CSP components (per band for FBCSP)")
    parser.add_argument("--cv", type=int, default=5,
                        help="Cross-validation folds")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    # Load train data
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    n_classes = len(np.unique(y_train))
    print(f"{args.method.upper()} + {args.classifier.upper()} "
          f"(n_components={args.n_components}, cv={args.cv})")
    print(f"Data: X={X_train.shape}, y={y_train.shape}, {n_classes} classes")

    # ── Run baseline ────────────────────────────────────────────────
    if args.method == "csp":
        if args.classifier == "svm":
            results = csp_svm_baseline(
                X_train, y_train,
                n_components=args.n_components, cv=args.cv,
            )
        else:
            results = csp_lda_baseline(
                X_train, y_train,
                n_components=args.n_components, cv=args.cv,
            )
    else:  # fbcsp
        results = fbcsp_classify(
            X_train, y_train,
            n_components=args.n_components,
            classifier=args.classifier, cv=args.cv,
        )

    # Log
    run_name = f"{args.method}_{args.classifier}"
    logger = ExperimentLogger(run_name=run_name)
    logger.log(**results)
    logger.close()

    print(f"Accuracy:  {results['accuracy']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"Per-fold:  {[round(s, 4) for s in results['scores']]}")

    # ── Optional: evaluate on held-out validation set ───────────────
    val_X_path = data_dir / "X_val.npy"
    val_y_path = data_dir / "y_val.npy"
    if val_X_path.exists() and val_y_path.exists():
        X_val = np.load(val_X_path)
        y_val = np.load(val_y_path)
        print(f"\nVal data: X={X_val.shape}, y={y_val.shape}")

        if args.method == "csp":
            _eval_csp_val(X_train, y_train, X_val, y_val,
                          n_components=args.n_components,
                          classifier=args.classifier)
        else:
            _eval_fbcsp_val(X_train, y_train, X_val, y_val,
                            n_components=args.n_components,
                            classifier=args.classifier)


def _eval_csp_val(X_train, y_train, X_val, y_val,
                  n_components, classifier):
    """Fit CSP on training set and evaluate on validation set."""
    from sklearn.pipeline import make_pipeline
    from mne.decoding import CSP
    from features.csp import _make_classifier

    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    clf = _make_classifier(classifier)
    pipe = make_pipeline(csp, clf)

    pipe.fit(X_train, y_train)
    val_acc = float(pipe.score(X_val, y_val))
    print(f"Val accuracy: {val_acc:.4f}")


def _eval_fbcsp_val(X_train, y_train, X_val, y_val,
                    n_components, classifier):
    """Fit FBCSP on training set and evaluate on validation set."""
    from features.csp import FilterBankCSP, _make_classifier

    fb = FilterBankCSP(n_components=n_components)
    X_train_feats = fb.fit(X_train, y_train).transform(X_train)
    X_val_feats = fb.transform(X_val)

    clf = _make_classifier(classifier)
    clf.fit(X_train_feats, y_train)
    val_acc = float(clf.score(X_val_feats, y_val))
    print(f"Val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
