"""
CSP + SVM baseline training.

Usage:
    python training/train_baseline.py --data_dir data/processed/subj01
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.csp import csp_svm_baseline
from utils.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--n_components", type=int, default=6)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    print(f"CSP+SVM baseline (n_components={args.n_components}, cv={args.cv})")
    print(f"Data: X={X_train.shape}, y={y_train.shape}")

    results = csp_svm_baseline(
        X_train, y_train,
        n_components=args.n_components,
        cv=args.cv,
    )

    logger = ExperimentLogger(run_name="csp_baseline")
    logger.log(**results)
    logger.close()

    print(f"Accuracy: {results['accuracy']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"Per-fold: {[round(s, 4) for s in results['scores']]}")


if __name__ == "__main__":
    main()
