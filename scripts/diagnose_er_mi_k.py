"""
Quick ER-MI K sweep: train on 29 subjects, test on S07.
Tests K=1,2,3,4,5 to verify if multi-step reasoning actually helps.

Usage:
    python scripts/diagnose_er_mi_k.py --data_dir data/loso_binary
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet_attn import create_model
from training.train_loso import load_per_subject_data, train_on_subjects, evaluate_on_subject


def run_single_fold(subjects, test_id, steps, device, epochs=80):
    """Train on all except test_id, evaluate on test_id."""
    train_subjs = [s for s in subjects if s["id"] != test_id]
    test_subj = [s for s in subjects if s["id"] == test_id][0]
    n_channels = test_subj["X"].shape[1]
    n_classes = len(np.unique(test_subj["y"]))

    model = create_model("er_mi", n_channels=n_channels, n_classes=n_classes,
                         steps=steps).to(device)
    model = train_on_subjects(train_subjs, "er_mi", device, epochs=epochs,
                              batch_size=64, lr=1e-3, seed=42,
                              model_kwargs={"steps": steps},
                              intermediate_loss_weight=0.3,
                              label_smoothing=0.0)
    metrics = evaluate_on_subject(model, test_subj, device)
    del model
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--test_subject", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = load_per_subject_data(args.data_dir, 30)
    test_id = args.test_subject

    print(f"ER-MI K Sweep: test=S{test_id:02d}, train=29 subjects")
    print(f"{'K':>4s}  {'Acc':>8s}  {'Kappa':>8s}")
    print("-" * 24)

    results = {}
    for K in [1, 2, 3, 4, 5]:
        metrics = run_single_fold(subjects, test_id, K, device, args.epochs)
        results[K] = metrics
        marker = " <-- default" if K == 3 else ""
        print(f"{K:>4d}  {metrics['accuracy']:8.4f}  {metrics['kappa']:8.4f}{marker}")

    best_K = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest K: {best_K} (acc={results[best_K]['accuracy']:.4f})")

    # Check: is K=1 as good as K=3?
    k1 = results[1]["accuracy"]
    k3 = results[3]["accuracy"]
    if k1 >= k3 - 0.01:
        print(f"WARNING: K=1 ({k1:.4f}) ≈ K=3 ({k3:.4f}) — multi-step reasoning may not help!")
    else:
        print(f"Multi-step reasoning confirmed: K=3 beats K=1 by {k3-k1:+.4f}")


if __name__ == "__main__":
    main()
