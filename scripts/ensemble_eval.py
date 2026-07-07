"""
Single-fold logits ensemble: train both models on N-1 subjects,
ensemble their logits on the held-out subject, sweep ensemble weight.

Usage:
    python scripts/ensemble_eval.py --data_dir data/loso_binary \
        --test_subject 13 --epochs 80 --alpha_sweep 0.0,0.3,0.5,0.7,1.0
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet_attn import create_model
from models.fbcnet import apply_filter_bank
from training.train_loso import load_per_subject_data, train_on_subjects
from utils.metrics import classification_report


def evaluate_ensemble(model_a, model_b, alpha, test_subj, device):
    """Ensemble logits with weight alpha on model_a, (1-alpha) on model_b."""
    X_test = test_subj["X"]
    y_test = test_subj["y"]

    # Filter bank for BRT-Det
    X_brt = apply_filter_bank(X_test)
    X_brt_t = torch.from_numpy(X_brt).float().to(device)

    # Raw EEG for Conformer/EEGNet
    X_raw_t = torch.from_numpy(X_test).float().to(device)

    ds = TensorDataset(X_raw_t, X_brt_t, torch.from_numpy(y_test).long())
    loader = DataLoader(ds, batch_size=64)

    model_a.eval()
    model_b.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for Xa, Xb, yb in loader:
            Xa, Xb = Xa.to(device), Xb.to(device)

            # Get logits from each model
            out_a = model_a(Xa)
            logits_a = out_a[-1] if isinstance(out_a, list) else out_a

            out_b = model_b(Xb)
            logits_b = out_b[-1] if isinstance(out_b, list) else out_b

            # Ensemble
            logits = alpha * logits_a + (1 - alpha) * logits_b
            all_preds.append(logits.argmax(-1).cpu())
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return classification_report(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Single-fold ensemble evaluation")
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--test_subject", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default=None)
    parser.add_argument("--alpha_sweep", type=str, default="0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0")
    parser.add_argument("--model_a", default="eeg_conformer", help="Model A (raw EEG)")
    parser.add_argument("--model_b", default="brt_det", help="Model B (filter bank)")
    parser.add_argument("--model_b_kwargs", type=str,
                        default='{"use_region_pool":false, "n_time_cells":24, '
                                '"dilations":[1,2,4], "agg_mode":"objectness", '
                                '"use_band_gate":true}')
    parser.add_argument("--label_a", default="Conformer")
    parser.add_argument("--label_b", default="BRT-Det")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_b_kwargs = json.loads(args.model_b_kwargs)
    alphas = [float(x.strip()) for x in args.alpha_sweep.split(",")]

    # Load data
    subjects = load_per_subject_data(args.data_dir, 30)
    subj_map = {s["id"]: s for s in subjects}
    test_subj = subj_map.get(args.test_subject)
    if test_subj is None:
        print(f"Subject {args.test_subject} not found")
        sys.exit(1)
    train_subjs = [s for s in subjects if s["id"] != args.test_subject]

    print(f"Test subject: S{args.test_subject:02d} "
          f"(trials={test_subj['X'].shape[0]})")
    print(f"Train subjects: {len(train_subjs)}")
    print(f"Ensemble weights: {alphas}")
    print(f"Model A: {args.label_a} ({args.model_a})")
    print(f"Model B: {args.label_b} ({args.model_b})")

    # Train model A (Conformer — raw EEG, no filter bank)
    print(f"\n-- Training {args.label_a} --")
    n_channels = test_subj["X"].shape[1]
    n_classes = len(np.unique(test_subj["y"]))
    model_a = create_model(args.model_a, n_channels=n_channels,
                           n_classes=n_classes).to(device)
    model_a = train_on_subjects(train_subjs, args.model_a, device,
                                epochs=args.epochs, batch_size=64, lr=1e-3,
                                label_smoothing=0.1)

    # Train model B (BRT-Det — filter bank input)
    print(f"\n-- Training {args.label_b} --")
    model_b = train_on_subjects(train_subjs, args.model_b, device,
                                epochs=args.epochs, batch_size=64, lr=1e-3,
                                model_kwargs=model_b_kwargs,
                                label_smoothing=0.1)

    # Evaluate each model alone
    print(f"\n-- Ensemble Sweep --")
    # Model A alone (alpha=1.0) requires raw EEG eval
    X_raw_t = torch.from_numpy(test_subj["X"]).float().to(device)
    y_t = torch.from_numpy(test_subj["y"]).long()
    ds_raw = TensorDataset(X_raw_t, y_t)
    loader_raw = DataLoader(ds_raw, batch_size=64)
    model_a.eval()
    preds_a, labels_a = [], []
    with torch.no_grad():
        for Xb, yb in loader_raw:
            Xb = Xb.to(device)
            out = model_a(Xb)
            logits = out[-1] if isinstance(out, list) else out
            preds_a.append(logits.argmax(-1).cpu())
            labels_a.append(yb)
    acc_a = (torch.cat(preds_a) == torch.cat(labels_a)).float().mean().item()

    # Model B alone
    X_brt = apply_filter_bank(test_subj["X"])
    X_brt_t = torch.from_numpy(X_brt).float().to(device)
    ds_brt = TensorDataset(X_brt_t, y_t)
    loader_brt = DataLoader(ds_brt, batch_size=64)
    model_b.eval()
    preds_b, labels_b = [], []
    with torch.no_grad():
        for Xb, yb in loader_brt:
            Xb = Xb.to(device)
            out = model_b(Xb)
            logits = out[-1] if isinstance(out, list) else out
            preds_b.append(logits.argmax(-1).cpu())
            labels_b.append(yb)
    acc_b = (torch.cat(preds_b) == torch.cat(labels_b)).float().mean().item()

    print(f"\n  {args.label_a} alone:  {acc_a:.4f}")
    print(f"  {args.label_b} alone: {acc_b:.4f}")
    print(f"  Best single:     {max(acc_a, acc_b):.4f}")

    # Sweep alpha
    best_acc = 0.0
    best_alpha = 0.5
    for alpha in alphas:
        if abs(alpha) < 0.001:
            acc = acc_b
        elif abs(alpha - 1.0) < 0.001:
            acc = acc_a
        else:
            metrics = evaluate_ensemble(model_a, model_b, alpha, test_subj, device)
            acc = metrics["accuracy"]
        marker = " <--" if acc > max(acc_a, acc_b) else ""
        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha
        print(f"  alpha={alpha:.1f} ({args.label_a}={alpha:.1f}, "
              f"{args.label_b}={1-alpha:.1f}): acc={acc:.4f}{marker}")

    gain = best_acc - max(acc_a, acc_b)
    print(f"\n  Best ensemble: alpha={best_alpha:.1f}, acc={best_acc:.4f} "
          f"(gain={gain:+.4f} over best single)")

    del model_a, model_b


if __name__ == "__main__":
    main()
