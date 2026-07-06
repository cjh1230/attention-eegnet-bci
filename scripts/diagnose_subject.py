"""
Single-subject diagnostic: train BRT-Det on one subject's own data.

Answers the question: is a subject inherently hard (weak MI signal, noisy data),
or is the low LOSO score purely a cross-subject generalization failure?

Usage:
    python scripts/diagnose_subject.py --data_dir data/loso_binary --subject 9
    python scripts/diagnose_subject.py --data_dir data/loso_binary --subject 9 --model eegnet
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet_attn import create_model
from utils.metrics import classification_report
from datasets.label_mapping import class_names as get_class_names


def main():
    parser = argparse.ArgumentParser(
        description="Single-subject diagnostic training")
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--subject", type=int, required=True,
                        help="Subject ID to diagnose (1-30)")
    parser.add_argument("--model", default="brt_det")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset", default="physionet_mi")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help="JSON kwargs for create_model()")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="K-fold CV within subject (default 5)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}

    # ── Load subject data ──
    subj_dir = Path(args.data_dir) / f"subj_{args.subject:02d}"
    X_path = subj_dir / "X.npy"
    y_path = subj_dir / "y.npy"
    if not X_path.exists():
        print(f"ERROR: {X_path} not found")
        sys.exit(1)

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    n_classes = len(np.unique(y))
    semantic_names = get_class_names(args.dataset)

    print("=" * 60)
    print(f"S{args.subject:02d} Single-Subject Diagnosis")
    print("=" * 60)
    print(f"  Trials: {len(y)}  Shape: {X.shape}  Classes: {n_classes}")
    print(f"  Label distribution:")

    for cls_id in range(n_classes):
        cls_name = semantic_names[cls_id] if cls_id < len(semantic_names) else f"cls_{cls_id}"
        count = (y == cls_id).sum()
        print(f"    {cls_name}: {count} trials ({count/len(y)*100:.1f}%)")

    # ── Per-channel amplitude check ──
    ch_names = ["FC3", "C3", "Cz", "C4", "FC4", "CP3", "CPz", "CP4"]
    print(f"\n  Per-channel RMS amplitude (μV):")
    for ch in range(min(8, X.shape[1])):
        rms = np.sqrt(np.mean(X[:, ch, :] ** 2))
        flag = " [LOW]" if rms < 1e-6 else ""
        print(f"    {ch_names[ch]:>4s}: {rms:.6f}{flag}")

    # ── Per-class channel difference (C3 - C4) ──
    print(f"\n  Per-class C3-C4 mean amplitude:")
    for cls_id in range(n_classes):
        cls_name = semantic_names[cls_id] if cls_id < len(semantic_names) else f"cls_{cls_id}"
        cls_mask = y == cls_id
        c3 = X[cls_mask, 1, :].mean()
        c4 = X[cls_mask, 3, :].mean()
        diff = c3 - c4
        print(f"    {cls_name}: C3={c3:.4f}, C4={c4:.4f}, C3-C4={diff:+.4f}")

    # ── K-fold within-subject training ──
    print(f"\n── {args.n_folds}-Fold Within-Subject CV ──")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_accs = []
    fold_kappas = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_f, X_test_f = X[train_idx], X[test_idx]
        y_train_f, y_test_f = y[train_idx], y[test_idx]

        model = create_model(args.model, n_channels=X.shape[1],
                             n_classes=n_classes, **model_kwargs).to(device)

        # Filter bank preprocessing
        if getattr(model, "input_requires_filter_bank", False):
            from models.fbcnet import apply_filter_bank
            X_train_f = apply_filter_bank(X_train_f)
            X_test_f = apply_filter_bank(X_test_f)

        train_ds = TensorDataset(
            torch.from_numpy(X_train_f).float(),
            torch.from_numpy(y_train_f).long())
        train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_idx)),
                                  shuffle=True)

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train_f), y=y_train_f)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.label_smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        for epoch in range(1, args.epochs + 1):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb)
                if isinstance(out, list):
                    loss = criterion(out[-1], yb)
                else:
                    loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Evaluate
        model.eval()
        test_ds = TensorDataset(
            torch.from_numpy(X_test_f).float(),
            torch.from_numpy(y_test_f).long())
        test_loader = DataLoader(test_ds, batch_size=64)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                out = model(Xb)
                all_preds.append(
                    (out[-1] if isinstance(out, list) else out).argmax(-1).cpu())
                all_labels.append(yb)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        metrics = classification_report(y_true, y_pred)
        fold_accs.append(metrics["accuracy"])
        fold_kappas.append(metrics["kappa"])
        print(f"  Fold {fold+1}: acc={metrics['accuracy']:.4f}  "
              f"κ={metrics['kappa']:.4f}")
        del model

    accs = np.array(fold_accs)
    kappas = np.array(fold_kappas)
    print(f"\n── Within-Subject Summary ──")
    print(f"  Accuracy: {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  Kappa:    {kappas.mean():.4f} ± {kappas.std():.4f}")

    # ── Diagnosis ──
    print(f"\n── Diagnosis ──")
    if accs.mean() >= 0.75:
        print(f"  [STRONG] S{args.subject:02d} within-subject >= 75%.")
        print(f"     Low LOSO = cross-subject generalization failure.")
        print(f"     Action: focus on domain adaptation / subject alignment.")
    elif accs.mean() >= 0.65:
        print(f"  [MODERATE] S{args.subject:02d} within-subject 65-75%.")
        print(f"     LOSO gap suggests both signal quality AND generalization issues.")
    elif accs.mean() >= 0.55:
        print(f"  [WEAK] S{args.subject:02d} within-subject 55-65%.")
        print(f"     MI signal is marginal — even within-subject training struggles.")
    else:
        print(f"  [VERY WEAK] S{args.subject:02d} within-subject < 55%.")
        print(f"     Possible data issues: check labels, trial alignment, channel order.")
        print(f"     May be BCI-inefficient — not all subjects can do MI.")


if __name__ == "__main__":
    main()
