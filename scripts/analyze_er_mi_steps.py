"""
Analyze ER-MI step-wise accuracy: does accuracy improve across reasoning steps?

Usage:
    python scripts/analyze_er_mi_steps.py --data_dir data/loso_binary --n_subjects 30 --epochs 80 --align --seed 42
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.er_mi import ERMI
from preprocessing.alignment import EuclideanAlignment


def load_subjects(data_dir, n_subjects):
    subjects = []
    for i in range(1, n_subjects + 1):
        subj_dir = Path(data_dir) / f"subj_{i:02d}"
        X = np.load(subj_dir / "X.npy").astype(np.float32)
        y = np.load(subj_dir / "y.npy").astype(np.int64)
        subjects.append({"id": i, "X": X, "y": y})
    return subjects


def train_ermi(train_subjects, device, epochs=80, lr=1e-3, seed=42):
    X_all = np.concatenate([s["X"] for s in train_subjects], axis=0)
    y_all = np.concatenate([s["y"] for s in train_subjects], axis=0)

    n_channels = X_all.shape[1]
    n_classes = len(np.unique(y_all))

    model = ERMI(n_channels=n_channels, n_classes=n_classes, steps=3).to(device)

    ds = TensorDataset(torch.from_numpy(X_all).float(), torch.from_numpy(y_all).long())
    loader = DataLoader(ds, batch_size=64, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))

    class_weights = compute_class_weight("balanced", classes=np.unique(y_all), y=y_all)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out[-1], yb)
            for step_logits in out[:-1]:
                loss = loss + 0.3 * criterion(step_logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def evaluate_per_step(model, subject, device):
    X_test = subject["X"]
    y_test = subject["y"]

    ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    loader = DataLoader(ds, batch_size=64)

    model.eval()
    all_step_preds = {}  # step -> [preds]
    all_labels = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            step_logits = model(Xb, return_all_steps=True)  # list of (B, n_classes)
            for step_i, logits in enumerate(step_logits):
                if step_i not in all_step_preds:
                    all_step_preds[step_i] = []
                all_step_preds[step_i].append(logits.argmax(-1).cpu())
            all_labels.append(yb)

    y_true = torch.cat(all_labels).numpy()
    step_accs = {}
    for step_i, preds in all_step_preds.items():
        y_pred = torch.cat(preds).numpy()
        step_accs[step_i] = (y_pred == y_true).mean()

    return step_accs, y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--n_subjects", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    subjects = load_subjects(args.data_dir, args.n_subjects)
    print(f"Loaded {len(subjects)} subjects | Device: {device}")

    all_step_accs = {0: [], 1: [], 2: []}
    per_subject = []

    for i, test_subj in enumerate(subjects):
        test_id = test_subj["id"]
        train_subjs = [s for s in subjects if s["id"] != test_id]

        # EA
        if args.align:
            ea = EuclideanAlignment()
            ea.fit([s["X"] for s in train_subjs])
            _train = [{"id": s["id"], "X": ea.transform(s["X"]), "y": s["y"]}
                      for s in train_subjs]
            train_subjs = _train
            test_subj = {"id": test_id, "X": ea.transform(test_subj["X"]), "y": test_subj["y"]}

        model = train_ermi(train_subjs, device, epochs=args.epochs, seed=args.seed)
        step_accs, _ = evaluate_per_step(model, test_subj, device)

        row = {"subject": f"S{test_id:02d}"}
        for step_i in sorted(step_accs):
            all_step_accs[step_i].append(step_accs[step_i])
            row[f"step{step_i+1}"] = f"{step_accs[step_i]:.4f}"
        row["trend"] = "↑" if step_accs[2] > step_accs[0] else ("↓" if step_accs[2] < step_accs[0] else "→")
        per_subject.append(row)

        print(f"Fold {i+1:2d}/30 S{test_id:02d}: "
              + " | ".join(f"S{t+1}={step_accs[t]:.4f}" for t in sorted(step_accs))
              + f"  {row['trend']}")

        del model

    # Summary
    print("\n" + "=" * 60)
    print("Step-wise Accuracy Summary")
    print("=" * 60)
    for step_i in sorted(all_step_accs):
        vals = np.array(all_step_accs[step_i])
        print(f"  Step {step_i+1}: mean={vals.mean():.4f}  std={vals.std():.4f}")

    # Count trends
    up = sum(1 for r in per_subject if r["trend"] == "↑")
    down = sum(1 for r in per_subject if r["trend"] == "↓")
    flat = sum(1 for r in per_subject if r["trend"] == "→")
    print(f"\nStep1→Step3 trend: ↑={up}  ↓={down}  →={flat} / {len(per_subject)} subjects")

    # Per-subject detail
    print("\n" + "=" * 60)
    print("Per-Subject Step Accuracy")
    print("=" * 60)
    print(f"{'Subject':>8} | {'Step1':>8} | {'Step2':>8} | {'Step3':>8} | Trend")
    print("-" * 50)
    for row in per_subject:
        print(f"{row['subject']:>8} | {row['step1']:>8} | {row['step2']:>8} | {row['step3']:>8} | {row['trend']}")


if __name__ == "__main__":
    main()
