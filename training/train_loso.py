"""
Leave-One-Subject-Out (LOSO) cross-validation for MI classification.

LOSO is the gold-standard evaluation for BCI — it measures how well a model
trained on N-1 subjects generalizes to a completely unseen subject.

Supports:
    - Pure LOSO: train on 29 subjects, test on 1. Repeat 30x.
    - LOSO + Few-shot FT: fine-tune on k trials of the target subject.

Usage:
    # Preprocess first:
    python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary

    # Then run LOSO:
    python training/train_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60
    python training/train_loso.py --data_dir data/loso_binary --n_subjects 30 --finetune 10 --model eegnet_spatiotemporal

Expected result: Within-subject MI binary typically 75-90% (vs 63% cross-subject).
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

from models.eegnet_attn import create_model
from utils.metrics import classification_report, per_class_accuracy


def load_per_subject_data(data_dir: str, n_subjects: int) -> list[dict]:
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


def train_on_subjects(
    train_subjects: list[dict],
    model_type: str,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    verbose: bool = False,
):
    """Train a model on a list of subjects' data. Returns trained model."""
    # Concatenate all training subjects
    X_all = np.concatenate([s["X"] for s in train_subjects], axis=0)
    y_all = np.concatenate([s["y"] for s in train_subjects], axis=0)

    n_channels = X_all.shape[1]
    n_classes = len(np.unique(y_all))

    train_ds = TensorDataset(
        torch.from_numpy(X_all).float(),
        torch.from_numpy(y_all).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = create_model(model_type, n_channels=n_channels, n_classes=n_classes).to(device)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_all), y=y_all)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def evaluate_on_subject(model, subject: dict, device: str) -> dict:
    """Evaluate model on a single subject. Returns metrics dict."""
    X_test = subject["X"]
    y_test = subject["y"]
    n_classes = len(np.unique(y_test))

    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )
    test_loader = DataLoader(test_ds, batch_size=64)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            all_preds.append(model(Xb).argmax(-1).cpu())
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return classification_report(y_true, y_pred)


def finetune_on_subject(
    model: nn.Module,
    subject: dict,
    n_finetune_trials: int,
    device: str,
    lr: float = 1e-4,
    epochs: int = 20,
) -> nn.Module:
    """
    Few-shot fine-tune on n_finetune_trials of the target subject.
    Returns the fine-tuned model.
    """
    X = subject["X"]
    y = subject["y"]

    # Split: first n_finetune_trials per class for FT, rest for test
    n_classes = len(np.unique(y))
    ft_indices = []
    test_indices = []
    for c in range(n_classes):
        c_idx = np.where(y == c)[0]
        n_ft = min(n_finetune_trials, len(c_idx) // 2)
        ft_indices.extend(c_idx[:n_ft].tolist())
        test_indices.extend(c_idx[n_ft:].tolist())

    X_ft = X[ft_indices]
    y_ft = y[ft_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Fine-tune
    ft_ds = TensorDataset(
        torch.from_numpy(X_ft).float(),
        torch.from_numpy(y_ft).long(),
    )
    ft_loader = DataLoader(ft_ds, batch_size=min(16, len(ft_indices)), shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for Xb, yb in ft_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluate on remaining test trials
    model.eval()
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long(),
    )
    test_loader = DataLoader(test_ds, batch_size=64)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            all_preds.append(model(Xb).argmax(-1).cpu())
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    return model, classification_report(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="LOSO cross-validation")
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--n_subjects", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", default="eegnet",
                        choices=["eegnet", "eegnet_spatiotemporal"])
    parser.add_argument("--finetune", type=int, default=0,
                        help="Few-shot FT trials per class (0 = pure LOSO)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip per-fold training (use pre-trained checkpoint)")
    parser.add_argument("--checkpoint", default=None,
                        help="Base checkpoint for --skip_train mode")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Model: {args.model}  FT: {args.finetune} trials/class")

    # Load per-subject data
    subjects = load_per_subject_data(args.data_dir, args.n_subjects)
    if len(subjects) < 2:
        print("Need at least 2 subjects for LOSO")
        return

    all_accs, all_kappas = [], []

    for i, test_subj in enumerate(subjects):
        test_id = test_subj["id"]
        train_subjs = [s for s in subjects if s["id"] != test_id]

        print(f"\n{'='*50}")
        print(f"Fold {i+1}/{len(subjects)}: Test=S{test_id:02d}, Train={len(train_subjs)} subjects")
        print(f"{'='*50}")

        # Train on N-1 subjects
        model = train_on_subjects(
            train_subjs, args.model, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )

        if args.finetune > 0:
            # Few-shot fine-tune on target subject
            model, metrics = finetune_on_subject(
                model, test_subj, n_finetune_trials=args.finetune,
                device=device,
            )
            print(f"  FT+Test: acc={metrics['accuracy']:.4f}  kappa={metrics['kappa']:.4f}")
        else:
            # Pure LOSO: evaluate directly
            metrics = evaluate_on_subject(model, test_subj, device)
            print(f"  Test:    acc={metrics['accuracy']:.4f}  kappa={metrics['kappa']:.4f}")

        all_accs.append(metrics["accuracy"])
        all_kappas.append(metrics["kappa"])

        del model  # free GPU memory

    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("LOSO Summary")
    print("=" * 60)
    accs = np.array(all_accs)
    kappas = np.array(all_kappas)
    print(f"Accuracy:  mean={accs.mean():.4f}  std={accs.std():.4f}")
    print(f"Kappa:     mean={kappas.mean():.4f}  std={kappas.std():.4f}")
    print(f"Per-subject: {[f'{a:.3f}' for a in all_accs]}")
    print(f"Best/Worst: {accs.max():.4f} / {accs.min():.4f}")


if __name__ == "__main__":
    main()
