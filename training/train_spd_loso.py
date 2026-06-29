"""
Leave-One-Subject-Out (LOSO) evaluation for SPDNet on SPD covariance matrices.

SPDNet operates on per-trial SPD covariance matrices (C×C), not raw EEG
trials (C×T).  This script handles the conversion inside each LOSO fold,
applying optional Euclidean Alignment before covariance estimation.

Usage:
    # Preprocess first:
    python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary

    # SPDNet LOSO:
    python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60
    python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align
    python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align --cov_estimator lwf

    # Quick test (2 subjects, 10 epochs):
    python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 2 --epochs 10 --align

Reference:
    Huang & Van Gool, "A Riemannian Network for SPD Matrix Learning", AAAI 2017.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from features.spd_covariance import compute_covariance
from models.spd_models import create_spdnet
from preprocessing.alignment import EuclideanAlignment
from utils.metrics import (
    classification_report,
    per_class_recall,
    per_class_specificity,
    per_class_f1,
)
from datasets.label_mapping import class_names as get_class_names


# ---------------------------------------------------------------------------
# Data loading — shared with train_loso.py / train_riemann_loso.py
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PyTorch training helpers
# ---------------------------------------------------------------------------


def _train_spdnet(
    model: nn.Module,
    C_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    verbose: bool = True,
) -> nn.Module:
    """Train SPDNet on pre-computed covariance matrices.

    Parameters
    ----------
    model : SPDNetModel
    C_train : (N_train, C, C)  SPD covariance matrices.
    y_train : (N_train,)  integer labels.
    device : torch.device
    epochs : int
    batch_size : int
    lr : float
    weight_decay : float
    patience : int  Early-stopping patience (0 = no early stopping).
    verbose : bool

    Returns
    -------
    model with best validation state loaded.
    """
    n_samples = len(C_train)

    # Train/val split (90/10, stratified)
    from sklearn.model_selection import train_test_split

    if n_samples >= 20 and patience > 0:
        C_tr, C_val, y_tr, y_val = train_test_split(
            C_train, y_train, test_size=0.10, stratify=y_train, random_state=42
        )
    else:
        C_tr, y_tr = C_train, y_train
        C_val, y_val = None, None

    tr_ds = TensorDataset(
        torch.from_numpy(C_tr).float(),
        torch.from_numpy(y_tr).long(),
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    if C_val is not None and patience > 0:
        val_ds = TensorDataset(
            torch.from_numpy(C_val).float(),
            torch.from_numpy(y_val).long(),
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Weighted CE for class imbalance
    classes = np.unique(y_tr)
    if len(classes) > 1:
        class_weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    else:
        class_weights = np.ones(1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        scheduler.step()

        avg_loss = total_loss / len(C_tr)

        # ---- Validate ----
        if C_val is not None and patience > 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    preds = model(Xb).argmax(-1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_acc = correct / total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stop @ epoch {epoch}  val_acc={val_acc:.4f}")
                break
        else:
            # No validation set — store latest state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and epoch % 20 == 0:
            status = f"    Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}"
            if C_val is not None and patience > 0:
                status += f"  val_acc={val_acc:.4f}"
            print(status)

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _evaluate_spdnet(
    model: nn.Module,
    C_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Evaluate SPDNet on test covariance matrices. Returns metrics dict."""
    ds = TensorDataset(
        torch.from_numpy(C_test).float(),
        torch.from_numpy(y_test).long(),
    )
    loader = DataLoader(ds, batch_size=batch_size)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            all_preds.append(model(Xb).argmax(-1).cpu())
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    metrics = classification_report(y_true, y_pred)
    metrics["per_class_recall"] = per_class_recall(y_true, y_pred)
    metrics["per_class_specificity"] = per_class_specificity(y_true, y_pred)
    metrics["per_class_f1"] = per_class_f1(y_true, y_pred)
    metrics["n_trials"] = len(y_true)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="LOSO evaluation for SPDNet on SPD covariance matrices",
    )
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--n_subjects", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--align", action="store_true",
        help="Apply Euclidean Alignment inside each LOSO fold",
    )
    parser.add_argument(
        "--cov_estimator", default="scm", choices=["scm", "lwf"],
        help="Covariance estimator (default: scm)",
    )
    parser.add_argument(
        "--bimap_dims", type=int, nargs="+", default=[8, 6, 4],
        help="BiMap dimensions (default: 8 6 4)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate before classifier",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device: 'auto', 'cpu', or 'cuda:0'",
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Directory for per-subject CSV and summary JSON",
    )
    parser.add_argument(
        "--dataset", default="physionet_mi",
        choices=["physionet_mi", "bci_iv_2a", "deepbci"],
        help="Dataset name for semantic class labels",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Seed ─────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ────────────────────────────────────────────────────────
    subjects = load_per_subject_data(args.data_dir, args.n_subjects)
    if len(subjects) < 2:
        print("Need at least 2 subjects for LOSO.")
        return

    n_channels = subjects[0]["X"].shape[1]
    n_classes = len(np.unique(np.concatenate([s["y"] for s in subjects])))
    print(f"Channels: {n_channels}  Classes: {n_classes}")

    semantic_names = get_class_names(args.dataset)
    per_subject_results = []

    # ── LOSO loop ────────────────────────────────────────────────────────
    for fold_idx, test_subj in enumerate(subjects):
        test_id = test_subj["id"]
        train_subjs = [s for s in subjects if s["id"] != test_id]

        print(f"\n{'=' * 55}")
        print(f"Fold {fold_idx + 1}/{len(subjects)}: Test=S{test_id:02d}, "
              f"Train={len(train_subjs)} subjects")
        print(f"{'=' * 55}")

        # ── EA alignment (per-fold, no leakage) ──────────────────────────
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

        # ── Covariance computation ───────────────────────────────────────
        X_train_raw = np.concatenate([s["X"] for s in train_subjs], axis=0)
        y_train = np.concatenate([s["y"] for s in train_subjs], axis=0)
        C_train = compute_covariance(X_train_raw, estimator=args.cov_estimator)

        X_test_raw = test_subj["X"]
        y_test = test_subj["y"]
        C_test = compute_covariance(X_test_raw, estimator=args.cov_estimator)

        print(f"  C_train: {C_train.shape}  C_test: {C_test.shape}")

        # ── Create model ─────────────────────────────────────────────────
        model = create_spdnet(
            n_channels=n_channels,
            n_classes=n_classes,
            bimap_dims=args.bimap_dims,
            dropout=args.dropout,
        ).to(device)

        # ── Train ────────────────────────────────────────────────────────
        model = _train_spdnet(
            model=model,
            C_train=C_train,
            y_train=y_train,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            verbose=True,
        )

        # ── Evaluate ─────────────────────────────────────────────────────
        metrics = _evaluate_spdnet(model, C_test, y_test, device)
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

    # ── Export ───────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_tag = f"_{args.dataset}" if args.dataset != "physionet_mi" else ""
    ea_tag = "_ea" if args.align else ""
    cov_tag = f"_{args.cov_estimator}" if args.cov_estimator != "scm" else ""

    csv_name = f"loso_spdnet{ds_tag}{ea_tag}{cov_tag}.csv"
    csv_path = output_dir / csv_name

    if per_subject_results:
        fieldnames = list(per_subject_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_subject_results)

    accs = np.array([r["accuracy"] for r in per_subject_results])
    kappas = np.array([r["kappa"] for r in per_subject_results])
    summary = {
        "method": "spdnet",
        "dataset": args.dataset,
        "n_subjects": len(subjects),
        "n_channels": n_channels,
        "n_classes": int(n_classes),
        "bimap_dims": args.bimap_dims,
        "cov_estimator": args.cov_estimator,
        "align": args.align,
        "epochs": args.epochs,
        "accuracy_mean": round(float(accs.mean()), 4),
        "accuracy_std": round(float(accs.std()), 4),
        "kappa_mean": round(float(kappas.mean()), 4),
        "kappa_std": round(float(kappas.std()), 4),
        "per_subject": per_subject_results,
    }

    print("\n" + "=" * 60)
    print("LOSO SPDNet Summary")
    print("=" * 60)
    print(f"Channels:  {n_channels}    Classes: {n_classes}")
    print(f"BiMap:     {args.bimap_dims}")
    print(f"Estimator: {args.cov_estimator}    Align: {args.align}")
    print(f"Accuracy:  mean={accs.mean():.4f}  std={accs.std():.4f}")
    print(f"Kappa:     mean={kappas.mean():.4f}  std={kappas.std():.4f}")
    print(f"Per-subject: {[f'{a:.3f}' for a in accs]}")
    print(f"Best/Worst: {accs.max():.4f} / {accs.min():.4f}")

    json_name = f"loso_spdnet{ds_tag}{ea_tag}{cov_tag}_summary.json"
    json_path = output_dir / json_name
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCSV  → {csv_path}")
    print(f"JSON → {json_path}")


if __name__ == "__main__":
    main()
