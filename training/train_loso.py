"""
Leave-One-Subject-Out (LOSO) cross-validation for MI classification.

LOSO is the gold-standard evaluation for BCI — it measures how well a model
trained on N-1 subjects generalizes to a completely unseen subject.

Supports:
    - Pure LOSO: train on 29 subjects, test on 1. Repeat 30x.
    - LOSO + Few-shot FT: fine-tune on k trials of the target subject.
    - LOSO + FT Sweep: test multiple FT trial counts in one run.

Usage:
    # Preprocess first:
    python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary

    # Then run LOSO:
    python training/train_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60
    python training/train_loso.py --data_dir data/loso_binary --n_subjects 30 --finetune 10 --model eegnet_spatiotemporal
    python training/train_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align
    python training/train_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --finetune_sweep 0,5,10,20,40

Expected result: Within-subject MI binary typically 75-90% (vs 63% cross-subject).
"""

import argparse
import copy
import csv
import json
import random
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
from preprocessing.alignment import EuclideanAlignment
from training.train_eegnet import load_checkpoint
from utils.metrics import (
    classification_report,
    per_class_accuracy,
    per_class_recall,
    per_class_specificity,
    per_class_f1,
)
from datasets.label_mapping import class_names as get_class_names


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
    augment: bool = False,
    seed: int = 42,
    verbose: bool = False,
    model_kwargs: dict | None = None,
    intermediate_loss_weight: float = 0.3,
    label_smoothing: float = 0.0,
    weight_decay: float = 0.0,
    obj_reg_weight: float = 0.0,
):
    """Train a model on a list of subjects' data. Returns trained model."""
    # Concatenate all training subjects
    X_all = np.concatenate([s["X"] for s in train_subjects], axis=0)
    y_all = np.concatenate([s["y"] for s in train_subjects], axis=0)

    if augment:
        from preprocessing.augment import augment_dataset

        X_all, y_all = augment_dataset(X_all, y_all, factor=2, seed=seed)
        print(f"  Augmented: X={X_all.shape}, y={y_all.shape}")

    n_channels = X_all.shape[1]
    n_classes = len(np.unique(y_all))

    model = create_model(
        model_type, n_channels=n_channels, n_classes=n_classes, **(model_kwargs or {})
    ).to(device)

    # Multi-band preprocessing (FBCNet)
    if getattr(model, "input_requires_filter_bank", False):
        from models.fbcnet import apply_filter_bank

        X_all = apply_filter_bank(X_all)

    train_ds = TensorDataset(
        torch.from_numpy(X_all).float(),
        torch.from_numpy(y_all).long(),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    class_weights = compute_class_weight("balanced", classes=np.unique(y_all), y=y_all)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()

            # Forward — request objectness map if the model supports it
            # and we're using objectness regularization
            if obj_reg_weight > 0 and hasattr(model, "use_objectness"):
                out = model(Xb, return_objectness=True)
                if isinstance(out, tuple):
                    logits, obj_map = out
                else:
                    logits, obj_map = out, None
            else:
                out = model(Xb)
                logits = out
                obj_map = None

            if isinstance(logits, list) and intermediate_loss_weight > 0:
                # ER-MI multi-step output: accumulate loss across steps
                loss = criterion(logits[-1], yb)
                for step_logits in logits[:-1]:
                    loss = loss + intermediate_loss_weight * criterion(step_logits, yb)
            else:
                loss = criterion(logits[-1] if isinstance(logits, list) else logits, yb)

            # Objectness entropy regularization
            if obj_map is not None:
                # obj_map: (B, nb, S, T_cells) in [0,1]
                eps = 1e-6
                obj_reg = -(
                    obj_map * torch.log(obj_map + eps)
                    + (1 - obj_map) * torch.log(1 - obj_map + eps)
                ).mean()
                loss = loss + obj_reg_weight * obj_reg

            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 20 == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:3d}/{epochs}  lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    return model


def evaluate_on_subject(model, subject: dict, device: str) -> dict:
    """Evaluate model on a single subject. Returns metrics dict with per-class breakdown."""
    X_test = subject["X"]
    y_test = subject["y"]

    if getattr(model, "input_requires_filter_bank", False):
        from models.fbcnet import apply_filter_bank

        X_test = apply_filter_bank(X_test)

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

    metrics = classification_report(y_true, y_pred)
    metrics["per_class_recall"] = per_class_recall(y_true, y_pred)
    metrics["per_class_specificity"] = per_class_specificity(y_true, y_pred)
    metrics["per_class_f1"] = per_class_f1(y_true, y_pred)
    metrics["n_trials"] = len(y_true)
    return metrics


def finetune_on_subject(
    model: nn.Module,
    subject: dict,
    n_finetune_trials: int,
    device: str,
    lr: float = 1e-4,
    epochs: int = 20,
    seed: int = 42,
    intermediate_loss_weight: float = 0.3,
    label_smoothing: float = 0.0,
    weight_decay: float = 0.0,
) -> nn.Module:
    """
    Few-shot fine-tune on n_finetune_trials of the target subject.
    Returns the fine-tuned model.
    """
    X = subject["X"]
    y = subject["y"]

    if getattr(model, "input_requires_filter_bank", False):
        from models.fbcnet import apply_filter_bank

        X = apply_filter_bank(X)

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
    ft_loader = DataLoader(
        ft_ds,
        batch_size=min(16, len(ft_indices)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    model.train()
    for epoch in range(epochs):
        for Xb, yb in ft_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            if isinstance(out, list) and intermediate_loss_weight > 0:
                loss = criterion(out[-1], yb)
                for step_logits in out[:-1]:
                    loss = loss + intermediate_loss_weight * criterion(step_logits, yb)
            else:
                loss = criterion(out[-1] if isinstance(out, list) else out, yb)
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

    metrics = classification_report(y_true, y_pred)
    metrics["per_class_recall"] = per_class_recall(y_true, y_pred)
    metrics["per_class_specificity"] = per_class_specificity(y_true, y_pred)
    metrics["per_class_f1"] = per_class_f1(y_true, y_pred)
    metrics["n_trials"] = len(y_true)
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="LOSO cross-validation")
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--n_subjects", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model",
        default="eegnet",
        choices=[
            "eegnet",
            "eegnet_se",
            "eegnet_mhsa",
            "eegnet_temporal",
            "eegnet_spatiotemporal",
            "fbcnet",
            "eeg_tcnet",
            "eeg_conformer",
            "fb_maa_eegnet",
            "maa_eegnet",
            "maa_eegnet_pre",
            "fb_tcnet",
            "spdnet",
            "er_mi",
            "er_mi_v2",
            "brt_det",
        ],
    )
    parser.add_argument(
        "--finetune",
        type=int,
        default=0,
        help="Few-shot FT trials per class (0 = pure LOSO)",
    )
    parser.add_argument(
        "--finetune_sweep",
        type=str,
        default=None,
        help="Comma-separated FT trial counts, e.g. '0,5,10,20,40'. "
        "Mutually exclusive with --finetune.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip per-fold training (use pre-trained checkpoint)",
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Base checkpoint for --skip_train mode"
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory for per-subject CSV and summary JSON",
    )
    parser.add_argument(
        "--dataset",
        default="physionet_mi",
        choices=["physionet_mi", "bci_iv_2a", "deepbci"],
        help="Dataset name for semantic class labels in CSV header",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Apply Euclidean Alignment (EA) inside each LOSO fold "
        "(R_bar computed from training subjects only)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation (2x) to training data",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default=None,
        help="JSON string of extra kwargs for create_model(), "
        "e.g. '{\"steps\": 5}' for ER-MI",
    )
    parser.add_argument(
        "--intermediate_loss_weight",
        type=float,
        default=0.3,
        help="Weight for intermediate step losses in multi-step "
        "models like ER-MI. 0.0 disables intermediate supervision. "
        "Default 0.3.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for CrossEntropyLoss (default 0.0)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam optimizer (default 0.0)",
    )
    parser.add_argument(
        "--obj_reg_weight",
        type=float,
        default=0.0,
        help="Objectness entropy regularization weight "
        "(0.0 = off, 0.001–0.01 recommended for BRT-Det)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag appended to output filenames to "
        "distinguish experiment variants (e.g. 'diff_ch', 'band_gate')",
    )
    args = parser.parse_args()

    if args.finetune > 0 and args.finetune_sweep:
        parser.error("--finetune and --finetune_sweep are mutually exclusive.")
    if args.skip_train and not args.checkpoint:
        parser.error("--skip_train requires --checkpoint.")
    if args.checkpoint and not args.skip_train:
        print(
            "WARNING: --checkpoint is only used with --skip_train; training will run normally."
        )

    # Parse model kwargs
    model_kwargs = {}
    if args.model_kwargs:
        model_kwargs = json.loads(args.model_kwargs)

    # ── Seed ────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve finetune sweep
    ft_sweep = None
    if args.finetune_sweep:
        ft_sweep = [int(x.strip()) for x in args.finetune_sweep.split(",")]
        print(f"Device: {device}  Model: {args.model}  FT sweep: {ft_sweep}")
    else:
        print(
            f"Device: {device}  Model: {args.model}  FT: {args.finetune} trials/class"
        )

    # Load per-subject data
    subjects = load_per_subject_data(args.data_dir, args.n_subjects)
    if len(subjects) < 2:
        print("Need at least 2 subjects for LOSO")
        return

    # ── Label validation ──────────────────────────────────────────
    from datasets.label_mapping import validate_labels, class_names as get_class_names

    semantic_names = get_class_names(args.dataset)
    # Binary PhysioNet MI: data labels are [left, right], not [rest, left, right].
    # Mirrors the override already present in train_spd_loso.py.
    n_data_classes = len(np.unique(np.concatenate([s["y"] for s in subjects])))
    if args.dataset == "physionet_mi" and n_data_classes == 2:
        semantic_names = ["Left Hand", "Right Hand"]
    total_classes = len(semantic_names)
    for subj in subjects:
        if not validate_labels(subj["y"], args.dataset):
            print(
                f"WARNING: Subject {subj['id']} has labels outside expected "
                f"range 0–{total_classes-1} for dataset '{args.dataset}'. "
                f"Got unique labels: {np.unique(subj['y'])}."
            )

    per_subject_results = []  # list of dicts for CSV export

    for i, test_subj in enumerate(subjects):
        test_id = test_subj["id"]
        train_subjs = [s for s in subjects if s["id"] != test_id]

        print(f"\n{'='*50}")
        print(
            f"Fold {i+1}/{len(subjects)}: Test=S{test_id:02d}, Train={len(train_subjs)} subjects"
        )
        print(f"{'='*50}")

        # ── Euclidean Alignment (per-fold, no data leakage) ──────────
        if args.align:
            ea = EuclideanAlignment()
            ea.fit([s["X"] for s in train_subjs])
            # Work on fresh dicts to avoid mutating shared subject data
            _train = []
            for s in train_subjs:
                _train.append({"id": s["id"], "X": ea.transform(s["X"]), "y": s["y"]})
            train_subjs = _train
            test_subj = {
                "id": test_subj["id"],
                "X": ea.transform(test_subj["X"]),
                "y": test_subj["y"],
            }

        # Train on N-1 subjects, or load a fixed base checkpoint for evaluation/FT.
        if args.skip_train:
            model = load_checkpoint(args.checkpoint, device)
            model_type = getattr(model, "model_type", args.model)
            print(f"  Skip train: loaded {model_type} from {args.checkpoint}")
        else:
            model = train_on_subjects(
                train_subjs,
                args.model,
                device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                augment=args.augment,
                seed=args.seed,
                model_kwargs=model_kwargs,
                intermediate_loss_weight=args.intermediate_loss_weight,
                label_smoothing=args.label_smoothing,
                weight_decay=args.weight_decay,
                obj_reg_weight=args.obj_reg_weight,
            )
            model_type = args.model
        base_state = copy.deepcopy(model.state_dict())

        if ft_sweep is not None:
            # ── FT Sweep mode ──────────────────────────────────────
            row = {"subject": f"S{test_id:02d}"}
            for ft_n in ft_sweep:
                if ft_n == 0:
                    metrics = evaluate_on_subject(model, test_subj, device)
                    print(
                        f"  FT={ft_n:>2d}:  acc={metrics['accuracy']:.4f}  "
                        f"kappa={metrics['kappa']:.4f}"
                    )
                else:
                    # Fresh model copy for each FT count.
                    model_ft = copy.deepcopy(model)
                    model_ft.load_state_dict(base_state)
                    _, metrics = finetune_on_subject(
                        model_ft,
                        test_subj,
                        n_finetune_trials=ft_n,
                        device=device,
                        seed=args.seed,
                        intermediate_loss_weight=args.intermediate_loss_weight,
                        label_smoothing=args.label_smoothing,
                        weight_decay=args.weight_decay,
                    )
                    print(
                        f"  FT={ft_n:>2d}:  acc={metrics['accuracy']:.4f}  "
                        f"kappa={metrics['kappa']:.4f}"
                    )
                    del model_ft

                row[f"acc_ft{ft_n}"] = metrics["accuracy"]
                row[f"kappa_ft{ft_n}"] = metrics["kappa"]
            per_subject_results.append(row)

        else:
            # ── Single FT / pure LOSO mode ─────────────────────────
            if args.finetune > 0:
                model, metrics = finetune_on_subject(
                    model,
                    test_subj,
                    n_finetune_trials=args.finetune,
                    device=device,
                    seed=args.seed,
                    intermediate_loss_weight=args.intermediate_loss_weight,
                    label_smoothing=args.label_smoothing,
                    weight_decay=args.weight_decay,
                )
                print(
                    f"  FT+Test: acc={metrics['accuracy']:.4f}  "
                    f"kappa={metrics['kappa']:.4f}"
                )
            else:
                metrics = evaluate_on_subject(model, test_subj, device)
                print(
                    f"  Test:    acc={metrics['accuracy']:.4f}  "
                    f"kappa={metrics['kappa']:.4f}"
                )

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

        del model

    # ---- Export ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ds_tag = f"_{args.dataset}" if args.dataset != "physionet_mi" else ""
    ft_tag = f"_ft{args.finetune}" if args.finetune > 0 else ""
    ea_tag = "_ea" if args.align else ""
    seed_tag = f"_seed{args.seed}"
    variant_tag = f"_{args.tag}" if args.tag else ""

    if ft_sweep is not None:
        # ── FT Sweep CSV ────────────────────────────────────────────
        sweep_tag = f"_ftsweep{ea_tag}"
        csv_path = (
            output_dir
            / f"loso_{args.model}{ds_tag}{sweep_tag}{seed_tag}{variant_tag}.csv"
        )

        if per_subject_results:
            fieldnames = list(per_subject_results[0].keys())
            # Reorder: subject, then sorted FT columns
            ft_keys = sorted(
                [k for k in fieldnames if k.startswith("acc_ft")],
                key=lambda k: int(k.split("ft")[1]),
            )
            kappa_keys = sorted(
                [k for k in fieldnames if k.startswith("kappa_ft")],
                key=lambda k: int(k.split("ft")[1]),
            )
            ordered = ["subject"] + [
                x for pair in zip(ft_keys, kappa_keys) for x in pair
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(per_subject_results)

        # ── FT Sweep Summary ────────────────────────────────────────
        summary = {
            "model": args.model,
            "dataset": args.dataset,
            "n_subjects": len(subjects),
            "finetune_counts": ft_sweep,
            "align": args.align,
            "seed": args.seed,
            "per_subject": per_subject_results,
        }
        print("\n" + "=" * 60)
        print("LOSO FT Sweep Summary")
        print("=" * 60)
        for ft_n in ft_sweep:
            acc_key = f"acc_ft{ft_n}"
            vals = np.array([r[acc_key] for r in per_subject_results])
            summary[f"ft{ft_n}_acc_mean"] = round(float(vals.mean()), 4)
            summary[f"ft{ft_n}_acc_std"] = round(float(vals.std()), 4)
            print(f"FT={ft_n:>2d}:  acc={vals.mean():.4f} ± {vals.std():.4f}")
        json_path = (
            output_dir
            / f"loso_{args.model}{ds_tag}{sweep_tag}{seed_tag}{variant_tag}_summary.json"
        )

    else:
        # ── Single FT / pure LOSO CSV ───────────────────────────────
        csv_path = (
            output_dir
            / f"loso_{args.model}{ds_tag}{ft_tag}{ea_tag}{seed_tag}{variant_tag}.csv"
        )
        if per_subject_results:
            fieldnames = list(per_subject_results[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_subject_results)

        accs = np.array([r["accuracy"] for r in per_subject_results])
        kappas = np.array([r["kappa"] for r in per_subject_results])
        summary = {
            "model": args.model,
            "dataset": args.dataset,
            "n_subjects": len(subjects),
            "finetune_trials": args.finetune,
            "align": args.align,
            "seed": args.seed,
            "accuracy_mean": round(float(accs.mean()), 4),
            "accuracy_std": round(float(accs.std()), 4),
            "kappa_mean": round(float(kappas.mean()), 4),
            "kappa_std": round(float(kappas.std()), 4),
            "per_subject": per_subject_results,
        }
        print("\n" + "=" * 60)
        print("LOSO Summary")
        print("=" * 60)
        print(f"Accuracy:  mean={accs.mean():.4f}  std={accs.std():.4f}")
        print(f"Kappa:     mean={kappas.mean():.4f}  std={kappas.std():.4f}")
        print(f"Per-subject: {[f'{a:.3f}' for a in accs]}")
        print(f"Best/Worst: {accs.max():.4f} / {accs.min():.4f}")
        json_path = (
            output_dir
            / f"loso_{args.model}{ds_tag}{ft_tag}{ea_tag}{seed_tag}{variant_tag}_summary.json"
        )

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCSV saved to {csv_path}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
