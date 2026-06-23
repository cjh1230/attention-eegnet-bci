"""
Standalone subject-wise evaluation — loads a trained checkpoint and
evaluates on each subject individually.

Usage:
    python training/evaluate_subjectwise.py --checkpoint checkpoints/eegnet_best.pt --data_dir data/loso_binary
    python training/evaluate_subjectwise.py --checkpoint checkpoints/eegnet_best.pt --data_dir data/loso_binary --output_dir results/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet_attn import create_model
from utils.metrics import (
    classification_report,
    per_class_recall,
    per_class_specificity,
    per_class_f1,
)
from datasets.label_mapping import class_names as get_class_names


def load_per_subject_data(data_dir: str, n_subjects: int) -> list[dict]:
    """Load per-subject .npy files."""
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


def load_checkpoint(path: str, device: str):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = create_model(
        "eegnet",  # default arch; override if stored in ckpt
        n_channels=cfg["n_channels"],
        n_classes=cfg["n_classes"],
    ).to(device)
    # Warm up lazy classifier
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, cfg["n_channels"], 750, device=device))
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint epoch={ckpt['epoch']} acc={ckpt['acc']:.4f}")
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Subject-wise evaluation from checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--n_subjects", type=int, default=30)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--dataset", default="physionet_mi",
                        choices=["physionet_mi", "bci_iv_2a", "deepbci"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, cfg = load_checkpoint(args.checkpoint, device)
    subjects = load_per_subject_data(args.data_dir, args.n_subjects)

    if not subjects:
        print("No subjects found.")
        return

    # Get semantic class names
    try:
        cls_names = get_class_names(args.dataset)
    except ValueError:
        cls_names = [str(i) for i in range(cfg["n_classes"])]

    results = []
    for subj in subjects:
        test_ds = TensorDataset(
            torch.from_numpy(subj["X"]).float(),
            torch.from_numpy(subj["y"]).long(),
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
        row = {
            "subject": f"S{subj['id']:02d}",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["f1_macro"],
            "kappa": metrics["kappa"],
            "n_trials": len(y_true),
        }
        recalls = per_class_recall(y_true, y_pred)
        for cls_id, val in recalls.items():
            label = cls_names[cls_id] if cls_id < len(cls_names) else f"class_{cls_id}"
            row[f"recall_{label}"] = val

        specs = per_class_specificity(y_true, y_pred)
        for cls_id, val in specs.items():
            label = cls_names[cls_id] if cls_id < len(cls_names) else f"class_{cls_id}"
            row[f"specificity_{label}"] = val

        results.append(row)
        print(f"  S{subj['id']:02d}: acc={metrics['accuracy']:.4f}  f1={metrics['f1_macro']:.4f}")

    # Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "subject_metrics.csv"
    fieldnames = list(results[0].keys()) if results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    accs = np.array([r["accuracy"] for r in results])
    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n_subjects": len(subjects),
        "accuracy_mean": round(float(accs.mean()), 4),
        "accuracy_std": round(float(accs.std()), 4),
        "per_subject": results,
    }
    json_path = output_dir / "subject_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMean acc: {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"CSV → {csv_path}")
    print(f"JSON → {json_path}")


if __name__ == "__main__":
    main()
