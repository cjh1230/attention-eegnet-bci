"""
Full LOSO ensemble benchmark: Conformer + BRT-Det logits ensemble.

For each LOSO fold, trains both models on N-1 subjects, then evaluates
logits ensemble on the held-out subject with a fixed alpha.

Usage:
    python scripts/ensemble_loso.py --data_dir data/loso_binary \
        --alpha 0.7 --epochs 80 --seed 42 --tag ensemble_a07
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet_attn import create_model
from models.fbcnet import apply_filter_bank
from training.train_loso import load_per_subject_data, train_on_subjects
from utils.metrics import classification_report, per_class_recall


def evaluate_ensemble_on_subject(model_a, model_b, alpha, subject, device,
                                  fb_a=False, fb_b=False):
    """Ensemble logits: alpha*logits_a + (1-alpha)*logits_b. Returns metrics dict."""
    X = subject["X"]
    y = subject["y"]

    X_a = apply_filter_bank(X) if fb_a else X
    X_b = apply_filter_bank(X) if fb_b else X
    X_a_t = torch.from_numpy(X_a).float()
    X_b_t = torch.from_numpy(X_b).float()
    y_t = torch.from_numpy(y).long()

    ds = TensorDataset(X_a_t, X_b_t, y_t)
    loader = DataLoader(ds, batch_size=64)

    model_a.eval()
    model_b.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for Xa, Xb, yb in loader:
            Xa, Xb, yb = Xa.to(device), Xb.to(device), yb.to(device)

            out_a = model_a(Xa)
            logits_a = out_a[-1] if isinstance(out_a, list) else out_a

            out_b = model_b(Xb)
            logits_b = out_b[-1] if isinstance(out_b, list) else out_b

            logits = alpha * logits_a + (1 - alpha) * logits_b
            all_preds.append(logits.argmax(-1).cpu())
            all_labels.append(yb.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return classification_report(y_true, y_pred)


def evaluate_single_model(model, subject, device, is_filter_bank=False):
    """Evaluate a single model. Returns metrics dict."""
    X = subject["X"]
    y = subject["y"]

    if is_filter_bank:
        X = apply_filter_bank(X)

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=64)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            out = model(Xb)
            logits = out[-1] if isinstance(out, list) else out
            all_preds.append(logits.argmax(-1).cpu())
            all_labels.append(yb)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return classification_report(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Full LOSO ensemble benchmark")
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--n_subjects", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Ensemble weight for Conformer (0-1)")
    parser.add_argument("--model_a", default="eeg_conformer")
    parser.add_argument("--model_b", default="brt_det")
    parser.add_argument("--label_a", default="Conformer")
    parser.add_argument("--label_b", default="BRT-Det")
    parser.add_argument("--model_a_kwargs", type=str, default="{}")
    parser.add_argument("--model_b_kwargs", type=str,
                        default='{"use_region_pool":false, "n_time_cells":24, '
                                '"dilations":[1,2,4], "agg_mode":"objectness", '
                                '"use_band_gate":true}')
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--align", action="store_true",
                        help="Apply Euclidean Alignment (EA) inside each fold")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_b_kwargs = json.loads(args.model_b_kwargs)
    model_a_kwargs = json.loads(args.model_a_kwargs)
    alpha = args.alpha

    # Seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    subjects = load_per_subject_data(args.data_dir, args.n_subjects)
    if len(subjects) < 2:
        print("Need at least 2 subjects")
        return

    variant_tag = f"_{args.tag}" if args.tag else ""
    ensemble_rows = []
    solo_a_rows = []
    solo_b_rows = []

    for i, test_subj in enumerate(subjects):
        test_id = test_subj["id"]
        train_subjs = [s for s in subjects if s["id"] != test_id]

        print(f"\n{'='*50}")
        print(f"Fold {i+1}/{len(subjects)}: Test=S{test_id:02d}, "
              f"Train={len(train_subjs)} subjects, alpha={alpha}")
        print(f"{'='*50}")

        # Euclidean Alignment (per-fold, no data leakage)
        if args.align:
            from preprocessing.alignment import EuclideanAlignment
            ea = EuclideanAlignment()
            ea.fit([s["X"] for s in train_subjs])
            train_subjs_aligned = []
            for s in train_subjs:
                train_subjs_aligned.append({"id": s["id"],
                                            "X": ea.transform(s["X"]), "y": s["y"]})
            train_subjs = train_subjs_aligned
            test_subj = {"id": test_subj["id"],
                         "X": ea.transform(test_subj["X"]), "y": test_subj["y"]}

        n_channels = test_subj["X"].shape[1]
        n_classes = len(np.unique(test_subj["y"]))

        # Train model A
        print(f"  Training {args.label_a}...", flush=True)
        model_a = train_on_subjects(
            train_subjs, args.model_a, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            seed=args.seed, model_kwargs=model_a_kwargs,
            label_smoothing=args.label_smoothing,
        )
        fb_a = model_a_kwargs.get("use_filter_bank", False) or getattr(model_a, "input_requires_filter_bank", False)

        # Train model B
        print(f"  Training {args.label_b}...", flush=True)
        model_b = train_on_subjects(
            train_subjs, args.model_b, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            seed=args.seed, model_kwargs=model_b_kwargs,
            label_smoothing=args.label_smoothing,
        )
        fb_b = model_b_kwargs.get("use_filter_bank", False) or getattr(model_b, "input_requires_filter_bank", False)

        # Evaluate solo
        metrics_a = evaluate_single_model(model_a, test_subj, device,
                                          is_filter_bank=fb_a)
        metrics_b = evaluate_single_model(model_b, test_subj, device,
                                          is_filter_bank=fb_b)
        metrics_e = evaluate_ensemble_on_subject(model_a, model_b, alpha,
                                                  test_subj, device,
                                                  fb_a=fb_a, fb_b=fb_b)

        print(f"  {args.label_a}:      acc={metrics_a['accuracy']:.4f}  "
              f"kappa={metrics_a['kappa']:.4f}")
        print(f"  {args.label_b}:     acc={metrics_b['accuracy']:.4f}  "
              f"kappa={metrics_b['kappa']:.4f}")
        print(f"  Ensemble (a={alpha:.1f}): acc={metrics_e['accuracy']:.4f}  "
              f"kappa={metrics_e['kappa']:.4f}  "
              f"gain={metrics_e['accuracy'] - max(metrics_a['accuracy'], metrics_b['accuracy']):+.4f}")

        solo_a_rows.append({"subject": f"S{test_id:02d}",
                            "accuracy": metrics_a["accuracy"],
                            "kappa": metrics_a["kappa"]})
        solo_b_rows.append({"subject": f"S{test_id:02d}",
                            "accuracy": metrics_b["accuracy"],
                            "kappa": metrics_b["kappa"]})
        ensemble_rows.append({"subject": f"S{test_id:02d}",
                              "accuracy": metrics_e["accuracy"],
                              "kappa": metrics_e["kappa"]})

        del model_a, model_b

    # Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_tag = f"_seed{args.seed}"
    alpha_tag = f"_a{str(args.alpha).replace('.', '')}"

    for rows, name in [(solo_a_rows, args.label_a.lower()),
                         (solo_b_rows, args.label_b.lower()),
                         (ensemble_rows, f"ensemble{alpha_tag}")]:
        csv_path = output_dir / f"loso_{name}_ea{seed_tag}{variant_tag}.csv"
        accs = np.array([r["accuracy"] for r in rows])
        kappas = np.array([r["kappa"] for r in rows])
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["subject", "accuracy", "kappa"])
            w.writeheader()
            w.writerows(rows)
        summary = {
            "model": name, "alpha": alpha if "ensemble" in name else None,
            "accuracy_mean": round(float(accs.mean()), 4),
            "accuracy_std": round(float(accs.std()), 4),
            "kappa_mean": round(float(kappas.mean()), 4),
            "kappa_std": round(float(kappas.std()), 4),
            "per_subject": rows,
        }
        json_path = csv_path.with_suffix(".json").as_posix().replace(".csv", "_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{name}: acc={accs.mean():.4f} +/- {accs.std():.4f}  "
              f"kappa={kappas.mean():.4f}")
        print(f"Saved: {csv_path}")

    gain = (np.array([r["accuracy"] for r in ensemble_rows]).mean() -
            max(np.array([r["accuracy"] for r in solo_a_rows]).mean(),
                np.array([r["accuracy"] for r in solo_b_rows]).mean()))
    print(f"\nEnsemble gain over best single: {gain:+.4f}")


if __name__ == "__main__":
    main()
