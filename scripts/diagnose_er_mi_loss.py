"""
Quick ER-MI aux loss weight sweep: test different intermediate_loss_weight
values on S06 (hard failure) and S09 (unstable).

Usage:
    python scripts/diagnose_er_mi_loss.py --data_dir data/loso_binary
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet_attn import create_model
from training.train_loso import load_per_subject_data, train_on_subjects, evaluate_on_subject


def run_single_fold(subjects, test_id, aux_weight, device, epochs=80):
    train_subjs = [s for s in subjects if s["id"] != test_id]
    test_subj = [s for s in subjects if s["id"] == test_id][0]
    n_channels = test_subj["X"].shape[1]
    n_classes = len(set(test_subj["y"]))
    model_kwargs = {"steps": 2, "use_filter_bank": True, "n_bands": 6}

    model = create_model("er_mi", n_channels=n_channels, n_classes=n_classes,
                         **model_kwargs).to(device)
    model = train_on_subjects(train_subjs, "er_mi", device, epochs=epochs,
                              batch_size=64, lr=1e-3, seed=42,
                              model_kwargs=model_kwargs,
                              intermediate_loss_weight=aux_weight,
                              label_smoothing=0.0)
    metrics = evaluate_on_subject(model, test_subj, device)
    del model
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    subjects = load_per_subject_data(args.data_dir, 30)

    weights = [0.0, 0.1, 0.2, 0.3, 0.5]
    for subj_id in [6, 9, 5]:
        print(f"\n{'='*50}")
        print(f"S{subj_id:02d} aux loss weight sweep")
        print(f"{'='*50}")
        print(f"  {'weight':>8s}  {'acc':>8s}  {'kappa':>8s}")
        for w in weights:
            m = run_single_fold(subjects, subj_id, w, device, args.epochs)
            marker = " <-- default" if abs(w - 0.3) < 0.01 else ""
            print(f"  {w:8.1f}  {m['accuracy']:8.4f}  {m['kappa']:8.4f}{marker}")


if __name__ == "__main__":
    main()
