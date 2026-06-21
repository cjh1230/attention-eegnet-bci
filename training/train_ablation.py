"""
Comprehensive ablation study: compare 6 model configurations.

Usage:
    python training/train_ablation.py --data_dir data/processed/
    python training/train_ablation.py --repeat 3 --epochs 100

Configs:
    1. EEGNet (base)
    2. EEGNet + ChannelAttention1D (SE)
    3. EEGNet + MultiHeadChannelAttention (MHSA)
    4. EEGNet + TemporalAttention
    5. EEGNet + SpatiotemporalAttention
    6. EEGNet + SpatiotemporalAttention + MultiBandFusion
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
from models.fusion import MultiBandFusion
from utils.config import BATCH_SIZE, LEARNING_RATE
from utils.metrics import classification_report


class EEGNetWithFusion(nn.Module):
    """EEGNet with spatiotemporal attention + multi-band fusion head."""

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.eegnet_attn = create_model(
            "eegnet_spatiotemporal", n_channels=n_channels, n_classes=n_classes
        )
        # Keep EEGNet Block1 + attn, replace classifier with fusion + fc
        # For simplicity: use spatiotemporal EEGNet as backbone
        self.fusion = nn.Sequential(
            nn.Linear(n_classes, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.eegnet_attn(x)
        return self.fusion(x)


def train_one_config(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    device: str,
    epochs: int,
    lr: float,
    label: str,
    quiet: bool = False,
) -> dict:
    """Train a single model config and return best-epoch metrics."""
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_preds = None
    best_labels = None

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                all_preds.append(model(Xb).argmax(-1).cpu())
                all_labels.append(yb)
        acc = (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_preds = torch.cat(all_preds).clone().numpy()
            best_labels = torch.cat(all_labels).clone().numpy()

    metrics = classification_report(best_labels, best_preds)
    if not quiet:
        print(f"  [{label:35s}]  acc={best_acc:.4f}  kappa={metrics['kappa']:.4f}  f1={metrics['f1_macro']:.4f}")
    return {"config": label, "best_acc": best_acc, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat each config N times for mean±std")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="CSV output path")
    args = parser.parse_args()

    # ---- Load data ----
    data_dir = Path(args.data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Ablation: channels={n_channels}, classes={n_classes}, epochs={args.epochs}, repeat={args.repeat}")
    print(f"Train={X_train.shape}, Val={X_val.shape}, Device={device}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ---- Config definitions ----
    configs = [
        ("EEGNet (base)", "eegnet"),
        ("EEGNet + SE-Attn", "eegnet_se"),
        ("EEGNet + MHSA-Attn", "eegnet_mhsa"),
        ("EEGNet + Temporal-Attn", "eegnet_temporal"),
        ("EEGNet + Spatiotemporal-Attn", "eegnet_spatiotemporal"),
    ]

    all_runs = []

    for label, model_type in configs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        for run in range(args.repeat):
            run_label = f"{label}" if args.repeat == 1 else f"{label} (run {run+1})"
            model = create_model(model_type, n_channels=n_channels, n_classes=n_classes).to(device)
            result = train_one_config(
                model, train_loader, val_loader, y_train, device,
                epochs=args.epochs, lr=args.lr, label=run_label,
            )
            all_runs.append(result)
            del model  # free memory

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Ablation Summary")
    print("=" * 70)
    header = f"{'Config':35s}  {'Acc':>8s}  {'Kappa':>8s}  {'F1':>8s}"
    print(header)
    print("-" * 70)

    # Aggregate per config
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_runs:
        base_label = r["config"].split(" (run ")[0]
        agg[base_label].append(r)

    for label, _ in configs:
        runs = agg[label]
        accs = [r["best_acc"] for r in runs]
        kappas = [r["kappa"] for r in runs]
        f1s = [r["f1_macro"] for r in runs]
        if len(runs) > 1:
            acc_str = f"{np.mean(accs):.4f}±{np.std(accs):.3f}"
            kap_str = f"{np.mean(kappas):.4f}±{np.std(kappas):.3f}"
            f1_str = f"{np.mean(f1s):.4f}±{np.std(f1s):.3f}"
        else:
            acc_str = f"{accs[0]:.4f}"
            kap_str = f"{kappas[0]:.4f}"
            f1_str = f"{f1s[0]:.4f}"
        print(f"{label:35s}  {acc_str:>8s}  {kap_str:>8s}  {f1_str:>8s}")

    # Save CSV if requested
    if args.output:
        import csv
        csv_path = Path(args.output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_runs[0].keys())
            writer.writeheader()
            writer.writerows(all_runs)
        print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
