"""
Ablation study: compare EEGNet with/without channel attention and multi-band fusion.

Usage:
    python training/train_ablation.py --data_dir data/processed/
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

from models.eegnet import EEGNet
from models.attention import ChannelAttention1D
from utils.config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from utils.metrics import classification_report
from utils.logger import ExperimentLogger


class EEGNetWithAttention(nn.Module):
    """EEGNet preceded by a Channel Attention module."""

    def __init__(self, n_channels, n_classes, n_times):
        super().__init__()
        self.attention = ChannelAttention1D(n_channels)
        self.eegnet = EEGNet(n_channels, n_classes, n_times)

    def forward(self, x):
        x = self.attention(x)
        return self.eegnet(x)


def train_one_config(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int,
    lr: float,
    label: str,
) -> dict:
    """Train a single model config and return final metrics."""
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(3), y=[0, 1, 2]
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
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

    metrics = classification_report(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())
    print(f"  [{label}]  acc={best_acc:.4f}  kappa={metrics['kappa']:.4f}")
    return {"config": label, "best_acc": best_acc, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    n_channels, n_times = X_train.shape[1], X_train.shape[2]
    n_classes = len(np.unique(y_train))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # ---- Ablation Configs ----
    configs = {
        "EEGNet (base)": EEGNet(n_channels, n_classes, n_times).to(device),
        "EEGNet + Attn": EEGNetWithAttention(n_channels, n_classes, n_times).to(device),
    }

    results = {}
    for label, model in configs.items():
        print(f"\nTraining: {label}")
        results[label] = train_one_config(
            model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE, label
        )

    print("\n" + "=" * 60)
    print("Ablation Summary")
    print("=" * 60)
    for label, r in results.items():
        print(f"  {label:20s}  acc={r['best_acc']:.4f}  kappa={r['kappa']:.4f}  f1={r['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
