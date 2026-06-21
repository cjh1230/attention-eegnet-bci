"""
Complete EEGNet training script.

Usage:
    python training/train_eegnet.py --data_dir data/processed/ --epochs 300

Data format expected:
    X_train.npy  — (N, C, T) float32
    y_train.npy  — (N,)    int
    X_val.npy    — (N, C, T) float32
    y_val.npy    — (N,)    int
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

# Allow running from project root or module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet import EEGNet
from utils.config import BATCH_SIZE, EPOCHS as DEFAULT_EPOCHS, LEARNING_RATE
from utils.metrics import classification_report, per_class_accuracy
from sklearn.metrics import confusion_matrix


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> EEGNet:
    """Load a saved EEGNet checkpoint, handling lazy classifier init."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = EEGNet(n_channels=cfg["n_channels"], n_classes=cfg["n_classes"])
    # Warm-up forward to build the lazy classifier
    dummy = torch.zeros(1, cfg["n_channels"], cfg.get("n_times", 750))
    model.eval()
    with torch.no_grad():
        model(dummy)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, acc={ckpt['acc']:.4f}")
    return model
from utils.logger import ExperimentLogger


def load_data(data_dir: str) -> tuple[np.ndarray, ...]:
    """Load preprocessed .npy files from data_dir."""
    p = Path(data_dir)
    files = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy"]
    for f in files:
        if not (p / f).exists():
            raise FileNotFoundError(f"Missing {p/f}. Run preprocessing first.")
    X_train = np.load(p / "X_train.npy")
    y_train = np.load(p / "y_train.npy")
    X_val = np.load(p / "X_val.npy")
    y_val = np.load(p / "y_val.npy")
    return X_train, y_train, X_val, y_val


def train(
    data_dir: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device: str = None,
    save_path: str = None,
):
    # ---- Setup ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X_train, y_train, X_val, y_val = load_data(data_dir)
    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape},   y={y_val.shape}")
    print(f"Classes: {n_classes}")

    # ---- DataLoaders ----
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # ---- Model ----
    model = EEGNet(n_channels=n_channels, n_classes=n_classes, F1=8, D=2, F2=16)
    model = model.to(device)

    # Class-balanced loss
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger = ExperimentLogger(run_name="eegnet")

    best_acc = 0.0
    best_ckpt = None
    best_preds = None
    best_labels = None

    # ---- Training Loop ----
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                val_loss += criterion(logits, yb).item() * Xb.size(0)
                all_preds.append(logits.argmax(-1).cpu())
                all_labels.append(yb.cpu())
        val_loss /= len(val_loader.dataset)

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        metrics = classification_report(y_true, y_pred)

        scheduler.step()

        # Log
        logger.log(
            epoch=epoch,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            **metrics,
        )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{epoch:3d}/{epochs}] "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"acc={metrics['accuracy']:.4f}  kappa={metrics['kappa']:.4f}"
            )

        # Save best
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_preds = y_pred.copy()
            best_labels = y_true.copy()
            best_ckpt = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt": optimizer.state_dict(),
                "acc": best_acc,
                "config": {
                    "n_channels": n_channels,
                    "n_classes": n_classes,
                    "n_times": X_train.shape[2],
                },
            }

    # ---- Save ----
    save_path = Path(save_path or ROOT / "checkpoints")
    save_path.mkdir(parents=True, exist_ok=True)
    ckpt_file = save_path / "eegnet_best.pt"
    torch.save(best_ckpt, ckpt_file)
    print(f"Best checkpoint saved to {ckpt_file} (acc={best_acc:.4f})")

    # ---- Final Eval (best epoch) ----
    print(f"\nPer-class accuracy (best epoch {best_ckpt['epoch']}):")
    pca = per_class_accuracy(best_labels, best_preds)
    for i, acc in enumerate(pca):
        print(f"  Class {i}: {acc:.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(best_labels, best_preds)}")

    logger.close()
    return model, best_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()
    train(**vars(args))
