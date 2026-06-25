"""
Complete EEGNet training script with model selection, augmentation, early stopping.

Usage:
    python training/train_eegnet.py --data_dir data/processed/ --epochs 300
    python training/train_eegnet.py --model eegnet_spatiotemporal --augment
    python training/train_eegnet.py --model eegnet_mhsa --label_smoothing 0.1 --early_stop 30

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
from sklearn.model_selection import StratifiedKFold

# Allow running from project root or module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet_attn import create_model
from utils.config import BATCH_SIZE, EPOCHS as DEFAULT_EPOCHS, LEARNING_RATE
from utils.logger import ExperimentLogger
from utils.metrics import classification_report, per_class_accuracy
from sklearn.metrics import confusion_matrix


def _infer_model_type(ckpt: dict, ckpt_path: str) -> str:
    """Return checkpoint model type with filename fallback for old checkpoints."""
    model_type = ckpt.get("model_type") or ckpt.get("config", {}).get("model_type")
    if model_type:
        return model_type

    name = Path(ckpt_path).name
    for candidate in [
        "eegnet_spatiotemporal",
        "eegnet_temporal",
        "eegnet_mhsa",
        "eegnet_se",
        "eeg_conformer",
        "eeg_tcnet",
        "fbcnet",
    ]:
        if candidate in name:
            return candidate
    return "eegnet"


def _checkpoint_model_kwargs(model_type: str, cfg: dict) -> dict:
    """Build constructor kwargs supported by the checkpoint model type."""
    kwargs = {
        "n_channels": cfg["n_channels"],
        "n_classes": cfg["n_classes"],
    }
    if model_type.startswith("eegnet") or model_type == "eeg_tcnet":
        kwargs.update(
            F1=cfg.get("F1", 8),
            D=cfg.get("D", 2),
            F2=cfg.get("F2", 16),
            dropout=cfg.get("dropout", 0.5),
        )
    elif model_type == "eeg_conformer":
        kwargs.update(
            F1=cfg.get("F1", 8),
            D=cfg.get("D", 2),
            dropout=cfg.get("dropout", 0.5),
        )
    elif model_type == "fbcnet":
        kwargs.update(dropout=cfg.get("dropout", 0.5))
    return kwargs


def warmup_model(model: nn.Module, n_channels: int, n_times: int, device: str) -> None:
    """Run one dummy forward for lazy classifiers before loading weights."""
    if getattr(model, "input_requires_filter_bank", False):
        n_bands = getattr(model, "n_bands", 9)
        dummy = torch.zeros(1, n_bands, n_channels, n_times, device=device)
    else:
        dummy = torch.zeros(1, n_channels, n_times, device=device)
    model.eval()
    with torch.no_grad():
        model(dummy)


def load_checkpoint(ckpt_path: str, device: str = "cpu") -> nn.Module:
    """Load a saved checkpoint, handling architecture and lazy classifier init."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_type = _infer_model_type(ckpt, ckpt_path)
    model = create_model(
        model_type,
        **_checkpoint_model_kwargs(model_type, cfg),
    ).to(device)

    warmup_model(model, cfg["n_channels"], cfg.get("n_times", 750), device)
    model.load_state_dict(ckpt["state_dict"])
    model.model_type = model_type
    model.eval()
    print(
        f"Loaded checkpoint: model={model_type}, "
        f"epoch={ckpt['epoch']}, acc={ckpt['acc']:.4f}"
    )
    return model


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


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * torch.log_softmax(logits, dim=-1), dim=-1))


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher γ → more focus on hard examples.
    alpha : float or None
        Class weight. If None, no weighting.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (n_classes,) tensor or None

    def forward(self, logits, target):
        ce_loss = torch.nn.functional.cross_entropy(
            logits, target, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)  # p_t for the correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train(
    data_dir: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    device: str = None,
    save_path: str = None,
    model_type: str = "eegnet",
    augment: bool = False,
    mixup_alpha: float = 0.0,
    label_smoothing: float = 0.0,
    grad_clip: float = 0.0,
    early_stop: int = 0,
    kfold: int = 0,
    loss_type: str = "ce",
):
    # ---- Setup ----
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {model_type}  Augment: {augment}  Mixup: {mixup_alpha}  "
          f"LabelSmooth: {label_smoothing}  Loss: {loss_type}")
    print(f"GradClip: {grad_clip}  EarlyStop: {early_stop}  KFold: {kfold}")

    X_train, y_train, X_val, y_val = load_data(data_dir)
    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape},   y={y_val.shape}")
    print(f"Classes: {n_classes}")

    # ---- Augmentation (applied once before training) ----
    if augment:
        from preprocessing.augment import augment_dataset
        print("Applying data augmentation...")
        X_train, y_train = augment_dataset(X_train, y_train, factor=2, seed=42)
        print(f"  Augmented: X={X_train.shape}, y={y_train.shape}")

    # ---- KFold cross-validation ----
    if kfold > 1:
        print(f"\nRunning {kfold}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=42)
        fold_scores = []
        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            X_tr, X_va = X_all[train_idx], X_all[val_idx]
            y_tr, y_va = y_all[train_idx], y_all[val_idx]
            fold_acc = _train_one_fold(
                X_tr, y_tr, X_va, y_va,
                model_type, n_channels, n_classes, device,
                epochs, batch_size, lr, label_smoothing, grad_clip, early_stop,
                loss_type=loss_type,
                fold_label=f"fold {fold+1}/{kfold}",
            )
            fold_scores.append(fold_acc)
            print(f"  Fold {fold+1}/{kfold}: acc={fold_acc:.4f}")

        print(f"\nKFold {kfold}-fold: mean={np.mean(fold_scores):.4f} ± {np.std(fold_scores):.3f}")
        return None, None

    # ---- Single train/val run ----
    model, best_ckpt = _train_one_run(
        X_train, y_train, X_val, y_val,
        model_type, n_channels, n_classes, device,
        epochs, batch_size, lr, label_smoothing, grad_clip, early_stop,
        mixup_alpha=mixup_alpha, loss_type=loss_type,
        save_path=save_path,
    )
    return model, best_ckpt


def _train_one_run(
    X_train, y_train, X_val, y_val,
    model_type, n_channels, n_classes, device,
    epochs, batch_size, lr, label_smoothing, grad_clip, early_stop,
    mixup_alpha=0.0,
    loss_type="ce",
    save_path=None,
):
    """Single train/val run with full logging and checkpointing."""
    # ---- Model ----
    model = create_model(model_type, n_channels=n_channels, n_classes=n_classes)
    model = model.to(device)

    # ---- Multi-band preprocessing (FBCNet) ----
    if getattr(model, "input_requires_filter_bank", False):
        from models.fbcnet import apply_filter_bank
        print("Applying filter bank (9 bands)...")
        X_train = apply_filter_bank(X_train)
        X_val = apply_filter_bank(X_val)
        print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

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

    # ---- Loss ----
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if loss_type == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    elif label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger = ExperimentLogger(run_name=model_type)

    best_acc = 0.0
    best_ckpt = None
    best_preds = None
    best_labels = None
    patience_counter = 0

    # ---- Training Loop ----
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            # Mixup
            if mixup_alpha > 0:
                from preprocessing.augment import mixup_batch, mixup_criterion
                Xb_np = Xb.cpu().numpy()
                yb_np = yb.cpu().numpy()
                Xb_mixed, y_a, y_b, lam = mixup_batch(Xb_np, yb_np, alpha=mixup_alpha)
                Xb = torch.from_numpy(Xb_mixed).float().to(device)
                y_a = torch.from_numpy(y_a).long().to(device)
                y_b = torch.from_numpy(y_b).long().to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            if mixup_alpha > 0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                "model_type": model_type,
                "state_dict": model.state_dict(),
                "opt": optimizer.state_dict(),
                "acc": best_acc,
                "config": {
                    "n_channels": n_channels,
                    "n_classes": n_classes,
                    "n_times": X_train.shape[-1],
                    "F1": getattr(model, "F1", 8),
                    "D": getattr(model, "D", 2),
                    "F2": getattr(model, "F2", 16),
                    "dropout": (
                        model.drop1.p if hasattr(model, "drop1")
                        else getattr(model, "dropout", 0.5)
                    ),
                },
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if early_stop > 0 and patience_counter >= early_stop:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ---- Save ----
    if save_path:
        save_path = Path(save_path or ROOT / "checkpoints")
        save_path.mkdir(parents=True, exist_ok=True)
        ckpt_file = save_path / f"{model_type}_best.pt"
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


def _train_one_fold(
    X_train, y_train, X_val, y_val,
    model_type, n_channels, n_classes, device,
    epochs, batch_size, lr, label_smoothing, grad_clip, early_stop,
    loss_type="ce",
    fold_label="",
):
    """Train on one fold and return best validation accuracy (lightweight)."""
    # Multi-band preprocessing for FBCNet
    model_temp = create_model(model_type, n_channels=n_channels,
                               n_classes=n_classes)
    if getattr(model_temp, "input_requires_filter_bank", False):
        from models.fbcnet import apply_filter_bank
        X_train = apply_filter_bank(X_train)
        X_val = apply_filter_bank(X_val)
    del model_temp

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = create_model(model_type, n_channels=n_channels, n_classes=n_classes).to(device)

    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if loss_type == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    elif label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                preds.append(model(Xb).argmax(-1).cpu())
                labels.append(yb)
        acc = (torch.cat(preds) == torch.cat(labels)).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            patience = 0
        else:
            patience += 1
            if early_stop > 0 and patience >= early_stop:
                break

    del model
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEGNet models")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--model", default="eegnet",
                        choices=["eegnet", "eegnet_se", "eegnet_mhsa",
                                 "eegnet_temporal", "eegnet_spatiotemporal",
                                 "fbcnet", "eeg_tcnet", "eeg_conformer"])
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--mixup", type=float, default=0.0,
                        help="Mixup alpha (0 = disabled, e.g. 0.2)")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing factor (e.g., 0.1)")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Gradient clipping max norm (0 = disabled)")
    parser.add_argument("--early_stop", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--kfold", type=int, default=0,
                        help="K-fold cross-validation (0 = disabled, use train/val split)")
    parser.add_argument("--loss", default="ce", choices=["ce", "focal"],
                        help="Loss function: ce (weighted cross-entropy) | focal (focal loss)")
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path,
        model_type=args.model,
        augment=args.augment,
        mixup_alpha=args.mixup,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        early_stop=args.early_stop,
        kfold=args.kfold,
        loss_type=args.loss,
    )
