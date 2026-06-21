"""
Hyperparameter sweep for EEGNet models using Optuna.

Usage:
    python training/train_sweep.py --trials 50 --data_dir data/processed/
    python training/train_sweep.py --trials 100 --model eegnet_mhsa

The sweep searches over:
    F1 (temporal filters), D (depth multiplier), F2 (pointwise filters),
    dropout rate, learning rate, batch size.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.eegnet_attn import create_model
from utils.config import EPOCHS as DEFAULT_EPOCHS
from utils.metrics import classification_report


def load_data(data_dir: str) -> tuple:
    p = Path(data_dir)
    X_train = np.load(p / "X_train.npy")
    y_train = np.load(p / "y_train.npy")
    X_val = np.load(p / "X_val.npy")
    y_val = np.load(p / "y_val.npy")
    return X_train, y_train, X_val, y_val


def train_one_trial(
    trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str,
    device: str,
    epochs: int,
    early_stop_patience: int = 30,
) -> float:
    """Single Optuna trial — returns best validation accuracy."""

    # ---- Sample hyperparams ----
    F1 = trial.suggest_categorical("F1", [4, 8, 16])
    D = trial.suggest_categorical("D", [1, 2, 4])
    F2 = trial.suggest_categorical("F2", [8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.25, 0.75, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))

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
    model = create_model(
        model_type, n_channels=n_channels, n_classes=n_classes,
        F1=F1, D=D, F2=F2, dropout=dropout,
    )
    model = model.to(device)

    # ---- Loss ----
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # Train
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
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        acc = (y_pred == y_true).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

        # Report intermediate value to Optuna
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep with Optuna")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--model", default="eegnet",
                        choices=["eegnet", "eegnet_se", "eegnet_mhsa",
                                 "eegnet_temporal", "eegnet_spatiotemporal"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Output JSON path for results")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        print("Optuna not installed. Run: pip install optuna")
        print("Falling back to manual grid search...")
        _manual_grid_search(args)
        return

    # Load data once
    X_train, y_train, X_val, y_val = load_data(args.data_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Sweep: model={args.model}, trials={args.trials}, epochs={args.epochs}")
    print(f"Data: train={X_train.shape}, val={X_val.shape}")
    print(f"Device: {device}")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    def objective(trial):
        return train_one_trial(
            trial, X_train, y_train, X_val, y_val,
            model_type=args.model, device=device, epochs=args.epochs,
        )

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # ---- Results ----
    print("\n" + "=" * 60)
    print("Sweep Results")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    results = {
        "model": args.model,
        "n_trials": args.trials,
        "best_accuracy": float(study.best_value),
        "best_params": study.best_params,
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    output_path = Path(args.output or ROOT / "logs" / f"sweep_{args.model}_{datetime.now():%Y%m%d_%H%M%S}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {output_path}")

    # Save best model
    best_params = study.best_params
    best_model = create_model(
        args.model,
        n_channels=X_train.shape[1],
        n_classes=len(np.unique(y_train)),
        F1=best_params["F1"],
        D=best_params["D"],
        F2=best_params["F2"],
        dropout=best_params["dropout"],
    ).to(device)

    # Quick train with best params to get checkpoint
    ckpt_path = ROOT / "checkpoints" / f"sweep_{args.model}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_model.state_dict(),
        "config": {
            "n_channels": X_train.shape[1],
            "n_classes": len(np.unique(y_train)),
            "n_times": X_train.shape[2],
            "F1": best_params["F1"],
            "D": best_params["D"],
            "F2": best_params["F2"],
            "dropout": best_params["dropout"],
        },
        "acc": study.best_value,
        "epoch": 0,
    }
    torch.save(ckpt, ckpt_path)
    print(f"Best config skeleton saved to {ckpt_path}")


def _manual_grid_search(args):
    """Fallback manual grid search when Optuna is not available."""
    X_train, y_train, X_val, y_val = load_data(args.data_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    grid = {
        "F1": [8, 16],
        "D": [2, 4],
        "F2": [16, 32],
        "dropout": [0.25, 0.5],
        "lr": [1e-3, 5e-3],
        "batch_size": [32, 64],
    }

    best_acc = 0.0
    best_config = None

    n_configs = np.prod([len(v) for v in grid.values()])
    print(f"Grid search: {n_configs} configs (grid is small for quick scan)")

    # Lazy cartesian product
    from itertools import product
    keys = list(grid.keys())
    for i, values in enumerate(product(*grid.values())):
        config = dict(zip(keys, values))
        print(f"\n[{i+1}/{n_configs}] {config}")

        n_channels = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        model = create_model(
            args.model, n_channels=n_channels, n_classes=n_classes,
            F1=config["F1"], D=config["D"], F2=config["F2"],
            dropout=config["dropout"],
        ).to(device)

        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long(),
        )
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"])

        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        best_val = 0.0
        for epoch in range(1, min(args.epochs, 50) + 1):  # shorter for grid
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(Xb), yb).backward()
                optimizer.step()
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    preds.append(model(Xb.to(device)).argmax(-1).cpu())
                    labels.append(yb)
            acc = (torch.cat(preds) == torch.cat(labels)).float().mean().item()
            if acc > best_val:
                best_val = acc

        print(f"  val_acc={best_val:.4f}")
        if best_val > best_acc:
            best_acc = best_val
            best_config = {**config, "acc": best_val}

    print(f"\nBest: acc={best_acc:.4f}, config={best_config}")


if __name__ == "__main__":
    main()
