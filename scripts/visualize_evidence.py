"""
Visualize BRT-Det evidence maps — band × channel × time heatmaps.

For each subject (or selected subjects), trains a LOSO model (N-1 subjects),
then extracts band-region-time evidence maps from the held-out subject.

Output per subject:
    results/evidence_maps/S{XX}/
        evidence_left_hand.png     — band×ch, band×time, ch×time for left
        evidence_right_hand.png    — same for right
        evidence_diff.png          — (left - right) difference map

Usage:
    # Single subject
    python scripts/visualize_evidence.py --data_dir data/loso_binary --subject 7

    # Compare strong vs weak subjects
    python scripts/visualize_evidence.py --data_dir data/loso_binary --subjects 7,29,9,18

    # All subjects (slow — 30 LOSO folds)
    python scripts/visualize_evidence.py --data_dir data/loso_binary --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models.eegnet_attn import create_model
from models.fbcnet import apply_filter_bank
from training.train_loso import load_per_subject_data, train_on_subjects

OUTPUT = Path("results/evidence_maps")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Channel / band / time labels
CH_NAMES = ["FC3", "C3", "Cz", "C4", "FC4", "CP3", "CPz", "CP4"]
BAND_NAMES = ["8-12", "12-16", "16-20", "20-24", "24-28", "28-30"]  # Hz
TIME_LABELS = [f"t{i}" for i in range(24)]  # 24 time cells


def visualize_subject(
    subject_id: int,
    train_subjects: list[dict],
    test_subject: dict,
    model_kwargs: dict,
    device: str,
    dataset: str = "physionet_mi",
):
    """Train LOSO model, extract evidence, save heatmaps for one subject."""
    from datasets.label_mapping import class_names as get_class_names
    class_names = get_class_names(dataset)

    n_classes = len(np.unique(test_subject["y"]))
    X_test = test_subject["X"]
    y_test = test_subject["y"]

    # Train on N-1 subjects
    print(f"  Training on {len(train_subjects)} subjects...", flush=True)
    model = train_on_subjects(
        train_subjects, "brt_det", device,
        epochs=80, batch_size=64, lr=1e-3,
        model_kwargs=model_kwargs,
        label_smoothing=0.1,
    )
    model.eval()

    # Apply filter bank to test subject
    X_fb = apply_filter_bank(X_test)
    X_t = torch.from_numpy(X_fb).float().to(device)

    # Extract evidence
    evidence_dict = model.extract_evidence(X_t)
    obj = evidence_dict["objectness"].cpu().numpy()       # (N, nb, S, T_cells)
    ev = evidence_dict["evidence"].cpu().numpy()           # (N, nb, S, T_cells, n_cls)
    spatial_labels = evidence_dict["spatial_labels"]
    S = len(spatial_labels)
    nb = ev.shape[1]
    T_cells = ev.shape[3]

    # Per-class average evidence
    subj_dir = OUTPUT / f"S{subject_id:02d}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    for cls_id in range(n_classes):
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        cls_mask = y_test == cls_id
        if cls_mask.sum() == 0:
            continue
        ev_cls = ev[cls_mask].mean(axis=0)   # (nb, S, T_cells, n_cls)
        ev_this = ev_cls[..., cls_id]         # (nb, S, T_cells) — evidence for THIS class
        obj_cls = obj[cls_mask].mean(axis=0)  # (nb, S, T_cells)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"S{subject_id:02d} — {cls_name} Evidence", fontsize=14,
                     fontweight="bold")

        # Panel 1: Band × Channel (average over time)
        ax = axes[0]
        bc = ev_this.mean(axis=2)  # (nb, S)
        im = ax.imshow(bc.T, aspect="auto", cmap="RdBu_r", origin="lower",
                       vmin=-abs(bc).max(), vmax=abs(bc).max())
        ax.set_xticks(range(nb))
        ax.set_xticklabels(BAND_NAMES[:nb], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(S))
        ax.set_yticklabels(spatial_labels, fontsize=9)
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("Channel / Region")
        ax.set_title("Band × Channel\n(avg over time)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Panel 2: Band × Time (average over channels)
        ax = axes[1]
        bt = ev_this.mean(axis=1)  # (nb, T_cells)
        im = ax.imshow(bt, aspect="auto", cmap="RdBu_r", origin="lower",
                       vmin=-abs(bt).max(), vmax=abs(bt).max())
        ax.set_xticks(range(T_cells))
        ax.set_xticklabels([str(i) for i in range(T_cells)], fontsize=7, rotation=90)
        ax.set_yticks(range(nb))
        ax.set_yticklabels(BAND_NAMES[:nb], fontsize=9)
        ax.set_xlabel("Time Cell")
        ax.set_ylabel("Frequency Band")
        ax.set_title("Band × Time\n(avg over channels)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Panel 3: Channel × Time (average over bands)
        ax = axes[2]
        ct = ev_this.mean(axis=0)  # (S, T_cells)
        im = ax.imshow(ct, aspect="auto", cmap="RdBu_r", origin="lower",
                       vmin=-abs(ct).max(), vmax=abs(ct).max())
        ax.set_xticks(range(T_cells))
        ax.set_xticklabels([str(i) for i in range(T_cells)], fontsize=7, rotation=90)
        ax.set_yticks(range(S))
        ax.set_yticklabels(spatial_labels, fontsize=9)
        ax.set_xlabel("Time Cell")
        ax.set_ylabel("Channel / Region")
        ax.set_title("Channel × Time\n(avg over bands)")
        plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        fname = subj_dir / f"evidence_{cls_name}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {fname}", flush=True)

    # ── Difference map (left vs right for binary) ──
    if n_classes == 2:
        cls_left = 1  # left_hand
        cls_right = 1  # same for binary?
        # For PhysioNet binary: class 0=rest, class 1=left_hand
        # Actually PhysioNet binary is left_hand vs right_hand (no rest)
        # Check the actual labels
        unique_labels = sorted(np.unique(y_test))
        if len(unique_labels) == 2:
            c0, c1 = unique_labels[0], unique_labels[1]
            ev_c0 = ev[y_test == c0].mean(axis=0)[..., c0]  # (nb, S, T_cells)
            ev_c1 = ev[y_test == c1].mean(axis=0)[..., c1]
            ev_diff = ev_c1 - ev_c0  # class1 minus class0

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            c0_name = class_names[c0] if c0 < len(class_names) else f"cls{c0}"
            c1_name = class_names[c1] if c1 < len(class_names) else f"cls{c1}"
            fig.suptitle(f"S{subject_id:02d} — Evidence Diff ({c1_name} − {c0_name})",
                         fontsize=14, fontweight="bold")

            # Band × Channel
            ax = axes[0]
            bc = ev_diff.mean(axis=2)
            vmax = max(abs(bc).max(), 0.01)
            im = ax.imshow(bc.T, aspect="auto", cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(nb))
            ax.set_xticklabels(BAND_NAMES[:nb], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(S))
            ax.set_yticklabels(spatial_labels, fontsize=9)
            ax.set_title("Band × Channel diff")
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Band × Time
            ax = axes[1]
            bt = ev_diff.mean(axis=1)
            vmax = max(abs(bt).max(), 0.01)
            im = ax.imshow(bt, aspect="auto", cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(T_cells))
            ax.set_xticklabels([str(i) for i in range(T_cells)], fontsize=7, rotation=90)
            ax.set_yticks(range(nb))
            ax.set_yticklabels(BAND_NAMES[:nb], fontsize=9)
            ax.set_title("Band × Time diff")
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Channel × Time
            ax = axes[2]
            ct = ev_diff.mean(axis=0)
            vmax = max(abs(ct).max(), 0.01)
            im = ax.imshow(ct, aspect="auto", cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(T_cells))
            ax.set_xticklabels([str(i) for i in range(T_cells)], fontsize=7, rotation=90)
            ax.set_yticks(range(S))
            ax.set_yticklabels(spatial_labels, fontsize=9)
            ax.set_title("Channel × Time diff")
            plt.colorbar(im, ax=ax, shrink=0.8)

            plt.tight_layout()
            fname = subj_dir / "evidence_diff.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved {fname}", flush=True)

    del model
    return True


def main():
    parser = argparse.ArgumentParser(description="BRT-Det Evidence Map Visualization")
    parser.add_argument("--data_dir", default="data/loso_binary")
    parser.add_argument("--subject", type=int, default=None,
                        help="Single subject ID to visualize")
    parser.add_argument("--subjects", type=str, default=None,
                        help="Comma-separated subject IDs, e.g. '7,29,9,18'")
    parser.add_argument("--all", action="store_true",
                        help="Visualize all subjects (slow — 30 LOSO folds)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--model_kwargs", type=str, default=None,
                        help="JSON kwargs for BRTDet")
    parser.add_argument("--dataset", default="physionet_mi")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {
        "use_region_pool": False,
        "n_time_cells": 24,
        "dilations": [1, 2, 4],
        "agg_mode": "objectness",
        "use_band_gate": True,
    }

    # Determine which subjects to process
    if args.all:
        subject_ids = list(range(1, 31))
    elif args.subjects:
        subject_ids = [int(x.strip()) for x in args.subjects.split(",")]
    elif args.subject:
        subject_ids = [args.subject]
    else:
        print("ERROR: specify --subject, --subjects, or --all")
        sys.exit(1)

    print(f"Device: {device}")
    print(f"Subjects: {subject_ids}")
    print(f"Output: {OUTPUT}")

    # Load all subjects once
    all_subjects = load_per_subject_data(args.data_dir, 30)
    subj_map = {s["id"]: s for s in all_subjects}

    for sid in subject_ids:
        test_subj = subj_map.get(sid)
        if test_subj is None:
            print(f"S{sid:02d}: not found, skipping")
            continue
        train_subjs = [s for s in all_subjects if s["id"] != sid]
        print(f"\n{'='*50}")
        print(f"S{sid:02d}: train={len(train_subjs)}, test={test_subj['X'].shape[0]} trials")
        print(f"{'='*50}")
        visualize_subject(sid, train_subjs, test_subj, model_kwargs, device,
                          dataset=args.dataset)


if __name__ == "__main__":
    main()
