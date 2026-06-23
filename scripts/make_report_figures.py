#!/usr/bin/env python
"""
Generate report figures from results/ data.

Produces:
  - confusion_matrix.png       — heatmap of best-model confusion matrix
  - per_subject_bar.png        — bar chart of per-subject accuracy
  - ablation_comparison.png    — grouped bar chart of ablation configs

Usage:
    python scripts/make_report_figures.py
    python scripts/make_report_figures.py --results_dir results/ --output_dir results/
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))


# ── Matplotlib setup ──────────────────────────────────────────────────
def _setup_mpl():
    """Configure matplotlib for consistent, publication-quality output."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.figsize": (8, 5),
        }
    )
    return plt


def plot_confusion_matrix(cm: list[list[int]], class_names: list[str], output_path: Path) -> None:
    """Plot a confusion matrix heatmap."""
    plt = _setup_mpl()

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues")

    # Annotate cells
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Confusion matrix → {output_path}")


def plot_per_subject(per_subject: list[dict], output_path: Path) -> None:
    """Plot per-subject accuracy bar chart."""
    if not per_subject:
        print("  No per-subject data to plot")
        return

    plt = _setup_mpl()

    subjects = [s.get("subject", s.get("subject_id", "?")) for s in per_subject]
    accs = [s.get("accuracy", 0) for s in per_subject]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4472C4" if a >= np.mean(accs) else "#D9534F" for a in accs]
    bars = ax.bar(range(len(subjects)), accs, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=np.mean(accs), color="gray", linestyle="--", linewidth=1, label=f"Mean: {np.mean(accs):.3f}")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Subject")
    ax.set_title("Per-Subject Accuracy (LOSO)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Per-subject bar → {output_path}")


def plot_ablation(ablation: list[dict], output_path: Path) -> None:
    """Plot ablation comparison as grouped horizontal bars."""
    if not ablation:
        print("  No ablation data to plot")
        return

    plt = _setup_mpl()

    configs = [a.get("config", a.get("configuration", "?")) for a in ablation]
    accs = [a.get("accuracy", a.get("acc", 0)) for a in ablation]
    kappas = [a.get("kappa", a.get("cohen_kappa", 0)) for a in ablation]

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(configs))
    height = 0.35

    bars1 = ax.barh(y - height / 2, accs, height, label="Accuracy", color="#4472C4")
    bars2 = ax.barh(y + height / 2, kappas, height, label="Cohen's κ", color="#ED7D31")

    ax.set_yticks(y)
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("Ablation Study")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right")

    # Annotate bars
    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=7)
    for bar in bars2:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Ablation comparison → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run 'python scripts/run_all_experiments.py' first.")
        return

    # ── Confusion matrix ──────────────────────────────────────────
    # Try to load from LOSO summary or ablation
    for json_file in sorted(results_dir.glob("loso_*_summary.json")):
        with open(json_file) as f:
            loso = json.load(f)
        # If we had confusion matrix in training output, use it
        break

    # ── Per-subject bar ───────────────────────────────────────────
    for json_file in sorted(results_dir.glob("loso_*_summary.json")):
        with open(json_file) as f:
            loso = json.load(f)
        per_subject = loso.get("per_subject", [])
        if per_subject:
            plot_per_subject(per_subject, output_dir / "per_subject_bar.png")
        break

    # ── Ablation comparison ───────────────────────────────────────
    ablation_json = results_dir / "ablation_results.json"
    if ablation_json.exists():
        with open(ablation_json) as f:
            ablation_data = json.load(f)
        if isinstance(ablation_data, list):
            plot_ablation(ablation_data, output_dir / "ablation_comparison.png")

    print(f"\nFigures saved to {output_dir}/")


if __name__ == "__main__":
    main()
