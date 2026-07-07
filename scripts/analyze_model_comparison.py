"""
Compare two LOSO model results — per-subject accuracy, error overlap,
4-quadrant classification, and cross-subject std.

Usage:
    python scripts/analyze_model_comparison.py \
        --csv_a results/loso_eeg_conformer_ea_seed42.csv \
        --csv_b results/loso_brt_det_ea_seed42_band_gate.csv \
        --label_a Conformer --label_b BRT-Det
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path("results/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_csv(path: str) -> dict[str, float]:
    """Load LOSO CSV, return {subject: accuracy}."""
    data = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            data[r["subject"]] = float(r["accuracy"])
    return data


def main():
    parser = argparse.ArgumentParser(description="Compare two LOSO model results")
    parser.add_argument("--csv_a", required=True, help="First model CSV")
    parser.add_argument("--csv_b", required=True, help="Second model CSV")
    parser.add_argument("--label_a", default="Model A")
    parser.add_argument("--label_b", default="Model B")
    args = parser.parse_args()

    data_a = load_csv(args.csv_a)
    data_b = load_csv(args.csv_b)

    # Find common subjects
    common = sorted(set(data_a) & set(data_b))
    if not common:
        print("ERROR: no common subjects found")
        sys.exit(1)

    acc_a = np.array([data_a[s] for s in common])
    acc_b = np.array([data_b[s] for s in common])
    labels = common
    n = len(common)

    # -- Basic stats --
    mean_a, std_a = acc_a.mean(), acc_a.std()
    mean_b, std_b = acc_b.mean(), acc_b.std()
    corr = np.corrcoef(acc_a, acc_b)[0, 1]
    diff = acc_a - acc_b

    print("=" * 60)
    print(f"Model Comparison: {args.label_a} vs {args.label_b}")
    print("=" * 60)
    print(f"\n  Subjects: {n}")
    print(f"  {args.label_a:>12s}: {mean_a:.4f} ± {std_a:.4f}  [true subject std]")
    print(f"  {args.label_b:>12s}: {mean_b:.4f} ± {std_b:.4f}  [true subject std]")
    print(f"  Mean diff ({args.label_a} - {args.label_b}): {diff.mean():+.4f}")
    print(f"  Correlation:  r = {corr:.4f}")
    print(f"  R^2:          {corr**2:.4f}")

    # -- 4-quadrant classification --
    THRESH = 0.55  # below this = "weak"
    both_strong = []
    both_weak = []
    a_strong_b_weak = []
    b_strong_a_weak = []
    middle = []

    for s, a, b in zip(labels, acc_a, acc_b):
        if a >= THRESH and b >= THRESH:
            both_strong.append((s, a, b))
        elif a < THRESH and b < THRESH:
            both_weak.append((s, a, b))
        elif a >= THRESH and b < THRESH:
            a_strong_b_weak.append((s, a, b))
        elif b >= THRESH and a < THRESH:
            b_strong_a_weak.append((s, a, b))
        else:
            middle.append((s, a, b))

    def print_group(title, items):
        if not items:
            print(f"\n  {title}: (none)")
            return
        print(f"\n  {title} ({len(items)} subjects):")
        for s, a, b in items:
            print(f"    {s}:  {args.label_a}={a:.3f}  {args.label_b}={b:.3f}  "
                  f"d={a-b:+.3f}")

    print_group(f"Both STRONG (>={THRESH:.0%})", both_strong)
    print_group(f"Both WEAK (<{THRESH:.0%}) -- shared failures, data bottleneck",
                both_weak)
    print_group(f"{args.label_a} strong / {args.label_b} weak — ensemble opportunity",
                a_strong_b_weak)
    print_group(f"{args.label_b} strong / {args.label_a} weak — ensemble opportunity",
                b_strong_a_weak)

    # -- Ensemble potential --
    complement = len(a_strong_b_weak) + len(b_strong_a_weak)
    print(f"\n-- Ensemble Potential --")
    print(f"  Complementary subjects: {complement}/{n} "
          f"({complement/n*100:.1f}%)")
    if complement > 0:
        # If we could "fix" complement subjects -> what's the ceiling?
        fixed_acc = []
        for s, a, b in zip(labels, acc_a, acc_b):
            fixed_acc.append(max(a, b))
        fixed_mean = np.mean(fixed_acc)
        print(f"  Oracle ensemble (take best per subject): {fixed_mean:.4f} "
              f"(d = {fixed_mean - max(mean_a, mean_b):+.4f})")

    # -- Shared failure rate --
    shared_weak = len(both_weak)
    print(f"\n  Shared failures (<{THRESH:.0%}): {shared_weak}/{n} "
          f"({shared_weak/n*100:.1f}%)")
    if shared_weak > 0:
        print(f"  --> These are data-limited, not model-limited.")

    # -- Per-subject detail table --
    print(f"\n-- Per-Subject Detail (sorted by |d|) --")
    sorted_idx = np.argsort(-np.abs(diff))
    print(f"  {'Subj':>6s}  {args.label_a:>10s}  {args.label_b:>10s}  "
          f"{'d':>8s}  {'Winner':>12s}")
    print("  " + "-" * 52)
    for i in sorted_idx:
        s = labels[i]
        d = diff[i]
        if abs(d) < 0.01:
            winner = "tie"
        elif d > 0:
            winner = args.label_a
        else:
            winner = args.label_b
        print(f"  {s:>6s}  {acc_a[i]:10.4f}  {acc_b[i]:10.4f}  "
              f"{d:+8.4f}  {winner:>12s}")

    # -- Scatter plot --
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(acc_a, acc_b, c="steelblue", s=60, zorder=3)
    # Annotate subjects
    for s, a, b in zip(labels, acc_a, acc_b):
        ax.annotate(s, (a, b), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")
    # Quadrant lines
    ax.axhline(THRESH, color="gray", ls="--", alpha=0.5)
    ax.axvline(THRESH, color="gray", ls="--", alpha=0.5)
    # Diagonal
    lims = [min(acc_a.min(), acc_b.min()) - 0.05,
            max(acc_a.max(), acc_b.max()) + 0.05]
    ax.plot(lims, lims, "r--", alpha=0.3, label="equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{args.label_a} Accuracy")
    ax.set_ylabel(f"{args.label_b} Accuracy")
    ax.set_title(f"{args.label_a} vs {args.label_b}\n"
                 f"r={corr:.3f}, mean d={diff.mean():+.3f}, "
                 f"n={n}")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fname = OUTPUT / f"scatter_{args.label_a.replace(' ','_')}_vs_{args.label_b.replace(' ','_')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Scatter saved: {fname}")


if __name__ == "__main__":
    main()
