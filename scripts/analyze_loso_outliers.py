"""
Analyze LOSO results to quantify the impact of outlier subjects.

Prints:
    1. Full 30-subject mean ± std
    2. Mean ± std excluding S09 (the worst subject)
    3. Mean ± std excluding all κ < 0 subjects
    4. Bottom-N subjects table with per-class recall/specificity

Usage:
    python scripts/analyze_loso_outliers.py --csv results/loso_brt_det_ea_seed42.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Analyze LOSO outlier impact")
    parser.add_argument("--csv", required=True, help="Path to LOSO per-subject CSV")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # Load CSV
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["accuracy"] = float(r["accuracy"])
            r["kappa"] = float(r["kappa"])
            # Convert all numeric fields to float
            for k in list(r.keys()):
                if k.startswith("recall_") or k.startswith("specificity_"):
                    r[k] = float(r[k])
            rows.append(r)

    accs = np.array([r["accuracy"] for r in rows])
    kappas = np.array([r["kappa"] for r in rows])

    # ── Full stats ──
    print("=" * 60)
    print("LOSO Outlier Impact Analysis")
    print("=" * 60)
    print(f"\nFile: {csv_path.name}")
    print(f"Subjects: {len(rows)}")

    print(f"\n── Full cohort (N={len(rows)}) ──")
    print(f"  Accuracy: {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  Kappa:    {kappas.mean():.4f} ± {kappas.std():.4f}")
    print(f"  Best:     S{np.argmax(accs)+1:02d} = {accs.max():.4f}")
    print(f"  Worst:    S{np.argmin(accs)+1:02d} = {accs.min():.4f}")

    # ── Excluding S09 ──
    s09_mask = np.array([r["subject"] != "S09" for r in rows])
    accs_no_s09 = accs[s09_mask]
    kappas_no_s09 = kappas[s09_mask]
    print(f"\n── Excluding S09 (N={s09_mask.sum()}) ──")
    print(f"  Accuracy: {accs_no_s09.mean():.4f} ± {accs_no_s09.std():.4f}  "
          f"(Δ = {accs_no_s09.mean() - accs.mean():+.4f})")
    print(f"  Kappa:    {kappas_no_s09.mean():.4f} ± {kappas_no_s09.std():.4f}")

    # ── Excluding all κ < 0 subjects ──
    neg_kappa_mask = kappas >= 0
    neg_kappa_subs = [r["subject"] for r, k in zip(rows, kappas) if k < 0]
    accs_no_neg = accs[neg_kappa_mask]
    kappas_no_neg = kappas[neg_kappa_mask]
    print(f"\n── Excluding all κ < 0 subjects (N={neg_kappa_mask.sum()}) ──")
    print(f"  Removed: {', '.join(neg_kappa_subs)}")
    print(f"  Accuracy: {accs_no_neg.mean():.4f} ± {accs_no_neg.std():.4f}  "
          f"(Δ = {accs_no_neg.mean() - accs.mean():+.4f})")
    print(f"  Kappa:    {kappas_no_neg.mean():.4f} ± {kappas_no_neg.std():.4f}")

    # ── Excluding bottom-3 (κ < 0) ──
    sorted_idx = np.argsort(accs)
    bottom_3_mask = np.ones(len(rows), dtype=bool)
    bottom_3_mask[sorted_idx[:3]] = False
    accs_no_b3 = accs[bottom_3_mask]
    kappas_no_b3 = kappas[bottom_3_mask]
    b3_subs = [rows[i]["subject"] for i in sorted_idx[:3]]
    print(f"\n── Excluding bottom-3 (N={bottom_3_mask.sum()}) ──")
    print(f"  Removed: {', '.join(b3_subs)}")
    print(f"  Accuracy: {accs_no_b3.mean():.4f} ± {accs_no_b3.std():.4f}  "
          f"(Δ = {accs_no_b3.mean() - accs.mean():+.4f})")
    print(f"  Kappa:    {kappas_no_b3.mean():.4f} ± {kappas_no_b3.std():.4f}")

    # ── Bottom-10 table ──
    print(f"\n── Bottom-10 subjects ──")
    header = f"  {'Subject':>8}  {'Acc':>7}  {'κ':>7}  {'Recall(Rest)':>13}  {'Recall(Left)':>13}  {'Spec(Rest)':>12}  {'Spec(Left)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in sorted_idx[:10]:
        r = rows[i]
        # Find recall/specificity columns — they vary by dataset but follow pattern
        recall_cols = [k for k in r if k.startswith("recall_")]
        spec_cols = [k for k in r if k.startswith("specificity_")]
        rec_str = "  ".join(f"{r[c]:.3f}" for c in recall_cols[:2])
        spec_str = "  ".join(f"{r[c]:.3f}" for c in spec_cols[:2])
        print(f"  {r['subject']:>8}  {r['accuracy']:.4f}  {r['kappa']:+.4f}  "
              f"{rec_str}  {spec_str}")

    # ── Histogram ──
    print(f"\n── Accuracy distribution ──")
    bins = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.0]
    hist, _ = np.histogram(accs, bins=bins)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        bar = "█" * hist[i]
        sub_list = [rows[j]["subject"] for j in range(len(rows))
                    if lo <= accs[j] < hi]
        print(f"  [{lo:.2f}-{hi:.2f}): {hist[i]:2d}  {bar}  {', '.join(sub_list)}")

    # ── Summary ──
    print(f"\n── Key takeaway ──")
    s09_acc = float(rows[8]["accuracy"])  # S09 is index 8 (0-based)
    gap = accs_no_s09.mean() - accs.mean()
    if gap > 0.02:
        print(f"  S09 alone drags the mean down by {gap:+.2%} — significant outlier impact.")
    elif gap > 0.005:
        print(f"  S09 has moderate impact ({gap:+.2%}) on the mean.")
    else:
        print(f"  S09 impact is negligible ({gap:+.2%}).")

    if len(neg_kappa_subs) > 3:
        print(f"  {len(neg_kappa_subs)} subjects have negative κ — "
              f"LOSO performance is highly subject-dependent.")


if __name__ == "__main__":
    main()
