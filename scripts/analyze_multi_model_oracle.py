"""
Multi-model oracle upper bound analysis.

Given per-subject LOSO CSVs from N models, compute:
  1. Oracle per-subject (best model per subject) — theoretical ceiling
  2. Per-trial oracle estimate (independence approximation)
  3. Shared failure rate (all models fail)
  4. Pairwise complementarity matrix
  5. Marginal gain per model (diminishing returns)
  6. Per-model kappa<0 count

Answers: "Can these models together exceed 70%?"

Usage:
    python scripts/analyze_multi_model_oracle.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Model definitions: (label, csv_path, column_name)
MODELS = [
    ("Conformer",    "loso_eeg_conformer_ea_seed42.csv",           "accuracy"),
    ("BRT-Det v8",   "loso_brt_det_ea_seed42_band_gate.csv",      "accuracy"),
    ("EEG-TCNet",    "loso_eeg_tcnet_ea_seed42.csv",              "accuracy"),
    ("ER-MI",        "loso_er_mi_ea_seed42.csv",                  "accuracy"),
    ("ER-MI v2",     "loso_er_mi_v2_ea_seed42.csv",               "accuracy"),
    ("Tangent-LDA",  "loso_riemann_tangent_ea.csv",               "accuracy"),
    ("FBCNet",       "loso_fbcnet_ftsweep_ea_seed42.csv",         "acc_ft0"),
]

THRESH_WEAK = 0.55


def load_model(path: str, col: str) -> dict[str, float]:
    """Load per-subject accuracy from a LOSO CSV."""
    data = {}
    full_path = RESULTS / path
    if not full_path.exists():
        print(f"  WARNING: {full_path} not found, skipping")
        return {}
    with open(full_path, newline="") as f:
        for r in csv.DictReader(f):
            data[r["subject"]] = float(r[col])
    return data


def main():
    print("=" * 65)
    print("Multi-Model Oracle Upper Bound Analysis")
    print("=" * 65)

    # Load all models
    all_data = {}
    all_labels = []
    for label, path, col in MODELS:
        data = load_model(path, col)
        if data:
            all_data[label] = data
            all_labels.append(label)

    if len(all_data) < 2:
        print("ERROR: need at least 2 models")
        sys.exit(1)

    # Find common subjects
    common = sorted(set.intersection(*[set(d.keys()) for d in all_data.values()]))
    n_subj = len(common)
    n_models = len(all_data)
    print(f"\nModels: {n_models}  Subjects: {n_subj}")

    # Build accuracy matrix: (n_models, n_subj)
    labels = list(all_data.keys())
    acc_matrix = np.zeros((n_models, n_subj))
    for i, label in enumerate(labels):
        for j, s in enumerate(common):
            acc_matrix[i, j] = all_data[label][s]

    # Per-model summary
    print(f"\n-- Per-Model Summary --")
    print(f"  {'Model':>15s}  {'Mean':>7s}  {'Std':>7s}  {'Min':>7s}  {'Max':>7s}")
    print("  " + "-" * 51)
    for i, label in enumerate(labels):
        row = acc_matrix[i]
        print(f"  {label:>15s}  {row.mean():7.4f}  {row.std():7.4f}  "
              f"{row.min():7.4f}  {row.max():7.4f}")

    # ── 1. Oracle per-subject ──
    oracle_subj = acc_matrix.max(axis=0)  # best model per subject
    oracle_mean = oracle_subj.mean()
    best_single = acc_matrix.mean(axis=1).max()

    print(f"\n-- Oracle Upper Bounds --")
    print(f"  Best single model:           {best_single:.4f}")
    print(f"  Oracle per-subject (N={n_models}): {oracle_mean:.4f}")
    print(f"  Oracle gain over best single: +{oracle_mean - best_single:.4f}")

    # ── 2. Per-trial oracle estimate (independence approximation) ──
    # P(all wrong) ≈ product of (1 - acc_i) for each model
    err_rates = 1 - acc_matrix
    per_trial_oracle = 1 - np.prod(err_rates, axis=0)
    pto_mean = per_trial_oracle.mean()
    print(f"  Per-trial oracle (indep est): {pto_mean:.4f}")
    print(f"  NOTE: per-trial oracle assumes independent errors — optimistic upper bound")

    # ── 3. Shared failure rate ──
    all_weak = np.all(acc_matrix < THRESH_WEAK, axis=0)
    shared_failures = [common[j] for j in range(n_subj) if all_weak[j]]
    print(f"\n-- Shared Failures (all models <{THRESH_WEAK:.0%}) --")
    if shared_failures:
        for s in shared_failures:
            vals = [f"{all_data[l][s]:.3f}" for l in labels]
            print(f"  {s}: {', '.join(f'{l}={v}' for l, v in zip(labels, vals))}")
    else:
        print(f"  (none)")

    # ── 4. Per-subject detail ──
    print(f"\n-- Per-Subject Oracle Detail --")
    print(f"  {'Subj':>6s}  {'Best':>7s}  {'Model':>15s}  {'Worst':>7s}  "
          f"{'Model':>15s}  {'Range':>7s}")
    print("  " + "-" * 68)
    for j, s in enumerate(common):
        best_idx = acc_matrix[:, j].argmax()
        worst_idx = acc_matrix[:, j].argmin()
        best_val = acc_matrix[best_idx, j]
        worst_val = acc_matrix[worst_idx, j]
        print(f"  {s:>6s}  {best_val:7.4f}  {labels[best_idx]:>15s}  "
              f"{worst_val:7.4f}  {labels[worst_idx]:>15s}  "
              f"{best_val - worst_val:7.4f}")

    # ── 5. Per-subject model count reaching threshold ──
    print(f"\n-- Models per Subject (>= {THRESH_WEAK:.0%}) --")
    n_strong = (acc_matrix >= THRESH_WEAK).sum(axis=0)
    for count in range(n_models + 1):
        subs = [common[j] for j in range(n_subj) if n_strong[j] == count]
        if subs:
            print(f"  {count} models strong: {', '.join(subs)}")

    # ── 6. Marginal gain (greedy — add best model first) ──
    print(f"\n-- Marginal Gain (greedy model addition) --")
    remaining = list(range(n_models))
    selected = []
    current_oracle = np.zeros(n_subj)
    for step in range(n_models):
        best_gain = -1
        best_idx = -1
        for i in remaining:
            new_oracle = np.maximum(current_oracle, acc_matrix[i])
            gain = new_oracle.mean() - current_oracle.mean()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_oracle = np.maximum(current_oracle, acc_matrix[best_idx])
        print(f"  +{labels[best_idx]:>15s}: oracle={current_oracle.mean():.4f}  "
              f"(+{best_gain:.4f})")

    # ── 7. Pairwise complementarity ──
    print(f"\n-- Pairwise Complementarity Matrix --")
    print(f"  {'':>15s}", end="")
    for l in labels:
        print(f"  {l:>10s}", end="")
    print()
    for i, la in enumerate(labels):
        print(f"  {la:>15s}", end="")
        for j, lb in enumerate(labels):
            if i == j:
                print(f"  {'---':>10s}", end="")
            else:
                corr = np.corrcoef(acc_matrix[i], acc_matrix[j])[0, 1]
                # Count where i better than j by >5pp
                i_better = (acc_matrix[i] > acc_matrix[j] + 0.05).sum()
                j_better = (acc_matrix[j] > acc_matrix[i] + 0.05).sum()
                print(f"  r={corr:.2f}", end="")
        print()

    # ── 8. Which model is best for each subject ──
    print(f"\n-- Best Model per Subject --")
    from collections import Counter
    best_counts = Counter()
    for j, s in enumerate(common):
        best_idx = acc_matrix[:, j].argmax()
        best_counts[labels[best_idx]] += 1
    for label, count in best_counts.most_common():
        subs = [common[j] for j in range(n_subj)
                if labels[acc_matrix[:, j].argmax()] == label]
        print(f"  {label}: {count} subjects — {', '.join(subs)}")

    # ── 9. Final verdict ──
    print(f"\n{'='*65}")
    print(f"FINAL VERDICT")
    print(f"{'='*65}")
    print(f"  Best single:          {best_single:.4f}")
    print(f"  Oracle per-subject:   {oracle_mean:.4f} (+{oracle_mean - best_single:.4f})")
    print(f"  Per-trial est:        {pto_mean:.4f}")
    if oracle_mean >= 0.70:
        print(f"  VERDICT: 70% is THEORETICALLY POSSIBLE with these models.")
        print(f"  Next: build legitimate fusion strategy.")
    else:
        gap = 0.70 - oracle_mean
        print(f"  VERDICT: 70% is NOT possible with current models alone.")
        print(f"  Gap to 70%: {gap:.4f} — need new information sources.")
        print(f"  Next: add EEGNet, ShallowConvNet, FBCSP-LDA OR accept "
              f"~{oracle_mean:.1%} as the hard ceiling.")


if __name__ == "__main__":
    main()
