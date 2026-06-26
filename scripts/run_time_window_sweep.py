#!/usr/bin/env python
"""
Time window sweep for MI-BCI.

Iterates through candidate epoch windows, runs preprocessing and
Riemannian LOSO for each, and produces a summary comparison table.

Usage:
    python scripts/run_time_window_sweep.py [--method tangent] [--align]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Candidate time windows (tmin, tmax) in seconds
WINDOW_CANDIDATES = [
    (-0.5, 2.5),   # current default
    (0.0, 2.0),
    (0.5, 2.5),
    (0.5, 3.0),
    (1.0, 3.0),
    (1.0, 4.0),
]


def run(cmd: list[str], desc: str = "") -> int:
    """Run a command via subprocess, streaming output."""
    header = f"  {desc}" if desc else f"  {' '.join(cmd)}"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main():
    parser = argparse.ArgumentParser(description="Time window sweep")
    parser.add_argument("--method", default="tangent",
                        choices=["tangent", "mdm", "fgmdm"])
    parser.add_argument("--align", action="store_true",
                        help="Apply EA inside each LOSO fold")
    parser.add_argument("--cov_estimator", default="scm")
    parser.add_argument("--metric", default="riemann")
    parser.add_argument("--input", default="data/raw/physionet_mi",
                        help="Raw data directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    results = []

    for tmin, tmax in WINDOW_CANDIDATES:
        tag = f"t{tmin:.1f}_t{tmax:.1f}".replace("-", "m").replace(".", "p")
        output_dir = f"data/loso_binary_{tag}"

        # --- Preprocessing ---
        preproc_cmd = [
            "python", "preprocessing/run_mne_pipeline.py",
            "--input", args.input,
            "--output", output_dir,
            "--channels", "motor8",
            "--binary",
            "--per_subject",
            "--tmin", str(tmin),
            "--tmax", str(tmax),
        ]
        if args.dry_run:
            print(f"[DRY RUN] {' '.join(preproc_cmd)}")
        else:
            rc = run(preproc_cmd, f"Preprocess tmin={tmin}, tmax={tmax}")
            if rc != 0:
                print(f"  WARNING: preprocessing failed for {tag}, skipping")
                continue

        # --- Riemannian LOSO ---
        riemann_cmd = [
            "python", "training/train_riemann_loso.py",
            "--data_dir", output_dir,
            "--n_subjects", "30",
            "--method", args.method,
            "--cov_estimator", args.cov_estimator,
            "--metric", args.metric,
            "--classifier", "lda",
            "--output_dir", f"results/window_sweep",
        ]
        if args.align:
            riemann_cmd.append("--align")

        if args.dry_run:
            print(f"[DRY RUN] {' '.join(riemann_cmd)}")
            results.append({"tmin": tmin, "tmax": tmax, "accuracy": None})
        else:
            rc = run(riemann_cmd, f"LOSO {args.method} on {tag}")
            if rc != 0:
                print(f"  WARNING: LOSO failed for {tag}, skipping")
                continue

            # Parse summary JSON
            summary_path = Path(ROOT) / "results" / "window_sweep" / \
                f"loso_riemann_{args.method}_ea_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                results.append({
                    "tmin": tmin,
                    "tmax": tmax,
                    "accuracy_mean": summary.get("accuracy_mean"),
                    "accuracy_std": summary.get("accuracy_std"),
                    "kappa_mean": summary.get("kappa_mean"),
                    "kappa_std": summary.get("kappa_std"),
                })
                print(f"  -> Accuracy: {summary.get('accuracy_mean', 0)*100:.2f}%")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("TIME WINDOW SWEEP RESULTS")
    print("=" * 70)
    print(f"{'tmin':>8}  {'tmax':>8}  {'Accuracy':>10}  {'Kappa':>8}")
    print("-" * 45)
    best_acc, best_window = 0, None
    for r in results:
        acc = r.get("accuracy_mean")
        if acc is None:
            continue
        print(f"{r['tmin']:>8.1f}  {r['tmax']:>8.1f}  "
              f"{acc*100:>9.2f}%  {r.get('kappa_mean', 0):>8.4f}")
        if acc > best_acc:
            best_acc = acc
            best_window = (r["tmin"], r["tmax"])

    if best_window:
        print(f"\nBest: tmin={best_window[0]:.1f}, tmax={best_window[1]:.1f} "
              f"({best_acc*100:.2f}%)")

    # Save CSV
    out_path = Path(ROOT) / "results" / "window_sweep_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("tmin,tmax,accuracy_mean,accuracy_std,kappa_mean,kappa_std\n")
        for r in results:
            f.write(f"{r['tmin']},{r['tmax']},"
                    f"{r.get('accuracy_mean','')},"
                    f"{r.get('accuracy_std','')},"
                    f"{r.get('kappa_mean','')},"
                    f"{r.get('kappa_std','')}\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
