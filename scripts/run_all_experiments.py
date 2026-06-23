#!/usr/bin/env python
"""
One-command experiment runner.

Orchestrates: preprocess → baseline → train (all model variants) → ablation → LOSO
and saves all outputs to results/.

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --dataset physionet_mi --epochs 60
    python scripts/run_all_experiments.py --skip_loso  # skip expensive LOSO

Output:
    results/
    ├── baseline_csp_svm.json
    ├── eegnet_results.json
    ├── eegnet_spatiotemporal_results.json
    ├── ablation_results.json
    ├── loso_eegnet.csv
    ├── loso_eegnet_summary.json
    ├── loso_eegnet_spatiotemporal.csv
    ├── loso_eegnet_spatiotemporal_summary.json
    └── summary.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# ── Experiment configuration ──────────────────────────────────────────
CONFIG = {
    "dataset": "physionet_mi",
    "channels": "motor8",
    "binary": True,
    "epochs": 60,
    "n_subjects": 30,
    "models": ["eegnet", "eegnet_spatiotemporal"],
    "skip_preprocess": False,
    "skip_baseline": False,
    "skip_loso": False,
}

MODEL_CONFIGS = [
    "eegnet",
    "eegnet_se",
    "eegnet_mhsa",
    "eegnet_temporal",
    "eegnet_spatiotemporal",
]


def run(cmd: list[str], desc: str = "") -> int:
    """Run a subprocess and report status."""
    label = f"  [{desc}]" if desc else ""
    print(f"\n{'='*60}")
    print(f"{label} {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  FAILED (code={result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all BCI experiments")
    parser.add_argument("--dataset", default="physionet_mi")
    parser.add_argument("--channels", default="motor8")
    parser.add_argument("--binary", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--n_subjects", type=int, default=30)
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_loso", action="store_true")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summary = {"timestamp": timestamp, "dataset": args.dataset, "results": {}}

    # ── Step 1: Preprocess ────────────────────────────────────────
    if not args.skip_preprocess:
        cmd = [
            sys.executable, "preprocessing/run_mne_pipeline.py",
            "--dataset", args.dataset,
            "--channels", args.channels,
            "--per_subject",
            "--output", "data/loso_binary",
        ]
        if args.binary:
            cmd.append("--binary")
        if run(cmd, "preprocess") != 0:
            print("Preprocessing failed. Aborting.")
            return

    # ── Step 2: CSP + SVM baseline ─────────────────────────────────
    if not args.skip_baseline:
        cmd = [
            sys.executable, "training/train_baseline.py",
            "--data_dir", "data/processed",
            "--cv", "5",
        ]
        run(cmd, "baseline")

    # ── Step 3: Train all model variants ───────────────────────────
    for model in MODEL_CONFIGS:
        cmd = [
            sys.executable, "training/train_eegnet.py",
            "--model", model,
            "--epochs", str(args.epochs),
            "--data_dir", "data/processed",
        ]
        if args.binary:
            cmd.append("--binary")
        run(cmd, f"train:{model}")

    # ── Step 4: Ablation study ─────────────────────────────────────
    cmd = [
        sys.executable, "training/train_ablation.py",
        "--epochs", str(min(args.epochs, 150)),
        "--repeat", "3",
    ]
    run(cmd, "ablation")

    # ── Step 5: LOSO ───────────────────────────────────────────────
    if not args.skip_loso:
        for model in args.models if hasattr(args, "models") else ["eegnet", "eegnet_spatiotemporal"]:
            cmd = [
                sys.executable, "training/train_loso.py",
                "--data_dir", "data/loso_binary",
                "--n_subjects", str(args.n_subjects),
                "--epochs", str(args.epochs),
                "--model", model,
                "--output_dir", str(RESULTS_DIR),
                "--dataset", args.dataset,
            ]
            run(cmd, f"loso:{model}")

    # ── Step 6: Export summary ─────────────────────────────────────
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All experiments complete!")
    print(f"Results → {RESULTS_DIR}/")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
