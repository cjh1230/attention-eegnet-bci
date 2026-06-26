#!/usr/bin/env python
"""
Full ablation study: compares 10 model configurations on PhysioNet MI binary LOSO.

Configurations:
  1. Tangent + LDA              (Riemannian baseline)
  2. Tangent + LDA + EA         (Riemannian + EA)
  3. FgMDM + LDA + EA           (Filter-bank Riemannian)
  4. EEGNet                     (DL baseline)
  5. EEGNet + EA                (DL + EA)
  6. EEGNet + Spatiotemporal + EA  (DL + attention + EA)
  7. FBCNet + EA                (DL + filter bank + EA)
  8. EEG-TCNet + EA             (DL + temporal conv + EA)
  9. EEG-Conformer + EA         (DL + transformer + EA)
  10. FB-MAA-EEGNet + EA        (DL + MAA + EA)

Usage:
    python scripts/run_ablation_all.py [--epochs 80] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = "data/loso_binary"
N_SUBJECTS = 30
OUTPUT_DIR = "results/ablation"


def run(cmd: list[str], desc: str) -> int:
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def parse_summary(result_dir: str, pattern: str) -> dict | None:
    """Find and parse the first JSON summary matching *pattern*."""
    import glob
    candidates = sorted(glob.glob(str(ROOT / result_dir / pattern)))
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Full ablation study")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs for DL models")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--skip-riemann", action="store_true",
                        help="Skip Riemannian baselines (already done)")
    parser.add_argument("--skip-dl", action="store_true",
                        help="Skip DL baselines")
    args = parser.parse_args()

    configs = []

    # ---- Riemannian baselines ----
    if not args.skip_riemann:
        configs.extend([
            {
                "id": 1, "name": "Tangent + LDA",
                "type": "riemann",
                "cmd": [
                    "python", "training/train_riemann_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--method", "tangent",
                    "--cov_estimator", "scm",
                    "--metric", "riemann",
                    "--classifier", "lda",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_riemann_tangent_summary.json",
            },
            {
                "id": 2, "name": "Tangent + LDA + EA",
                "type": "riemann",
                "cmd": [
                    "python", "training/train_riemann_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--method", "tangent",
                    "--cov_estimator", "scm",
                    "--metric", "riemann",
                    "--classifier", "lda",
                    "--align",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_riemann_tangent_ea_summary.json",
            },
            {
                "id": 3, "name": "FgMDM + LDA + EA",
                "type": "riemann",
                "cmd": [
                    "python", "training/train_riemann_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--method", "fgmdm",
                    "--cov_estimator", "scm",
                    "--metric", "riemann",
                    "--classifier", "lda",
                    "--align",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_riemann_fgmdm_ea_summary.json",
            },
        ])

    # ---- DL baselines ----
    if not args.skip_dl:
        configs.extend([
            {
                "id": 4, "name": "EEGNet",
                "type": "dl",
                "cmd": [
                    "python", "training/train_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--model", "eegnet",
                    "--epochs", str(args.epochs),
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_eegnet_summary.json",
            },
            {
                "id": 5, "name": "EEGNet + EA",
                "type": "dl",
                "cmd": [
                    "python", "training/train_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--model", "eegnet",
                    "--epochs", str(args.epochs),
                    "--align",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_eegnet_ea_summary.json",
            },
            {
                "id": 6, "name": "EEGNet + Spatiotemporal + EA",
                "type": "dl",
                "cmd": [
                    "python", "training/train_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--model", "eegnet_spatiotemporal",
                    "--epochs", str(args.epochs),
                    "--align",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_eegnet_spatiotemporal_ea_summary.json",
            },
            {
                "id": 7, "name": "FBCNet + EA",
                "type": "dl",
                "cmd": [
                    "python", "training/train_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--model", "fbcnet",
                    "--epochs", str(args.epochs),
                    "--align",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_fbcnet_ea_summary.json",
            },
            {
                "id": 8, "name": "EEG-TCNet + EA",
                "type": "dl",
                "cmd": [
                    "python", "training/train_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--model", "eeg_tcnet",
                    "--epochs", str(args.epochs),
                    "--align",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_eeg_tcnet_ea_summary.json",
            },
            {
                "id": 9, "name": "EEG-Conformer + EA",
                "type": "dl",
                "cmd": [
                    "python", "training/train_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--model", "eeg_conformer",
                    "--epochs", str(args.epochs),
                    "--align",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_eeg_conformer_ea_summary.json",
            },
            {
                "id": 10, "name": "FB-MAA-EEGNet + EA",
                "type": "dl",
                "cmd": [
                    "python", "training/train_loso.py",
                    "--data_dir", DATA_DIR,
                    "--n_subjects", str(N_SUBJECTS),
                    "--model", "fb_maa_eegnet",
                    "--epochs", str(args.epochs),
                    "--align",
                    "--output_dir", OUTPUT_DIR,
                ],
                "summary_glob": "loso_fb_maa_eegnet_ea_summary.json",
            },
        ])

    # --- Execute ---
    results = []
    for cfg in configs:
        if args.dry_run:
            print(f"[DRY RUN #{cfg['id']}] {cfg['name']}: {' '.join(cfg['cmd'])}")
            results.append({"id": cfg["id"], "name": cfg["name"],
                            "accuracy": None, "kappa": None})
            continue

        rc = run(cfg["cmd"], f"#{cfg['id']} {cfg['name']}")
        if rc != 0:
            print(f"  WARNING: #{cfg['id']} {cfg['name']} failed (rc={rc})")
            results.append({"id": cfg["id"], "name": cfg["name"],
                            "accuracy": None, "kappa": None})
            continue

        # Parse result
        summary = parse_summary(OUTPUT_DIR, cfg["summary_glob"])
        if summary:
            results.append({
                "id": cfg["id"],
                "name": cfg["name"],
                "accuracy_mean": summary.get("accuracy_mean"),
                "accuracy_std": summary.get("accuracy_std"),
                "kappa_mean": summary.get("kappa_mean"),
                "kappa_std": summary.get("kappa_std"),
            })
            print(f"  -> {summary.get('accuracy_mean', 0)*100:.2f}% "
                  f"± {summary.get('accuracy_std', 0)*100:.2f}%")
        else:
            results.append({"id": cfg["id"], "name": cfg["name"],
                            "accuracy": None, "kappa": None})

    # --- Print table ---
    print("\n" + "=" * 75)
    print("ABLATION STUDY RESULTS")
    print("=" * 75)
    print(f"{'#':>3}  {'Configuration':<35}  {'Accuracy':>10}  {'Kappa':>8}")
    print("-" * 75)
    for r in results:
        acc = r.get("accuracy_mean")
        kap = r.get("kappa_mean")
        if acc is not None:
            acc_str = f"{acc*100:.2f}% ± {r.get('accuracy_std', 0)*100:.2f}%"
            kap_str = f"{kap:.4f}"
        else:
            acc_str = "N/A"
            kap_str = "N/A"
        print(f"{r['id']:>3}  {r['name']:<35}  {acc_str:>10}  {kap_str:>8}")

    # --- Save CSV ---
    out_csv = Path(ROOT) / OUTPUT_DIR / "ablation_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("id,name,accuracy_mean,accuracy_std,kappa_mean,kappa_std\n")
        for r in results:
            f.write(f"{r['id']},{r['name']},"
                    f"{r.get('accuracy_mean','')},"
                    f"{r.get('accuracy_std','')},"
                    f"{r.get('kappa_mean','')},"
                    f"{r.get('kappa_std','')}\n")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
