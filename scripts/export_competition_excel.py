#!/usr/bin/env python
"""
Generate competition-format Excel validation report from results/ directory.

Reads result CSVs and JSONs from results/ and produces a polished Excel
workbook matching the competition report format.

Usage:
    python scripts/export_competition_excel.py
    python scripts/export_competition_excel.py --results_dir results/ --output competition_report.xlsx
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys_path = str(ROOT)
import sys

sys.path.insert(0, sys_path)


def gather_results(results_dir: Path) -> dict[str, Any]:
    """
    Scan results/ directory and aggregate all available data into a
    single results dict suitable for report_excel.generate_validation_report().
    """
    results: dict[str, Any] = {
        "summary": {},
        "per_class": [],
        "confusion_matrix": [],
        "ablation": [],
        "per_subject": [],
    }

    # ── Find LOSO summary (most recent / best model) ─────────────────
    for json_file in sorted(results_dir.glob("loso_*_summary.json")):
        with open(json_file) as f:
            loso = json.load(f)
        results["summary"] = {
            "accuracy": loso.get("accuracy_mean", "N/A"),
            "kappa": loso.get("kappa_mean", "N/A"),
            "f1_macro": "N/A",
            "precision": "N/A",
            "recall": "N/A",
            "latency_ms": "N/A",
            "dataset": loso.get("dataset", "N/A"),
            "model": loso.get("model", "N/A"),
            "n_subjects": loso.get("n_subjects", "N/A"),
            "n_trials": "N/A",
            "n_channels": 8,
        }
        results["per_subject"] = [
            {
                "subject_id": s.get("subject", "N/A"),
                "accuracy": s.get("accuracy", "N/A"),
                "n_trials": s.get("n_trials", "N/A"),
            }
            for s in loso.get("per_subject", [])
        ]
        break  # Use first found

    # ── Find ablation results ────────────────────────────────────────
    ablation_json = results_dir / "ablation_results.json"
    if ablation_json.exists():
        with open(ablation_json) as f:
            ablation_data = json.load(f)
        if isinstance(ablation_data, list):
            results["ablation"] = ablation_data

    return results


def main():
    parser = argparse.ArgumentParser(description="Export competition Excel report")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output", default="competition_report.xlsx")
    parser.add_argument("--demo", action="store_true", help="Generate demo report")
    args = parser.parse_args()

    from utils.report_excel import generate_validation_report, generate_demo_report

    if args.demo:
        generate_demo_report(args.output)
        return

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run 'python scripts/run_all_experiments.py' first, or use --demo for a template.")
        return

    results = gather_results(results_dir)
    generate_validation_report(results, args.output)
    print(f"Report → {args.output}")


if __name__ == "__main__":
    main()
