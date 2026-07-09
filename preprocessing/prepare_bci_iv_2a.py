"""
Prepare BCI Competition IV 2a data for LOSO training.

Reads the combined X.npy / y.npy saved by data/download.py --bci_iv_2a,
selects 8 motor-cortex channels, and splits into per-subject files.

The raw MOABB download produces X.npy with shape (5184, 22, 1001):
  - 9 subjects × 576 trials each
  - 22 EEG channels (3 EOG + 1 STI already stripped by MOABB)
  - 1001 time points (~4s at 250 Hz)

After this script:
  data/bci_iv_2a_processed/
  ├── subj_01/X.npy (576, 8, 1001)
  ├── subj_02/X.npy (576, 8, 1001)
  ├── ...
  └── subj_09/X.npy (576, 8, 1001)

Usage:
    python preprocessing/prepare_bci_iv_2a.py
    python preprocessing/prepare_bci_iv_2a.py --input data/raw/bci_iv_2a --output data/bci_iv_2a_processed
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from preprocessing.alignment import EuclideanAlignment
from utils.config import MOTOR_CHANNELS_BCI4

# ── BCI IV 2a EEG channel order (from BNCI2014_001 specification) ──
# Verified against MOABB raw data: 22 EEG channels in this exact order.
# The 3 EOG + 1 STI channels are already stripped by MOABB's paradigm.get_data().
BCI4_EEG_CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

# Map 8 motor-cortex channel names to their indices in the 22-channel array
MOTOR_CHANNEL_INDICES = [BCI4_EEG_CHANNELS.index(ch) for ch in MOTOR_CHANNELS_BCI4]

N_SUBJECTS = 9
TRIALS_PER_SUBJECT = 576  # 5184 / 9


def prepare_bci_iv_2a(
    input_path: str | Path = "data/raw/bci_iv_2a",
    output_path: str | Path = "data/bci_iv_2a_processed",
    n_subjects: int = N_SUBJECTS,
    trials_per_subject: int = TRIALS_PER_SUBJECT,
    align: bool = False,
) -> None:
    """
    Load combined X.npy/y.npy, select 8 motor channels, save per-subject.

    Parameters
    ----------
    input_path : str or Path
        Directory containing X.npy and y.npy.
    output_path : str or Path
        Directory to write per-subject subj_XX/ folders.
    n_subjects : int
        Number of subjects (default 9).
    trials_per_subject : int
        Trials per subject (default 576).
    align : bool
        If True, apply Euclidean Alignment globally before per-subject split.
        For LOSO, prefer using --align in train_loso.py instead (per-fold EA).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    X = np.load(input_path / "X.npy").astype(np.float32)
    y = np.load(input_path / "y.npy").astype(np.int64)

    total_trials = n_subjects * trials_per_subject
    if X.shape[0] != total_trials:
        print(
            f"WARNING: Expected {total_trials} trials ({n_subjects} subjects × "
            f"{trials_per_subject}), got {X.shape[0]}. Adjusting trials_per_subject."
        )
        trials_per_subject = X.shape[0] // n_subjects

    # Select 8 motor-cortex channels
    X = X[:, MOTOR_CHANNEL_INDICES, :]
    print(
        f"Selected {len(MOTOR_CHANNEL_INDICES)} motor channels: {MOTOR_CHANNELS_BCI4}"
    )
    print(f"Data: X={X.shape}, y={y.shape}, labels={np.unique(y)}")

    # Euclidean Alignment — global (across all subjects)
    if align:
        print(
            "\n" + "!" * 70 + "\n"
            "LEAKAGE WARNING: --align applies GLOBAL Euclidean Alignment across\n"
            "ALL subjects BEFORE the per-subject split. If these files feed LOSO,\n"
            "every held-out test subject participates in the shared alignment\n"
            "estimate — this LEAKS test-subject information into training.\n"
            "For a clean LOSO estimate, generate files WITHOUT --align and pass\n"
            "--align to training/train_loso.py instead (per-fold EA).\n" + "!" * 70
        )
        ea = EuclideanAlignment()
        X = ea.fit_transform([X])[0]
        print(f"EA aligned (GLOBAL — leaky for LOSO): X={X.shape}")

    # Split into per-subject blocks and save
    output_path.mkdir(parents=True, exist_ok=True)
    for subj_idx in range(n_subjects):
        start = subj_idx * trials_per_subject
        end = start + trials_per_subject
        X_subj = X[start:end]
        y_subj = y[start:end]

        subj_dir = output_path / f"subj_{subj_idx + 1:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        np.save(subj_dir / "X.npy", X_subj)
        np.save(subj_dir / "y.npy", y_subj)
        print(
            f"  subj_{subj_idx + 1:02d}: X={X_subj.shape}, "
            f"y={y_subj.shape}, classes={np.unique(y_subj)}"
        )

    print(f"\nSaved {n_subjects} subjects to {output_path.resolve()}/")
    print(
        "Next: python main.py loso --data_dir data/bci_iv_2a_processed "
        "--n_subjects 9 --epochs 60 --dataset bci_iv_2a"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare BCI IV 2a data for LOSO training"
    )
    parser.add_argument(
        "--input",
        default="data/raw/bci_iv_2a",
        help="Directory containing X.npy and y.npy",
    )
    parser.add_argument(
        "--output",
        default="data/bci_iv_2a_processed",
        help="Output directory for per-subject files",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Apply Euclidean Alignment globally before per-subject split",
    )
    args = parser.parse_args()
    prepare_bci_iv_2a(
        input_path=args.input,
        output_path=args.output,
        align=args.align,
    )


if __name__ == "__main__":
    main()
