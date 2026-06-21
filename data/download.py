"""
Download public MI-EEG datasets for the project.

Supported datasets:
  - BCI Comp IV 2a    (mne.datasets.eegbci — PhysioNet MI)
  - BCI Comp IV 2b    (via MOABB)
  - Sample MNE data   (for quick testing)

Usage:
    python data/download.py          # auto-download PhysioNet MI
    python data/download.py --sample # quick MNE sample test
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def download_physionet_mi(output_dir: Path):
    """
    Download PhysioNet MI dataset.

    This is the standard 2-class MI dataset:
      - 109 subjects
      - 64 EEG channels (@ 160 Hz)
      - Task: left fist vs right fist motor imagery
      - ~45 trials per subject

    We'll download a subset (subjects 1–10) for initial experiments.
    """
    import mne
    from mne.datasets import eegbci

    output_dir = Path(output_dir) / "physionet_mi"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading PhysioNet MI dataset...")
    print("  Subjects: 1–30")
    print("  Runs: 4, 8, 12 (MI left/right fist)")

    all_raws = []
    for subject in range(1, 31):
        subject_dir = output_dir / f"subj_{subject:02d}"
        subject_dir.mkdir(exist_ok=True)

        # Runs 4, 8, 12 = motor imagery (left fist / right fist)
        raw_fif = subject_dir / f"subj_{subject:02d}_mi.fif"

        if raw_fif.exists():
            print(f"  Subject {subject:2d}: already downloaded — loading")
            raw = mne.io.read_raw_fif(raw_fif, preload=True)
        else:
            print(f"  Subject {subject:2d}: downloading runs 4,8,12...")
            raw_list = []
            for run in [4, 8, 12]:
                try:
                    fnames = eegbci.load_data(
                        subject, runs=run, path=str(subject_dir), update_path=True
                    )
                    raw_run = mne.io.read_raw_edf(fnames[0], preload=True)
                    raw_list.append(raw_run)
                except Exception as e:
                    print(f"    Run {run} failed: {e}")
                    continue

            if raw_list:
                raw = mne.concatenate_raws(raw_list)
                raw.save(raw_fif, overwrite=True)
            else:
                continue

        all_raws.append(raw)
        print(f"      channels={len(raw.ch_names)}, samples={len(raw.times)}, duration={raw.times[-1]:.0f}s")

    print(f"\n[DONE] Downloaded {len(all_raws)} subjects -> {output_dir}")
    return output_dir


def download_mne_sample(output_dir: Path):
    """Download MNE sample dataset for quick pipeline testing."""
    import mne

    output_dir = Path(output_dir) / "mne_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MNE sample data...")
    data_path = mne.datasets.sample.data_path(path=str(output_dir), update_path=True)
    print(f"  Path: {data_path}")

    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    if raw_fname.exists():
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
        print(f"  Loaded: channels={len(raw.ch_names)}, duration={raw.times[-1]:.0f}s")
    else:
        print("  Raw file not found — sample download may be incomplete")

    print(f"\n✅ MNE sample data → {data_path}")
    return data_path


def download_bci_iv_2a(output_dir: Path):
    """
    Download BCI Competition IV dataset 2a via MOABB.

    Dataset 2a: 9 subjects, 22 EEG channels @ 250 Hz, 4-class MI
    (left hand, right hand, feet, tongue).
    This is the standard benchmark for MI classification.
    """
    import numpy as np

    output_dir = Path(output_dir) / "bci_iv_2a"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError:
        print("MOABB not installed. Run: pip install moabb")
        return None

    print("Downloading BCI Competition IV dataset 2a via MOABB...")
    print("  9 subjects, 22 EEG channels @ 250 Hz, 4-class MI")
    print("  (left_hand, right_hand, feet, tongue)")

    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        events=["left_hand", "right_hand", "feet", "tongue"],
        n_classes=4,
    )

    # Label mapping: string → int
    label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}

    X_list, y_list = [], []
    for subject in dataset.subject_list:
        print(f"  Subject {subject}...", end=" ")
        try:
            X, y_str, meta = paradigm.get_data(
                dataset=dataset, subjects=[subject]
            )
            # Map string labels to ints
            y_int = np.array([label_map[s] for s in y_str], dtype=np.int64)
            X_list.append(X.astype(np.float32))
            y_list.append(y_int)
            print(f"X={X.shape}, y={X.shape[0]} trials")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

    if X_list:
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        print(f"\n  Total: X={X_all.shape}, y={y_all.shape}")
        print(f"  Classes: {np.unique(y_all, return_counts=True)}")

        # Save as .npy
        np.save(str(output_dir / "X.npy"), X_all)
        np.save(str(output_dir / "y.npy"), y_all)
        print(f"  Saved to {output_dir}/")
    else:
        print("  No data downloaded.")

    print(f"\n✅ BCI IV 2a → {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Download public EEG datasets")
    parser.add_argument("--sample", action="store_true", help="Download MNE sample (quick)")
    parser.add_argument("--bci_iv_2a", action="store_true", help="Download BCI Competition IV 2a")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.sample:
        download_mne_sample(output_dir)
    elif args.bci_iv_2a:
        download_bci_iv_2a(output_dir)
    else:
        download_physionet_mi(output_dir)


if __name__ == "__main__":
    main()
