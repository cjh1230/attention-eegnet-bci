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
    print("  Subjects: 1–10")
    print("  Runs: 4, 8, 12 (MI left/right fist)")

    all_raws = []
    for subject in range(1, 11):
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


def main():
    parser = argparse.ArgumentParser(description="Download public EEG datasets")
    parser.add_argument("--sample", action="store_true", help="Download MNE sample (quick)")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.sample:
        download_mne_sample(output_dir)
    else:
        download_physionet_mi(output_dir)


if __name__ == "__main__":
    main()
