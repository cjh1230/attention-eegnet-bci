"""
End-to-end MNE preprocessing — multi-subject support, aggregates across subjects.

Usage:
    python preprocessing/run_mne_pipeline.py
    python preprocessing/run_mne_pipeline.py --input data/raw/physionet_mi --output data/processed/
    python preprocessing/run_mne_pipeline.py --ica   # with ICA artifact removal
"""
import argparse
import sys
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import SFREQ, FREQ_BANDS, T_MIN, T_MAX

# Motor-cortex 16-channel subset (10-10 system) for MI classification
MOTOR_16_CHANNELS = [
    "Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.",
    "C5..", "C3..", "C1..", "Cz..", "C2..", "C4..", "C6..",
    "Cp3.", "Cp4.",
]


def find_eeg_files(input_path: Path) -> list[Path]:
    exts = {".edf", ".fif", ".gdf", ".set", ".bdf", ".vhdr"}
    if input_path.is_file():
        return [input_path]
    files = []
    for ext in exts:
        for f in input_path.rglob(f"*{ext}"):
            # Skip MNE-eegbci-data subdirectories (to avoid double-counting with .fif files)
            if "MNE-eegbci-data" not in str(f):
                files.append(f)
    return sorted(files)


def load_and_filter(file_path: Path) -> mne.io.Raw:
    print(f"  Loading: {file_path.name}")
    raw = mne.io.read_raw(file_path, preload=True, verbose=False)
    print(f"    channels={len(raw.ch_names)}, sfreq={raw.info['sfreq']}, duration={raw.times[-1]:.0f}s")

    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ)

    # Pick EEG only
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False)
    if len(eeg_picks) > 0:
        raw.pick(eeg_picks)

    raw.filter(FREQ_BANDS["full"][0], FREQ_BANDS["full"][1], fir_design="firwin", verbose=False)
    raw.notch_filter(50.0, fir_design="firwin", verbose=False)
    # Common Average Reference — removes shared noise across channels
    raw.set_eeg_reference("average", verbose=False)
    return raw


def extract_events(raw: mne.io.Raw) -> np.ndarray:
    try:
        events = mne.find_events(raw, stim_channel="auto", verbose=False)
    except (ValueError, RuntimeError):
        events, _ = mne.events_from_annotations(raw, verbose=False)
    print(f"    Events: {len(events)}")
    return events


def process_subject(fp: Path, channel_picks: list[str] | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Process one subject file and return (X, y) arrays."""
    raw = load_and_filter(fp)

    # Select specific channels if requested
    if channel_picks is not None:
        available = [ch for ch in channel_picks if ch in raw.ch_names]
        if len(available) < len(channel_picks):
            missing = set(channel_picks) - set(available)
            print(f"    WARNING: missing channels: {missing}")
        raw.pick(available)
        print(f"    Picked {len(available)} motor channels")

    events = extract_events(raw)
    if len(events) == 0:
        return None

    unique_ev = np.unique(events[:, -1])
    if len(unique_ev) < 2:
        print(f"    Only {len(unique_ev)} event type(s) — skipping")
        return None

    # Remap event IDs to consecutive 0,1,2...
    event_id_map = {v: i for i, v in enumerate(sorted(unique_ev))}
    events_remapped = events.copy()
    for old, new in event_id_map.items():
        events_remapped[events[:, -1] == old, -1] = new

    epochs = mne.Epochs(
        raw, events_remapped,
        event_id={str(k): k for k in event_id_map.values()},
        tmin=T_MIN, tmax=T_MAX,
        baseline=None,
        preload=True, verbose=False,
    )

    X = epochs.get_data().astype(np.float32)
    y = epochs.events[:, -1].astype(np.int64)

    # Optionally downsample time dim
    n_times_target = int((T_MAX - T_MIN) * SFREQ)
    if X.shape[2] > n_times_target:
        X = X[:, :, :n_times_target]

    print(f"    -> X={X.shape}, y={y.shape}  (classes: {np.unique(y)})")
    return X, y


def main():
    parser = argparse.ArgumentParser(description="MNE preprocessing pipeline")
    parser.add_argument("--input", default="data/raw/physionet_mi", help="Input directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--ica", action="store_true", help="Apply ICA artifact removal")
    parser.add_argument("--max-channels", type=int, default=64, help="Cap channels to first N")
    parser.add_argument("--channels", default="motor16", choices=["motor16", "all"],
                        help="Channel preset: motor16 (C3/Cz/C4 area) or all (64ch)")
    parser.add_argument("--binary", action="store_true",
                        help="Binary classification: left vs right only (drop T0/rest)")
    args = parser.parse_args()

    # Resolve channel picks
    if args.channels == "motor16":
        channel_picks = MOTOR_16_CHANNELS
    else:
        channel_picks = None

    files = find_eeg_files(Path(args.input))
    if not files:
        print(f"No EEG files found in {args.input}")
        return

    print(f"Found {len(files)} file(s)")

    all_X, all_y = [], []
    subj_labels = []

    for i, fp in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] {fp.parent.name}/{fp.name}")
        result = process_subject(fp, channel_picks)
        if result is None:
            continue
        X, y = result

        # Ensure consistent channel count (only when using "all" mode)
        if channel_picks is None and X.shape[1] > args.max_channels:
            X = X[:, :args.max_channels, :]

        all_X.append(X)
        all_y.append(y)
        subj_labels.extend([i] * len(y))

    if not all_X:
        print("No valid subjects processed.")
        return

    # Aggregate
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"\nTotal: X={X_all.shape}, y={y_all.shape}")
    print(f"Classes: {np.unique(y_all, return_counts=True)}")

    # Binary mode: drop rest (label 0), remap left→0, right→1
    if args.binary:
        mask = y_all != 0
        X_all = X_all[mask]
        y_all = y_all[mask]
        y_all = y_all - 1  # remap: 1→0 (left), 2→1 (right)
        print(f"Binary mode: X={X_all.shape}, y={y_all.shape}")
        print(f"Classes: {np.unique(y_all, return_counts=True)}")

    # Train/val split (subject-independent: 75/25)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42, stratify=y_all
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in [
        ("X_train.npy", X_train), ("y_train.npy", y_train),
        ("X_val.npy", X_val), ("y_val.npy", y_val),
    ]:
        np.save(output_dir / name, arr)
        print(f"Saved {output_dir/name}: {arr.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
