"""
End-to-end MNE preprocessing — single script, ready to run.

Usage:
    python preprocessing/run_mne_pipeline.py       # auto-detect data/raw/*.edf
    python preprocessing/run_mne_pipeline.py --input data/raw/subj01.edf --output data/processed/
    python preprocessing/run_mne_pipeline.py --input data/raw/ --output data/processed/  # batch

Performs:
  1. Load raw EEG (.edf, .fif, .gdf, .set)
  2. Pick channels, set montage
  3. Bandpass 8–30 Hz, notch 50 Hz
  4. ICA artifact removal (optional, --ica)
  5. Epoch around events
  6. Train/val split per subject
  7. Export as numpy [N, C, T]
"""
import argparse
import sys
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import train_test_split

# Project import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config import (
    N_CHANNELS, SFREQ,
    CHANNEL_NAMES,
    FREQ_BANDS,
    T_MIN, T_MAX,
    EVENT_IDS,
)


def find_eeg_files(input_path: Path) -> list[Path]:
    """Recursively find all EEG files in input_path."""
    exts = {".edf", ".fif", ".gdf", ".set", ".bdf", ".vhdr"}
    if input_path.is_file():
        return [input_path]
    files = []
    for ext in exts:
        files.extend(input_path.rglob(f"*{ext}"))
    return sorted(files)


def load_and_filter(file_path: Path) -> mne.io.Raw:
    """Load raw file and apply bandpass + notch filters."""
    print(f"  Loading: {file_path.name}")
    raw = mne.io.read_raw(file_path, preload=True)
    print(f"    channels={len(raw.ch_names)}, sfreq={raw.info['sfreq']}, duration={raw.times[-1]:.1f}s")

    # Resample if needed
    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ)

    # Pick EEG channels only (exclude EOG, ECG, STIM)
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False)
    if len(eeg_picks) > 0:
        raw.pick(eeg_picks)

    # Filter
    raw.filter(FREQ_BANDS["full"][0], FREQ_BANDS["full"][1], fir_design="firwin")
    raw.notch_filter(50.0, fir_design="firwin")
    return raw


def extract_events(raw: mne.io.Raw) -> np.ndarray:
    """Extract events from STIM channel or annotations."""
    try:
        events = mne.find_events(raw, stim_channel="auto", verbose=False)
    except (ValueError, RuntimeError):
        print("    No STIM channel found — trying annotations...")
        events, _ = mne.events_from_annotations(raw, verbose=False)
    print(f"    Events found: {len(events)}")
    return events


def epoch_and_export(raw: mne.io.Raw, events: np.ndarray, output_dir: Path, subject_id: str):
    """Create epochs, split, and save as .npy."""
    # Map events to 0, 1, 2 labels
    unique_ev = np.unique(events[:, -1])
    if len(unique_ev) < 2:
        print(f"    WARNING: only {len(unique_ev)} event types, skipping")
        return

    event_id_map = {v: i for i, v in enumerate(sorted(unique_ev))}
    # Re-map events
    events_remapped = events.copy()
    for old, new in event_id_map.items():
        events_remapped[events[:, -1] == old, -1] = new

    epochs = mne.Epochs(
        raw,
        events_remapped,
        event_id={str(k): k for k in event_id_map.values()},
        tmin=T_MIN,
        tmax=T_MAX,
        baseline=(T_MIN, 0),
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()          # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]       # (n_epochs,)

    # Downsample time dimension for lighter training
    n_times_target = int((T_MAX - T_MIN) * SFREQ)
    if X.shape[2] > n_times_target:
        X = X[:, :, :n_times_target]

    # Train/val split (75/25)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

    # Save
    for name, arr in [
        ("X_train.npy", X_train),
        ("y_train.npy", y_train),
        ("X_val.npy", X_val),
        ("y_val.npy", y_val),
    ]:
        np.save(output_dir / name, arr.astype(np.float32 if "X" in name else np.int64))

    print(f"    Saved: train=({X_train.shape}, {y_train.shape}), val=({X_val.shape}, {y_val.shape})")
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser(description="MNE preprocessing pipeline")
    parser.add_argument("--input", default="data/raw", help="Input file or directory")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    parser.add_argument("--ica", action="store_true", help="Apply ICA artifact removal")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_eeg_files(input_path)
    if not files:
        print(f"No EEG files found in {input_path}")
        return

    print(f"Found {len(files)} EEG file(s)")

    for i, fp in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing {fp.name}")
        raw = load_and_filter(fp)

        if args.ica:
            print("  Applying ICA...")
            ica = mne.preprocessing.ICA(n_components=15, random_state=42)
            ica.fit(raw)
            eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
            ica.exclude = eog_idx
            raw = ica.apply(raw)

        events = extract_events(raw)
        if len(events) == 0:
            print("  No events — skipping.")
            continue

        subj_id = fp.stem
        subject_dir = output_dir / subj_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        epoch_and_export(raw, events, subject_dir, subj_id)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
