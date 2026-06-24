"""
DeepBCI session loader — converts recorded sessions to training-ready .npy files.

Converts:
    data/subjects/sub_XXX/session_YYYYMMDD_HHMMSS/
    ├── raw.csv       (continuous EEG, Fortran-order flattened per row)
    ├── events.csv    (trial markers: timestamp_s, event_label, class_id)
    ├── metadata.json (session info: n_channels, sfreq, subject_id, ...)
    └── notes.md

To:
    data/deepbci_processed/subj_XXX/
    ├── X.npy  [N, C, T]  float32
    └── y.npy  [N]         int64

Usage:
    python datasets/deepbci_loader.py -i data/subjects/sub_001/session_xxx
    python datasets/deepbci_loader.py --all
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from utils.config import N_CHANNELS, SFREQ, T_MIN, T_MAX


class DeepBCILoader:
    """
    Convert a recorded DeepBCI session to epoched training data.

    Reverses the Fortran-order flattening applied by DeepBCIRecorder,
    then extracts fixed-length epochs around each event marker.

    Parameters
    ----------
    session_dir : str or Path
        Path to session directory (containing raw.csv, events.csv, metadata.json).
    tmin : float
        Seconds before event to start epoch (default -0.5).
    tmax : float
        Seconds after event to end epoch (default 2.5).
    n_channels : int
        Number of EEG channels (default 8).
    sfreq : int
        Sampling frequency in Hz (default 250).
    """

    def __init__(
        self,
        session_dir: str | Path,
        tmin: float = T_MIN,
        tmax: float = T_MAX,
        n_channels: int = N_CHANNELS,
        sfreq: int = SFREQ,
    ):
        self.session_dir = Path(session_dir)
        self.tmin = tmin
        self.tmax = tmax
        self.n_channels = n_channels
        self.sfreq = sfreq
        self._epoch_samples = int((tmax - tmin) * sfreq)

    # ── Public API ─────────────────────────────────────────────

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load session, epoch events, return (X, y).

        Returns
        -------
        X : np.ndarray, shape (N, n_channels, epoch_samples), float32
        y : np.ndarray, shape (N,), int64
            Canonical class IDs from events.csv.
        """
        metadata = self._load_metadata()
        n_channels = metadata.get("n_channels", self.n_channels)
        sfreq = metadata.get("sfreq", self.sfreq)

        # Step 1: load continuous EEG
        timestamps, continuous = self._load_raw(n_channels)

        # Step 2: load events
        events = self._load_events()

        if not events:
            raise ValueError(f"No events found in {self.session_dir / 'events.csv'}")

        # Step 3: epoch each event
        epochs = []
        labels = []
        skipped = 0
        for event_ts, _event_label, class_id in events:
            epoch_data = self._extract_epoch(
                continuous, timestamps, event_ts, n_channels, sfreq
            )
            if epoch_data is None:
                skipped += 1
                continue
            epochs.append(epoch_data)
            labels.append(class_id)

        if skipped > 0:
            warnings.warn(
                f"Skipped {skipped} event(s) near session boundaries "
                f"(insufficient data for {self._epoch_samples}-sample epoch)."
            )

        if not epochs:
            raise ValueError(
                f"No valid epochs could be extracted from {self.session_dir}. "
                f"All {len(events)} events were outside valid data range."
            )

        X = np.stack(epochs, axis=0).astype(np.float32)
        y = np.array(labels, dtype=np.int64)
        return X, y

    # ── Private helpers ────────────────────────────────────────

    def _load_metadata(self) -> dict:
        meta_path = self.session_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _load_raw(
        self, n_channels: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load raw.csv → (timestamps, continuous_signal).

        Reverses Fortran-order flattening applied by DeepBCIRecorder.
        Chunk size is computed per row (may vary).

        raw.csv row format:
            timestamp_s, ch0_s0, ch0_s1, ..., ch0_sM,
                         ch1_s0, ch1_s1, ..., ch1_sM, ...

        Returns
        -------
        timestamps : np.ndarray, shape (total_samples,), float64
        continuous : np.ndarray, shape (n_channels, total_samples), float32
        """
        raw_path = self.session_dir / "raw.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"raw.csv not found: {raw_path}")

        chunks = []
        timestamps_list = []

        with open(raw_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                ts = float(parts[0])
                values = np.array([float(v) for v in parts[1:]], dtype=np.float32)

                n_values = len(values)
                if n_values % n_channels != 0:
                    raise ValueError(
                        f"Row with timestamp {ts} has {n_values} values, "
                        f"not divisible by n_channels={n_channels}"
                    )
                chunk_size = n_values // n_channels

                # Reverse Fortran-order flattening
                chunk = values.reshape(n_channels, chunk_size, order="F")
                chunks.append(chunk)

                # Per-sample timestamps (linear interpolation within chunk)
                chunk_duration = chunk_size / self.sfreq
                sample_times = np.linspace(
                    ts, ts + chunk_duration, chunk_size, endpoint=False
                )
                timestamps_list.append(sample_times)

        if not chunks:
            raise ValueError(f"No data rows found in {raw_path}")

        continuous = np.concatenate(chunks, axis=1)
        all_timestamps = np.concatenate(timestamps_list)
        return all_timestamps, continuous

    def _load_events(self) -> list[tuple[float, str, int]]:
        """Load events.csv → list of (timestamp_s, event_label, class_id)."""
        events_path = self.session_dir / "events.csv"
        if not events_path.exists():
            raise FileNotFoundError(f"events.csv not found: {events_path}")

        events = []
        with open(events_path, "r") as f:
            header = f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                ts = float(parts[0])
                label = parts[1]
                class_id = int(parts[2])
                events.append((ts, label, class_id))

        return events

    def _extract_epoch(
        self,
        continuous: np.ndarray,
        timestamps: np.ndarray,
        event_ts: float,
        n_channels: int,
        sfreq: int,
    ) -> np.ndarray | None:
        """
        Extract epoch window [event_ts + tmin, event_ts + tmax].

        Uses binary search (searchsorted) for O(log n) timestamp lookup.
        Returns None if the window extends beyond available data.

        Returns
        -------
        epoch : np.ndarray, shape (n_channels, epoch_samples) or None
        """
        start_time = event_ts + self.tmin
        end_time = event_ts + self.tmax

        start_idx = int(np.searchsorted(timestamps, start_time, side="left"))
        end_idx = int(np.searchsorted(timestamps, end_time, side="right"))

        if end_idx - start_idx < self._epoch_samples:
            return None

        return continuous[:, start_idx : start_idx + self._epoch_samples].copy()


# ── CLI ────────────────────────────────────────────────────────


def process_session(
    session_dir: str | Path,
    output_root: str | Path,
    tmin: float = T_MIN,
    tmax: float = T_MAX,
) -> Path:
    """
    Process one session → save X.npy / y.npy to output_root/subj_XXX/.

    Returns the output directory path.
    """
    loader = DeepBCILoader(session_dir, tmin=tmin, tmax=tmax)
    X, y = loader.load()

    session_path = Path(session_dir)
    subj_name = session_path.parent.name  # e.g., "sub_001"

    output_dir = Path(output_root) / subj_name
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)

    n_classes = len(np.unique(y))
    print(f"  {subj_name}: X={X.shape}, y={y.shape}, classes={n_classes}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="DeepBCI session loader — convert recorded sessions to X.npy/y.npy"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i",
        help="Path to a session directory (raw.csv + events.csv + metadata.json)",
    )
    group.add_argument(
        "--all", "-a", action="store_true",
        help="Batch process all sessions under data/subjects/",
    )
    parser.add_argument(
        "--output", "-o", default="data/deepbci_processed",
        help="Output root directory (default: data/deepbci_processed)",
    )
    parser.add_argument(
        "--tmin", type=float, default=T_MIN,
        help=f"Pre-event window in seconds (default: {T_MIN})",
    )
    parser.add_argument(
        "--tmax", type=float, default=T_MAX,
        help=f"Post-event window in seconds (default: {T_MAX})",
    )

    args = parser.parse_args()

    if args.all:
        subjects_root = Path("data/subjects")
        if not subjects_root.exists():
            print(f"ERROR: subjects directory not found: {subjects_root}")
            sys.exit(1)

        session_dirs = sorted(subjects_root.glob("sub_*/session_*/"))
        if not session_dirs:
            print(f"No sessions found under {subjects_root}/sub_*/session_*/")
            sys.exit(1)

        print(f"Batch processing {len(session_dirs)} sessions...")
        success = 0
        failed = 0
        for sd in session_dirs:
            try:
                process_session(sd, args.output, tmin=args.tmin, tmax=args.tmax)
                success += 1
            except Exception as e:
                print(f"  FAILED: {sd} — {e}")
                failed += 1
        print(f"\nDone: {success} succeeded, {failed} failed / {len(session_dirs)} total")
    else:
        process_session(args.input, args.output, tmin=args.tmin, tmax=args.tmax)
        print("Done.")


if __name__ == "__main__":
    main()
