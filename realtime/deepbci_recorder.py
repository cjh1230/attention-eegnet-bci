"""
DeepBCI data recorder — saves streaming EEG to disk during data collection.

Output per subject:
    data/subjects/sub_XXX/
    ├── raw.csv          # continuous EEG (n_channels × n_samples)
    ├── events.csv       # trial markers (timestamp, event_label, class_id)
    ├── metadata.json    # {subject_id, date, n_channels, sfreq, notes, n_trials}
    └── notes.md         # free-text experiment notes

Usage:
    from realtime.deepbci_recorder import DeepBCIRecorder

    recorder = DeepBCIRecorder(data_root="data/subjects")
    recorder.start_session(subject_id=1, notes="First pilot run — resting state")
    recorder.record_trial(chunk, timestamp=0.0, event_label="rest")
    recorder.record_trial(chunk, timestamp=2.0, event_label="left_hand_cue")
    recorder.end_session()
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.config import N_CHANNELS, SFREQ
from datasets.label_mapping import LABEL_MAPS


class DeepBCIRecorder:
    """
    Records EEG streams to disk with trial markers and metadata.

    Parameters
    ----------
    data_root : str or Path
        Root directory for subject data (e.g. "data/subjects").
    n_channels : int
    s_freq : int
    """

    def __init__(
        self,
        data_root: str | Path = "data/subjects",
        n_channels: int = N_CHANNELS,
        s_freq: int = SFREQ,
    ):
        self.data_root = Path(data_root)
        self.n_channels = n_channels
        self.s_freq = s_freq

        self._session_dir: Path | None = None
        self._raw_file: Path | None = None
        self._events_file: Path | None = None
        self._raw_writer = None
        self._events_writer = None
        self._subject_id: int | None = None
        self._notes: str = ""
        self._trial_count: int = 0
        self._session_start: str = ""

    # ── Public API ─────────────────────────────────────────────────

    def start_session(
        self,
        subject_id: int,
        notes: str = "",
        dataset: str = "deepbci",
    ) -> Path:
        """
        Begin a recording session for *subject_id*.

        Creates data/subjects/sub_XXX/ with metadata.

        Parameters
        ----------
        subject_id : int
            Subject number (1-based).
        notes : str
            Free-text notes saved to notes.md.
        dataset : str
            Dataset key for label mapping (default "deepbci").

        Returns
        -------
        session_dir : Path
        """
        self._subject_id = subject_id
        self._notes = notes
        self._trial_count = 0
        self._session_start = datetime.now().isoformat()

        self._session_dir = self.data_root / f"sub_{subject_id:03d}"
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Notes
        notes_path = self._session_dir / "notes.md"
        notes_path.write_text(
            f"# Subject {subject_id:03d}\n\n"
            f"Date: {self._session_start}\n\n"
            f"{notes}\n",
            encoding="utf-8",
        )

        # Raw CSV
        self._raw_file = self._session_dir / "raw.csv"
        self._raw_writer = open(self._raw_file, "w", newline="")
        self._csv_raw = csv.writer(self._raw_writer)

        # Events CSV
        self._events_file = self._session_dir / "events.csv"
        self._events_writer = open(self._events_file, "w", newline="")
        self._csv_events = csv.writer(self._events_writer)
        self._csv_events.writerow(["timestamp_s", "event_label", "class_id"])

        print(f"Session started: {self._session_dir}")
        return self._session_dir

    def record_chunk(
        self,
        chunk: np.ndarray,
        timestamp_s: float,
        event_label: str | None = None,
    ) -> None:
        """
        Record one chunk of EEG data.

        Parameters
        ----------
        chunk : np.ndarray, shape (n_channels, n_samples)
        timestamp_s : float
            Time in seconds since session start.
        event_label : str or None
            If not None, writes an event marker to events.csv.
            Uses LABEL_MAPS["deepbci"] for class_id lookup.
        """
        if self._raw_writer is None:
            raise RuntimeError("Session not started. Call start_session() first.")

        # Write raw EEG row: timestamp, ch0_sample0, ch0_sample1, ..., chN_sampleM
        row = [f"{timestamp_s:.4f}"] + [f"{v:.6f}" for v in chunk.flatten(order="F")]
        self._csv_raw.writerow(row)

        # Write event marker if present
        if event_label is not None:
            class_id = LABEL_MAPS.get("deepbci", {}).get(event_label, -1)
            self._csv_events.writerow([f"{timestamp_s:.4f}", event_label, class_id])
            self._trial_count += 1

    def end_session(self) -> None:
        """Close files and write metadata.json."""
        if self._raw_writer is not None:
            self._raw_writer.close()
            self._raw_writer = None

        if self._events_writer is not None:
            self._events_writer.close()
            self._events_writer = None

        if self._session_dir is None:
            return

        metadata = {
            "subject_id": self._subject_id,
            "date": self._session_start,
            "n_channels": self.n_channels,
            "sfreq": self.s_freq,
            "n_trials": self._trial_count,
            "notes": self._notes,
        }
        meta_path = self._session_dir / "metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Session ended. {self._trial_count} trials → {self._session_dir}")
        self._session_dir = None
