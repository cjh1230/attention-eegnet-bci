"""
Dataset metadata — channel info, paradigm, subject counts.

Used for exporting metadata.json and for downstream tools that need
dataset-specific information (e.g. channel selection, paradigm timing).
"""

import json
from pathlib import Path
from typing import Any

DATASET_META: dict[str, dict[str, Any]] = {
    "physionet_mi": {
        "name": "PhysioNet Motor Imagery (eegbci)",
        "n_subjects": 30,
        "n_channels_available": 64,
        "sfreq": 160,
        "paradigm": "motor_imagery",
        "classes": ["rest", "left_hand", "right_hand"],
        "trials_per_class": 30,
        "source": "https://physionet.org/content/eegmmidb/",
        "notes": "BCI2000 64ch; tasks: baseline (T0), left fist (T1), right fist (T2)",
    },
    "bci_iv_2a": {
        "name": "BCI Competition IV — Dataset 2a",
        "n_subjects": 9,
        "n_channels_available": 22,
        "sfreq": 250,
        "paradigm": "motor_imagery",
        "classes": ["left_hand", "right_hand", "feet", "tongue"],
        "trials_per_class": 72,
        "source": "https://www.bbci.de/competition/iv/#dataset2a",
        "notes": "22 EEG channels; cue-based 4-class MI",
    },
    "deepbci": {
        "name": "DeepBCI — Self-collected",
        "n_subjects": 0,
        "n_channels_available": 8,
        "sfreq": 250,
        "paradigm": "motor_imagery",
        "classes": ["idle", "left_hand", "right_hand"],
        "trials_per_class": 0,
        "source": "Self-collected via DeepBCI hardware",
        "notes": "8ch motor-cortex montage; 250 Hz",
    },
}


def get_dataset_meta(dataset: str) -> dict[str, Any]:
    """Return metadata dict for *dataset*."""
    if dataset not in DATASET_META:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {sorted(DATASET_META.keys())}."
        )
    return dict(DATASET_META[dataset])


def export_metadata(output_path: str | Path) -> None:
    """
    Export all dataset metadata as a JSON file.

    Parameters
    ----------
    output_path : str or Path
        Path to write metadata.json.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(DATASET_META, f, indent=2, ensure_ascii=False)
    print(f"Metadata exported to {output_path}")
