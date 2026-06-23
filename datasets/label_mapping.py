"""
Canonical label mappings for all supported datasets.

This is the single source of truth for event-ID → semantic-label conversion.
All preprocessing and training code MUST use these mappings rather than
auto-sorting event IDs, which produces semantically-wrong labels.

Dataset             Raw Event IDs           Canonical Labels
─────────────────────────────────────────────────────────────
physionet_mi        T0→1, T1→2, T2→3        0=rest, 1=left_hand, 2=right_hand
bci_iv_2a           769, 770, 771, 772       0=left_hand, 1=right_hand, 2=feet, 3=tongue
deepbci             (device-dependent)       0=idle, 1=left_hand, 2=right_hand
"""

from typing import Optional

import numpy as np

# ── Semantic label maps (human-readable → canonical int) ──────────────
LABEL_MAPS: dict[str, dict[str, int]] = {
    "physionet_mi": {
        "rest": 0,
        "left_hand": 1,
        "right_hand": 2,
    },
    "bci_iv_2a": {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
        "tongue": 3,
    },
    "deepbci": {
        "idle": 0,
        "left_hand": 1,
        "right_hand": 2,
    },
}

# ── Raw MNE event ID → canonical label ───────────────────────────────
RAW_EVENT_TO_LABEL: dict[str, dict[int, int]] = {
    "physionet_mi": {1: 0, 2: 1, 3: 2},        # T0→rest, T1→left, T2→right
    "bci_iv_2a": {769: 0, 770: 1, 771: 2, 772: 3},
}

# For physionet_mi binary: T1→left, T2→right (no rest)
RAW_EVENT_TO_LABEL_BINARY: dict[str, dict[int, int]] = {
    "physionet_mi": {2: 0, 3: 1},
}

# ── Auto-detection signatures ─────────────────────────────────────────
DATASET_EVENT_SIGNATURES: dict[str, frozenset[int]] = {
    "physionet_mi": frozenset([1, 2, 3]),
    "bci_iv_2a": frozenset([769, 770, 771, 772]),
}


def get_label_map(dataset: str) -> dict[int, int]:
    """
    Return raw-event-ID → canonical-label mapping for *dataset*.

    Raises ValueError for unknown datasets — there is no sorted-remap fallback.
    """
    if dataset not in RAW_EVENT_TO_LABEL:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {list(RAW_EVENT_TO_LABEL.keys())}. "
            f"Use --dataset to specify one of these."
        )
    return dict(RAW_EVENT_TO_LABEL[dataset])


def get_label_map_binary(dataset: str) -> dict[int, int]:
    """Return binary raw-event-ID → canonical-label mapping (left=0, right=1)."""
    if dataset not in RAW_EVENT_TO_LABEL_BINARY:
        raise ValueError(
            f"No binary mapping defined for dataset '{dataset}'. "
            f"Supported: {list(RAW_EVENT_TO_LABEL_BINARY.keys())}."
        )
    return dict(RAW_EVENT_TO_LABEL_BINARY[dataset])


def get_semantic_labels(dataset: str) -> dict[str, int]:
    """Return semantic-name → canonical-label mapping."""
    if dataset not in LABEL_MAPS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: {list(LABEL_MAPS.keys())}."
        )
    return dict(LABEL_MAPS[dataset])


def list_datasets() -> list[str]:
    """Return list of supported dataset names."""
    return sorted(LABEL_MAPS.keys())


def validate_labels(y: np.ndarray, dataset: str) -> bool:
    """
    Check that all labels in *y* are valid for *dataset*.

    Returns True iff every value in y is within the expected label range.
    """
    if dataset not in LABEL_MAPS:
        return False
    n_classes = len(LABEL_MAPS[dataset])
    return bool(np.all((y >= 0) & (y < n_classes)))


def class_names(dataset: str) -> list[str]:
    """Return ordered class names for *dataset* (index = canonical label)."""
    if dataset not in LABEL_MAPS:
        raise ValueError(f"Unknown dataset '{dataset}'.")
    # Sort by canonical label value
    pairs = sorted(LABEL_MAPS[dataset].items(), key=lambda kv: kv[1])
    return [name for name, _ in pairs]


def auto_detect_dataset(unique_event_ids: frozenset[int]) -> Optional[str]:
    """
    Auto-detect dataset from a set of raw event IDs.

    Returns the dataset name string, or None if no signature matches.
    """
    for name, signature in DATASET_EVENT_SIGNATURES.items():
        if unique_event_ids.issubset(signature):
            return name
    return None
