"""
Dataset registry — canonical label mapping, metadata, and validation.

Usage:
    from datasets import get_label_map, get_dataset_meta, list_datasets
    from datasets import LABEL_MAPS, DATASET_META, validate_labels
"""

from datasets.label_mapping import (
    LABEL_MAPS,
    RAW_EVENT_TO_LABEL,
    DATASET_EVENT_SIGNATURES,
    get_label_map,
    get_semantic_labels,
    list_datasets,
    validate_labels,
    class_names,
)

from datasets.metadata import (
    DATASET_META,
    get_dataset_meta,
    export_metadata,
)

__all__ = [
    "LABEL_MAPS",
    "RAW_EVENT_TO_LABEL",
    "DATASET_EVENT_SIGNATURES",
    "get_label_map",
    "get_semantic_labels",
    "list_datasets",
    "validate_labels",
    "class_names",
    "DATASET_META",
    "get_dataset_meta",
    "export_metadata",
]
