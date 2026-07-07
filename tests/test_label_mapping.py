"""Tests for datasets/label_mapping.py — canonical label mappings."""
import numpy as np
import pytest

from datasets.label_mapping import (
    LABEL_MAPS,
    RAW_EVENT_TO_LABEL,
    RAW_EVENT_TO_LABEL_BINARY,
    DATASET_EVENT_SIGNATURES,
    get_label_map,
    get_label_map_binary,
    get_semantic_labels,
    list_datasets,
    validate_labels,
    class_names,
    auto_detect_dataset,
)


# ── Constants ───────────────────────────────────────────────────────────────

class TestConstants:
    def test_physionet_mi_in_maps(self):
        assert "physionet_mi" in LABEL_MAPS
        assert "physionet_mi" in RAW_EVENT_TO_LABEL
        assert "physionet_mi" in RAW_EVENT_TO_LABEL_BINARY

    def test_bci_iv_2a_in_maps(self):
        assert "bci_iv_2a" in LABEL_MAPS
        assert "bci_iv_2a" in RAW_EVENT_TO_LABEL

    def test_deepbci_in_labmaps(self):
        assert "deepbci" in LABEL_MAPS

    def test_physionet_mi_semantic_labels(self):
        labels = LABEL_MAPS["physionet_mi"]
        assert labels["rest"] == 0
        assert labels["left_hand"] == 1
        assert labels["right_hand"] == 2

    def test_physionet_mi_raw_to_label(self):
        mapping = RAW_EVENT_TO_LABEL["physionet_mi"]
        assert mapping[1] == 0  # T0 → rest
        assert mapping[2] == 1  # T1 → left
        assert mapping[3] == 2  # T2 → right

    def test_physionet_mi_binary_mapping(self):
        mapping = RAW_EVENT_TO_LABEL_BINARY["physionet_mi"]
        assert mapping[2] == 0  # T1 → left
        assert mapping[3] == 1  # T2 → right
        assert 1 not in mapping   # T0 (rest) absent in binary

    def test_bci_iv_2a_raw_to_label(self):
        mapping = RAW_EVENT_TO_LABEL["bci_iv_2a"]
        assert mapping[769] == 0  # left
        assert mapping[770] == 1  # right
        assert mapping[771] == 2  # feet
        assert mapping[772] == 3  # tongue

    def test_event_signatures(self):
        assert DATASET_EVENT_SIGNATURES["physionet_mi"] == frozenset([1, 2, 3])
        assert DATASET_EVENT_SIGNATURES["bci_iv_2a"] == frozenset([769, 770, 771, 772])


# ── get_label_map ───────────────────────────────────────────────────────────

class TestGetLabelMap:
    def test_physionet_mi(self):
        m = get_label_map("physionet_mi")
        assert m[1] == 0
        assert m[2] == 1
        assert m[3] == 2

    def test_bci_iv_2a(self):
        m = get_label_map("bci_iv_2a")
        assert m[769] == 0
        assert m[770] == 1

    def test_returns_copy(self):
        m1 = get_label_map("physionet_mi")
        m2 = get_label_map("physionet_mi")
        assert m1 is not m2  # should be independent copies

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_label_map("unknown_dataset")


# ── get_label_map_binary ────────────────────────────────────────────────────

class TestGetLabelMapBinary:
    def test_physionet_mi(self):
        m = get_label_map_binary("physionet_mi")
        assert m[2] == 0
        assert m[3] == 1
        assert 1 not in m

    def test_returns_copy(self):
        m1 = get_label_map_binary("physionet_mi")
        m2 = get_label_map_binary("physionet_mi")
        assert m1 is not m2

    def test_no_binary_mapping_raises(self):
        with pytest.raises(ValueError, match="No binary mapping"):
            get_label_map_binary("bci_iv_2a")


# ── get_semantic_labels ─────────────────────────────────────────────────────

class TestGetSemanticLabels:
    def test_physionet_mi(self):
        labels = get_semantic_labels("physionet_mi")
        assert labels["rest"] == 0
        assert labels["left_hand"] == 1
        assert labels["right_hand"] == 2

    def test_bci_iv_2a(self):
        labels = get_semantic_labels("bci_iv_2a")
        assert labels["left_hand"] == 0
        assert labels["feet"] == 2
        assert labels["tongue"] == 3

    def test_deepbci(self):
        labels = get_semantic_labels("deepbci")
        assert labels["idle"] == 0
        assert labels["left_hand"] == 1
        assert labels["right_hand"] == 2

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_semantic_labels("fake")


# ── list_datasets ───────────────────────────────────────────────────────────

class TestListDatasets:
    def test_returns_sorted_list(self):
        ds = list_datasets()
        assert isinstance(ds, list)
        assert ds == sorted(ds)
        assert "physionet_mi" in ds
        assert "bci_iv_2a" in ds
        assert "deepbci" in ds


# ── validate_labels ─────────────────────────────────────────────────────────

class TestValidateLabels:
    def test_valid_physionet(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        assert validate_labels(y, "physionet_mi") is True

    def test_valid_bci_iv_2a(self):
        y = np.array([0, 1, 2, 3])
        assert validate_labels(y, "bci_iv_2a") is True

    def test_invalid_label(self):
        y = np.array([0, 1, 3])  # 3 not valid for physionet_mi
        assert validate_labels(y, "physionet_mi") is False

    def test_negative_label(self):
        y = np.array([0, -1, 2])
        assert validate_labels(y, "physionet_mi") is False

    def test_unknown_dataset(self):
        assert validate_labels(np.array([0, 1]), "fake") is False

    def test_empty_array(self):
        assert validate_labels(np.array([]), "physionet_mi") is True


# ── class_names ─────────────────────────────────────────────────────────────

class TestClassNames:
    def test_physionet_mi(self):
        names = class_names("physionet_mi")
        assert names == ["rest", "left_hand", "right_hand"]

    def test_bci_iv_2a(self):
        names = class_names("bci_iv_2a")
        assert names == ["left_hand", "right_hand", "feet", "tongue"]

    def test_deepbci(self):
        names = class_names("deepbci")
        assert names == ["idle", "left_hand", "right_hand"]

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            class_names("nonexistent")


# ── auto_detect_dataset ─────────────────────────────────────────────────────

class TestAutoDetectDataset:
    def test_detect_physionet(self):
        result = auto_detect_dataset(frozenset([1, 2, 3]))
        assert result == "physionet_mi"

    def test_detect_physionet_subset(self):
        """Should match even if only a subset of events is present."""
        result = auto_detect_dataset(frozenset([1, 2]))
        assert result == "physionet_mi"

    def test_detect_bci_iv_2a(self):
        result = auto_detect_dataset(frozenset([769, 770, 771, 772]))
        assert result == "bci_iv_2a"

    def test_no_match(self):
        result = auto_detect_dataset(frozenset([999]))
        assert result is None

    def test_empty_set(self):
        """Empty set is a subset of all signatures — returns first match (physionet_mi)."""
        result = auto_detect_dataset(frozenset())
        assert result == "physionet_mi"  # empty is subset of everything → first match
