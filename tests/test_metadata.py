"""Tests for datasets/metadata.py — dataset metadata export."""
import json
import pytest

from datasets.metadata import (
    DATASET_META,
    get_dataset_meta,
    export_metadata,
)


class TestConstants:
    def test_physionet_meta(self):
        meta = DATASET_META["physionet_mi"]
        assert meta["n_subjects"] == 30
        assert meta["n_channels_available"] == 64
        assert meta["sfreq"] == 160
        assert meta["paradigm"] == "motor_imagery"
        assert len(meta["classes"]) == 3

    def test_bci_iv_2a_meta(self):
        meta = DATASET_META["bci_iv_2a"]
        assert meta["n_subjects"] == 9
        assert meta["n_channels_available"] == 22
        assert meta["sfreq"] == 250
        assert len(meta["classes"]) == 4

    def test_deepbci_meta(self):
        meta = DATASET_META["deepbci"]
        assert meta["n_channels_available"] == 8
        assert meta["sfreq"] == 250
        assert meta["paradigm"] == "motor_imagery"


class TestGetDatasetMeta:
    def test_physionet_mi(self):
        meta = get_dataset_meta("physionet_mi")
        assert meta["name"] == "PhysioNet Motor Imagery (eegbci)"
        assert meta["n_subjects"] == 30

    def test_bci_iv_2a(self):
        meta = get_dataset_meta("bci_iv_2a")
        assert meta["n_subjects"] == 9

    def test_deepbci(self):
        meta = get_dataset_meta("deepbci")
        assert meta["name"] == "DeepBCI — Self-collected"

    def test_returns_copy(self):
        m1 = get_dataset_meta("physionet_mi")
        m2 = get_dataset_meta("physionet_mi")
        assert m1 is not m2
        assert m1 == m2

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_meta("nonexistent")


class TestExportMetadata:
    def test_creates_json_file(self, tmp_path):
        path = tmp_path / "metadata.json"
        export_metadata(str(path))
        assert path.exists()

    def test_valid_json(self, tmp_path):
        path = tmp_path / "metadata.json"
        export_metadata(str(path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "physionet_mi" in data
        assert "bci_iv_2a" in data
        assert "deepbci" in data

    def test_utf8_encoding(self, tmp_path):
        path = tmp_path / "metadata.json"
        export_metadata(str(path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # DeepBCI has em-dash in name
        assert "—" in data["deepbci"]["name"]

    def test_all_keys_preserved(self, tmp_path):
        path = tmp_path / "metadata.json"
        export_metadata(str(path))
        with open(path, encoding="utf-8") as f:
            exported = json.load(f)
        for ds_name in DATASET_META:
            assert ds_name in exported
            for key in DATASET_META[ds_name]:
                assert key in exported[ds_name]

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "subdir" / "nested" / "metadata.json"
        export_metadata(str(path))
        assert path.exists()

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "metadata.json"
        path.write_text("old content")
        export_metadata(str(path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "physionet_mi" in data
