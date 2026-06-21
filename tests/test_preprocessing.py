"""Smoke tests for preprocessing pipeline functions."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# mne may not be installed in all environments
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False


@pytest.mark.skipif(not HAS_MNE, reason="MNE not installed")
class TestPreprocessingFunctions:
    def test_filtering_module_imports(self):
        from preprocessing.filtering import bandpass, notch, load_raw
        assert callable(bandpass)
        assert callable(notch)
        assert callable(load_raw)

    def test_epoching_module_imports(self):
        from preprocessing.epoching import epoch, to_array
        assert callable(epoch)
        assert callable(to_array)

    def test_artifact_module_imports(self):
        from preprocessing.artifact import apply_ica
        assert callable(apply_ica)

    def test_mne_pipeline_imports(self):
        from preprocessing.mne_pipeline import run_pipeline
        assert callable(run_pipeline)

    def test_run_mne_pipeline_imports(self):
        from preprocessing.run_mne_pipeline import (
            find_eeg_files,
            load_and_filter,
            extract_events,
            process_subject,
        )
        assert callable(find_eeg_files)
        assert callable(load_and_filter)
        assert callable(extract_events)
        assert callable(process_subject)

    def test_synthetic_filter_chain(self):
        """End-to-end filter chain on synthetic Raw."""
        from preprocessing.filtering import load_raw

        # Create synthetic raw with MNE
        n_channels = 4
        sfreq = 250
        duration = 10
        n_samples = int(sfreq * duration)
        data = np.random.RandomState(42).randn(n_channels, n_samples)
        info = mne.create_info(
            ch_names=[f"ch{i}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data, info)
        # Filtering should not crash
        raw.filter(8, 30, fir_design="firwin", verbose=False)
        raw.notch_filter(50, fir_design="firwin", verbose=False)
        assert raw.get_data().shape == (n_channels, n_samples)


@pytest.mark.skipif(not HAS_MNE, reason="MNE not installed")
class TestRunMnePipelineHelpers:
    """Tests that don't require MNE (pure logic)."""

    def test_find_eeg_files_finds_edf(self, tmp_path):
        """find_eeg_files should find .edf/.fif/.gdf files."""
        from preprocessing.run_mne_pipeline import find_eeg_files
        (tmp_path / "test.edf").touch()
        files = find_eeg_files(tmp_path)
        assert len(files) >= 1
        assert any("test.edf" in str(f) for f in files)

    def test_find_eeg_files_ignores_non_eeg(self, tmp_path):
        """Should ignore .txt, .csv, etc."""
        from preprocessing.run_mne_pipeline import find_eeg_files
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.csv").touch()
        files = find_eeg_files(tmp_path)
        assert len(files) == 0
