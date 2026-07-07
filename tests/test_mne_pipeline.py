"""Tests for preprocessing/mne_pipeline.py — full MNE pipeline programmatic API."""
import os

import numpy as np
import pytest

from preprocessing.mne_pipeline import run_pipeline


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_edf(tmp_path):
    """Create a minimal synthetic EDF file for testing the pipeline."""
    import mne
    from mne import create_info
    from mne.io import RawArray

    # 8-channel, 250 Hz, 10 seconds of noise
    n_channels = 8
    sfreq = 250
    duration = 10
    n_samples = sfreq * duration

    rng = np.random.RandomState(42)
    data = rng.randn(n_channels, n_samples).astype(np.float64)

    # Use standard 10-20 channel names
    ch_names = ["FC3", "C3", "Cz", "C4", "FC4", "CP3", "CPz", "CP4"]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = RawArray(data, info)

    path = tmp_path / "test_raw.fif"
    raw.save(path, overwrite=True)
    return str(path)


@pytest.fixture
def events_array():
    """MNE events array: (n_events, 3). 9 events spaced 200 samples apart.

    Event codes 0, 1, 2 match EVENT_IDS = {rest:0, left_hand:1, right_hand:2}.
    Epoch window is [-0.5, 2.5]s = 750 samples. Data is 10s = 2500 samples.
    Onsets: 150..1750 → last epoch ends at 1750+625=2375 < 2500.  ✓
    """
    n_events = 9
    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = np.arange(n_events) * 200 + 150
    events[:, 1] = 0
    events[:, 2] = np.tile([0, 1, 2], n_events // 3 + 1)[:n_events]
    return events


# ── Tests ───────────────────────────────────────────────────────────────────

class TestRunPipeline:
    def test_basic_run(self, sample_edf, events_array):
        """Pipeline should run without error and return correct shapes."""
        X, y = run_pipeline(sample_edf, events_array)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 3  # (n_trials, n_channels, n_times)
        assert X.shape[0] == len(events_array)
        assert X.shape[1] == 8  # 8 channels
        assert y.shape == (len(events_array),)

    def test_no_ica(self, sample_edf, events_array):
        """With ICA disabled, should still work."""
        X, y = run_pipeline(sample_edf, events_array, apply_ica_flag=False)
        assert X.ndim == 3
        assert X.shape[0] > 0

    def test_with_ica(self, sample_edf, events_array):
        """With ICA enabled on synthetic data, may fail gracefully or succeed
        with warnings. Either outcome is acceptable — synthetic noise is not
        a valid ICA target."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                X, y = run_pipeline(sample_edf, events_array, apply_ica_flag=True)
                # If it succeeds, basic sanity checks
                assert X.ndim == 3
                assert X.shape[0] > 0
            except (ValueError, RuntimeError) as e:
                # ICA on synthetic noise may legitimately fail
                err_msg = str(e).lower()
                assert any(kw in err_msg for kw in ("ica", "rank", "covariance"))

    def test_dtype_float(self, sample_edf, events_array):
        """Output X should be a floating-point type."""
        X, _ = run_pipeline(sample_edf, events_array)
        assert np.issubdtype(X.dtype, np.floating)

    def test_y_is_integer(self, sample_edf, events_array):
        """Labels should be integer type."""
        _, y = run_pipeline(sample_edf, events_array)
        assert np.issubdtype(y.dtype, np.integer)

    def test_no_nan_in_output(self, sample_edf, events_array):
        X, y = run_pipeline(sample_edf, events_array)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_all_values_finite(self, sample_edf, events_array):
        X, _ = run_pipeline(sample_edf, events_array)
        assert np.isfinite(X).all()

    def test_reproducible(self, sample_edf, events_array):
        """Same input should produce same output (pipeline is deterministic)."""
        X1, y1 = run_pipeline(sample_edf, events_array)
        X2, y2 = run_pipeline(sample_edf, events_array)
        assert np.allclose(X1, X2)
        assert np.array_equal(y1, y2)

    def test_events_have_labels(self, sample_edf, events_array):
        """Output labels should match the events' label column."""
        _, y = run_pipeline(sample_edf, events_array)
        expected_labels = events_array[:, 2]
        assert np.array_equal(y, expected_labels)

    def test_n_trials_matches_n_events(self, sample_edf, events_array):
        X, y = run_pipeline(sample_edf, events_array)
        assert X.shape[0] == len(events_array)
        assert y.shape[0] == len(events_array)

    def test_time_dimension_positive(self, sample_edf, events_array):
        X, _ = run_pipeline(sample_edf, events_array)
        assert X.shape[2] > 0  # time dimension should be non-zero
