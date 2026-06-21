"""Sanity checks on global config constants."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import config


class TestConfigSanity:
    def test_sfreq_positive(self):
        assert config.SFREQ > 0

    def test_n_channels_positive(self):
        assert config.N_CHANNELS > 0

    def test_epoch_window_valid(self):
        assert config.T_MAX > config.T_MIN

    def test_buffer_window_positive(self):
        assert config.BUFFER_WINDOW > 0

    def test_batch_size_positive(self):
        assert config.BATCH_SIZE > 0

    def test_learning_rate_in_range(self):
        assert 0 < config.LEARNING_RATE < 1

    def test_motor_channels_not_empty(self):
        assert len(config.MOTOR_CHANNELS) > 0

    def test_freq_bands_have_three_bands(self):
        assert "mu" in config.FREQ_BANDS
        assert "beta" in config.FREQ_BANDS
        assert "full" in config.FREQ_BANDS

    def test_band_ranges_valid(self):
        for name, (lo, hi) in config.FREQ_BANDS.items():
            assert lo > 0
            assert hi > lo

    def test_event_ids_three_classes(self):
        assert len(config.EVENT_IDS) == 3

    def test_predict_interval_positive(self):
        assert config.PREDICT_INTERVAL > 0
