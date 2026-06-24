"""Tests for realtime/deepbci_source.py — DeepBCISource placeholder."""

import pytest

from realtime.deepbci_source import DeepBCISource
from realtime.sources import EEGSource


class TestDeepBCISource:
    """DeepBCI hardware placeholder tests."""

    def test_constructor_defaults(self):
        """Constructor uses 8 channels, 250 Hz by default."""
        source = DeepBCISource()
        assert source.n_channels == 8
        assert source.s_freq == 250

    def test_constructor_custom_params(self):
        """Constructor accepts custom n_channels and s_freq."""
        source = DeepBCISource(n_channels=16, s_freq=500)
        assert source.n_channels == 16
        assert source.s_freq == 500

    def test_open_raises_not_implemented(self):
        """open() raises NotImplementedError with a helpful message."""
        source = DeepBCISource()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            source.open()

    def test_read_chunk_raises_not_implemented(self):
        """read_chunk() raises NotImplementedError."""
        source = DeepBCISource()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            source.read_chunk()

    def test_close_raises_not_implemented(self):
        """close() raises NotImplementedError."""
        source = DeepBCISource()
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            source.close()

    def test_satisfies_eegsource_protocol(self):
        """DeepBCISource structurally satisfies EEGSource Protocol."""
        source = DeepBCISource()
        assert isinstance(source, EEGSource)
