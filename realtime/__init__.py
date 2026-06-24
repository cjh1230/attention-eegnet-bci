"""Real-time inference pipeline: sources, buffer, inference, recording."""

from realtime.sources import EEGSource
from realtime.file_replay import FileReplaySource
from realtime.stream import DummyStream
from realtime.buffer import RingBuffer
from realtime.inference import MIInference
from realtime.deepbci_recorder import DeepBCIRecorder
from realtime.deepbci_protocol import MIProtocol
from realtime.deepbci_source import DeepBCISource

__all__ = [
    "EEGSource",
    "FileReplaySource",
    "DummyStream",
    "RingBuffer",
    "MIInference",
    "DeepBCIRecorder",
    "MIProtocol",
    "DeepBCISource",
]
