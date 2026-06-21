"""
Real-time MI inference loop.

Connects stream → buffer → model → action output.
"""
import numpy as np
import torch

from utils.config import N_CHANNELS, SFREQ


class MIInference:
    """
    Real-time inference wrapper for a trained PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained EEGNet (or similar) model.
    buffer : RingBuffer
        Sliding-window buffer.
    device : str
        Torch device.
    """

    def __init__(self, model: torch.nn.Module, buffer, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.buffer = buffer
        self.device = device

    def predict(self) -> tuple[int, float]:
        """
        Read buffer, run inference, return (class_id, confidence).

        Returns
        -------
        class_id : int
            0=Idle, 1=Left, 2=Right
        confidence : float
            Softmax probability of the predicted class.
        """
        data = self.buffer.read()  # (C, T)
        tensor = torch.from_numpy(data).unsqueeze(0).to(self.device)  # (1, C, T)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)
            class_id = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs.max().item())

        return class_id, confidence
