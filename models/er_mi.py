"""
ER-MI: Evidence Reasoning Network for Motor Imagery.

Architecture (ER-MI-Lite):
    EEG → EEGNet Encoder (Block1 + Block2)
        → Evidence Vector (Linear projection)
        → GRUCell Reasoning × K steps
        → Step-wise Classifier
        → [logits_1, ..., logits_K] (train) / logits_K (eval)

Core idea: MI decoding as multi-step evidence accumulation rather than
one-shot classification. The model revisits the same evidence vector K times,
refining its hidden state via GRUCell before producing the final prediction.

Reference: Project original (XH-202610).
"""

import torch
import torch.nn as nn


class ERMI(nn.Module):
    """
    Evidence Reasoning Network for Motor Imagery (Lite version).

    Parameters
    ----------
    n_channels : int
        Number of EEG channels. Default 8 (motor8 montage).
    n_classes : int
        Number of output classes. Default 2 (binary MI).
    F1 : int
        Number of temporal filters (EEGNet Block1).
    D : int
        Depth multiplier for spatial filters.
    F2 : int
        Number of pointwise filters (EEGNet Block2).
    hidden_dim : int
        Dimension of evidence vector and GRU hidden state.
    steps : int
        Number of reasoning steps (K). Default 3.
    dropout : float
        Dropout rate applied after each pooling layer.
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        hidden_dim: int = 64,
        steps: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.hidden_dim = hidden_dim
        self.steps = steps

        # ── Block 1: Temporal conv → Spatial depthwise ──────────────
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Conv2d(
            F1, D * F1, kernel_size=(n_channels, 1), groups=F1, bias=False,
        )
        self.bn_depth = nn.BatchNorm2d(D * F1)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # ── Block 2: Separable conv ─────────────────────────────────
        self.separable = nn.Sequential(
            nn.Conv2d(
                D * F1, D * F1,
                kernel_size=(1, 16), padding=(0, 8), groups=D * F1, bias=False,
            ),
            nn.Conv2d(D * F1, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # ── Evidence projection (lazy — built on first forward) ─────
        self.evidence_proj = None

        # ── Reasoning cell ──────────────────────────────────────────
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

        # ── Step classifier ─────────────────────────────────────────
        self.step_classifier = nn.Linear(hidden_dim, n_classes)

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _build_evidence_proj(self, n_features: int, device: torch.device):
        """Create evidence projection after flatten dim is known."""
        self.evidence_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, self.hidden_dim),
        )
        self.evidence_proj.to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_all_steps: bool = False):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, C, T) or (B, 1, C, T).
        return_all_steps : bool
            If True, always return list of all step logits (even in eval mode).

        Returns
        -------
        training mode or return_all_steps : list of torch.Tensor
            [logits_1, ..., logits_K], each (B, n_classes).
        eval mode (default) : torch.Tensor
            logits_K, (B, n_classes) — final prediction only.
        """
        # Normalise shape → (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # ── Block 1 ─────────────────────────────────────────────────
        x = self.conv1(x)           # (B, F1, C, T)
        x = self.depthwise(x)       # (B, D*F1, 1, T)
        x = self.bn_depth(x)
        x = self.act1(x)
        x = self.pool1(x)           # (B, D*F1, 1, T//4)
        x = self.drop1(x)

        # ── Block 2 ─────────────────────────────────────────────────
        x = self.separable(x)       # (B, F2, 1, T//32)

        # ── Lazy evidence projection ────────────────────────────────
        if self.evidence_proj is None:
            n_features = x.reshape(x.size(0), -1).size(1)
            self._build_evidence_proj(n_features, x.device)

        evidence = self.evidence_proj(x)   # (B, hidden_dim)

        # ── Multi-step reasoning ────────────────────────────────────
        logits_list = []
        h = evidence                        # h₀ = evidence

        for _ in range(self.steps):
            h = self.gru_cell(evidence, h)  # input = evidence (fixed), hidden evolves
            logits_t = self.step_classifier(h)
            logits_list.append(logits_t)

        if self.training or return_all_steps:
            return logits_list
        else:
            return logits_list[-1]
