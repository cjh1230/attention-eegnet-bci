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
        evidence_depth: int = 1,
        use_filter_bank: bool = False,
        n_bands: int = 6,
        use_band_gate: bool = True,  # only used with filter_bank
        gate_mode: str = "sigmoid",  # sigmoid | softmax | residual | temperature
        gate_tau: float = 1.0,       # temperature for softmax gate
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
        self.evidence_depth = evidence_depth
        self.use_filter_bank = use_filter_bank
        self.n_bands = n_bands
        self.use_band_gate = use_band_gate
        self.gate_mode = gate_mode
        self.gate_tau = gate_tau
        self.dropout_rate = dropout

        # Signal training scripts to apply apply_filter_bank()
        self.input_requires_filter_bank = use_filter_bank

        # ── Band gate (only with filter bank) ──
        if use_filter_bank and use_band_gate and gate_mode in ("sigmoid", "residual", "softmax", "temperature"):
            self.band_gate_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

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
        if self.evidence_depth == 1:
            self.evidence_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_features, self.hidden_dim),
            )
        elif self.evidence_depth == 2:
            mid_dim = (n_features + self.hidden_dim) // 2
            self.evidence_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_features, mid_dim),
                nn.BatchNorm1d(mid_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(mid_dim, self.hidden_dim),
            )
        else:
            raise ValueError(f"evidence_depth must be 1 or 2, got {self.evidence_depth}")
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
            With use_filter_bank: (B, n_bands, C, T).
        return_all_steps : bool
            If True, always return list of all step logits (even in eval mode).

        Returns
        -------
        training mode or return_all_steps : list of torch.Tensor
            [logits_1, ..., logits_K], each (B, n_classes).
        eval mode (default) : torch.Tensor
            logits_K, (B, n_classes) — final prediction only.
        """
        # ── Filter bank path ─────────────────────────────────────────
        if self.use_filter_bank:
            B, nb, C, T_in = x.shape
            # Shared encoder per band
            x = x.reshape(B * nb, C, T_in)
            x = x.unsqueeze(1)          # (B*nb, 1, C, T)
            x = self.conv1(x)
            x = self.depthwise(x)
            x = self.bn_depth(x)
            x = self.act1(x)
            x = self.pool1(x)
            x = self.drop1(x)
            x = self.separable(x)       # (B*nb, F2, 1, T//32)

            # Lazy evidence projection
            if self.evidence_proj is None:
                n_features = x.reshape(x.size(0), -1).size(1)
                self._build_evidence_proj(n_features, x.device)

            # Per-band evidence
            evidence_per_band = self.evidence_proj(x)  # (B*nb, hidden_dim)
            evidence_per_band = evidence_per_band.reshape(B, nb, self.hidden_dim)  # (B, nb, hidden_dim)

            # Band fusion: gate or mean pooling
            if self.use_band_gate and self.gate_mode != "mean":
                gate_raw = self.band_gate_proj(
                    evidence_per_band.reshape(B * nb, self.hidden_dim)
                ).reshape(B, nb, 1)  # (B, nb, 1)

                if self.gate_mode == "sigmoid":
                    gate = torch.sigmoid(gate_raw)
                elif self.gate_mode == "softmax":
                    gate = torch.softmax(gate_raw / self.gate_tau, dim=1)  # compete across bands
                elif self.gate_mode == "temperature":
                    gate = torch.softmax(gate_raw / self.gate_tau, dim=1)
                elif self.gate_mode == "residual":
                    gate = torch.sigmoid(gate_raw)
                    evidence = evidence_per_band.mean(dim=1) + (gate * evidence_per_band).sum(dim=1)
                    # Skip the standard weighted sum below
                    h = evidence
                    logits_list = []
                    for _ in range(self.steps):
                        h = self.gru_cell(evidence, h)
                        logits_t = self.step_classifier(h)
                        logits_list.append(logits_t)
                    if self.training or return_all_steps:
                        return logits_list
                    return logits_list[-1]
                else:
                    gate = torch.sigmoid(gate_raw)  # fallback

                evidence = (gate * evidence_per_band).sum(dim=1)
            else:
                evidence = evidence_per_band.mean(dim=1)  # mean pool across bands

            # Multi-step reasoning
            logits_list = []
            h = evidence
            for _ in range(self.steps):
                h = self.gru_cell(evidence, h)
                logits_t = self.step_classifier(h)
                logits_list.append(logits_t)

            if self.training or return_all_steps:
                return logits_list
            return logits_list[-1]

        # ── Standard path (raw EEG) ──────────────────────────────────
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.bn_depth(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x)

        if self.evidence_proj is None:
            n_features = x.reshape(x.size(0), -1).size(1)
            self._build_evidence_proj(n_features, x.device)

        evidence = self.evidence_proj(x)

        logits_list = []
        h = evidence
        for _ in range(self.steps):
            h = self.gru_cell(evidence, h)
            logits_t = self.step_classifier(h)
            logits_list.append(logits_t)

        if self.training or return_all_steps:
            return logits_list
        else:
            return logits_list[-1]
