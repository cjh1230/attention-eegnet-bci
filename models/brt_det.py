"""
BRT-Det: Band-Region-Time Evidence Detector for MI-BCI.

Reframes MI-EEG decoding from "whole-trial classification" to
"band-region-time evidence detection":

    EEG trial → Filter Bank → [Region Pooling] → Time Cell Pooling
    → Evidence Map (B, n_bands, spatial_dim, n_time_cells)
    → Detection Head (objectness + class scores per cell)
    → Evidence Aggregation (weighted sum) → logits

Two variants:
    - Region  (use_region_pool=True):  6 bands × 3 regions × 12 time cells
    - Channel (use_region_pool=False): 6 bands × 8 channels × 12 time cells

Reference:
    XH-202610 BRT-Det design doc (2026-07-04).

Model receives filter-bank input (B, n_bands, C, T) — set
``input_requires_filter_bank = True`` so training scripts call
``apply_filter_bank()`` automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BRTDet(nn.Module):
    """Band-Region-Time Evidence Detector for Motor Imagery classification.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default 8 for motor8 montage).
    n_classes : int
        Number of output classes (default 2 for binary MI).
    n_bands : int
        Number of frequency bands in filter-bank input (default 6).
    n_regions : int
        Number of spatial regions (default 3: FC / C / CP).
        Only used when ``use_region_pool=True``.
    n_time_cells : int
        Number of temporal grid cells (default 12).
    hidden : int
        Hidden channels in the detection backbone (default 32).
    topk : int or None
        If set, only aggregate the top-K evidence cells by objectness.
        None (default) uses all cells.
    use_region_pool : bool
        If True (default), pool channels into 3 regions (FC/C/CP).
        If False, keep all 8 channels — preserves C3/C4 lateralization.
    """

    # Signal to training scripts: apply apply_filter_bank() before feeding data.
    input_requires_filter_bank: bool = True

    # Channel → Region mapping for 8ch motor montage.
    # Channel order: FC3, C3, Cz, C4, FC4, CP3, CPz, CP4
    REGION_INDICES: list[list[int]] = [
        [0, 4],         # FC:  FC3, FC4
        [1, 2, 3],      # C:   C3, Cz, C4
        [5, 6, 7],      # CP:  CP3, CPz, CP4
    ]

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 2,
        n_bands: int = 6,
        n_regions: int = 3,
        n_time_cells: int = 12,
        hidden: int = 32,
        topk: int | None = None,
        use_region_pool: bool = True,
        use_objectness: bool = True,
        multi_scale: bool = False,
        dilations: list[int] | None = None,
        agg_mode: str = "objectness",
        agg_tau: float = 0.5,
        agg_topk: int = 10,
        use_band_mixer: bool = False,
        use_diff_channels: bool = False,
        use_band_gate: bool = False,
        use_temporal_gate: bool = False,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_bands = n_bands
        self.n_regions = n_regions
        self.n_time_cells = n_time_cells
        self.hidden = hidden
        self.topk = topk
        self.use_region_pool = use_region_pool
        self.use_objectness = use_objectness
        self.multi_scale = multi_scale
        self.dilations = dilations or [1, 1, 1]
        self.agg_mode = agg_mode
        self.agg_tau = agg_tau
        self.agg_topk = agg_topk
        self.use_band_mixer = use_band_mixer
        self.use_diff_channels = use_diff_channels
        self.use_band_gate = use_band_gate
        self.use_temporal_gate = use_temporal_gate

        # Spatial dimension after pooling
        self._spatial_dim = n_regions if use_region_pool else n_channels
        # If diff channels enabled, spatial dim increases (8 -> 11)
        if use_diff_channels and not use_region_pool:
            self._spatial_dim = n_channels + 3  # 8 original + 3 diff pairs

        # ── Temporal Stem ──
        # Two-layer temporal conv BEFORE grid pooling — extracts local
        # temporal features from each (band, spatial_pos) independently.
        # Input:  (B*nb*S, 1, T) → Output: (B*nb*S, temporal_hidden, T)
        self.temporal_hidden = temporal_hidden = 24
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(1, temporal_hidden // 2, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(temporal_hidden // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Conv1d(temporal_hidden // 2, temporal_hidden, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(temporal_hidden),
            nn.ELU(),
            nn.Dropout(0.1),
        )

        # ── Spatial Mixing ──
        # Cross-channel conv on the evidence grid.
        # Input:  (B*nb, temporal_hidden, S, n_time_cells)
        # Output: (B*nb, temporal_hidden, S, n_time_cells)
        self.spatial_mix = nn.Sequential(
            nn.Conv2d(temporal_hidden, temporal_hidden, kernel_size=(3, 1),
                      padding=(1, 0), bias=False),
            nn.BatchNorm2d(temporal_hidden),
            nn.ELU(),
            nn.Dropout(0.1),
        )

        # ── Cross-Band Mixer ──
        # Linear mixer across n_bands.  Lets the model learn band
        # interactions (e.g., mu-beta complementarity) instead of
        # treating each band independently.
        if use_band_mixer:
            # Mixer: for each (th, S, T_cells) position, apply Linear(nb→nb)
            # Only 36 weights + 6 biases = 42 learnable params
            self.band_mixer = nn.Sequential(
                nn.Linear(n_bands, n_bands * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(n_bands * 2, n_bands),
            )

        # ── Band Gate ──
        # Per-band scalar gate: learns which frequency bands carry
        # useful MI evidence for the current input.  Preserves filter
        # bank independence — each band is weighted, not mixed.
        if use_band_gate:
            self.band_gate_proj = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        # ── Temporal Gate ──
        # Per-time-cell scalar gate: learns which temporal windows
        # carry MI evidence.  ~200 params.  Applied after backbone,
        # before detection head — weights the evidence grid along the
        # time axis independently for each trial.
        if use_temporal_gate:
            self.temporal_gate_proj = nn.Sequential(
                nn.Linear(hidden, hidden // 4),
                nn.ReLU(),
                nn.Linear(hidden // 4, 1),
            )

        # ── Detection Backbone ──
        # 3-layer 2D conv on the band×channel×time evidence grid.
        # Dilation along time axis expands temporal receptive field
        # without extra params or multi-scale forward passes.
        # Input:  (B*n_bands, temporal_hidden, _spatial_dim, n_time_cells)
        # Output: (B*n_bands, hidden, _spatial_dim, n_time_cells)
        backbone_layers = []
        in_ch = temporal_hidden
        for i, d in enumerate(self.dilations):
            backbone_layers.extend([
                nn.Conv2d(in_ch, hidden,
                          kernel_size=(3, 3),
                          padding=(1, d), dilation=(1, d),
                          bias=False),
                nn.BatchNorm2d(hidden),
                nn.ELU(),
                nn.Dropout(0.2),
            ])
            in_ch = hidden
        self.backbone = nn.Sequential(*backbone_layers)

        # ── Detection Head ──
        # 1×1 conv: hidden → (1 + n_classes) per cell
        #   channel 0:         objectness (raw logit)
        #   channels 1..n_cls: class scores (raw logits)
        self.head = nn.Conv2d(hidden, 1 + n_classes, kernel_size=1)

        # ── Learned aggregation weight (for agg_mode="softmax_weight") ──
        if agg_mode == "softmax_weight":
            self.agg_weight = nn.Sequential(
                nn.Conv2d(hidden, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Conv2d(1, 1, kernel_size=1),
            )

    # ------------------------------------------------------------------
    # Region pooling
    # ------------------------------------------------------------------

    def _region_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool channels into spatial regions by averaging.

        Parameters
        ----------
        x : (B, n_bands, C, T)

        Returns
        -------
        out : (B, n_bands, n_regions, T)
        """
        regions = []
        for idx in self.REGION_INDICES:
            # Mean over channels within each region → (B, n_bands, T)
            regions.append(x[:, :, idx, :].mean(dim=2))
        return torch.stack(regions, dim=2)  # (B, n_bands, n_regions, T)

    # ------------------------------------------------------------------
    # Evidence aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        class_scores: torch.Tensor,   # (B, nb, S, T_cells, n_classes)
        pred: torch.Tensor,            # (B, nb, S, T_cells, 1+n_classes)
        feat: torch.Tensor,            # (B*nb, hidden, S, T_cells)
        B: int, nb: int, S: int,
    ) -> torch.Tensor:
        """Aggregate per-cell class scores into logits."""
        n_cells = class_scores.shape
        mode = self.agg_mode

        if mode == "mean":
            return class_scores.mean(dim=(1, 2, 3))

        if mode == "objectness":
            obj = torch.sigmoid(pred[..., 0])
            li = (obj.unsqueeze(-1) * class_scores).sum(dim=(1, 2, 3))
            return li / (obj.sum(dim=(1, 2, 3)).unsqueeze(-1) + 1e-6)

        if mode == "topk":
            # Flatten all cells, select top-k strongest per class
            flat = class_scores.reshape(B, -1, self.n_classes)  # (B, N_cells, n_cls)
            k = min(self.agg_topk, flat.shape[1])
            top_vals, _ = flat.topk(k, dim=1)  # (B, k, n_cls)
            return top_vals.mean(dim=1)          # (B, n_cls)

        if mode == "logsumexp":
            # Soft-max: between mean and max, controlled by temperature
            tau = self.agg_tau
            # Compute logsumexp per class across all cells
            return tau * torch.logsumexp(
                class_scores / tau, dim=(1, 2, 3),
            )  # (B, n_cls)

        if mode == "softmax_weight":
            # Learned per-cell attention weights
            if not hasattr(self, "agg_weight"):
                raise RuntimeError("agg_mode='softmax_weight' but agg_weight not built")
            w = self.agg_weight(feat)           # (B*nb, 1, S, T_cells)
            w = w.reshape(B, nb, S, -1)         # (B, nb, S, T_cells)
            w = F.softmax(w.reshape(B, -1), dim=1)  # softmax across all cells
            w = w.reshape(B, nb, S, -1)
            return (w.unsqueeze(-1) * class_scores).sum(dim=(1, 2, 3))

        raise ValueError(f"Unknown agg_mode: {mode}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_objectness: bool = False):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, n_bands, C, T)
            Multi-band EEG input (already filter-banked).
        return_objectness : bool
            If True, also return the objectness map (B, nb, S, T_cells)
            for entropy regularization.

        Returns
        -------
        logits : torch.Tensor, shape (B, n_classes)
        objectness : torch.Tensor, shape (B, nb, S, T_cells) — only if
            return_objectness=True
        """
        B, nb, C, T_in = x.shape

        # 0. Diff channels: augment with C3/C4 lateralization features
        if self.use_diff_channels and not self.use_region_pool:
            # Channel order: FC3(0), C3(1), Cz(2), C4(3), FC4(4), CP3(5), CPz(6), CP4(7)
            diff_fc = x[:, :, 0:1, :] - x[:, :, 4:5, :]   # FC3 - FC4
            diff_c  = x[:, :, 1:2, :] - x[:, :, 3:4, :]   # C3 - C4
            diff_cp = x[:, :, 5:6, :] - x[:, :, 7:8, :]   # CP3 - CP4
            x = torch.cat([x, diff_fc, diff_c, diff_cp], dim=2)  # (B, nb, 11, T)

        # 1. Spatial grouping: region pool or keep channels
        if self.use_region_pool:
            x = self._region_pool(x)  # (B, nb, R, T)
        # else: x stays as (B, nb, C, T)

        S = x.shape[2]  # spatial dim (n_regions or n_channels)

        # 2. Temporal stem: extract local temporal features per (band, spatial_pos)
        #    (B, nb, S, T) → (B*nb*S, 1, T) → (B*nb*S, th, T)
        x = x.reshape(B * nb * S, 1, T_in)
        x = self.temporal_stem(x)

        # 3. Reshape: (B*nb*S, th, T) → (B*nb, th, S, T)
        th = self.temporal_hidden
        x = x.reshape(B * nb, th, S, T_in)

        # 4. Time → Time cells
        if self.multi_scale:
            # Multi-scale: pool to 3 temporal resolutions, share backbone,
            # average the logits.
            scales = [self.n_time_cells // 2, self.n_time_cells, self.n_time_cells * 2]
            scale_logits = []
            for n_cells in scales:
                x_pool = F.adaptive_avg_pool2d(x, (S, n_cells))
                x_pool = self.spatial_mix(x_pool)
                if self.use_band_mixer:
                    x_pool = x_pool.reshape(B, nb, self.temporal_hidden, S, n_cells)
                    x_pool = x_pool.permute(0, 2, 3, 4, 1)
                    x_pool = self.band_mixer(x_pool)
                    x_pool = x_pool.permute(0, 4, 1, 2, 3)
                    x_pool = x_pool.reshape(B * nb, self.temporal_hidden, S, n_cells)
                feat = self.backbone(x_pool)
                p = self.head(feat)
                p = p.reshape(B, nb, 1 + self.n_classes, S, n_cells)
                p = p.permute(0, 1, 3, 4, 2)
                class_s = p[..., 1:]
                li = self._aggregate(class_s, p, feat, B, nb, S)
                scale_logits.append(li)
            logits = torch.stack(scale_logits, dim=0).mean(dim=0)
            if return_objectness:
                return logits, None
            return logits

        # Single-scale
        x = F.adaptive_avg_pool2d(x, (S, self.n_time_cells))
        x = self.spatial_mix(x)

        # Cross-band mixer (optional)
        if self.use_band_mixer:
            # (B*nb, th, S, T_cells) → (B, nb, th, S, T_cells)
            x = x.reshape(B, nb, self.temporal_hidden, S, self.n_time_cells)
            # Mix across bands: permute to (B*th*S*T_cells, nb) → Linear → back
            x = x.permute(0, 2, 3, 4, 1)  # (B, th, S, T_cells, nb)
            x = self.band_mixer(x)          # mix across last dim (nb)
            x = x.permute(0, 4, 1, 2, 3)  # (B, nb, th, S, T_cells)
            x = x.reshape(B * nb, self.temporal_hidden, S, self.n_time_cells)

        feat = self.backbone(x)         # (B*nb, hidden, S, T_cells)

        # Temporal gate: per-time-cell scalar weight
        # Pool over bands and spatial dims to get per-time-cell features,
        # then learn which time cells are most informative.
        if self.use_temporal_gate:
            gate_in = feat.reshape(B, nb, self.hidden, S, self.n_time_cells)
            gate_in = gate_in.mean(dim=(1, 3))          # (B, hidden, T_cells)
            gate_in = gate_in.permute(0, 2, 1)           # (B, T_cells, hidden)
            gate_t = torch.sigmoid(self.temporal_gate_proj(gate_in))  # (B, T_cells, 1)
            gate_t = gate_t.squeeze(-1)                   # (B, T_cells)
            gate_t = gate_t.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # (B, 1, 1, 1, T_cells)
            feat = feat.reshape(B, nb, self.hidden, S, self.n_time_cells)
            feat = feat * gate_t                         # broadcast over nb, hidden, S
            feat = feat.reshape(B * nb, self.hidden, S, self.n_time_cells)

        # Band gate: per-band scalar weight (learnable noise suppression)
        if self.use_band_gate:
            # Pool each band's features → gate scalar
            gate_in = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)  # (B*nb, hidden)
            gate_in = gate_in.reshape(B, nb, self.hidden)  # (B, nb, hidden)
            gate = torch.sigmoid(self.band_gate_proj(gate_in))  # (B, nb, 1)
        else:
            gate = None

        pred = self.head(feat)          # (B*nb, 1+n_classes, S, T_cells)
        pred = pred.reshape(B, nb, 1 + self.n_classes, S, self.n_time_cells)
        pred = pred.permute(0, 1, 3, 4, 2)

        class_scores = pred[..., 1:]

        # Apply band gate to class scores before aggregation
        if gate is not None:
            # gate: (B, nb, 1) → (B, nb, 1, 1, 1)
            class_scores = class_scores * gate.reshape(B, nb, 1, 1, 1)

        logits = self._aggregate(class_scores, pred, feat, B, nb, S)

        if return_objectness:
            obj = torch.sigmoid(pred[..., 0])  # (B, nb, S, T_cells)
            return logits, obj
        return logits

    # ------------------------------------------------------------------
    # Evidence map extraction (for visualization / analysis)
    # ------------------------------------------------------------------

    def extract_evidence(
        self, x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extract evidence maps without aggregation (for analysis).

        Parameters
        ----------
        x : (B, n_bands, C, T)

        Returns
        -------
        dict with keys:
            objectness   : (B, n_bands, S, n_time_cells)
            class_scores : (B, n_bands, S, n_time_cells, n_classes)
            evidence     : (B, n_bands, S, n_time_cells, n_classes)
                           = objectness × class_scores (per-class evidence)
            spatial_labels : list[str] — region or channel names
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            B, nb, C, T_in = x.shape

            if self.use_region_pool:
                x_r = self._region_pool(x)
                spatial_labels = ["FC", "C", "CP"]
            else:
                x_r = x
                spatial_labels = [
                    "FC3", "C3", "Cz", "C4", "FC4", "CP3", "CPz", "CP4",
                ]

            S = x_r.shape[2]
            th = self.temporal_hidden

            # Temporal stem
            x_r = x_r.reshape(B * nb * S, 1, T_in)
            x_r = self.temporal_stem(x_r)
            x_r = x_r.reshape(B * nb, th, S, T_in)

            # Grid pooling (mean only — matches forward pass) + spatial mixing + backbone + head
            x_r = F.adaptive_avg_pool2d(x_r, (S, self.n_time_cells))
            x_r = self.spatial_mix(x_r)
            feat = self.backbone(x_r)
            pred = self.head(feat)
            pred = pred.reshape(B, nb, 1 + self.n_classes, S, self.n_time_cells)
            pred = pred.permute(0, 1, 3, 4, 2)

            objectness = torch.sigmoid(pred[..., 0])
            class_scores = pred[..., 1:]
            evidence = objectness.unsqueeze(-1) * class_scores

        if was_training:
            self.train()
        return {
            "objectness": objectness,
            "class_scores": class_scores,
            "evidence": evidence,
            "spatial_labels": spatial_labels,
        }
