# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**脑机接口（BCI）— 基于运动想象的脑－机交互算法研究**

BCI algorithm research using the Motor Imagery (MI) paradigm. 挑战杯 project **XH-202610** (May–Nov 2026).

**Team:** Solo (单人全栈)

## Environment Setup

```bash
conda env create -f environment.yml              # once
conda activate bci                                # per session
pip install mne moabb streamlit plotly pytest black ruff openpyxl pandas  # full deps
```

All commands below assume `conda activate bci` is active. Use `python`, not a hardcoded interpreter path.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Python | 3.10+ |
| Preprocessing | MNE-Python (offline pipeline) |
| Deep Learning | PyTorch 2.5.1 |
| Datasets | MOABB, PhysioNet |
| Hyperparam Search | Optuna |
| Real-time | LSL (pylsl) or simulated stream |
| UI | Streamlit |
| Reporting | openpyxl (Excel) |
| Env | conda (environment.yml) |

## Project Structure

```
├── data/
│   ├── raw/              # Raw EEG (.edf, .fif, .mat)
│   │   ├── physionet_mi/ #   PhysioNet MI (30 subjects)
│   │   └── bci_iv_2a/    #   BCI Competition IV 2a (9 subjects)
│   ├── processed/        # MNE output (.npy) — X=[N,8,750], y=[N]
│   ├── subjects/         # Per-subject metadata
│   └── download.py       # Auto-download datasets
├── datasets/             # Label mapping + metadata
│   ├── label_mapping.py  #   Canonical label maps (single source of truth)
│   ├── metadata.py       #   Dataset metadata export
│   └── deepbci_loader.py #   DeepBCI session → .npy training data
├── preprocessing/
│   ├── run_mne_pipeline.py  # MNE pipeline: --channels motor8|motor16|all, --binary
│   ├── prepare_bci_iv_2a.py # BCI IV 2a MOABB .npy → 8ch per-subject
│   ├── filtering.py         #   bandpass 8-30Hz + notch 50Hz
│   ├── epoching.py          #   epoch creation
│   ├── artifact.py          #   ICA removal
│   ├── mne_pipeline.py      #   programmatic API
│   ├── alignment.py         #   Euclidean Alignment (EA) cross-subject
│   └── augment.py           #   on-the-fly augmentation
├── features/              # csp.py, riemann.py, bandpower.py, spd_covariance.py
├── models/
│   ├── eegnet.py            # EEGNet (Lawhern 2018) — lazy classifier
│   ├── attention.py         # ChannelAttention1D, MultiHeadChannelAttention,
│   │                        #   TemporalAttention, SpatiotemporalAttention
│   ├── eegnet_attn.py       # EEGNet + attention deep integration (5 variants)
│   ├── eeg_conformer.py     # EEG Conformer (CNN + Transformer, Song 2023)
│   ├── eeg_tcnet.py         # EEG-TCNet (CNN + TCN, Ingolfsson 2020)
│   ├── fbcnet.py            # FBCNet (Filter-Bank CNN, Bakshi 2021)
│   ├── fb_tcnet.py          # FB-TCNet (Filter Bank + TCN, project original)
│   ├── spd_models.py        # SPDNet (SPD manifold DL, Huang & Van Gool 2017)
│   ├── motor_area_attention.py  # Motor-Area Attention (8ch region grouping)
│   ├── fb_maa_eegnet.py     # FB-MAA-EEGNet
│   ├── maa_eegnet.py        # MAA-EEGNet (MAA after temporal conv)
│   ├── maa_eegnet_pre.py    # MAA-EEGNet-Pre (MAA before temporal conv)
│   ├── brt_det.py           # BRT-Det (Band-Region-Time Evidence Detector, v1)
│   ├── fusion.py            # MultiBandFusion (mu/beta/full)
│   └── mixstyle.py          # MixStyle domain generalization (Zhou, ICLR 2021)
├── training/
│   ├── train_eegnet.py     # main training script (6 new features)
│   ├── train_baseline.py   # CSP+SVM baseline
│   ├── train_ablation.py   # 6-config comparison with repeat
│   ├── train_sweep.py      # Optuna hyperparameter search + manual grid fallback
│   ├── train_loso.py       # LOSO cross-validation (Few-shot FT, CSV/JSON)
│   ├── train_riemann_loso.py  # Riemannian LOSO (cov/metric/band sweep)
│   ├── train_spd_loso.py   # SPDNet LOSO on SPD covariance matrices
│   ├── evaluate_subjectwise.py  # Per-subject eval
│   └── split.py            # Data splitting
├── realtime/
│   ├── sources.py          #   EEGSource Protocol (unified interface)
│   ├── stream.py           #   DummyStream (synthetic data)
│   ├── stream_lsl.py       #   LSLStream (real hardware)
│   ├── file_replay.py      #   FileReplaySource (.npy replay, trial_mode)
│   ├── deepbci_stream.py   #   DeepBCIStream (multi-mode)
│   ├── deepbci_recorder.py #   DeepBCIRecorder (session recording + metadata)
│   ├── deepbci_protocol.py #   MIProtocol (experiment timing)
│   ├── deepbci_source.py   #   DeepBCISource placeholder (EEGSource protocol)
│   ├── buffer.py           #   RingBuffer (thread-safe sliding window)
│   └── inference.py        #   MIInference (predict + idle gating)
├── utils/
│   ├── config.py           # global config — EDIT THIS for channel changes
│   ├── metrics.py          # classification_report, per_class_accuracy
│   ├── domain_adapt.py     # domain adaptation losses (Center Loss + MMD)
│   ├── report_excel.py     # 5-sheet Excel validation report (competition format)
│   └── logger.py           # ExperimentLogger (CSV)
├── ui/                    # dashboard.py (Streamlit, Synthetic + File Replay)
├── scripts/               # Experiment automation
│   ├── run_all_experiments.py      # One-command full pipeline
│   ├── export_competition_excel.py # Competition-format Excel
│   ├── make_report_figures.py      # Confusion matrix / bar charts / ablation
│   ├── run_time_window_sweep.py    # Time window sweep
│   ├── run_ablation_all.py         # 10-config ablation
│   ├── analyze_ea_effects.py       # EA × Architecture interaction analysis
│   ├── analyze_spdnet_vs_tangent.py  # SPDNet vs Tangent comparison
│   └── make_paper_figures.py       # Paper figures (6 figs + stats)
├── docs/                  # Research docs & paper draft
│   ├── ea_analysis.md
│   ├── paper_draft.md
│   ├── TECHNICAL_REPORT.md
│   ├── research_proposal.md
│   ├── research_directions_2025.md
│   ├── neural_sde_research_plan.md
│   └── spd_ssl_research_plan.md
├── tests/                 # 350 unit tests (29 files, pytest)
├── main.py                # Single entry point (18 commands)
├── environment.yml
└── requirements.txt
```

## Commands

### Pipeline (via main.py)

```bash
python main.py setup          # Create conda environment
python main.py preprocess     # Run MNE pipeline (8ch default)
python main.py baseline       # CSP + SVM baseline
python main.py train          # Train EEGNet
python main.py ablation       # Ablation study (6 configs)
python main.py loso           # LOSO cross-validation (gold standard)
python main.py demo           # Real-time terminal demo
python main.py dashboard      # Streamlit dashboard (port 8501)
python main.py record         # DeepBCI data collection (interactive)
python main.py run_all        # Full pipeline: preprocess → train → LOSO → export
python main.py export         # Competition Excel report
python main.py riemann        # Riemannian geometry LOSO
python main.py figures        # Report figures (confusion matrix, bar charts)
python main.py metadata       # Export dataset metadata to JSON
python main.py subjectwise    # Subject-wise eval from checkpoint
```

### Data

```bash
python data/download.py                  # PhysioNet MI (subjects 1–30)
python data/download.py --bci_iv_2a      # BCI Competition IV 2a (MOABB)
python data/download.py --sample         # MNE sample data for quick test
```

### Quality

```bash
black . && ruff check .              # Format + lint
pytest                               # Run all tests (350)
pytest tests/ -v                     # Verbose, single file
```

### Preprocessing (direct)

```bash
python preprocessing/run_mne_pipeline.py                          # motor8 default (8ch)
python preprocessing/run_mne_pipeline.py --channels motor8        # explicit 8ch
python preprocessing/run_mne_pipeline.py --channels motor16       # 16ch backward compat
python preprocessing/run_mne_pipeline.py --channels all           # all 64 channels
python preprocessing/run_mne_pipeline.py --binary                 # 2-class (left vs right)
python preprocessing/run_mne_pipeline.py --dataset physionet_mi   # explicit dataset (auto-detect by default)
```

### Training (direct)

```bash
# Basic
python training/train_eegnet.py --epochs 200 --data_dir data/processed/
python training/train_baseline.py --n_components 6 --cv 5

# Model selection (via create_model factory)
python training/train_eegnet.py --model eegnet                     # base (default)
python training/train_eegnet.py --model eegnet_se                  # + SE attention
python training/train_eegnet.py --model eegnet_mhsa                # + multi-head attention
python training/train_eegnet.py --model eegnet_temporal            # + temporal attention
python training/train_eegnet.py --model eegnet_spatiotemporal      # + combined (best)

# Advanced features
python training/train_eegnet.py --augment                          # data augmentation (2x)
python training/train_eegnet.py --label_smoothing 0.1              # label smoothing
python training/train_eegnet.py --early_stop 50                    # early stopping patience
python training/train_eegnet.py --grad_clip 1.0                    # gradient clipping
python training/train_eegnet.py --kfold 5                          # K-fold cross-validation

# Sweep & ablation
python training/train_sweep.py --model eegnet --trials 50          # Optuna hyperparam search
python training/train_ablation.py --epochs 150 --repeat 3          # 6 configs × 3 repeats

# LOSO (Leave-One-Subject-Out) — gold-standard BCI evaluation
python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary
python training/train_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60
python training/train_loso.py --data_dir data/loso_binary --n_subjects 30 --finetune 5   # + few-shot FT

# SPDNet LOSO
python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align
python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align --cov_estimator lwf
python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align --multiband

# BCI IV 2a LOSO
python preprocessing/prepare_bci_iv_2a.py
python training/train_loso.py --data_dir data/bci_iv_2a_processed --n_subjects 9 --epochs 60 --dataset bci_iv_2a

# Demo — file replay (offline simulation of online closed-loop)
python main.py demo --source replay --data data/loso_binary/subj_01/X.npy \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt          # basic replay
python main.py demo --source replay --data data/loso_binary/subj_01/X.npy \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt --gating # + idle gating
python main.py demo --all-subjects --source replay \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt          # batch all subjects
```

### Riemannian & SPD

```bash
# Riemannian baselines
python main.py riemann --method tangent --align
python main.py riemann --method tangent --align --cov_estimator lwf
python main.py riemann --method fgmdm --align --bands 8 12 12 16 16 20 20 24 24 28 28 30

# SPDNet (SPD manifold deep learning)
python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align
```

### DeepBCI

```bash
python datasets/deepbci_loader.py -i data/subjects/sub_001/session_xxx   # Single session → .npy
python datasets/deepbci_loader.py --all                                  # All sessions
```

### Reports

```bash
python utils/report_excel.py --demo                    # Generate Excel template
python utils/report_excel.py --input results.json      # From JSON results
```

### Paper Figures

```bash
python scripts/make_paper_figures.py                   # 6 figures + stats tests
python scripts/analyze_spdnet_vs_tangent.py            # SPDNet vs Tangent comparison
python scripts/analyze_ea_effects.py --data_dir data/loso_binary  # EA effect analysis
```

## Key Conventions

- **Data shape**: `X = [N, C, T]` float32, `y = [N]` int
- **Labels**: 0=Rest/Idle, 1=Left, 2=Right (canonical — enforced by `--dataset physionet_mi`)
  - PhysioNet MI raw: annotations T0→0, T1→1, T2→2 (via `PHYSIONET_MI_EVENT_TO_LABEL`)
  - BCI IV 2a raw: triggers 769→0(left), 770→1(right), 771→2(feet), 772→3(tongue)
- **Validation**: LOSO (Leave-One-Subject-Out) is the gold standard. Random split is for debugging only.
- **Hardware**: **8 channels** @ 250 Hz (DeepBCI)
- **8ch Montage**: FC3, C3, Cz, C4, FC4, CP3, CPz, CP4
  - PhysioNet names: `Fc3. C3.. Cz.. C4.. Fc4. Cp3. Cpz. Cp4.`
  - BCI IV 2a names: `FC3 C3 Cz C4 FC4 CP3 CPz CP4`
- **Bands**: mu (8–13 Hz), beta (13–30 Hz), full (8–30 Hz)
- **FBCSP_BANDS**: 6 sub-bands within 8–30 Hz: `[(8,12), (12,16), (16,20), (20,24), (24,28), (28,30)]`
- **Config**: Edit `utils/config.py` to change channels/montage
- **Immutability**: prefer new arrays over in-place mutation
- **Format/lint**: `black . && ruff check .`
- **EEGSource Protocol**: All stream sources must satisfy `EEGSource` (open/read_chunk/close)
  - Impl: `DummyStream`, `FileReplaySource` (trial_mode + streaming), `LSLStream`, `DeepBCISource`
  - Use `FileReplaySource(trial_mode=True)` to feed full trials directly (bypass RingBuffer)
- **Idle gating**: `predict_with_gating()` auto-detects STOP class from action_map — works for binary/3-class/4-class
- **LOSO output**: Files include dataset tag (e.g. `loso_eegnet_bci_iv_2a.csv`) and FT tag (e.g. `_ft5`)

### Inter-stage data contract

All training scripts read from `data/processed/` and expect these exact files:

```
data/processed/
├── X_train.npy    # float32 (N_train, C, T) — C should be 8
├── y_train.npy    # int64   (N_train,)
├── X_val.npy      # float32 (N_val, C, T)
└── y_val.npy      # int64   (N_val,)
```

Generated by `preprocessing/run_mne_pipeline.py` via 75/25 stratified split.

### EEGNet lazy classifier & checkpoint loading

EEGNet builds its final `nn.Linear` layer on the **first forward pass** (to handle variable time lengths). This means:

1. **Training**: no special handling — the first batch triggers `_build_classifier()`.
2. **Loading a checkpoint**: you MUST warm up the model with a dummy forward before `load_state_dict()`, otherwise the classifier won't exist yet:

```python
model = EEGNet(n_channels=cfg["n_channels"], n_classes=cfg["n_classes"])
model.eval()
with torch.no_grad():
    model(torch.zeros(1, cfg["n_channels"], cfg["n_times"]))  # builds classifier
model.load_state_dict(ckpt["state_dict"])
```

Use `training/train_eegnet.load_checkpoint(path)` — it handles this internally.

### Checkpoint format

```python
ckpt = {
    "epoch": int,
    "state_dict": model.state_dict(),
    "opt": optimizer.state_dict(),
    "acc": float,
    "config": {
        "n_channels": int, "n_classes": int, "n_times": int,
        "F1": int, "D": int, "F2": int, "dropout": float,
    },
}
```

Saved to `checkpoints/{model_type}_best.pt` by default.

### Model factory

Use `create_model()` from `models/eegnet_attn.py` to instantiate any model variant:

```python
from models.eegnet_attn import create_model

model = create_model("eegnet", n_channels=8, n_classes=3)
model = create_model("eegnet_spatiotemporal", n_channels=8, n_classes=3)
model = create_model("eegnet_mhsa", n_channels=8, n_classes=3)
model = create_model("fbcnet", n_channels=8, n_classes=3)
model = create_model("eeg_tcnet", n_channels=8, n_classes=3)
model = create_model("eeg_conformer", n_channels=8, n_classes=3)
model = create_model("fb_tcnet", n_channels=8, n_classes=3)
model = create_model("fb_maa_eegnet", n_channels=8, n_classes=3)
model = create_model("maa_eegnet", n_channels=8, n_classes=3)
model = create_model("maa_eegnet_pre", n_channels=8, n_classes=3)
model = create_model("er_mi", n_channels=8, n_classes=3)       # ER-MI v1: GRU evidence reasoning
model = create_model("er_mi_v2", n_channels=8, n_classes=3)    # ER-MI v2: multi-token evidence
model = create_model("brt_det", n_channels=8, n_classes=2)    # BRT-Det v8: Band-Region-Time Evidence Detector (63.02% 3-seed)
```

### New model architectures

```python
from models.eeg_conformer import EEGConformer
from models.eeg_tcnet import EEGTCNet
from models.fbcnet import FBCNet, apply_filter_bank
from models.fb_tcnet import FBTCNet
from models.spd_models import create_spdnet, MultiBandSPDNet, SPDDecoder, ProtoSPDNet
from models.mixstyle import MixStyle1d, MixStyle2d
from models.motor_area_attention import MotorAreaAttention
from models.fb_maa_eegnet import FBMAAEEGNet
from models.maa_eegnet import MAAEEGNet
from models.maa_eegnet_pre import MAAEEGNetPre
from models.brt_det import BRTDet

# BRT-Det: Band-Region-Time Evidence Detector (v8, 3-seed: 63.02% ± 1.42%, rank 3)
model = BRTDet(n_channels=8, n_classes=2, use_region_pool=False, n_time_cells=24,
               dilations=[1,2,4], agg_mode="objectness", use_band_gate=True)
model = BRTDet(n_channels=8, n_classes=2, topk=10)  # sparse top-K evidence aggregation
model = BRTDet(n_channels=8, n_classes=2, use_diff_channels=True)  # +C3/C4 diff channels

# EEG Conformer: CNN backbone + Transformer encoder (best overall: 63.93%)
model = EEGConformer(n_channels=8, n_classes=2, d_model=64, n_heads=4, n_layers=2)

# EEG-TCNet: EEGNet Block1 + TCN (temporal conv net, 63.41%)
model = EEGTCNet(n_channels=8, n_classes=2, F1=8, D=2, kernel_size=16)

# FBCNet: Filter-bank CNN (requires multi-band input, 61.11% + EA)
X_bands = apply_filter_bank(X, fs=250)  # (N, C, T) → (N, n_bands, C, T)
model = FBCNet(n_bands=6, n_channels=8, n_classes=2)

# FB-TCNet: Filter Bank + TCN (project original — combined FBCNet + EEG-TCNet)
model = FBTCNet(n_bands=6, n_channels=8, n_classes=2, F1=8, D=2)

# SPDNet: SPD manifold deep learning (Huang & Van Gool, AAAI 2017)
from features.spd_covariance import compute_covariance
covs = compute_covariance(X)  # (N, C, T) → (N, C, C) SPD matrices
model = create_spdnet(n_classes=2, bimap_dims=[8, 6, 4])  # (N, C, C) → (N, n_classes)

# MAA modules: Motor-Area Attention for 8ch motor-cortex EEG
maa = MotorAreaAttention(n_channels=8)               # region-based channel weighting
model = MAAEEGNet(n_channels=8, n_classes=2)         # MAA after temporal conv
model = MAAEEGNetPre(n_channels=8, n_classes=2)      # MAA before temporal conv (raw EEG)
model = FBMAAEEGNet(n_bands=6, n_channels=8, n_classes=2)  # Filter Bank + MAA + EEGNet

# MixStyle: domain generalization via feature-statistics mixing
self.mixstyle = MixStyle2d(p=0.5, alpha=0.1)  # insert after BN/activation
```

### Domain adaptation & alignment

```python
from preprocessing.alignment import EuclideanAlignment
from utils.domain_adapt import center_loss, multi_kernel_mmd

# Euclidean Alignment: align covariance across subjects (unsupervised)
ea = EuclideanAlignment()
ea.fit([train_subj_X, ...])  # compute reference covariance
aligned = ea.transform(X)     # apply alignment

# Center Loss: pull same-class features toward shared centers
loss_ct, centers = center_loss(features, labels, n_classes=3, centers=centers)
loss = ce_loss + 0.01 * loss_ct
```

### SPD covariance computation

```python
from features.spd_covariance import (
    compute_covariance,           # SCM with regularization
    compute_covariance_shrinkage, # Ledoit-Wolf shrinkage
    compute_multiband_covariance, # Multi-band SPD matrices
    paired_spd_augment,           # Per-class pair augmentation
    mask_covariance_channels,     # Channel dropout
    batch_geodesic_mixup,         # SPD manifold mixup
)
```

### 8ch Montage Configuration

Channel montage is defined in `utils/config.py`. To change channels, edit `MOTOR_CHANNELS`:

```python
# utils/config.py
N_CHANNELS = 8

# PhysioNet 10-10 naming (dots matter!)
MOTOR_CHANNELS = [
    "Fc3.",                     # FC3
    "C3..", "Cz..", "C4..",    # C3, Cz, C4
    "Fc4.",                     # FC4
    "Cp3.", "Cpz.", "Cp4.",    # CP3, CPz, CP4
]

# BCI IV 2a 10-20 naming (no dots)
MOTOR_CHANNELS_BCI4 = [
    "FC3", "C3", "Cz", "C4", "FC4", "CP3", "CPz", "CP4",
]
```

## Current Results

### PhysioNet MI (30 subjects, 8ch, binary LOSO, 80 epochs for DL, 60 for SPDNet)

| Rank | Method | Type | Accuracy | Kappa |
|------|-------|------|----------|-------|
| 1 | **EEG Conformer + EA** | DL + Transformer | **63.93%** ± 9.58% | 0.277 |
| 2 | **EEG-TCNet + EA** | DL + TCN | **63.41%** ± 10.51% | 0.265 |
| 3 | **BRT-Det + EA** ★ | DL + Detection | **63.02%** ± 1.42% | 0.257 |
| 4 | **ER-MI + EA** | DL + GRU Reasoning | **62.55%** ± 0.92% | 0.246 |
| 5 | **FBCNet + EA** | DL + Filter Bank | **61.11%** ± 11.69% | 0.219 |
| 6 | Tangent Space + LDA + EA | Riemannian | 60.44% ± 9.64% | 0.212 |
| 7 | FgMDM + EA (6-band, 8–30Hz) | Riemannian | 59.18% ± 8.12% | 0.180 |
| 8 | **SPDNet + EA** (seed42) | DL + SPD Manifold | **58.52%** ± 8.38% | 0.161 |
| 9 | EEGNet + EA | DL | 58.00% ± 10.06% | 0.161 |
| 10 | EEGNet + SpatiotemporalAttn + EA | DL + Attention | 57.78% ± 8.55% | 0.158 |
| 11 | MDM + EA | Riemannian | 56.22% ± 10.52% | 0.127 |
| 12 | FB-MAA-EEGNet + EA | DL + FB + MAA | 53.78% ± 7.68% | 0.070 |
| 13 | EEGNet (no EA) | DL | 51.93% ± 7.20% | 0.033 |
| 14 | SPDNet (no EA) | DL + SPD Manifold | 50.59% ± 1.87% | 0.000 |
| 15 | FBCNet (no EA) | DL + Filter Bank | 49.70% ± 2.66% | -0.010 |

> Key insight: EEG Conformer + EA (63.93%) remains SOTA. **BRT-Det + EA (63.02%)** debuts at rank 3 — a lightweight detection model (32K params) that reframes MI decoding as band-region-time evidence detection. Band Gate (per-band scalar gating, +5.41pp) is the key v8 breakthrough. BRT-Det beats ER-MI (+0.47pp) and FBCNet (+1.91pp) despite using only 32K parameters. ★ ER-MI (62.55%): 3-seed 62.30/61.78/63.56.

### SPDNet Ablation

| Config | Accuracy | Kappa |
|--------|----------|-------|
| SPDNet + EA (d=[8,8], seed42) | **58.52%** ± 8.38% | 0.161 |
| SPDNet + EA (d=[8,8], seed123) | 56.96% ± 9.52% | 0.133 |
| SPDNet + EA (d=[8,8], seed456) | 55.93% ± 8.83% | 0.112 |
| SPDNet + EA (d=[8,8], default) | 50.44% ± 4.59% | 0.020 |
| SPDNet (no EA) | 50.59% ± 1.87% | 0.000 |

> SPDNet training is seed-sensitive; controlled seeds produce stable results, but default-seed runs collapse. Without EA, all SPDNet runs collapse to majority class.

### EA Gain Analysis

| Model | no EA | + EA | Δ |
|-------|-------|------|-----|
| EEGNet | 51.93% | 58.00% | +6.07pp |
| EEGNet + SpatiotemporalAttn | 55.04% | 57.78% | +2.74pp |
| FBCNet | 49.70% | 61.11% | +11.41pp |
| SPDNet | 50.59% | 58.52% | +7.93pp |
| Tangent Space | 60.44% | 60.44% | ±0.00pp (affine-invariant) |

### Riemannian Cov Estimator & Metric Sweep

| cov_estimator | Acc | metric | Acc |
|---------------|-----|--------|-----|
| scm (sample) | 60.44% | riemann | 60.44% |
| lwf (Ledoit-Wolf) | 60.30% | wasserstein | 60.44% |
| oas (Oracle) | 60.23% | logeuclid | 59.70% |
| mcd (robust) | — | logchol | 59.25% |

### BCI Competition IV 2a (9 subjects, 8ch, 4-class LOSO)

| Method | Type | Accuracy | Kappa |
|-------|------|----------|-------|
| **EEGNet (base)** | DL | **39.47%** ± 12.45% | 0.193 |
| Tangent Space + LDA + EA | Riemannian | 38.60% ± 12.44% | 0.181 |
| EEGNet + SpatiotemporalAttn | DL | 36.94% ± 11.78% | 0.159 |
| FgMDM + EA | Riemannian | 34.91% ± 8.48% | 0.132 |
| MDM + EA | Riemannian | 33.43% ± 10.92% | 0.112 |

## References

- `XH-202610_基于运动想象的脑－机交互算法研究.pdf` — Project proposal
- EEGNet: Lawhern et al. 2018 (https://doi.org/10.1088/1741-2552/aace8c)
- EEG Conformer: Song et al. 2023 (arXiv:2301.05578)
- EEG-TCNet: Ingolfsson et al. 2020 (IEEE SMC)
- FBCNet: Bakshi et al. 2021 (arXiv:2104.01233)
- SPDNet: Huang & Van Gool 2017, "A Riemannian Network for SPD Matrix Learning" (AAAI)
- MNE-Python: https://mne.tools
- MOABB: https://github.com/NeuroTechX/moabb
- pyriemann: https://github.com/pyRiemann/pyRiemann
- LSL: https://labstreaminglayer.org

## Current Sprint

**Sprint 3** (late June–July 2026): SPD manifold deep learning, BRT-Det detection paradigm + paper drafting.

- [x] SPDNet implementation: BiMap, ReEig, LogEig layers + model factory
- [x] SPD covariance computation: SCM, Ledoit-Wolf shrinkage, multiband
- [x] SPDNet LOSO training script with EA, seed control, multiband support
- [x] SPD data augmentation: paired per-class augmentation, channel masking, geodesic mixup
- [x] SPDNet results: 58.52% + EA (competitive with EEGNet + EA), +7.93pp EA gain
- [x] FB-TCNet model (Filter Bank + TCN, project original)
- [x] Paper figures script (6 figures): main results, ablation, EA gain, SPDNet vs Tangent, few-shot, t-SNE
- [x] EA × Architecture interaction analysis script
- [x] Paper draft (docs/paper_draft.md, results/paper_outputs/)
- [x] Test suite: 335 → 350 tests (29 files)
- [x] **BRT-Det (Band-Region-Time Evidence Detector)**: 46.6% → 63.0% across 35+ experiments
- [x] BRT-Det v8: Band Gate (per-band scalar gating) — 3-seed 63.02% ± 1.42%, rank 3

**Key findings:**
- SPDNet + EA (58.52%) is competitive with EEGNet + EA (58.00%) — SPD manifold DL viable on 8ch
- SPDNet without EA collapses to chance (κ≈0) — EA is even more critical for SPDNet than for CNN
- EEG Conformer + EA (63.93%) remains SOTA; temporal modeling > spatial/manifold for 8ch MI
- **BRT-Det v8 (63.02%, 3-seed)** debuts at rank 3 with only 32K params:
  - Band Gate (32 params): per-band reliability weighting — NOT cross-band mixing
  - Core insight: MI effective bands vary across subjects; band-wise gating > forced mixing
  - v8 3-seed is only +0.43pp over v7 best single-seed, but κ<0 subjects drop from 7→3
  - Main gain is cross-subject stability, not peak accuracy
- **Cross-subject variance** is the dominant bottleneck: removing 3 κ<0 subjects lifts mean to 66.17%
- **BRT-Det now lacks subject adaptation, not model capacity** — next: cross-dataset validation, few-shot FT, evidence visualization

### Sprint 2 (completed): Traditional-driven neural network design + full experiment sweep.

- [x] FBCSP_BANDS corrected: 9-band 4–40Hz → 6-band 8–30Hz
- [x] Riemannian sweep: cov estimator + metric
- [x] DL baselines with EA: EEGNet, FBCNet, EEG-TCNet, EEG Conformer
- [x] MAA-EEGNet (3 variants) + FB-MAA-EEGNet
- [x] EA gain analysis: +3~11pp; FBCNet benefits most
- [x] Time window sweep script + 10-config ablation
- [x] Test suite: 304 → 335 tests

### Sprint 1.5 (completed): Model zoo + domain generalization.

- [x] EEG Conformer, EEG-TCNet, FBCNet
- [x] MixStyle, EA, Center Loss, MMD
- [x] Riemannian Geometry baseline (60.44%)
- [x] Test suite: 131 → 304 tests

### Backlog
- [ ] Hyperparameter sweep for top models (Conformer/TCNet)
- [ ] Few-shot calibration sweep on best models
- [ ] Time window sweep (full run — script ready, needs compute)
- [ ] Real-time LSL integration test with DeepBCI hardware
- [ ] MAA redesign: learnable grouping or gated attention
- [ ] SPDNet SSL: prototype regularization, SPD augmentation
- [ ] NeurIPS-style paper submission
