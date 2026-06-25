# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**脑机接口（BCI）— 基于运动想象的脑－机交互算法研究**

BCI algorithm research using the Motor Imagery (MI) paradigm. 挑战杯 project **XH-202610** (May–Nov 2026).

**Team:** Solo (单人全栈)

## Environment Setup

```bash
conda env create -f environment.yml              # once
conda activate bci                                # per session
pip install mne moabb streamlit plotly pytest black ruff openpyxl  # full deps
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
│   └── metadata.py       #   Dataset metadata export
├── preprocessing/
│   ├── run_mne_pipeline.py  # MNE pipeline: --channels motor8|motor16|all, --binary
│   ├── prepare_bci_iv_2a.py # BCI IV 2a MOABB .npy → 8ch per-subject
│   ├── filtering.py         #   bandpass 8-30Hz + notch 50Hz
│   ├── epoching.py          #   epoch creation
│   ├── artifact.py          #   ICA removal
│   ├── mne_pipeline.py      #   programmatic API
│   ├── alignment.py         #   Euclidean Alignment (EA) cross-subject
│   └── augment.py           #   on-the-fly augmentation
├── features/              # csp.py (CSP + FBCSP), riemann.py (Tangent/MDM/FgMDM), bandpower.py
├── models/
│   ├── eegnet.py            # EEGNet (Lawhern 2018) — lazy classifier
│   ├── attention.py         # ChannelAttention1D, MultiHeadChannelAttention,
│   │                        #   TemporalAttention, SpatiotemporalAttention
│   ├── eegnet_attn.py       # EEGNet + attention deep integration (5 variants)
│   ├── eeg_conformer.py     # EEG Conformer (CNN + Transformer, Song 2023)
│   ├── eeg_tcnet.py         # EEG-TCNet (CNN + TCN, Ingolfsson 2020)
│   ├── fbcnet.py            # FBCNet (Filter-Bank CNN, Bakshi 2021)
│   ├── fusion.py            # MultiBandFusion (mu/beta/full)
│   └── mixstyle.py          # MixStyle domain generalization (Zhou, ICLR 2021)
├── training/
│   ├── train_eegnet.py     # main training script (6 new features)
│   ├── train_baseline.py   # CSP+SVM baseline
│   ├── train_ablation.py   # 6-config comparison with repeat
│   └── train_sweep.py      # Optuna hyperparameter search + manual grid fallback
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
│   └── make_report_figures.py      # Confusion matrix / bar charts / ablation
├── tests/                 # 267 unit tests (pytest)
├── main.py                # Single entry point (16 commands)
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
pytest                               # Run all tests (267)
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

# Model selection (5 variants)
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

Use `create_model()` from `models/eegnet_attn.py` to instantiate any EEGNet variant:

```python
from models.eegnet_attn import create_model

model = create_model("eegnet", n_channels=8, n_classes=3)
model = create_model("eegnet_spatiotemporal", n_channels=8, n_classes=3)
model = create_model("eegnet_mhsa", n_channels=8, n_classes=3)
```

### New model architectures (Sprint 1.5)

```python
from models.eeg_conformer import EEGConformer
from models.eeg_tcnet import EEGTCNet
from models.fbcnet import FBCNet, apply_filter_bank
from models.mixstyle import MixStyle1d, MixStyle2d

# EEG Conformer: CNN backbone + Transformer encoder
model = EEGConformer(n_channels=8, n_classes=3, d_model=64, n_heads=4, n_layers=2)

# EEG-TCNet: EEGNet Block1 + TCN (temporal conv net)
model = EEGTCNet(n_channels=8, n_classes=3, F1=8, D=2, kernel_size=16)

# FBCNet: Filter-bank CNN (requires multi-band input)
X_bands = apply_filter_bank(X, fs=250)  # (N, C, T) → (N, n_bands, C, T)
model = FBCNet(n_bands=9, n_channels=8, n_classes=3)

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

### PhysioNet MI (30 subjects, 8ch, binary LOSO)

| Method | Type | LOSO Accuracy | Kappa |
|-------|------|--------------|-------|
| EEGNet (base) | DL | 51.93% ± 7.20% | 0.033 |
| EEGNet + Few-shot FT (5 trials) | DL | 54.95% ± 8.04% | 0.099 |
| EEGNet + SpatiotemporalAttn | DL | 55.04% ± 7.86% | 0.096 |
| MDM + EA | Riemannian | 56.30% ± 10.94% | 0.129 |
| FgMDM + EA | Riemannian | 60.00% ± 9.04% | 0.198 |
| **Tangent Space + LDA + EA** | **Riemannian** | **60.30%** ± 9.75% | **0.208** |

### BCI Competition IV 2a (9 subjects, 8ch, 4-class LOSO)

| Method | Type | LOSO Accuracy | Kappa |
|-------|------|--------------|-------|
| MDM + EA | Riemannian | 33.43% ± 10.92% | 0.112 |
| FgMDM + EA | Riemannian | 34.91% ± 8.48% | 0.132 |
| EEGNet + SpatiotemporalAttn | DL | 36.94% ± 11.78% | 0.159 |
| Tangent Space + LDA + EA | Riemannian | 38.60% ± 12.44% | 0.181 |
| **EEGNet (base)** | **DL** | **39.47%** ± 12.45% | **0.193** |

> Key insight: Riemannian dominates DL by +8pp on binary, but EEGNet edges ahead on 4-class. The setting ceiling is ~60%, not ~55%.

## References

- `XH-202610_基于运动想象的脑－机交互算法研究.pdf` — Project proposal
- EEGNet: Lawhern et al. 2018 (https://doi.org/10.1088/1741-2552/aace8c)
- MNE-Python: https://mne.tools
- MOABB: https://github.com/NeuroTechX/moabb
- LSL: https://labstreaminglayer.org

## Current Sprint

**Sprint 1.5** (June 2026): Model zoo + domain generalization + code review.

- [x] Repo structure + skeleton modules
- [x] MNE preprocessing (PhysioNet MI, 30 subjects)
- [x] EEGNet model with lazy classifier
- [x] Channel attention (SE, MHSA, Temporal, Spatiotemporal)
- [x] EEGNet + attention deep integration (5 variants)
- [x] Training scripts (baseline, EEGNet, ablation, sweep)
- [x] Data augmentation pipeline
- [x] Real-time pipeline (dummy + LSL scaffold)
- [x] Streamlit dashboard (model loading, CSV export)
- [x] 8-channel adaptation (hardware-aligned montage)
- [x] BCI Competition IV 2a dataset (downloaded)
- [x] Unit test suite (267 tests)
- [x] Excel validation report generator
- [x] CSP baseline: 38.6% (3-class, 8ch)
- [x] EEGNet: 57.6% (3-class, 8ch), 63.0% (2-class, 8ch)
- [x] LOSO evaluation (PhysioNet MI binary: EEGNet 51.93%, Spatiotemporal 55.04%, FT5 54.95%)
- [x] EEGSource Protocol + FileReplaySource + demo --source replay + idle gating
- [x] DeepBCI recorder session subdirectories + metadata fields
- [x] BCI IV 2a preprocessing (8ch motor selection, per-subject split)
- [x] DeepBCISource placeholder (EEGSource protocol, raises NotImplementedError)
- [x] BCI IV 2a LOSO (4-class: EEGNet 39.47%, Spatiotemporal 36.94%; chance=25%)
- [x] Dashboard File Replay integration (Synthetic + File Replay, dynamic n_classes)
- [x] Multi-subject batch demo (--all-subjects)
- [x] Action map fix (binary/4-class gating, auto-idle detection)
- [x] FBCSP feature extraction (Filter Bank CSP + LDA/SVM)
- [x] EEG Conformer model (CNN + Transformer for MI-EEG)
- [x] EEG-TCNet model (CNN + TCN for embedded deployment)
- [x] FBCNet model (Filter-Bank CNN with variance pooling)
- [x] MixStyle domain generalization (feature-statistics mixing)
- [x] Euclidean Alignment (EA) cross-subject covariance alignment
- [x] Domain adaptation losses (Center Loss + MMD)
- [x] DeepBCI session loader (session → .npy training data)
- [x] Full code review + bug fixes across all modules
- [x] Test suite expanded (131 → 304 tests)
- [x] Riemannian Geometry baseline (Tangent Space / MDM / FgMDM) — 60.30% binary LOSO
- [ ] Hyperparameter sweep (ready, needs GPU time)
- [ ] Real-time LSL integration test with DeepBCI hardware
