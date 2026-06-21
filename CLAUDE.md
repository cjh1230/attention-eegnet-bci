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
├── preprocessing/         # run_mne_pipeline.py — end-to-end MNE
│   ├── filtering.py       #   bandpass 8-30Hz + notch 50Hz
│   ├── epoching.py        #   epoch creation
│   ├── artifact.py        #   ICA removal
│   ├── mne_pipeline.py    #   programmatic API
│   ├── augment.py         #   on-the-fly augmentation (noise/dropout/shift/scale)
│   └── run_mne_pipeline.py #  main entry: --channels motor8|motor16|all, --binary
├── features/              # csp.py, bandpower.py
├── models/
│   ├── eegnet.py           # EEGNet (Lawhern 2018) — lazy classifier
│   ├── attention.py        # ChannelAttention1D, MultiHeadChannelAttention,
│   │                       #   TemporalAttention, SpatiotemporalAttention
│   ├── eegnet_attn.py      # EEGNet + attention deep integration (5 variants)
│   └── fusion.py           # MultiBandFusion (mu/beta/full)
├── training/
│   ├── train_eegnet.py     # main training script (6 new features)
│   ├── train_baseline.py   # CSP+SVM baseline
│   ├── train_ablation.py   # 6-config comparison with repeat
│   └── train_sweep.py      # Optuna hyperparameter search + manual grid fallback
├── realtime/              # stream.py, stream_lsl.py, buffer.py, inference.py
├── utils/
│   ├── config.py           # global config — EDIT THIS for channel changes
│   ├── metrics.py          # classification_report, per_class_accuracy
│   ├── report_excel.py     # 5-sheet Excel validation report (competition format)
│   └── logger.py           # ExperimentLogger (CSV)
├── ui/                    # dashboard.py (Streamlit)
├── tests/                 # 66 unit tests (pytest)
├── main.py                # Single entry point
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
python main.py demo           # Real-time terminal demo (100 steps)
python main.py dashboard      # Streamlit dashboard (port 8501)
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
pytest                               # Run all tests (66)
pytest tests/ -v                     # Verbose, single file
```

### Preprocessing (direct)

```bash
python preprocessing/run_mne_pipeline.py                          # motor8 default (8ch)
python preprocessing/run_mne_pipeline.py --channels motor8        # explicit 8ch
python preprocessing/run_mne_pipeline.py --channels motor16       # 16ch backward compat
python preprocessing/run_mne_pipeline.py --channels all           # all 64 channels
python preprocessing/run_mne_pipeline.py --binary                 # 2-class (left vs right)
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
```

### Reports

```bash
python utils/report_excel.py --demo                    # Generate Excel template
python utils/report_excel.py --input results.json      # From JSON results
```

## Key Conventions

- **Data shape**: `X = [N, C, T]` float32, `y = [N]` int
- **Labels**: 0=Idle, 1=Left, 2=Right
- **Hardware**: **8 channels** @ 250 Hz (DeepBCI)
- **8ch Montage**: FC3, C3, Cz, C4, FC4, CP3, CPz, CP4
  - PhysioNet names: `Fc3. C3.. Cz.. C4.. Fc4. Cp3. Cpz. Cp4.`
  - BCI IV 2a names: `FC3 C3 Cz C4 FC4 CP3 CPz CP4`
- **Bands**: mu (8–13 Hz), beta (13–30 Hz), full (8–30 Hz)
- **Config**: Edit `utils/config.py` to change channels/montage
- **Immutability**: prefer new arrays over in-place mutation
- **Format/lint**: `black . && ruff check .`

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

Use `create_model()` from `models/eegnet_attn.py` to instantiate any variant:

```python
from models.eegnet_attn import create_model

model = create_model("eegnet", n_channels=8, n_classes=3)
model = create_model("eegnet_spatiotemporal", n_channels=8, n_classes=3)
model = create_model("eegnet_mhsa", n_channels=8, n_classes=3)
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

## Current Results (PhysioNet MI, 30 subjects, 8ch)

| Model | 3-Class | 2-Class |
|-------|---------|---------|
| CSP + SVM | 38.6% | — |
| EEGNet (base) | 53.8% | 56.2% |
| EEGNet + SpatiotemporalAttn | **57.6%** | **63.0%** |

Best binary confusion matrix:
```
           Pred L  Pred R
True Left     116      55      (67.8%)
True Right     70      97      (58.1%)
```

## References

- `XH-202610_基于运动想象的脑－机交互算法研究.pdf` — Project proposal
- EEGNet: Lawhern et al. 2018 (https://doi.org/10.1088/1741-2552/aace8c)
- MNE-Python: https://mne.tools
- MOABB: https://github.com/NeuroTechX/moabb
- LSL: https://labstreaminglayer.org

## Current Sprint

**Sprint 1** (now → July 10): Refinement + results.

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
- [x] Unit test suite (66 tests)
- [x] Excel validation report generator
- [x] CSP baseline: 38.6% (3-class, 8ch)
- [x] EEGNet: 57.6% (3-class, 8ch), 63.0% (2-class, 8ch)
- [ ] BCI IV 2a preprocessing + training
- [ ] Hyperparameter sweep (ready, needs GPU time)
- [ ] Real-time LSL integration test with DeepBCI hardware
