# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**脑机接口（BCI）— 基于运动想象的脑－机交互算法研究**

BCI algorithm research using the Motor Imagery (MI) paradigm. 挑战杯 project **XH-202610** (May–Nov 2026).

**Team:** Solo (单人全栈)

## Environment Setup

```bash
conda env create -f environment.yml
conda activate bci
```

Or quickly:
```bash
python main.py setup
```

## Tech Stack

| Layer | Choice |
|-------|--------|
| Python | 3.10 |
| Preprocessing | MNE-Python (offline pipeline) |
| Deep Learning | PyTorch 2.x |
| Real-time | LSL (pylsl) or simulated stream |
| UI | Streamlit |
| Env | conda (environment.yml) |

## Project Structure

```
bci_project/
├── data/raw/              # Raw EEG (.edf, .fif, .gdf)
├── data/processed/        # MNE output (.npy) — X=[N,C,T], y=[N]
├── data/subjects/         # Per-subject metadata
├── preprocessing/         # run_mne_pipeline.py — end-to-end MNE
│   ├── filtering.py       #   bandpass + notch
│   ├── epoching.py        #   epoch creation
│   ├── artifact.py        #   ICA removal
│   └── mne_pipeline.py    #   programmatic API
├── features/              # csp.py, bandpower.py
├── models/                # eegnet.py, attention.py, fusion.py
├── training/              # train_eegnet.py, train_baseline.py, train_ablation.py
├── realtime/              # stream.py, stream_lsl.py, buffer.py, inference.py
├── utils/                 # config.py, metrics.py, logger.py
├── ui/                    # dashboard.py (Streamlit)
├── main.py                # Single entry point
├── environment.yml
└── requirements.txt
```

## Commands (via main.py)

```bash
python main.py setup          # Create conda environment
python main.py preprocess     # Run MNE pipeline on data/raw/
python main.py baseline       # CSP+SVM baseline
python main.py train          # Train EEGNet
python main.py ablation       # Ablation study
python main.py demo           # Real-time terminal demo
python main.py dashboard      # Streamlit dashboard (streamlit run ui/dashboard.py)
```

## Key Conventions

- **Data shape**: `X = [N, C, T]` float32, `y = [N]` int
- **Labels**: 0=Idle, 1=Left, 2=Right
- **Bands**: mu (8–13 Hz), beta (13–30 Hz), full (8–30 Hz)
- **Default HW**: 16 channels @ 250 Hz (DeepBCI) — edit `utils/config.py`
- **Immutability**: prefer new arrays over in-place mutation
- **Format/lint**: `black . && ruff check .`

## References

- `XH-202610_基于运动想象的脑－机交互算法研究.pdf` — Project proposal
- EEGNet: Lawhern et al. 2018 (https://doi.org/10.1088/1741-2552/aace8c)
- MNE-Python: https://mne.tools
- LSL: https://labstreaminglayer.org

## Current Sprint

**Sprint 0** (now → July 10): Software + theory.
- [x] Repo structure
- [x] Skeleton modules
- [x] MNE preprocessing pipeline
- [x] EEGNet model
- [x] Training scripts (baseline, EEGNet, ablation)
- [x] Real-time pipeline (dummy + LSL)
- [x] Streamlit dashboard
- [ ] Run on public dataset (BCI Competition IV 2a)
- [ ] CSP baseline result
- [ ] EEGNet trained result
