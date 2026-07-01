# 基于运动想象的脑-机交互算法研究

> **XH-202610** · 挑战杯 2026 · `master` [![tests](https://img.shields.io/badge/tests-350%20passed-brightgreen)]()

面向 8 通道 DeepBCI 的少通道运动想象 BCI 多方法对比与在线闭环系统。使用 MNE-Python 预处理，对比 EEGNet、EEG Conformer、EEG-TCNet、FBCNet、SPDNet 等深度模型与 Riemannian Geometry 传统强基线，构建 8 通道跨被试 MI 识别、消融实验与在线闭环原型。

**关键词**：运动想象 · EEGNet · EEG Conformer · EEG-TCNet · SPDNet · 时空注意力 · Riemannian Geometry · FBCNet · 域泛化 · LOSO 交叉验证 · EUCLIDEAN ALIGNMENT · 在线闭环 · 空闲门控

---

## 快速开始

```bash
# 1. 环境
conda env create -f environment.yml && conda activate bci
pip install mne moabb streamlit pytest black ruff openpyxl pandas

# 2. 下载公开数据
python data/download.py                    # PhysioNet MI (30 subjects)
python data/download.py --bci_iv_2a        # BCI Competition IV 2a (9 subjects)

# 3. PhysioNet MI 预处理 + LOSO
python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary
python main.py loso --data_dir data/loso_binary --epochs 80 --align

# 4. BCI IV 2a 预处理 + LOSO
python preprocessing/prepare_bci_iv_2a.py
python main.py loso --data_dir data/bci_iv_2a_processed --n_subjects 9 --epochs 60 --dataset bci_iv_2a

# 5. 在线 Demo（文件回放模拟）
python main.py demo --source replay --data data/loso_binary/subj_01/X.npy \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt --gating
```

---

## 实验结果（LOSO 金标准，8ch）

### PhysioNet MI (30 subjects, 8ch, binary LOSO, 80 epochs for DL, 60 for SPDNet)

| Rank | 方法 | 类型 | Accuracy | Kappa |
|------|------|------|----------|-------|
| 1 | **EEG Conformer + EA** | DL + Transformer | **63.93%** ± 9.58% | 0.277 |
| 2 | **EEG-TCNet + EA** | DL + TCN | **63.41%** ± 10.51% | 0.265 |
| 3 | **FBCNet + EA** | DL + Filter Bank | **61.11%** ± 11.69% | 0.219 |
| 4 | Tangent Space + LDA + EA | Riemannian | 60.44% ± 9.64% | 0.212 |
| 5 | Tangent Space + LDA | Riemannian | 60.44% ± 9.64% | 0.212 |
| 6 | FgMDM + EA (6-band, 8–30Hz) | Riemannian | 59.18% ± 8.12% | 0.180 |
| 7 | **SPDNet + EA** (seed42) | DL + SPD Manifold | **58.52%** ± 8.38% | 0.161 |
| 8 | EEGNet + EA | DL | 58.00% ± 10.06% | 0.161 |
| 9 | EEGNet + SpatiotemporalAttn + EA | DL + Attention | 57.78% ± 8.55% | 0.158 |
| 10 | MDM + EA | Riemannian | 56.22% ± 10.52% | 0.127 |
| 11 | MAA-EEGNet-Pre + EA ⚠️ | DL + MAA | 56.00% ± 9.42% | 0.121 |
| 12 | MAA-EEGNet + EA ⚠️ | DL + MAA | 55.33% ± 8.73% | 0.102 |
| 13 | EEGNet + SpatiotemporalAttn | DL + Attention | 55.04% ± 7.86% | 0.096 |
| 14 | FB-MAA-EEGNet + EA ⚠️ | DL + FB + MAA | 53.78% ± 7.68% | 0.070 |
| 15 | EEGNet (no EA) | DL | 51.93% ± 7.20% | 0.033 |
| 16 | SPDNet (no EA) | DL + SPD Manifold | 50.59% ± 1.87% | 0.000 |
| 17 | FBCNet (no EA) | DL + Filter Bank | 49.70% ± 2.66% | -0.010 |

> **核心发现**: 深度模型可以超越传统 Riemannian 基线。EEG Conformer + EA（63.93%）和 EEG-TCNet + EA（63.41%）均超过 Tangent Space + LDA + EA（60.44%）。SPDNet + EA（58.52%）首次在 8ch MI 上验证了 SPD 流形深度学习的可行性，与 EEGNet + EA（58.00%）竞争。EA 增益与架构内部归一化强度负相关：FBCNet（无归一化）+11.41pp → SPDNet（无归一化）+7.93pp → EEGNet（BN）+6.07pp → Conformer/TCNet（LN+残差）+1.8~2.6pp → Tangent（仿射不变）±0pp。详细分析见 [EA × Architecture Interaction Analysis](docs/ea_analysis.md)。

### SPDNet 消融实验

| 配置 | Accuracy | Kappa |
|------|----------|-------|
| SPDNet + EA (d=[8,8], seed42) | **58.52%** ± 8.38% | 0.161 |
| SPDNet + EA (d=[8,8], seed123) | 56.96% ± 9.52% | 0.133 |
| SPDNet + EA (d=[8,8], seed456) | 55.93% ± 8.83% | 0.112 |
| SPDNet + EA (d=[8,8], default) | 50.44% ± 4.59% | 0.020 |
| SPDNet (no EA) | 50.59% ± 1.87% | 0.000 |

> SPDNet 训练对随机种子敏感；受控种子下稳定，默认种子可能 collapse 到多数类。无 EA 时所有 SPDNet 均 collapse（κ≈0），EA 对 SPDNet 比 CNN 更加关键。

### Few-shot Calibration

| Calibration Trials/Class | EEG-Conformer + EA |
|--------------------------|-------------------|
| 0 (pure LOSO) | 65.33% |
| 5 | 66.38% |
| 10 | **67.47%** |
| 20 | 66.38% |
| 40 | 66.67% |

> 10 trials/class 的少样本校准带来额外 +2.14pp，最终达 67.47%。

### Riemannian 协方差估计器与度量 Sweep

| cov_estimator | Accuracy | metric | Accuracy |
|---------------|----------|--------|----------|
| scm (样本协方差) | **60.44%** | riemann | **60.44%** |
| lwf (Ledoit-Wolf) | 60.30% | wasserstein | 60.44% |
| oas (Oracle Approx.) | 60.23% | logeuclid | 59.70% |
| mcd (鲁棒) | — | logchol | 59.25% |

> 收缩估计器对 8 通道小样本无明显增益。scm 已是最优选择。

### BCI Competition IV 2a (9 subjects, 8ch, 4-class LOSO)

| 方法 | 类型 | Accuracy | Kappa |
|------|------|----------|-------|
| **EEGNet (base)** | DL | **39.47%** ± 12.45% | 0.193 |
| Tangent Space + LDA + EA | Riemannian | 38.60% ± 12.44% | 0.181 |
| EEGNet + SpatiotemporalAttn | DL | 36.94% ± 11.78% | 0.159 |
| FgMDM + EA | Riemannian | 34.91% ± 8.48% | 0.132 |
| MDM + EA | Riemannian | 33.43% ± 10.92% | 0.112 |

> Chance level: PhysioNet 50% (binary), BCI IV 2a 25% (4-class).

---

## EA 增益分析（关键发现）

Euclidean Alignment 是当前 8ch LOSO 任务中最强的单一技术。但其增益**高度依赖架构**：

| 模型 | 无 EA | + EA | 增益 | 内部归一化 |
|------|-------|------|------|-----------|
| FBCNet | 49.70% | 61.11% | **+11.41pp** | ❌ 无 |
| SPDNet | 50.59% | 58.52% | **+7.93pp** | ❌ 无（SPD 流形层） |
| EEGNet | 51.93% | 58.00% | **+6.07pp** | BatchNorm |
| SpatiotemporalAttn | 55.04% | 57.78% | +2.74pp | Attention 重加权 |
| EEG-Conformer | 61.33% | 63.93% | +2.60pp | LayerNorm + 残差 |
| EEG-TCNet | 61.56% | 63.41% | +1.85pp | BatchNorm + 残差 |
| Tangent Space | 60.44% | 60.44% | ±0.00pp | 仿射不变 |

**机制**: EA 将跨被试 Riemannian 协方差距离减少 **85.1%**。FBCNet 的方差池化层和 SPDNet 的 SPD 流形层直接读取协方差结构，因此受益最大；TCN/Transformer 的内部归一化（LayerNorm/BatchNorm + 残差）已消除大部分分布偏移，EA 边际增益小。

### 跨数据集验证

EA 增益**并非纯架构属性**，还受数据集特征调节：

| 模型 | PhysioNet EA增益 | BCI IV 2a EA增益 |
|------|-----------------|-----------------|
| EEGNet | +6.07pp | **-1.54pp** ⚠️ |
| EEG-TCNet | +1.85pp | +0.45pp |
| EEG-Conformer | +2.60pp | +0.05pp |
| FBCNet | +11.41pp | **+1.35pp** |

BCI IV 2a（9 subjects, 576 trials/subject, 4-class）上 EA 增益几乎为零甚至为负。PhysioNet（30 subjects, ~45 trials/subject）被试间变异性更大，EA 收益更显著。

```text
EA 增益 = f(架构内部归一化 × 数据集被试间变异性)
```

> 详细分析见 [`docs/ea_analysis.md`](docs/ea_analysis.md) — 包含双数据集对比、修正后的双因素模型、设计原则。

---

## 8 通道 Montage

对齐 DeepBCI 硬件，标准 10-20 系统运动皮层覆盖：

| 区域 | 通道 | Index | 生理意义 |
|------|------|-------|---------|
| 前运动区 | FC3, FC4 | 0, 4 | 运动准备 / SMA |
| 中央运动区 | C3, Cz, C4 | 1, 2, 3 | 初级运动皮层 (MI 核心) |
| 中央-顶叶 | CP3, CPz, CP4 | 5, 6, 7 | 体感反馈 |

---

## 环境

| 组件 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.5.1 |
| MNE-Python | 1.12+ |
| MOABB | 1.5+ |
| pyriemann | 0.7+ |
| 目标硬件 | DeepBCI **8 通道** @ 250 Hz |

---

## 项目结构

```
├── data/
│   ├── raw/                        # 原始数据
│   │   ├── physionet_mi/           #   PhysioNet MI (30 subjects)
│   │   └── bci_iv_2a/              #   BCI Competition IV 2a (9 subjects)
│   ├── processed/                  # 预处理输出 (.npy) X=[N,8,750] y=[N]
│   ├── loso_binary/                # LOSO 逐被试数据 (30 folders)
│   └── download.py                 # 数据集自动下载
├── datasets/                       # 数据集标签映射与元数据
│   ├── label_mapping.py            #   规范标签映射 (单点真理)
│   ├── metadata.py                 #   数据集元数据导出
│   └── deepbci_loader.py           #   DeepBCI 会话 → .npy 训练数据
├── features/                       # 特征提取
│   ├── csp.py                      #   CSP/FBCSP + SVM/LDA
│   ├── riemann.py                  #   Riemannian Geometry (Tangent/MDM/FgMDM)
│   ├── spd_covariance.py           # ★ SPD 协方差矩阵计算 (SCM/LWF/多频段)
│   └── bandpower.py                #   频带功率特征
├── preprocessing/
│   ├── run_mne_pipeline.py         #   PhysioNet MNE 预处理 (+ --tmin/--tmax)
│   ├── prepare_bci_iv_2a.py        #   BCI IV 2a MOABB → 8ch per-subject
│   ├── filtering.py                #   带通 8–30Hz + 陷波 50Hz
│   ├── epoching.py                 #   事件分段
│   ├── artifact.py                 #   ICA 去伪迹
│   ├── mne_pipeline.py             #   编程 API
│   ├── alignment.py                #   Euclidean Alignment (跨被试协方差对齐)
│   └── augment.py                  #   数据增强
├── models/
│   ├── eegnet.py                   #   EEGNet (Lawhern 2018, lazy classifier)
│   ├── attention.py                #   注意力模块 (SE/MHSA/Temporal/Spatiotemporal)
│   ├── eegnet_attn.py              #   EEGNet + 注意力 (5 变体 + 工厂函数)
│   ├── eeg_conformer.py            #   EEG Conformer (CNN + Transformer, Song 2023)
│   ├── eeg_tcnet.py                #   EEG-TCNet (CNN + TCN, Ingolfsson 2020)
│   ├── fbcnet.py                   #   FBCNet (Filter-Bank CNN, Bakshi 2021)
│   ├── fb_tcnet.py                 # ★ FB-TCNet (Filter Bank + TCN, 本项目原创)
│   ├── spd_models.py               # ★ SPDNet (SPD 流形 DL, Huang & Van Gool 2017)
│   ├── motor_area_attention.py     #   Motor-Area Attention (8ch 区域分组注意力)
│   ├── fb_maa_eegnet.py            #   FB-MAA-EEGNet (Filter Bank + MAA + EEGNet)
│   ├── maa_eegnet.py               #   MAA-EEGNet (MAA after temporal conv)
│   ├── maa_eegnet_pre.py           #   MAA-EEGNet-Pre (MAA before temporal conv)
│   ├── fusion.py                   #   多频段融合 (μ/β/full)
│   └── mixstyle.py                 #   MixStyle 域泛化 (Zhou, ICLR 2021)
├── training/
│   ├── train_eegnet.py             #   训练脚本 (增强/平滑/早停/裁剪/K折)
│   ├── train_baseline.py           #   CSP+SVM 基线
│   ├── train_ablation.py           #   消融实验
│   ├── train_sweep.py              #   超参搜索 (Optuna + 网格回退)
│   ├── train_loso.py               #   LOSO 交叉验证 (Few-shot FT, CSV/JSON)
│   ├── train_riemann_loso.py       #   Riemannian LOSO (cov/metric/band sweep)
│   ├── train_spd_loso.py           # ★ SPDNet LOSO (SPD 协方差 + EA)
│   ├── evaluate_subjectwise.py     #   逐被试评估
│   └── split.py                    #   数据分割
├── realtime/                       # 实时推理管线
│   ├── sources.py                  #   EEGSource Protocol (统一接口)
│   ├── stream.py                   #   DummyStream (合成数据)
│   ├── stream_lsl.py               #   LSLStream (真实设备)
│   ├── file_replay.py              #   FileReplaySource (.npy 文件回放)
│   ├── deepbci_stream.py           #   DeepBCIStream (多模式)
│   ├── deepbci_recorder.py         #   DeepBCIRecorder (会话录制 + 元数据)
│   ├── deepbci_protocol.py         #   MIProtocol (实验时序)
│   ├── deepbci_source.py           #   DeepBCISource 预留骨架
│   ├── buffer.py                   #   RingBuffer (线程安全环形缓冲)
│   └── inference.py                #   MIInference (推理 + 空闲门控)
├── utils/
│   ├── config.py                   #   全局配置 (8ch montage, FBCSP_BANDS)
│   ├── metrics.py                  #   分类指标
│   ├── domain_adapt.py             #   域自适应损失 (Center Loss + MMD)
│   ├── report_excel.py             #   Excel 验证报告 (5-Sheet 竞赛格式)
│   └── logger.py                   #   实验日志
├── ui/
│   └── dashboard.py                #   Streamlit 实时看板
├── scripts/
│   ├── run_all_experiments.py      #   全流程一键运行
│   ├── export_competition_excel.py #   Excel 报告导出
│   ├── make_report_figures.py      #   图表生成 (混淆矩阵/消融/逐被试)
│   ├── run_time_window_sweep.py    #   时间窗 sweep 实验
│   ├── run_ablation_all.py         #   10 配置消融实验
│   ├── analyze_ea_effects.py       # ★ EA × 架构交互分析
│   ├── analyze_spdnet_vs_tangent.py  # ★ SPDNet vs Tangent 对比分析
│   └── make_paper_figures.py       # ★ 论文图表生成 (6 figures + 统计检验)
├── docs/                           # ★ 研究文档与论文草稿
│   ├── ea_analysis.md              #   EA 增益双因素分析
│   ├── paper_draft.md              #   论文草稿
│   ├── TECHNICAL_REPORT.md         #   技术报告
│   ├── research_proposal.md        #   研究计划
│   ├── neural_sde_research_plan.md #   Neural SDE 研究计划
│   └── spd_ssl_research_plan.md    #   SPD SSL 研究计划
├── tests/                          # 350 个单元测试 (29 文件)
├── main.py                         # 统一入口 (18 个命令)
├── environment.yml
└── CLAUDE.md
```

---

## 全部命令

```bash
# === 数据 ===
python data/download.py                      # PhysioNet MI (默认)
python data/download.py --bci_iv_2a          # BCI IV 2a
python data/download.py --sample             # MNE 示例数据

# === 预处理 ===
python main.py preprocess                    # PhysioNet MI 8ch 三分类
python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary
python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --tmin 0.5 --tmax 3.0 --output data/loso_window
python preprocessing/prepare_bci_iv_2a.py    # BCI IV 2a 8ch per-subject

# === 训练 ===
python main.py baseline                      # CSP+SVM
python main.py train --model eegnet_spatiotemporal --augment
python main.py ablation                      # 消融实验

# === LOSO 交叉验证 ★ ===
# DL 模型
python main.py loso --data_dir data/loso_binary --epochs 80 --align                     # EEGNet + EA
python main.py loso --data_dir data/loso_binary --model eeg_conformer --epochs 80 --align # Conformer 🏆
python main.py loso --data_dir data/loso_binary --model eeg_tcnet --epochs 80 --align     # TCNet
python main.py loso --data_dir data/loso_binary --model fbcnet --epochs 80 --align        # FBCNet
python main.py loso --data_dir data/loso_binary --model fb_maa_eegnet --epochs 80 --align # FB-MAA-EEGNet
python main.py loso --data_dir data/loso_binary --model maa_eegnet --epochs 80 --align    # MAA-EEGNet
# Few-shot calibration
python main.py loso --data_dir data/loso_binary --model eeg_conformer --epochs 80 --align --finetune_sweep 0,5,10,20,40

# SPDNet (SPD 流形深度学习)
python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align
python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align --cov_estimator lwf
python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align --multiband

# Riemannian 方法
python main.py riemann --method tangent --align                            # Tangent + LDA + EA
python main.py riemann --method tangent --align --cov_estimator lwf        # Ledoit-Wolf 收缩
python main.py riemann --method tangent --align --metric logeuclid         # Log-Euclidean 度量
python main.py riemann --method fgmdm --align --bands 8 12 12 16 16 20 20 24 24 28 28 30  # FgMDM 6-band
python main.py riemann --method mdm --align                                # MDM

# BCI IV 2a
python main.py loso --data_dir data/bci_iv_2a_processed --n_subjects 9 --dataset bci_iv_2a

# === 在线 Demo ===
python main.py demo                                    # 合成数据
python main.py demo --source replay --data data/loso_binary/subj_01/X.npy \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt --gating

# === 实时 ===
python main.py dashboard             # Streamlit 看板 (端口 8501)
python main.py record                # DeepBCI 数据采集

# === 实验脚本 ===
python scripts/run_time_window_sweep.py --method tangent --align     # 时间窗 sweep
python scripts/run_ablation_all.py --epochs 80                        # 10 配置消融
python scripts/analyze_ea_effects.py --data_dir data/loso_binary      # EA 效果分析
python scripts/analyze_spdnet_vs_tangent.py                           # SPDNet vs Tangent
python scripts/make_paper_figures.py                                  # 论文图表 (6 figs)

# === 工具 ===
python main.py metadata              # 数据集元数据
python main.py run_all               # 一键全流程
python main.py export                # Excel 报告
python main.py figures               # 报告图表

# === 质量 ===
pytest tests/ -v                     # 350 个测试
black . && ruff check .              # 格式化 + Lint
```

---

## 预处理流程

```
原始 EEG (64ch, 160 Hz)
  → 重采样至 250 Hz
  → 选择 8 运动皮层通道 (FC3, C3, Cz, C4, FC4, CP3, CPz, CP4)
  → 带通滤波 8–30 Hz (FIR)
  → 陷波 50 Hz
  → CAR (共平均参考)
  → 分段 (tmin~tmax 相对 cue, 默认 -0.5~2.5s)
  → 导出 X=[N, 8, 750] float32, y=[N] int64
```

> **频带一致性**: 预处理使用 8–30 Hz 带通。FBCSP_BANDS 已更新为 8–30 Hz 内的 6 个子频带 `[(8,12), (12,16), (16,20), (20,24), (24,28), (28,30)]`，与预处理通带对齐，避免外频带能量不足。

---

## 模型

### 模型库总览

| 模型 | 类型 | 论文 | 特点 |
|------|------|------|------|
| `eegnet` | DL | Lawhern et al. 2018 | 紧凑 CNN，时序+空域可分离卷积 |
| `eegnet_spatiotemporal` | DL + Attention | — | EEGNet + 时空注意力 (MHSA + Temporal) |
| `eeg_conformer` | DL + Transformer | Song et al. 2023 | CNN 骨干 + Transformer Encoder |
| `eeg_tcnet` | DL + TCN | Ingolfsson et al. 2020 | EEGNet Block1 + 时序卷积网络 |
| `fbcnet` | DL + Filter Bank | Bakshi et al. 2021 | 多频段 + 逐频段空域卷积 + 方差池化 |
| `fb_tcnet` | DL + FB + TCN | 本项目 (2026) | Filter Bank + TCN 时序建模 + EA |
| `spdnet` | DL + SPD Manifold | Huang & Van Gool 2017 | SPD 流形上的 BiMap/ReEig/LogEig 层 |
| `fb_maa_eegnet` | DL + FB + MAA | 本项目 | Filter Bank + 运动区注意力 + EEGNet |
| `maa_eegnet` | DL + MAA | 本项目 | MAA (temporal conv 后) + EEGNet |
| `maa_eegnet_pre` | DL + MAA | 本项目 | MAA (原始 EEG 预处理) + EEGNet |
| `Tangent Space + LDA` | Riemannian | Barachant et al. 2012 | SPD → 切空间 → 线性分类 |
| `MDM` | Riemannian | Congedo et al. 2017 | 流形最小黎曼距离 |
| `FgMDM` | Riemannian | — | 多频带 + 每带切空间 + 融合 |

### EEGNet + 时空注意力

```
Input (B, 1, C, T)
  → Block 1: Temporal Conv(1×64) → DepthwiseConv(C×1) → Pool
  → [★ SpatiotemporalAttention: MHSA(channels) → TemporalAttention(time)]
  → Block 2: Separable Conv → Pool
  → Linear → N classes
```

### SPDNet (SPD 流形深度学习)

```
Input (B, C, T) raw EEG
  → compute_covariance → (B, C, C) SPD matrices
  → BiMap (C→d1) → ReEig → BiMap (d1→d2) → ReEig → LogEig
  → Upper triangle flatten → (B, d2*(d2+1)/2)
  → Linear → N classes
```

SPDNet 直接在 SPD 流形上学习，核心层：
- **BiMap**: 双线性映射 `W @ C @ W.T`，降维并保持 SPD 结构
- **ReEig**: 特征值整流（SPD 流形上的非线性激活）
- **LogEig**: 矩阵对数（SPD 流形 → 切空间 / 欧氏空间）

### Riemannian Geometry 基线

| 方法 | 说明 |
|------|------|
| **Tangent Space + LDA** | SPD 协方差矩阵 → 切空间映射 → 线性判别分析 |
| **MDM** | SPD 流形上的黎曼距离最小均值分类 |
| **FgMDM** | 多频带滤波 + 每带切空间 + 特征拼接 + LDA |

### 域泛化与自适应

| 技术 | 来源 | 说明 |
|------|------|------|
| **Euclidean Alignment (EA)** | He & Wu 2018 | 跨被试协方差对齐，无监督，**+3~11pp DL 增益** |
| Center Loss | Wen et al. 2016 | 同类特征收缩 |
| MMD Loss | Gretton et al. 2012 | 多核最大均值差异 |
| MixStyle | Zhou et al. 2021 | 实例级均值/方差混合 |

---

## 验证方法论

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| **LOSO** ⭐ | N-1 人训练, 1 人测试, 轮 N 次 | **论文/比赛报告** |
| **LOSO + EA** | 每折内用训练被试计算 EA 参考矩阵 | **跨被试泛化** |
| **LOSO + Few-shot FT** | LOSO 后用目标被试少量 trial 微调 | 在线系统校准 |

---

## 实时系统架构

```
DataSource (EEGSource Protocol)
  ├── DummyStream          # 合成随机数据
  ├── FileReplaySource     # .npy 文件回放 (离线模拟)
  ├── LSLStream            # Lab Streaming Layer
  └── DeepBCISource        # DeepBCI 硬件 (预留)
       ↓
  RingBuffer (线程安全, 2s 滑动窗口)
       ↓
  MIInference
  ├── predict()            # 原始预测
  └── predict_with_gating() # + 空闲门控 + 置信度阈值
       ↓
  Action: STOP / LEFT / RIGHT
```

---

## 交付物

| # | 文件 | 状态 |
|---|------|------|
| 1 | PDF 技术报告 | ⏳ 草稿完成 |
| 2 | Excel 验证数据 | ✅ |
| 3 | MP4 演示视频 | 等硬件 |
| 4 | 源码 | ✅ |

---

## 参考文献

- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *J. Neural Eng.*, 15(5). [DOI:10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
- Song, Y., et al. (2023). EEG Conformer: Convolutional Transformer for EEG Decoding. *arXiv:2301.05578*.
- Ingolfsson, T. M., et al. (2020). EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded Motor-Imagery BCI. *IEEE SMC*.
- Bakshi, K., et al. (2021). FBCNet: A Multi-view Convolutional Neural Network for BCI. *arXiv:2104.01233*.
- Huang, Z. & Van Gool, L. (2017). A Riemannian Network for SPD Matrix Learning. *AAAI*.
- Barachant, A., et al. (2012). Multiclass Brain-Computer Interface Classification by Riemannian Geometry. *IEEE Trans. Biomed. Eng.*, 59(4).
- Congedo, M., et al. (2017). Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. *Brain-Computer Interfaces*, 4(3).
- He, H. & Wu, D. (2018). Transfer Learning for BCI: A Euclidean Space Data Alignment Approach. *arXiv:1808.05464*.
- Ang, K. K., et al. (2008). Filter Bank Common Spatial Pattern (FBCSP). *Int. Joint Conf. Neural Networks*.
- Zhou, K., et al. (2021). MixStyle: Domain Generalization via Feature Statistics Mixing. *ICLR*.
- PhysioNet: [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/)
- BCI Competition IV 2a: [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets)
- MNE-Python: [mne.tools](https://mne.tools)
- MOABB: [github.com/NeuroTechX/moabb](https://github.com/NeuroTechX/moabb)
- pyriemann: [github.com/pyRiemann/pyRiemann](https://github.com/pyRiemann/pyRiemann)
