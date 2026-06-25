# 基于运动想象的脑-机交互算法研究

> **XH-202610** · 挑战杯 2026 · `master` [![tests](https://img.shields.io/badge/tests-304%20passed-brightgreen)]()

基于运动想象（Motor Imagery）的脑-机接口算法研究与实时系统。使用 MNE-Python 预处理，结合 EEGNet、时空注意力与 Riemannian Geometry 方法，构建 8 通道跨被试 MI 识别、多方法对比评估与在线闭环原型系统。

**关键词**：运动想象 · EEGNet · 时空注意力 · Riemannian Geometry · EEG Conformer · FBCNet · 域泛化 · LOSO 交叉验证 · 在线闭环 · 空闲门控

---

## 快速开始

```bash
# 1. 环境
conda env create -f environment.yml && conda activate bci
pip install mne moabb streamlit pytest black ruff openpyxl

# 2. 下载公开数据
python data/download.py                    # PhysioNet MI (30 subjects)
python data/download.py --bci_iv_2a        # BCI Competition IV 2a (9 subjects)

# 3. PhysioNet MI 预处理 + LOSO
python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary
python main.py loso --data_dir data/loso_binary --epochs 60

# 4. BCI IV 2a 预处理 + LOSO
python preprocessing/prepare_bci_iv_2a.py
python main.py loso --data_dir data/bci_iv_2a_processed --n_subjects 9 --epochs 60 --dataset bci_iv_2a

# 5. 在线 Demo（文件回放模拟）
python main.py demo --source replay --data data/loso_binary/subj_01/X.npy \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt --gating
```

---

## 实验结果（LOSO 金标准，8ch）

### PhysioNet MI (30 subjects, 8ch, binary)

| 方法 | 类型 | LOSO Accuracy | Kappa |
|------|------|--------------|-------|
| EEGNet (base) | DL | 51.93% ± 7.20% | 0.033 |
| EEGNet + SpatiotemporalAttn | DL | 55.04% ± 7.86% | 0.096 |
| EEGNet + Few-shot FT (5 trials) | DL | 54.95% ± 8.04% | 0.099 |
| MDM + EA | Riemannian | 56.30% ± 10.94% | 0.129 |
| FgMDM + EA | Riemannian | **60.00%** ± 9.04% | 0.198 |
| **Tangent Space + LDA + EA** | **Riemannian** | **60.30%** ± 9.75% | **0.208** |

### BCI Competition IV 2a (9 subjects, 8ch, 4-class)

| 方法 | 类型 | LOSO Accuracy | Kappa |
|------|------|--------------|-------|
| MDM + EA | Riemannian | 33.43% ± 10.92% | 0.112 |
| FgMDM + EA | Riemannian | 34.91% ± 8.48% | 0.132 |
| EEGNet + SpatiotemporalAttn | DL | 36.94% ± 11.78% | 0.159 |
| Tangent Space + LDA + EA | Riemannian | 38.60% ± 12.44% | 0.181 |
| **EEGNet (base)** | **DL** | **39.47%** ± 12.45% | **0.193** |

> **关键发现**: Binary 任务 Tangent Space + LDA 达 60.30%（vs EEGNet base +8.37pp，vs 最优 DL 模型 +5.26pp）；MDM 56.30% 最弱；FgMDM ≈ Tangent（频带分解在 8ch 设置下无增益）。4-class 任务 EEGNet 39.47% 略优于 Tangent 38.60%。设置上限并非 55%，而是 **~60%**。

> Chance level: PhysioNet 50% (binary), BCI IV 2a 25% (4-class).  Random-split 结果（仅用于 pipeline 验证）：CSP+SVM 38.6%, EEGNet 53.8% (3-class).

---

## 环境

| 组件 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.5.1 |
| MNE-Python | 1.12+ |
| MOABB | 1.5+ |
| 目标硬件 | DeepBCI **8 通道** @ 250 Hz |

---

## 8 通道 Montage

对齐 DeepBCI 硬件，标准 10-20 系统运动皮层覆盖：

| 区域 | 通道 | 生理意义 |
|------|------|---------|
| 前运动区 | FC3, FC4 | 运动准备 / SMA |
| 中央运动区 | C3, Cz, C4 | 初级运动皮层 (MI 核心) |
| 中央-顶叶 | CP3, CPz, CP4 | 体感反馈 |

---

## 项目结构

```
├── data/
│   ├── raw/                        # 原始数据
│   │   ├── physionet_mi/           #   PhysioNet MI (30 subjects)
│   │   └── bci_iv_2a/              #   BCI Competition IV 2a (9 subjects)
│   ├── processed/                  # 预处理输出 (.npy) X=[N,8,750] y=[N]
│   └── download.py                 # 数据集自动下载
├── datasets/                       # ★ 数据集标签映射与元数据
│   ├── label_mapping.py            #   规范标签映射 (单点真理)
│   ├── metadata.py                 #   数据集元数据导出
│   └── deepbci_loader.py           #   DeepBCI 会话 → .npy 训练数据
├── features/                       # ★ 特征提取 (CSP + FBCSP + Riemannian)
│   ├── csp.py                      #   CSP/FBCSP + SVM/LDA
│   └── riemann.py                  # ★ Riemannian Geometry (Tangent Space / MDM / FgMDM)
├── preprocessing/
│   ├── run_mne_pipeline.py         # ★ PhysioNet MNE 预处理 (motor8/motor16/all)
│   ├── prepare_bci_iv_2a.py        # ★ BCI IV 2a MOABB .npy → 8ch per-subject
│   ├── filtering.py                #   带通 8-30Hz + 陷波 50Hz
│   ├── epoching.py                 #   事件分段
│   ├── artifact.py                 #   ICA 去伪迹
│   ├── mne_pipeline.py             #   编程 API
│   ├── alignment.py                #   Euclidean Alignment 跨被试协方差对齐
│   └── augment.py                  #   数据增强
├── models/
│   ├── eegnet.py                   # EEGNet (Lawhern 2018, lazy classifier)
│   ├── attention.py                # 注意力模块 (SE/MHSA/Temporal/Spatiotemporal)
│   ├── eegnet_attn.py              # EEGNet + 注意力 (5 变体 + 工厂函数)
│   ├── eeg_conformer.py            # EEG Conformer (CNN + Transformer, Song 2023)
│   ├── eeg_tcnet.py                # EEG-TCNet (CNN + TCN, Ingolfsson 2020)
│   ├── fbcnet.py                   # FBCNet (Filter-Bank CNN, Bakshi 2021)
│   ├── fusion.py                   # 多频段融合 (μ/β/full)
│   └── mixstyle.py                 # MixStyle 域泛化 (Zhou, ICLR 2021)
├── training/
│   ├── train_eegnet.py             # ★ 训练脚本 (增强/平滑/早停/裁剪/K折)
│   ├── train_baseline.py           #   CSP+SVM 基线
│   ├── train_ablation.py           #   消融实验
│   ├── train_sweep.py              #   超参搜索 (Optuna + 网格回退)
│   ├── train_loso.py               # ★ LOSO 交叉验证 (Few-shot FT, CSV/JSON 导出)
│   ├── train_riemann_loso.py       # ★ Riemannian LOSO (Tangent/MDM/FgMDM + EA)
│   ├── evaluate_subjectwise.py     #   逐被试评估
│   └── split.py                    #   数据分割工具
├── realtime/                       # ★ 实时推理管线
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
│   ├── config.py                   # ★ 全局配置 (8ch montage)
│   ├── metrics.py                  # 分类指标
│   ├── domain_adapt.py             # 域自适应损失 (Center Loss + MMD)
│   ├── report_excel.py             # Excel 验证报告 (5-Sheet 竞赛格式)
│   └── logger.py                   # 实验日志
├── ui/
│   └── dashboard.py                # Streamlit 实时看板 (Synthetic + File Replay)
├── scripts/
│   ├── run_all_experiments.py      # 全流程一键运行
│   ├── export_competition_excel.py # Excel 报告导出
│   └── make_report_figures.py      # 图表生成 (混淆矩阵/消融/逐被试)
├── tests/                          # ★ 304 个单元测试
├── main.py                         # 统一入口 (16 个命令)
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
python preprocessing/prepare_bci_iv_2a.py    # BCI IV 2a 8ch per-subject

# === 训练 ===
python main.py baseline                      # CSP+SVM
python main.py train --model eegnet_spatiotemporal --augment
python main.py ablation                      # 消融实验

# === LOSO 交叉验证 ★ ===
python main.py loso --data_dir data/loso_binary --epochs 60                          # PhysioNet MI (EEGNet)
python main.py loso --data_dir data/loso_binary --finetune 5                          # + few-shot FT
python main.py loso --data_dir data/bci_iv_2a_processed --n_subjects 9 --dataset bci_iv_2a  # BCI IV 2a
python main.py riemann --method tangent --align                      # Riemannian LOSO (最强基线)
python main.py riemann --method mdm --align                          # MDM
python main.py riemann --method fgmdm --align                        # Filter-bank Riemannian

# === 在线 Demo ===
python main.py demo                                    # 合成数据 (100 steps)
python main.py demo --source replay --data data/loso_binary/subj_01/X.npy \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt             # 文件回放
python main.py demo --source replay --data data/loso_binary/subj_01/X.npy \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt --gating    # + 空闲门控
python main.py demo --all-subjects --source replay \
    --checkpoint checkpoints/eegnet_spatiotemporal_best.pt             # 全被试批量

# === 实时 ===
python main.py dashboard             # Streamlit 看板 (端口 8501)
python main.py record                # DeepBCI 数据采集 (交互式)

# === 工具 ===
python main.py metadata              # 导出数据集元数据
python main.py subjectwise           # 逐被试评估
python main.py run_all               # 一键全流程
python main.py export                # 导出 Excel 报告
python main.py figures               # 生成报告图表

# === DeepBCI 数据 ===
python datasets/deepbci_loader.py -i data/subjects/sub_001/session_xxx   # 单会话
python datasets/deepbci_loader.py --all                                  # 所有会话

# === 质量 ===
pytest tests/ -v                     # 304 个测试
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
  → 分段 (-0.5s ~ 2.5s 相对 cue)
  → 导出 X=[N, 8, 750] float32, y=[N] int64
```

> **注意**: 预处理阶段使用 8–30 Hz 带通。FBCSP/FBCNet/FgMDM 默认的 4–40 Hz 9 频带中，4–8 Hz、32–36 Hz、36–40 Hz 落在通带外。实验表明频带分解在此 8ch 设置下无显著增益（FgMDM ≈ Tangent Space），因此**不推荐**将 FBCSP_BANDS 作为默认滤波器组 — 使用全频段协方差（Tangent Space）或仅限 8–30 Hz 内子带。

---

## 模型

### EEGNet + 时空注意力

```
Input (B, 1, C, T)
  → Block 1: Temporal Conv(1×64) → DepthwiseConv(C×1) → Pool
  → [★ SpatiotemporalAttention: MHSA(channels) → TemporalAttention(time)]
  → Block 2: Separable Conv → Pool
  → Linear → N classes
```

### 注意力变体

| 类型 | 模块 | 说明 |
|------|------|------|
| `se` | `ChannelAttention1D` | Squeeze-and-Excitation |
| `mhsa` | `MultiHeadChannelAttention` | 多头自注意力 |
| `temporal` | `TemporalAttention` | 时域点加权 |
| `spatiotemporal` | `SpatiotemporalAttention` | MHSA(ch) + Temporal(t) 串联 |

### 创新点

1. **时空注意力深度融合** — 在 EEGNet 的 Block1/Block2 之间插入，比独立注意力模块更有效
2. **空闲状态置信度门控** — 在线推理时自动过滤低置信度和 IDLE 预测，防止误触发
3. **少样本在线校准** — LOSO + Few-shot FT，用目标被试 5 个 trial 即可提升 ~3pp

### 扩展模型库 (Sprint 1.5)

| 模型 | 论文 | 特点 |
|------|------|------|
| `EEGConformer` | Song et al. 2023 | CNN 骨干 + Transformer Encoder，小样本 MI 优化 |
| `EEGTCNet` | Ingolfsson et al. 2020 | EEGNet Block1 + TCN 时序卷积，适合嵌入式部署 |
| `FBCNet` | Bakshi et al. 2021 | 多频段滤波 + 逐频段空间卷积 + 方差池化 |
| `MixStyle` | Zhou et al. (ICLR 2021) | 特征统计混合域泛化，无需域标签 |

### Riemannian Geometry 基线 ★

| 方法 | 说明 | 论文 |
|------|------|------|
| **Tangent Space + LDA** | SPD 协方差矩阵 → 切空间映射 → 线性分类 | Barachant et al. (IEEE TBME, 2012) |
| **MDM** | 流形上的最小黎曼距离分类 | Congedo et al. (BCI, 2017) |
| **FgMDM** | 多频带滤波 + 每带切空间 + LDA | 类似 Ang et al. (2008) 的频带分解思路 |

Riemannian 方法是 MI-BCI 领域公认的最强传统基线。在当前 8ch binary LOSO 设置下，Tangent Space + LDA + EA 达 60.30%，相比 EEGNet base 提升 8.37pp，相比最优深度模型（EEGNet + SpatiotemporalAttn）提升 5.26pp。使用 `pyriemann` 库实现，提供与 `csp.py` 一致的 sklearn Pipeline API。

### 域泛化与自适应

| 技术 | 来源 | 说明 |
|------|------|------|
| **Euclidean Alignment (EA)** | He & Wu 2018 | 跨被试协方差对齐，无监督，计算轻量；与 Riemannian 协同使用 |
| **Center Loss** | Wen et al. 2016 | 同类特征向中心收缩，隐式减少被试间散度 |
| **MMD Loss** | Gretton et al. 2012 | 多核 RBF 最大均值差异，显式对齐特征分布 |
| **MixStyle** | Zhou et al. 2021 | 实例级均值/方差混合，模拟域偏移 |

---

## 验证方法论

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| **Random split** | 混洗所有 trial 后 `train_test_split` | 快速 debug, 算法迭代 |
| **LOSO** ⭐ | N-1 人训练, 1 人测试, 轮 N 次 | **论文/比赛报告, 泛化性评估** |
| **LOSO + Few-shot FT** | LOSO 后用目标被试少量 trial 微调 | 在线系统标定 |

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

| # | 文件 | 要求 | 状态 |
|---|------|------|------|
| 1 | PDF 技术报告 | 算法设计 + 系统实现 + 实验分析 | ⏳ |
| 2 | Excel 验证数据 | 标准数据集性能 | ✅ 生成器就绪 |
| 3 | MP4 演示视频 | 实时在线系统全流程 | 等硬件 |
| 4 | 源码 | 完整可运行代码 | ✅ |

---

## 参考文献

- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *J. Neural Eng.*, 15(5). [DOI:10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
- Barachant, A., et al. (2012). Multiclass Brain-Computer Interface Classification by Riemannian Geometry. *IEEE Trans. Biomed. Eng.*, 59(4).
- Congedo, M., et al. (2017). Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review. *Brain-Computer Interfaces*, 4(3).
- He, H. & Wu, D. (2018). Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach. *arXiv:1808.05464*.
- Ang, K. K., et al. (2008). Filter Bank Common Spatial Pattern (FBCSP). *Int. Joint Conf. Neural Networks*.
- Schalk, G., et al. (2004). BCI2000. *IEEE Trans. Biomed. Eng.*
- PhysioNet: [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/)
- BCI Competition IV 2a: [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets)
- MNE-Python: [mne.tools](https://mne.tools)
- MOABB: [Mother of All BCI Benchmarks](https://github.com/NeuroTechX/moabb)
- LSL: [labstreaminglayer.org](https://labstreaminglayer.org)
- pyriemann: [github.com/pyRiemann/pyRiemann](https://github.com/pyRiemann/pyRiemann)
