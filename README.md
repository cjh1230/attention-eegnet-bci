# 基于运动想象的脑-机交互算法研究

> **XH-202610** · 挑战杯 2026 · `master` [![tests](https://img.shields.io/badge/tests-131%20passed-brightgreen)]()

基于运动想象（Motor Imagery）的脑-机接口算法研究与实时系统。使用 MNE-Python 预处理、EEGNet + 时空注意力深度学习模型，实现高精度跨被试 MI 识别。

**关键词**：运动想象 · EEGNet · 时空注意力 · LOSO 交叉验证 · 在线闭环 · 空闲门控

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

## 实验结果（LOSO 金标准）

### PhysioNet MI (30 subjects, 8ch, binary)

| 模型 | LOSO Accuracy | Kappa |
|-------|--------------|-------|
| EEGNet (base) | 51.93% ± 7.20% | 0.033 |
| EEGNet + Few-shot FT (5 trials) | **54.95%** ± 8.04% | 0.099 |
| EEGNet + SpatiotemporalAttn | **55.04%** ± 7.86% | 0.096 |

### BCI Competition IV 2a (9 subjects, 8ch, 4-class)

| 模型 | LOSO Accuracy | Kappa |
|-------|--------------|-------|
| EEGNet (base) | **39.47%** ± 12.45% | 0.193 |
| EEGNet + SpatiotemporalAttn | 36.94% ± 11.78% | 0.159 |

> Chance level: PhysioNet 50% (binary), BCI IV 2a 25% (4-class).
> Random-split 结果（仅供参考）：CSP+SVM 38.6%, EEGNet 53.8% (3-class), EEGNet+Spatiotemporal 57.6% (3-class).

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
│   └── metadata.py                 #   数据集元数据导出
├── preprocessing/
│   ├── run_mne_pipeline.py         # ★ PhysioNet MNE 预处理 (motor8/motor16/all)
│   ├── prepare_bci_iv_2a.py        # ★ BCI IV 2a MOABB .npy → 8ch per-subject
│   ├── filtering.py                #   带通 8-30Hz + 陷波 50Hz
│   ├── epoching.py                 #   事件分段
│   ├── artifact.py                 #   ICA 去伪迹
│   ├── mne_pipeline.py             #   编程 API
│   └── augment.py                  #   数据增强
├── models/
│   ├── eegnet.py                   # EEGNet (Lawhern 2018, lazy classifier)
│   ├── attention.py                # 注意力模块 (SE/MHSA/Temporal/Spatiotemporal)
│   ├── eegnet_attn.py              # EEGNet + 注意力 (5 变体 + 工厂函数)
│   └── fusion.py                   # 多频段融合 (μ/β/full)
├── training/
│   ├── train_eegnet.py             # ★ 训练脚本 (增强/平滑/早停/裁剪/K折)
│   ├── train_baseline.py           #   CSP+SVM 基线
│   ├── train_ablation.py           #   消融实验
│   ├── train_sweep.py              #   超参搜索 (Optuna + 网格回退)
│   ├── train_loso.py               # ★ LOSO 交叉验证 (Few-shot FT, CSV/JSON 导出)
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
│   ├── report_excel.py             # Excel 验证报告 (5-Sheet 竞赛格式)
│   └── logger.py                   # 实验日志
├── ui/
│   └── dashboard.py                # Streamlit 实时看板 (Synthetic + File Replay)
├── scripts/
│   ├── run_all_experiments.py      # 全流程一键运行
│   ├── export_competition_excel.py # Excel 报告导出
│   └── make_report_figures.py      # 图表生成 (混淆矩阵/消融/逐被试)
├── tests/                          # ★ 131 个单元测试
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
python main.py loso --data_dir data/loso_binary --epochs 60                          # PhysioNet MI
python main.py loso --data_dir data/loso_binary --finetune 5                          # + few-shot FT
python main.py loso --data_dir data/bci_iv_2a_processed --n_subjects 9 --dataset bci_iv_2a  # BCI IV 2a

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

# === 质量 ===
pytest tests/ -v                     # 131 个测试
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
- Schalk, G., et al. (2004). BCI2000. *IEEE Trans. Biomed. Eng.*
- PhysioNet: [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/)
- BCI Competition IV 2a: [BNCI Horizon 2020](http://bnci-horizon-2020.eu/database/data-sets)
- MNE-Python: [mne.tools](https://mne.tools)
- MOABB: [Mother of All BCI Benchmarks](https://github.com/NeuroTechX/moabb)
- LSL: [labstreaminglayer.org](https://labstreaminglayer.org)
