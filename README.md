# 基于运动想象的脑-机交互算法研究

> **XH-202610** · 挑战杯 2026 · `master` [![tests](https://img.shields.io/badge/tests-66%20passed-brightgreen)]()

基于运动想象（Motor Imagery）的脑-机接口算法研究与实时系统。使用 MNE-Python 预处理、EEGNet + 时空注意力深度学习模型，实现**空闲/左手/右手**三分类高精度实时识别。

---

## 快速开始

```bash
# 1. 环境
conda env create -f environment.yml && conda activate bci
pip install mne moabb streamlit pytest black ruff openpyxl

# 2. 下载公开数据
python data/download.py                    # PhysioNet MI (30 subjects)
python data/download.py --bci_iv_2a        # BCI Competition IV 2a (9 subjects)

# 3. 预处理 → 导出自定义8通道
python preprocessing/run_mne_pipeline.py --channels motor8

# 4. 训练基线
python main.py baseline

# 5. 训练 EEGNet (随机分割，快速验证)
python main.py train --model eegnet_spatiotemporal --augment

# 6. LOSO 交叉验证 (金标准 BCI 评估)
python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary
python main.py loso --data_dir data/loso_binary --epochs 60
```

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
├── preprocessing/
│   ├── run_mne_pipeline.py         # ★ 端到端 MNE 预处理 (支持 motor8/motor16/all)
│   ├── filtering.py                #   带通 8-30Hz + 陷波 50Hz
│   ├── epoching.py                 #   事件分段
│   ├── artifact.py                 #   ICA 去伪迹
│   ├── mne_pipeline.py             #   编程 API
│   └── augment.py                  #   数据增强 (噪声/通道dropout/时移/幅值缩放)
├── features/
│   ├── csp.py                      # CSP 特征提取 + SVM 基线
│   └── bandpower.py                # μ/β 频带功率
├── models/
│   ├── eegnet.py                   # EEGNet (Lawhern 2018, lazy classifier)
│   ├── attention.py                # 注意力模块 (SE/MHSA/Temporal/Spatiotemporal)
│   ├── eegnet_attn.py              # EEGNet + 注意力深度融合 (5种变体)
│   └── fusion.py                   # 多频段融合 (μ/β/full)
├── training/
│   ├── train_eegnet.py             # ★ 训练脚本 (6大特性)
│   ├── train_baseline.py           #   CSP+SVM 基线
│   ├── train_ablation.py           #   消融实验 (6配置 × N重复)
│   ├── train_sweep.py              #   超参搜索 (Optuna + 网格回退)
│   └── train_loso.py               #   LOSO 交叉验证 (金标准, +few-shot FT)
├── realtime/
│   ├── stream_lsl.py               # LSL 真实设备流
│   ├── stream.py                   # DummyStream (离线模拟)
│   ├── buffer.py                   # 环形缓冲 (线程安全)
│   └── inference.py                # 实时推理封装
├── utils/
│   ├── config.py                   # ★ 全局配置 (8ch montage 定义处)
│   ├── metrics.py                  # 分类指标 (acc/kappa/f1/precision/recall)
│   ├── report_excel.py             # Excel 验证报告生成 (5-Sheet 竞赛格式)
│   └── logger.py                   # 实验日志 (CSV)
├── ui/
│   └── dashboard.py                # Streamlit 实时看板 (模型加载/导出CSV)
├── tests/                          # ★ 66 个单元测试
│   ├── conftest.py                 #   共享 fixtures
│   ├── test_eegnet.py, test_attention.py, test_fusion.py
│   ├── test_buffer.py, test_inference.py
│   ├── test_csp.py, test_metrics.py, test_config.py
│   └── test_preprocessing.py
├── main.py                         # 统一入口
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
python main.py preprocess                    # 8ch 三分类 (默认)
python preprocessing/run_mne_pipeline.py --channels motor8              # 同上
python preprocessing/run_mne_pipeline.py --channels motor8 --binary     # 二分类 (左右手)
python preprocessing/run_mne_pipeline.py --channels motor16             # 16 通道
python preprocessing/run_mne_pipeline.py --channels all                 # 全部通道

# === 训练 ===
python main.py baseline                      # CSP+SVM
python main.py train                         # EEGNet 基线 (200 epochs)
python main.py train --model eegnet_spatiotemporal --augment --label_smoothing 0.1
python main.py ablation                      # 消融实验

# === 高级训练参数 ===
python training/train_eegnet.py \
    --model eegnet_spatiotemporal \          # eegnet|eegnet_se|eegnet_mhsa|eegnet_temporal|eegnet_spatiotemporal
    --data_dir data/processed_binary \       # 数据目录
    --epochs 200 --early_stop 50 \           # 训练轮次 + 早停
    --augment \                               # 数据增强 (2x)
    --label_smoothing 0.1 \                  # 标签平滑
    --grad_clip 1.0 \                        # 梯度裁剪
    --kfold 5                                # K 折交叉验证

python training/train_ablation.py --epochs 150 --repeat 3    # 6 模型 × 3 次
python training/train_sweep.py --model eegnet --trials 50     # 超参搜索 (需 optuna)

# === LOSO 交叉验证 (金标准 BCI 评估) ===
# Step 1: 按被试导出 .npy
python preprocessing/run_mne_pipeline.py --channels motor8 --binary --per_subject --output data/loso_binary
# Step 2: 30-fold LOSO
python main.py loso --data_dir data/loso_binary --n_subjects 30 --epochs 60
python main.py loso --data_dir data/loso_binary --n_subjects 30 --finetune 10   # + few-shot fine-tune

# === 实时 ===
python main.py demo                  # 终端模拟 Demo
python main.py dashboard             # Streamlit 看板 (端口 8501)

# === 工具 ===
python utils/report_excel.py --demo             # 生成 Excel 验证报告模板
pytest tests/ -v                                 # 运行全部测试 (66 个)
black . && ruff check .                          # 格式化 + Lint
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

### EEGNet (Lawhern et al. 2018)

```
Input (B, 1, C, T)
  → Block 1: Temporal Conv(1×64) → DepthwiseConv(C×1) → Pool
  → [Attention] ← 时空注意力插入点
  → Block 2: Separable Conv → Pool
  → Linear → N classes
```

### 注意力变体

| 类型 | 模块 | 说明 |
|------|------|------|
| `se` | `ChannelAttention1D` | Squeeze-and-Excitation 通道注意力 |
| `mhsa` | `MultiHeadChannelAttention` | 多头自注意力 (通道间) |
| `temporal` | `TemporalAttention` | 时域点加权 |
| `spatiotemporal` | `SpatiotemporalAttention` | MHSA + Temporal 串联 |

### 数据增强

- 高斯噪声 (σ=0.05 × per-channel std)
- 通道 Dropout (p=0.1, 模拟电极脱落)
- 时间偏移 (±50ms, 模拟 trial 对齐抖动)
- 幅度缩放 (0.8–1.2×, 模拟阻抗变化)

---

## 实验结果 (PhysioNet MI, 30 人, 8 通道)

> **验证方式**: 以下结果为 **随机 train/val split**（跨被试混洗），仅用于快速验证 pipeline 通断。
> **比赛级评估**需使用 **LOSO (Leave-One-Subject-Out)** — 见下方 [验证方法论](#验证方法论)。

| 模型 | 3-Class | 2-Class | 备注 |
|------|---------|---------|------|
| CSP + SVM | 38.6% | — | CSP-8 + 线性 SVM |
| EEGNet (base) | 53.8% | 56.2% | 无注意力, 无增强 |
| **EEGNet + SpatiotemporalAttn** | **57.6%** | **63.0%** | +增强 +label_smoothing |

> ⚠️ PhysioNet 的 3 分类存在天然数据不均衡 (Idle ≈ 50%)，真实的 "空闲-想象" 区分能力依赖自采数据验证。

### 验证方法论

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| **Random split** | 混洗所有 trial 后 `train_test_split` | 快速 debug, 算法迭代 |
| **LOSO** ⭐ | 29 人训练, 1 人测试, 轮 30 次 | **比赛报告, 论文, 泛化性评估** |
| **LOSO + Few-shot FT** | LOSO 后用目标被试少量 trial 微调 | 在线系统标定 (calibration) |

LOSO 是 BCI 领域的金标准 — 它直接回答"模型对新被试是否有效"，评委也会关注这点。

---

## 交付物

| # | 文件 | 要求 | 状态 |
|---|------|------|------|
| 1 | PDF 技术报告 | 算法设计 + 系统实现 + 实验分析 | ⏳ |
| 2 | Excel 验证数据 | 标准数据集性能 (acc/recall/specificity/latency) | ✅ 生成器就绪 |
| 3 | MP4 演示视频 | 实时在线系统全流程 (≤10 min) | 等硬件 |
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

