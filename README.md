# 基于运动想象的脑-机交互算法研究

> **XH-202610** · 挑战杯 2026 · 单人项目

基于运动想象（Motor Imagery）范式的脑-机接口算法研究与实时系统实现。使用 MNE-Python 预处理、EEGNet 深度学习模型，实现对**静息/左手想象/右手想象**三种认知状态的高精度实时分类。

## 快速开始

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate bci
pip install mne streamlit plotly pytest black ruff

# 2. 下载公开数据 (30 人)
python data/download.py

# 3. 预处理
python main.py preprocess

# 4. 训练基线
python main.py baseline

# 5. 训练 EEGNet
python main.py train
```

## 环境

| 组件 | 版本/型号 |
|------|-----------|
| Python | 3.10 |
| PyTorch | 2.5.1 |
| MNE-Python | 1.12+ |
| 目标硬件 | DeepBCI (16 通道, 250 Hz) |

## 项目结构

```
├── data/
│   ├── raw/                    # 原始 EEG (.edf, .fif)
│   ├── processed/              # 预处理后 (.npy) X=[N,C,T], y=[N]
│   ├── subjects/               # 被试元数据
│   └── download.py             # 自动下载 PhysioNet MI
├── preprocessing/
│   ├── run_mne_pipeline.py     # ★ 一键 MNE 预处理
│   ├── filtering.py            #   带通 + 陷波滤波
│   ├── epoching.py             #   分段
│   ├── artifact.py             #   ICA 去伪迹
│   └── mne_pipeline.py         #   编程 API
├── features/
│   ├── csp.py                  # CSP 特征提取
│   └── bandpower.py            # mu/beta 功率比
├── models/
│   ├── eegnet.py               # EEGNet (Lawhern 2018)
│   ├── attention.py            # 通道注意力
│   └── fusion.py               # 多频段融合
├── training/
│   ├── train_eegnet.py         # ★ EEGNet 训练
│   ├── train_baseline.py       # ★ CSP+SVM 基线
│   └── train_ablation.py       #   消融实验
├── realtime/
│   ├── stream_lsl.py           # LSL 真实设备接入
│   ├── stream.py               # DummyStream (离线测试)
│   ├── buffer.py               # 环形缓冲 (线程安全)
│   └── inference.py            # 实时推理封装
├── utils/
│   ├── config.py               # 全局参数
│   ├── metrics.py              # 分类指标
│   └── logger.py               # 实验日志
├── ui/
│   └── dashboard.py            # Streamlit 实时看板
├── main.py                     # 统一入口
├── environment.yml             # Conda 环境
└── CLAUDE.md                   # AI 辅助开发指南
```

## 命令

```bash
python main.py setup           # 创建 conda 环境
python main.py preprocess      # MNE 预处理 → data/processed/
python main.py baseline        # CSP + SVM 基线
python main.py train           # 训练 EEGNet
python main.py ablation        # 消融实验 (EEGNet vs +Attention)
python main.py demo            # 实时推演 Demo (模拟流)
python main.py dashboard       # Streamlit 看板
```

## 分类任务

| 标签 | 含义 | 说明 |
|------|------|------|
| 0 | 静息 (Rest) | 无运动想象 |
| 1 | 左手 (Left) | 左手运动想象 |
| 2 | 右手 (Right) | 右手运动想象 |

数据格式：`X = [N, C, T]` float32，`y = [N]` int

## 预处理流程

```
原始 EEG (160 Hz, 64ch)
  → 重采样至 250 Hz
  → 选择 16 运动皮层通道 (FC/C/CP)
  → 带通滤波 8–30 Hz
  → 陷波 50 Hz
  → CAR (共平均参考)
  → 分段 (-0.5s ~ 2.5s)
  → 导出 [N, 16, 750]
```

### 16 通道选择

| 区域 | 通道 |
|------|------|
| 前运动区 (FC) | FC5, FC3, FC1, FCz, FC2, FC4, FC6 |
| 中央运动区 (C) | C5, C3, C1, Cz, C2, C4, C6 |
| 中央-顶叶 (CP) | CP3, CP4 |

## 模型

### EEGNet

基于 Lawhern et al. (2018)，自适应时间窗口长度：

```
Input (B, 1, 16, 750)
  → Temporal Conv (1×64)
  → Depthwise Conv (16×1) — 空间滤波
  → Separable Conv
  → Linear → 3 classes
```

### 通道注意力

Squeeze-and-Excitation 风格，自动学习 C3/Cz/C4 权重。

### 多频段融合

三路并行：mu (8–13 Hz) / beta (13–30 Hz) / full (8–30 Hz) → 融合分类。

## 实验结果 (PhysioNet MI, 30 人)

| 模型 | 3-Class Acc | 2-Class Acc | 备注 |
|------|-------------|-------------|------|
| CSP + SVM | 29.0% | 52.5% | 线性基线 |
| EEGNet | **52.7%** | **62.8%** | 30 subjects, 16ch |

3-Class per-class accuracy: rest 55.2% / left 38.0% / right 49.7%

## 实时系统架构

```
DeepBCI / LSL Stream
    ↓ 125ms chunk
Ring Buffer (2s 滑动窗口)
    ↓ (16, 500)
EEGNet 推理
    ↓
动作输出 (0=Idle, 1=Left, 2=Right)
```

## 交付物

| 文件 | 要求 | 截止 |
|------|------|------|
| PDF 技术报告 | 算法设计与系统实现 | 9月10日 |
| Excel 验证数据 | 标准数据集性能测试 | 9月10日 |
| MP4 演示视频 | 实时在线系统演示 (≤10min) | 9月10日 |
| 源码 | 完整可运行代码 | 9月10日 |

## 参考文献

- Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5). DOI: [10.1088/1741-2552/aace8c](https://doi.org/10.1088/1741-2552/aace8c)
- Schalk, G., et al. (2004). BCI2000: a general-purpose brain-computer interface (BCI) system. *IEEE Trans. Biomed. Eng.*
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23).
- MNE-Python: https://mne.tools
- Lab Streaming Layer: https://labstreaminglayer.org

## License

本仓库为挑战杯 XH-202610 竞赛作品，保留所有权利。
