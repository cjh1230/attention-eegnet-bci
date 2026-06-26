# EA × Architecture Interaction Analysis

> **核心问题**: Euclidean Alignment 为什么对不同深度模型的提升幅度差异巨大？

> **新发现 (2026-06-26)**: EA 增益并非纯架构属性 — 它受到数据集被试间变异性的显著调节。

## PhysioNet MI 完整 EA 增益表

| 模型 | 无 EA | + EA | 增益 | 内部归一化机制 |
|------|-------|------|------|---------------|
| **FBCNet** | 49.70% | 61.11% | **+11.41pp** | ❌ 无（方差池化直接读取协方差） |
| **EEGNet** | 51.93% | 58.00% | **+6.07pp** | BatchNorm（通道维） |
| **SpatiotemporalAttn** | 55.04% | 57.78% | **+2.74pp** | Attention 隐式重加权 |
| **EEG-Conformer** | 61.33% | 63.93% | **+2.60pp** | LayerNorm + 残差连接 |
| **EEG-TCNet** | 61.56% | 63.41% | **+1.85pp** | BatchNorm + 残差连接 |
| **Tangent Space** | 60.44% | 60.44% | **±0.00pp** | 仿射不变（黎曼度量） |

**实验设置**: PhysioNet MI, 30 subjects, 8ch, binary LOSO, 80 epochs (DL), scm+riemann+lda (Riemannian).

---

## EA 机制量化

### 1. 协方差对齐

EA 后，测试被试与训练被试之间的 **Riemannian 距离减少 85.1%**：

| 指标 | EA 前 | EA 后 | 变化 |
|------|-------|-------|------|
| 平均 Riemannian 距离 | 42.50 | 4.29 | **-85.1% ± 17.4%** |
| 中位数 | — | — | -89.9% |
| 范围 | — | — | [12.8%, 96.0%] |

### 2. 频带方差稳定性

EA 后各频带的被试间变异系数（CV）均下降，**12-16Hz（μ/β 交界）最稳定**：

| 频带 | EA 前 CV | EA 后 CV | 降低 |
|------|----------|----------|------|
| 8-12Hz | 0.449 | 0.405 | -9.9% |
| **12-16Hz** | **0.258** | **0.215** | **-16.6%** |
| 16-20Hz | 0.282 | 0.250 | -11.5% |
| 20-24Hz | 0.321 | 0.303 | -5.5% |
| 24-28Hz | 0.526 | 0.452 | -14.2% |
| 28-30Hz | 0.548 | 0.501 | -8.5% |

---

## 解释：为什么增益不同？

### FBCNet (+11.41pp) — 协方差结构敏感

```
X → Filter Bank (6频带) → Spatial Conv → Temporal Depthwise Conv
  → Variance Pooling → Log → FC → Classifier
```

FBCNet 的核心操作是 **方差池化**（variance pooling）：对每个频带的每个空间滤波器，计算时间维度的方差，然后取 log。方差直接来自输入的协方差结构。

- EA 对齐了跨被试协方差 → 每个频带的方差特征变得可比
- **无内部时序归一化**（没有 BN on time dim, 无残差连接）
- 因此对协方差偏移最敏感，EA 收益最大

### EEGNet (+6.07pp) — BatchNorm 部分缓解

```
X → Temporal Conv → BN → Depthwise Spatial Conv → BN → ELU → Pool
  → Separable Conv → BN → ELU → Pool → FC
```

- BatchNorm 在通道维上做归一化，部分缓解分布偏移
- 但 BatchNorm 的统计量来自训练集，测试被试的偏移仍会影响
- EA 减少了这种偏移 → 中等增益

### SpatiotemporalAttn (+2.74pp) — 注意力重加权

EEGNetWithAttention 中的注意力插入在 Block1（空间卷积）之后：

```
Block1 (Temporal→Spatial Conv) → 此时空间维已压缩为 1
  → Attention over D*F1 channels → Block2 (Separable Conv) → FC
```

- 空间维已通过 depthwise spatial conv 压缩，EA 对原始通道协方差的对齐被"稀释"
- Attention 本身提供隐式特征重加权 → 减少了对输入协方差的依赖

### EEG-Conformer (+2.60pp) — LayerNorm + 残差

```
X → CNN Backbone → Transformer Encoder
  ├── LayerNorm → MHSA → Residual → LayerNorm → FFN → Residual
  └── FC
```

- **LayerNorm** 在每个 token 上独立归一化，天然消除分布偏移
- **残差连接** 提供梯度高速公路，稳定训练
- Multi-head self-attention 建模的是通道/时间之间的**相对关系**，而非绝对值

### EEG-TCNet (+1.85pp) — BatchNorm + 残差

```
X → EEGNet Block1 → TCN Block
  ├── Dilated Conv1d (groups=in_channels) → BN → ELU → Dropout
  └── + Residual (if shape matches)
```

- TCN 的膨胀卷积 + BatchNorm 在时序维上反复归一化
- 残差连接吸收分布偏移
- 是**所有 DL 模型中 EA 增益最小的**

### Tangent Space (±0.00pp) — 仿射不变

```
X → Covariance Matrix (SPD) → Tangent Space Projection → LDA
```

- 切线空间映射使用仿射不变黎曼度量
- EA 本质上是仿射变换（白化）：`X' = R^{-1/2} @ X`
- 仿射变换不改变 SPD 流形上的黎曼距离 → 分类边界不变

---

## 结论

### 设计原则

1. **协方差敏感度决定 EA 收益**：方差池化 > 简单 CNN > BatchNorm > LayerNorm/残差 > 仿射不变
2. **内部归一化越强，EA 增益越小**：LayerNorm + 残差（Conformer/TCNet）几乎不需要 EA
3. **EA 对频带方差稳定性贡献有限**（~11%），主要贡献在协方差对齐（~85%）
4. **12-16Hz 是最稳定的 MI 频带**，EA 后 CV 仅 0.215

### 对架构设计的启示

- 如果架构已有强内部归一化（LayerNorm/BN + residual），EA 的边际收益小
- 如果架构依赖原始协方差/方差特征（如 FBCNet 的 variance pooling），EA 至关重要
- 在少通道（8ch）设置下，EA 是替代复杂归一化结构的轻量方案
- **建议**: 在设计面向 DeepBCI 8ch 的模型时，如使用 variance/log-variance 特征层，必须配合 EA；如使用 TCN/Transformer 结构，EA 是可选增强

---

## Few-shot Calibration

对最优模型（EEG-Conformer + EA）进行少样本校准：

| Calibration Trials/Class | Accuracy |
|--------------------------|----------|
| 0 (pure LOSO) | 65.33% |
| 5 | 66.38% |
| **10** | **67.47%** |
| 20 | 66.38% |
| 40 | 66.67% |

> Few-shot calibration 带来额外 +2.14pp 提升（10 trials/class），说明少量目标被试数据即可进一步提升在线性能。

---

## 跨数据集验证: BCI Competition IV 2a

> **关键发现**: EA 增益是**数据集相关的**，不是纯架构属性。

### BCI IV 2a (9 subjects, 8ch, 4-class LOSO, 60 epochs)

| 模型 | 无 EA | + EA | 增益 | vs PhysioNet |
|------|-------|------|------|-------------|
| EEGNet | 39.85% | 38.31% | **-1.54pp** ⚠️ | +6.07pp（反转！） |
| EEG-TCNet | 40.33% | 40.78% | +0.45pp | +1.85pp（一致低增益） |
| EEG-Conformer | 40.07% | 40.12% | +0.05pp | +2.60pp（一致低增益） |
| FBCNet | ⏳ | ⏳ | — | +11.41pp |

### 双数据集对比

```
PhysioNet (30 subjects, ~45 trials/subject, binary):
  EA 增益显著: FBCNet +11.41 > EEGNet +6.07 > Conformer +2.60 > TCNet +1.85 > Tangent ±0

BCI IV 2a (9 subjects, 576 trials/subject, 4-class):
  EA 增益微小或负: EEGNet -1.54, TCNet +0.45, Conformer +0.05
```

### 解释: 修正后的 EA 增益模型

原有假设（仅架构维度）:
```
EA 增益 = f(内部归一化强度)
```

修正后的双因素模型:
```
EA 增益 = f(架构内部归一化 × 数据集被试间变异性)
```

| 数据集特征 | PhysioNet MI | BCI IV 2a |
|-----------|-------------|-----------|
| 被试数 | **30** | 9 |
| Trials/被试 | **~45** | 576 |
| 被试间变异 | **大**（少 trials，高方差） | 小（多 trials，稳定估计） |
| EA 效果 | **显著**（+2~+11pp） | 微弱（-1.5~+0.5pp） |

**机制**:
- EA 的核心作用是减少被试间协方差分布差异
- 当被试间差异本来就大（PhysioNet: 30 被试，每人仅 ~45 trials）→ EA 收益大
- 当被试间差异较小（BCI IV 2a: 9 被试，每人 576 trials → 协方差估计更稳定）→ EA 边际收益小
- 4-class 任务本身更难（chance=25% vs PhysioNet 50%），EA 的协方差对齐不足以弥补类别决策边界的复杂度

**对架构设计的启示（更新版）**:
1. 少被试、少 trial 的场景（如真实 BCI 校准）→ EA 至关重要
2. 多被试、多 trial 的场景 → 内部归一化（BN/LN）已足够
3. **EA 对 FBCNet 的 +11pp 增益可能也受 PhysioNet 特定数据特征影响，需在 BCI IV 2a 上验证**
4. 跨数据集迁移时，不能假设 EA 增益可复现

---

## 完整排行榜

| Rank | 模型 | Accuracy | 类型 |
|------|------|----------|------|
| 1 | EEG-Conformer + EA + FT10 | **67.47%** | DL + EA + Calibration |
| 2 | EEG-Conformer + EA | 63.93% | DL + Transformer + EA |
| 3 | EEG-TCNet + EA | 63.41% | DL + TCN + EA |
| 4 | FBCNet + EA | 61.11% | DL + Filter Bank + EA |
| 5 | Tangent Space + LDA + EA | 60.44% | Riemannian 传统基线 |

---

*分析日期: 2026-06-26*
