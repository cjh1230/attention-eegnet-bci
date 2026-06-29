# 低通道运动想象 EEG 跨被试解码中的轻量化 SPD 流形深度学习

> **修订稿 v3** | 2026-06-29  
> 叙事主线：轻量 SPDNet + Euclidean Alignment (EA) + 自监督预训练边界分析  
> 当前状态：实验结果已写入；图表、统计检验和参考文献格式仍需终稿核对。
>
> **论文定位**：本文不主张提出一种能够大幅刷新性能的新自监督方法，而是系统评估低通道 MI-EEG 跨被试解码中 SPD 流形深度学习的效率边界、必要条件和失败模式。核心结论包括：轻量 SPDNet 能以极低参数量接近复杂深度模型；EA 是 SPDNet 稳定工作的关键条件；基础自监督预训练在 8 x 8 SPD 矩阵上未带来增益，提示该场景存在明确的信息瓶颈。

---

## 摘要

低通道运动想象脑机接口（motor imagery brain-computer interface, MI-BCI）需要在有限电极、低信噪比和显著个体差异下实现可靠解码。现有深度模型通常直接处理原始 EEG 时间序列，能够取得较高精度，但参数量和跨被试稳定性仍限制其在便携式系统中的部署。本文从对称正定（symmetric positive definite, SPD）流形视角出发，将 EEG 协方差矩阵作为紧凑表征，系统评估 SPD 流形深度学习在 8 通道 MI-EEG 跨被试解码中的效率边界。

我们在 PhysioNet MI 数据集前 30 名被试上进行二分类留一被试交叉验证（leave-one-subject-out, LOSO）。所有实验均采用 8 个运动皮层通道（FC3, C3, Cz, C4, FC4, CP3, CPz, CP4），并在相同预处理和划分条件下比较 SPDNet、传统黎曼方法和多种深度学习基线。结果显示，单层 BiMap SPDNet 仅含 260 个可学习参数，在施加 Euclidean Alignment (EA) 后达到 61.04% ± 10.42% 的跨被试准确率（Cohen's kappa = 0.219）。该结果超过 Tangent Space + LDA + EA（60.44% ± 9.64%），并接近参数量约高两个数量级的 FBCNet + EA（61.11% ± 11.69%）。

进一步实验表明，EA 是 SPDNet 正常工作的关键条件。未施加 EA 时，SPDNet 准确率降至 50.59% ± 1.87%，接近随机水平；施加 EA 后提升 10.45 个百分点。结构消融显示，在 8 x 8 SPD 矩阵上，加深或加宽 SPDNet 均未带来增益。我们还探索了 SPD 流形上的两种自监督预训练策略，即对比学习和掩码重建，但二者分别使性能下降 0.89 和 2.97 个百分点。

这些结果表明，低通道 MI-EEG 中的 SPD 流形深度学习具有明确的效率优势，但其性能高度依赖跨被试几何对齐。本文提供了一个面向便携式 MI-BCI 的轻量化 SPD 学习基准，并指出低维 SPD 表征上的自监督学习需要更精细的任务匹配与增强设计。

---

## 1. 引言

运动想象脑机接口通过解码用户想象肢体运动时产生的脑电活动，实现不依赖外周肌肉输出的人机交互。该技术在神经康复、辅助控制和智能交互中具有应用潜力。然而，MI-EEG 信号具有低信噪比、强非平稳性和显著个体差异，使跨被试解码成为该领域最困难的问题之一。对于便携式 BCI 系统，这一问题更加突出，因为低通道硬件虽然更易部署，却会进一步压缩空间信息。

深度学习已显著推动 MI-EEG 解码的发展。EEGNet、EEG Conformer、EEG-TCNet 和 FBCNet 等模型能够从原始 EEG 时间序列中学习时空特征，并在多个公开数据集上取得有竞争力的结果。然而，这些模型通常在欧氏空间中处理原始信号，未显式利用 EEG 协方差矩阵的几何结构。与之相对，基于黎曼几何的方法将 EEG 试次表示为 SPD 流形上的点，并通过切空间投影、MDM 或滤波器组黎曼分类器实现稳健分类。这类方法参数少、泛化稳定，但表达能力通常受限于线性分类器或固定度量。

SPDNet 提供了连接这两条路线的可能性。它通过 BiMap、ReEig 和 LogEig 等层，在保持 SPD 结构的同时引入端到端可学习变换。已有研究主要在通道数较多的数据集或被试内设置中评估 SPD 深度学习。对于更接近便携式应用的 8 通道 MI-EEG、LOSO 跨被试协议和低样本条件，SPDNet 的效率边界、架构需求和对齐依赖尚缺乏系统评估。

本文关注一个具体问题：在仅使用 8 个运动皮层通道时，SPD 流形深度学习能否以极低参数量达到或超过传统黎曼方法，并接近复杂深度模型？围绕这一问题，我们进一步考察三个子问题。第一，SPDNet 在 8 x 8 低维 SPD 矩阵上需要多深、多宽的结构。第二，EA 对 SPDNet 的作用有多关键，以及如何从 SPD 流形视角理解其机制。第三，SPD 流形上的基础自监督预训练能否进一步提升跨被试泛化。

我们的实验给出三个主要发现。首先，单层 BiMap SPDNet 在 260 个参数下即可达到 61.04% 的 LOSO 准确率，超过 Tangent Space + LDA + EA，并几乎追平 FBCNet + EA。其次，EA 对 SPDNet 带来 10.45 个百分点的增益，是模型稳定工作的必要条件。第三，对比学习和掩码重建未提升下游分类性能，提示 8 x 8 SPD 矩阵中的可用判别信息有限，常规自监督增强可能破坏而非保留类别相关结构。

---

## 2. 相关工作

### 2.1 BCI 中的黎曼几何方法

黎曼几何方法在 BCI 中的核心思想是将每个 EEG 试次的协方差矩阵视为 SPD 流形上的点，并利用该空间的测地距离或切空间表示进行分类。Barachant 等人的工作奠定了这一方向，随后 Tangent Space + LDA、Minimum Distance to Mean (MDM) 和 Filter-bank Riemannian 方法被广泛用于 MI-EEG 解码。此类方法通常不需要大规模训练数据，在跨被试场景中具有较好稳定性，但其表达能力受限于固定特征映射和相对简单的分类器。

### 2.2 SPD 流形上的深度学习

Huang 和 Van Gool 提出的 SPDNet 将深度学习引入 SPD 流形。其核心层包括 BiMap（双线性映射）、ReEig（特征值整流）和 LogEig（矩阵对数映射）。这些操作在保持 SPD 结构的同时实现非线性表征学习。后续研究进一步将多频段分解、图神经网络和张量表示引入 SPD 学习框架，用于提升 EEG 解码性能。

现有 SPD 深度学习研究多集中于通道数较高的数据集，如 BCI Competition IV 2a，或采用被试内评估。对于 8 通道运动皮层 montage、PhysioNet MI 数据集和 LOSO 跨被试评估的组合，目前仍缺少系统基准。本文将这一低通道设置作为主要对象，重点考察 SPDNet 在低维协方差矩阵上的有效性和边界。

### 2.3 EEG 中的自监督学习

自监督学习正在成为 EEG 表征学习的重要方向。已有工作通过掩码重建、生成式预训练或对比学习从未标注 EEG 中学习可迁移表征，并在部分 MI-EEG 任务上取得了有前景的结果。然而，大多数方法直接作用于原始 EEG 时间序列。SPD 流形上的自监督预训练仍处于探索阶段，尤其是在 8 x 8 低维协方差矩阵上的适用性尚不明确。

### 2.4 Euclidean Alignment

Euclidean Alignment (EA) 通过估计训练数据的参考协方差矩阵，并将各试次线性对齐到共享参考空间，从而减小跨被试分布偏移。EA 已被证明能提升多种深度模型的跨被试性能。对于 SPD 学习，EA 可视为对协方差矩阵施加同余变换，使不同被试的分布更接近单位矩阵邻域。本文系统量化 EA 对 SPDNet 的影响，并将其作为理解 SPDNet 跨被试泛化的关键因素。

---

## 3. 方法

### 3.1 SPD 流形表示

对称正定矩阵集合记为：

```text
S_d^{++} = {C in R^{d x d} | C = C^T, x^T C x > 0 for all x != 0}.
```

每个 EEG 试次 `X in R^{C x T}` 被表示为正则化样本协方差矩阵：

```text
C = X X^T / T + epsilon I.
```

在本文的 8 通道设置中，`C` 为 8 x 8 SPD 矩阵，包含 36 个独立元素。两个 SPD 矩阵之间的仿射不变黎曼距离为：

```text
delta_R(C_1, C_2) = ||log(C_1^{-1/2} C_2 C_1^{-1/2})||_F.
```

该表示保留通道间协方差结构，并将原始 EEG 试次压缩为低维几何对象。

### 3.2 SPDNet 架构

本文使用轻量化 SPDNet 作为主要模型。基础结构为：

```text
BiMap(8 -> 8) -> ReEig -> LogEig -> upper-triangular flatten -> Linear(2).
```

BiMap 层将输入 SPD 矩阵映射到新的 SPD 矩阵：

```text
C_out = W C_in W^T,
```

其中 `W in R^{d_out x d_in}` 为可学习权重。ReEig 层对特征值进行下限截断，以保证输出矩阵的数值稳定性。LogEig 层将 SPD 矩阵映射到切空间，随后提取上三角元素并输入线性分类器。基础模型总参数量为 260。

为评估结构复杂度的影响，我们比较了四种 BiMap 维度配置：[8, 8]、[8, 8, 8]、[8, 10, 8] 和 [8, 6, 4]。这些配置分别测试单层、加深、加宽和压缩瓶颈设计。

### 3.3 EA 的几何解释

EA 首先在训练被试数据上估计参考协方差矩阵：

```text
R_bar = (1 / N) sum_i X_i X_i^T / T.
```

随后对训练和测试试次施加相同变换：

```text
X_aligned = R_bar^{-1/2} X.
```

对应到协方差矩阵，EA 等价于同余变换：

```text
C_aligned = R_bar^{-1/2} C R_bar^{-1/2}.
```

在 SPD 流形视角下，该操作将不同被试的协方差分布对齐到共同参考点附近，同时保留仿射不变度量下的重要几何关系。本文在每一折 LOSO 中仅使用训练被试估计 `R_bar`，并将该变换应用于训练和测试数据，以避免数据泄漏。

需要强调的是，EA 不是严格意义上的黎曼平行移动。更准确地说，EA 是一种基于同余变换的全局协方差对齐或白化操作。本文使用“几何对齐”描述其作用，以避免过度数学化表述。

### 3.4 SPD 流形上的自监督预训练

我们探索两类基础自监督预训练策略。

**对比学习**：对同一 SPD 矩阵生成两种增强视图，包括通道随机丢弃和协方差扰动。增强样本经 SPDNet 编码器和投影头后，使用 NT-Xent 损失拉近正样本对并推远批内负样本。

**掩码重建**：随机遮蔽一个通道对应的协方差行列，将遮蔽后的矩阵输入 SPDNet 编码器，并由三层 MLP 解码器在 Log-Euclidean 空间中重建完整对数协方差矩阵。损失函数为重建矩阵与目标矩阵之间的均方误差。

预训练使用 LOSO 训练折中的全部未标注试次。预训练完成后，丢弃投影头或解码器，在编码器后添加线性分类器进行有监督微调。

---

## 4. 实验

### 4.1 数据集与预处理

本文使用 PhysioNet MI 数据集前 30 名被试。每名被试采用运动想象 runs 4、8 和 12，构成左手与右手二分类任务。原始数据为 64 通道 EEG，原始采样率为 160 Hz。预处理包括 8-30 Hz 带通滤波、50 Hz 陷波滤波、重采样至 250 Hz，以及 3 s 试次提取。每名被试得到 45 个试次，输入形状为 `(45, 8, 750)`。

通道选择固定为 FC3、C3、Cz、C4、FC4、CP3、CPz 和 CP4。这一 montage 覆盖运动皮层核心区域，并与低通道便携式 BCI 的部署需求相匹配。

### 4.2 评估协议

所有主要实验均采用 LOSO 协议。每一折以 1 名被试作为测试集，其余 29 名被试作为训练集，共进行 30 折。报告平均准确率、标准差和 Cohen's kappa。EA 在每一折内仅基于训练被试计算，并应用于训练集和测试集。

统计检验当前作为待补充项保留。终稿应基于逐被试结果进行配对 t 检验或 Wilcoxon 符号秩检验，并报告效应量和多重比较修正策略。

### 4.3 训练配置

SPDNet 使用 AdamW 优化器训练，学习率为 1e-3，权重衰减为 1e-4，批量大小为 64，最大训练轮数为 60，早停耐心值为 30。样本协方差矩阵使用 1e-4 正则化项并进行迹归一化，以提高数值稳定性。所有 SPDNet 实验在 CPU 上完成。

### 4.4 基线方法

我们在相同数据划分和预处理条件下比较以下基线：

| 类别 | 方法 | 说明 |
|---|---|---|
| 传统黎曼 | Tangent Space + LDA + EA | 协方差矩阵、切空间投影和线性判别分析 |
| 传统黎曼 | FgMDM + EA | 滤波器组协方差和 MDM 分类 |
| 深度学习 | EEGNet + EA | 紧凑卷积神经网络 |
| 深度学习 | EEG Conformer + EA | CNN 与 Transformer 编码器 |
| 深度学习 | EEG-TCNet + EA | CNN 与时间卷积网络 |
| 深度学习 | FBCNet + EA | 滤波器组 CNN |

---

## 5. 结果

### 5.1 主要性能比较

表 1 汇总 PhysioNet MI 8 通道二分类 LOSO 实验结果。

**表 1. PhysioNet MI 8ch binary LOSO 主要结果**

| 方法 | 准确率 (%) | Kappa | 参数量 |
|---|:---:|:---:|:---:|
| EEG Conformer + EA | **63.93 ± 9.58** | 0.277 | ~40K |
| EEG-TCNet + EA | 63.41 ± 10.51 | 0.265 | ~10K |
| FBCNet + EA | 61.11 ± 11.69 | 0.219 | ~50K |
| **SPDNet [8,8] + EA (本文)** | **61.04 ± 10.42** | **0.219** | **260** |
| Tangent Space + LDA + EA | 60.44 ± 9.64 | 0.212 | - |
| FgMDM + EA | 59.18 ± 8.12 | 0.180 | - |
| EEGNet + EA | 58.00 ± 10.06 | 0.161 | ~2K |

SPDNet [8,8] + EA 达到 61.04% ± 10.42% 的准确率，超过 Tangent Space + LDA + EA，并与 FBCNet + EA 基本持平。与复杂深度模型相比，SPDNet 的绝对准确率低于 EEG Conformer + EA 和 EEG-TCNet + EA，但参数量分别约为其 1/154 和 1/38。该结果表明，低维 SPD 表征可以在极低参数预算下保留有效判别信息。

### 5.2 结构消融

表 2 比较不同 BiMap 维度配置下的 SPDNet 性能。

**表 2. SPDNet 结构消融**

| BiMap 维度 | 层数 | 参数量 | 准确率 (%) | Kappa |
|---|:---:|:---:|:---:|:---:|
| [8, 8] | 1 | **260** | **61.04 ± 10.42** | **0.219** |
| [8, 8, 8] | 2 | 392 | 59.78 ± 8.75 | 0.190 |
| [8, 10, 8] | 2 | 422 | 59.18 ± 8.20 | 0.180 |
| [8, 6, 4] | 2 | 94 | 56.89 ± 7.60 | 0.132 |

加深或加宽 SPDNet 均降低了跨被试性能。单层 BiMap [8,8] 在 8 x 8 SPD 矩阵上取得最佳效率和性能平衡。该结果说明，在低维协方差空间中，模型容量很快达到饱和；额外结构可能主要增加过拟合风险，而不是引入有效表达能力。

### 5.3 EA 增益分析

表 3 给出 EA 对 SPDNet 和其他模型的影响。

**表 3. EA 增益分析**

| 方法 | 无 EA (%) | +EA (%) | 增益 |
|---|:---:|:---:|:---:|
| **SPDNet [8,8]** | **50.59 ± 1.87** | **61.04 ± 10.42** | **+10.45 pp** |
| FBCNet | 49.70 ± 2.66 | 61.11 ± 11.69 | +11.41 pp |
| EEGNet | 51.93 ± 7.20 | 58.00 ± 10.06 | +6.07 pp |
| EEGNet + SpatiotemporalAttn | 55.04 ± 8.55 | 57.78 ± 8.55 | +2.74 pp |
| Tangent Space + LDA | 60.44 ± 9.64 | 60.44 ± 9.64 | ±0.00 pp |

SPDNet 对 EA 的依赖最为突出。未施加 EA 时，模型准确率接近随机水平，并出现类别坍缩现象；施加 EA 后，准确率提高 10.45 个百分点。FBCNet 同样从 EA 中获得较大增益，而 Tangent Space + LDA 的结果不变，这与其仿射不变几何性质一致。

### 5.4 SPD 自监督预训练结果

表 4 展示两类自监督预训练策略的下游性能。

**表 4. SPD 流形自监督预训练结果**

| 预训练策略 | 准确率 (%) | Kappa | 相对基线 |
|---|:---:|:---:|:---:|
| 无预训练（全监督基线） | **61.04 ± 10.42** | 0.219 | - |
| 对比学习（SimCLR 风格） | 60.15 ± 10.41 | 0.202 | -0.89 pp |
| 掩码重建（MAE 风格） | 58.07 ± 8.98 | 0.160 | -2.97 pp |

两种自监督策略均未提升下游分类性能。预训练阶段损失正常收敛，说明模型能够优化预训练目标，但所得表征未能有效迁移到 MI 分类任务。这一结果提示，在 8 x 8 SPD 矩阵上，常规增强或重建目标可能与下游判别目标不匹配。

### 5.5 多频段协方差实验

在 8-30 Hz 全频段协方差基础上，我们进一步测试 mu（8-13 Hz）和 beta（13-30 Hz）子带协方差的双分支融合。结果见表 5。

**表 5. 多频段协方差结果**

| 配置 | 准确率 (%) | 相对基线 |
|---|:---:|:---:|
| 单频段 [8,8]（8-30 Hz） | **61.04** | - |
| 双频段 [8,8]（mu + beta） | 60.82 | -0.22 pp |

多频段拆分未带来增益。由于预处理已经保留 8-30 Hz 的主要 MI 频段，进一步拆分为 mu 和 beta 子带未引入额外判别信息，反而可能因每个子带的信息量减少而略微损害性能。

---

## 6. 讨论

### 6.1 低维 SPD 表征的效率边界

本文最重要的发现是，8 x 8 SPD 矩阵上的极简 SPDNet 可以达到与传统黎曼方法相当甚至略优的性能，同时显著降低参数量。协方差矩阵将 EEG 试次压缩为通道间统计结构。在 8 通道条件下，该表示仅包含 36 个独立元素，因此单层 BiMap 已足以学习主要判别方向。更深或更宽的网络没有改善性能，反而降低泛化能力。

这一结果对便携式 BCI 具有实际意义。若目标是极低计算开销和可部署性，而非追求最高离线准确率，轻量 SPDNet 提供了一种有吸引力的选择。它的性能低于 EEG Conformer + EA 和 EEG-TCNet + EA，但参数量降低约两个数量级，并且显式利用协方差几何结构。

### 6.2 EA 在 SPDNet 中的作用

EA 对 SPDNet 的 +10.45 个百分点增益表明，跨被试分布对齐不是可选预处理，而是低参数 SPD 深度学习的必要条件。无 EA 时，单层 BiMap 需要同时处理被试间分布偏移和类别判别结构。对于仅含 260 个参数的模型，这一负担过重，最终导致接近随机水平的结果。

从 SPD 流形视角看，EA 通过同余变换将不同被试的协方差分布移至共同参考空间，使 BiMap 层能够在更一致的几何坐标中学习判别特征。这也解释了为什么 EA 对 SPDNet 和 FBCNet 等小型或结构受限模型的增益更大，而对容量较高的模型增益相对较小。较大模型可能在参数空间中部分吸收被试间分布偏移，但轻量模型更依赖显式对齐。

### 6.3 SPD 自监督学习的边界

自监督预训练的负结果并不意味着 SPD 流形不适合自监督学习，而是说明常规目标在低维 SPD 场景中可能失配。8 x 8 SPD 矩阵只有 36 个独立元素，判别信息可能集中在少数协方差模式中。通道丢弃、协方差扰动或掩码重建会改变这些细微结构，从而削弱与左右手运动想象相关的类别信息。

另一种可能解释是，预训练目标学习到的是被试身份、整体能量或通道统计依赖，而不是运动想象类别所需的特定协方差模式。因此，未来的 SPD 自监督学习需要更贴近下游任务。例如，可以探索监督对比学习、类别保持型流形增强、多任务预训练，或在更高通道数 SPD 矩阵上验证自监督目标是否更有效。

### 6.4 局限性

本文仍有若干限制。第一，主要实验仅在 PhysioNet MI 前 30 名被试上完成，跨数据集泛化仍需在 BCI Competition IV 2a 等数据集上进一步验证。第二，自监督实验只测试了两种基础策略，尚未覆盖更精细的增强设计和多任务目标。第三，本文未探索原始 EEG 深度模型与 SPD 后端结合的混合架构。第四，当前统计检验仍需补充逐被试显著性分析、效应量和多重比较校正。

---

## 7. 结论

本文系统评估了低通道 MI-EEG 跨被试解码中的 SPD 流形深度学习。结果表明，单层 BiMap SPDNet 在仅 260 个参数下即可达到 61.04% 的 LOSO 准确率，超过 Tangent Space + LDA + EA，并接近 FBCNet + EA。结构消融显示，8 x 8 SPD 矩阵上的模型复杂度应保持克制，额外深度或宽度并未带来收益。

EA 是该框架的关键条件。它将跨被试协方差分布对齐到共享参考空间，使轻量 SPDNet 能够专注于学习判别结构。相反，基础自监督预训练在该低维 SPD 场景中未能提升性能，提示预训练目标和增强策略必须更严格地保护类别相关协方差信息。

总体而言，本文的贡献不在于追求最高离线准确率，而在于明确低通道 MI-EEG 中 SPD 深度学习的效率边界和方法条件。未来工作应进一步验证跨数据集泛化，设计判别性更强的 SPD 自监督目标，并探索兼顾原始 EEG 时序建模与 SPD 几何约束的混合架构。

---

## 参考文献

[1] Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.

[2] Song, Y., et al. (2023). EEG Conformer: Convolutional Transformer for EEG Decoding. *arXiv:2301.05578*.

[3] Ingolfsson, T. M., et al. (2020). EEG-TCNet: An accurate temporal convolutional network for embedded motor imagery brain-machine interfaces. *IEEE SMC 2020*.

[4] Bakshi, K., et al. (2021). FBCNet: A multi-view convolutional neural network for brain-computer interface. *arXiv:2104.01233*.

[5] Barachant, A., et al. (2012). Multiclass brain-computer interface classification by Riemannian geometry. *IEEE Transactions on Biomedical Engineering*, 59(4), 920-928.

[6] Barachant, A., et al. (2013). Classification of covariance matrices using a Riemannian-based kernel for BCI applications. *Neurocomputing*, 112, 172-178.

[7] Ang, K. K., et al. (2008). Filter bank common spatial pattern (FBCSP) in brain-computer interface. *IEEE IJCNN 2008*.

[8] Huang, Z., & Van Gool, L. (2017). A Riemannian network for SPD matrix learning. *AAAI 2017*.

[9] Ju, C., & Guan, C. (2023). Tensor-CSPNet: A novel geometric deep learning framework for motor imagery classification. *IEEE Transactions on Neural Networks and Learning Systems*, 34(12), 10955-10969.

[10] Ju, C., & Guan, C. (2023). Graph neural networks on SPD manifolds for motor imagery classification. *IEEE Transactions on Neural Networks and Learning Systems*.

[11] Aristimunha, B., et al. (2026). SPD Learn: A geometric deep learning Python library for neural decoding through trivialization. *arXiv:2602.22895*.

[12] Liu, et al. (2025). MIRepNet: A pipeline and foundation model for EEG-based motor imagery classification. *arXiv:2507.20254*.

[13] Wenhui, et al. (2024). Neuro-GPT: Developing an EEG foundation model. *GitHub*.

[14] He, H., & Wu, D. (2020). Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach. *IEEE Transactions on Biomedical Engineering*, 67(7), 1906-1916.

---

## 待办

- [ ] 补充图表：性能对比柱状图、结构消融图、EA 增益图、t-SNE 或 covariance-space 可视化。
- [ ] 补充统计检验：配对 t 检验或 Wilcoxon 符号秩检验，报告效应量和多重比较校正。
- [ ] 核对所有参考文献 DOI、期刊名、卷期页码和引用编号。
- [ ] 若 BCI Competition IV 2a 跨数据集验证已经完成，补充到结果或补充材料。
- [ ] 明确目标期刊后，按具体 author guidelines 调整摘要长度、图表格式和参考文献格式。
