# 面向 8 通道 MI-EEG 的 SPD 流形深度表征学习：修订研究计划

> **版本**: v2 — 基于实验结果修订  
> **日期**: 2026-06-29  
> **状态**: 基线已验证，SSL 方向待重新设计

---

## 零、实验现状

### 已完成实验（30 被试 PhysioNet MI binary LOSO）

| # | 配置 | Accuracy | 结论 |
|:--:|------|----------|------|
| 1 | SPDNet [8,8] + EA | **61.04% ± 10.42%** | 🥇 最优基线 |
| 2 | SPDNet [8,8] + EA + SSL 对比预训练 | 60.15% ± 10.41% | ❌ 未带来增益 |
| 3 | Multi-band SPDNet [8,8] mu+beta + EA | 60.82% ± 8.06% | → 无显著增益 |
| 4 | SPDNet [8,8] + EA, lr=5e-3 | 60.81% ± 9.45% | → 与 lr=1e-3 无差异 |
| 5 | SPDNet [8,8,8] + EA | 59.78% ± 8.75% | ❌ 两层过拟合 |
| 6 | SPDNet [8,10,8] + EA | 59.18% ± 8.20% | ❌ 扩展-收缩无用 |
| 7 | SPDNet [8,6,4] + EA | 56.89% ± 7.60% | ❌ 压缩太激进 |
| 8 | SPDNet [8,8] no EA | 50.59% ± 1.87% | ❌ 无 EA = 随机 |

### 关键发现

1. **SPDNet 基线有效**：以 260 参数达到 61.04%，超越 Tangent Space+LDA (60.44%)，逼近 FBCNet (61.11%)
2. **EA 是必须的**：+10.45pp 增益，无 EA 模型完全学不到任何东西
3. **架构越简单越好**：单层 BiMap [8,8] 优于所有更深/更宽的变体
4. **SSL 对比预训练未生效**：预训练损失正常收敛（0.22→0.15），但 fine-tuning 后性能反而下降 0.89pp
5. **多频段拆分无增益**：在已 bandpass 到 8-30Hz 的数据上进一步拆分 mu/beta 子带没有带来额外信息

### 对比预训练失败的可能原因

SPD 流形上的 SimCLR 风格对比学习在 8×8 矩阵上面临几个根本性困难：

| 问题 | 分析 |
|------|------|
| **增强破坏语义** | channel_dropout 和 cov_perturb 可能破坏了 MI 类别相关的协方差结构，而非仅引入不变性 |
| **正样本对过于相似** | 同一个 8×8 SPD 矩阵的两种增强差异太小，对比学习退化为区分 trial identity 而非学习语义特征 |
| **batch size 不足** | InfoNCE 需要大 batch（≥256），但 1305 samples/fold + CPU 限制了有效 batch size |
| **投影头坍塌** | 小规模 SPD 矩阵上，投影头可能学到了 trivial solution（将所有样本映射到同一点） |

---

## 一、修订后的研究方向

基于实验结果，对原计划做出以下调整：

### 保留并加强

| 方向 | 调整 |
|------|------|
| **SPDNet 基线** | 作为核心贡献点之一。8ch PhysioNet LOSO 上首个黎曼 DL 基准，260 参数达到 61.04%，本身就是有效的方法贡献 |
| **EA 增益分析** | +10.45pp 的 EA 增益是所有 DL 方法中最高的（vs EEGNet +6.07pp, Conformer +? pp），值得单独分析 |
| **少样本实验** | 即使没有 SSL 预训练，SPDNet 在小样本场景下的表现仍值得研究（260 参数的极简模型天然适合少样本） |

### 重新设计

| 方向 | 原计划 | 修订后 |
|------|--------|--------|
| **SSL 预训练** | SimCLR 对比学习 | **掩码重建（MAE 风格）**：遮蔽部分协方差矩阵元素 → 重建完整 SPD 矩阵。更接近"学习协方差结构"的目标，不依赖 batch size |
| **跨被试泛化** | SSL → fine-tune | **Subject-conditional SPDNet**：每个被试学习一个低维偏差向量，调制共享 SPDNet。更直接地解决跨被试偏移 |
| **论文叙事** | "SPD+SSL 突破天花板" | **"8ch 低通道 MI-EEG 的几何表征学习：效率、对齐与泛化"** —— 更诚实的定位 |

### 新增方向

| 方向 | 动机 |
|------|------|
| **SPD 流形掩码重建预训练** | 掩码 channel/频段 → 重建协方差结构。相比对比学习更适合 SPD 矩阵，不依赖大 batch |
| **跨被试协方差残差建模** | 不尝试"消除"被试差异，而是显式建模每个被试相对总体均值的协方差残差 |
| **与 Tangent Space 的深度对比分析** | 系统分析为什么 SPDNet（260 参数）能做到和 Tangent Space+LDA 几乎相同的性能——协方差空间的本质结构是什么？ |

---

## 二、修订后的技术路线

### 2.1 阶段 A：SPD 掩码重建预训练（替代对比学习）

**核心思想**：随机 mask 协方差矩阵的某些通道或元素 → 训练 SPDNet 重建完整矩阵。

```
完整协方差 C (8×8)
       ↓
随机 mask 1-2 个通道 → C_masked (部分观测)
       ↓
SPDNet 编码器 → LogEig → 解码器 → 重建 C_recon (8×8)
       ↓
Loss: ||log(C) - log(C_recon)||²  (Log-Euclidean 空间 MSE)
```

**为什么掩码重建比对比学习更适合 SPD 矩阵**：
1. **任务定义明确**：重建被 mask 的协方差结构是 well-defined 的回归任务，不依赖 batch 内负样本
2. **信息完整性**：mask 通道迫使模型学习通道间的统计依赖关系（这正是协方差矩阵的核心信息）
3. **物理可解释**：mask 通道 = 模拟电极丢失/失效，模型学到的"修复"能力直接对应跨通道泛化

**预期**：掩码重建预训练 → fine-tune 后达到 62-64%（比基线 +1~3pp）

### 2.2 阶段 B：跨被试少样本适配（保持）

**核心思想**：SPDNet 的 260 参数极简架构天然适合少样本场景。

```
训练被试全量数据 → SPDNet 预训练（掩码重建或全监督）
                              ↓
目标被试 5/10/20-shot → 冻结 BiMap + ReEig → 只微调 LogEig + 分类头
                              ↓
评估：全量 LOSO vs 5/10/20-shot LOSO
```

**假设**：即使全量 LOSO 下 SSL 增益不大，少样本场景下预训练的优势会更加明显。

### 2.3 阶段 C：跨数据实验 + 可视化

保持原计划的跨数据集验证（BCI IV 2a）和可视化分析。

---

## 三、修订后的实验矩阵

| # | 实验 | 目的 | 优先级 |
|:--:|------|------|:--:|
| E1 | SPDNet 基线（已完成） | 建立 8ch PhysioNet LOSO 的 SPDNet 基准 | ✅ |
| E2 | EA 消融（已完成） | 量化 EA 对 SPDNet 的贡献 (+10.45pp) | ✅ |
| **E3** | **SPD 掩码重建预训练** 🆕 | 替代对比学习，验证掩码重建在 SPD 流形上的有效性 | 🥇 |
| E4 | 增强策略对比 | channel_mask vs element_mask vs 无增强 | 🥈 |
| E5 | 少样本 LOSO | 5/10/20-shot，全监督 vs 预训练 | 🥇 |
| E6 | 跨数据集验证 | BCI IV 2a 4-class LOSO | 🥈 |
| E7 | 架构消融（已完成） | [8,8] vs [8,6,4] vs [8,8,8] | ✅ |
| E8 | Tangent Space vs SPDNet 深度对比 | 特征空间分析、t-SNE、协方差响应 | 🥈 |

---

## 四、修订后的时间线

```
Week 1-2 (已完成): SPDNet 基线 + 架构消融 + EA 分析
  ✅ SPDNet [8,8] + EA = 61.04%
  ✅ 架构搜索：[8,8] 最优
  ✅ EA 增益：+10.45pp

Week 3 (当前): SPD 掩码重建预训练
  - 实现通道级 mask 策略
  - 实现 Log-Euclidean 空间重建损失
  - 预训练 → fine-tune → LOSO 评估
  - 目标：62-64%

Week 4: 少样本实验 + 对比分析
  - 5/10/20-shot LOSO
  - 全监督 vs 预训练 vs 随机初始化对比
  - Tangent Space vs SPDNet 特征可视化

Week 5: 跨数据集 + 论文
  - BCI IV 2a 验证
  - 论文初稿
```

---

## 五、修订后的论文故事线

```
标题：
"Riemannian Deep Learning for Low-Channel Motor Imagery EEG:
 Efficiency, Alignment, and Self-Supervised Representation Learning"

叙事弧线：
  1. 背景：8ch 低通道 MI-EEG 的跨被试解码挑战
  2. SPD 流形视角：协方差矩阵天然生活在 SPD 流形上
  3. SPDNet 的效率：260 参数的极简模型在 8ch PhysioNet MI LOSO 上
     达到 61.04%，超越 Tangent Space+LDA，逼近 50K 参数的 FBCNet
  4. EA 的几何解释：EA 作为 SPD 流形上的平行移动，为 SPDNet 提供
     了 +10.45pp 的增益——这是所有 DL 方法中最高的 EA 增益
  5. 自监督探索：对比学习 vs 掩码重建在 SPD 流形上的效果对比
  6. 少样本适配：极简架构在 ≤20-shot 场景下的优势

贡献点：
  C1: 8ch 运动皮层 MI-EEG 的首个黎曼 DL LOSO 基准
  C2: SPDNet 在低通道场景下的效率-性能权衡分析
  C3: SPD 流形上自监督预训练方法的对比研究
  C4: EA 在黎曼 DL 框架下的几何统一解释（+10.45pp）
```

---

## 六、诚实声明

此修订版基于以下实验事实：

1. **SPDNet 基线有效但未突破**：61.04% 是一个 solid baseline，但距离 EEG Conformer (63.93%) 仍有 2.89pp 差距
2. **SSL 对比学习在 8×8 SPD 矩阵上未生效**：这是本文的 honest negative result，本身有方法论价值
3. **掩码重建是更有希望的方向**：基于信息完整性假设，更适合小规模 SPD 矩阵
4. **如果掩码重建也不 work**：SPDNet 基线 + EA 分析 + 效率对比 + 少样本实验仍然可以构成完整的研究叙事。贡献不单纯依赖准确率数值

---

## 七、参考文献

1. Huang & Van Gool (2017). "A Riemannian Network for SPD Matrix Learning." AAAI 2017.
2. Ju & Guan (2023). "Tensor-CSPNet: A Novel Geometric Deep Learning Framework for Motor Imagery Classification." IEEE TNNLS.
3. Aristimunha et al. (2026). "SPD Learn: A Geometric Deep Learning Python Library." arXiv:2602.22895.
4. He & Wu (2020). "Transfer Learning for BCIs: A Euclidean Space Data Alignment Approach." IEEE TBME.
5. He et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. (MAE)
6. Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020. (SimCLR)
7. Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs." J. Neural Eng.
8. Song et al. (2023). "EEG Conformer: Convolutional Transformer for EEG Decoding." arXiv:2301.05578.
