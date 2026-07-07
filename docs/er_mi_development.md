# ER-MI 开发日志：Evidence Reasoning Network for MI-BCI

> 2026-07-04 ~ 2026-07-07

---

## 一、设计动机

把 MI-EEG 分类从"一次性前向推断"重新定义为**多步证据累积过程**：

```text
EEG → Encoder → Evidence Vector
       ↓
   GRU Cell × K steps:  h_t = GRU(evidence, h_{t-1})
       ↓
   Step Classifier:  logits_t = Linear(h_t)
       ↓
   Loss = CE(logits_K, y) + 0.3 * Σ CE(logits_t, y)
```

核心假设：**单次 forward 可能不够，让模型在同一个 evidence 上"多想几步"能提升决策质量。**

---

## 二、架构

```
EEG (B, C, T) → (B, 1, C, T)
  → Block 1: Conv2d(1→F1, k=(1,64)) + DepthwiseConv2d(F1→D*F1, k=(C,1))
      → BN + ELU + AvgPool(1,4) + Dropout
      → (B, 16, 1, 187)  for T=750
  → Block 2: Separable Conv2d (depthwise k=(1,16) + pointwise)
      → BN + ELU + AvgPool(1,8) + Dropout
      → (B, 16, 1, 23)
  → Flatten: (B, 368)
  → Evidence Projection: Linear(368, 64)
  → GRU Cell × K steps: h_t = GRU(evidence, h_{t-1})
  → Shared Classifier: Linear(64, n_classes) @ each step
  → Return: [logits_1, ..., logits_K]
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| F1 | 8 | 时间滤波器数 |
| D | 2 | 空间深度乘数 |
| F2 | 16 | 逐点滤波器 |
| hidden_dim | 64 | Evidence 维度 = GRU 隐藏维度 |
| steps (K) | 3 | 推理步数 |
| dropout | 0.5 | |
| 总参数 | ~10K | 极轻量 |

---

## 三、v1 → v2 迭代

| 版本 | 改动 | seed42 Acc | 结论 |
|------|------|:---:|------|
| v1 | 单 evidence vector + GRU×3 | **61.92%** | baseline |
| v2 | 4 token evidence (mu/beta/spatial/global) + Transformer + GRU×3 | **61.93%** | ❌ 无效 |

v2 的核心改动：
- 用 4 分支 tokenizer 替代全局 evidence projection
  - mu token: Conv1d(k=65) 针对 mu 节律
  - beta token: Conv1d(k=33) 针对 beta 节律
  - spatial token: channel attention + 1×1 conv
  - global token: adaptive avg pool
- 1 层 TransformerEncoder 做 token 交互
- 均值聚合 + GRU 推理

**结论：多 token evidence 不带来增益。v1 的单 evidence vector + GRU 已经足够。**

---

## 四、已有实验结果

### 4.1 3-seed LOSO

| Seed | Accuracy | Kappa |
|------|:---:|:---:|
| 42 | 61.92% ± 12.17% | 0.233 |
| 123 | 61.78% ± 8.97% | 0.231 |
| 456 | 63.56% ± 11.28% | 0.267 |
| **Mean** | **62.42% ± 0.82%** | **0.244** |

### 4.2 步数消融（analyze_er_mi_steps.py）

从步骤 1 到步骤 3 的准确率趋势已测量，但未系统扫 K=1/2/4/5。

### 4.3 多视角 Oracle 分析

| 指标 | 值 |
|------|-----|
| ER-MI 是最优模型的被试数 | **9/30**（所有模型中最多的！） |
| BRT-Det + ER-MI oracle | 68.08% (+4.00pp) |
| BRT-Det + ER-MI correlation | **r = 0.46** (最低之一) |
| ER-MI 共享失败被试 | S06, S17, S18, S20, S24 |

> **ER-MI 是被最低估的模型：均值只有 61.92%，但覆盖了 9/30 被试——超过任何其他单模型。**

### 4.4 ER-MI vs BRT-Det 职责分工

| 模型 | 最优被试 | 被试特征 |
|------|:---:|------|
| BRT-Det v8 | 14 人 | S01, S02, S05, S06, S10, S16, S17, S21, S22, S23, S24, S25, S28, S29 |
| ER-MI | 12 人 | S03, S04, S07, S08, S09, S12, S13, S14, S15, S18, S19, S27 |
| ER-MI v2 | 4 人 | S11, S20, S26, S30 |

> **两者覆盖几乎不相交。BRT-Det 偏"稳定被试"，ER-MI 偏"BRT-Det 搞不定的被试"。**

---

## 五、持续失败被试

| 被试 | ER-MI 3-seed 均值 | 特征 |
|------|:---:|------|
| S06 | 44.44% | 所有 seed 完全一致崩溃，κ 始终为负 |
| S09 | 50.37% | BCI-inefficient (within-subject CV=55.56%) |
| S18 | 48.89% | 所有 seed κ<0 |
| S17 | 52.59% | 偏向 rest 类 (recall_left≈0) |
| S20 | 53.33% | 和 S17 相同模式 |
| S24 | 51.85% | 跨 seed 波动大 |

---

## 六、关键结论（当前）

1. **GRU 多步推理 vs 单步的增益未经充分验证。** K=3 是默认值，未 sweep。
2. **v2 多 token evidence 无效。** 单 evidence vector 已经够用，token 分支引入复杂度但没有收益。
3. **ER-MI 和 BRT-Det 高度互补（r=0.46）。** 这是 ER-MI 在项目中最大的价值——不是单模型 SOTA，而是最稀缺的互补视角。
4. **ER-MI 用 raw EEG，没有 filter bank。** BRT-Det 的 Band Gate 突破来自频带自适应选择——ER-MI 可能也存在同样的瓶颈。
5. **Evidence projection 极简单（单层 Linear）。** 当前直接把 368 维 flatten 特征映射到 64 维，没有任何中间表示学习。

---

## 七、尚未尝试的方向

| 优先级 | 方向 | 理由 | 预期 |
|:---:|------|------|:---:|
| 1 | **Step sweep (K=1~5)** | 验证多步推理是否真有效，找最优 K | 快速诊断 |
| 2 | **Filter bank 输入** | BRT-Det 的 Band Gate 是最大突破，ER-MI 可能同理 | 可能 +2-3pp |
| 3 | **Deeper evidence projection** | Linear(368→64) 太简单 → MLP(368→128→64) | 快速验证 |
| 4 | **GRU → LSTM / TCN** | GRU 可能不是最优推理单元 | 中等 |
| 5 | **Per-step attention** | 不同步看 evidence 的不同部分 | 复杂 |
| 6 | **Aux loss weight sweep** | 当前 0.3 是硬编码 | 快速扫 |
| 7 | **Dropout sweep** | 当前 0.5 可能过高/过低 | 快速扫 |

---

## 八、当前定位（2026-07-07）

### 核心判断

> **ER-MI 不是单模型冲 SOTA 的主力，而是多模型融合里的"互补模型"。**

```text
BRT-Det v8: 负责稳定被试 (14/30 最优)
ER-MI:      负责 BRT-Det 搞不定的一批被试 (12/30 最优)
ER-MI v2:   目前不值得继续加复杂度
```

### 不建议继续做

```text
ER-MI v3 = 更多 token + 更复杂 Transformer   ← v2 已证无效
ER-MI v3 = 更深 GRU / 更多 attention         ← 没找到瓶颈就堆模块
ER-MI v3 = 更复杂 reasoning 模块             ← 盲目堆
```

---

## 九、Phase 1 诊断：K Sweep + MLP + Filter Bank（2026-07-07）

### 9.1 K Sweep（单折 S07 + S09）

| K | S07 (强) | S09 (弱) | 判断 |
|:---:|:---:|:---:|:---:|
| 1 | 80.00% | 48.89% | 无推理 baseline |
| **2** | **86.67%** | **51.11%** | ✅ 两端最优或次优 |
| 3 | 64.44% | 46.67% | ❌ 默认值反而是最差的 |
| 4 | 77.78% | — | 中等 |
| 5 | 91.11% | 42.22% | ⚠️ 极化效应 |

> **K=3（默认值）是局部最低点。K=2 最安全，K=5 有极化效应。**

### 9.2 K=2 全 LOSO

| | K=3 (default) | K=2 | Δ |
|------|:---:|:---:|:---:|
| Acc | 61.92% | **62.96%** | **+1.04pp** |
| κ<0 | 4 | 3 | -1 |

有效但温和——在不同被试之间搬收益，净正。

### 9.3 MLP Evidence Projection ❌

| Subject | depth=1 (50K) | depth=2 (120K) |
|------|:---:|:---:|
| S07 | 51.11% | 48.89% |
| S09 | 53.33% | 53.33% |

> **加深 projection 无效。和 v2 多 token 一样——增加参数反而更差。29 个训练被试撑不起 120K 参数。**

### 9.4 🔥 Filter Bank (FB-ER-MI)

和 BRT-Det 一样的突破路线：raw EEG → 6-band filter bank → shared encoder per band → band gate → GRU reasoning。

| 版本 | Acc | Kappa | κ<0 | Δ |
|------|:---:|:---:|:---:|:---:|
| ER-MI K=3 (raw, 默认) | 61.92% | 0.233 | 4 | — |
| ER-MI K=2 (raw) | 62.96% | 0.255 | 3 | +1.04pp |
| **FB-ER-MI K=2** | **64.59%** | **0.291** | **1** | **+2.67pp** |

**最大提升：S26 53%→87% (+33pp), S22 44%→60% (+16pp), S01 53%→67% (+13pp)**

### 9.5 Phase 1 总结

```text
诊断轮结果:

K=3 → K=2:   61.92% → 62.96%  (+1.04pp)  ✅ 改默认值
MLP depth=2:  无效                     ❌ 和v2一样
Filter bank:  62.96% → 64.59%  (+1.63pp)  🔥 最大突破

总计: 61.92% → 64.59% (+2.67pp)
κ<0:  4 → 1

和 BRT-Det 完全一样的剧本:
  Filter Bank + Band Gate 是核心突破
  加深 projection/加 token 无效
```

### 9.6 纯自研 Oracle 更新

| | Before | After |
|------|:---:|:---:|
| 最佳单模型 | BRT-Det 64.08% | **FB-ER-MI 64.59%** |
| Oracle (3 models) | 68.52% | **69.04%** |
| 共享失败 | S06, S09, S17, S18 | **S06, S09** |
| 到 70% 差距 | 1.48pp | **0.96pp** |

> **FB-ER-MI 现在是最强自研单模型（64.59%），首次超过 BRT-Det v8（64.08%）。** 纯自研 oracle 离 70% 只差 0.96pp。

---

## 十、FB-ER-MI 最终架构

```text
EEG (B, 8ch, 750T)
  → Filter Bank (6 bands, scipy filtfilt)
  → Shared EEGNet Encoder (Block1 + Block2) per band
  → Per-band Evidence Vector: Linear(flatten → 64)
  → Band Gate: Linear(64→32→1) + sigmoid → per-band weight
  → Fused Evidence: Σ(gate_b * evidence_b)
  → GRU Cell × 2 steps: h_t = GRU(evidence, h_{t-1})
  → Step Classifier: Linear(64, n_classes) @ each step
  → Return: [logits_1, logits_2] (train) / logits_2 (eval)
```

**参数：52,051 | 训练：K=2, intermediate_loss_weight=0.3, Adam lr=1e-3, CosineAnnealing**

---

## 十一、关键结论（更新）

1. **K=3 是坏的默认值。** K=2 更安全，K=5 有极化效应。
2. **v2 多 token 无效。** 和 MLP depth=2 一样——加复杂度不带来收益。
3. **Filter Bank 是 ER-MI 的最大突破。** 和 BRT-Det 一模一样——频带分离 + band gate = 核心。
4. **FB-ER-MI (64.59%) 超过 BRT-Det v8 (64.08%)，成为最强自研单模型。**
5. **ER-MI 和 BRT-Det 仍然高度互补（r=0.46），FB-ER-MI 覆盖 13/30 被试。**
6. **纯自研 oracle = 69.04%，离 70% 差 0.96pp。** 共享失败仅剩 S06/S09。

---

## 十二、3-Seed + Ablation 完整结果（2026-07-07）

### 12.1 FB-ER-MI 3-Seed

| Seed | Acc | Kappa | κ<0 |
|------|:---:|:---:|:---:|
| 42 | 64.59% | 0.291 | 1 |
| 123 | 63.85% | 0.273 | 3 |
| 456 | 63.48% | 0.267 | 6 |
| **Mean** | **63.97% ± 0.46%** | **0.277** | — |

vs ER-MI K=3 原始 3-seed: 62.42% ± 0.82% → **+1.55pp, seed 稳定性更好（±0.46% vs ±0.82%）**

### 12.2 Ablation Chain (seed42)

| Variant | Acc | Kappa | κ<0 | Δ |
|------|:---:|:---:|:---:|:---:|
| Raw K=3 (baseline) | 61.92% | 0.233 | 4 | — |
| Raw K=2 | 62.96% | 0.255 | 3 | +1.04pp |
| FB no gate K=2 | **64.67%** | 0.288 | 3 | **+1.71pp** |
| FB gate K=2 | 64.59% | 0.291 | **1** | -0.08pp |

> **和 BRT-Det 完全一样的剧本！Filter bank 提供主要准确率增益，Band gate 改善稳定性（κ<0 3→1）。FB no gate 的准确率甚至略高于 gate 版（64.67% vs 64.59%），但 gate 版 κ<0 仅 1 人。**

### 12.3 论文逻辑链

```text
Raw EEG + EEGNet encoder → 61.92%
  + K=2 fix → 62.96% (+1.04pp, 证明 K=3 是坏的默认值)
  + Filter bank → 64.67% (+1.71pp, 证明频带分离是主要瓶颈)
  + Band gate → 64.59% (−0.08pp acc, κ<0 4→1, 证明 gate 改善稳定性)
```

### 12.4 S06/S09 状态

| 被试 | FB-ER-MI seed42 | FB-ER-MI seed123 | FB-ER-MI seed456 |
|------|:---:|:---:|:---:|
| S06 | 53.33% (κ=+0.05) | 42.22% (κ=-0.17) | 44.44% (κ=-0.11) |
| S09 | 53.33% (κ=+0.04) | 60.00% (κ=+0.17) | 53.33% (κ=+0.03) |

> S09 在 seed123 到 60%！不是完全无解。S06 仍在 42-53% 波动。

---

## 十三、最终定性和路线（2026-07-07）

### 主结果（论文用）

```text
FB-ER-MI: 63.97% ± 0.46% (3-seed mean), κ = 0.277
vs ER-MI K=3: 62.42% ± 0.82%, +1.55pp, seed 稳定性更好
```

> **论文主表写 63.97%（3-seed mean），不是 64.59%（seed42 单次）。64.59% 放在 best-seed 分析或 ablation 里。**

### Ablation 一句话总结

```text
Filter bank improves average accuracy (+1.71pp),
while band gate improves cross-subject robustness (κ<0: 3→1).
```

中文：

```text
滤波器组主要提升平均准确率，频带门控主要改善跨被试鲁棒性。
```

### 最终模型选择：FB gate，不是 no gate

虽然 FB no gate (64.67%) 略高于 FB gate (64.59%)，但 gate 版 κ<0 从 3→1。论文写法：

> Although mean fusion achieves marginally higher accuracy, the gated variant substantially reduces negative-kappa subjects, indicating improved cross-subject robustness. Therefore FB-ER-MI with band gate is selected as the final model.

### 当前风险

```text
seed42:  κ<0 = 1
seed123: κ<0 = 3  
seed456: κ<0 = 6 (S03, S05, S06, S09, S18, S24)
→ S06 所有 seed collapse, 其余是 seed 间不稳定
→ S09/S18 与 BRT-Det 共享, S03/S05/S06/S24 FB-ER-MI 特有
```

---

## 十四、ER 优化收尾：Aux Loss + Gate 变体（2026-07-07）

### Aux Loss Sweep

| weight | S06 | S09 | S05 | 结论 |
|:---:|:---:|:---:|:---:|------|
| 0.0 | 46.7% | 57.8% | 55.6% | |
| **0.1** | **57.8%** | 55.6% | 42.2% | S06 最优 |
| **0.2** | 46.7% | **62.2%** | **71.1%** | S09/S05 最优 |
| 0.3 | 40.0% | 48.9% | 53.3% | ← 默认值，最差 |
| 0.5 | 46.7% | 51.1% | 60.0% | |

> **默认 aux_loss=0.3 在三个弱被试上都是最差或接近最差。但降至 0.1-0.2 的单折增益未能在全 LOSO 中保持（+0.08pp，不显著）。**

### Gate Mode Sweep（单折）

| mode | S06 | S09 | S05 | S18 | 结论 |
|------|:---:|:---:|:---:|:---:|------|
| sigmoid | 53.3% | 57.8% | 51.1% | 46.7% | baseline |
| softmax | **62.2%** | 46.7% | 55.6% | 48.9% | S06 有效，S09 差 |
| residual | 55.6% | **62.2%** | 53.3% | 48.9% | S09 有效，S06 一般 |

> **单折上有分化，但全 LOSO 后 residual 仅 64.30%（-0.29pp vs sigmoid）。单折增益不可靠。**

### 最终结论

```text
FB-ER-MI sigmoid gate + aux_loss=0.3 已是最优配置。

微调 aux_loss / gate_mode 在单折上看似有戏，
但全 LOSO 后回归均值 — 单模型优化已触达天花板。

ER 单模型: 64.59% (seed42), 63.97% (3-seed)
进一步优化应转向融合而非单模型。
```

---

## 十五、FB-ER-MI + BRT-Det 合法融合（2026-07-07）

### 结果（α=0.5, seed42, with EA）

| 模型 | Acc | Kappa | κ<0 |
|------|:---:|:---:|:---:|
| FB-ER-MI | 64.74% | 0.289 | 3 |
| BRT-Det v8 | 62.52% | 0.249 | 4 |
| **Ensemble α=0.5** | **66.00%** | **0.317** | **3** |

**Gain: +1.26pp over best single model。**

### 对比

| Ensemble | Acc | Gain |
|------|:---:|:---:|
| Conformer + BRT-Det (α=0.7) | 64.37% | +0.96pp |
| **FB-ER-MI + BRT-Det (α=0.5)** | **66.00%** | **+1.26pp** |

> **自研双模型融合优于 Conformer+BRT 融合。FB-ER-MI 和 BRT-Det 是当前最优自研组合。**

### 纯自研 Oracle 最终版

| 模型 | Acc | 最优被试数 |
|------|:---:|:---:|
| FB-ER-MI K=2 | 64.59% | 13 |
| BRT-Det v8 | 64.08% | 13 |
| ER-MI v2 | 61.93% | 4 |
| **Oracle (3 models)** | **69.04%** | — |

```text
合法融合已到 66.00%，离 oracle 69.04% 差 3.04pp。
下一步: adaptive routing / few-shot calibration 缩小 oracle gap。
```

### 项目主线演进

```text
第一阶段: BRT-Det 证明 region/band/time evidence detection 有效
第二阶段: ER-MI 证明 multi-step evidence reasoning 有互补价值
第三阶段: FB-ER-MI 把 filter bank + evidence reasoning 合并
          → 63.97% (3-seed)，成为论文主模型之一
第四阶段: FB-ER-MI + BRT-Det 合法融合 → 目标 67-69%
```

### 当前模型定位

| 模型 | 3-seed Acc | 角色 |
|------|:---:|------|
| **FB-ER-MI K=2** | **63.97% ± 0.46%** | 论文主模型 |
| BRT-Det v8 | 63.02% ± 1.42% | 结构解释性强，互补 |
| ER-MI v2 | 61.93% | 补充视角 |

### FB-ER-MI 一句话定位

```text
中文: 一种基于滤波器组证据提取、频带门控融合与两步循环推理的
      轻量级运动想象 EEG 分类模型。

英文: A lightweight filter-bank evidence reasoning network with
      band-wise gating and two-step recurrent decision refinement.
```

### 下一步（融合路线）

| 优先级 | 实验 | 目的 |
|:---:|------|------|
| 1 | α sweep (0.0-1.0) | 找最优固定权重 |
| 2 | 3-seed fixed ensemble | 确认 66.00% 可复现 |
| 3 | 统一实验口径 | FB-ER-MI/BRT-Det/Ensemble 同一批 fold 报告 |
| 4 | 无标签 adaptive router | 缩小 oracle gap (69.04% - 66.00% = 3.04pp) |

### 项目最终路线

```text
单模型:  FB-ER-MI 63.97% ± 0.46% (3-seed)
固定融合: FB-ER-MI + BRT-Det = 66.00% (seed42)
自适应:  router / entropy-based → 目标 67-69%
Oracle:  69.04% (3 自研模型), 71.04% (7 模型)
```

### 不建议继续做

```text
- ER 单模型结构优化                ← 已触天花板
- 更深 MLP / 更多 token            ← 已证无效
- GRU → LSTM / per-step attention  ← 未找到瓶颈
- Transformer evidence reasoning   ← v2 路线已封存
```

---

## 十六、Related Work 定位（2026-07-07）

### 与已有工作的关系

自研模型不是凭空出现，而是将 MI-EEG 中成熟思想重新组合：

| 已有方向 | 相似度 | 自研模型中的体现 |
|------|:---:|------|
| EEGNet (Lawhern 2018) | 高 | ER-MI / FB-ER-MI encoder backbone |
| FBCNet (Bakshi 2021) | 高 | FB-ER-MI / BRT-Det 的 filter bank 输入 |
| CNN-GRU (多篇) | 中 | ER-MI 的 CNN + GRU 组合 |
| EEG-TCNet (Ingolfsson 2020) | 中 | BRT-Det 的 temporal stem + dilated conv |
| FBMSNet / MSFBCNN | 中 | 滤波器组 + CNN 的 multi-view 思路 |
| MBEEGSE / MBEEGCBAM | 中 | Band gate / channel attention |
| EEG Conformer (Song 2023) | 中 | ER-MI v2 的 token + Transformer |
| ATCNet / TCFormer | 中 | multi-branch + attention |

### 自研辨识度：不是基础模块新，是组合和框架新

```text
不是: 首次提出 filter bank / EEGNet encoder / attention / GRU
而是: 提出了一套面向跨被试 MI-EEG 的证据建模框架

BRT-Det:
  将 MI-EEG 分类重构为 band-channel-time 网格上的分布式证据检测问题，
  每个单元预测 objectness + class evidence，通过 objectness 加权聚合。
  → 这是最有自研辨识度的部分

ER-MI:
  引入循环式证据精炼机制：EEG encoder → global evidence vector →
  GRU 对同一证据多步决策修正。
  → CNN-GRU 已有很多，但 same-evidence repeated reasoning 相对有差异

FB-ER-MI:
  将 filter bank、band gate、GRU reasoning 组合起来，
  消融证明 filter bank 涨准确率、band gate 降 collapse。

BRT-Det + FB-ER-MI 融合:
  利用互补归纳偏置：BRT-Det 检测局部分布式证据，
  FB-ER-MI 做全局证据推理，互补性真实存在 (r<0.5)。
```

### 最严谨的定位

```text
不是发明了全新的基础模块，
而是提出了一套面向跨被试 MI-EEG 的证据建模框架：
BRT-Det 做局部证据检测，
FB-ER-MI 做全局证据推理，
二者通过互补融合提升跨被试泛化。
```

### 推荐代码参考

| 方向 | 推荐仓库 |
|------|------|
| EEGNet backbone | `vlawhern/arl-eegmodels`, Braindecode |
| Filter-bank CNN | FBCNet (arXiv:2104.01233) |
| Temporal CNN | `iis-eth-zurich/eeg-tcnet` |
| Attention EEG | `Altaheri/EEG-ATCNet` |
| Fusion baseline | MOABB, Braindecode pipelines |
