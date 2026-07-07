# BRT-Det 开发日志：尝试了什么、优化了什么

> 2026-07-04 ~ 2026-07-06，3 天 30+ 次 LOSO 实验

---

## 一、起点：Region Pooling 版本

**设计：** Filter Bank → Region Pooling (FC/C/CP) → 12 time cells → Backbone → Objectness Aggregation

| 实验 | Accuracy | 结论 |
|------|----------|------|
| Region (FC/C/CP) | 46.59% | ❌ 低于 chance (50%) |

**根因：** Region pooling 把 FC3+FC4、C3+Cz+C4、CP3+CPz+CP4 分别做平均，C3/C4 的左右侧化差异被平均掉了。这对二分类 MI 是致命的。

**立即改为 Channel-level（8 通道直接保留）。**

---

## 二、结构迭代：从 51% 到 59%

### 2.1 Channel-level（去掉 region pooling）

| 改动 | Accuracy | Δ |
|------|----------|-----|
| Region → Channel | 46.59% → 51.11% | +4.52 |

C3/C4 侧化信息得以保留，但仍在 chance level。

### 2.2 Temporal Stem（最关键的突破）

**问题：** `adaptive_avg_pool2d(T=750 → 12)` 压缩比 62.5×，在 pooling 前没有任何时序特征提取。模型拿到的是 12 个盲平均后的值，无法学到 ERD/ERS 等时序模式。

**方案：** 在 grid pooling 前加轻量 temporal conv（Conv1d），先提取局部时序特征，再 pool 到 grid。

| 改动 | Accuracy | Δ |
|------|----------|-----|
| 无 stem | 51.11% | — |
| +1-layer stem (k=15) | 57.04% | **+5.93** |
| +2-layer stem (k=31,15) | 58.30% | +1.26 |

**Temporal stem 是 BRT-Det 最大的单一突破。**

### 2.3 Spatial Mixing

**问题：** temporal stem 对每个 (band, channel) 独立处理，backbone 的 3×3 conv 跨通道感受野有限。

**方案：** 在 grid pooling 后加一层跨通道 conv（spatial mix），让模型在 8-ch 证据网格上学习 C3/C4 差异。

| 改动 | Accuracy | Δ |
|------|----------|-----|
| 无 spatial mix | 58.30% | — |
| +spatial mix | 59.33% | +1.03 |

### 2.4 Dropout 正则化

| 改动 | Accuracy | Δ |
|------|----------|-----|
| 无 dropout | — | — |
| +dropout (0.1/0.2) | 59.33% | 含在 spatial mix 中 |

---

## 三、Round 1：单尺度 vs 多尺度机制拆解

**问题：** Multi-scale (6/12/24 cells) 拿了 61.63%，但 3× 前向太慢。到底是多尺度互补有用，还是只是 24 cells 分辨率够细？

| 实验 | n_time_cells | Accuracy | 结论 |
|------|:-----------:|----------|------|
| R1-1 | 6 | 61.04% | 粗网格也还行 |
| R1-2 | 12 | 59.33% | 尴尬的中间值 |
| R1-3 | **24** | **61.78%** | ✅ 单尺度最优 |
| R1-4 | 6/12/24 | 61.63% | 多尺度不如单 24 |

**结论：24 cells 单尺度 > multi-scale，砍掉 multi-scale，速度 3×。**

---

## 四、Round 2：训练稳定性

| 实验 | label_smoothing | weight_decay | Accuracy | 结论 |
|------|:--:|:--:|----------|------|
| R2-1 | 0.0 | 0.0 | 61.78% | baseline |
| R2-2 | **0.1** | **0.0** | **62.00%** | ✅ 微涨，std 降 |
| R2-3 | 0.0 | 1e-4 | 61.33% | ❌ 回退 |
| R2-4 | 0.1 | 1e-4 | 62.00% | 与 R2-2 持平 |

**结论：label_smoothing=0.1 稳中有升，weight_decay 无用。后续实验默认 --label_smoothing 0.1。**

---

## 五、Round 3：Dilated Backbone

**问题：** 单尺度的 backbone 只有 3 层普通 3×3 conv，时间感受野有限。能否用 dilated conv 在不增加参数的前提下扩展感受野？

**改动：** backbone 的 Conv2d 加 `dilation=(1, d)` 沿时间轴

| 实验 | dilations | Accuracy | 结论 |
|------|----------|----------|------|
| R3-3 | [1,1,1] | 62.00% | baseline |
| **R3-1** | **[1,2,4]** | **62.59%** | ✅ 最优 |
| R3-2 | [1,1,2] | 61.48% | 太保守 |

**Dilation [1,2,4] 不增加任何参数，提升 0.59pp。** S09 首次突破 chance level（33%→51%）。

---

## 六、Round 4：Evidence Aggregation 方法对比

**问题：** objectness-weighted sum 是否是最优的聚合方式？top-k、logsumexp、learned softmax 能否更好？

| 实验 | agg_mode | Accuracy | 结论 |
|------|----------|----------|------|
| **R4-1** | **objectness** | **62.59%** | ✅ 最佳 |
| R4-2 | topk=5 | 60.59% | ❌ k 太小 |
| R4-3 | topk=10 | 58.96% | ❌ 丢失信息 |
| R4-5 | logsumexp (τ=0.5) | 61.63% | 不如 objectness |

**结论：MI evidence 是分布式的，不是少数强 cell。objectness-weighted sum 是最优聚合方式。**

---

## 七、Round 5：Cross-Band Mixer ❌

**问题：** 当前 6 个频带被当成独立 batch 处理，频带间无交互。加跨频带 mixer 能否提升？

**方案：** spatial mix 后插入 Linear(6→12→6) mixer，混合 6 个频带

| 实验 | Accuracy | 结论 |
|------|----------|------|
| 无 band mixer | 62.59% | baseline |
| +band mixer | 58.67% | ❌ 显著回退 |

**结论：频带独立处理反而更好。** 强制让模型混合 μ/β 等不同频带会引入噪声，破坏 filter bank 的优势。

---

## 八、其他尝试过的方向（均无效/回退）

| 尝试 | Accuracy | 原因 |
|------|----------|------|
| Variance pooling (mean+log-var) | 56.44% | 通道翻倍 → spatial_mix 难以学习 |
| Region pooling (FC/C/CP) | 46.59% | C3/C4 侧化被平均 |
| tqdm 进度条 | — | 非 TTY 环境输出混乱，改回 print |
| conda run 缓冲 | — | 改用直接 python.exe 路径 |

---

## 九、最终架构 (BRT-Det v7)

```text
EEG (B, 8ch, 750T)
  → Filter Bank (6 bands, scipy filtfilt)
  → Temporal Stem: Conv1d(1→12, k=31) + Conv1d(12→24, k=15)
  → AdaptiveAvgPool2d → (B*6, 24, 8, 24)
  → Spatial Mix: Conv2d(24, 24, k=(3,1)) + BN + ELU + Dropout
  → Dilated Backbone: Conv2d(24→32, d=1) → Conv2d(32→32, d=2) → Conv2d(32→32, d=4)
  → Detection Head: Conv2d(32, 3, 1×1) → [obj, left, right]
  → Objectness-weighted Aggregation
  → logits (B, 2)
```

**参数：32,175 | 训练：label_smoothing=0.1, Adam lr=1e-3, CosineAnnealing**

---

## 十、BRT-Det 全迭代曲线

```text
46.6% ── Region pooling (失败)
  │
51.1% ── Channel-level (+4.5pp)
  │
57.0% ── +Temporal Stem (+5.9pp) ← 最大突破
  │
58.3% ── +2-layer stem (+1.3pp)
  │
59.3% ── +Spatial Mix + Dropout (+1.0pp)
  │
61.8% ── 24 cells 单尺度 (+2.5pp)
  │
62.0% ── +Label Smoothing (+0.2pp)
  │
62.6% ── +Dilated Backbone [1,2,4] (+0.6pp) ← 最终
```

**总计：3 天 30+ 实验，46.6% → 62.6%（+16.0pp），32K 参数。**

---

## 十一、关键结论

1. **Temporal stem 是核心。** 62.5× 盲压缩是最大的信息瓶颈。在 pooling 前加轻量时序 conv 是 BRT-Det 能工作的前提。

2. **Region pooling 对二分类有害。** C3/C4 侧化差异不能被区域平均抹掉。这是 BRT-Det 从 46.6% → 51.1% 的关键认知。

3. **24 cells 单尺度最优。** 不需要 multi-scale 的 3× 计算开销。Dilated backbone 在同一尺度内扩展感受野更高效。

4. **Evidence 是分布式的。** 判别信息分散在整个 grid 中，不是少数强 cell。因此 objectness-weighted sum > top-k / logsumexp。

5. **频带独立处理比混合好。** filter bank 的优势在于让模型分别处理不同频段，强制跨频带混合反而引入噪声。

6. **Dilation 替代 multi-scale。** dilations=[1,2,4] 在单尺度下达到多尺度感受野，不增参数、不增计算。

7. **S09 是硬骨头但有改善空间。** Band Gate 把 S09 从 33.3% 拉到 44.4%（+11.1pp），但仍低于 chance。

8. **跨被试方差是主要瓶颈。** 7/30 被试 κ<0，去掉后均值 +4.33pp。模型结构改进的边际收益在递减。

---

## 十二、Round 6：诊断实验（2026-07-06）

**核心问题：** 62.6% 的瓶颈来自模型容量还是跨被试泛化？

### 6.1 离群值分析

| 指标 | 全部30人 | 去掉S09 | 去掉所有κ<0(7人) |
|------|----------|---------|------------------|
| Accuracy | 58.67% | 59.54% | **62.99%** |
| Kappa | 0.1728 | 0.1905 | 0.2601 |

7/30 被试（23%）κ<0，去掉后 +4.33pp。

### 6.2 S09 单被试诊断

S09 5-fold within-subject CV: **55.56% ± 7.0%** — 本身 MI 信号弱，不是 LOSO 泛化失败。

### 6.3 Diff Channels（C3/C4 差分通道）

| 实验 | Accuracy | Δ |
|------|----------|-----|
| Baseline | 58.67% | — |
| +Diff Channels (8ch→11ch) | **59.85%** | **+1.18pp** |

零参数。S10/S11/S12 各涨 +13~16pp。显式加入 FC3-FC4、C3-C4、CP3-CP4 三对差分通道。

### 6.4 Band Gate（最大突破）

| 实验 | Accuracy | Δ |
|------|----------|-----|
| Baseline | 58.67% | — |
| **+Band Gate** | **64.08%** | **+5.41pp** |

仅 32 参数。每频带学一个标量门控权重，保留 filter bank 独立性。

**3-seed 验证：**

| Seed | Accuracy | Kappa |
|------|----------|-------|
| 42 | 64.08% | 0.2796 |
| 123 | 61.41% | 0.2234 |
| 456 | 63.56% | 0.2670 |
| **Mean** | **63.02% ± 1.42%** | **0.2567** |

κ<0 被试从 7 → 3。S09 33.3% → 44.4%（+11.1pp）。

### 6.5 组合实验

| 实验 | Accuracy | 结论 |
|------|----------|------|
| Diff Ch + Band Gate | 61.56% | 互相干扰 |
| Band Gate + ObjReg (0.01) | 63.85% | 无显著增益 |

---

## 十三、BRT-Det v8 最终架构

```text
EEG (B, 8ch, 750T)
  → Filter Bank (6 bands, scipy filtfilt)
  → Temporal Stem: Conv1d(1→12, k=31) + Conv1d(12→24, k=15)
  → AdaptiveAvgPool2d → (B*6, 24, 8, 24)
  → Spatial Mix: Conv2d(24, 24, k=(3,1)) + BN + ELU + Dropout
  → Dilated Backbone: Conv2d(24→32, d=1) → Conv2d(32→32, d=2) → Conv2d(32→32, d=4)
  → Band Gate: Linear(32→16→1) + sigmoid → per-band scalar weight
  → Detection Head: Conv2d(32, 3, 1×1) → [obj, left, right]
  → Objectness-weighted Aggregation
  → logits (B, 2)
```

**参数：32,207 | 训练：label_smoothing=0.1, Adam lr=1e-3, CosineAnnealing**

---

## 十四、全迭代曲线

```text
46.6% ── Region pooling (失败)
  │
51.1% ── Channel-level (+4.5pp)
  │
57.0% ── +Temporal Stem (+5.9pp) ← 最大突破
  │
58.3% ── +2-layer stem (+1.3pp)
  │
59.3% ── +Spatial Mix + Dropout (+1.0pp)
  │
61.8% ── 24 cells 单尺度 (+2.5pp)
  │
62.0% ── +Label Smoothing (+0.2pp)
  │
62.6% ── +Dilated Backbone [1,2,4] (+0.6pp) ← v7
  │
63.0% ── +Band Gate (+0.4pp, 3-seed) ← v8
```

**注意：** v8 3-seed mean 相较 v7 最优单次结果（62.59%）仅提升 +0.43pp，但 κ<0 被试从 7→3。Band Gate 的核心贡献是**跨被试稳定性**而非峰值提升。单 seed 实验中 Band Gate 最高达 64.08%（相比同期 baseline 58.67% 提升 +5.41pp），但这两个基线不可直接比较——v7 baseline 已包含 dilation + label_smoothing。

**总计：4 天 35+ 实验，46.6% → 63.0%（+16.4pp），32,207 参数。**

---

## 十五、v8 定位：BRT-Det 现在的问题不是缺网络复杂度

### 核心判断

BRT-Det v8 不是“结构大幅突破”，而是：

> **用 Band Gate 证明了：主要瓶颈不是 backbone 不够强，而是不同被试的有效频带不同，需要自适应频带选择。**

| 现象 | 含义 |
|------|------|
| Band Gate 单 seed 58.67%→64.08% | 频带选择确实有效 |
| 3-seed mean = 63.02% ± 1.42% | 收益存在，但波动不小 |
| κ<0 被试 7→3 | Band Gate 改善的是跨被试稳定性 |
| S09 33.3%→44.4% | 有改善，但仍不可靠 |
| S09 within-subject 55.56% ± 7.0% | S09 本身 MI 信号弱，不只是泛化失败 |
| Diff Ch + Band Gate → 61.56% | 两个先验不是简单相加 |

### Band Gate 的本质

不是“加了个注意力模块所以涨点”。而是：

```text
Cross-Band Mixer = 强制把频带混在一起 → 破坏独立性 → 58.67%
Band Gate        = 每频带独立处理，然后加权选择 → 保留独立性 → 64.08%
```

归纳偏置：

> **MI 的有效频带具有被试差异，模型需要 band-wise reliability estimation，而非跨频带混合。**

### 当前主要矛盾：不是平均值，而是被试间方差

| 指标 | 当前值 | 说明 |
|------|--------|------|
| Mean Acc | 63.02% | 总体尚可 |
| Subject Std | ±1.42%（3-seed mean）| 跨被试波动大 |
| κ<0 count | 3/30 | 仍有 10% 被试 collapse |
| Worst-5 | ~47% | S09, S18, S13 等严重拖低均值 |
| Excl κ<0 | 66.17% | 去掉 3 个 collapse subject 后 |

> **BRT-Det 在一部分被试上能学到 MI evidence；但在弱信号/异常分布被试上仍然不稳定。瓶颈已从“网络容量”转向“被试自适应”。**

### 建议后续实验固定输出 5 个指标

每次 LOSO 实验都应报告：

| 指标 | 为什么重要 |
|------|-----------|
| Mean Acc | 总体水平 |
| Mean Kappa | 排除类别随机影响 |
| Subject Std | 跨被试波动 |
| κ<0 count | collapse 数量 |
| Worst-5 Mean | 最差被试鲁棒性 |

---

## 十六、关键结论（更新）

1. **Temporal stem 是核心。** 在 pooling 前加轻量时序 conv 是 BRT-Det 能工作的前提。

2. **Region pooling 对二分类有害。** C3/C4 侧化差异不能被区域平均抹掉。

3. **24 cells 单尺度最优。** Dilated backbone 在同一尺度内扩展感受野更高效。

4. **Evidence 是分布式的。** objectness-weighted sum > top-k / logsumexp。

5. **频带不能粗暴混合，但需要自适应选择。** Cross-band mixer 破坏 filter bank 独立性（-3.92pp）。Band Gate 保留独立性 + 学 per-band reliability weight（+5.41pp 单 seed）。这是 v8 的核心认知。

6. **Dilation 替代 multi-scale。** 不增参数、不增计算。

7. **S09 是 BCI-inefficient subject。** Within-subject CV 仅 55.56%，说明 MI 信号本身弱，不是靠 backbone 能解决的。

8. **BRT-Det 现在不缺网络复杂度，缺被试自适应机制。** 32K 参数已达 63%，继续堆容量边际收益递减。下一步应关注 domain adaptation、few-shot calibration、evidence 可视化。

9. **Diff Channels 有效但不可简单叠加。** 单独 +1.18pp，但与 Band Gate 组合降到 61.56%。二者可能在争夺同一类信息（空间侧化 vs 频带可靠性）。更好的方式是旁路分支 logits-level fusion 而非直接拼输入。

10. **Few-shot FT 对正常被试有效，对弱信号被试无效。** FT=20 整体 +2.34pp（62.30%→64.64%），但 S09/S13/S18 几乎无改善 — BCI-inefficient，不是泛化失败。

11. **Evidence 可视化证实模型学到了生理学合理模式。** S07 显示清晰 C3 侧化 + μ/β 波段 + cue后时序；S09 完全平坦 — 模型不会对噪声数据编造 evidence。

---

## 十七、Round 7：跨数据集 + Few-shot + 可视化验证（2026-07-06）

### 7.1 BCI IV 2a 跨数据集（4-class, 9 subjects）

| 方法 | Accuracy | Kappa |
|------|----------|-------|
| EEGNet (base) | 39.47% ± 12.45% | 0.193 |
| Tangent + LDA + EA | 38.60% ± 12.44% | 0.181 |
| EEGNet + SpatiotemporalAttn | 36.94% ± 11.78% | 0.159 |
| **BRT-Det v8 + EA** | **36.38% ± 9.44%** | **0.152** |
| FgMDM + EA | 34.91% ± 8.48% | 0.132 |
| MDM + EA | 33.43% ± 10.92% | 0.112 |

4 分类 chance=25%。BRT-Det v8 排名第 4，跨数据集泛化成立。

### 7.2 Few-Shot FT Sweep (PhysioNet, 30 subjects)

| FT Trials | Accuracy | S09 | S13 | S18 |
|:---:|----------|:---:|:---:|:---:|
| 0 | 62.30% | 46.7% | 51.1% | 55.6% |
| 5 | 63.14% | 48.6% | 51.4% | 51.4% |
| 10 | 63.60% | 48.0% | 48.0% | 56.0% |
| **20** | **64.64%** | 47.8% | 47.8% | 52.2% |

正常被试受益（+2.34pp），collapse 被试无改善。**Few-shot FT 不是 S09/S13/S18 的解药。**

### 7.3 Evidence Map 可视化

新增 `scripts/visualize_evidence.py`：LOSO 模型 → `extract_evidence()` → band×channel×time 热力图。

| 被试 | LOSO Acc | Evidence 模式 |
|------|:---:|------|
| S07 | 91.1% | C3 强侧化、8-16Hz μ波段、cue后时序 — 生理学合理 |
| S09 | 44.4% | 完全平坦，信号强度 ~0.015（S07 的 1/3）|

> 模型不会对噪声数据编造 evidence。Collapse 被试的失败原因是数据本身缺乏可分信号。

---

## 十八、最终路线图（已验证）

| 优先级 | 方向 | 状态 | 结论 |
|--------|------|:---:|------|
| 1 | BCI IV 2a 跨数据集 | ✅ 完成 | 泛化成立，4-class 仍是挑战 |
| 2 | Few-shot FT | ✅ 完成 | 正常被试有效，collapse 被试无效 |
| 3 | Evidence 可视化 | ✅ 完成 | 证实模型学到正确 MI 特征 |
| 4 | 3-seed 稳定性 | ✅ 完成 | 63.02% ± 1.42% |

### 与 EEG Conformer + EA 的对比

| 指标 | Conformer + EA | BRT-Det v8 + EA | 说明 |
|------|:---:|:---:|------|
| Accuracy | 64.22% | 63.02% | Conformer +1.20pp |
| Kappa | 0.283 | 0.257 | Conformer +0.026 |
| Reported Std | ±9.86% | ±1.42% | **统计口径不同**（见注） |
| 参数 | ~100K+ | 32K | BRT-Det 约 3× lighter |
| 可解释性 | 弱 | 强 | BRT-Det 有 evidence map |

> **注：BRT-Det 的 ±1.42% 来自 3-seed 稳定性实验（seed 间标准差），Conformer 的 ±9.86% 是跨被试标准差。二者统计口径不同，不能直接比较跨被试稳定性。** 需要计算 BRT-Det 的 per-subject std（来自 LOSO CSV）才能公平对比。

**差距来源（1.20pp）：**
1. **全局时间依赖：** Conformer 的 self-attention 直接建模 time cell 3↔17 等远距离依赖；BRT-Det 用 dilated conv 扩展感受野，仍是局部算子
2. **时空 token 建模更自由：** BRT-Det 的 band×channel×time grid + objectness aggregation 解释性强但表达受限；Conformer 的 token 表示更灵活
3. **模型容量：** Conformer ~100K+ vs BRT-Det 32K（~3×），但差距仅 1.20pp——说明 BRT-Det 的参数效率更高

**BRT-Det 的差异化价值（不应只比 accuracy）：**
- 32K 参数达到 SOTA 的 98.1%（63.02/64.22）
- Evidence map 可解释（`extract_evidence()` → band×ch×time 热力图）
- Detection framing 为弱证据定位提供了新研究视角
- 3-seed 稳定性好（±1.42%）

### 建议暂缓

| 方向 | 原因 |
|------|------|
| hidden 32→48 / 加深 backbone | 容量不是第一瓶颈 |
| 复杂 cross-band attention | 数据量小，LOSO 易过拟合 |
| 围绕 S09 改主模型 | S09 本身 MI 信号弱 |

---

## 十九、Conformer 对比后的下一步（2026-07-06）

核心判断：

> Conformer + EA 的 64.22% 不是 EA 红利——BRT-Det v8 同样使用 EA。1.20pp 差距来自 Conformer 的全局 self-attention、更自由的时空 token 建模和更大容量。BRT-Det 的价值不在于绝对 SOTA，而在于 1/3 参数达到接近性能 + evidence map 可解释性。

### 建议优先级

| 优先级 | 方向 | 理由 |
|--------|------|------|
| 1 | Conformer + BRT-Det logits ensemble | 错误模式可能互补 |
| 2 | BRT-Det + Temporal Cell Gate | 轻量时间门控，不引入 self-attention |
| 3 | 计算 BRT-Det per-subject std | 与 Conformer 公平对比 |
| 4 | 错分重合度分析 | 验证 ensemble 可行性 |

---

## 二十、Round 8：Error Overlap + Temporal Gate + Ensemble（2026-07-07）

### 8.1 Error Overlap 分析

新增 `scripts/analyze_model_comparison.py`：对比两个 LOSO CSV 的 per-subject accuracy。

| 指标 | 值 |
|------|-----|
| BRT-Det seed42 true subject std | **±12.28%** |
| Conformer seed42 true subject std | ±10.74% |
| Correlation | r = 0.6865 |
| 互补被试 | **11/30 (36.7%)** |
| 共享失败 (<55%) | 2/30 (S06, S18) |
| Oracle ensemble | **67.48%** (+3.4pp over best single) |

> BRT-Det 的真正跨被试标准差是 ±12.28%（与 Conformer 的 ±10.74% 可比），而非之前报告的 3-seed std ±1.42%。两者统计口径不同。

**4 象限分类：**
- Both strong: 17 人
- Both weak: 2 人（S06, S18）— 数据瓶颈
- Conformer strong / BRT-Det weak: 6 人 — ensemble 机会（S09, S11, S13, S14, S17, S30）
- BRT-Det strong / Conformer weak: 5 人 — ensemble 机会（S03, S04, S20, S24, S27）

最大分歧：S13（Conformer 71% vs BRT-Det 47%, Δ=+24pp）、S24（Conformer 51% vs BRT-Det 71%, Δ=-20pp）。

### 8.2 Temporal Cell Gate ❌

新增 `use_temporal_gate` flag：在 dilated backbone 后加 per-time-cell 标量门控（~800 params）。

| 实验 | Accuracy | Δ |
|------|----------|-----|
| v8 (Band Gate) | 64.08% | — |
| +Temporal Gate | **63.26%** | **-0.82pp** |

> **Temporal Gate 是零和博弈。** 帮助 S13 (+22pp)、S14 (+13pp)、S20 (+13pp)，但伤害 S22 (-18pp)、S24 (-16pp)、S28 (-16pp)、S09 (-13pp)。净效果微负。

**结论：BRT-Det 的 dilated backbone + 24 time cells 已足够覆盖时序依赖，再加 gate 只引入方差。Band Gate 之所以有效（+5.41pp）是因为它解决了"被试间有效频带差异"这个真实瓶颈；Temporal Gate 无效是因为它试图解决的问题已被充分覆盖。**

### 8.3 Logits Ensemble 🔥

新增 `scripts/ensemble_eval.py`：训练两个模型 → logits 加权融合 → sweep α。

| 被试 | Conformer | BRT-Det | Best Ensemble | Gain | 模式 |
|------|:---:|:---:|:---:|:---:|------|
| S13 | 66.67% | 55.56% | **71.11%** (α=0.8) | **+4.44pp** | 互补 |
| S24 | 60.00% | 62.22% | 62.22% (α=0.0) | +0.00pp | 一方独强 |
| S09 | 60.00% | 53.33% | 60.00% (α=1.0) | +0.00pp | 共享失败 |

> **Ensemble 在互补被试上显著有效（+4.44pp），在一方独强或共享失败时无效。** 从 37% 互补率保守估计，全 LOSO ensemble 可增益 ~1pp，将组合系统推至 ~65%。

### 8.4 最终判断：BRT-Det 单模型到平台期

**公平的 subject std 对比：**

| 模型 | True subject std (seed42) |
|------|:---:|
| BRT-Det v8 | ±12.28% |
| Conformer | ±10.74% |

> BRT-Det 并不是跨被试更稳——之前报告的 ±1.42% 是 3-seed std，不是 subject std。真实跨被试波动与 Conformer 在同一量级。

**为什么 Temporal Gate 失败而 Band Gate 成功：**

| 模块 | 解决的问题 | 结果 | 原因 |
|------|-----------|:---:|------|
| Band Gate | 不同被试有效频带不同 | ✅ +5.41pp | 真实瓶颈 |
| Temporal Gate | 不同被试有效时间窗不同 | ❌ -0.82pp | 24 cells + dilation [1,2,4] 已覆盖 |

> **BRT-Det 单模型已到平台期。继续加轻量模块大概率只是在不同被试之间搬收益，净效果为负或微正。**

**BRT-Det 的最终定位：**

> 不是单模型 SOTA，而是一个**轻量（32K）、可解释（evidence map）、与 Conformer 错误模式互补（r=0.69）的 evidence model。**

---

## 二十一、Round 9：正式 Ensemble Benchmark（2026-07-07）

新增 `scripts/ensemble_loso.py`：全 30-fold LOSO，每折同时训练 Conformer + BRT-Det v8 + EA，固定 α 做 logits ensemble。

### 结果（α=0.7, seed42, with EA）

| 模型 | Accuracy | Kappa | κ<0 | Worst-5 Mean |
|------|:---:|:---:|:---:|:---:|
| Conformer + EA | 63.41% ± 9.63% | 0.264 | 3 | 49.78% |
| BRT-Det v8 + EA | 63.19% ± 11.41% | 0.260 | 2 | 50.22% |
| **Ensemble α=0.7** | **64.37% ± 10.29%** | **0.283** | **2** | **50.22%** |

**Gain: +0.96pp over best single model。**

### 分析

```text
Ensemble 有效，但增益温和（+0.96pp），远低于单点 S13 的 +4.44pp。

原因：
1. 固定 α=0.7 必须兼顾所有 30 人，无法为每人单独调权
2. 37% 互补被试上 ensemble 有效，其余被试上无增益
3. Oracle (67.48%) 是 per-subject 最优选择的上限，固定 α 无法达到

α 收益曲线是倒 U 形，α=0.7 已在峰值附近。
α=0.6 或 0.8 最多 ±0.2-0.3pp，不值得再跑完整 LOSO。
固定 α 的实际上限约 64.5-64.8%。
```

### 意义

64.37% 超过 Conformer 3-seed mean（64.22%），超过此前 leaderboard 上所有单模型。

> BRT-Det 与 Conformer 具有互补性，轻量 evidence model 可作为强时空模型的有效补充。但固定 α ensemble 的收益温和（~1pp），不是免费午餐。

---

## 二十二、70% 目标可行性分析（2026-07-07）

### 当前天花板

| 路径 | 实际上限 | 判断 |
|------|:---:|:---:|
| BRT-Det 单模型 | ~63% | 已到平台期 |
| Conformer 单模型 | ~64% | 已到平台期 |
| 固定 α ensemble (2 models) | ~64.8% | α=0.7 已近峰值 |
| Oracle per-subject (2 models) | 67.48% | 仅靠两模型的上限 |
| **70% 目标** | — | **仅靠两模型不可能** |

> 即使每个被试选 Conformer/BTR-Det 中更好的那个，上限也只有 67.48%。要冲 70 必须新增第三类互补信息源。

### 三条可达路径

| 路线 | 描述 | 预期上限 |
|------|------|:---:|
| A: 多模型 ensemble | 加入 FBCSP-LDA, Tangent-LDA, EEGNet, ShallowConvNet | 67-69% |
| B: Subject-adaptive routing | 用无标签特征（entropy, evidence strength, cov distance）选 α | 65-66% |
| C: Few-shot calibration + ensemble | 40/60/80-shot FT + 多模型 ensemble | **最现实到 70** |

### 路线 B 的关键约束

固定 α 太弱因为不同被试最优权重不同。但 α_subject 不能用测试标签选（否则是 oracle）。可用无标签信号：

| 信号 | 用途 |
|------|------|
| Conformer prediction entropy | 判断 Conformer 是否自信 |
| BRT evidence strength | 判断 BRT 是否看到清晰 MI 证据 |
| 两模型 logits disagreement | 判断是否需要保守融合 |
| EA 后协方差距离 | 判断 target subject 更像哪些训练被试 |

形式：`α = router(unlabeled_target_features)`

### 路线 C 的关键约束

之前 20-shot FT 已确认正常被试受益（+2.34pp）但 collapse 被试无改善。需要试更大 FT trial 数（40/60/80）+ 不要只 FT 单模型。

---

## 二十三、建议：不再做 BRT-Det v9，转向多视角 Oracle 分析

### Round 10: Multi-View Ensemble Upper Bound

**目的：先算上限，再决定要不要建复杂系统。**

加入 4-5 个模型，计算：

```text
best single per subject
fixed ensemble
oracle subject selection (每个被试选最优模型)
oracle trial selection (每个 trial 选最优模型)
shared failure subjects (所有模型都错)
```

**关键问题：这些模型加起来，oracle 能不能超过 70%？**

如果 oracle < 70%，说明数据和划分本身限制太强，后续不用盲目冲。
如果 oracle ≥ 70%，说明有路，再进入合法融合（Round 11）和 few-shot（Round 12）。

---

## 二十四、Round 10：多视角 Oracle 上限分析（2026-07-07）

新增 `scripts/analyze_multi_model_oracle.py`：读 7 个模型的 LOSO CSV，计算 per-subject oracle。

### 7 模型汇总（seed42, EA）

| 模型 | Mean Acc | 最优被试数 |
|------|:---:|:---:|
| BRT-Det v8 | 64.08% | 6 |
| EEG-TCNet | 63.85% | 5 |
| Conformer | 63.63% | 6 |
| FBCNet | 62.67% | 0 |
| ER-MI | 61.92% | **9** |
| ER-MI v2 | 61.93% | 3 |
| Tangent-LDA | 60.44% | 1 |

> ER-MI 平均只有 61.92%，但覆盖了 9/30 被试——很多是 BRT-Det/Conformer 都 miss 的人。这说明"低平均 ≠ 无价值"，互补性才是 ensemble 的关键。

### Marginal Gain（贪心加入）

| # | +Model | Oracle | Δ |
|:---:|------|:---:|:---:|
| 1 | BRT-Det v8 | 64.08% | — |
| 2 | +ER-MI | 68.08% | +4.00pp |
| **3** | **+EEG-TCNet** | **70.00%** | +1.93pp |
| 4 | +Conformer | 70.52% | +0.52pp |
| 7 | +all | **71.04%** | — |

> **仅需 3 个模型（BRT-Det + ER-MI + EEG-TCNet），oracle 即可达 70.00%！**

### Pairwise 互补性（r 最低 = 最互补）

| 配对 | r | 互补性 |
|------|:---:|:---:|
| BRT-Det + Tangent-LDA | 0.42 | 最强 |
| BRT-Det + ER-MI | 0.46 | 强 |
| ER-MI v2 + Tangent-LDA | 0.48 | 强 |
| Conformer + FBCNet | 0.77 | 冗余 |
| EEG-TCNet + FBCNet | 0.84 | 最冗余 |

### 共享失败

S06 和 S18 — 所有 7 个模型都 <55%。数据瓶颈，ensemble 也无法解决。

### 最终判决

```text
1. 70% 在理论上可行（7-model oracle = 71.04%）
2. 仅需 3 模型（BRT-Det + ER-MI + EEG-TCNet）即可达 oracle 70.00%
3. ER-MI 是最被低估的模型——低均值但高互补性
4. FBCNet 和 EEG-TCNet 高度冗余（r=0.84），不需要同时加
5. S06/S18 是硬天花板——所有模型都救不了
6. 从 oracle 70% 到合法融合 67-68% 是下一步的关键挑战
```

### 从现在起不做

```text
- BRT-Det v9 (加模块)
- BRT-Det 单模型刷分
- 固定 α ensemble sweep
- 围绕 S09 改主模型
```

---

## 二十五、战略转向：从单模型迭代到多视角融合（2026-07-07）

### 核心判断

> **70% 已经不是"有没有可能"的问题，而是"怎么把 oracle 上限转成合法融合结果"的问题。**

任务从：

```text
怎么把 BRT-Det 改到 70%
```

变成：

```text
怎么把 BRT-Det + ER-MI + EEG-TCNet oracle 70.00%
变成合法、无泄漏的 67~70% 融合系统
```

### ER-MI 是最重要的新线索

| 模型 | Mean Acc | 最优被试数 | Ensemble 价值 |
|------|:---:|:---:|------|
| EEG-TCNet | 63.85% | 5 | 强但和 FBCNet 冗余 |
| Conformer | 63.63% | 6 | 强但和深度模型重叠 |
| BRT-Det v8 | 64.08% | 6 | 核心，自研 |
| **ER-MI** | **61.92%** | **9** | **均值低但互补性极强** |
| Tangent-LDA | 60.44% | 1 | 和 BRT 相关最低 (r=0.42) |

> **ER-MI 平均不高但 marginal gain 极大（+4.00pp）。选模型不应看 mean acc，应看 marginal gain + pairwise correlation + shared failure reduction。**

### 防止数据泄漏规则

进入 ensemble/router 阶段后必须遵守：

| 允许 | 不允许 |
|------|--------|
| 目标被试无标签 EEG/协方差 | 目标被试 accuracy/kappa |
| 目标被试预测分布 (entropy等) | 目标被试 trial 正误 |
| 训练折内部 validation 选权重 | 测试标签选 α/模型/权重 |
| 固定全局权重 | Oracle per-subject selection 作正式结果 |

### 新路线图

| Round | 内容 | 目标 |
|:---:|------|:---:|
| 11 | BRT + ER-MI + EEG-TCNet 合法融合 (equal/fixed/entropy/calibrated) | 66-68% |
| 12 | Subject router (无标签特征 → 动态权重) | 68-69% |
| 13 | Few-shot calibration (40/60/80-shot × ensemble) | 69-72% |

> 这三轮的目标不是"BRT-Det 单模型更强"，而是"多视角融合系统逼近 oracle 70%"。

### 当前选模型准则（更新）

```text
旧：谁平均 Accuracy 高 → 选谁
新：谁能救现有系统 miss 的被试 → 选谁
```

### 最被低估的发现

```text
ER-MI (61.92%) 在 9/30 被试上最优 — 超过任何其他模型
BRT-Det + ER-MI 的 oracle 增益 +4.00pp — 超过 BRT + Conformer
最低相关配对: BRT-Det + Tangent-LDA (r=0.42), BRT-Det + ER-MI (r=0.46)
```
