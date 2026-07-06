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

---

## 十七、下一步路线图

### 第一优先级：跨数据集验证
所有结论来自 PhysioNet MI，需要 BCI IV 2a 验证 Temporal Stem / Dilation / Band Gate 不是数据集特化。

### 第二优先级：Few-shot FT
验证：给少量目标被试数据后，collapse subject（S09, S18, S13）能否恢复到可用水平？如果 5-shot/10-shot 能把 κ<0 被试拉到 60%+，实用价值大幅增强。

### 第三优先级：Evidence Map 可视化
证明模型不是黑盒乱学。输出 band × channel × time_cell 的 objectness × class_score heatmap，验证 C3/C4 和 μ/β 段有明显响应。

### 建议暂缓

| 方向 | 原因 |
|------|------|
| hidden 32→48 / 加深 backbone | 容量不是第一瓶颈，泛化才是 |
| 复杂 cross-band attention | 数据量小，LOSO 容易过拟合 |
| 围绕 S09 改主模型 | S09 单被试都只有 55%，不适合主导架构设计 |
