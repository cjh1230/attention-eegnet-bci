# SPD 流形黎曼深度学习 + 自监督预训练：研究计划

> **主线方案**：SPD 流形上的深度表征学习 + 自监督预训练  
> **优先级**：🥇 主攻（先做）  
> **预计周期**：4–6 周  
> **输入**：8ch 运动皮层 EEG（PhysioNet MI + BCI IV 2a）  
> **评估**：LOSO 跨被试

---

## 一、为什么选这个方向

### 1.1 实验数据给出的信号

| 方法 | 特征空间 | 准确率 (PhysioNet binary LOSO) |
|------|----------|-------------------------------|
| EEGNet (no EA) | 原始 EEG | 51.93% |
| EEG Conformer + EA | 原始 EEG | **63.93%** |
| Tangent Space + LDA + EA | **协方差** | **60.44%** |

协方差 + 线性分类（60.44%）距离原始 EEG + 深度非线性（63.93%）仅差 3.49pp。这表明**协方差结构保留了 EEG 空间模式中的大量判别信息**——深度网络在原始 EEG 上花费可观容量隐式逼近的，很大程度上正是这种通道间相关结构。

### 1.2 竞争格局：差异化空间明确

| 维度 | 已有工作 | 本文 |
|------|----------|------|
| 通道数 | 22ch（BNCI/MOABB） | **8ch 运动皮层蒙太奇** |
| 数据集 | BCI IV 2a, BNCI 系列 | **PhysioNet MI (30 subjects)** |
| 评估协议 | 10-fold CV（被试内） | **LOSO（跨被试）** |
| 训练范式 | 全监督 | **SSL 预训练 + 微调** |
| SPD 流形 + SSL 预训练 | **现有研究较少关注** | **本文探索方向** |

**差异化来自问题设置和训练范式的组合，而非单一组件的新颖性。** 已有黎曼 DL 工作集中在 22ch、被试内评估、全监督训练；本文面向 8ch 低通道场景、跨被试 LOSO 评估，并探索 SSL 预训练在 SPD 流形上的可行性，这四个维度的叠加构成了与现有工作的实质差异。

### 1.3 风险结构

```
阶段一（必做，1-2 周）: SPDNet 8ch PhysioNet LOSO 基线
  → 建立 8ch PhysioNet MI 场景下的黎曼 DL LOSO 基线
  → 即使阶段二效果不如预期，该基线本身可作为阶段性成果

阶段二（核心探索，2-3 周）: SPD 流形上的 SSL 预训练
  → 探索 SPD 流形上自监督预训练的可行性与效果
  → 若效果不显著: 阶段一基线 + 消融分析 + 8ch 差异化分析 → 撰写实验报告

阶段三（扩展，1 周）: 跨数据集验证 + 可视化
```

**这一结构确保即使 SSL 探索未达预期，仍可基于 SPDNet 基线 + 8ch 差异化形成完整研究叙事。**

---

## 二、核心思路

### 2.1 整体框架

```
8ch 原始 EEG
     ↓
预处理（带通滤波 8–30 Hz + 陷波 50 Hz + 试次截取）
     ↓
EA 对齐（跨被试协方差对齐 = SPD 流形上的平行移动）
     ↓
频段划分（mu 8–13 Hz / beta 13–30 Hz / full 8–30 Hz）
     ↓
协方差矩阵计算（8×8 SPD 矩阵）
     ↓
┌──────────────────────────────────────────┐
│  🆕 SPD 流形上的 SSL 预训练（核心创新）      │
│  - SPD 矩阵增强策略（流形感知的数据增强）     │
│  - 对比学习（同一试次的不同增强 → 正样本对）  │
│  - 掩码重建（遮蔽通道/频段 → 重建协方差）     │
│  → 预训练编码器                             │
└──────────────────────────────────────────┘
     ↓
SPDNet 编码器（BiMap → ReEig → LogEig）
     ↓
分类器 → LOSO 跨被试评估
```

### 2.2 SPD 流形 SSL 的核心挑战

SPD 流形上的 SSL 与欧氏空间 SSL 的关键区别：

| | 欧氏空间 SSL | SPD 流形 SSL |
|------|-------------|-------------|
| 数据形式 | 向量 | 8×8 正定矩阵 |
| 几何空间 | 欧氏空间 | SPD 流形（弯曲空间） |
| 数据增强 | 裁剪、翻转、颜色抖动 | 协方差扰动、通道mask、频段mask |
| 距离度量 | 欧氏距离 | 仿射不变黎曼距离 / Log-Euclidean 距离 |
| 对比损失 | InfoNCE（欧氏点积相似度） | 需适配到流形上的相似度度量 |

**SPD 流形上的 SSL 预训练是本研究计划探索的关键方向**：已有黎曼 DL 工作均为全监督训练，而对比学习与掩码重建在 SPD 流形上的适配尚处于探索不足的状态。

### 2.3 SPD 流形上的数据增强策略

为 SPD 协方差矩阵设计以下增强（是关键贡献之一）：

| 增强方式 | 操作 | 物理含义 |
|----------|------|----------|
| **通道 Dropout** | 随机 mask 1–2 个通道 → 重算 6×6 或 7×7 协方差 | 模拟电极接触不良 |
| **频段 Mask** | 在 mu/beta 内随机遮蔽部分频率分量 | 模拟频段信息缺失 |
| **协方差扰动** | 对 SPD 矩阵施加小幅度随机扰动（保持正定性） | 模拟噪声波动 |
| **时间裁剪** | 只取试次的一部分窗口计算协方差 | 模拟时间不确定性 |
| **被试 Mixup** | 两个被试的协方差矩阵在 SPD 流形上插值 | 模拟被试间过渡状态 |

```python
# 核心操作示例
def spd_augment(C: Tensor[8, 8], strategy: str) -> Tensor[8, 8]:
    """在 SPD 流形上做数据增强，保持正定性"""
    if strategy == "channel_dropout":
        kept = random.sample(range(8), k=6)  # 随机保留 6 个通道
        C_aug = C[kept][:, kept]
    elif strategy == "cov_perturb":
        eps = torch.randn(8, 8) * 0.01
        eps = (eps + eps.T) / 2  # 对称化
        C_aug = C + eps
        # 确保正定性：对负特征值做 clamp
        eigvals, eigvecs = torch.linalg.eigh(C_aug)
        eigvals = torch.clamp(eigvals, min=1e-6)
        C_aug = eigvecs @ torch.diag(eigvals) @ eigvecs.T
    elif strategy == "band_mask":
        # 在频域随机遮蔽部分频率 bin
        ...
    return C_aug
```

### 2.4 SPD 流形上的对比学习

**SimCLR 风格，但适配 SPD 流形**：

```
同一 trial → 两种增强 (C_i, C_j) → SPDNet → (z_i, z_j)
                                              ↓
                            sim(z_i, z_j) = -δ_R²(z_i, z_j) / τ
                                              ↓
                            InfoNCE loss: -log exp(sim(z_i,z_j)) / Σ_k exp(sim(z_i,z_k))
```

关键差异：
- 相似度函数不直接用余弦相似度，改用**负黎曼距离**或**Log-Euclidean 内积**
- 投影头也需适配 SPD 结构（小 SPDNet → LogEig → 欧氏投影）

**掩码重建（MAE 风格），适配 SPD 矩阵**：

```
完整协方差 C → 随机 mask 某些通道/频段 → 部分协方差 C_masked
                                            ↓
                              SPDNet 编码器 → LogEig → 解码器 → 重建完整 C
                                            ↓
                              MSE on Log-Euclidean space: ||log(C) - log(C_recon)||²
```

### 2.5 模型架构（具体）

```
┌──────────────────────────────────────────────┐
│ 输入: 8×8 SPD 协方差矩阵                      │
├──────────────────────────────────────────────┤
│ BiMap(8, 6)   → 6×6  (流形上的线性变换)       │
│ ReEig         → 6×6  (流形上的非线性激活)     │
│ BiMap(6, 4)   → 4×4                          │
│ ReEig         → 4×4                          │
│ LogEig        → 4×4 → flatten → 10d          │
├──────────────────────────────────────────────┤
│ 线性分类器 → 2 类 (PhysioNet) / 4 类 (BCI IV 2a) │
└──────────────────────────────────────────────┘
```

总参数量 ~2K，极其轻量。

**运动皮层拓扑先验（可选增强）**：

8 个通道的解剖分组：

```
左半球:  FC3, C3, CP3  (通道 0,1,2)
中线:    Cz, CPz       (通道 3,4)
右半球:  FC4, C4, CP4  (通道 5,6,7)
```

可以在第一个 BiMap 层之前加入分组卷积或掩码：

```python
# 通道分组：BiMap 的权重矩阵可以初始化为块对角结构
# 让模型先学习半球内的协方差模式，再学习半球间的交互
mask = torch.block_diag(
    torch.ones(3,3),  # left
    torch.ones(2,2),  # central
    torch.ones(3,3),  # right
)
```

---

## 三、实验设计

### 3.1 评估协议

- **主评估**：LOSO（Leave-One-Subject-Out）30 被试 × PhysioNet MI binary
- **跨数据集**：BCI IV 2a (9 subjects, 4-class LOSO)
- **统计检验**：配对 t 检验 / Wilcoxon 符号秩检验

### 3.2 实验矩阵

| 实验 | 目的 | 对比项 | 预计时间 |
|------|------|--------|----------|
| **E1: 基线建立** | 验证 SPDNet 在 8ch PhysioNet LOSO 上的基础性能 | Tangent Space+LDA, EEGNet, EEG Conformer, SPDNet(no EA) | Week 1 |
| **E2: EA 消融** | 验证 EA 对齐对 SPD 方法的贡献 | SPDNet ± EA | Week 1 |
| **E3: 频段消融** | 验证多频段协方差的贡献 | mu only / beta only / full band / multi-band fusion | Week 2 |
| **E4: 增强策略消融** | 找到最优 SPD 增强组合 | 5 种增强策略的独立 + 组合效果 | Week 2-3 |
| **E5: SSL 主实验** | 验证 SSL 预训练的整体效果 | 全监督 vs 对比学习预训练 vs 掩码重建预训练 | Week 3-4 |
| **E6: 少样本实验** | 验证 SSL 在少样本场景的优势 | 5/10/20/50-shot LOSO，预训练 vs 无预训练 | Week 4 |
| **E7: 跨数据集验证** | 验证方法在不同数据集上的泛化 | BCI IV 2a 4-class LOSO | Week 5 |
| **E8: 可视化 + 可解释性** | 理解模型学到的表征 | 协方差响应图、通道贡献、t-SNE、频段重要性 | Week 5-6 |

### 3.3 预期结果

| 目标层次 | 描述 | 参考值 |
|----------|------|--------|
| **保底** | SPDNet 在 8ch LOSO 上稳定运行 | 建立基准 |
| **合格** | SPDNet + EA 超过 EEG Conformer + EA | >63.93% |
| **良好** | SPDNet + EA + SSL 预训练 | 66–70% |
| **优秀** | SSL 预训练在小样本场景（≤20-shot）显著优于全监督 | ≥5pp 提升 |

---

## 四、实施计划

### Week 1：SPDNet 基线（必做）

```
Day 1-2: 搭建 8ch → 协方差 → SPDNet pipeline
  - 使用 spd-learn 库的 SPDNet 实现
  - 适配 8ch 输入（现有 SPDNet 默认适配 22ch）
  - 编写 LOSO 评估循环
  - 目标：端到端跑通，确认无 bug

Day 3-4: 基线实验
  - SPDNet ± EA on PhysioNet binary LOSO
  - 复现 Tangent Space + LDA + EA (60.44%)
  - 对比 EEGNet, EEG Conformer (用已有结果)
  - 目标：SPDNet > 60.44%

Day 5: 结果分析 + 调试
  - 如果 SPDNet < Tangent Space → 分析原因，调整架构
  - 记录所有 baselines
```

**Week 1 交付**：SPDNet 8ch PhysioNet LOSO 基线结果表

### Week 2：多频段 + 拓扑先验（必做）

```
Day 1-2: 多频段协方差
  - mu (8-13Hz), beta (13-30Hz), full (8-30Hz) 分别计算协方差
  - 三路 BiMap → 融合 → LogEig → 分类
  - 消融：单频段 vs 多频段

Day 3-4: 运动皮层拓扑先验
  - 通道分组 BiMap（左/中/右）
  - 对比：分组 vs 不分组的性能差异

Day 5: EA 消融 + 频段消融完整结果
```

**Week 2 交付**：频段消融 + 拓扑先验消融结果

### Week 3：SPD 流形 SSL 预训练（主创新）

```
Day 1-2: SPD 增强实现
  - 实现 5 种 SPD 流形感知的数据增强
  - 验证增强后的矩阵保持正定性
  - 单元测试

Day 3-4: 对比学习 pre-training
  - SimCLR 风格对比学习框架
  - SPD 流形上的投影头
  - 用所有训练被试的 trial 做预训练（无标签）
  - 预训练 → 微调 → LOSO 评估

Day 5: 掩码重建 pre-training
  - MAE 风格的协方差掩码重建
  - 对比 vs 重建的预训练效果
```

**Week 3 交付**：SPD 流形 SSL 预训练框架 + 初步实验结果

### Week 4：SSL 消融 + 少样本实验

```
Day 1-2: 增强策略消融
  - 5 种增强的独立效果
  - 最佳组合确定

Day 3-4: 少样本 LOSO
  - 5/10/20/50-shot LOSO
  - 全监督 vs SSL 预训练 vs 随机初始化
  - 这是论文的核心亮点图

Day 5: 初步结果汇总
```

**Week 4 交付**：SSL 消融 + 少样本实验结果

### Week 5：跨数据集 + 扩展实验

```
Day 1-3: BCI IV 2a 跨数据集验证
  - 9 subjects, 4-class LOSO
  - SPDNet ± SSL 预训练
  - 对比 EEGNet, Conformer

Day 4-5: 补充实验
  - 更多被试数量的 scaling 分析
  - 不同时间窗口的敏感性
```

### Week 6：可视化 + 论文

```
Day 1-2: 可视化
  - 协方差矩阵响应图（频段 × 通道）
  - t-SNE 对比（预训练前后）
  - 通道贡献热力图
  - 少样本学习曲线

Day 3-5: 论文初稿
  - 引言 + 相关工作 + 方法 + 实验 + 讨论
```

---

## 五、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|:---:|------|------|
| SPDNet 不如 Tangent Space + LDA | 低 | 高 | 检查 8×8 协方差估计质量；调大 BiMap 维度；增加正则化；若确实不如，分析 8×8 低维 SPD 流形的信息瓶颈，作为方法层面的发现 |
| 8×8 协方差信息量不足 | 中 | 高 | 引入多频段协方差（mu+beta+full 三个 8×8）增加信息量；考虑 phase 信息（Phase-SPDNet 的做法） |
| SSL 预训练不收敛 | 中 | 中 | 先用较简单的增强策略；降低对比学习 batch size 要求（累积梯度）；如果对比学习不行，先做掩码重建（更稳定） |
| PhysioNet 30 被试 × ~100 trials 样本不足以做 SSL | 中 | 中 | 引入 BCI IV 2a 数据扩充预训练集；或使用 trial 内的子段做增强（增加有效样本数） |
| SSL 带来的提升太小（<2pp） | 中 | 中 | SPDNet 基线 + 消融 + 8ch 差异化分析仍可形成完整实验报告；少样本场景的提升是更关键的故事线 |

---

## 六、关键基础设施

| 组件 | 库/工具 | 备注 |
|------|---------|------|
| SPDNet 实现 | `spd-learn` (pip install spd-learn) | BSD-3 许可，内置 SPDNet/TensorCSPNet/PhaseSPDNet/TSMNet |
| 协方差计算 | `pyRiemann` | 已有依赖，已验证 |
| EA 对齐 | 自实现 (`preprocessing/alignment.py`) | 已有，已验证 |
| 对比学习框架 | `lightly` 或自实现 | SimCLR/BYOL 等 |
| LOSO 评估 | 自实现 (`training/train_loso.py`) | 已有，已验证 |
| 预处理 | 自实现 (`preprocessing/`) | 已有，已验证 |

---

## 七、论文故事线

```
标题候选：
- "Self-Supervised Pre-Training on SPD Manifolds for 
   Low-Channel Motor Imagery EEG Decoding"
- "Riemannian Deep Learning with Self-Supervised Pre-training 
   for Cross-Subject 8-Channel MI-EEG Decoding"

故事弧线：
  1. 背景：跨被试 MI-EEG 解码的核心挑战——分布偏移 + 标注数据稀缺
           低通道（8ch）场景进一步加剧了信息不足的问题
  2. 动机：协方差矩阵是 EEG 空间结构的紧凑表示，
            现有黎曼 DL 方法（SPDNet, Tensor-CSPNet 等）已验证其有效性，
            但这些方法均采用全监督训练，
            SPD 流形上的自监督预训练尚未得到充分探索
  3. 方法：面向 8ch 运动皮层 MI-EEG，构建 SPD 流形上的 SSL 预训练框架
            ——SPD 流形感知的数据增强策略
            ——基于黎曼距离度量的对比学习预训练
            ——掩码协方差重建预训练
  4. 实验：PhysioNet MI (30 subjects, binary LOSO) 主实验
             + BCI IV 2a (9 subjects, 4-class) 跨数据集验证
             + 少样本校准场景 (5/10/20-shot)
  5. 发现：SPD 流形 SSL 预训练能否提升低通道跨被试泛化？
             少样本场景是否比全监督有更明显优势？

三个贡献点：
  C1: 8ch 运动皮层 MI-EEG 的 SPD 流形深度学习方法在 LOSO 跨被试评估下的实验分析
  C2: SPD 流形上的自监督预训练策略（流形感知增强 + 对比学习 + 掩码重建）
  C3: 低通道少样本跨被试适配场景下的性能分析与讨论
```

> **注意**：C1 不声称“首个”，C2 不声称“完全空白”，C3 不声称“显著提升”。
> 研究的价值在于**在 8ch 低通道 LOSO 跨被试这一特定场景下，系统探索 SPD 流形 + SSL 预训练的可行性与效果**，而非声称发明了全新的方法类别。

---

## 八、参考文献

1. Huang & Van Gool (2017). "A Riemannian Network for SPD Matrix Learning." AAAI 2017.
2. Ju & Guan (2023). "Tensor-CSPNet: A Novel Geometric Deep Learning Framework for Motor Imagery Classification." IEEE TNNLS.
3. Aristimunha et al. (2026). "SPD Learn: A Geometric Deep Learning Python Library." arXiv:2602.22895.
4. Wilson et al. (2025). "Deep Riemannian Networks for End-to-End EEG Decoding." Imaging Neuroscience.
5. Collas et al. (2024). "Geometric Neural Network based on Phase Space for BCI-EEG decoding." arXiv:2403.05645.
6. He & Wu (2020). "Transfer Learning for BCIs: A Euclidean Space Data Alignment Approach." IEEE TBME.
7. Chen et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020. (SimCLR)
8. He et al. (2022). "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. (MAE)
9. Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs." J. Neural Eng.
10. Song et al. (2023). "EEG Conformer: Convolutional Transformer for EEG Decoding." arXiv:2301.05578.
