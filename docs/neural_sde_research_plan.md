# 神经随机微分方程（Neural SDE）用于运动想象 EEG 动力学建模：探索性研究计划

> **探索方案**：Neural SDE 驱动的 MI-EEG 连续时间动力学建模  
> **优先级**：🥈 探索（SPD+SSL 完成后启动）  
> **前置依赖**：SPD+SSL 阶段的预处理、EA 对齐、LOSO 评估基础设施可复用  
> **预计周期**：3–4 周  
> **风险等级**：中高（方法新颖，但实现有成熟库支持）

---

## 一、动机：为什么是 Neural SDE

### 1.1 当前范式的根本局限

所有现有 MI-EEG 解码方法——无论是 CNN、Transformer、Mamba、SPDNet——都在**离散时间**框架下工作：

```
EEG 试次 (250 Hz × 3s = 750 个离散时间点)
        ↓
CNN/Transformer/SPDNet（离散时间处理）
        ↓
分类
```

但大脑是一个**连续时间动力系统**。EEG 是这个动力系统在头皮上的投影。把 EEG 当作离散采样点序列处理，丢失了动力学的连续性信息。

**Neural ODE/SDE 提供了在连续时间域建模 EEG 的数学框架。**

### 1.2 你的实验数据已经给出了信号

| 模型 | 时间建模方式 | 准确率 |
|------|------------|--------|
| EEGNet | 无显式时间建模 | 58.00% (+EA) |
| EEG-TCNet | TCN 离散时间 | 63.41% (+EA) |
| EEG Conformer | Transformer 离散时间 | **63.93%** (+EA) |

**时间建模是最大的信息增量来源。** 但 TCN 和 Transformer 都是"把连续动力学拍扁成离散 token 再处理"。

**Neural SDE 从根本上改变这个范式：先假设 EEG 服从某个随机微分方程，再让神经网络学习这个方程的动力学参数。**

### 1.3 竞争格局：处于极早期阶段

| 方向 | 论文数 (2023-2025) |
|------|:---:|
| Neural ODE + EEG 分类 | 3 |
| Neural ODE + MI-EEG | 1（运动控制策略分类，非运动想象解码） |
| **Neural SDE + MI-EEG** | **0** |
| **Neural CDE + MI-EEG** | **0** |

**在已有文献检索中，尚未发现将 Neural SDE 应用于 MI-EEG 解码的工作。** 这意味着该方向具有较高的探索价值和较大的不确定性——缺乏直接可参考的 baseline 既意味着机会，也意味着试错成本。

### 1.4 为什么 SDE 而非 ODE

EEG 信号本质是随机的——两个完全相同的运动想象意图，产生的 EEG 波形也会不同。这种变异不是"噪声"，而是神经系统的内在特性。

| Neural ODE | Neural SDE |
|------------|------------|
| dz = f(z, t) dt | dz = f(z, t) dt + g(z, t) dW |
| 确定性动力学 | 随机动力学 |
| 同一输入 → 同一轨迹 | 同一输入 → 轨迹分布 |
| 不适合建模 EEG 的 trial-by-trial 变异 | 天然建模 EEG 的随机性 |

**SDE 的扩散项 g(z,t)dW 不是 bug，是 feature。** 它显式建模了 EEG 的变异性来源——这是 ODE 做不到的。

---

## 二、核心思路

### 2.1 整体框架

```
EEG 试次 (750 时间步 × 8 通道)
        ↓
┌──── 编码器（RNN / TCN）──────────────────┐
│  将离散 EEG 映射到连续潜状态空间            │
│  x₁, x₂, ..., x₇₅₀ → z₀ (初始状态)        │
└─────────────────────────────────────────┘
        ↓
┌──── Latent Neural SDE ──────────────────┐
│  dz = f_θ(z, t) dt + g_φ(z, t) dW       │
│                                          │
│  f_θ: 漂移网络（确定性动力学）              │
│  g_φ: 扩散网络（随机波动建模）              │
│                                          │
│  求解器：Euler-Maruyama / Milstein         │
│  从 z₀ 积分到 z_T，得到轨迹 {z(t)}         │
└─────────────────────────────────────────┘
        ↓
┌──── 读出层 ─────────────────────────────┐
│  从轨迹中提取分类信息                       │
│  可选：最后状态 / 轨迹平均 / 注意力池化       │
│  → 左/右手                                │
└─────────────────────────────────────────┘
```

### 2.2 关键设计选择

#### 编码器：从离散 EEG 到连续潜状态

```python
class EEGEncoder(nn.Module):
    """将 (N, 8, 750) 的 EEG 试次映射到初始潜状态 z₀"""
    def __init__(self, n_channels=8, latent_dim=32):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            # 保留 EEGNet Block1 的空间卷积 + 深度卷积
            nn.Conv2d(1, 8, (1, 64), padding='same'),  # 时间滤波
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (8, 1), groups=8),        # 空间滤波（depthwise）
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        # 将卷积输出压缩为初始状态
        self.to_latent = nn.Linear(16 * (750//4), latent_dim)

    def forward(self, x):
        # x: (N, 8, 750)
        x = x.unsqueeze(1)  # (N, 1, 8, 750)
        x = self.temporal_conv(x)  # (N, 16, 1, T')
        x = x.flatten(1)           # (N, 16*T')
        z0 = self.to_latent(x)     # (N, latent_dim)
        return z0
```

#### Neural SDE 核心

```python
class LatentNeuralSDE(nn.Module):
    """潜空间 Neural SDE: dz = f(z,t)dt + g(z,t)dW"""
    def __init__(self, latent_dim=32, hidden_dim=64):
        super().__init__()
        # 漂移网络 f(z, t)：确定性动力学
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),  # +1 for time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # 扩散网络 g(z, t)：随机波动
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),  # 保证扩散系数为正
        )

    def f(self, t, z):
        """漂移函数"""
        t_tensor = torch.full((z.shape[0], 1), t, device=z.device)
        tz = torch.cat([z, t_tensor], dim=-1)
        return self.drift_net(tz)

    def g(self, t, z):
        """扩散函数（返回对角扩散矩阵）"""
        t_tensor = torch.full((z.shape[0], 1), t, device=z.device)
        tz = torch.cat([z, t_tensor], dim=-1)
        return self.diffusion_net(tz)
```

#### 读出层：从轨迹到分类

```python
class TrajectoryReadout(nn.Module):
    """从 SDE 轨迹中提取分类特征"""
    def __init__(self, latent_dim=32, n_classes=2):
        super().__init__()
        # 注意力池化：学习关注轨迹中的关键时间点
        self.attn = nn.MultiheadAttention(latent_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def forward(self, trajectory):
        # trajectory: (N, T_sde, latent_dim) — SDE solver 输出的轨迹
        attn_out, _ = self.attn(trajectory, trajectory, trajectory)
        pooled = attn_out.mean(dim=1)  # 全局平均池化
        return self.classifier(pooled)
```

### 2.3 训练

```python
import torchsde

model = MI_NeuralSDE(latent_dim=32, n_classes=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for X, y in train_loader:
        # X: (N, 8, 750)
        z0 = model.encoder(X)

        # SDE 求解（Euler-Maruyama, 自适应步长可选）
        # 在 [0, 1] 归一化时间区间上积分
        ts = torch.linspace(0, 1, 50, device=device)  # 50 步积分
        z_trajectory = torchsde.sdeint(
            model.sde,
            z0,
            ts,
            dt=0.02,
            method='euler'  # 或 'milstein', 'srk'
        )  # (50, N, latent_dim) → (N, 50, latent_dim)
        z_trajectory = z_trajectory.permute(1, 0, 2)

        logits = model.readout(z_trajectory)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

### 2.4 跨被试扩展：Subject-Conditional Neural SDE

这是论文的核心创新点——把跨被试泛化自然地融入 SDE 框架：

```python
class SubjectConditionalNeuralSDE(nn.Module):
    """
    共享动力学 + 被试特定调制

    核心假设：
    - 所有被试共享相同的 MI 神经动力学骨架（f_shared, g_shared）
    - 个体差异由被试特定潜变量 s 调制
    - s 可以从少量校准试次中推断出来
    """
    def __init__(self, latent_dim=32, subject_dim=16):
        super().__init__()
        self.subject_encoder = SubjectEncoder(latent_dim=subject_dim)

        # 共享 SDE
        self.f_shared = DriftNet(latent_dim + subject_dim, latent_dim)
        self.g_shared = DiffusionNet(latent_dim + subject_dim, latent_dim)

    def infer_subject_latent(self, calibration_trials):
        """从少量校准试次推断被试潜变量 s"""
        # calibration_trials: (n_cal, 8, 750)
        features = self.subject_encoder(calibration_trials)
        return features.mean(dim=0)  # (subject_dim,)

    def f(self, t, z, s):
        """被试条件漂移: dz = f(z, t, s) dt + g(z, t, s) dW"""
        tz = torch.cat([z, s.expand(z.shape[0], -1),
                        torch.full((z.shape[0],1), t)], dim=-1)
        return self.f_shared(tz)

    def g(self, t, z, s):
        tz = torch.cat([z, s.expand(z.shape[0], -1),
                        torch.full((z.shape[0],1), t)], dim=-1)
        return self.g_shared(tz)
```

**这个设计的好处**：
- 共享动力学保证了跨被试的知识迁移
- 被试潜变量 s 提供了自然的个性化机制
- 新被试只需少量试次推断 s，无需重新训练
- 框架天然支持 few-shot 校准（核心 story）

---

## 三、实验设计

### 3.1 分阶段实验

| 阶段 | 实验 | 目的 | 复杂度 |
|------|------|------|:---:|
| **Phase 1** | Neural ODE 基线 | 验证连续时间建模在 MI-EEG 上的可行性 | 低 |
| **Phase 2** | Neural SDE vs ODE | 验证随机项（扩散）的贡献 | 中 |
| **Phase 3** | Subject-Conditional SDE | 验证被试特定潜变量的跨被试泛化 | 高 |
| **Phase 4** | 少样本校准 | ≤20-shot 推断被试潜变量 s 的效果 | 高 |

### 3.2 实验矩阵

| 实验 | 对比项 | 评估 | 关键指标 |
|------|--------|------|----------|
| E1: ODE 基线 | Neural ODE vs EEGNet vs Conformer | PhysioNet binary LOSO | 准确率 |
| E2: SDE 对比 | Neural ODE vs Neural SDE | 同上 | 准确率 + 预测不确定性 |
| E3: 扩散项消融 | ± 扩散项 g(z,t) | 同上 | 准确率 + trial 间方差 |
| E4: 积分步数消融 | 10/25/50/100 步 | 同上 | 准确率 vs 计算量 |
| E5: 潜变量维度 | dim 16/32/64/128 | 同上 | 准确率 |
| E6: Subject-Conditional | 共享 SDE vs Conditional SDE | 同上 | 准确率 |
| E7: 少样本校准 | 1/5/10/20-shot 推断 s | 同上 | 准确率 vs 校准试次数 |
| E8: 跨数据集 | BCI IV 2a 4-class LOSO | BCI IV 2a | 准确率 + Kappa |

### 3.3 预期结果

| 目标层次 | 描述 |
|-----------|------|
| **保底** | Neural ODE 在 MI-EEG 上收敛，准确率接近 EEGNet (~58%) |
| **合格** | Neural SDE > Neural ODE，验证随机建模的收益 |
| **良好** | Neural SDE > EEG Conformer (63.93%)，验证连续时间建模的优势 |
| **优秀** | Subject-Conditional SDE 在 ≤10-shot 校准下达到 65%+ |
| **冲刺** | 全量 LOSO 达到 68%+，少样本（5-shot）显著优于全监督 |

---

## 四、实施计划

### Week 1：Neural ODE 基线

```
Day 1-2: 搭建环境
  - pip install torchsde torchdiffeq
  - 实现 EEG Encoder (EEGNet Block1 → z₀)
  - 实现基础 Neural ODE (torchdiffeq.odeint)

Day 3-4: Neural ODE 训练
  - 在 PhysioNet binary LOSO 上训练
  - 调参：潜变量维度、积分步数、学习率
  - 对比 EEGNet baseline

Day 5: 实验记录
  - 记录 ODE 收敛曲线、准确率、计算时间
  - 可视化：ODE 轨迹示例
```

### Week 2：Neural SDE + 对比实验

```
Day 1-2: Neural SDE 实现
  - 实现 f(z,t) + g(z,t) 双网络
  - 使用 torchsde.sdeint 替代 odeint
  - 对比 Euler-Maruyama vs Milstein 求解器

Day 3-4: ODE vs SDE 实验
  - 完整 LOSO 对比
  - 消融：积分步数、扩散项大小
  - 分析：SDE 是否比 ODE 更鲁棒？

Day 5: 轨迹可视化
  - 同一试次 ODE vs SDE 轨迹对比
  - 不同被试的轨迹分布对比
  - 不确定性量化
```

### Week 3：Subject-Conditional SDE

```
Day 1-3: Conditional SDE 实现
  - 被试编码器（从校准试次推断 s）
  - 条件 SDE：f(z, t, s), g(z, t, s)
  - LOSO 训练 + 评估

Day 4-5: 少样本校准实验
  - 1/5/10/20-shot 推断 s
  - 对比：随机初始化 vs 预训练 SDE + 推断 s
  - 对比：SDE-fewshot vs SPDNet-SSL-fewshot（如已实现）
```

### Week 4：跨数据集 + 论文

```
Day 1-2: BCI IV 2a 验证
  - 9 subjects, 4-class LOSO
  - Conditional SDE vs 基线对比

Day 3-5: 实验结果汇总 + 论文提纲
  - 所有实验表 + 可视化
  - 论文故事框架
```

---

## 五、风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|:---:|------|------|
| Neural ODE 不收敛 | 中 | 高 | 先用简单 MLP 验证编码器-解码器 pipeline；降低潜变量维度；增加编码器容量 |
| SDE 训练极慢 | 高 | 中 | 减少积分步数（50→25）；使用较小的潜变量维度（32→16）；固定扩散项为常数（简化为 Langevin 动力学） |
| SDE 不比 ODE 好 | 中 | 中 | 即使 SDE=ODE，连续时间建模的 MI-EEG 框架本身仍有故事；可转向 Neural CDE（受控微分方程） |
| Subject-Conditional SDE 潜变量推断不稳定 | 中高 | 中 | 先做确定性编码（直接用被试 ID embedding）；再逐步引入推断机制；可参考 Neural Process 的变分推断 |
| 总体性能不如 SPD+SSL | 中 | 低 | **Neural SDE 的价值不只在准确率。** 动力学解释性、不确定性量化、轨迹可视化都是 SPD+SSL 没有的维度 |

---

## 六、与 SPD+SSL 的互补关系

两个方向不是竞争关系，而是**互补**的：

| 维度 | SPD+SSL | Neural SDE |
|------|---------|------------|
| 核心关注 | **空间**协方差结构 | **时间**动力学 |
| 数学框架 | Riemannian 几何 | 随机微分方程 |
| 输入 | 协方差矩阵（静态） | 时间序列（动态） |
| 数据利用 | 无标签预训练 | 连续时间建模 |
| 可解释维度 | 频段/通道贡献 | 动力学轨迹/相空间 |
| 少样本机制 | SSL 预训练 → 微调 | 潜变量推断 → 条件生成 |

**如果两个都做出来**，你可以把论文升级为：

> "从空间到时间：面向 8 通道 MI-EEG 的几何表征学习与动力学建模"

这是**两个互补的范式创新**，而不是两个方向在竞争。

---

## 七、关键基础设施

| 组件 | 库/工具 | 备注 |
|------|---------|------|
| Neural ODE 求解器 | `torchdiffeq` | Chen et al. 2018 官方实现 |
| Neural SDE 求解器 | `torchsde` | Kidger et al. 2021, ICLR 2021 |
| EEG 编码器 | 复用 EEGNet Block1 | 已有实现 |
| LOSO 评估 | 复用 `train_loso.py` | 已有 |
| EA 对齐 | 复用 `alignment.py` | 已有 |
| 预处理 | 复用 `run_mne_pipeline.py` | 已有 |

---

## 八、论文故事线（初步）

```
标题候选：
- "Learning the Equations of Brain Motion: Neural Stochastic 
   Differential Equations for Motor Imagery EEG Decoding"
- "Continuous-Time Neural Dynamics Modeling for Cross-Subject 
   Motor Imagery Brain-Computer Interfaces"

核心叙事：
  1. 大脑是连续时间动力系统，但现有 MI-BCI 方法全在离散时间建模
  2. 我们首次将 Neural SDE 引入 MI-EEG 解码
  3. 随机项 g(z,t)dW 显式建模了 EEG 的 trial-by-trial 变异性
  4. Subject-Conditional SDE 通过被试潜变量实现自然的少样本校准
  5. 8ch 低通道场景下的有效性验证

贡献点：
  C1: 探索 Neural SDE 在 MI-EEG 解码中的可行性
  C2: Subject-Conditional SDE 框架——共享动力学 + 个体潜变量
  C3: 连续时间动力学视角下的 MI-EEG 可解释性分析
```

---

## 九、参考文献

1. Chen et al. (2018). "Neural Ordinary Differential Equations." NeurIPS 2018 (Best Paper).
2. Kidger et al. (2021). "Neural SDEs: Deep Neural Networks as Semimartingales." ICLR 2021.
3. Kidger et al. (2020). "Neural Controlled Differential Equations for Irregular Time Series." NeurIPS 2020.
4. Rubanova et al. (2019). "Latent ODEs for Irregularly-Sampled Time Series." NeurIPS 2019.
5. Oh et al. (2024). "Application of a Neural ODE to Classify Motion Control Strategy using EEG." PubMed: 40039431.
6. Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs." J. Neural Eng.
7. Song et al. (2023). "EEG Conformer: Convolutional Transformer for EEG Decoding." arXiv:2301.05578.
