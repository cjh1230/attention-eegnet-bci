# Results Registry

> **目的**: 记录 README 中每个结果的确切来源（命令、seed、epochs、输出文件），
> 区分"有 JSON 文件可直接验证"和"仅 README 声明（无备份文件）"两类结果。
>
> **最后更新**: 2026-07-01

---

## 一、可验证结果（JSON 摘要文件存在）

### PhysioNet MI, 30 subjects, 8ch, binary LOSO

| # | 模型 | 配置 | JSON 文件 | Acc (JSON) | README Acc | 匹配? |
|---|------|------|-----------|------------|------------|-------|
| 1 | EEG-TCNet | + EA, 80ep, 默认seed | `loso_eeg_tcnet_ea_summary.json` | **63.48%** ± 11.56% | 63.41% ± 10.51% | ≈ (0.07pp) |
| 2 | Tangent + LDA | + EA, scm | `loso_riemann_tangent_ea_summary.json` | **60.44%** ± 9.64% | 60.44% ± 9.64% | ✅ 精确 |
| 3 | SPDNet d=[8,8] | + EA, seed42, 60ep | `loso_spdnet_ea_d8_8_seed42_summary.json` | **58.52%** ± 8.38% | 58.52% ± 8.38% | ✅ 精确 |
| 4 | SPDNet d=[8,8] | + EA, seed123, 60ep | `loso_spdnet_ea_d8_8_seed123_summary.json` | **59.56%** ± 8.56% | 56.96% ± 9.52% | ❌ +2.60pp |
| 5 | SPDNet d=[8,8] | + EA, seed456, 60ep | `loso_spdnet_ea_d8_8_seed456_summary.json` | **61.48%** ± 9.02% | 55.93% ± 8.83% | ❌ +5.55pp |
| 6 | SPDNet d=[8,8] | + EA, 默认seed, 60ep | `loso_spdnet_ea_summary.json` | **50.44%** ± 4.59% | 50.44% ± 4.59% | ✅ 精确 |
| 7 | SPDNet d=[8,8] | no EA, seed42, 60ep | `loso_spdnet_d8_8_seed42_summary.json` | **50.59%** ± 1.87% | 50.59% ± 1.87% | ✅ 精确 |
| 8 | SPDNet d=[8,8] | no EA, 默认seed, 60ep | `loso_spdnet_summary.json` | **50.59%** ± 1.87% | 50.59% ± 1.87% | ✅ 精确 |
| 9 | EEGNet | + EA, 60ep(?)*, 默认seed | `loso_eegnet_ea_summary.json` | **52.96%** ± 8.12% | 58.00% ± 10.06% | ❌ −5.04pp |

### ER-MI (Evidence Reasoning Network) — 新增 2026-07-04

| # | 模型 | 配置 | JSON 文件 | Acc (JSON) | 说明 |
|---|------|------|-----------|------------|------|
| ER1 | **ER-MI S3** | + EA, seed42, 80ep | `loso_er_mi_ea_seed42_summary.json` | **62.30%** ± 11.64% | 基准，κ=0.241 |
| ER2 | **ER-MI S3** | + EA, seed123, 80ep | `loso_er_mi_ea_seed123_summary.json` | **61.78%** ± 8.97% | κ=0.231 |
| ER3 | **ER-MI S3** | + EA, seed456, 80ep | `loso_er_mi_ea_seed456_summary.json` | **63.56%** ± 11.28% | κ=0.267 |
| ER4 | **ER-MI 3-seed mean** | + EA, 80ep | — | **62.55%** ± 0.92% | κ=0.246, 跨seed极稳定 |
| ER5 | ER-MI S1 | + EA, seed42, 80ep | `loso_er_mi_ea_seed42_summary.json`† | **62.15%** ± 12.22% | 步数消融: 一步推理 |
| ER6 | ER-MI S5 | + EA, seed42, 80ep | `loso_er_mi_ea_seed42_summary.json`† | **61.85%** ± 12.29% | 步数消融: 五步推理 |
| ER7 | ER-MI S3 无中间监督 | + EA, seed42, 80ep | `loso_er_mi_ea_seed42_summary.json`† | **?** | 中间监督消融 |

> † 被后续运行覆盖，结果仅记录在本文档中。建议后续实验使用 `--output_dir` 区分输出目录。
> ER-MI Step-wise 分析: S1=61.78% → S2=62.00% → S3=62.30% (微弱递增, +0.52pp)

> \* EEGNet + EA JSON 文件日期较早，epochs 数未记录在 JSON 中。52.96% 远低于 README 声明的 58.00%，推测 README 使用了更长的训练（80 epochs）或不同的 seed。
> 该 JSON 的 kappa=0.058 也显著低于 README 的 kappa=0.161，进一步证实并非同一次运行。

### BCI IV 2a, 9 subjects, 8ch, 4-class LOSO

| # | 模型 | 配置 | JSON 文件 | Acc (JSON) | README Acc | 匹配? |
|---|------|------|-----------|------------|------------|-------|
| 10 | SPDNet d=[8,8] | + EA, 60ep | `loso_spdnet_bci_iv_2a_ea_summary.json` | **38.97%** ± 13.99% | — | 未列入 README |

---

## 二、不可验证结果（无 JSON 备份文件）

以下结果在 README 中有声明，但 `results/` 目录下**没有对应的 JSON 或 CSV 文件**。
它们可能是从终端输出手工转录的，或 JSON 文件已丢失/被覆盖。

### PhysioNet MI — 缺失文件

| # | README 声明 | Acc | 应存在的文件名 | 说明 |
|---|-------------|-----|---------------|------|
| 11 | EEG Conformer + EA | 63.93% ± 9.58% | `loso_eeg_conformer_ea_summary.json` | **排行榜第1名，无备份** |
| 12 | EEG Conformer (no EA) | 61.33% ± ? | `loso_eeg_conformer_summary.json` | 无备份 |
| 13 | FBCNet + EA | 61.11% ± 11.69% | `loso_fbcnet_ea_summary.json` | 无备份 |
| 14 | FBCNet (no EA) | 49.70% ± 2.66% | `loso_fbcnet_summary.json` | 无备份 |
| 15 | EEGNet (no EA) | 51.93% ± 7.20% | `loso_eegnet_summary.json` | 无备份 |
| 16 | EEGNet + SpatiotemporalAttn + EA | 57.78% ± 8.55% | `loso_eegnet_spatiotemporal_ea_summary.json` | 无备份 |
| 17 | EEGNet + SpatiotemporalAttn | 55.04% ± 7.86% | `loso_eegnet_spatiotemporal_summary.json` | 无备份 |
| 18 | EEG-TCNet (no EA) | 61.56% ± ? | `loso_eeg_tcnet_summary.json` | 无备份 |
| 19 | MAA-EEGNet-Pre + EA | 56.00% ± 9.42% | — | 无备份 |
| 20 | MAA-EEGNet + EA | 55.33% ± 8.73% | — | 无备份 |
| 21 | FB-MAA-EEGNet + EA | 53.78% ± 7.68% | — | 无备份 |
| 22 | FgMDM + EA | 59.18% ± 8.12% | — | Riemannian, 无备份 |
| 23 | MDM + EA | 56.22% ± 10.52% | — | Riemannian, 无备份 |
| 24 | Tangent + LDA (no EA) | 60.44% ± 9.64% | — | Riemannian, 无备份 |

### Few-shot Calibration — 全部缺失

| # | README 声明 | Acc | 应存在的文件名 | 说明 |
|---|-------------|-----|---------------|------|
| 25 | Conformer + EA, 0-shot | 65.33% | `loso_eeg_conformer_ftsweep_ea_summary.json` | few-shot sweep 输出 |
| 26 | Conformer + EA, 5-shot | 66.38% | 同上 | |
| 27 | Conformer + EA, 10-shot | 67.47% | 同上 | **项目最高分，无备份** |
| 28 | Conformer + EA, 20-shot | 66.38% | 同上 | |
| 29 | Conformer + EA, 40-shot | 66.67% | 同上 | |

### BCI IV 2a — 缺失文件

| # | README 声明 | Acc | 应存在的文件名 | 说明 |
|---|-------------|-----|---------------|------|
| 30 | EEGNet base | 39.47% ± 12.45% | — | 无备份 |
| 31 | Tangent + LDA + EA | 38.60% ± 12.44% | — | 无备份 |
| 32 | EEGNet + SpatiotemporalAttn | 36.94% ± 11.78% | — | 无备份 |
| 33 | FgMDM + EA | 34.91% ± 8.48% | — | 无备份 |
| 34 | MDM + EA | 33.43% ± 10.92% | — | 无备份 |

---

## 三、关键差异分析

### 差异 1: EEGNet + EA — JSON 52.96% vs README 58.00%（差 5.04pp）

**根因**: JSON 文件 (`loso_eegnet_ea_summary.json`) 是旧版运行。JSON 中未记录 epochs 数，但 kappa=0.058 远低于 README 的 kappa=0.161，表明该次运行欠拟合或 epochs 较少。

**结论**: JSON 文件不可信。需要重新运行 EEGNet + EA 80 epochs 来生成可验证的结果文件。

### 差异 2: SPDNet seed123 — JSON 59.56% vs README 56.96%（差 2.60pp）

**根因**: README 中的数字很可能是转录错误。JSON 文件 (`loso_spdnet_ea_d8_8_seed123_summary.json`) 是机器生成的权威来源。

**结论**: README 应更新为 JSON 中的值 (59.56%)。按准确率排序的话，seed456 (61.48%) > seed123 (59.56%) > seed42 (58.52%)，与 README 中 seed42 > seed123 > seed456 的降序模式完全相反。

### 差异 3: SPDNet seed456 — JSON 61.48% vs README 55.93%（差 5.55pp）

同上。JSON 权威值为 61.48%，README 的 55.93% 是转录错误。

### 差异 4: Conformer 排行榜 63.93% vs Few-shot 表 0-shot 65.33%（差 1.40pp）

**可能原因**:
1. 两次运行使用的 epochs 数不同（few-shot sweep 可能用了更多 epochs 训练 base model）
2. seed 不同（few-shot sweep 碰巧抽到了更优的 seed）
3. few-shot sweep 的 0-shot 分支可能在训练逻辑上有细微差异

**结论**: 两者均无 JSON 备份，无法溯源。需要用受控 seed 重跑来消除歧义。

---

## 四、推荐重跑命令

以下命令用于重新生成缺失的结果文件，**全部使用 Task 2 新增的 `--seed` 参数**：

### 高优先级（排行榜前 3 + 差异项）

```bash
# Conformer + EA, 3 seeds
python main.py loso --data_dir data/loso_binary --model eeg_conformer --epochs 80 --align --seed 42
python main.py loso --data_dir data/loso_binary --model eeg_conformer --epochs 80 --align --seed 123
python main.py loso --data_dir data/loso_binary --model eeg_conformer --epochs 80 --align --seed 456

# TCNet + EA, 3 seeds
python main.py loso --data_dir data/loso_binary --model eeg_tcnet --epochs 80 --align --seed 42
python main.py loso --data_dir data/loso_binary --model eeg_tcnet --epochs 80 --align --seed 123
python main.py loso --data_dir data/loso_binary --model eeg_tcnet --epochs 80 --align --seed 456

# FBCNet + EA, 1 seed (then ± EA pair)
python main.py loso --data_dir data/loso_binary --model fbcnet --epochs 80 --align --seed 42
python main.py loso --data_dir data/loso_binary --model fbcnet --epochs 80 --seed 42

# EEGNet ± EA (resolve the 52.96% vs 58.00% discrepancy)
python main.py loso --data_dir data/loso_binary --model eegnet --epochs 80 --align --seed 42
python main.py loso --data_dir data/loso_binary --model eegnet --epochs 80 --seed 42
```

### 中优先级（补全缺失的 baseline）

```bash
# EEGNet + SpatiotemporalAttn ± EA
python main.py loso --data_dir data/loso_binary --model eegnet_spatiotemporal --epochs 80 --align --seed 42
python main.py loso --data_dir data/loso_binary --model eegnet_spatiotemporal --epochs 80 --seed 42

# BCI IV 2a FBCNet ± EA (fill ⏳ in docs/ea_analysis.md)
python main.py loso --data_dir data/bci_iv_2a_processed --n_subjects 9 --dataset bci_iv_2a --model fbcnet --epochs 60 --align --seed 42
python main.py loso --data_dir data/bci_iv_2a_processed --n_subjects 9 --dataset bci_iv_2a --model fbcnet --epochs 60 --seed 42
```

### 低优先级（few-shot sweep）

```bash
# Conformer few-shot sweep (reproduce the 67.47% claim)
python main.py loso --data_dir data/loso_binary --model eeg_conformer --epochs 80 --align --finetune_sweep 0,5,10,20,40 --seed 42

# TCNet few-shot sweep
python main.py loso --data_dir data/loso_binary --model eeg_tcnet --epochs 80 --align --finetune_sweep 0,5,10,20,40 --seed 42

# FBCNet few-shot sweep
python main.py loso --data_dir data/loso_binary --model fbcnet --epochs 80 --align --finetune_sweep 0,5,10,20,40 --seed 42
```

---

## 五、现有 JSON 文件来源推断

基于文件内容和命名模式，推断以下命令：

| JSON 文件 | 推断命令 |
|-----------|---------|
| `loso_eeg_tcnet_ea_summary.json` | `python training/train_loso.py --data_dir data/loso_binary --model eeg_tcnet --epochs 80 --align` |
| `loso_eegnet_ea_summary.json` | `python training/train_loso.py --data_dir data/loso_binary --model eegnet --epochs 60 --align` (推测) |
| `loso_riemann_tangent_ea_summary.json` | `python training/train_riemann_loso.py --data_dir data/loso_binary --n_subjects 30 --method tangent --align` |
| `loso_spdnet_ea_d8_8_seed42_summary.json` | `python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align --seed 42` |
| `loso_spdnet_ea_d8_8_seed123_summary.json` | `python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align --seed 123` |
| `loso_spdnet_ea_d8_8_seed456_summary.json` | `python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align --seed 456` |
| `loso_spdnet_ea_summary.json` | `python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --align` (默认seed) |
| `loso_spdnet_summary.json` | `python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60` |
| `loso_spdnet_d8_8_seed42_summary.json` | `python training/train_spd_loso.py --data_dir data/loso_binary --n_subjects 30 --epochs 60 --seed 42` |
| `loso_spdnet_bci_iv_2a_ea_summary.json` | `python training/train_spd_loso.py --data_dir data/bci_iv_2a_processed --n_subjects 9 --dataset bci_iv_2a --epochs 60 --align` |

---

## 六、建议

1. **立即**: 用 `--seed` 重跑 Conformer/TCNet/FBCNet/EEGNet ± EA，生成可追溯的结果文件
2. **更新 README**: SPDNet seed123/seed456 的数字应修正为 JSON 权威值
3. **版本化结果**: 所有重跑后的 JSON/CSV 保存在 `results/` 下，文件名包含 seed 标签
4. **不再手工转录**: 所有报告数字必须有对应的 JSON 文件作为证据
