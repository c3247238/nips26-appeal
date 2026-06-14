# Iteration 4 实验计划

**日期**: 2026-03-18
**目标评分**: 7.0-7.5（从 Iter 3 的 5.5 提升）
**核心策略**: 修复地基优先，扩展范围其次

---

## 一、总体概览

### 实验矩阵

| 阶段 | 内容 | GPU 需求 | 预计时间 |
|------|------|----------|----------|
| Phase 0 | Metric 修复 + SGD 数据分析 | 0 GPU | 3-4 小时 |
| Phase 1 | Pilot 验证 | 1-2 GPU | 15-30 分钟 |
| Phase 2A | VGG-16-BN CIFAR-10/100 (AdamW) | 4-8 GPU | 3-4 小时 |
| Phase 2B | ResNet-20 额外种子 (789, 999) | 4 GPU | 2-3 小时 |
| Phase 2C | VGG-16-BN SGD 对照 | 4 GPU | 2-3 小时 |
| Phase 3 | 统计分析 + 可视化 | 0 GPU | 4-6 小时 |
| Phase 4 | ImageNet ResNet-50 (可选) | 8 GPU | 10-14 小时 |

**总计**: 约 22-30 GPU 小时，可在 8 卡并行下 1-2 天完成。

---

## 二、Phase 0: 零算力修复（P0 优先级）

### 0.1 修复三个诊断指标

#### BEM (Budget Equivalence Metric)
- **问题**: half_lambda BEM=0.000 是代码 bug；声称 [0,1] 有界但数学上不成立
- **根因分析**: `compute_bem()` 计算 `|mean(lambda_t) - lambda_const| / lambda_const`。对 half_lambda 方法，`get_metrics()` 返回 `effective_wd = self.phi.base_wd`（因为 HalfLambdaPhi 没有设置 `wd_schedule` 或 `mask_ratio` 诊断量），所以记录的 `effective_wd` 是 `base_wd=5e-4` 而非实际的 `2.5e-4`，导致 BEM=0
- **修复方案**:
  1. HalfLambdaPhi 添加 `self._diagnostics['effective_wd_scalar'] = self.base_wd * 0.5`
  2. `get_metrics()` 中统一使用 `effective_wd_scalar` 字段
  3. 改用有向 BEM（去掉绝对值）：`BEM_signed = (mean_wd - constant_wd) / constant_wd`
  4. 移除 [0,1] 有界性声明，改为说明 BEM 的符号含义（正=预算超支，负=预算不足）
- **验证**: half_lambda 预期 BEM = -0.5，cosine_schedule 预期 BEM ≈ -0.5

#### AIS (Alignment Informativeness Score)
- **问题**: 声称范围 [0,1]，但 cosine similarity 范围是 [-1,1]；当前实现取 `abs(cos)` 丢失方向信息
- **修复方案**:
  1. 保留 `abs(cos)` 的 entropy 计算作为 AIS（因为我们关心的是对齐的多样性，不是方向）
  2. 修正论文中范围声明为 [0,1]（这实际上是对的，因为实现取了 abs 后做 entropy 归一化）
  3. 补充逐层平均公式的显式表达：$\text{AIS} = H(\{|\cos(\mathbf{g}_l, \mathbf{w}_l)|\}_{l=1}^L) / \log(K)$
  4. 明确 $\Delta L$ 符号约定：$\Delta L_l = L(w - \eta g) - L(w)$（下降为负）

#### CSI (Coupling Stability Index)
- **问题**: 三个组件量纲不一致（weight norm 变化 vs 有效 WD vs gradient norm），等权加权无意义
- **修复方案**:
  1. 简化为单一定义：weight norm 变化序列的 CV（变异系数）
  2. $\text{CSI} = \text{std}(\Delta \|\mathbf{w}\|) / (\text{mean}(|\Delta \|\mathbf{w}\||) + \epsilon)$
  3. 归一化：$\text{CSI}_{\text{rel}} = \text{CSI}_{\text{method}} / \text{CSI}_{\text{constant}}$
  4. 这样 constant 基线的 CSI_rel = 1.0，其他方法相对于基线的稳定性一目了然

### 0.2 SGD Baseline 数据重新分析

- **数据位置**: `iter_003/exp/results/sgd_baseline/cifar10/resnet20/`
- **可用方法**: constant, cosine_schedule, cwd_hard, half_lambda, no_wd, random_mask, swd
- **种子**: 42, 123, 456（3 seeds）

**分析步骤**:
1. 从每个 `summary.json` 提取 `best_test_acc`
2. 用 scipy 重新计算配对 t 检验（以 constant 为基准）
3. 应用 Bonferroni-Holm 多重比较校正
4. 计算 Cohen's d 效应量
5. 与 AdamW 结果对比，计算效应量倍率差

**预期关键发现**:
- SGD constant vs no_wd 差异 ~0.91%（Cohen's d > 10）
- AdamW constant vs no_wd 差异 ~0.05%（不显著）
- 效应量倍率差 ~18x → 优化器特异性的核心证据

---

## 三、Phase 1: Pilot 验证（10-15 分钟/run）

### 1.1 VGG-16-BN 可行性 Pilot

**目的**: 验证 VGG-16-BN 训练管线正常，估算全量时间

| 配置 | 值 |
|------|-----|
| 架构 | VGG-16-BN |
| 数据集 | CIFAR-10（前 100 个样本） |
| 方法 | constant, cwd_hard, no_wd |
| 种子 | 42 |
| Epochs | 10 |
| 超时 | 900 秒 |
| GPU | 1 卡 |

**验证项**:
- 训练是否正常收敛（loss 下降）
- epoch 用时估算（外推到 200 epochs 全量数据）
- 诊断指标（CSI/AIS/BEM）是否有值（非 0/NaN）
- GPU 显存占用

### 1.2 Metric 修复验证 Pilot

**目的**: 验证修复后的 BEM/AIS/CSI 计算正确

| 配置 | 值 |
|------|-----|
| 架构 | ResNet-20 |
| 数据集 | CIFAR-10（全量） |
| 方法 | half_lambda, cosine_schedule |
| 种子 | 42 |
| Epochs | 5 |
| GPU | 1 卡 |

**验证项**:
- half_lambda BEM ≈ -0.5（修复后）
- cosine_schedule BEM ≈ -0.5（积分平均值应约为 base_wd/2）
- CSI_rel 基线归一化正确

---

## 四、Phase 2: 全量实验

### 2A: VGG-16-BN on CIFAR-10/100 (AdamW)

**实验矩阵**:

| 维度 | 值 |
|------|-----|
| 架构 | VGG-16-BN (14.7M params) |
| 数据集 | CIFAR-10, CIFAR-100 |
| 优化器 | AdamW |
| WD 方法 | constant, cosine_schedule, cwd_hard, cwd_soft, swd, half_lambda, no_wd |
| 种子 | 42, 123, 456, 789, 999 |
| Epochs | 200 |
| LR | 1e-3 |
| WD | 5e-4 |
| LR schedule | cosine annealing |
| Batch size | 128 |

**总 runs**: 7 方法 x 2 数据集 x 5 种子 = **70 runs**

**GPU 分配**: 8 卡并行，每卡约 8-9 runs，预计 3-4 小时

**关键假设验证**:
- Phi Invariance 是否跨架构成立？
- VGG-16-BN 和 ResNet-20 的 CSI/AIS 模式是否一致？
- 如果 VGG 破坏不变性 → 转向"架构调制"叙事

### 2B: ResNet-20 额外种子 (AdamW)

**补充种子**: 789, 999（现有 42, 123, 456）

| 维度 | 值 |
|------|-----|
| 架构 | ResNet-20 |
| 数据集 | CIFAR-10, CIFAR-100 |
| 优化器 | AdamW |
| WD 方法 | constant, cosine_schedule, cwd_hard, cwd_soft, swd, half_lambda, no_wd |
| 种子 | 789, 999（仅新增） |
| Epochs | 200 |

**总 runs**: 7 方法 x 2 数据集 x 2 种子 = **28 runs**

**GPU 分配**: 4 卡，约 2-3 小时

**统计收益**: 最小可检测效应从 ~0.93% 降至 ~0.50%（5 seeds, alpha=0.05, power=0.80）

### 2C: VGG-16-BN SGD 对照

| 维度 | 值 |
|------|-----|
| 架构 | VGG-16-BN |
| 数据集 | CIFAR-10, CIFAR-100 |
| 优化器 | SGD (momentum=0.9) |
| WD 方法 | constant, cosine_schedule, cwd_hard, no_wd |
| 种子 | 42, 123, 456 |
| Epochs | 200 |
| LR | 0.1 |
| LR schedule | multistep [80, 120, 160] |

**总 runs**: 4 方法 x 2 数据集 x 3 种子 = **24 runs**

**目的**: 验证 SGD 下动态 WD 的效果是否跨架构一致

---

## 五、Phase 3: 统计分析与可视化

### 3.1 TOST 等价检验

对每对 (method, constant) 在 AdamW 下执行 Two One-Sided Tests：
- **等价边界 Delta**: ±0.3%（主要报告）、±0.5%（补充）
- **检验**: paired t-test based TOST
- **每个组合**: 5 seeds，4 df
- **预期结论**: 大部分 AdamW 方法对在 Delta=0.3% 下等价

### 3.2 Power Analysis

- n=5 seeds, alpha=0.05
- 最小可检测效应 (MDE) 计算
- 基于观测到的组内 std (~0.15-0.30%)
- 报告 power > 0.80 的阈值效应量

### 3.3 可视化计划

| 图表 | 内容 | 数据来源 |
|------|------|----------|
| Fig: Training Curves | test accuracy vs epoch, mean±std | epoch_metrics.jsonl |
| Fig: Weight Norm Trajectories | weight norm vs epoch, 所有方法叠加 | epoch_metrics.jsonl |
| Fig: SGD vs AdamW Effect Sizes | 分组条形图，Cohen's d | summary.json |
| Fig: TOST Results | 等价区间图 | 统计分析结果 |
| Fig: CSI/AIS/BEM Heatmap | 方法 x 数据集 热力图 | summary.json |
| Fig: Cross-Architecture Comparison | ResNet-20 vs VGG-16-BN 准确率差异 | summary.json |

---

## 六、Phase 4: ImageNet ResNet-50（时间允许时执行）

| 维度 | 值 |
|------|-----|
| 架构 | ResNet-50 (25.6M params) |
| 数据集 | ImageNet-1K |
| 优化器 | AdamW |
| WD 方法 | constant, cosine_schedule, cwd_hard, no_wd |
| 种子 | 42, 123, 456 |
| Epochs | 90 |
| LR | 1e-3 |
| WD | 1e-4 |
| Batch size | 256 |

**总 runs**: 4 方法 x 3 种子 = **12 runs** (AdamW)
**SGD 对照**: constant + no_wd x 3 种子 = **6 runs**
**总计**: 18 runs，8 卡并行约 10-14 小时

**ImageNet 特殊处理**:
- 需要预下载 ImageNet 数据集到本地
- 可能需要 mixed precision (fp16) 加速
- 预备两条叙事路径：不变性成立 vs 规模依赖转变

---

## 七、超参数配置汇总

### AdamW 超参数（所有 CIFAR 实验统一）

| 参数 | 值 |
|------|-----|
| lr | 1e-3 |
| betas | (0.9, 0.999) |
| eps | 1e-8 |
| wd (base) | 5e-4 |
| batch_size | 128 |
| lr_schedule | cosine annealing |
| epochs | 200 |

### SGD 超参数

| 参数 | 值 |
|------|-----|
| lr | 0.1 |
| momentum | 0.9 |
| wd (base) | 5e-3 |
| batch_size | 128 |
| lr_schedule | multistep [80, 120, 160], gamma=0.1 |
| epochs | 200 |

### CWD 特定参数

| 参数 | 值 |
|------|-----|
| cwd_beta (soft) | 100.0 |
| swd_sensitivity | 1.0 |
| mask_prob (random) | 0.5 |

---

## 八、时间线与里程碑

### Day 1（立即开始）

| 时段 | 任务 | 预计用时 |
|------|------|----------|
| 上午 | Phase 0: 代码修复（BEM/AIS/CSI） | 2-3 小时 |
| 上午 | Phase 0: SGD 数据重新分析 | 1-2 小时 |
| 下午 | Phase 1: Pilot 验证 | 30 分钟 |
| 下午-晚上 | Phase 2A+2B: VGG-16-BN + ResNet-20 额外种子 | 3-4 小时（并行） |

### Day 2

| 时段 | 任务 | 预计用时 |
|------|------|----------|
| 上午 | Phase 2C: VGG-16-BN SGD 对照 | 2-3 小时 |
| 下午 | Phase 3: 统计分析 + 可视化 | 4-6 小时 |
| 晚上 | Phase 4: ImageNet 启动（可选） | 持续 10-14 小时 |

### Day 3

| 时段 | 任务 | 预计用时 |
|------|------|----------|
| 全天 | ImageNet 完成 + 结果整合 | — |
| 下午 | 论文更新（实验章节、表格、图表） | 4-6 小时 |

### 里程碑

| # | 里程碑 | 验收标准 |
|---|--------|---------|
| M1 | Metric 修复完成 | half_lambda BEM=-0.5，Pilot 通过 |
| M2 | SGD 分析完成 | Table 5 所有数字从 summary.json 自动生成 |
| M3 | VGG-16-BN 实验完成 | 70 runs (AdamW) 全部 _DONE |
| M4 | 5-seed 统计完成 | ResNet-20 28 runs 完成，TOST 结果生成 |
| M5 | 可视化完成 | 至少 6 张图表生成 |
| M6 | ImageNet 完成（可选） | 18 runs 完成，叙事路径确定 |

---

## 九、风险应对

| 风险 | 概率 | 应对 |
|------|------|------|
| VGG-16-BN 破坏 Phi Invariance | 低-中 | 转向"架构调制发现"叙事 |
| ImageNet 破坏不变性 | 中 | 预备"scale-dependent transition"叙事 |
| GPU 资源争用 | 低 | 本地 8 卡独占，Phase 2A/2B 可完全并行 |
| 5-seed TOST 仍无法通过 | 低 | 放宽 Delta 到 0.5%，或承认 power 不足 |
| 训练不稳定（VGG-16-BN）| 低 | 已有 BN + Kaiming init + Dropout 0.5 |
