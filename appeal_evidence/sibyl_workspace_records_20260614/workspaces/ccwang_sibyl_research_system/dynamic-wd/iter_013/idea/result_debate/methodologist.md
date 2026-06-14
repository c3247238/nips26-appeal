# 方法论审查报告：Equilibrium-Driven Weight Decay (EqWD)

**审查日期**: 2026-03-25
**审查者**: 方法论者 Agent
**实验版本**: Iteration 3+（包含 ImageNet ResNet-50 全量实验）

---

## 执行摘要

EqWD 在 ImageNet ResNet-50（45 epoch）上以 72.271±0.197% 位居第一，超越 FixedWD 0.38%，在 CIFAR-100 ResNet-20 上达到 65.05±0.36%（第三名，低于 FixedWD 0.14%）。从方法论角度看，实验设计存在若干系统性问题，影响结论的可推广性和置信度，但核心发现仍具有一定可信度。以下逐项详述。

---

## 1. 45 Epoch ImageNet 训练：外部效度威胁

### 问题描述

ImageNet ResNet-50 的行业标准训练方案为 **90 epoch**（He et al., 2016 原文）或 **100+ epoch**（PyTorch 官方基线 Top-1 76.15%）。本实验采用 45 epoch，是标准方案的一半。FixedWD 在 45 epoch 下的均值 71.891% 对比标准 90 epoch 的约 76.1%，差距约 4.2%，说明模型尚未充分收敛。

### 对比与说明

| 方案 | Epochs | FixedWD Top-1 | EqWD Top-1 |
|------|--------|--------------|-----------|
| 本研究 | 45 | 71.891% | 72.271% |
| PyTorch baseline | 90 | ~76.15% | 未测试 |
| 标准 SGDW | 90+ | ~76.3% | 未知 |

### 具体威胁

**正面效应（支持外部效度）**: 在欠拟合阶段（训练未收敛），正则化方法间的差异往往被放大。如果 EqWD 在欠收敛条件下仍能领先，说明其在优化动态方面确有优势。

**负面效应（削弱外部效度）**:
- 充分训练至收敛后，不同 WD 策略的精度差距可能收窄至统计噪声水平。历史上多个 "训练时 trick 有效" 在完整收敛后消失的案例（例如，Fixup 初始化在短训练中优势显著但长训练后差异减小）。
- 45 epoch 方案对 warmup 期（通常前 5 epoch）的占比过高，而大多数动态 WD 方法（包括 EqWD）主要在中后期发挥作用。
- 余弦学习率衰减在 45 epoch 结束时 LR 已降至接近 0，但此时模型权重范数分布尚未达到 90 epoch 时的稳态——这使 EqWD 的均衡估计 r* 可能未完全收敛。

**评估结论**: 45 epoch 实验是有意义的，但**外部效度存在中等程度风险**。建议在论文中明确声明此局限性，并通过理论论证（如 EqWD 的优势在训练早中期更显著）来构建防御性论据，而非简单忽视。

---

## 2. 仅使用 SGD 优化器：适用性限制

### 问题描述

所有方法均基于 SGD（含动量，SGDW 框架）。当前深度学习实践中，**Adam/AdamW 是 NLP、ViT、Diffusion 模型的主流选择**，SGD 主要用于视觉分类任务。

### 适用性分析

**SGD 的合理性**（支持现有选择）:
- 权重衰减（WD）与 SGD 的耦合问题是 Defazio (2025) 和 Sun et al. (CVPR 2025) 工作的核心语境
- EqWD 的理论推导（梯度-权重比均衡 r* = lambda/gamma）基于 SGDW 的动力学假设
- CIFAR + ResNet 的 SGD 基线是社区的标准实验方案

**主要局限**:
- EqWD 在 Adam 框架下的行为未经验证。Adam 有自适应二阶矩估计，梯度-权重比的意义与 SGD 中不同，r* 的均衡分析不能直接移植
- 论文标题强调 "Unified Framework"，但统一仅覆盖 SGD 系列方法——对 AdamW 的缺位是显著的范围限制
- 在 Adam 优化下，CWD 和 CAWD 的行为（binary alignment masking）与 SGD 下可能有实质差异

**评估结论**: 适用性局限**中等严重**。论文须在 Scope 节明确声明 "本框架适用于 SGDW 系列"，并提供在 Adam 优化下的初步实验或理论分析作为 Future Work。若审稿人要求 Adam 对比，缺乏实验数据将是弱点。

---

## 3. 统计显著性分析：3 个种子是否足够？

### 现有数据的统计结构

**ImageNet ResNet-50（n=3）**:

| 方法 | 均值 | 标准差 | 95% CI（±t×σ/√n） |
|------|------|--------|-------------------|
| EqWD | 72.271 | 0.197 | ±0.490 |
| SWD | 72.044 | 0.401 | ±0.997 |
| FixedWD | 71.891 | 0.235 | ±0.584 |

注：t₂,₀.₀₂₅ = 4.303（双尾，df=2）

**EqWD vs. FixedWD 两样本 t 检验估计**:
- 差异 Δ = 0.380
- 合并标准差 ≈ √((0.197² + 0.235²)/2) ≈ 0.217
- t 统计量 ≈ 0.380 / (0.217 × √(2/3)) ≈ 2.03
- df = 4（pooled），p ≈ 0.11（双尾）

**结论：p > 0.05，差异在统计上不显著。**

**CIFAR-100 ResNet-20（n=3）**:
- EqWD 65.053 vs. FixedWD 65.193，差异仅 −0.14%
- EqWD 甚至排第三，低于 FixedWD

### 3 个种子的已知问题

- **自由度极低**（df=2 或 df=4）：t 分布的临界值极大，通常需要 Δ/σ > 3.5 才能达到 p<0.05
- **方差估计不稳定**：3 个点的样本标准差可能低估真实方差达 30-50%（小样本偏差）
- **多重比较问题**：7 种方法两两比较需要 Bonferroni 校正，显著性阈值应降至 p < 0.05/21 ≈ 0.0024

### 建议

1. **最低要求**: 补充到 5 个种子（df=8，统计功效显著提升）
2. **推荐方案**: 采用 bootstrap 置信区间（1000 次重采样），对每对方法报告置信区间重叠情况
3. **当前论文写作策略**: 如无法补充实验，应采用 "numerical improvement" 而非 "statistically significant improvement" 的措辞，如 "EqWD consistently outperforms FixedWD by 0.38% on ImageNet across all three random seeds"

**评估结论**: **统计显著性是本研究的最大方法论弱点**。ImageNet 主要结果（EqWD vs FixedWD，+0.38%）无法通过 p<0.05 检验。论文须诚实报告这一局限，或补充实验以增强置信度。

---

## 4. 超参数选择的公平性：EqWD beta 的调优问题

### 关键发现：beta 选择与公平性

实验中 EqWD 使用 **β=2.0，α=0.9**。ablation 数据显示：

| Beta | CIFAR-100 Best Acc（单次） |
|------|--------------------------|
| 0.1 | 65.21% |
| 0.5 | 65.07% |
| 1.0 | 65.39% |
| 2.0 | 65.35% |
| **5.0** | **66.07%** |

ImageNet 主实验使用 β=2.0 而非 ablation 中最佳的 β=5.0——这有一定合理性（避免过度调优），但存在以下问题：

**公平性争议点**:

1. **EqWD 经过 beta 搜索，其他方法是否也经过同等调优？**

   根据预算等价实验（BEM）的数据：FixedWD 用 Optuna 搜索了 15 次 trial，找到了 (lr=0.261, wd=0.0028) 的最优组合，达到 68.73%；EqWD 同样搜索了 15 次，找到 (lr=0.285, wd=0.00213, beta=4.40, ema_alpha=0.881)，达到 68.49%。

   **在 budget equivalence 设置下（均衡调优计算量），EqWD 的均值（68.30%）反而低于 FixedWD（68.21% ← 注意这里 FixedWD 搜索后结果更好）和 SWD（68.57%）**。BEM 结论明确：`dynamic_beats_fixed: False`。

2. **主实验中其他方法的 WD lambda 固定为 5e-4**，而 EqWD 的有效 WD 是动态放大的（β=2.0 意味着偏差大时 WD 可达 3×基础值）——这等同于 EqWD 使用了更大的实际 WD。

3. **CIFAR-100 VGG16-BN 上，EqWD 最好的 beta（pilot 结果中 beta=2.0 最佳）也在 main experiment 中使用**，但该设置来自不同架构的 pilot——存在跨架构超参泄漏风险。

**对比方法的调优情况**:
- SWD：使用了原论文（NeurIPS 2023）的默认参数
- CWD：使用了原论文（ICLR 2026）的默认参数
- CPR：使用了原论文（NeurIPS 2024）的默认参数

**结论**: 如果基线方法使用默认参数而 EqWD 使用调优后参数，比较是不公平的。**EqWD 在公平的 budget equivalence 测试（BEM）中未能显示优势**（dynamic_beats_fixed = False）是一个非常值得关注的负面结果，论文必须诚实呈现。

---

## 5. 可复现性评估

### 代码与数据可用性

**当前状态**:
- 实验代码存在于 `exp/code/` 目录
- 数据：CIFAR-100 标准公开数据集（可复现）；ImageNet-1K 在 `/home/ccwang/dataset/imagenet-1k`（HuggingFace parquet 格式，非标准 ILSVRC 格式）

**可复现性检查清单**:

| 要素 | 状态 | 评估 |
|------|------|------|
| 随机种子固定 | 是（42, 123, 456） | 良好 |
| PyTorch 版本记录 | 未知 | 需补充 |
| CUDA 版本记录 | 未知 | 需补充 |
| 硬件配置记录 | 8x RTX PRO 6000，98GB | 良好 |
| AMP 精度设置 | 已记录（enabled） | 良好 |
| 数据预处理步骤 | 未在报告中详细说明 | 需补充 |
| ImageNet 数据格式说明 | HuggingFace parquet | 非标准，可复现性风险 |
| 代码公开计划 | 未记录 | 需声明 |

**ImageNet 数据可复现性风险**:
- 标准 ILSVRC2012 使用 tar 格式；HuggingFace parquet 格式的预处理步骤（如 resize、augmentation pipeline）可能影响结果约 0.1-0.3%
- 需要明确记录 `transforms.Compose` 的具体参数（RandomResizedCrop 大小、ColorJitter 参数等）

**EqWD 算法复现**:
- 算法仅需 3 行代码（proposal 中有伪代码），理论上极易复现
- 然而 EMA 初始化策略（r* 的冷启动）和数值稳定性处理（eps 值）需要明确记录
- 层类型判断逻辑（normalized vs non-normalized）需要代码级说明

**评估结论**: 可复现性**中等**，需补充软件版本、数据预处理细节，以及代码开源承诺。

---

## 6. 缺失的对照实验

### 严重缺失

**6.1 Adam/AdamW 优化器对照**

适用于 Transformer、ViT 的 WD 调度实验完全缺失。建议至少增加 ViT-S 或 DeiT-S 在 Adam 优化器下的对照，哪怕只是 CIFAR-100 快速实验（~30 分钟）。

**6.2 更大规模：ResNet-101 或 EfficientNet**

根据 Project Memory，ResNet-101 是规划中的架构，但实验中未出现。规模扩展验证对 "unified framework" 的声称至关重要。

**6.3 收敛后的公平比较（90 epoch ImageNet）**

BEM 实验使用 100 epoch（CIFAR-100），但 ImageNet 主实验只有 45 epoch。建议至少对 EqWD 和 FixedWD 做一组 90 epoch ImageNet 的对照实验（仅 2 个方法 × 1 个种子也有参考价值）。

### 中等重要性缺失

**6.4 WD 预算显式控制的对照**

EqWD 动态调高 WD 意味着实际 WD 预算高于 lambda_base=5e-4。一个公平的对照是：固定总 WD 预算（sum_t lambda_t * gamma_t 相等），对比 FixedWD 与 EqWD 的效果。当前 BEM 实验设计（搜索最优 lambda）部分解决了这个问题，但 main experiment 和 BEM 结论矛盾，需要统一解释。

**6.5 BEM 负面结果的解释实验**

BEM 发现 `dynamic_beats_fixed: False`，即在公平调优条件下 EqWD 未超越 FixedWD（68.30% vs. 68.21%，差距仅 0.09%）。这与 main experiment 的 0.38% 差距形成矛盾。原因可能是：
- main experiment 的 FixedWD lambda=5e-4 是次优选择（BEM 找到的最优 lambda 是 0.0028，高 5.6 倍）
- 即 EqWD 的性能优势部分来自其动态机制**隐式地实现了更大的有效 WD**，而非因为动态性本身

这是一个重要的发现，需要增加一个控制实验：`FixedWD with lambda=0.0028` vs. `EqWD with lambda=5e-4`。

**6.6 Normalization 层的 WD 处理**

nobn_ablation 结果显示所有方法准确率均为 1.00%（随机水平），表明不含 BN 的 VGG-16 完全无法收敛。这个结果虽然说明了 BN 的重要性，但也暗示 EqWD 的层感知扩展（layer-type-aware formulation）在无 BN 场景下同样失效，且在有 BN 场景下（ablation 结果）layeraware 版本（62.32%）反而低于 uniform 版本（62.81%）。这一负面结果削弱了论文的层感知设计贡献，需要深入分析或降低对该设计的声称。

---

## 7. 总体方法论评分与建议

### 威胁汇总

| 威胁类型 | 严重程度 | 是否可在论文中防御 |
|---------|---------|-----------------|
| 45 epoch vs 90 epoch 外部效度 | 中等 | 可以（注明局限 + 理论论据） |
| 仅 SGD，缺乏 Adam 对照 | 中等 | 部分（明确 scope） |
| 统计显著性不足（p~0.11） | 高 | 困难（需补充实验或调整声称） |
| EqWD 超参调优不公平 | 中等 | 可以（BEM 实验已做，需整合叙述） |
| BEM 负面结果与主实验矛盾 | 高 | 困难（需控制实验 + 机制解释） |
| 可复现性信息不完整 | 低 | 容易（补充 requirements/环境） |
| 层感知设计结果为负 | 中等 | 部分（降低声称，作为 future work） |

### 关键建议（优先级排序）

1. **必须处理**（否则论文论证不成立）:
   - 诚实呈现 BEM 负面结果，解释 main experiment 与 BEM 之间的矛盾
   - 将统计声称改为 "numerically consistent" 而非 "statistically significant"
   - 增加 FixedWD(lambda=0.0028) 控制实验，区分 "动态性优势" vs. "有效 WD 量增加的效果"

2. **强烈建议**（显著提升论文质量）:
   - 补充至少 1 组 90 epoch ImageNet 实验（EqWD vs FixedWD，1 个种子即可）
   - 增加 seeds 至 5 个，或使用 bootstrap 报告置信区间
   - 降低层感知设计的贡献声称，改为 "preliminary analysis shows mixed results"

3. **建议处理**（提升严谨性）:
   - 明确 scope：本框架面向 SGDW 系列
   - 补充软件环境（PyTorch 版本、CUDA 版本、数据预处理细节）
   - 至少 1 个 Adam 优化器的初步实验（哪怕是 CIFAR-100 ViT-Tiny）

---

## 结论

EqWD 的核心思路（用梯度-权重比均衡偏差作为 WD 调制信号）在方法论上是合理且新颖的。实验设计在资源约束下达到了可接受水平，但存在若干可辩护的局限性。**最关键的两个问题是**：(1) 在公平调优条件下（BEM 测试），EqWD 未能超越 FixedWD，这一负面结果与主实验的正面结论形成矛盾，必须在论文中诚实面对；(2) 3 个种子的统计功效不足以支持 "significant improvement" 的声称。建议在投稿前补充控制实验并谨慎措辞，以应对可能的方法论审查。
