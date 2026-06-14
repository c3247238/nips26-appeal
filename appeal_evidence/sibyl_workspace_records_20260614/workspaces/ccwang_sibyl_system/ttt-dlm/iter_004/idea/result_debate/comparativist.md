# Comparativist Analysis: BSD + RACFG 实验结果的文献定位

**分析时间**: 2026-03-10
**分析范围**: Iteration 4 全部 pilot 结果（Countdown-16, GSM8K-16），ablation 结果，compute-fair 对比
**数据来源**: arXiv 搜索、Google Scholar、Web 搜索

---

## 1. SOTA 对比：当前结果在已有工作中的位置

### 1.1 Countdown Benchmark

我们的实验在 Countdown-16 (pilot) 上的结果：

| 方法 | 准确率 | FLOPs | 来源 |
|------|--------|-------|------|
| Vanilla (128 steps) | 0.0-6.2% | 1.0x | 本实验 (不同 run) |
| DMI (alpha=0.3) | 0.0-12.5% | ~1.05x | 本实验 |
| BSD (k=0.75) | 6.2-12.5% | ~1.1x | 本实验 |
| A-CFG (w=1.5) | 12.5% | ~2.0x | 本实验 |
| BSD+A-CFG combo | 6.2% | ~2.1x | 本实验 |
| RACFG (JSD) | 0.0% | ~1.8x | 本实验 (FALSIFIED) |

**关键对比**：Dream 原论文报告 Countdown 16.0（8-shot），而我们的 vanilla baseline 在自生成 Countdown 问题上仅 0-6.2%。第 3 轮 500 样本 3 seed 验证显示 vanilla 4.7%。这一差距来源于：评估设置差异（zero-shot vs 8-shot）、问题集差异、生成参数差异。

**已发表方法在 Countdown 上的已知结果**：

| 方法 | 模型 | Countdown | 类型 | 来源 |
|------|------|-----------|------|------|
| Dream vanilla (8-shot) | Dream-7B | 16.0% | Baseline | Dream 原论文 |
| AGRPO | LLaDA-8B | +59.4% (相对) | RL 训练 | arXiv 2510.04019 |
| MDPO | Dream-7B | +54.2% (相对) | RL 训练 | arXiv 2508.13148 |
| MGDM | 从头训练 | 91.5% | 任务特定训练 | arXiv 2410.14157 |
| d2 (improved techniques) | LLaDA-8B | 显著提升 | 训练改进 | arXiv 2509.21474 |
| LookUM | LLaDA-8B | 匹配 RL 后训练 | Training-free | arXiv 2511.05563 |
| POKE (Path LL优化) | LLaDA/Dream | 改善 | Training-free | arXiv 2602.03496 |
| **Ours: A-CFG (best)** | **Dream-7B** | **12.5% (n=16)** | **Training-free** | **本实验** |
| **Ours: DMI (iter 3)** | **Dream-7B** | **9.3% (n=500, 3 seeds)** | **Training-free** | **本实验** |

**坦率评估**：在 training-free 方法阵营中，A-CFG 在 16 样本 pilot 上的 12.5% 与 DMI 的 12.5% 持平，但 A-CFG 需要 2x FLOPs 而 DMI 仅 1.05x。更重要的是，A-CFG 的原始论文（arXiv 2505.20199, NeurIPS 2025）已经在 LLaDA-8B 上报告了 GSM8K 73.5，**我们并未在 Dream-7B 上复现出与该论文可比的强力结果**。LookUM（arXiv 2511.05563）仅需 2-3 条路径即达峰值且匹配 RL 后训练效果，是更强的竞争者。

### 1.2 GSM8K Benchmark

| 方法 | 模型 | GSM8K | 类型 | 来源 |
|------|------|-------|------|------|
| Dream vanilla (8-shot) | Dream-7B | 77.2% | Baseline | Dream 原论文 |
| A-CFG | LLaDA-8B | 73.5% (4-shot) | Training-free | arXiv 2505.20199 |
| A-CFG | Dream-7B | 77.9% | Training-free | arXiv 2505.20199 |
| wd1 | LLaDA-8B | 84.5% | RL 训练 | arXiv 2507.08838 |
| DCoLT | LLaDA-8B | +9.8pp | RL 训练 | arXiv 2505.10446 |
| MetaState | Dream-7B | +1.2pp vs Instruct | 需训练 | arXiv 2603.01331 |
| LookUM | LLaDA-8B | +4pp | Training-free | arXiv 2511.05563 |
| LATTS | 多模型 | +4.1% | Training-free | OpenReview |
| Reward-Guided Stitching | 多模型 | +23.8% (平均) | 需 PRM + AR | arXiv 2602.22871 |
| **Ours: A-CFG (pilot)** | **Dream-7B** | **37.5% (n=16)** | **Training-free** | **本实验** |
| **Ours: Vanilla (iter 3)** | **Dream-7B** | **~29.6% (n=1300)** | **Baseline** | **本实验** |

**关键问题**：我们的 Dream-7B vanilla GSM8K ~29.6% 远低于 Dream 原论文的 77.2%。这表明我们的评估设置（zero-shot, origin 模式, 128 gen tokens）与论文设置（8-shot, 可能更长生成）有根本性差异。**这意味着我们的绝对数字不可直接与文献对比**，只有相对改善（方法 vs vanilla 的 delta）才有意义。

A-CFG pilot 的 37.5% vs vanilla 25.0%（+12.5pp，n=16）看似正面，但 95% CI 为 [-12.5%, +37.5%]，**完全不具统计显著性**。

### 1.3 RACFG 完败的定位

**H5 FALSIFIED**：RACFG（JSD 跨步稳定性引导）在 Dream-7B 上获得 0% 准确率，而 A-CFG（单步置信度重遮蔽）获得 12.5%。

**根本原因**：Dream-7B 的跨步 logit 分布极其稳定（JSD 稳定性 ~0.997），使 JSD 无法区分"推理关键"位置。这是**模型特性**，非算法缺陷——A-CFG 原论文在 LLaDA-8B 上工作良好是因为 LLaDA 的跨步分布变化更大。

这一发现对论文有两层含义：
1. **负面**：RACFG 作为 "BSD 的正交增强组件" 的提案完全失败
2. **正面但有限**：发现 Dream-7B 的 JSD 稳定性特性本身是有学术价值的观察

---

## 2. 贡献边际（Contribution Margin）评估

### 2.1 BSD 的实际贡献

BSD（Belief-State Diffusion）在 pilot 上的最佳结果：
- Countdown-16: 6.2-12.5%（不同 run），vs vanilla 0-6.2%
- 信念熵单调递减（Spearman rho=-0.95），终端熵低于 vanilla（0.001 vs 0.002）
- 计算开销极小（~1.1x）

**坦率评估**：

1. **H1（BSD >= 14%）未验证**：Pilot 仅 16 样本，最佳 12.5%，但 bootstrap CI 包含 0。需要 500 样本 3 seed 验证。

2. **BSD 的核心思想——用概率加权 embedding 混合替代 mask embedding——与以下已发表工作高度重叠**：
   - **Soft-Masked DLMs** (arXiv 2510.17206, NeurIPS 2025): "动态混合 mask embedding 与 top-k 预测 embedding"
   - **ReMix** (arXiv 2602.22868): "Continuous Mixing State 作为 masked state 和 decoded token state 之间的中间态"
   - **LRD** (arXiv 2510.11052): 信念状态精化，GSM8K +2.9
   - **EvoToken-DLM** (arXiv 2601.07351): 连续 token 进化
   - **DMI**（我们自己的 iter 3）: 前步 logits 加权 embedding 注入

   BSD 声称的差异化——"完全替代 mask embedding"而非"混合"——在 pilot 阶段并未产生显著优势（BSD ~= DMI ~= A-CFG 在 12.5% 附近）。

3. **信念熵分析是 BSD 最独特的贡献**。熵-准确率相关系数 r=0.78 (p=0.0003) 具有统计显著性。但这一分析性贡献能否支撑一篇论文取决于 full-scale 验证。

### 2.2 A-CFG 的结果定位

A-CFG 在 Dream-7B 上的表现：
- Countdown-16: 12.5% (w=1.5), vs vanilla 0%
- GSM8K-16: 37.5%, vs vanilla 25.0%
- 计算开销: ~2.0x

**关键问题**：A-CFG 是已发表方法（arXiv 2505.20199, NeurIPS 2025）。我们在 Dream-7B 上的复现**不构成新贡献**。A-CFG 原论文已在 Dream-7B 上报告 GSM8K 77.9%，远高于我们的 37.5%（评估设置差异导致）。

**我们能声称的最多是**：在我们的评估设置下，A-CFG 是唯一在 Countdown 和 GSM8K 上同时正面的方法。但这只是对已有方法的确认，不是新贡献。

### 2.3 BSD+A-CFG 组合的失败

**H7 FALSIFIED**：组合（6.2%）< max(BSD, A-CFG)（12.5%）。两个组件的正交互补假说不成立。

这是**致命的**。提案的核心架构——BSD 表示层 + RACFG/A-CFG 预测层的三层框架——在 pilot 上表现为 sub-additive。可能原因：
- BSD 的信念向量改变了输入分布，使 A-CFG 的置信度信号失效
- 两种修改同时作用导致 OOD 效应叠加

### 2.4 Compute-Fair 对比的残酷现实

**Pilot compute-fair 结果显示：vanilla 是 Pareto 最优的。**

在所有 FLOPs 水平上，简单增加 vanilla 步数都不劣于我们的任何方法。这意味着：
- BSD (1.1x FLOPs, 6.2%) < vanilla 1.1x (12.5%)
- A-CFG (2.0x FLOPs, 0%) < vanilla 2.0x (12.5%)（注意这是不同 run 的结果，表明 n=16 方差极大）

**警告**：n=16 的 compute-fair 结果随机性极大，不应作为最终结论。但方向性信号是负面的。

---

## 3. 并发工作检查（Concurrent Work）

### 3.1 直接竞争者（更新）

| 工作 | 发表时间 | 与我们的重叠 | 威胁等级 |
|------|---------|-------------|---------|
| **A-CFG** (arXiv 2505.20199, NeurIPS 2025) | 2025-05 | 我们的 RACFG 是 A-CFG 的（失败的）变体 | **致命** |
| **Soft-Masked DLMs** (arXiv 2510.17206, NeurIPS 2025) | 2025-10 | 与 BSD/DMI 核心思想高度重叠 | **高** |
| **MetaState** (arXiv 2603.01331) | 2026-03-02 | 直接解决 Information Island | **高** |
| **ReMix** (arXiv 2602.22868) | 2026-02 | Continuous Mixing State ≈ BSD 概念 | **高** |
| **LookUM** (arXiv 2511.05563) | 2025-11 | Training-free, 2-3 路径匹配 RL | **中** |
| **POKE** (arXiv 2602.03496) | 2026-02 | Path likelihood 优化, Countdown 改善 | **中** |
| **dllm** (arXiv 2602.22661) | 2026-02 | 简化 DLM 采样超参数调优 | **低** |

### 3.2 最严重威胁：A-CFG 已是 NeurIPS 2025 论文

A-CFG 已经被 NeurIPS 2025 接收，且在 LLaDA-8B 上报告 GSM8K 73.5%（超越 LLaMA3 8B 的 53.1%）和 Dream-7B GSM8K 77.9%。我们的 RACFG 试图在 A-CFG 之上添加"跨步稳定性信号"和"时间调度"，但**完全失败**。这使得我们在 CFG 方向上没有任何新贡献可言。

### 3.3 BSD vs Soft-Masked vs ReMix vs LRD

四个独立工作都在做"用连续表示替代/增强 mask embedding"：

| 维度 | Soft-Masked | ReMix | LRD | BSD (Ours) |
|------|------------|-------|-----|------------|
| 核心思想 | Top-k 预测 emb 混合 | 连续混合态 | 信念精化 | 概率加权 emb 替代 |
| 需要训练 | 是 (continued pretraining) | 否 | 否 | 否 |
| 跨步记忆 | 隐式 (via 混合) | 否 | 否 | 是 (EMA) |
| 验证基准 | 代码 (Dream-7B) | 加速 | GSM8K +2.9 | Countdown (pilot) |
| 发表状态 | NeurIPS 2025 | arXiv 2026-02 | arXiv 2025-10 | 未发表 |

**BSD 的差异化空间极窄**。唯一的潜在差异是"完全替代 mask embedding + EMA 跨步累积"，但 pilot 未能证明这比 DMI 的简单混合更好。

---

## 4. 新颖性评估

### 4.1 BSD 新颖性

**低-中**：核心思想（连续表示替代 mask embedding）已被 Soft-Masked (NeurIPS 2025)、ReMix、LRD 等覆盖。BSD 的 EMA 跨步累积是增量创新。信念熵分析有一定新颖性但属于分析贡献。

### 4.2 RACFG 新颖性

**已证伪**：RACFG 的核心假说（JSD 跨步稳定性 > 单步置信度）在 Dream-7B 上不成立。作为负面结果有诊断价值，但不构成正面贡献。

### 4.3 组合框架新颖性

**已证伪**：BSD + A-CFG 组合的正交互补假说（H7）被 pilot 否定。

### 4.4 整体论文新颖性

从 4 轮迭代的完整视角看：
- Iter 1-2: TTT 6 变体全部失败 (p=0.88)
- Iter 3: DMI 是唯一成功方法 (+4.6pp)，但与 Soft-Masked 概念重叠
- Iter 4: BSD 增量改进 DMI，RACFG 失败，组合失败

**最诚实的贡献定位**是一篇诊断性/分析论文："Why Inference-Time Parameter Adaptation Fails and Representation Enhancement Barely Helps in Masked Diffusion Language Models"，记录 20+ 方法变体的系统消融和失败原因分析。

---

## 5. 发表可行性评估

### 5.1 当前数据支持的投稿等级

**以目前仅有 pilot 数据（n=16），不支持任何投稿。** Pilot 的 McNemar 检验全部不显著（p >= 0.45），Bootstrap CI 全部包含 0。

### 5.2 最佳情况（Full-scale 确认 A-CFG + BSD 均有效）

如果 Countdown-500 x 3 seeds 显示 A-CFG >= 10% 且 BSD >= 8%（显著高于 vanilla 4.7%）：
- 可投 **EMNLP/ACL**: "Representation Enhancement vs Guidance for Masked Diffusion Language Models: A Systematic Study"
- 但须承认 A-CFG 是已有方法的复现，BSD 是 Soft-Masked 的 training-free 变体

### 5.3 中等情况（仅 A-CFG 有效，BSD ≈ DMI）

- 可投 **Workshop / Findings**: "Classifier-Free Guidance Works for Dream-7B Reasoning: An Empirical Confirmation with Negative Results on Cross-Step Stability"
- 核心价值在于确认 A-CFG 在 Dream-7B 上的效果 + RACFG 失败的根因分析

### 5.4 降级情况（Full-scale 确认所有方法 ≈ vanilla）

这是 iter 3 的 PPL 实验已经发生的情况（24-25% pilot 改善在 full-scale 上完全消失）。如果 iter 4 重演：
- 可投 **ACL Findings / EMNLP Findings**: 负面结果论文
- "Four Iterations of Failure: A Systematic Study of Why Training-Free Inference-Time Scaling Remains Elusive for Masked Diffusion Language Models"
- 价值在于：系统记录了 20+ 方法变体为什么在 DLM 上不工作

### 5.5 坦率的概率估计

基于历史模式（iter 3 的 pilot→full-scale 衰减 20-25pp）：
- A-CFG full-scale 显著 > vanilla 的概率: **40-50%**（A-CFG 有 NeurIPS 论文背书，但我们的评估设置可能不利）
- BSD full-scale 显著 > DMI 的概率: **20-30%**（pilot 差异 < 1 样本）
- 组合 > 单独最佳的概率: **<10%**（pilot 已否定）

---

## 6. 需要补充的基线和对比

### 6.1 缺失的关键基线

| 基线 | 重要性 | 原因 |
|------|--------|------|
| **Soft-Masked DLM** (Dream-7B finetuned) | **必须** | 与 BSD/DMI 直接竞争，NeurIPS 2025 |
| **LookUM** (2-3 paths) | **必须** | Training-free SOTA，compute-matched |
| **Best-of-N (vanilla, N=2-4)** | **必须** | 最简单推理时扩展基线 |
| **POKE** (Path LL optimization) | 高 | 直接在 Countdown 上验证过 |
| **CORE** (context-robust remasking) | 高 | Training-free, MBPP +9.2pp |
| **Prism** (HTS + SVF) | 中 | Training-free 搜索，多模型验证 |

### 6.2 缺失的分析

1. **与 Dream 原论文设置对齐的 baseline**：我们的 vanilla ~5%/~30% 远低于论文的 16%/77%，需要 8-shot 设置复现原论文数字
2. **LLaDA-8B 上的验证**：A-CFG 原论文主要在 LLaDA 上验证，我们应在 LLaDA 上复现以确认评估框架正确性
3. **Full-scale 统计检验**：n=500, 3 seeds, McNemar + Bootstrap CI + Bonferroni correction
4. **不同 Countdown 难度层的分层分析**：pilot 显示 69% 问题是 "hard"，方法间差异主要在 easy/hard 边界

---

## 7. 总结与建议

### 7.1 当前结果的诚实定位

**正面**：
- BSD 信念熵分析是一个有趣的学术观察（r=0.78, p<0.001）
- A-CFG 在 Dream-7B 上的 Countdown 和 GSM8K pilot 结果方向正确
- RACFG 失败的根因分析（Dream-7B JSD 稳定性 ~0.997）有诊断价值
- 系统消融覆盖了完整的方法谱系（表示层 × 预测层 × 组合）

**负面**：
- RACFG 核心假说被证伪
- BSD+A-CFG 组合假说被证伪
- BSD 未能显著超越 DMI
- Compute-fair 对比中 vanilla 是 Pareto 最优
- 所有结果基于 n=16，无统计显著性
- BSD 核心思想被 3+ 已发表工作覆盖（Soft-Masked, ReMix, LRD）
- A-CFG 是已有 NeurIPS 2025 论文的复现

### 7.2 风险缓解建议

1. **最优先**：启动 Countdown-500 x 3 seeds 的 full-scale A-CFG 和 BSD 评估。这是唯一能改变论文命运的数据。
2. **立即放弃**：BSD+A-CFG 组合和 RACFG。Pilot 已给出明确负面信号。
3. **对冲**：准备负面结果论文框架——如果 full-scale 重演 iter 3 的衰减，"系统性失败记录"本身有学术价值。
4. **基线补全**：实现 Best-of-N (N=2) 和 LookUM (2 paths) 作为 compute-matched 基线，确保论文审稿不被"比 Best-of-N 都差"卡住。
5. **评估设置对齐**：至少在 8-shot 设置下复现 Dream 原论文的 baseline 数字，确保绝对值可与文献对比。

### 7.3 对论文前景的坦率判断

基于 4 轮迭代的完整数据，**本项目产出顶会主会论文的概率约为 10-15%**。最可能的结局是一篇 Findings/Workshop 级别的论文，核心贡献是：
1. 在 Dream-7B 上确认 A-CFG 有效（复现性贡献）
2. 证明 JSD 跨步稳定性信号在 Dream-7B 上无效（负面结果）
3. BSD 信念熵分析（分析贡献）
4. 20+ 方法变体的系统消融（实验性贡献）

**如果要追求更高影响力**，建议考虑：
- 将研究重心从"新方法"转向"理论/实证分析"——为什么 training-free 推理时扩展在 DLM 上如此困难？
- 利用 4 轮迭代的丰富负面数据，构建一个"DLM 推理时扩展的经验性理论"
- 这种诊断性论文（如 "Scaling Laws for Inference-Time Compute in Diffusion Language Models"）可能比另一个 marginal 方法论文有更大的学术影响力

---

*本分析基于以下文献搜索综合整理：*
- *arXiv: 2505.20199 (A-CFG, NeurIPS 2025), 2510.17206 (Soft-Masked, NeurIPS 2025), 2603.01331 (MetaState), 2602.22868 (ReMix), 2511.05563 (LookUM), 2602.03496 (POKE), 2602.22661 (dllm), 2510.11052 (LRD), 2601.07351 (EvoToken), 2602.22871 (Reward-Guided Stitching), 2509.21474 (d2), 2507.08838 (wd1), 2505.10446 (DCoLT), 2410.14157 (MGDM)*
- *Google Scholar: masked diffusion language model inference-time scaling benchmark results*
- *Web 搜索: Dream 7B Countdown GSM8K SOTA, A-CFG LLaDA Dream GSM8K results*
