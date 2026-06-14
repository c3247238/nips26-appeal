

## Project Spec
# 项目: ablation-no-debate

## 研究主题
研究稀疏自编码器（SAE）中的特征吸收（feature absorption）现象：系统分析和量化其成因、规律及对可解释性的影响。

## 背景与动机
稀疏自编码器（SAE）是机械可解释性研究的核心工具，用于从语言模型激活中提取人类可解释的特征。然而，SAE 存在"特征吸收"现象：某些低频但语义独立的特征会被高频特征"吸收"，导致 SAE 特征不完整，影响特征质量和下游解释性研究的可靠性。

目前，该现象的普遍程度、成因机制和定量影响尚不清楚，需要系统化的分析和量化。

## 初始想法
- 设计系统化的分析框架，量化特征吸收现象的普遍程度（跨不同模型层、不同 SAE 配置）
- 探索特征吸收的成因：与特征共现频率、稀疏惩罚强度、SAE 字典大小等因素的关系
- 开发可复现的评估指标，衡量吸收现象对下游解释性任务（如电路发现、概念探测）的影响
- 研究方法以 training-free 分析为主，基于现有的预训练 SAE（如 SAELens 库中的 SAE）进行分析

## 关键参考文献
- SAELens 库及其预训练 SAE（GemmaScope, GPT2-small SAE 等）
- Feature absorption 相关文献（待 Sibyl 文献调研补全）

## 可用资源
- GPU: 本地 GPU（有 SSH 访问）
- 服务器: default（SSH MCP 连接）
- 远程路径: /home/qhxie/sibyl_system

## 实验约束
- 实验类型: training-free（分析现有预训练 SAE，不重新训练）
- 模型规模: 小到中等（GPT-2, Gemma-2B 等）
- 时间预算: 单实验 ≤ 1 小时，pilot 10-15 分钟

## 目标产出
- 学术论文（NeurIPS/ICLR 级别）
- 包含：特征吸收的量化分析、成因分析、对可解释性影响的实验

## 特殊需求
- 以 training-free 分析为主，充分利用 SAELens 现有预训练模型
- 论文应包含可复现的评估框架，方便社区后续研究


## User's Initial Ideas
- 设计系统化的分析框架，量化特征吸收现象的普遍程度（跨不同模型层、不同 SAE 配置）
- 探索特征吸收的成因：与特征共现频率、稀疏惩罚强度、SAE 字典大小等因素的关系
- 开发可复现的评估指标，衡量吸收现象对下游解释性任务（如电路发现、概念探测）的影响
- 研究方法以 training-free 分析为主，基于现有的预训练 SAE（如 SAELens 库中的 SAE）进行分析

## Seed References (from user)
- SAELens 库及其预训练 SAE（GemmaScope, GPT2-small SAE 等）
- Feature absorption 相关文献（待 Sibyl 文献调研补全）

## 文献调研报告（请仔细阅读，避免重复已有工作）
# Literature Survey Report

**Research Topic**: 研究稀疏自编码器（SAE）中的特征吸收（feature absorption）现象：系统分析和量化其成因、规律及对可解释性的影响。

**Survey Date**: 2026-04-27

**arXiv Search Keywords**: ["sparse autoencoder feature absorption interpretability", "SAE superposition dictionary learning", "mechanistic interpretability autoencoder transformer", "SAE feature sensitivity measurement evaluation"]

**Web Search Keywords**: ["sparse autoencoder feature absorption phenomenon 2025", "SAE superposition feature visualization interpretability", "SAE feature absorption hierarchy superposition LLM", "SAELens GEMMA scope sparse autoencoder open source"]

## 1. Field Overview

稀疏自编码器（Sparse Autoencoders, SAEs）已成为大语言模型（LLM）机械可解释性（Mechanistic Interpretability）研究的核心工具。SAE 通过稀疏字典学习将 LLM 密集、多义的激活分解为人类可解释的潜在特征方向，其理论基础建立在**线性表示假说（Linear Representation Hypothesis）**和**叠加（Superposition）**现象之上。

然而，近期研究揭示了 SAE 的一个根本性局限——**特征吸收（Feature Absorption）**现象。当底层特征形成层级结构时（如"以 S 开头"和"short"），SAE 的稀疏性优化目标会驱动更具体的特征"窃取"更一般特征的方向表示，导致看似单义的潜在特征在其应该激活的情况下失效。这对 SAE 作为可靠分类器的应用构成了严峻挑战。

当前研究领域呈现以下趋势：
- **特征分裂（Feature Splitting）**：较宽 SAE 中特征分裂为更细粒度的子特征（如"以 L 开头"分裂为"大写 L 开头"和"小写 l 开头"）
- **特征吸收作为分裂的负面变体**：更具体特征吸收一般特征方向，造成假阴性
- **评估方法多元化**：从最大激活样本分析转向 precision/recall/F1 量化和敏感性测量
- **层次语义建模**：尝试将语义层级结构显式编码到 SAE 架构中

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv 2409.14507 | 2024 | **首次系统定义特征吸收现象**，提出吸收率（absorption rate）量化指标，通过玩具模型证明层级特征导致吸收的数学机制 | 仅验证到 layer 17（ ablation 依赖），无法直接迁移到高层 |
| 2 | Measuring Sparse Autoencoder Feature Sensitivity | arXiv 2509.23717 | 2025 | 提出特征敏感性（feature sensitivity）作为评估新维度，发现许多可解释特征敏感性差 | 需要 LLM 生成相似文本，计算开销大 |
| 3 | Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs | arXiv 2403.19647 | 2024 | 将 SAE 特征与因果图结合，支持通过特征进行干预和编辑 | 特征吸收会影响稀疏因果路径的完整性 |
| 4 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | DeepMind/Gemma | 2024 | 发布 Gemma 2 全层的 16k/65k/1m 宽度 SAE，覆盖 27 层 | 特征吸收在不同层均有发生 |
| 5 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Anthropic | 2024 | 展示了大规模 SAE 提取可解释特征的能力 | precision/recall 不平衡问题 |
| 6 | End-to-End Sparse Dictionary Learning | arXiv 2405.12241 | 2024 | 提出端到端 SAE 训练方法，确保特征功能重要性 | 计算成本更高 |
| 7 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv 2505.20254 | 2025 | 强调 SAE 特征一致性（feature consistency）的重要性，提出 PW-MCC 指标 | 不同训练运行的特征不一致性 |
| 8 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | arXiv 2506.01197 | 2025 | 提出显式建模语义层级的 SAE 架构 | 仍在探索阶段 |
| 9 | Route Sparse Autoencoder to Interpret Large Language Models | arXiv 2503.08200 | 2025 | 提出 RouteSAE，通过路由机制跨层提取特征 | 未考虑特征吸收问题 |
| 10 | Superposition as Lossy Compression: Measure with SAEs | arXiv 2512.13568 | 2025 | 信息论框架量化叠加程度，连接叠加与对抗鲁棒性 | 特征吸收导致的欠表示未充分讨论 |

## 3. SOTA Methods and Benchmarks

### 当前最佳方法

| 方法 | 描述 | 主要贡献 |
|------|------|----------|
| **JumpReLU SAE** | 使用跳跃 ReLU 激活的 SAE 变体 | 改善重建保真度 |
| **TopK SAE** | 每步激活 top-k 个潜在特征 | 避免 L1 正则化调参 |
| **BatchTopK SAE** | 跨 batch 选择 top-k*B 个激活 | 提高平均重建质量 |
| **端到端 SAE (E2E)** | 最小化 KL 散度而非重建损失 | 确保特征功能重要性 |
| **Meta-SAE** | 在已有 SAE decoder 上训练新 SAE | 可分解复合特征 |

### 主流数据集与评测

| 数据集/资源 | 说明 | 获取方式 |
|-------------|------|----------|
| **Gemma Scope SAEs** | Gemma 2 27 层的 16k/65k/1m SAE | [DeepMind](https://arxiv.org/abs/2408.00516) |
| **SAELens** | SAE 训练和分析库 | [GitHub](https://github.com/jbloomAus/SAELens) |
| **Neuronpedia** | 在线 SAE 特征仪表盘平台 | [neuronpedia.org](https://neuronpedia.org) |
| **SAE Spelling Dataset** | 首个字母识别任务的测试集 | [GitHub](https://github.com/lasr-spelling/sae-spelling) |

### 评估指标

- **Precision/Recall/F1**：基于线性探针的分类性能
- **Feature Sensitivity**：特征在语义相似文本上的激活可靠性
- **Absorption Rate**：吸收率量化层级特征的吸收程度
- **Explained Variance**：重建质量度量
- **L0（激活稀疏度）**：活跃潜在特征的平均数量

## 4. Identified Research Gaps

### Gap 1: 特征吸收的理论解决方案
**问题**：仅改变 SAE 大小或稀疏度不足以解决特征吸收。现有 toy model 证明层级特征的吸收是稀疏性优化的必然结果。

**探索方向**：
- Group Lasso 或层次稀疏编码（Hierarchical Sparse Coding）
- Attribution Dictionary Learning
- Meta-SAE 扩展：允许吸收发生并利用其恢复特征层级

### Gap 2: 高层特征的吸收检测
**问题**：当前吸收率指标依赖 ablation 实验，仅适用于 layer 0-17。深层 attention 已将信息移到最终 token 位置，ablation 影响消失。

**探索方向**：
- 利用编码器-解码器不对称性检测吸收
- 结合注意力流分析推断深层吸收模式

### Gap 3: 特征一致性与可靠性
**问题**：不同训练运行学到的特征集合不同（feature consistency 低），影响可解释性研究的可复现性。

**探索方向**：
- PW-MCC 指标指导架构选择
- 标准化训练流程和初始化策略

### Gap 4: 跨模态泛化
**问题**：SAE 在音频、视觉领域的应用面临独特挑战（密集性语义缺失），特征吸收在不同模态的表现尚不清楚。

**探索方向**：
- 针对音频 latent space 的 SAE 训练框架
- 视觉 Transformer 的 SAE 可解释性（Prisma 工具包）

## 5. Available Resources

### Open-source Code

| Repository | Description | Language | License |
|------------|-------------|----------|---------|
| [SAELens](https://github.com/jbloomAus/SAELens) | SAE 训练、分析和评估库 | Python | Apache 2.0 |
| [sae-spelling](https://github.com/lasr-spelling/sae-spelling) | 特征吸收研究的实验代码 | Python | MIT |
| [Prisma](https://github.com/Prisma-Multimodal/ViT-Prisma) | 视觉/视频机械可解释性工具包 | Python | - |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | 机械可解释性分析框架 | Python | BSD |
| [RouteSAE](https://github.com/swei2001/RouteSAEs) | 跨层路由 SAE 实现 | Python | - |

### Pretrained Models & Weights

| Resource | Description | Access |
|----------|-------------|--------|
| Gemma Scope SAEs | Gemma 2 27 层全覆盖 SAE (16k/65k/1m) | HuggingFace / DeepMind |
| Neuronpedia | 50+ 模型的在线特征仪表盘 | Web UI |
| E2E SAE | 功能重要性导向的预训练 SAE | ApolloResearch GitHub |

### Datasets

| Dataset | Description | Task |
|---------|-------------|------|
| First-letter ICL prompts | "{token}hasthefirstletter:{letter}" 格式 | 字母识别 |
| Enron Email | PII 泄露隐私研究 | 隐私保护 |
| Chess/Othello boards | 合成任务验证 ground truth | 电路发现 |

## 6. Implications for Idea Generation

### Worth Exploring

1. **层次感知 SAE 架构**：显式建模特征的层级关系，可能从根本上解决吸收问题
2. **吸收作为层级发现信号**：利用吸收的编码器-解码器不对称性逆向推断特征层级
3. **多粒度特征融合**：结合 feature splitting 和 absorption 的观察，设计多尺度表示
4. **跨层特征追踪**：RouteSAE 思路扩展到吸收感知的跨层特征传播分析

### Saturated Directions

1. 单纯扩大 SAE 宽度或降低 L0
2. 纯可视化分析（最大激活样本展示）
3. 单层、单模型的孤立分析

### Cross-domain Potential

- **代码合成**：SAE 在代码模型中的应用，识别 API 调用、函数定义等层级特征
- **多语言迁移**：特征吸收在不同语言中的表现差异
- **安全可解释性**：利用吸收分析检测模型的隐藏行为（如欺骗性）

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens library | High | Apache 2.0 | Adopt | 功能完整的 SAE 训练和分析框架，支持 Gemma Scope |
| Gemma Scope SAEs | High | - | Adopt | DeepMind 发布的高质量预训练 SAE，可直接用于实验 |
| Neuronpedia | High | - | Adopt | 在线交互式分析工具，加速特征理解 |
| Feature Absorption Paper Code (sae-spelling) | High | MIT | Extend | 提供了吸收率指标和 toy model 实现，可扩展到新实验 |
| E2E SAE | Medium | - | Extend | 功能重要性导向的 SAE，可与吸收分析结合 |
| RouteSAE | Medium | - | Compose | 跨层特征提取思路，结合层次感知设计 |
| Meta-SAE | Medium | - | Compose | 特征分解技术，可用于吸收的后处理分析 |

### Recommended Research Pipeline

```
1. 基础分析 (SAELens + Gemma Scope)
   ├── 加载预训练 SAE
   ├── 计算各层吸收率基线
   └── 特征可视化与分类

2. 吸收量化方法开发
   ├── 扩展吸收率指标到高层
   ├── 编码器-解码器不对称性分析
   └── 层级推断算法

3. 解决方案探索
   ├── 层次感知 SAE 架构设计
   ├── Group Lasso 正则化
   └── 吸收感知的训练目标

4. 应用验证
   ├── 稀疏特征电路构建
   ├── 跨模型泛化（Llama, Qwen, Gemma）
   └── 安全关键场景测试
```

---

## References

### Key Papers
- Chanin et al. (2024). "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507
- Tian et al. (2025). "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717
- Song et al. (2025). "Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs." arXiv:2505.20254
- Muchane et al. (2025). "Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures." arXiv:2506.01197
- Lieberum et al. (2024). "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2."

### Tools & Libraries
- SAELens: https://github.com/jbloomAus/SAELens
- SAE Spelling: https://github.com/lasr-spelling/sae-spelling
- Neuronpedia: https://neuronpedia.org
- Prisma: https://github.com/Prisma-Multimodal/ViT-Prisma


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Research Proposal: Feature Absorption in SAEs Under Random Baseline Scrutiny

## Metadata

- **Iteration**: 1 (initial synthesis)
- **Date**: 2026-04-27
- **Front-runner**: Candidate P1 (Pragmatist) + Candidate E1 (Empiricist) hybrid
- **Status**: Pre-pilot, no experimental results yet

---

## Abstract

Feature absorption -- the phenomenon where parent features in SAEs fail to fire correctly and are instead "handled" by their child features -- is a fundamental limitation for mechanistic interpretability. We propose a rigorous controlled study with two complementary components: (1) establishing whether absorption is a genuine SAE artifact or a statistical inevitability by comparing against random baselines, and (2) testing whether absorbed features causally fail downstream tasks via steering intervention. Together, these provide both a sanity check on the phenomenon itself and a causal account of its practical consequences.

---

## Motivation

The SAE field has documented absorption as a problem (Chanin et al., 2024) but lacks two critical pieces of evidence:

1. **Sanity check**: Do SAEs actually learn absorption, or is it an artifact of how we measure it? Korznikov et al. (2026) showed random baselines match SAEs on many interpretability metrics. No work has tested whether this extends to absorption-specific metrics.

2. **Causal link**: Is absorption merely a descriptive curiosity, or does it actively degrade SAE reliability? Without intervention-based causal evidence, we cannot prioritize absorption mitigation over other failure modes.

This proposal addresses both gaps through controlled synthetic experiments and causal intervention.

---

## Research Questions

1. Do trained SAEs exhibit higher absorption rates than random baselines on controlled synthetic feature hierarchies?
2. Does artificially correcting absorbed feature directions improve feature sensitivity (causal test)?
3. What is the relationship between absorption rate and downstream task utility?
4. Do absorbed features show systematically different encoder-decoder geometry compared to non-absorbed features?

---

## Hypotheses

| ID | Hypothesis | Falsification Criterion | Expected Outcome |
|----|-----------|------------------------|-----------------|
| H1 | Trained SAEs show higher absorption rates than random baselines on synthetic hierarchies | Baseline absorption rate not significantly lower than SAE (t-test p > 0.05) | SAEs absorb; baselines do not |
| H2 | Absorbed features have measurably higher encoder-decoder asymmetry (norm ratio != 1) | Absorbed and non-absorbed features show identical asymmetry distributions | Asymmetry metric is a valid proxy |
| H3 | Steering absorbed features toward parent directions improves sensitivity | No significant sensitivity improvement after intervention (paired t-test p > 0.05) | Absorption is not causally responsible for sensitivity failures |
| H4 | Absorption rate inversely correlates with feature activation frequency | No correlation (Pearson r > -0.2, p > 0.05) | Frequency-based predictions of absorption are invalid |

---

## Method

### Part A: Baseline Comparison (Pragmatist)

**Synthetic Feature Hierarchy Generation**
- Create controlled hierarchies with 3 levels (parent, child, grandchild)
- Example: "animals" (parent) -> "dogs/cats" (children) -> "poodles/huskies/tabbies/persians" (grandchildren)
- Encode as linear feature directions in d=512 space; parent direction is a linear combination of children

**SAE Training**
- Train standard SAEs using SAELens on synthetic activations
- Architectures: width expansion 8x, L0 targets {16, 32, 64}
- 5 random seeds per configuration

**Baseline Construction**
- **Baseline A (Random decoder)**: Same architecture, Xavier initialization, no training
- **Baseline B (Shuffled features)**: Real activations with permuted feature index assignments
- **Baseline C (Permuted encoder)**: Trained SAE with encoder weights randomly shuffled

**Absorption Measurement**
- Ablation-based absorption rate: parent ablation should not affect child activation if absorption is absent
- Measure: % of parent feature activations that are "explained" by child features

### Part B: Causal Intervention (Empiricist)

**Feature Selection**
- 50 features with known high absorption (absorption rate > 0.3) from Part A
- 50 matched features with low absorption (absorption rate < 0.1)

**Intervention Design**
- Compute parent feature direction via decoder analysis
- Add steering vector (alpha in {0.05, 0.1, 0.15, 0.2, 0.25}) to SAE activations during inference
- Compare sensitivity before/after steering

**Evaluation**
- Feature sensitivity: activation rate on semantically similar text
- Reconstruction quality: monitor explained variance during steering
- Statistical test: paired t-test with Bonferroni correction

### Part C: Encoder-Decoder Asymmetry Analysis (Supplementary)

- Compute Asymmetry Index (AI): ||W_encoder[i]|| / ||W_decoder[:, i]||
- For features from Part A, correlate AI with measured absorption rate
- Validate AI as a training-free absorption proxy

---

## Experimental Plan

| Phase | Task | Duration | Notes |
|-------|------|----------|-------|
| 1a | Generate synthetic hierarchies + validate pipeline | 30 min | Pilot for measurement methods |
| 1b | Train 3 SAEs (L0=16,32,64) x 5 seeds | 30 min | ~45 GPU-min |
| 2 | Construct baselines and measure absorption | 20 min | Compare all conditions |
| 3 | Statistical analysis + asymmetry computation | 15 min | t-tests, correlation |
| 4 | Steering intervention on absorbed features | 20 min | Causal validation |
| 5 | Downstream task evaluation | 15 min | Probing, circuit discovery |

**Total estimated runtime**: ~2.5 hours GPU time (within 1-hour task budget with parallelization)

---

## Evidence-Driven Revisions

N/A - no prior pilot results exist yet.

---

## Revisions from Prior Feedback

N/A - this is the initial proposal.

---

## Novelty Assessment

| Candidate | Novelty Finding |
|-----------|----------------|
| Baseline comparison | **Novel** - No prior work tests absorption metrics against random baselines. Korznikov et al. (2026) tested general interpretability but not absorption-specific metrics. |
| Encoder-decoder asymmetry | **Novel** - No prior work mathematically connects encoder-decoder geometry to absorption detection. arXiv search found no matching work. |
| Steering intervention | **Novel** - No prior work tests whether causal intervention on absorbed features improves sensitivity. |
| Safety-critical application | **Novel** - No prior work specifically examines absorption for rare safety-critical features. |

**Known close prior art**: "Ensembling Sparse Autoencoders" (Gadgil et al., arXiv:2505.16077, May 2025) proposes SAE ensembles for reconstruction and diversity but does not address absorption detection or hierarchical feature recovery. Our multi-resolution approach differs in goal (absorption recovery via sparsity diversity) and mechanism (diagnostic experiment with orthogonality test).

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Absorption not detectable in toy model | Medium | High | Use more complex hierarchies; fall back to real activations |
| Baselines show absorption too | Low-Medium | High | Would be significant finding; document carefully as negative result |
| Steering destabilizes reconstructions | Medium | Medium | Monitor explained variance; use small alpha values |
| Asymmetry metric fails to correlate | Medium | Medium | Pivot to alternative metrics; drop Part C if invalid |

---

## Contributions

1. **First rigorous sanity check for absorption**: Establishes whether absorption is a genuine SAE phenomenon or an artifact of sparse optimization interacting with measurement
2. **Causal evidence for absorption impact**: Tests whether absorbed features actively fail downstream tasks via steering intervention
3. **Training-free absorption proxy**: Encoder-decoder asymmetry as a detectable signal for absorption without ablation
4. **Open-source experimental framework**: Reproducible pipeline for controlled absorption evaluation

---

## References

- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Korznikov et al. (2026). Sanity Checks for SAEs. (pending publication)
- Tian et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717
- Song et al. (2025). Feature Consistency Position. arXiv:2505.20254
- Muchane et al. (2025). Hierarchical SAE. arXiv:2506.01197
- Lieberum et al. (2024). Gemma Scope
- Tang et al. (2025). Unified Theory of SDL. arXiv:2512.05534
- Gadgil et al. (2025). Ensembling SAEs. arXiv:2505.16077
- Karvonen et al. (2024). Board Game SAE Evaluation. arXiv:2408.00113


## 当前可检验假设
# Testable Hypotheses

## Primary Hypotheses (Front-runner: cand_p1)

### H1: Trained SAEs Show Higher Absorption Rates Than Random Baselines

**Hypothesis**: Trained SAEs on synthetic 3-level feature hierarchies exhibit significantly higher absorption rates than random decoder baselines and shuffled feature baselines.

**Test**: Compare absorption rate across conditions:
- Trained SAE (5 seeds, L0 ∈ {16, 32, 64})
- Random decoder baseline
- Shuffled feature baseline
- Permuted encoder baseline

**Expected outcome**: Trained SAE absorption rate ~20-40%, baselines near 0% or random.
**Falsification**: Baseline absorption rate not significantly lower than SAE (t-test p > 0.05).

---

### H2: Absorbed Features Have Higher Encoder-Decoder Asymmetry

**Hypothesis**: Features identified as absorbed (via ablation) show measurably higher encoder-decoder asymmetry than non-absorbed features, measured by norm ratio and cosine similarity.

**Test**: Compute asymmetry metrics for all features in trained SAEs from H1 experiment. Correlate with ablation-based absorption labels.

**Expected outcome**: Pearson correlation between asymmetry index and absorption rate > 0.4.
**Falsification**: Absorbed and non-absorbed features show identical asymmetry distributions.

---

### H3: Steering Absorbed Features Improves Sensitivity

**Hypothesis**: For absorbed features, artificially steering activations toward parent feature directions (via decoder analysis) significantly improves feature sensitivity scores.

**Test**: Before/after comparison on absorbed vs. non-absorbed features with 5 steering alpha values. Paired t-test across features.

**Expected outcome**: Significant sensitivity improvement for absorbed features (p < 0.01), no improvement for non-absorbed controls.
**Falsification**: No significant sensitivity improvement after steering (p > 0.05) OR non-absorbed features show equal improvement.

---

### H4: Absorption Rate Inversely Correlates With Feature Frequency

**Hypothesis**: Features with lower activation frequency show systematically higher absorption rates, consistent with the competitive exclusion prediction from sparse optimization.

**Test**: Plot absorption rate vs. activation frequency for all features. Compute Pearson/Spearman correlation.

**Expected outcome**: Negative correlation (r < -0.3, p < 0.01).
**Falsification**: No significant negative correlation.

---

## Supplementary Hypotheses (Backup candidates)

### H_Asym1: Encoder-Decoder Norm Ratio Correlates With Ablation Absorption

**Hypothesis**: The encoder-decoder norm ratio ||W_encoder[i]|| / ||W_decoder[:, i]|| correlates with ablation-based absorption rate across all features.

**Test**: Compute norm ratio for all features in Gemma Scope SAEs (layers 0-8). Correlate with absorption rates from Chanin et al.

**Expected outcome**: Pearson r > 0.5.
**Falsification**: r < 0.3.

---

### H_Ens1: Multi-Resolution Ensemble Recovers Absorbed Features

**Hypothesis**: Child features recoverable in high-L0 SAE (L0=256) are absorbed in low-L0 SAE (L0=16), and cross-SAE matching correctly identifies parent-child pairs.

**Test**: Train ensemble (L0=16, 64, 256). Match features via decoder cosine similarity. Verify absorbed features in low-L0 SAE have recovered representations in high-L0 SAE.

**Expected outcome**: >50% of absorbed features have recoverable children in ensemble.
**Falsification**: <20% recovery rate.

---

### H_Safe1: Safety-Critical Features Have Elevated Absorption

**Hypothesis**: Features annotated as safety-critical show higher absorption rates than matched non-safety features.

**Test**: Human annotation of 40 Gemma Scope features as safety-critical vs. non-safety. Compare absorption rates.

**Expected outcome**: Safety features show mean absorption rate > non-safety features (Mann-Whitney p < 0.05).
**Falsification**: No significant difference in absorption rates.

---

## Summary Table

| ID | Candidate | Hypothesis | Metric | Threshold | 
|----|-----------|-----------|--------|-----------|
| H1 | front_runner | SAEs > baselines on absorption | Absorption rate | t-test p < 0.05 |
| H2 | front_runner | Asymmetry = absorption proxy | Pearson r | r > 0.4 |
| H3 | front_runner | Steering improves sensitivity | Paired t-test | p < 0.01 |
| H4 | front_runner | Frequency = inverse absorption | Pearson r | r < -0.3 |
| H_Asym1 | backup | Norm ratio = absorption | Pearson r | r > 0.5 |
| H_Ens1 | backup | Ensemble recovers absorbed | Recovery rate | >50% |
| H_Safe1 | backup | Safety = higher absorption | Mann-Whitney | p < 0.05 |


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_p1",
      "title": "Feature Absorption Under Random Baseline Scrutiny: A Controlled Study",
      "status": "front_runner",
      "summary": "Test whether trained SAEs exhibit higher absorption rates than random baselines on synthetic feature hierarchies. Combines pragmatic implementation (toy model + baselines) with empirical validation and causal steering intervention. Addresses the sanity check gap identified by Korznikov et al. (2026) for absorption-specific metrics.",
      "hypotheses": [
        "H1: Trained SAEs show higher absorption rates than random baselines on synthetic hierarchies (t-test p < 0.05)",
        "H2: Absorbed features have measurably higher encoder-decoder asymmetry than non-absorbed features",
        "H3: Steering absorbed features toward parent directions improves sensitivity (paired t-test p < 0.01)",
        "H4: Absorption rate inversely correlates with feature activation frequency (Pearson r < -0.3)"
      ],
      "pilot_focus": "Single 3-level hierarchy, L0=32, 3 seeds. Validate measurement pipeline. Compare trained SAE vs. 2 baselines. Estimated 30 minutes.",
      "key_papers": [
        "Chanin et al. 2024",
        "Korznikov et al. 2026",
        "Tian et al. 2025"
      ],
      "merge_contributions": {
        "pragmatist": "Low-risk experimental design with clear positive/negative outcome. Baseline comparison validates whether absorption is real.",
        "empiricist": "Causal steering intervention establishes whether absorption actively degrades downstream utility.",
        "theoretical": "Encoder-decoder asymmetry metric provides training-free absorption proxy.",
        "innovator": "Cross-layer detection extends beyond ablation-limited lower layers.",
        "contrarian": "Safety-critical framing strengthens impact narrative."
      },
      "novelt_flag": "none",
      "estimated_runtime_hours": 2.5,
      "drop_trigger": "If all 4 hypotheses falsified simultaneously"
    },
    {
      "candidate_id": "cand_a1",
      "title": "Encoder-Decoder Asymmetry as Training-Free Absorption Proxy",
      "status": "backup",
      "summary": "Analyze encoder-decoder weight asymmetry (norm ratio, cosine similarity) as a proxy for absorption detection across all layers. Theoretically motivated by compressive sensing geometry. Validated on chess/Othello ground-truth features before extrapolation to high layers.",
      "hypotheses": [
        "H_asym1: Encoder-decoder norm ratio correlates with ablation-based absorption rate (Pearson r > 0.5)",
        "H_asym2: Absorbed features have systematically lower encoder-decoder cosine similarity",
        "H_asym3: Asymmetry metric enables absorption detection in layers 18+ where ablation fails"
      ],
      "pilot_focus": "Compute asymmetry index for Gemma Scope layer 0-8 SAEs. Validate against known absorption rates. Estimated 30 minutes.",
      "key_papers": [
        "Chanin et al. 2024",
        "Karvonen et al. 2024",
        "Tang et al. 2025"
      ],
      "novelt_flag": "novel_metric",
      "novelty_note": "No prior work mathematically connects encoder-decoder geometry to absorption detection. arXiv search returned no direct matches.",
      "estimated_runtime_hours": 3,
      "drop_trigger": "If correlation between asymmetry and absorption is < 0.3"
    },
    {
      "candidate_id": "cand_a2",
      "title": "SAE Ensembles for Hierarchical Feature Recovery (MRSAE)",
      "status": "backup",
      "summary": "Train ensemble of SAEs with varying L0 targets to collectively recover hierarchical features. Grounded in immune system clonal selection analogy. Each sparsity level captures different resolution features; combined ensemble recovers full hierarchy.",
      "hypotheses": [
        "H_ens1: High-L0 SAE captures child features that low-L0 SAE has absorbed",
        "H_ens2: Cross-SAE feature matching identifies absorbed parent-child pairs",
        "H_ens3: Ensemble collectively achieves lower effective absorption rate than single SAE"
      ],
      "pilot_focus": "Train 3 SAEs (L0=16,64,256) on synthetic hierarchical data. Match features across SAEs. Test recovery on 5 known absorbed pairs. Estimated 45 minutes.",
      "key_papers": [
        "Muchane et al. 2025",
        "Gadgil et al. 2025"
      ],
      "novelt_flag": "partial_overlap",
      "novelty_note": "Overlaps with Gadgil et al. (May 2025) on SAE ensembles but differs in goal (absorption recovery via sparsity diversity vs. general reconstruction/diversity). Must differentiate by focusing specifically on hierarchical feature recovery with orthogonality diagnostic.",
      "estimated_runtime_hours": 4,
      "drop_trigger": "If ensemble does not recover significantly more absorbed features than single SAE"
    },
    {
      "candidate_id": "cand_a3",
      "title": "Safety-Critical Rare Features and Absorption",
      "status": "backup",
      "summary": "Test whether SAEs systematically fail to represent rare safety-critical features due to absorption. If confirmed, SAE-based safety analysis (deception detection, jailbreak identification) is unreliable precisely for the cases that matter most.",
      "hypotheses": [
        "H_safe1: Safety-critical features have higher absorption rates than matched non-safety features",
        "H_safe2: SAE-based steering is less effective on absorbed safety features than on non-absorbed ones",
        "H_safe3: Absorbed safety features show lower sensitivity scores"
      ],
      "pilot_focus": "Select 20 human-annotated safety-relevant features from Gemma Scope. Compare absorption rates with 20 matched non-safety features. Estimated 20 minutes.",
      "key_papers": [
        "Makelov et al. 2024",
        "Chanin et al. 2024"
      ],
      "novelt_flag": "novel_application",
      "novelty_note": "No prior work specifically examines absorption for rare safety-critical features. Contrarian framing is novel for this application.",
      "estimated_runtime_hours": 3,
      "drop_trigger": "If safety-relevant features do not show elevated absorption rates"
    }
  ],
  "pool_status": {
    "front_runner": "cand_p1",
    "backups": ["cand_a1", "cand_a2", "cand_a3"],
    "dropped": [],
    "pivot_priority": ["cand_a1", "cand_a2", "cand_a3"]
  }
}


## 上一轮 validation 决策意见
# Idea Validation Decision

## Pilot Evidence Summary

| Task | Status | Key Metrics |
|------|--------|-------------|
| Task 1: Synthetic Data | PASS | Hierarchy validated; children-grandchildren similarity=1.0; grandchildren orthogonal=-0.034 |
| Task 2: SAE Training | PASS | L0=32.0, explained variance=97.7%, loss converged |
| Task 3-4: Baselines+Absorption | PARTIAL PASS | Overlap: trained SAE=0.50, random baseline=0.25, DIFF=0.25. Ablation: both=1.0 (FAIL) |

**Key Findings**:
- **H1 (overlap method): SUPPORTED** - Trained SAE shows 0.50 absorption vs 0.25 for random baseline (delta=0.25)
- **H1 (ablation method): FALSIFIED** - Both trained SAE and random baseline reach 1.0 absorption; ablation cannot differentiate
- **H2 (asymmetry): NOT SUPPORTED** - Asymmetry index nearly identical: trained SAE=0.487+/-0.013 vs random baseline=0.471+/-0.009; no separation
- **H3 (steering): NOT TESTED** - Pilot did not run steering intervention
- **H4 (frequency): NOT TESTED** - Pilot did not run frequency correlation analysis
- **Critical red flag**: The ablation method failure means the primary H1 falsification criterion (t-test on ablation-based absorption) cannot be met with current measurement approach

## Decision Matrix

### cand_p1 (Front-runner, only candidate with pilot data)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | Overlap method shows delta=0.25 (signal), but ablation method saturates at 1.0 for both conditions (no signal). Mixed evidence. |
| Hypothesis survival | 0.25 | 2 | H1 partially survives (overlap). H2 directly contradicted by data. H3/H4 untested. Risk that H1 falsification criterion (ablation-based) cannot be satisfied. |
| Path to full result | 0.20 | 3 | Clear pipeline exists; need to fix ablation measurement methodology. Steering and frequency analysis not yet attempted. |
| Novelty (from report) | 0.15 | 3 | Novelty score 6/10. Korznikov collision is significant. Differentiation via absorption-specific metrics + steering combination is valid but needs repositioning. |
| Resource efficiency | 0.10 | 4 | Pilot completed in ~45 min. Full plan is 3.5 hours, within budget. Synthetic data approach is efficient. |
| **Weighted Score** | **1.00** | **2.95** | **REFINE** |

### cand_a1, cand_a2, cand_a3 (Backups, no pilot data)

| Candidate | Evidence | Weighted Score Estimate |
|-----------|---------|------------------------|
| cand_a1 | No pilot run. Asymmetry metric needs validation. Korznikov/OrtSAE collision noted. | N/A (no pilot) |
| cand_a2 | No pilot run. Overlaps with Gadgil. L0-diversity mechanism needs proof. | N/A (no pilot) |
| cand_a3 | No pilot run. Highest novelty (9/10) but requires human annotation (40 features). Safety framing compelling. | N/A (no pilot) |

### Sanity Checks
- [x] Compared all candidates with data (only cand_p1 has pilot data)
- [x] Penalized candidate for failed falsification criterion (ablation method failure penalized H1 survival)
- [x] Did NOT let sunk cost sway decision (pilot effort is irrelevant; only evidence matters)
- [x] Did NOT blindly advance when pilot was partially inconclusive (ablation failure is a real concern)

## Decision Rationale

**REFINE** is the correct decision because:

1. **The ablation measurement failure is a critical methodological issue.** H1's falsification criterion explicitly requires ablation-based absorption to differentiate trained SAE from baseline. The current ablation method saturates at 1.0 for both conditions, meaning we cannot reliably measure the phenomenon we're studying.

2. **The overlap method provides a genuine signal.** The 0.25 delta (trained SAE 0.50 vs random baseline 0.25) is real and consistent with the absorption hypothesis. This validates that the synthetic hierarchy and measurement pipeline work.

3. **The asymmetry metric (H2) failed immediately.** Trained SAE and random baseline have nearly identical asymmetry indices. If H2 cannot be rescued, the "training-free proxy" contribution is weakened.

4. **Two of four hypotheses remain untested.** H3 (steering) and H4 (frequency) were not in the pilot scope. We have no evidence for or against them.

5. **The Korznikov collision is manageable but requires repositioning.** The novelty report correctly identifies that the baseline methodology is not novel in isolation. The combination of absorption-specific metrics + steering intervention is the actual contribution.

**Why not ADVANCE**: The ablation failure means we do not have a reliable measurement for the core phenomenon. Advancing to a full experiment with flawed measurement would produce uninterpretable results.

**Why not PIVOT**: The overlap method evidence (delta=0.25) is genuine. The idea has merit; only the measurement methodology needs refinement. Pivot would waste the validated synthetic hierarchy and pipeline.

**Confidence**: 0.45 (moderate). The decision is constrained by mixed pilot evidence and untested hypotheses.

## Next Actions

1. **FIX MEASUREMENT (BLOCKING)**: Design a more granular ablation-based absorption metric. The current method (parent ablation effect on single child) saturates. Options:
   - Multi-child ablation: ablate multiple top children and measure residual effect
   - Proportional absorption: measure what fraction of parent activation is "explained" by children, not just binary presence/absence
   - Ablation on held-out samples: measure absorption generalization, not just in-sample fit

2. **VALIDATE ASYMMETRY METRIC (H2)**: The current norm ratio did not differentiate. Try:
   - Cosine similarity between encoder/decoder columns
   - Projection-based asymmetry: measure how much of decoder[i] lies in encoder[i]'s orthogonal complement
   - If H2 cannot be rescued, consider dropping it or pivoting to cand_a1's approach

3. **RUN MULTI-SEED PILOT**: Current pilot is single-seed (seed=42). Run 3 seeds to establish variance. This is needed for the t-test falsification criterion.

4. **TEST H3 (STEERING) AND H4 (FREQUENCY)**: These were deferred from pilot but are needed for full hypothesis coverage.

5. **PARALLEL BACKUP DEVELOPMENT**: While refining cand_p1 methodology:
   - Start cand_a3 pilot (highest novelty, no Korznikov collision) to have a backup if measurement cannot be fixed
   - cand_a3 requires only 20 annotated features and ~20 minutes; it is a low-cost hedge

SELECTED_CANDIDATE: cand_p1
CONFIDENCE: 0.45
DECISION: REFINE


## 上一轮 validation 结构化决策
{
  "decision": "REFINE",
  "selected_candidate_id": "cand_p1",
  "confidence": 0.45,
  "candidate_scores": {
    "cand_p1": {
      "weighted_score": 2.95,
      "verdict": "REFINE",
      "component_scores": {
        "pilot_signal_strength": 3,
        "hypothesis_survival": 2,
        "path_to_full_result": 3,
        "novelty": 3,
        "resource_efficiency": 4
      },
      "evidence": "Overlap method: trained SAE=0.50, random baseline=0.25 (delta=0.25, signal present). Ablation method: both=1.0 (saturation, no signal). Asymmetry index: trained SAE=0.487+/-0.013, random baseline=0.471+/-0.009 (no differentiation). H3 and H4 not tested."
    },
    "cand_a1": {
      "weighted_score": null,
      "verdict": "NO_PILOT",
      "evidence": "No pilot data available. Asymmetry metric needs validation. Collision with OrtSAE noted."
    },
    "cand_a2": {
      "weighted_score": null,
      "verdict": "NO_PILOT",
      "evidence": "No pilot data available. Collision with Gadgil et al. on ensemble mechanism."
    },
    "cand_a3": {
      "weighted_score": null,
      "verdict": "NO_PILOT",
      "evidence": "No pilot data available. Highest novelty (score 9/10) but requires 40 human-annotated features."
    }
  },
  "reasons": [
    "Overlap method supports H1: trained SAE absorption 0.50 vs random baseline 0.25 (delta=0.25)",
    "Ablation method fails: both trained SAE and random baseline saturate at 1.0 absorption",
    "H2 directly contradicted: asymmetry index indistinguishable between conditions (0.487 vs 0.471)",
    "H3 (steering) and H4 (frequency) remain untested",
    "Ablation measurement failure blocks H1 falsification criterion (t-test requires differentiation)",
    "Single-seed pilot (seed=42) insufficient for statistical significance",
    "Korznikov collision requires repositioning but is manageable with absorption-specific focus"
  ],
  "critical_findings": [
    "Ablation-based absorption measurement saturates at 1.0 for both trained SAE and random baseline - measurement methodology must be fixed before full experiment",
    "Asymmetry index (norm ratio) does not differentiate trained from random: H2 not supported by pilot data",
    "Overlap-based absorption measurement shows genuine signal (delta=0.25) validating the synthetic hierarchy pipeline",
    "SAE training works correctly: L0=32, explained variance=97.7%"
  ],
  "next_actions": [
    "FIX: Design granular ablation-based absorption metric (multi-child ablation, proportional absorption, or held-out sample generalization)",
    "VALIDATE: Test alternative asymmetry formulations for H2 (encoder-decoder cosine similarity, projection-based asymmetry)",
    "EXPAND: Run multi-seed pilot (3 seeds) to establish variance for t-test falsification criterion",
    "TEST: Run H3 (steering intervention) and H4 (frequency correlation) pilots to establish signal",
    "HEDGE: Start cand_a3 pilot (highest novelty, no Korznikov collision) in parallel as backup",
    "ACKNOWLEDGE: Reposition around absorption-specific metrics + causal steering (avoid direct baseline methodology claim)"
  ],
  "blocked_hypotheses": ["H1 (ablation method)", "H2 (asymmetry norm ratio)"],
  "supported_hypotheses": ["H1 (overlap method)"],
  "untested_hypotheses": ["H3 (steering)", "H4 (frequency)"],
  "dropped_candidates": [],
  "pivot_trigger_condition": "If ablation measurement cannot be fixed to differentiate trained SAE from random baseline after 2 refinement iterations, pivot to cand_a3 (safety-critical features) as primary candidate"
}


## 上一轮新颖性检查报告（必须针对发现的撞车问题进行修正）
# Novelty Report: Feature Absorption in SAEs Under Random Baseline Scrutiny

**Workspace**: ablation-no-debate / iter_001
**Date**: 2026-04-27
**Checker**: sibyl-novelty-checker

---

## Executive Summary

This proposal studies feature absorption in sparse autoencoders (SAEs) through two complementary lenses: (1) random baseline comparison and (2) causal steering intervention. The most significant prior-art collision is **Korznikov et al. (2026)** (arXiv:2602.14111), which overlaps substantially with Part A (baseline comparison). However, the combination with Part B (causal steering) and the focus on absorption-specific metrics provides partial differentiation. The three backup candidates show varying degrees of overlap with recent literature.

---

## Candidate-by-Candidate Analysis

---

### cand_p1: Feature Absorption Under Random Baseline Scrutiny

**Novelty Score: 6/10** (partial_overlap, proceed with significant repositioning)

#### Collisions

**1. Korznikov et al. (2026) — "Sanity Checks for SAEs: Do SAEs Beat Random Baselines?" (arXiv:2602.14111)**
- **Severity**: exact_match on method, partial_overlap on metric focus
- **Overlap**: Both use identical baseline methodology (random decoder, shuffled features, permuted encoder) applied to SAEs. Korznikov tests on interpretability, sparse probing, and causal editing. The proposal applies the same baselines to absorption-specific metrics.
- **Critical overlap**: Korznikov's synthetic ground-truth evaluation shows SAEs recover only 9% of true features despite 71% explained variance — this directly informs the front-runner's experimental setup.
- **Differentiation remaining**: The proposal focuses specifically on **absorption rates** as the metric, whereas Korznikov uses different evaluation tasks. This is a meaningfully different research question (absorption is one specific failure mode, not the full picture of SAE reliability). However, the baseline methodology is **not novel** — it is directly copied from Korznikov.

**2. Chanin et al. (2024) — "A is for Absorption" (arXiv:2409.14507)**
- **Severity**: related_work
- **Overlap**: Defines the absorption phenomenon and introduces the ablation-based absorption metric. Already cited in the proposal.
- **Differentiation**: Chanin documents absorption; the proposal tests whether it is a genuine SAE artifact vs. statistical artifact.

**3. OrtSAE (Korznikov et al., 2025, arXiv:2509.22033)**
- **Severity**: partial_overlap
- **Overlap**: Reduces absorption by 65% via orthogonal constraints. Uses encoder-decoder geometry concepts.
- **Differentiation**: OrtSAE is a mitigation method; the proposal is a diagnostic framework.

**4. "SAEs Are Good for Steering" (Arad et al., 2025, arXiv:2505.20063)**
- **Severity**: partial_overlap with H3 (steering intervention)
- **Overlap**: Shows that steering with SAEs works well after filtering low-output-score features. Distinguishes input features vs. output features.
- **Differentiation**: The proposal tests whether steering absorbed features toward parent directions improves sensitivity — a specific causal question Arad et al. do not address.

**5. "Feature-Guided SAE Steering" (Bhargav & Zhu, 2025, arXiv:2511.00029)**
- **Severity**: related_work
- **Overlap**: Uses SAE steering for safety-utility tradeoff.
- **Differentiation**: Neither paper tests whether steering absorbed features is less effective than steering non-absorbed features.

#### Recommendation

**Proceed with significant repositioning.** The Korznikov (2026) paper substantially overlaps on methodology (random baselines). The proposal's contribution must be clearly reframed around:

1. **Absorption-specific metric focus**: Korznikov uses interpretability/sparse probing/causal editing; the proposal uses absorption rate as metric. This is a meaningful but narrow differentiation.
2. **Novel combined design**: The **combination** of random baseline comparison + causal steering intervention on absorbed features is novel. Korznikov does not test steering absorbed features.
3. **H3/H4 contributions**: The causal steering test (H3) and frequency-absorption correlation (H4) are not in Korznikov.

**Required revision**: The proposal must explicitly acknowledge Korznikov (2026) as having already established random baselines for SAEs, and clearly state what the proposal adds beyond their findings. The contribution should be: "We extend Korznikov's sanity-check framework to absorption-specific metrics and add causal validation of absorption's practical impact."

---

### cand_a1: Encoder-Decoder Asymmetry as Training-Free Absorption Proxy

**Novelty Score: 7/10** (novel_metric, minor overlap, proceed with caveats)

#### Collisions

**1. OrtSAE (Korznikov et al., 2025, arXiv:2509.22033)**
- **Severity**: minor_overlap
- **Overlap**: OrtSAE uses encoder-decoder cosine similarity as part of their orthogonality constraint during training. The proposal uses encoder-decoder norm ratio/cosine similarity as a **detection** metric post-hoc.
- **Critical distinction**: OrtSAE uses the geometry to **mitigate** absorption; cand_a1 uses the same geometry to **detect** absorption. These are different goals. OrtSAE does not claim or test that asymmetry correlates with absorption rate.
- **Remaining risk**: If OrtSAE's training procedure inherently produces asymmetric weights as a byproduct of the orthogonal constraint, the metric may not generalize to standard SAEs. Validation on non-OrtSAE SAEs is essential.

**2. "SAEs Are Good for Steering" (Arad et al., 2025)**
- **Severity**: minor_overlap
- **Overlap**: Introduces input vs. output feature scores as feature quality metrics. Different from asymmetry.

#### Recommendation

**Proceed.** The asymmetry metric for absorption detection is not directly claimed in prior work. Caveat: must be validated on non-OrtSAE SAEs to avoid circular reasoning.

---

### cand_a2: SAE Ensembles for Hierarchical Feature Recovery (MRSAE)

**Novelty Score: 5/10** (partial_overlap, needs repositioning)

#### Collisions

**1. Gadgil et al. (2025) — "Ensembling Sparse Autoencoders" (arXiv:2505.16077)**
- **Severity**: partial_overlap
- **Overlap**: Trains multiple SAEs with different initializations and ensembles them. Improves reconstruction, diversity, and downstream tasks.
- **Differentiation remaining**: The proposal focuses on **multi-resolution SAEs with different L0 targets** (not different initializations) to specifically recover absorbed parent features in child features at different sparsity levels. This is a different ensemble strategy and a different goal.
- **Critical overlap**: Gadgil already showed that ensemble diversity captures more features. The proposal's claim that high-L0 SAE captures absorbed child features that low-L0 SAE misses is plausible but must be experimentally validated — no prior work has shown this.

**2. SCALAR (Fillingham et al., 2025, arXiv:2511.07572)**
- **Severity**: minor_overlap
- **Overlap**: Uses weight-sharing across SAEs (staircase SAEs) to limit upstream feature duplication. Related ensemble idea but different mechanism.

**3. Muchane et al. (2025) — "Incorporating Hierarchical Semantics in SAE"**
- **Severity**: partial_overlap
- **Overlap**: Explicitly models hierarchical feature structure in SAEs. However, Muchane adds hierarchy to the architecture; cand_a2 detects hierarchy post-hoc via multi-resolution matching.

#### Recommendation

**Proceed with repositioning.** The multi-resolution L0 diversity idea for absorption recovery is mechanistically plausible and not directly claimed in prior work. Must differentiate clearly from Gadgil (different initialization ensemble) by emphasizing the L0 variation as the key differentiator, not just "ensemble." Consider renaming to emphasize L0 diversity.

---

### cand_a3: Safety-Critical Rare Features and Absorption

**Novelty Score: 9/10** (genuinely novel, proceed)

#### Collisions

**1. "Feature-Guided SAE Steering for Refusal-Rate Control" (Bhargav & Zhu, 2025, arXiv:2511.00029)**
- **Severity**: related_work
- **Overlap**: Uses SAE steering for safety. Does not address absorption.
- **Differentiation**: Neither Bhargav nor any prior work tests whether safety-critical features are disproportionately absorbed.

**2. "Detecting Strategic Deception Using Linear Probes" (Goldowsky-Dill et al., 2025)**
- **Severity**: related_work
- **Overlap**: Studies safety-critical deception features in LLMs. Does not involve SAEs or absorption.

**3. Makelov et al. (2024) — cited in proposal**
- **Severity**: related_work
- **Overlap**: Relevant to safety-critical features but not absorption.

#### Recommendation

**Proceed.** No prior work examines whether safety-critical features are systematically more absorbed. This is the most novel candidate. The contrarian framing (SAE-based safety analysis is unreliable precisely for the cases that matter most) is compelling and not claimed elsewhere.

---

## Overall Novelty Assessment

| Candidate | Score | Collision Count | Severity | Recommendation |
|-----------|-------|----------------|----------|----------------|
| cand_p1 | 6 | 5 | partial_overlap | proceed with repositioning |
| cand_a1 | 7 | 2 | minor_overlap | proceed |
| cand_a2 | 5 | 3 | partial_overlap | proceed with repositioning |
| cand_a3 | 9 | 2 | related_work | proceed |

**Overall novelty: medium**

**Primary risk**: Korznikov et al. (2026) preempts much of the front-runner's methodology. The proposal must pivot toward absorption-specific metrics and causal steering contribution to survive.

**Primary opportunity**: cand_a3 is genuinely novel and should be elevated if cand_p1 needs substantial revision.

---

## Key References

- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
- Korznikov et al. (2025). OrtSAE. arXiv:2509.22033
- Gadgil et al. (2025). Ensembling SAEs. arXiv:2505.16077
- Arad et al. (2025). SAEs Are Good for Steering. arXiv:2505.20063
- Bhargav & Zhu (2025). Feature-Guided SAE Steering. arXiv:2511.00029
- Fillingham et al. (2025). SCALAR. arXiv:2511.07572
- Muchane et al. (2025). Hierarchical SAE. arXiv:2506.01197
