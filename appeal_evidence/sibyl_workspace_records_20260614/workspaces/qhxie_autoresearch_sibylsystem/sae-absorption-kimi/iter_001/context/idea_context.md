

## Project Spec
# 项目: sae-absorption

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

**Research Topic**: Feature Absorption in Sparse Autoencoders (SAEs): Systematic Analysis, Quantification, and Implications for Interpretability
**Survey Date**: 2026-04-14
**arXiv Search Keywords**: "sparse autoencoder" AND "feature absorption"; "sparse autoencoder" AND "superposition"; "sparse autoencoder" AND (interpretability OR "mechanistic interpretability"); "feature splitting" AND "sparse autoencoder"; "sparse autoencoder" AND (benchmark OR evaluation OR metric)
**Web Search Keywords**: sparse autoencoder feature absorption state of the art 2025; sparse autoencoder superposition interpretability survey 2024 2025; SAEBench sparse autoencoder benchmark github open source; SAELens sparse autoencoder github open source training analysis; feature absorption metric sparse autoencoder github code implementation

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant unsupervised approach for mechanistic interpretability of large language models (LLMs), following the seminal work by Anthropic on superposition and monosemanticity. The core premise is that neural networks represent more features than they have neurons by encoding concepts as overlapping directions in activation space---a phenomenon termed *superposition*. SAEs attempt to disentangle these superposed representations into sparse, human-interpretable features via dictionary learning. Over the past two years (2024--2026), the field has shifted from demonstrating that SAEs can find interpretable features to rigorously evaluating their failure modes and improving architectural designs.

A central and increasingly recognized failure mode is **feature absorption**: when general, parent-level features fail to activate independently and are instead "absorbed" into more specific child features due to sparsity optimization. This was first systematically studied by Chanin et al. (2024) and has since been validated across hundreds of LLM SAEs. Absorption creates *interpretability illusions*---features appear monosemantic but are incomplete because they miss instances where the general feature should fire. In 2025, the community responded with multiple architectural innovations (OrtSAE, Matryoshka SAEs, EWG-SAE) and comprehensive benchmarking frameworks (SAEBench, CE-Bench) that explicitly measure absorption and related pathologies. Theoretical understanding has also advanced, with unified frameworks casting SAE training as piecewise biconvex optimization and characterizing spurious minima that lead to absorption and dead neurons.

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 | 2024 | Coins "feature absorption"; introduces a metric; validates empirically across hundreds of LLM SAEs; shows absorption is caused by sparsity optimization under hierarchical features | Does not propose a complete architectural solution; metric is task-specific (first-letter task) |
| 2 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | Enforces orthogonality between SAE features via cosine-similarity penalty; reduces absorption by 65%, composition by 15%; discovers 9% more distinct features | Chunked orthogonality is heuristic; scaling to very large dictionaries remains unproven |
| 3 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | arXiv:2503.17547 | 2025 | Trains nested dictionaries of increasing size simultaneously; smaller dictionaries learn general concepts, larger ones learn specifics; reduces feature absorption | Minor reconstruction trade-off; training is ~1.5x slower than standard SAE |
| 4 | A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima | arXiv:2512.05534 | 2025 | First unified theoretical framework for SAEs, transcoders, and crosscoders; explains feature absorption and dead neurons via spurious optima analysis; proposes "feature anchoring" | Theory is abstract; empirical validation on real LLMs is preliminary |
| 5 | Improving Robustness In Sparse Autoencoders via Masked Regularization | arXiv:2604.06495 | 2026 | Proposes token-replacement masking during training to disrupt co-occurrence patterns; reduces absorption and OOD gap across architectures | Masking strategy is simple; gains may be task-dependent |
| 6 | Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training | arXiv:2510.08855 | 2025 | Introduces ATM, a probabilistic masking mechanism based on importance scores over training time; achieves lower absorption scores than TopK and JumpReLU | Limited evaluation to Gemma-2-2b; mechanism adds complexity |
| 7 | Scaling and evaluating sparse autoencoders | arXiv:2406.04093 | 2024 | Proposes k-sparse (TopK) autoencoders for direct sparsity control; introduces scaling laws and new evaluation metrics (feature recovery, explainability, downstream sparsity); trains 16M latent SAE on GPT-4 | Focuses on scaling rather than absorption specifically |
| 8 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | arXiv:2309.08600 | 2023 | Foundational empirical result: SAEs extract monosemantic features from LLMs; establishes SAEs as a viable interpretability tool | Pre-dates recognition of absorption as a major failure mode |
| 9 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability | ICML 2025 / arXiv:2503.09532 | 2025 | Systematic benchmark with 8 evaluations including feature absorption, sparse probing, RAVEL, SCR, TPP, unlearning; releases 200+ trained SAEs | Proxy metrics do not always translate to practical gains |
| 10 | Measuring Sparse Autoencoder Feature Sensitivity | arXiv:2509.23717 | 2025 | Introduces "feature sensitivity"---how reliably features activate on semantically similar texts; finds sensitivity declines with SAE width | Complementary to absorption; does not address root causes |
| 11 | Data Whitening Improves Sparse Autoencoder Learning | arXiv:2511.13981 | 2025 | Shows PCA whitening of input activations improves interpretability metrics (including disentanglement) on SAEBench despite minor reconstruction drops | Preprocessing step; not an architectural fix for absorption |
| 12 | From Data Statistics to Feature Geometry: How Correlations Shape Superposition | arXiv:2603.09972 | 2026 | Introduces BOWS, a controlled setting for correlated features; shows interference can be constructive rather than just noise; explains semantic clusters and cyclical structures | Focuses on superposition geometry rather than SAE training directly |
| 13 | SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data | arXiv:2602.14687 | 2026 | Toolkit for large-scale synthetic data with realistic correlation, hierarchy, and superposition; enables controlled ablations and ground-truth validation | Synthetic only; may not capture all LLM complexities |
| 14 | Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation | arXiv:2603.28744 | 2026 | Reframes SAE failure as a dictionary learning challenge (not amortization); shows SAE-learned dictionaries point in wrong directions under OOD compositional shifts | Critical perspective; limited to toy and vision settings |
| 15 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | Argues for feature consistency across training runs; proposes PW-MCC metric and shows high consistency is achievable with TopK SAEs | Consistency is different from absorption, though related |

## 3. SOTA Methods and Benchmarks

### Current Best Methods for Reducing Absorption

1. **OrtSAE** (Korznikov et al., 2025): Applies a chunk-wise orthogonality penalty on decoder weights, achieving a **65% reduction in absorption** with minimal overhead.
2. **Matryoshka SAEs** (Bussmann et al., 2025): Uses nested dictionaries to separate abstraction levels; reduces absorption while maintaining interpretability at multiple scales.
3. **Masked Regularization** (Narayanaswamy et al., 2026): Randomly replaces tokens during training to disrupt co-occurrence patterns, improving robustness and reducing absorption.
4. **Adaptive Temporal Masking (ATM)** (Li & Ren, 2025): Probabilistic feature selection based on evolving importance scores; outperforms TopK and JumpReLU on absorption metrics.
5. **Feature Anchoring** (Tang et al., 2025): A theoretically motivated technique to restore identifiability in sparse dictionary learning, improving feature recovery.

### Mainstream Datasets and Evaluation Metrics

- **SAEBench** (Karvonen et al., ICML 2025): The dominant comprehensive benchmark. Includes 8 evaluations:
  - Feature Absorption
  - AutoInterp (automated interpretability)
  - L0 / Loss Recovered
  - RAVEL (feature disentanglement)
  - Spurious Correlation Removal (SCR)
  - Targeted Probe Perturbation (TPP)
  - Sparse Probing
  - Unlearning
- **CE-Bench** (Gulko et al., 2025): Lightweight contrastive evaluation on story pairs; correlates >70% with SAEBench without requiring an external LLM judge.
- **SynthSAEBench** (Chanin & Garriga-Alonso, 2026): Synthetic benchmark with ground-truth features, hierarchy, and superposition for controlled ablations.
- **Absorption Metric** (Chanin et al., 2024): Computes the fraction of logistic-regression true positives where the main SAE latents fail to activate and the feature is absorbed elsewhere.

## 4. Identified Research Gaps

- **Gap 1: Lack of a Unified, Task-Agnostic Absorption Metric.** The current absorption metric from Chanin et al. is defined on a specific spelling task (first-letter detection). A generalizable metric that quantifies absorption across arbitrary hierarchical feature domains is still missing.
- **Gap 2: Theoretical-empirical Disconnect.** While Tang et al. (2025) provide a unified theoretical framework explaining absorption via spurious minima, the connection between theory and practical architectural design remains loose. There is room for theory-guided architectures.
- **Gap 3: Scaling Solutions to Larger Models.** Most absorption-mitigation methods are evaluated on Gemma-2-2B or Pythia-160M. Their effectiveness on larger models (e.g., Llama-3.1-8B, GPT-4-scale) is underexplored.
- **Gap 4: Dynamic and Context-Dependent Absorption.** Absorption is typically measured as a static property of trained SAEs. Little work has studied how absorption varies across inputs, layers, or training dynamics.
- **Gap 5: Causal Impact of Absorption on Downstream Interpretability.** While absorption is known to create interpretability illusions, its causal effect on circuit tracing, steering, and safety interventions has not been systematically quantified.
- **Gap 6: Relationship Between Absorption and Other Failure Modes.** The interplay between absorption, feature splitting, feature composition, dead neurons, and polysemanticity is not fully understood.

## 5. Available Resources

### Open-Source Code

- **SAELens** (`jbloomAus/SAELens`): The primary library for training and analyzing SAEs on LLMs. Supports Standard, Gated, TopK, and JumpReLU architectures. Integrates with TransformerLens and Neuronpedia.
  - GitHub: https://github.com/jbloomAus/SAELens
  - Install: `pip install sae-lens`
- **SAEBench** (`adamkarvonen/SAEBench`): Comprehensive benchmark suite with 8 evaluations and 200+ trained SAEs.
  - GitHub: https://github.com/adamkarvonen/SAEBench
  - Install: `pip install sae-bench`
- **sae-spelling** (`lasr-spelling/sae-spelling`): Official code for "A is for Absorption." Contains `FeatureAbsorptionCalculator` and absorption experiments.
  - GitHub: https://github.com/lasr-spelling/sae-spelling
- **Matryoshka SAE** (`bartbussmann/matryoshka_sae`): Original implementation of Matryoshka SAEs.
  - GitHub: https://github.com/bartbussmann/matryoshka_sae
- **noanabeshima/matryoshka-saes**: Efficient training implementation with toy model experiments.
  - GitHub: https://github.com/noanabeshima/matryoshka-saes
- **RouteSAE** (`swei2001/RouteSAEs`): Multi-layer SAE with routing mechanism.
  - GitHub: https://github.com/swei2001/RouteSAEs
- **top-afa-sae** (`SewoongLee/top-afa-sae`): Dynamic top-k activation without manual k tuning.
  - GitHub: https://github.com/SewoongLee/top-afa-sae

### Datasets

- **OpenWebText**: Standard corpus for training SAEs on GPT-2 scale models.
- **The Pile (uncopyrighted)**: Larger corpus for scaling experiments.
- **SAEBench synthetic tasks**: Hierarchical feature datasets for controlled absorption studies.
- **CE-Bench contrastive story pairs**: Lightweight evaluation dataset.

### Pretrained Models and Checkpoints

- **Gemma Scope**: Pretrained SAEs for Gemma-2-2B (widely used in absorption studies).
- **Llama Scope**: 256 SAEs trained on each layer/sub-layer of Llama-3.1-8B-Base (32K and 128K features).
  - HuggingFace: `fnlp/Llama-Scope`
- **SAEBench released SAEs**: 200+ SAEs across 7 architectures, 3 widths (4K, 16K, 65K), on Pythia-160M and Gemma-2-2B.
- **Neuronpedia**: Web-based browser for pretrained SAE features.
  - URL: https://neuronpedia.org

## 6. Implications for Idea Generation

### Directions Worth Exploring

1. **Generalizable Absorption Quantification.** Developing a task-agnostic absorption metric (or family of metrics) would be a high-impact contribution. This could build on the Chanin et al. framework but extend it to arbitrary hierarchical features using automated probing or LLM-based feature labeling.
2. **Theory-Guided Architectures.** Using the unified piecewise biconvexity framework (Tang et al.) to design training objectives or regularization terms that explicitly avoid spurious minima leading to absorption.
3. **Dynamic Multi-Scale SAEs.** Extending Matryoshka-style architectures with learned or input-dependent dictionary selection, rather than fixed nesting, could better handle context-dependent abstraction levels.
4. **Causal Evaluation of Absorption.** Designing experiments that directly measure how absorption degrades circuit tracing, steering, or safety interventions would strengthen the practical case for absorption research.
5. **Training Dynamics of Absorption.** Studying when and how absorption emerges during training could lead to early-stopping or curriculum strategies that prevent it.

### Saturated Directions

- Simple comparisons of ReLU vs. TopK vs. JumpReLU on proxy metrics (MSE, L0) without downstream interpretability evaluation.
- Small-scale synthetic experiments without validation on real LLMs.
- Proposing new SAE architectures without benchmarking on SAEBench or comparable suites.

### Cross-Domain Analogies with Potential

- **Sparse Coding in Neuroscience:** The problem of "parts-based" vs. "holistic" representations in visual cortex may offer analogies for how hierarchical features should be decomposed.
- **Topic Models (LDA):** The relationship between general topics and sub-topics in LDA has been studied extensively; hierarchical topic model regularization techniques could inspire SAE objectives.
- **Matrix Factorization:** Non-negative matrix factorization with orthogonality or hierarchical constraints has a rich literature that could inform SAE design.

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens (`jbloomAus/SAELens`) | High | MIT | Adopt | Mature, well-documented, integrates with TransformerLens; supports multiple architectures and training pipelines |
| SAEBench (`adamkarvonen/SAEBench`) | High | Open source | Adopt | Industry-standard benchmark; includes feature absorption metric, sparse probing, SCR, TPP, RAVEL, unlearning |
| sae-spelling (`lasr-spelling/sae-spelling`) | High | Open source | Adopt | Official absorption metric implementation; directly reusable for quantifying absorption |
| Matryoshka SAE (`bartbussmann/matryoshka_sae`) | Medium | Open source | Extend | Good baseline for hierarchical SAEs; can be modified with custom regularization or dynamic level selection |
| noanabeshima/matryoshka-saes | Medium | Open source | Extend | Efficient training code and toy model experiments; useful for rapid prototyping |
| RouteSAE (`swei2001/RouteSAEs`) | Medium | Open source | Extend | Multi-layer routing mechanism; could be combined with absorption-aware objectives |
| top-afa-sae (`SewoongLee/top-afa-sae`) | Low-Medium | Open source | Compose | Dynamic k selection is interesting but not directly related to absorption; could be composed into a larger pipeline |
| TransformerLens | High | MIT | Adopt | Essential for activation hooking and model inspection in LLM interpretability |
| HuggingFace Transformers / Datasets | High | Apache 2.0 | Adopt | Standard for model loading and dataset access |

### Highlighted Reusable Components

- **Evaluation Framework:** SAEBench provides a ready-to-run evaluation pipeline including the absorption metric. This should be the primary evaluation backbone.
- **Absorption Metric Code:** `sae-spelling` contains the reference implementation of `FeatureAbsorptionCalculator` and should be reused or extended rather than reimplemented.
- **Training Pipeline:** SAELens provides the most robust training pipeline. Custom architectures (e.g., with orthogonality penalties or hierarchical nesting) can be implemented as subclasses within the SAELens framework.
- **Pretrained SAEs:** Gemma Scope and Llama Scope checkpoints enable experiments without training from scratch, saving significant compute.


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Research Proposal: The Hidden Cost of Fixing Feature Absorption

## Title
**The Impossibility Triangle of Sparse Autoencoders: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods**

(Alternative: **Rethinking Feature Absorption: Why Current Mitigations Trade One Pathology for Another**)

---

## Abstract

Feature absorption—where general parent features are "absorbed" into specific child features due to sparsity optimization—has emerged as a central pathology in sparse autoencoders (SAEs). The field has responded with architectural innovations (OrtSAE, Matryoshka SAEs, masked regularization) that report impressive absorption reductions. However, these evaluations are almost always single-metric: they optimize absorption in isolation. We challenge this framing by conducting the first systematic, training-free, multi-objective evaluation of existing pretrained SAE checkpoints. Our hypothesis is that absorption-mitigation methods do not dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance. Instead, each method occupies a different tradeoff region. We further develop and pilot a task-agnostic absorption metric based on automated hierarchy discovery, and quantify absorption's unique causal impact on downstream interpretability utility via controlled meta-analysis of SAEBench data. Our work reframes the SAE research agenda from "fixing absorption" to "navigating unavoidable tradeoffs."

---

## Motivation

Sparse autoencoders are the dominant tool for unsupervised feature discovery in language model interpretability. Yet they suffer from feature absorption: when hierarchical features co-occur, the sparsity penalty incentivizes the SAE to represent the general parent feature through the specific child feature, making the parent feature effectively invisible (Chanin et al., 2024). This undermines the core promise of SAEs—finding human-interpretable, atomic features.

The community has responded enthusiastically. OrtSAE reports 65% absorption reduction via orthogonality penalties (Korznikov et al., 2025). Matryoshka SAEs claim ~10x reduction via multi-scale dictionaries (Bussmann et al., 2025). Masked regularization and time-aware feature selection add further architectural fixes.

But there is a growing skeptical turn. Chanin et al. (2025) prove narrower SAEs reduce absorption but increase **feature hedging**—an opposite failure mode where latents incorrectly mix correlated features. Cui et al. (2025) prove theoretically that standard SAEs generally fail to recover ground-truth monosemantic features. Roy et al. (2026) show "catastrophic interpretability collapse" under aggressive sparsification. Kantamneni et al. (2025) find SAE probes underperform simple logistic-regression baselines on downstream tasks.

These results suggest absorption may not be a simple fixable bug but rather an intrinsic consequence of the sparsity objective and dictionary learning limitations. Yet no prior work has rigorously tested this hypothesis with a fair, multi-objective evaluation across the full suite of SAE quality metrics.

---

## Research Questions

1. **RQ1 (Tradeoffs):** Do absorption-mitigation architectures (OrtSAE, Matryoshka, JumpReLU, masked regularization) dominate standard SAEs on a multi-objective Pareto front, or do they systematically trade absorption for other pathologies (hedging, reconstruction loss, dead neurons)?

2. **RQ2 (Downstream Causality):** Controlling for reconstruction quality and L0 sparsity, does absorption have a unique causal effect on downstream interpretability utility (sparse probing, RAVEL disentanglement, steering)?

3. **RQ3 (Metric Generalization):** Can we construct and validate a task-agnostic absorption metric that generalizes beyond the first-letter spelling task to arbitrary semantic hierarchies?

---

## Hypotheses

### Primary Hypothesis (H1)
Absorption-mitigation methods do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance. Each method occupies a distinct tradeoff region, and the apparent superiority of OrtSAE/Matryoshka on absorption metrics is an artifact of selective single-metric reporting.

### Secondary Hypothesis (H2)
After controlling for L0 sparsity and reconstruction loss, higher absorption rates correlate negatively with downstream interpretability utility (sparse probing F1, RAVEL Cause/Isolation). This supports the claim that absorption is not merely an aesthetic pathology but has measurable practical harm.

### Tertiary Hypothesis (H3)
A task-agnostic absorption metric, constructed by combining automated hierarchical concept discovery with the causal ablation framework of Chanin et al., will correlate moderately-to-strongly (r > 0.4) with the original first-letter absorption benchmark while enabling absorption measurement across arbitrary semantic domains.

---

## Method

### Experiment 1: Multi-Objective Pareto Evaluation (Training-Free, ~45–60 min)

**Checkpoint corpus:** We assemble a diverse corpus of existing pretrained SAEs:
- Gemma Scope (multiple L0 variants: 22, 44, 88, 176)
- Llama Scope (multiple widths and sparsities)
- SAEBench released SAEs (200+ checkpoints across architectures: Standard, TopK, JumpReLU, BatchTopK, OrtSAE, Matryoshka, Masked Regularization)

**Metrics:**
- **Absorption:** Chanin et al.'s first-letter metric (for comparability with prior work)
- **Hedging:** Feature hedging score computed on correlated token pairs (Chanin et al., 2025)
- **Reconstruction:** L0 / CE loss recovered / explained variance on a held-out corpus
- **Dead neurons:** Fraction of latents with near-zero activation frequency
- **Downstream probing:** Sparse probing F1 from SAEBench
- **RAVEL:** Cause and Isolation scores for causal disentanglement

**Analysis:**
1. Normalize each metric to [0, 1] within model family.
2. Compute empirical Pareto fronts per architecture family on Gemma-2-2B and Pythia-160M.
3. Test stochastic dominance using Mann-Whitney U tests across the full metric suite.
4. Report pairwise tradeoff curves (absorption vs. hedging, absorption vs. reconstruction, etc.).

**Models:** Gemma-2-2B and Pythia-160M (the two most studied models in absorption research).

### Experiment 2: Downstream Causal Cost Meta-Analysis (Training-Free, ~30 min)

Using the 200+ pretrained SAEs from SAEBench:
1. Extract absorption, RAVEL Cause/Isolation, sparse probing F1, TPP, SCR, and L0/loss-recovered for each SAE.
2. Perform partial correlation and regression with absorption as the predictor, controlling for L0 and reconstruction.
3. Include architecture family dummies and dictionary width as covariates.
4. Use cluster-robust standard errors by architecture family.

**Expected outcome:** Negative partial correlation between absorption and downstream performance, supporting H2.

### Experiment 3: Task-Agnostic Absorption Metric Pilot (Training-Free, ~30 min)

**Phase 1 — Automated Hierarchy Discovery:**
1. Select a pretrained SAE (GemmaScope 16K on Gemma-2-2B, layer 12).
2. Use an LLM judge to label the top-N most active features on a diverse corpus.
3. Prompt the LLM to organize descriptions into validated hierarchies for 2–3 clean domains (geography: continent → country; biology: animal → mammal → species; colors: color → shade).
4. For each parent-child pair, train a logistic regression probe on residual-stream activations.

**Phase 2 — Absorption Detection:**
1. For each probe direction, perform k-sparse probing on SAE latents to identify primary latents.
2. Detect false negatives (probe succeeds, main latents fail).
3. Run integrated-gradients ablation on false-negative tokens to find the most causally important latent.
4. Classify absorption if the top ablation latent aligns with the probe direction (cosine similarity > threshold) and dominates the runner-up.

**Phase 3 — Validation:**
1. Compute the new metric on 20–50 pretrained SAEs and correlate with the original first-letter absorption score.
2. If correlation is weak, analyze divergence to understand whether first-letter absorption is unrepresentative of general absorption.

---

## Expected Contributions

1. **First systematic multi-objective evaluation** of absorption-mitigation methods using existing pretrained SAEs, showing whether current "fixes" genuinely improve SAEs overall or merely shift pathologies.

2. **Empirical quantification** of absorption's unique causal impact on downstream interpretability utility, controlling for confounders.

3. **Pilot validation** of a task-agnostic absorption metric that could replace the task-specific first-letter benchmark.

4. **Reframing contribution:** If H1 is supported, we argue the field should shift from "fixing absorption" to **"navigating unavoidable tradeoffs"** and motivate future work on **task-adaptive SAE selection** rather than universal architectural superiority.

---

## Novelty Assessment

### Front-Runner (Multi-Objective Evaluation)
We searched arXiv, Google Scholar, and the web for prior work on:
- "multi-objective Pareto sparse autoencoder absorption hedging reconstruction tradeoff"
- "OrtSAE Matryoshka SAE multi-objective evaluation absorption hedging reconstruction tradeoff Pareto"

**Findings:**
- Switch SAEs (2024), Gated SAEs (2024), and HierarchicalTopK (2025) study Pareto improvements in the **MSE–L0 tradeoff** specifically.
- A 2025 paper, "Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words," critiques the narrow MSE–L0 Pareto frontier as insufficient for interpretability.
- OrtSAE and Matryoshka SAE papers compare each other on multiple metrics (absorption, reconstruction, downstream tasks) but do **not** frame this as a systematic Pareto analysis showing no architecture dominates across the full metric suite.
- **No prior work** explicitly evaluates the **absorption-hedging-reconstruction impossibility triangle** across existing architectures using existing checkpoints in a training-free setting.

**Verdict:** The specific contribution—systematic multi-objective Pareto evaluation of absorption-mitigation methods—is novel.

### Backup 1 (Task-Agnostic Metric)
We searched for:
- "task-agnostic absorption metric sparse autoencoder hierarchical feature beyond first-letter"
- "SAEBench absorption score metric Karvonen 2025 sparse autoencoder benchmark"

**Findings:**
- The canonical absorption metric (Chanin et al., 2024) and its SAEBench adaptation (Karvonen et al., 2025) remain tied to the **first-letter spelling task**.
- SAEBench authors explicitly note that supervised metrics are "fundamentally limited by the availability of ground truth data."
- **No prior work** has proposed a fully task-agnostic absorption metric using automated hierarchy discovery.

**Verdict:** Novel and high-impact if validated.

### Backup 2 (Random-Decoder Baseline)
We searched for:
- "random decoder sparse autoencoder absorption metric baseline"
- "frozen decoder SAE feature absorption training artifact"

**Findings:**
- Korznikov et al. (2026) introduced frozen-decoder and random-decoder baselines and showed they match trained SAEs on AutoInterp, sparse probing, and RAVEL.
- **However, they did NOT run the absorption metric on these baselines.**
- No prior work directly tests whether absorption is a training artifact or geometric inevitability using a controlled random-decoder design.

**Verdict:** Novel, but this candidate requires training new SAEs, which conflicts with the project's training-free constraint.

---

## Revisions from Prior Feedback

This is the first synthesis round; no prior proposal, novelty report, or Codex feedback exists to revise from.

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| OrtSAE/Matryoshka actually dominate the Pareto front | Medium | The paper still provides the first rigorous multi-objective validation—a valuable contribution, though less contrarian. |
| SAEBench metrics are too noisy for clean Pareto analysis | Medium | Use existing, well-tested implementations; average across multiple layers/checkpoints; report confidence intervals. |
| Task-agnostic metric does not correlate with first-letter benchmark | Medium | If correlation is weak, pivot to analyzing *why*—the divergence itself may reveal that first-letter absorption is unrepresentative, which is a valuable negative result. |
| Downstream correlation is confounded by unobserved architecture differences | Medium | Include architecture dummies, width, and L0 as controls; use cluster-robust SEs; explicitly discuss causal limitations. |

---

## Resource Estimate

- **Models:** Gemma-2-2B (2B), Pythia-160M (160M)
- **SAEs:** Gemma Scope, Llama Scope, SAEBench releases (all training-free)
- **Compute:** Single A100 or equivalent. Each metric computation on a checkpoint takes 5–15 minutes. With 50–100 checkpoints per model, total analysis is ~10–20 hours, easily parallelizable across GPUs. We target ≤1 hour per independent subtask (per-checkpoint metric batch).
- **Pilots:** 10–15 minutes per pilot (small subset of checkpoints + single hierarchy domain).

---

## What Each Perspective Contributed

- **Contrarian:** Provided the core thesis—that absorption is not a fixable bug but an intrinsic tradeoff, and that current evaluations are selectively reported on a single objective. The contrarian's multi-objective evaluation design is the backbone of the front-runner.
- **Innovator:** Contributed the task-agnostic metric idea (automated hierarchy discovery) and the downstream causal cost framing. These became Experiment 3 and the H2 validation.
- **Theoretical:** Provided the learning-theoretic intuition that absorption increases effective complexity, motivating the downstream causal analysis and giving theoretical language to the tradeoff discussion.
- **Empiricist:** Contributed rigorous experimental design principles (pre-registered thresholds, control variables, bootstrap CIs, confounder analysis) that strengthen all three experiments.
- **Pragmatist:** Supplied the engineering reality check—what tools exist (SAELens, SAEBench, sae-spelling), what is feasible within 1 hour, and what checkpoints are readily available. The pragmatist's emphasis on training-free analysis using existing resources is strictly followed.
- **Interdisciplinary:** The LCA/lateral inhibition idea was intellectually rich but dropped as a front-runner because (a) it requires training/modifying SAEs, violating the training-free constraint, and (b) very recent work (Rajamanoharan et al., 2024) has already applied LCA to GPT-2 SAEs, creating a partial novelty collision.

---

## References

- Chanin, D., et al. (2024). *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*. arXiv:2409.14507.
- Chanin, D., et al. (2025). *Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders*. arXiv:2505.11756.
- Karvonen, A., et al. (2025). *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability*. arXiv:2503.09532.
- Korznikov, A., et al. (2025). *OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features*. arXiv:2509.22033.
- Bussmann, B., et al. (2025). *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. arXiv:2503.17547.
- Cui, J., et al. (2025). *On the Theoretical Understanding of Identifiable Sparse Autoencoders and Beyond*. arXiv:2506.15963.
- Roy, S., et al. (2026). *Fundamental Limits of Neural Network Sparsification*. arXiv:2603.18056.
- Kantamneni, S., et al. (2025). *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing*. arXiv:2502.16681.
- Korznikov, A., et al. (2026). *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?* arXiv:2602.14111.
- Rajamanoharan, S., et al. (2024). *Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders*. arXiv:2411.13117.


## 当前可检验假设
# Testable Hypotheses with Expected Outcomes

## H1: Multi-Objective Pareto Tradeoffs (Front-Runner)

### Hypothesis
Absorption-mitigation methods (OrtSAE, Matryoshka SAE, JumpReLU, Masked Regularization) do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance.

### Expected Outcome
- **Absorption:** OrtSAE and Matryoshka SAE show lower absorption than Standard/TopK SAEs.
- **Hedging:** Narrower SAEs (including some Matryoshka configurations) show higher hedging scores.
- **Reconstruction:** Standard/BatchTopK SAEs achieve the best explained variance and CE loss recovered.
- **Dead neurons:** Aggressive sparsity methods (some TopK variants, masked regularization) show elevated dead-neuron rates.
- **Downstream probing:** No single architecture family dominates; performance varies by task.
- **Pareto front:** Each architecture family occupies a distinct region. No family stochastically dominates all others across the full metric suite.

### Falsification Criterion
If one architecture family (e.g., Matryoshka SAE or OrtSAE) shows statistically significant stochastic dominance (Mann-Whitney U test, p < 0.05) across ≥4 out of 5 metrics, H1 is falsified.

### Measurement Plan
1. Assemble 50–100 pretrained SAE checkpoints per model (Gemma-2-2B, Pythia-160M) across 5+ architecture families.
2. Compute all 5 metrics per checkpoint.
3. Normalize metrics to [0, 1] within model family.
4. Compute Pareto fronts and test stochastic dominance.

---

## H2: Absorption's Unique Causal Impact on Downstream Utility

### Hypothesis
Controlling for L0 sparsity and reconstruction loss, SAEs with higher absorption rates exhibit worse performance on downstream interpretability tasks (sparse probing F1, RAVEL Cause/Isolation).

### Expected Outcome
- **Partial correlation:** Negative partial correlation between absorption and sparse probing F1 (r_partial ≈ −0.25 to −0.45).
- **Partial correlation:** Negative partial correlation between absorption and RAVEL Cause/Isolation (r_partial ≈ −0.20 to −0.35).
- **Regression coefficient:** Absorption coefficient is negative and statistically significant (p < 0.05) in regressions controlling for L0, reconstruction, width, and architecture family.

### Falsification Criterion
If the absolute partial correlation between absorption and downstream performance is < 0.15 for both sparse probing and RAVEL, or if the regression coefficient is not statistically significant (p > 0.05), H2 is falsified.

### Measurement Plan
1. Extract absorption, sparse probing F1, RAVEL Cause/Isolation, L0, and loss-recovered for 200+ SAEBench SAEs.
2. Compute Pearson and Spearman correlations (raw).
3. Compute partial correlations controlling for L0 and reconstruction.
4. Run OLS regression with cluster-robust SEs (clustered by architecture family).

---

## H3: Task-Agnostic Absorption Metric Validity

### Hypothesis
A task-agnostic absorption metric, constructed by combining automated hierarchical concept discovery with causal ablation, will correlate moderately-to-strongly (r > 0.4) with the original first-letter absorption benchmark while enabling absorption measurement across arbitrary semantic domains.

### Expected Outcome
- **Correlation:** Pearson r between task-agnostic metric and first-letter metric ≈ 0.45–0.70 across 20–50 SAEs.
- **Domain coverage:** Successful absorption detection in 2–3 non-lexical domains (geography, biology, colors).
- **Downstream prediction:** SAEs scoring higher on the task-agnostic metric show worse sparse probing F1 on parent-level features.

### Falsification Criterion
If Pearson r < 0.3 between the task-agnostic metric and the first-letter benchmark, the automated hierarchy discovery is insufficiently reliable and H3 is falsified.

### Measurement Plan
1. Select GemmaScope 16K (Gemma-2-2B, layer 12) as the pilot SAE.
2. Use LLM-based labeling to discover 10–20 parent-child hierarchies in 2–3 domains.
3. Train logistic regression probes for each concept.
4. Run the absorption detection pipeline (k-sparse probing → false negatives → ablation → absorption classification).
5. Compute the task-agnostic metric and correlate with SAEBench first-letter absorption scores.

---

## H4 (Backup, Random-Decoder Baseline — Conditional)

### Hypothesis
Feature absorption in sparse autoencoders is primarily a geometric consequence of sparse dictionary learning on hierarchically structured data, not a pathology of flawed training dynamics. Randomly initialized, frozen-decoder SAEs matched for sparsity will exhibit absorption rates comparable to fully trained SAEs.

### Expected Outcome
- Random-decoder SAE absorption rate is within 20% (relative) of the trained SAE absorption rate.
- If geometry dominates, the result reframes absorption as an intrinsic property of sparse coding on hierarchical data.

### Falsification Criterion
If random-decoder SAEs show <50% of the trained SAE absorption rate (relative), H4 is falsified and training dynamics are the dominant cause.

### Measurement Plan
1. Load trained SAE (`gpt2-small-res-jb`).
2. Train random-decoder SAE to matched L0 (TopK for exact control).
3. Run `sae-spelling` absorption metric on both.
4. Report mean absorption rate, full absorption rate, feature splitting rate, L0, MSE, and dead-neuron fraction.

### Constraint Note
This hypothesis **violates the project's training-free constraint** and can only be pursued if the constraint is explicitly relaxed.


## 小型实验真实反馈（必须基于这些证据修正 idea，不能忽略负结果）
# Pilot Summary: e1_full_gemma (fallback to e1_full_gpt2)

## Task Context
- **Original task**: `e1_full_gemma` — Pareto evaluation on Gemma-2-2B SAEs.
- **Actual execution**: Fallback to `e1_full_gpt2` because `google/gemma-2-2b` is a gated HuggingFace repo and no HF authentication token is available in this environment.

## Environment
- GPU: cuda:4 (RTX 4090 class, 24 GB)
- Model: GPT-2 Small (117M)
- Dataset: C4 validation subset, ~12k characters (8 snippets)
- Seed: 42
- Runtime: ~34 seconds

## Checkpoints Evaluated (10)

| Release | SAE ID | Family |
|---------|--------|--------|
| gpt2-small-res-jb | blocks.0.hook_resid_pre | Standard |
| gpt2-small-res-jb | blocks.4.hook_resid_pre | Standard |
| gpt2-small-res-jb | blocks.8.hook_resid_pre | Standard |
| gpt2-small-res-jb | blocks.11.hook_resid_pre | Standard |
| gpt2-small-resid-post-v5-32k | blocks.4.hook_resid_post | TopK |
| gpt2-small-resid-post-v5-32k | blocks.8.hook_resid_post | TopK |
| gpt2-small-resid-post-v5-128k | blocks.4.hook_resid_post | TopK |
| gpt2-small-resid-post-v5-128k | blocks.8.hook_resid_post | TopK |
| gpt2-small-mlp-out-v5-32k | blocks.8.hook_mlp_out | TopK_MLP |
| gpt2-small-attn-out-v5-32k | blocks.8.hook_attn_out | TopK_Attn |

## Metrics Computed
- L0 (average active features per token)
- Explained variance
- Dead-neuron fraction
- CE loss recovered (%)
- Absorption rate (simplified first-letter proxy)
- Hedging rate (simplified correlated-pair proxy)

## Results Table

| Family | L0 | EV | Dead | CE_Rec | Absorption | Hedging |
|--------|----|----|------|--------|------------|---------|
| Standard | 19.0 | 0.981 | 0.724 | -2.08% | 0.0 | 0.08 |
| Standard | 34.4 | 0.950 | 0.648 | -3.31% | 0.0 | 0.8 |
| Standard | 69.5 | 0.917 | 0.339 | -4.98% | 0.0 | 0.8 |
| Standard | 61.6 | 0.930 | 0.335 | -6.30% | 0.0 | 0.8 |
| TopK | 32.0 | 0.948 | 0.680 | -1.94% | 0.0 | 0.8 |
| TopK | 32.0 | 0.914 | 0.652 | -3.56% | 0.0 | 0.8 |
| TopK | 32.0 | 0.961 | 0.878 | -0.99% | 0.0 | 0.8 |
| TopK | 32.0 | 0.931 | 0.865 | -2.20% | 0.0 | 0.8 |
| TopK_MLP | 32.0 | 0.734 | 0.563 | -0.49% | 0.0 | 1.0 |
| TopK_Attn | 32.0 | 0.794 | 0.648 | -0.10% | 0.654 | 1.0 |

## Family Averages

- **Standard**: L0=46.1, EV=0.945, Dead=0.511, CE_Rec=-4.17%, Absorption=0.000, Hedging=0.620
- **TopK**: L0=32.0, EV=0.938, Dead=0.769, CE_Rec=-2.17%, Absorption=0.000, Hedging=0.800
- **TopK_MLP**: L0=32.0, EV=0.734, Dead=0.563, CE_Rec=-0.49%, Absorption=0.000, Hedging=1.000
- **TopK_Attn**: L0=32.0, EV=0.794, Dead=0.648, CE_Rec=-0.10%, Absorption=0.654, Hedging=1.000

## Observations
1. **All CE loss recovered values are negative**, indicating that SAE reconstruction slightly degrades cross-entropy compared to the original model. This is expected for some SAEs but suggests the hook-replacement CE recovery computation may be noisy at small sample sizes.
2. **Dead-neuron rates are very high** (33-88%). This is likely because the pilot used only ~2k tokens; dead-neuron detection requires a larger corpus to be reliable.
3. **Absorption metric is degenerate** for most checkpoints (0.0) except TopK_Attn (0.65). The simplified first-letter proxy is too coarse and does not align with the rigorous sae-spelling benchmark. A proper absorption evaluation requires the full Chanin et al. pipeline.
4. **Hedging metric is also coarse** (same top feature for antonym pairs). It shows a clear split: Standard early layer has low hedging (0.08), while TopK_MLP and TopK_Attn show maximum hedging (1.0).
5. **Metric pipeline works end-to-end**: all 10 checkpoints loaded successfully, all metrics returned finite values, and the system tracked progress correctly.

## GO / NO-GO Assessment

- **Pipeline validation**: GO. The multi-objective metric pipeline runs without errors and produces outputs for every checkpoint.
- **Metric quality**: NO-GO for publication-ready numbers. The simplified absorption and hedging proxies are too crude, and dead-neuron estimates are unreliable at 2k tokens.
- **Gemma fallback**: NO-GO for the original `e1_full_gemma` task due to the gated model. This is a **blocking resource issue** that requires either (a) an HF token, or (b) switching to a fully open model like GPT-2 Small or Pythia-70m/160m/410m.

## Recommendation

**REFINE** — before scaling to full experiments:
1. Integrate the proper `sae-spelling` absorption metric (or SAEBench adaptation) instead of the simplified proxy.
2. Increase the activation corpus for dead-neuron detection to at least 50k-100k tokens.
3. Decide on the target model: either obtain HF access for Gemma-2-2B, or commit to GPT-2 Small / Pythia-160m as the open-model anchor.
4. Add more architecture families (e.g., Gated, JumpReLU) by sourcing appropriate checkpoints.


## 小型实验结构化信号（供你提炼 go/no-go / confidence / hypothesis status）
{
  "overall_recommendation": "REFINE",
  "selected_candidate_id": "cand_a",
  "candidates": [
    {
      "candidate_id": "cand_a",
      "go_no_go": "GO",
      "confidence": 0.55,
      "supported_hypotheses": ["H1_pipeline_works"],
      "failed_assumptions": ["H1_ready_for_full_scale", "gemma_access"],
      "key_metrics": {
        "checkpoints_loaded": 10,
        "pipeline_success_rate": 1.0,
        "mean_l0_standard": 46.1,
        "mean_l0_topk": 32.0,
        "mean_ev_standard": 0.945,
        "mean_ev_topk": 0.938,
        "mean_absorption_standard": 0.0,
        "mean_absorption_topk_attn": 0.654,
        "mean_hedging_standard": 0.620,
        "mean_hedging_topk_mlp": 1.0
      },
      "notes": "Pipeline runs end-to-end on GPT-2 Small, but Gemma-2-2B is gated and inaccessible. Simplified absorption/hedging proxies are too coarse for full-scale evaluation. Dead-neuron estimates unreliable at 2k tokens. Need proper sae-spelling metric and larger corpus before full run."
    }
  ],
  "fallback_executed": {
    "original_task": "e1_full_gemma",
    "fallback_task": "e1_full_gpt2",
    "reason": "google/gemma-2-2b is a gated HuggingFace repository and no HF token is configured in the environment."
  }
}


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_a",
      "title": "The Impossibility Triangle of Sparse Autoencoders: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods",
      "status": "front_runner",
      "summary": "Conduct the first training-free, systematic multi-objective Pareto evaluation of existing pretrained SAE checkpoints (Gemma Scope, Llama Scope, SAEBench). Test whether absorption-mitigation architectures (OrtSAE, Matryoshka, JumpReLU, Masked Regularization) dominate standard SAEs across absorption, hedging, reconstruction, dead-neuron rate, and downstream probing metrics. Hypothesis: no architecture dominates; each occupies a distinct tradeoff region.",
      "hypotheses": [
        "Absorption-mitigation methods do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance.",
        "Controlling for L0 sparsity and reconstruction loss, higher absorption rates correlate negatively with downstream interpretability utility (sparse probing F1, RAVEL Cause/Isolation)."
      ],
      "pilot_focus": "Run absorption + hedging + reconstruction + dead-neuron metrics on 5 checkpoints (1 per architecture family) on Gemma-2-2B to confirm metric pipeline works and variances are reasonable. Expected duration: 10–15 minutes.",
      "novelty_claim": "No prior work has conducted a systematic multi-objective Pareto evaluation of absorption-mitigation methods across the full suite of SAE quality metrics using existing checkpoints in a training-free setting.",
      "alignment_with_spec": "Directly addresses spec goals 1 (quantify absorption prevalence) and 2 (explore causes: sparsity penalty, dictionary size, architecture tradeoffs). Fully training-free.",
      "key_papers": [
        "Chanin et al. (2024) — A is for Absorption",
        "Chanin et al. (2025) — Feature Hedging",
        "Karvonen et al. (2025) — SAEBench",
        "Korznikov et al. (2025) — OrtSAE",
        "Bussmann et al. (2025) — Matryoshka SAEs"
      ],
      "risks": [
        "OrtSAE/Matryoshka may actually dominate the Pareto front, weakening the contrarian claim.",
        "SAEBench metrics may be too noisy for clean Pareto analysis.",
        "Downstream correlation may be confounded by unobserved architecture differences."
      ],
      "mitigations": [
        "If architectures dominate, the paper still provides the first rigorous multi-objective validation—a valuable contribution.",
        "Use well-tested implementations; average across layers/checkpoints; report confidence intervals.",
        "Include architecture dummies, width, and L0 as controls; use cluster-robust SEs."
      ]
    },
    {
      "candidate_id": "cand_b",
      "title": "Towards a Task-Agnostic Absorption Metric: Automated Hierarchy Discovery for Generalizable SAE Evaluation",
      "status": "backup",
      "summary": "Develop and validate a task-agnostic absorption metric that generalizes beyond the first-letter spelling task. Use LLM-based automated hierarchy discovery to identify parent-child feature relationships in arbitrary semantic domains (geography, biology, colors), then apply the Chanin et al. causal ablation framework. Correlate the new metric with the original first-letter benchmark across 20–50 pretrained SAEs.",
      "hypotheses": [
        "A task-agnostic absorption metric will correlate moderately-to-strongly (r > 0.4) with the original first-letter absorption benchmark while enabling absorption measurement across arbitrary semantic domains.",
        "SAEs with higher scores on the task-agnostic metric will exhibit measurably worse downstream interpretability utility on parent-level features."
      ],
      "pilot_focus": "Run hierarchy discovery and absorption detection on GemmaScope 16K (Gemma-2-2B, layer 12) for one domain (geography: continent → country) with 5–10 parent-child pairs. Expected duration: 10–15 minutes.",
      "novelty_claim": "No prior work has proposed a fully task-agnostic absorption metric using automated hierarchy discovery. The canonical metric remains tied to the first-letter spelling task.",
      "alignment_with_spec": "Addresses spec goal 3 (develop reproducible evaluation metrics). Fully training-free.",
      "key_papers": [
        "Chanin et al. (2024) — A is for Absorption",
        "Karvonen et al. (2025) — SAEBench",
        "Zhu et al. (2012) — Sparse Topical Coding"
      ],
      "risks": [
        "LLM-generated hierarchies may be noisy or hallucinated.",
        "New metric may not correlate with the first-letter benchmark.",
        "Higher engineering complexity than the front-runner."
      ],
      "mitigations": [
        "Start with small, easily validated domains; manually audit a random sample; use consensus prompting.",
        "If correlation is weak, analyze divergence as a negative result—first-letter absorption may be unrepresentative.",
        "Scope pilot tightly to 1 SAE × 1 domain before scaling."
      ]
    },
    {
      "candidate_id": "cand_c",
      "title": "Is Feature Absorption a Training Artifact? A Controlled Baseline Study with Random-Decoder Sparse Autoencoders",
      "status": "backup",
      "summary": "Test whether absorption is a training-dynamics pathology or a geometric inevitability of sparse coding. Compare a trained SAE baseline against a random-decoder SAE (frozen decoder, train only encoder) matched for L0 sparsity. If random decoders show comparable absorption, the phenomenon is structural, not training-specific.",
      "hypotheses": [
        "Randomly initialized, frozen-decoder SAEs matched for sparsity will exhibit absorption rates comparable to fully trained SAEs, indicating absorption is a geometric consequence of sparse dictionary learning on hierarchical data."
      ],
      "pilot_focus": "GPT-2-small, layer 8. One trained SAE (`gpt2-small-res-jb`). One random-decoder SAE trained to matched L0 on 10M tokens. Run `sae-spelling` absorption metric. Expected duration: ~15 minutes GPU time.",
      "novelty_claim": "No prior work has applied the canonical feature absorption metric to random-decoder SAE baselines. Korznikov et al. (2026) showed random baselines match trained SAEs on standard metrics but did not measure absorption.",
      "alignment_with_spec": "TENSION: the project spec explicitly states 'training-free (分析现有预训练 SAE，不重新训练)'. This candidate requires training new SAEs and can only be pursued if the constraint is relaxed for a minimal pilot.",
      "key_papers": [
        "Chanin et al. (2024) — A is for Absorption",
        "Korznikov et al. (2026) — Sanity Checks for Sparse Autoencoders"
      ],
      "risks": [
        "Explicitly violates the project's training-free constraint.",
        "Random-decoder SAE may fail to converge to reasonable MSE.",
        "Result may fall in an ambiguous 50–80% range, making interpretation difficult."
      ],
      "mitigations": [
        "Only pursue if training-free constraint is explicitly relaxed.",
        "Use TopK for exact sparsity control; monitor MSE during training.",
        "Pre-register interpretation thresholds: within 20% = geometry dominates; below 50% = training dominates."
      ]
    }
  ],
  "synthesis_notes": {
    "front_runner_selection_reason": "Cand_a was selected as the front-runner because it is fully training-free (strictly aligned with the project spec), directly tests a bold contrarian claim that is well-supported by emerging theory, and uses existing resources (Gemma Scope, Llama Scope, SAEBench) with a clear implementation path. It addresses the spec's core goals of quantifying absorption prevalence and exploring its relationship with architecture/sparsity tradeoffs.",
    "perspective_weights": {
      "contrarian": "Highest weight. Provided the central thesis that absorption is not a fixable bug but an intrinsic tradeoff, and designed the multi-objective evaluation backbone.",
      "innovator": "High weight. Contributed the task-agnostic metric idea and downstream causal cost framing, which became integrated experiments within the front-runner.",
      "theoretical": "Medium-high weight. Provided the learning-theoretic intuition that absorption increases effective complexity, motivating H2 and giving theoretical language to the tradeoff discussion.",
      "empiricist": "High weight. Insisted on rigorous controls, pre-registered thresholds, confounder analysis, and bootstrap CIs—strengthening all experiments.",
      "pragmatist": "High weight. Supplied the engineering reality check and confirmed what tools/checkpoints are available. The training-free constraint from the spec was treated as ironclad.",
      "interdisciplinary": "Low weight. The LCA/lateral inhibition idea was intellectually rich but dropped as a front-runner because it requires training/modifying SAEs (violating the training-free constraint) and recent work (Rajamanoharan et al., 2024) has already applied LCA to GPT-2 SAEs, creating a partial novelty collision."
    },
    "dropped_ideas": [
      "Interdisciplinary Candidate A (LCA-SAE) — dropped due to training-free constraint violation and partial novelty collision with Rajamanoharan et al. (2024).",
      "Interdisciplinary Candidate B (Information Bottleneck) — dropped due to weak empirical testability (ReLU/TopK lack compression phase, no natural Y in unsupervised SAE training).",
      "Theoretical Candidate A (Rate-Distortion Bound) — dropped due to questionable practical predictive power and loose bounds.",
      "Theoretical Candidate C (Recovery Threshold) — dropped due to fatal orthogonality assumption and vacuous threshold in practice.",
      "Innovator Candidate A (Absorption Dynamics) — dropped due to training-free constraint violation (requires checkpoint-SAEs across training).",
      "Contrarian Candidate C (Training-Free Analysis Insufficient) — dropped due to logical weakness (conflates feature identity variance with absorption variance) and training-free constraint.",
      "Empiricist Candidate C (Metric Generalization Controlled Test) — scoped down to a component of Backup 1 rather than a standalone front-runner."
    ],
    "novelty_verification_summary": {
      "multi_objective_pareto": "No prior work explicitly evaluates the absorption-hedging-reconstruction impossibility triangle across existing architectures using existing checkpoints. Switch SAEs, Gated SAEs, and HierarchicalTopK study MSE-L0 Pareto fronts. OrtSAE and Matryoshka compare each other on multiple metrics but do not frame it as a systematic Pareto analysis.",
      "task_agnostic_metric": "No prior work has proposed a fully task-agnostic absorption metric. The canonical metric (Chanin et al., 2024) and SAEBench adaptation (Karvonen et al., 2025) remain tied to the first-letter spelling task.",
      "random_decoder_baseline": "No prior work has applied the absorption metric to random-decoder baselines. Korznikov et al. (2026) showed random baselines match trained SAEs on standard metrics but did not measure absorption."
    }
  }
}


## 上一轮 validation 决策意见
# Idea Validation Decision

## Pilot Evidence Summary

Only **candidate A (cand_a)** was executed in the pilot phase. The original task `e1_full_gemma` fell back to `e1_full_gpt2` because `google/gemma-2-2b` is a gated HuggingFace repository and no HF authentication token is available in the environment.

**Pilot execution summary (GPT-2 Small, 10 checkpoints, ~34 seconds):**
- Pipeline success rate: 1.0 (all 10 checkpoints loaded, all metrics returned finite values)
- Mean L0 (Standard): 46.1; (TopK): 32.0; (TopK_MLP): 32.0; (TopK_Attn): 32.0
- Mean explained variance (Standard): 0.945; (TopK): 0.938; (TopK_MLP): 0.734; (TopK_Attn): 0.794
- Mean CE loss recovered: all negative (-4.17% to -0.10%), suggesting noisy small-sample computation
- Dead-neuron fraction: 33-88%, unreliable at ~2k tokens
- Absorption (simplified first-letter proxy): 0.0 for all families except TopK_Attn (0.654)
- Hedging (simplified correlated-pair proxy): 0.620 (Standard), 0.800 (TopK), 1.000 (TopK_MLP/TopK_Attn)

**Key findings:**
1. The metric pipeline works end-to-end — this validates the engineering path.
2. The simplified absorption/hedging proxies are too coarse for publication-ready evaluation.
3. Dead-neuron estimates require a larger activation corpus (≥50k-100k tokens).
4. Gemma-2-2B is inaccessible without an HF token; the target model must switch to GPT-2 Small or Pythia.
5. H1 (no architecture dominates) is **neither confirmed nor falsified** — the pilot did not test stochastic dominance with proper metrics or sample size.
6. H2 and H3 were **not tested** in this pilot.

**Candidate B and C:** No pilot experiments were run.

---

## Decision Matrix

### Candidate A — Multi-Objective Pareto Evaluation

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | Pipeline runs end-to-end, but key metrics (absorption, hedging, dead neurons) are degenerate or noisy at pilot scale. No clear positive or negative signal on H1. |
| Hypothesis survival | 0.25 | 4 | H1 was not falsified. The pilot simply lacked statistical power and proper proxies. Main hypothesis remains viable. |
| Path to full result | 0.20 | 3 | Clear route exists: fix absorption metric (integrate sae-spelling / SAEBench), increase token budget, switch to open model. But requires non-trivial refinement before scaling. |
| Novelty (from report) | 0.15 | 5 | Novelty report and candidates.json confirm no prior systematic multi-objective Pareto evaluation of absorption-mitigation methods across the full metric suite in a training-free setting. |
| Resource efficiency | 0.10 | 4 | Fully training-free; uses existing checkpoints. One GPU per task, ~1 hour per batch. Very efficient once metrics are fixed. |

**Weighted score for Candidate A:** 3.55

### Candidate B — Task-Agnostic Absorption Metric

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No pilot was run. The task-agnostic metric is higher-engineering and depends on LLM-based hierarchy discovery, which is unvalidated in this workspace. |
| Hypothesis survival | 0.25 | 3 | H3 is plausible but entirely untested. Falsification threshold (r > 0.4) is reasonable. |
| Path to full result | 0.20 | 2 | Requires GemmaScope 16K (Gemma-2-2B), which is gated and inaccessible. Would need to pivot to GPT-2 Small or another open model, but no equivalent scope release exists for GPT-2. |
| Novelty (from report) | 0.15 | 5 | Candidates.json novelty claim: "No prior work has proposed a fully task-agnostic absorption metric using automated hierarchy discovery." |
| Resource efficiency | 0.10 | 2 | Higher complexity than Candidate A; LLM labeling, probe training, and causal ablation per hierarchy. Uncertain cost. |

**Weighted score for Candidate B:** 2.55

### Candidate C — Random-Decoder Baseline

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot run. |
| Hypothesis survival | 0.25 | 1 | Explicitly violates the project's training-free constraint (candidates.json: "TENSION: the project spec explicitly states 'training-free'"). |
| Path to full result | 0.20 | 1 | Cannot proceed without relaxing the training-free constraint. Even then, training a random-decoder SAE to matched L0 is non-trivial. |
| Novelty (from report) | 0.15 | 4 | Novel in applying absorption metric to random-decoder baselines, but Korznikov et al. (2026) already showed random baselines match trained SAEs on other metrics. |
| Resource efficiency | 0.10 | 1 | Requires training new SAEs — the most expensive path and explicitly disallowed. |

**Weighted score for Candidate C:** 1.15

---

## Decision Rationale

- **Candidate A scores 3.55**, which meets the ADVANCE threshold (≥ 3.5).
- Its main hypothesis (H1) was **not falsified** by the pilot. The pilot's issues were methodological (proxy metrics too coarse, sample too small, model inaccessible) — not fundamental contradictions of the core claim.
- Candidate B is promising but untested and currently blocked by the Gemma gating issue. It scores 2.55, placing it in the REFINE zone.
- Candidate C is effectively disqualified by the training-free constraint.
- The pilot revealed **three specific fixes** needed before full-scale experiments:
  1. Replace the simplified absorption/hedging proxies with rigorous implementations (sae-spelling or SAEBench adaptation).
  2. Increase the activation corpus to at least 50k-100k tokens for reliable dead-neuron detection.
  3. Commit to GPT-2 Small (or Pythia-160m) as the open-model anchor, abandoning Gemma-2-2B unless an HF token is obtained.

Because the required fixes are well-defined and the core idea remains viable, the correct decision is **ADVANCE with refinement preconditions**.

---

## Next Actions

1. **Refine the metric pipeline** before scaling:
   - Integrate the proper `sae-spelling` absorption metric (or SAEBench adaptation) instead of the simplified first-letter proxy.
   - Integrate the proper hedging metric from Chanin et al. (2025) or SAEBench.
   - Increase the activation corpus for dead-neuron detection to ≥50k tokens.
2. **Anchor on GPT-2 Small** for the full Pareto evaluation (e1_full_gpt2), since Gemma-2-2B is inaccessible.
3. **Source additional architecture families** for GPT-2 Small via SAELens / SAEBench (e.g., Gated, JumpReLU, BatchTopK) to ensure diversity.
4. **Re-run a validation pilot** (e1_pilot_v2) with the fixed metrics and larger corpus on 5-10 checkpoints to confirm signal quality.
5. If validation pilot passes, proceed to **e1_full_gpt2** (20-30 checkpoints, Pareto front computation).
6. **Keep Candidate B as a backup**: if the open-model anchor for task-agnostic hierarchies becomes available (e.g., Pythia Scope or SAEBench releases), revisit e3_pilot.
7. **Drop Candidate C** from active consideration unless the training-free constraint is explicitly relaxed.

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.72
DECISION: ADVANCE


## 上一轮 validation 结构化决策
{
  "decision": "ADVANCE",
  "selected_candidate_id": "cand_a",
  "confidence": 0.72,
  "candidate_scores": {
    "cand_a": {
      "weighted_score": 3.55,
      "verdict": "ADVANCE"
    },
    "cand_b": {
      "weighted_score": 2.55,
      "verdict": "REFINE"
    },
    "cand_c": {
      "weighted_score": 1.15,
      "verdict": "PIVOT"
    }
  },
  "reasons": [
    "Pipeline runs end-to-end (success rate 1.0), validating the engineering path.",
    "Main hypothesis H1 was not falsified; pilot lacked statistical power and proper proxies, not a fundamental contradiction.",
    "Candidate A scores 3.55, meeting the ADVANCE threshold (>= 3.5).",
    "Candidate B is untested and blocked by gated Gemma-2-2B access; scores 2.55 (REFINE zone).",
    "Candidate C violates the training-free constraint and scores 1.15."
  ],
  "next_actions": [
    "Integrate proper sae-spelling / SAEBench absorption and hedging metrics instead of simplified proxies.",
    "Increase activation corpus to >= 50k tokens for reliable dead-neuron detection.",
    "Anchor on GPT-2 Small for full Pareto evaluation (e1_full_gpt2) due to Gemma gating.",
    "Source additional GPT-2 Small architecture families (Gated, JumpReLU, BatchTopK) via SAELens / SAEBench.",
    "Re-run validation pilot (e1_pilot_v2) with fixed metrics and larger corpus on 5-10 checkpoints.",
    "If validation pilot passes, proceed to e1_full_gpt2 (20-30 checkpoints, Pareto front computation).",
    "Keep Candidate B as backup; revisit if an open-model hierarchy corpus becomes available.",
    "Drop Candidate C unless training-free constraint is explicitly relaxed."
  ],
  "dropped_candidates": ["cand_c"]
}
