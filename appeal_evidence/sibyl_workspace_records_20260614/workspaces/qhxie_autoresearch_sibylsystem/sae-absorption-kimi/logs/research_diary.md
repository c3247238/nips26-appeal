# Iteration 1 研究日记

## 项目初始化
- 项目名: sae-absorption
- 主题: 研究稀疏自编码器（SAE）中的特征吸收（feature absorption）现象
- 时间: 2026-04-14

## 文献调研
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


## 初始想法
- 设计系统化的分析框架，量化特征吸收现象的普遍程度（跨不同模型层、不同 SAE 配置）
- 探索特征吸收的成因：与特征共现频率、稀疏惩罚强度、SAE 字典大小等因素的关系
- 开发可复现的评估指标，衡量吸收现象对下游解释性任务（如电路发现、概念探测）的影响
- 研究方法以 training-free 分析为主，基于现有的预训练 SAE（如 SAELens 库中的 SAE）进行分析

## 上下文


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



# Iteration 0

**Score**: 5.5/10
**Issues**: 12
**Trajectory**: stagnant

## Reflection
# Reflection Report: Iteration 001

## Iteration Summary

Iteration 001 produced a first draft of "The Impossibility Triangle of Sparse Autoencoders" with a genuinely novel conceptual reframing---shifting SAE research from "fixing absorption" to "navigating unavoidable tradeoffs." The paper earned a supervisor score of **5.5 / 10** (verdict: CONTINUE). While the framing and literature positioning are strong, the empirical foundation is undermined by a degenerate absorption proxy and a failed pilot-to-full escalation gate. Experiment 2 (N=314 meta-analysis) is the clearest asset; Experiments 1 and 3 require substantial repair before the paper can be considered submission-ready.

---

## Issue Analysis by Category

### EXPERIMENT (Critical)

1. **Degenerate absorption proxy.** The simplified first-letter absorption metric returns exactly 0.0 on 26 of 27 GPT-2 checkpoints in E1 and 9 of 10 checkpoints in E3. A metric with near-zero variance cannot support claims about architectural tradeo

## Review Summary
continue The paper makes a genuinely novel conceptual reframing—shifting SAE research from 'fixing absorption' to 'navigating tradeoffs'—but its empirical foundation is undermined by a degenerate absorption proxy that returns zero on 96% of E1 checkpoints and 90% of E3 checkpoints. The negative correlation between the task-agnostic and first-letter metrics (r = -0.59) is more alarming than acknowledged, suggesting the two measures capture different constructs rather than a shared 'absorption' ph

## Critique Summary
# Writing Critique

## Overall Assessment

The manuscript is structurally sound and the prose is generally clear, but it suffers from three major writing flaws that would draw negative reviewer attention at a top venue: (1) a missing forward reference for Figure 4, (2) unlabeled inline tables instead of proper LaTeX tables, and (3) terminology drift that undermines precision. The paper also overreaches with causal language in an observational study.

## Major Issues

### 1. Missing Forward Refer


# Iteration 1

**Score**: 5.5/10
**Issues**: 12
**Trajectory**: stagnant

## Reflection
# Reflection Report: Iteration 001

## Iteration Summary

Iteration 001 produced a first draft of "The Impossibility Triangle of Sparse Autoencoders" with a genuinely novel conceptual reframing---shifting SAE research from "fixing absorption" to "navigating unavoidable tradeoffs." The paper earned a supervisor score of **5.5 / 10** (verdict: CONTINUE). While the framing and literature positioning are strong, the empirical foundation is undermined by a degenerate absorption proxy and a failed pilot-to-full escalation gate. Experiment 2 (N=314 meta-analysis) is the clearest asset; Experiments 1 and 3 require substantial repair before the paper can be considered submission-ready.

---

## Issue Analysis by Category

### EXPERIMENT (Critical)

1. **Degenerate absorption proxy.** The simplified first-letter absorption metric returns exactly 0.0 on 26 of 27 GPT-2 checkpoints in E1 and 9 of 10 checkpoints in E3. A metric with near-zero variance cannot support claims about architectural tradeo

## Review Summary
continue The paper makes a genuinely novel conceptual reframing—shifting SAE research from 'fixing absorption' to 'navigating tradeoffs'—but its empirical foundation is undermined by a degenerate absorption proxy that returns zero on 96% of E1 checkpoints and 90% of E3 checkpoints. The negative correlation between the task-agnostic and first-letter metrics (r = -0.59) is more alarming than acknowledged, suggesting the two measures capture different constructs rather than a shared 'absorption' ph

## Critique Summary
The project has made substantial empirical progress since the last critic round, but the paper draft has not kept pace. The most critical issue: the newly completed E2 full experiments (GPT-2 and Pythia) show that the official sae-spelling absorption metric has meaningful variance across all families, directly falsifying H2's prediction of degeneracy. Yet the paper still claims the first-letter benchmark is 'degenerate' based on the old simplified proxy. This creates a severe internal contradict


# Iteration 2

**Score**: 5.5/10
**Issues**: 19
**Fixed**: 4
**Trajectory**: declining

## Reflection
# Iteration 002 Reflection Report

## Iteration Summary

Iteration 002 executed a construct-validity study of the SAEBench feature absorption metric, testing whether first-letter absorption scores predict semantic-hierarchy absorption across 8 SAE architectures on Pythia-160M. The study was a pivot from Iteration 001's degenerate absorption proxy problem. All 8 experiments completed successfully in ~42 minutes with zero GPU idle time.

**Key Results:**
- H1 (construct validity): r = 0.463, CI [-0.389, 0.981] --- below pre-registered threshold (r > 0.6)
- H2 (hierarchy specificity): reversed --- non-hierarchy (0.331) > hierarchy (0.235), t = -4.748, p = 0.0032
- H3 (tau_fs robustness): stable across thresholds (r = 0.468, 0.463, 0.471)
- Random-SAE control: identical to Standard SAE (0.352) on semantic tasks, near-zero (0.030) on first-letter
- GPT-2 replication: near-zero scores (0.000, 0.003) with ceiling effect (k_sparse_acc near 1.0)

**Supervisor Score: 5.5/10** (Borderline Reject,

## Review Summary
continue This paper proposes the first construct-validity study of the SAEBench feature absorption metric, a timely and important question. However, the execution suffers from critical methodological flaws that undermine its central claims: (1) an internal contradiction in the Random-SAE control description (encoder vs decoder permutation) that the paper does not resolve, (2) a fundamental metric design flaw where perfect probe accuracy causes the absorption formula to collapse to k-sparse probi

## Critique Summary
This construct-validity study of the SAEBench absorption metric reveals significant methodological weaknesses: (1) an inconclusive correlation (r=0.463, CI spanning negative to near-perfect) due to severely underpowered sample size (n=7 SAEs), (2) a failed hierarchy-specificity test where non-hierarchy features show higher 'absorption' than hierarchies, (3) a degenerate Random-SAE control that matches trained SAEs on semantic tasks, and (4) critical internal contradictions in the paper's descrip


# Iteration 3

**Score**: 6.5/10
**Issues**: 16
**Fixed**: 4
**Trajectory**: declining

## Reflection
# Reflection Report: Iteration 003

## Iteration Summary

Iteration 003 attempted a pivot from a construct-validity study to a Goodhart's Law critique of the SAEBench absorption metric. The conceptual reframing is timely and valuable, but the execution introduced critical methodological problems that undermine the central claims. The supervisor score dropped from 5.5 (iter_002) to 6.5 (iter_003)---a modest improvement in score but driven by the novelty of the reframing rather than empirical quality. The critic's assessment is far harsher, characterizing the iteration as a "framing catastrophe" where the paper promises experiments (H1-H4) that were not actually delivered.

The core iter_002 results---H2 hierarchy specificity failure (non-hierarchy > hierarchy, t=-4.75, p=0.003) and the Random=Standard identity on semantic hierarchies---remain the strongest evidence and survive scrutiny. However, iter_003 added estimated data and underpowered tests that weaken rather than strengthen the 

## Review Summary
continue This iteration pivots from a construct-validity study to a Goodhart's Law critique of the SAEBench absorption metric, a timely and important reframing. The paper makes a genuinely novel contribution by connecting metric decomposition (random/PCA baselines), hierarchy-specificity failure, and downstream utility disconnect into a unified argument. However, critical methodological weaknesses undermine the central claims: (1) the PCA SAE and low-cooccurrence controls are estimated/theoretic

## Critique Summary



# Iteration 4

**Score**: 7.0/10
**Issues**: 13
**Fixed**: 7
**Trajectory**: improving

## Reflection
# Reflection Report: Iteration 004

## Iteration Summary

Iteration 004 successfully reverted from the problematic Goodhart's Law framing of iter_003 back to a clean construct-validity study. The paper now presents three research questions (H1-H3) with real empirical data throughout. The supervisor score improved from 6.5 (iter_003) to 7.0 (iter_004), driven by honest reframing, removal of estimated data, and methodological soundness. The core iter_002 results---H2 hierarchy specificity failure (t=-4.748, p=0.003, d=-1.794)---remain the strongest evidence and survive scrutiny.

However, iter_004 introduced a new and serious data integrity issue: the paper reports Random-SAE semantic-hierarchy absorption as 0.352 (identical to Standard SAE), but the actual experimental data shows Random=0.175. This appears to be a copy-paste error from iter_003's e1_decomposition_results.json, where estimated Random scores were copied from Standard. The iter_004 paper inherited this error in Table 1 and

## Review Summary
continue This iteration successfully addresses the previous review's critical concerns by removing the Goodhart's Law overreach, eliminating estimated PCA and low-cooccurrence data, and reverting to a clean construct-validity study framed around three research questions (H1-H3). The paper now presents an honest, well-scoped contribution: the first empirical test of whether the SAEBench absorption metric generalizes from first-letter to semantic hierarchies. The hierarchy-specificity failure (H2 

## Critique Summary



# Iteration 5

**Score**: 5.0/10
**Issues**: 22
**Trajectory**: improving

## Reflection
# Reflection Report: Iteration 5

## Iteration Summary

Iteration 5 produced a component-isolated causal analysis of SAE absorption-reduction mechanisms on SynthSAEBench-16k. The paper's core finding---that TopK sparsity (Cohen's d = 5.51) dominates over multi-scale decomposition and orthogonality---is striking and well-supported by the data. The paper is well-structured, internally consistent, and honestly reports negative results and limitations.

However, critical methodological flaws undermine several key claims. The most serious issue is the persistent gap between provisional data (3 of 6 variants with full replicates) and definitive-sounding claims in the Abstract and Conclusion. Additionally, the r~+0.93 sparsity-absorption correlation is based on n=4 points with two at identical L0=50, making it mathematically fragile. The comparison between TopK (full experiment) and MultiScale (pilot) violates like-with-like principles.

Writing quality score: 8/10 (from writing/review.md). T

## Review Summary


## Critique Summary
This paper presents a component-isolated causal analysis of SAE absorption-reduction mechanisms on SynthSAEBench-16k. While the core finding (TopK sparsity dominates with d=5.51) is striking and well-supported by the data, critical methodological flaws undermine several key claims: (1) the r~-0.97 sparsity-absorption correlation is based on n=4 points with two points at identical L0=50, making it mathematically fragile and potentially confounded; (2) the paper reports 3 of 6 variants with full d


# Iteration 6

**Score**: 5.0/10
**Issues**: 0
**Trajectory**: stagnant

## Reflection


## Review Summary


## Critique Summary



# Iteration 7

**Score**: 5.5/10
**Issues**: 17
**Fixed**: 8
**Trajectory**: stagnant

## Reflection
# Reflection Report: Iteration 009

## Iteration Summary

This iteration produced a paper titled "L0-Matched or Misleading?" that makes an important methodological observation: L0 (sparsity level) is the dominant driver of absorption rate, confounding cross-architecture comparisons. The paper reports experiments on 6 SAE variants (Baseline L1, Gated, OrtSAE, Matryoshka, TopK, Random control) with 5 replicates each, plus 3 ablation studies and a dose-response lambda sweep (5 levels x 5 seeds = 25 measurements).

**Supervisor Score: 5.5 / 10 (Borderline Reject, Verdict: REVISE)**
**Critic Assessment: Critical metric issues, computational anomalies, insufficient statistical analysis**
**Writing Quality Score: 7 / 10**

The core idea has merit but the execution is compromised by critical data errors, an invalid central metric, and overstated conclusions.

---

## Issue Analysis by Category

### EXPERIMENT (Critical: 3, Major: 3, Minor: 3)

**Critical Issue 1: MCC at Chance Level**
Feature 

## Review Summary
revise The paper makes an important methodological point---that L0 confounds cross-architecture absorption comparisons---but the execution is undermined by critical data errors, metric invalidity, and overstated conclusions. The central claim that 'absorption does not causally predict downstream interpretability' rests on an MCC metric that is at chance level (~0.22) for ALL variants including an untrained Random control. Table 1 reports 0% dead latents for all variants, but raw data shows TopK 

## Critique Summary
This iteration (009) makes a genuinely important methodological point: L0 confounds cross-architecture absorption comparisons. However, the paper overstates its conclusions, contains metric gaming risks, has reproducibility gaps, and fails to address several critical methodological issues. The central claim—that absorption does not causally predict downstream interpretability—is supported by only one metric (MCC) on synthetic data with extremely low feature recovery across all conditions (~0.22)


# Iteration 8

**Score**: 5.0/10
**Issues**: 0
**Trajectory**: stagnant

## Reflection


## Review Summary


## Critique Summary



# Iteration 9

**Score**: 5.0/10
**Issues**: 0
**Trajectory**: stagnant

## Reflection


## Review Summary


## Critique Summary

