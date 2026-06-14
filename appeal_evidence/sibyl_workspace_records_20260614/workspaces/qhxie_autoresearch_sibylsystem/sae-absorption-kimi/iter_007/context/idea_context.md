

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


## 文献调研报告（请仔细阅读，避免重复已有工作）
# Literature Survey Report

**Research Topic**: Systematic Analysis and Quantification of Feature Absorption in Sparse Autoencoders (SAEs): Causes, Patterns, and Impact on Interpretability
**Survey Date**: 2026-04-25
**arXiv Search Keywords**: ["sparse autoencoder feature absorption", "SAE feature splitting absorption", "Matryoshka sparse autoencoder", "hierarchical sparse autoencoder", "orthogonal sparse autoencoder OrtSAE", "TopK SAE interpretability"]
**Web Search Keywords**: ["SAE feature absorption mechanism 2025", "SAEBench benchmark absorption evaluation", "Matryoshka SAE absorption reduction", "sparse autoencoder GitHub open source"]

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant paradigm for mechanistic interpretability of large language models (LLMs), following Anthropic's foundational work on monosemanticity (Bricken et al., 2023). The core premise is that SAEs can decompose polysemantic neural activations into sparse, human-interpretable feature directions. Over 2024-2025, the field has shifted from demonstrating feasibility to rigorously diagnosing failure modes and building standardized benchmarks.

Feature absorption stands out as one of the most theoretically consequential failure modes. Identified by Chanin et al. (2024) in "A is for Absorption," it describes how sparsity incentives cause parent features in a semantic hierarchy to be subsumed by their child features---creating "holes" in feature coverage and undermining the reliability of SAE-based interpretability. This discovery has catalyzed a wave of follow-up work: benchmark integration (SAEBench), hierarchical architectures (Matryoshka SAEs, HSAEs), orthogonality constraints (OrtSAE), and theoretical analyses of training dynamics (feature hedging, bias adaptation). The current state of the field is characterized by an active tension between improving reconstruction/sparsity trade-offs and ensuring that learned features are not merely interpretable-looking but causally faithful and structurally sound.

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 (NeurIPS 2025) | 2024 | Introduced feature absorption as a sparsity-driven failure mode; proposed a detection metric; validated on hundreds of LLM SAEs. | Toy-model focus (first-letter tasks); no general theoretical solution proposed. |
| 2 | Towards Monosemanticity: Decomposing Language Models with Dictionary Learning | Transformer Circuits Thread | 2023 | Foundational SAE work; demonstrated monosemantic feature recovery in a 512-neuron MLP; introduced feature splitting. | Did not identify or address absorption. |
| 3 | Scaling and Evaluating Sparse Autoencoders | arXiv:2406.04093 (ICLR 2025) | 2024 | Large-scale SAE scaling study on GPT-2/GPT-4; established training best practices and evaluation metrics. | Focused on aggregate metrics; absorption not a central concern. |
| 4 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders | arXiv:2503.09532 | 2025 | Standardized 8-evaluation benchmark including Feature Absorption, AutoInterp, Sparse Probing, SCR, RAVEL. | Absorption evaluation is computationally expensive (~26 min per SAE). |
| 5 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | arXiv:2503.17547 | 2025 | Proposed nested dictionary SAEs that learn features at multiple scales; dramatically reduced absorption rates (0.05 vs. 0.49). | Inner levels act as narrow SAEs, exacerbating feature hedging (Chanin et al. 2025). |
| 6 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | arXiv:2505.11756 | 2025 | Identified feature hedging as a reconstruction-loss-driven pathology in narrow SAEs; proposed balanced Matryoshka variant. | Focuses on narrow-width regime; less relevant to very wide SAEs. |
| 7 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders | arXiv:2602.11881 | 2026 | Proposed HSAE with explicit parent-child relationships and structural constraint loss to learn feature hierarchies. | Very recent preprint; limited community validation so far. |
| 8 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | Enforced decoder orthogonality via chunk-wise penalty; reduced absorption by 65% and composition by 15%. | Adds ~4-11% compute overhead; chunk size is a new hyperparameter. |
| 9 | Are Sparse Autoencoders Useful? A Case Study in Sparse Probing | arXiv:2502.16681 | 2025 | Critical evaluation showing SAEs do not consistently outperform strong non-SAE baselines on downstream probing tasks. | Does not isolate absorption as the cause of underperformance. |
| 10 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | Argued for feature consistency (PW-MCC metric) as a community priority; showed high consistency is achievable with TopK SAEs. | Consistency does not guarantee absence of absorption or causal validity. |
| 11 | Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders | arXiv:2506.14002 (ICLR 2026) | 2025 | First SAE algorithm with theoretical feature recovery guarantees; introduced Group Bias Adaptation (GBA). | Theory assumes a specific generative model; real-world LLM features may not fit. |
| 12 | On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy | arXiv:2506.15963 | 2025 | Closed-form theoretical solution for SAEs; proves extreme sparsity needed for full recovery; proposes WSAE with adaptive weighting. | Simplified theoretical model; limited empirical validation. |
| 13 | CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of Interpretability of Sparse Autoencoders | arXiv:2509.00691 | 2025 | LLM-free deterministic evaluation using contrastive story pairs; >70% Spearman correlation with SAEBench. | Newer benchmark; less community validation than SAEBench. |
| 14 | Data Whitening Improves Sparse Autoencoder Learning | arXiv:2511.13981 | 2025 | PCA whitening of input activations improves interpretability metrics; challenges assumption that optimal sparsity-fidelity aligns with interpretability. | Minor reconstruction quality drop. |
| 15 | Building a Structured Feature Forest with Hierarchical Sparse Autoencoders | arXiv:2602.11881 | 2026 | Tree-structured hierarchical SAEs with explicit parent-child relationships. | Very recent preprint; limited community validation. |
| 16 | A Survey on Sparse Autoencoders: Interpreting the Internal Representations of LLMs | arXiv:2503.05613 | 2025 | Comprehensive survey evaluating LLaMA Scope, Pythia SAE, Gemma Scope using SAEBench metrics. | Critical of SAE utility vs. simple linear probes. |
| 17 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | arXiv:2407.14435 | 2024 | JumpReLU architecture with learnable per-feature threshold; improved reconstruction fidelity. | Does not directly address absorption. |
| 18 | Gated Sparse Autoencoders | arXiv:2405.14719 | 2024 | Separate gate and magnitude paths for better sparsity control. | Does not directly address absorption. |
| 19 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Anthropic Blog / arXiv | 2024 | Scaled SAEs to production LLM; 34M features across all layers. | Proprietary; limited methodological detail. |
| 20 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | Argued for feature consistency (PW-MCC metric) as community priority; high consistency achievable with TopK SAEs. | Consistency does not guarantee absence of absorption. |

---

## 3. SOTA Methods and Benchmarks

### Current Best Methods

| Method | Core Innovation | Absorption Impact | Trade-off |
|--------|----------------|-------------------|-----------|
| **BatchTopK SAE** | Explicit k-sparsity selection per batch | Lower absorption than ReLU SAEs | Reconstruction fidelity |
| **Matryoshka SAE** | Nested dictionaries with multi-scale reconstruction | Absorption rate ~0.05 vs. 0.49 (BatchTopK) | Feature hedging in inner levels |
| **Balanced Matryoshka SAE** | Tuned loss coefficients across hierarchy levels | Better absorption-hedging Pareto frontier | Additional hyperparameter tuning |
| **OrtSAE** | Chunk-wise orthogonality penalty on decoder weights | -65% absorption, -15% composition | ~4-11% compute overhead |
| **HSAE** | Explicit tree-structured parent-child constraints | Reduced absorption in hierarchical settings | Complex architecture, early validation |
| **GBA (Group Bias Adaptation)** | Adaptive bias for frequency-matched activation sparsity | Theoretically avoids absorption under model assumptions | Empirical validation up to 1.5B params |

### Mainstream Benchmarks

- **SAEBench** (Karvonen et al., 2025): The dominant community benchmark. Includes 8 evaluations:
  1. Feature Absorption
  2. AutoInterp
  3. Sparse Probing
  4. Spurious Correlation Removal (SCR)
  5. RAVEL
  6. Targeted Probe Perturbation (TPP)
  7. Unlearning
  8. L0 / Loss Recovered

- **CE-Bench** (Peng et al., 2025): LLM-free alternative using 5,000 contrastive story pairs across 1,000 subjects. Achieves >70% Spearman correlation with SAEBench results while being fully deterministic and reproducible.

- **Absorption Evaluation Protocol** (from SAEBench, based on Chanin et al. 2024):
  - Train ground-truth probes (logistic regression) on model residual stream activations for token properties (e.g., starting letter).
  - Use k-sparse probing (k=1..10) to identify main latents for each property.
  - Detect feature splitting via F1 threshold tau_fs = 0.03.
  - Find false negatives: test examples where main feature-split latents fail to activate but the LR probe correctly classifies.
  - Detect absorption via integrated-gradients ablation on false-negative tokens.
  - Absorption criteria: absorbing latent must have cosine similarity >= 0.025 with LR probe; ablation effect >= 1.0x larger than second-highest; main SAE latents must not activate.
  - Final formula: absorption_rate = num_absorptions / lr_probe_true_positives
  - Report `1 - absorption_score` (higher is better).

### Evaluation Metrics

| Metric | What It Measures | Typical Target |
|--------|------------------|----------------|
| L0 | Average active features per token | 50-200 |
| CE Loss Score | Cross-entropy recovered vs. original | 80-95% |
| Explained Variance | Reconstruction quality | >90% |
| Absorption Score | Degree of parent-feature subsumption | As low as possible |
| Feature Splitting Rate | Fragmentation of concepts | Context-dependent |
| PW-MCC | Pairwise dictionary mean correlation (consistency) | ~0.80+ |
| SCR (Spurious Correlation Removal) | Can removing top latents reduce unwanted biases? | Higher is better |
| TPP (Targeted Probe Perturbation) | Do latents causally isolate specific concepts? | Higher is better |

---

## 4. Identified Research Gaps

- **Gap 1: Theoretical understanding of absorption dynamics.** While Chanin et al. (2024) identified absorption and Bussmann et al. (2025) mitigated it, there is no unified theoretical framework predicting *when* absorption will occur for a given feature hierarchy, SAE width, and sparsity level.

- **Gap 2: Scalable causal validation of absorbed features.** Existing metrics are correlational (probe-based). There is limited work on causal interventions that definitively establish whether a latent "knows about" vs. "uses" a parent feature, especially under absorption.

- **Gap 3: Cross-architecture absorption patterns.** Most absorption studies focus on residual-stream SAEs. How absorption manifests in attention-output SAEs, MLP SAEs, or multimodal (vision-language) SAEs remains underexplored.

- **Gap 4: Training dynamics and emergence time.** It is unclear at what point during SAE training absorption emerges, whether it is reversible, and how curriculum learning or temporal masking might prevent it.

- **Gap 5: Unified objective trade-offs.** No single training objective dominates across all interpretability goals. Methods that reduce absorption (Matryoshka, OrtSAE) introduce new pathologies (hedging, overhead) or trade off reconstruction fidelity.

- **Gap 6: Real-world downstream impact.** Kantamneni et al. (2025) showed SAEs often fail to outperform baselines on sparse probing. The causal link between absorption rates and downstream task underperformance has not been systematically quantified.

- **Gap 7: Limited quantification scope.** The absorption metric from Chanin et al. (2024) is validated primarily on first-letter spelling tasks. There is no systematic quantification of absorption across diverse semantic domains (syntactic, factual, safety-related features).

- **Gap 8: No systematic analysis of absorption patterns.** While absorption has been shown to exist, there is no comprehensive study of: (a) which types of features are most prone to absorption, (b) how absorption varies across model layers, (c) how absorption scales with model size and SAE dictionary size.

- **Gap 9: Temporal dynamics of absorption emergence.** Feature absorption may evolve during training; no study tracks how absorption emerges and changes over the training trajectory.

- **Gap 10: Cross-architecture absorption comparison.** No systematic study compares how different SAE architectures (ReLU, TopK, Gated, JumpReLU) affect absorption rates under controlled conditions.

---

## 5. Available Resources

### Open-source Code

- **SAELens** (`https://github.com/jbloomAus/SAELens`): The primary PyTorch library for training and analyzing SAEs on LLMs. Supports standard, gated, TopK, and JumpReLU architectures. Integrates with TransformerLens and Neuronpedia.
- **SAEBench** (`https://github.com/adamkarvonen/SAEBench`): Comprehensive benchmarking suite with absorption evaluation, AutoInterp, sparse probing, and more. PyPI: `sae-bench`.
- **TransformerLens** (`https://github.com/neelnanda-io/TransformerLens`): Essential for extracting activations and running causal interventions on transformer models.
- **Neuronpedia** (`https://neuronpedia.org`): Browser for pre-trained SAE features (e.g., GPT-2 small, Gemma).

### Datasets

- **The Pile / pile-uncopyrighted** (`monology/pile-uncopyrighted`): Standard training corpus for SAEs.
- **OpenWebText**: Common alternative for GPT-2 scale models.
- **First-letter classification tokens** (from Chanin et al. 2024): Controlled synthetic evaluation set for absorption measurement.

### Pretrained Models and SAEs

- **GPT-2 Small SAEs** (`gpt2-small-res-jb` release on SAELens): Widely used baseline for absorption and interpretability research.
- **Gemma-2-2B SAEs** (`gemma-2b-res` release): Used in Matryoshka SAE and OrtSAE papers.
- **Gemma Scope** (Lieberum et al., 2024): Large-scale open SAEs across layers and model sizes.

---

## 6. Implications for Idea Generation

**Directions worth exploring:**
- **Training-dynamic analysis of absorption:** Designing experiments that track when and how absorption emerges during SAE training, and whether early-stopping or curriculum strategies can prevent it.
- **Causal intervention benchmarks for absorption:** Moving beyond probe-based metrics to establish causal criteria (e.g., activation patching, ablation) that verify whether absorbed parent features are truly "missing" or merely "hidden."
- **Cross-layer absorption propagation:** Studying whether absorption in early residual-stream SAEs propagates to and compounds in deeper layers.
- **Quantifying the downstream cost of absorption:** Systematically varying absorption rates (via architecture or hyperparameters) and measuring sparse probing, steering, and safety-monitoring performance to establish a dose-response relationship.

**Saturated or crowded directions:**
- Simply proposing a new SAE architecture and showing lower absorption on first-letter tasks is becoming crowded (Matryoshka, OrtSAE, HSAE, GBA all exist).
- Purely descriptive studies of absorption in a single model/layer are unlikely to be novel.

**Cross-domain analogies with potential:**
- **Dictionary learning in signal processing:** The concept of "incoherence" in compressed sensing parallels OrtSAE's orthogonality penalty; more principled incoherence constraints could be imported.
- **Hierarchical topic models (LDA, HDP):** The explicit tree-structured priors in topic modeling could inspire probabilistic SAE variants with built-in hierarchical structure.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens (training + analysis) | High | MIT | Adopt | Dominant community library; supports multiple architectures out of the box. |
| SAEBench (absorption eval) | High | Open source | Adopt | Standardized benchmark; directly provides the absorption metric and evaluation protocol from Chanin et al. (2024). |
| TransformerLens (activation extraction) | High | MIT | Adopt | De facto standard for mechanistic interpretability on transformers. |
| Matryoshka SAE code (from paper authors) | Medium | TBD | Extend | If publicly released, fork to study inner-level hedging vs. absorption trade-offs. |
| OrtSAE code (from paper authors) | Medium | TBD | Extend | If released, useful for comparing orthogonality-based absorption reduction against hierarchical methods. |
| Neuronpedia (feature browser) | Low | N/A | Compose | Use for qualitative validation and feature inspection, not for quantitative experiments. |

**Highlight:**
- **Reusable evaluation framework:** SAEBench's absorption evaluation (`sae_bench/evals/absorption/`) is the highest-priority resource to adopt. It provides ground-truth probe training, k-sparse probing, and absorption scoring.
- **Reusable data pipeline:** SAELens's `ActivationsStore` and `LanguageModelSAERunnerConfig` provide standardized activation buffering and training loops.
- **Reusable pretrained models:** GPT-2 small and Gemma-2-2B SAEs from SAELens/HuggingFace provide immediate baselines without requiring expensive training from scratch.


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Research Proposal: L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders (Iteration 9)

## Title
**L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders**

---

## Abstract

Feature absorption---where parent features in semantic hierarchies are subsumed by child features under sparsity pressure---is a recognized pathology in Sparse Autoencoders (SAEs). While Chanin et al. (2024) identified the phenomenon and subsequent work (Matryoshka SAE, OrtSAE, GBA) proposed architectural mitigations, a critical gap remains: no study has systematically established whether absorption actually causes downstream interpretability failures, or merely correlates with them. Existing absorption metrics are validated on first-letter spelling tasks, leaving open whether findings transfer to real semantic features.

This paper re-evaluates architecture claims for feature absorption under controlled, L0-matched conditions. Our pilot evidence reveals that the apparent architectural advantage of TopK and Matryoshka SAEs is confounded by sparsity: at matched L0=50, the difference may vanish. Furthermore, a dose-response study falsifies the causal link between absorption and downstream interpretability (feature recovery MCC remains flat at ~0.22 regardless of absorption rate). The expected contribution is a methodologically rigorous re-evaluation that challenges the "mitigation" framing and redirects community effort toward understanding *when* absorption matters rather than *how to reduce* it unconditionally.

---

## 1. Motivation

### 1.1 The Absorption Problem

SAEs decompose neural activations into sparse, interpretable features. Under sparsity pressure, hierarchical parent features (e.g., "starts with S") can be absorbed into child features (e.g., "short"), creating "holes" in feature coverage. Chanin et al. (2024) showed absorption rates of 10-50% across hundreds of SAEs.

### 1.2 The Causal Gap

Despite recognition of absorption as a failure mode, the field lacks causal evidence that absorption *causes* downstream harm. Key questions remain unanswered:
- Does a SAE with 30% absorption perform worse at sparse probing than one with 5%?
- Does absorption degrade steering efficacy, or do absorbed parent features remain causally active through child features?
- Are first-letter spelling tasks representative of real semantic absorption?

The contrarian perspective raises a provocative possibility: absorption may be a feature, not a bug. Hierarchical representation through child features mirrors human cognition. Without causal evidence, the "mitigation" framing is premature.

### 1.3 Why Now

2025-2026 has seen an explosion of absorption-mitigating architectures (Matryoshka, OrtSAE, HSAE, GBA, ATM). The community needs to know which mitigations matter---not just which reduce absorption rates on toy tasks, but which improve real interpretability. This work provides that evidence.

---

## 2. Research Questions

**RQ1 (Architecture)**: Under L0-matched conditions, which SAE architectures (ReLU/L1, TopK, JumpReLU, Gated, Matryoshka, OrtSAE) exhibit genuinely different absorption rates, independent of sparsity confounds?

**RQ2 (Causality)**: Does absorption rate causally predict downstream interpretability performance (sparse probing accuracy, steering efficacy, circuit-tracing precision)?

**RQ3 (Theory)**: Can decoder mutual coherence predict absorption probability, and does this theoretical predictor generalize across architectures?

**RQ4 (Generalization)**: Do findings from first-letter tasks transfer to real semantic features (syntactic, factual, safety-related)?

---

## 3. Hypotheses

### H1: L0-Matched Architecture Effects (RQ1)
- **H1a**: Matryoshka SAE maintains lower absorption than Baseline even at matched L0 (replicating prior claim under controlled conditions).
- **H1b**: OrtSAE shows no absorption benefit over L0-matched Baseline when lambda_ortho is not tuned (consistent with iter 006 null result).
- **H1c**: TopK and JumpReLU show *higher* absorption than L0-matched Baseline (consistent with SAEBench finding that reconstruction-optimized architectures worsen absorption).

### H2: Absorption-Downstream Causality (RQ2)
- **H2a**: Absorption rate negatively correlates with sparse probing F1 (dose-response relationship).
- **H2b**: Absorption rate negatively correlates with steering efficacy (measured by ablation effect size).
- **H2c**: The correlation is causal, not merely correlational: artificially inducing absorption (via targeted sparsity increase) degrades downstream performance.

### H3: Mutual Coherence Predictor (RQ3)
- **H3a**: Decoder mutual coherence (max off-diagonal cosine similarity) positively correlates with absorption rate.
- **H3b**: The theoretical threshold mu < 1/(2k-1) from compressed sensing theory predicts absorption onset.

### H4: Task Generalization (RQ4)
- **H4a**: Absorption patterns on first-letter tasks correlate with absorption on semantic features (syntactic, factual).
- **H4b**: If H4a is falsified, first-letter absorption metrics are insufficient proxies for real interpretability.

---

## 4. Method Design

### 4.1 Experimental Framework

**Ground-truth foundation**: SynthSAEBench-16k synthetic data with known feature hierarchies eliminates probe-based confounds. Each synthetic feature has a known parent-child structure, enabling exact absorption detection without logistic regression probes.

**Validation layer**: GPT-2 small (124M) layer 8 residual stream SAEs from SAELens provide real-LLM validation. We use SAEBench absorption eval for comparability with prior work.

**Statistical rigor**: All experiments use 5 random seeds. Report mean +/- std, Cohen's d with pooled standard deviation, and Welch's t-test. Pre-register analysis plan before viewing data.

### 4.2 RQ1: Cross-Architecture L0-Matched Comparison

**Variants** (3-4 per iteration, following lessons learned scope constraint):

| Variant | Core Mechanism | Prior Absorption Claim |
|---------|---------------|----------------------|
| Baseline L1 | L1 sparse penalty | Reference (high) |
| TopK | Explicit k-sparse selection | Worse than Baseline at low L0 (SAEBench) |
| Matryoshka | Nested multi-scale dictionary | ~90% reduction (Bussmann et al.) |
| OrtSAE | Decoder orthogonality penalty | ~65% reduction (claim) |

**L0-matching protocol**:
1. Train each variant to target L0 = 50 (typical sparse regime) and L0 = 200 (moderate regime)
2. For Baseline, sweep lambda_L1 to match each variant's achieved L0
3. Compare absorption rates at matched L0

**Controls**:
- Random SAE (untrained dictionary): validates metric discrimination
- Shuffled feature labels: validates that absorption detection is not artifactual

**Expected outcome**: Separates true architectural effects from sparsity confounds. If TopK shows higher absorption at matched L0, this confirms SAEBench's finding and is a novel contribution under controlled conditions.

### 4.3 RQ2: Dose-Response Causality Study

**Design**: Systematically vary absorption rate via two independent manipulations:

**Manipulation A (Architectural)**: Use variants from RQ1 with naturally different absorption rates (range: ~5% to ~50%).

**Manipulation B (Sparsity-induced)**: Fix architecture (Baseline L1) and vary lambda_L1 to create a sparsity gradient. Higher sparsity should increase absorption.

**Downstream metrics**:
1. **Sparse probing F1**: Train linear probes on SAE latents for synthetic concept classification
2. **Steering efficacy**: Measure ablation effect size when steering with parent vs. child features
3. **Circuit-tracing precision**: Fraction of true parent-child edges correctly identified by attribution

**Causal inference**: If both Manipulation A and B show the same absorption-downstream correlation, this strengthens causal interpretation. If only B shows the correlation, the effect is sparsity-driven, not absorption-specific.

### 4.4 RQ3: Mutual Coherence Theory

**Computation**: For each trained SAE, compute decoder mutual coherence:
```
mu(W_dec) = max_{i != j} |cosine_similarity(W_dec[:, i], W_dec[:, j])|
```

**Predictions**:
- Plot mu(W_dec) vs. absorption rate across all variants and seeds
- Test H3b: Does mu < 1/(2k-1) predict absorption onset? (k = average L0)

**Theoretical contribution**: If H3a holds, this provides the first theoretically grounded absorption predictor. If H3b holds, it bridges compressed sensing theory to SAE interpretability.

### 4.5 RQ4: Task Generalization

**First-letter tasks**: Use SAEBench absorption eval on GPT-2 small (standard benchmark).

**Semantic tasks**: Design 3 semantic feature categories:
1. **Syntactic**: Part-of-speech features (noun, verb, adjective)
2. **Factual**: Country-capital relationships
3. **Safety**: Refusal-related features (harmful request detection)

**Comparison**: Compute absorption rates for each category and test correlation with first-letter absorption.

### 4.6 Data Integrity Pipeline (Iter 007-008 Enhanced)

Mandatory checks for every experiment:
1. **Feature count validation**: num_features == declared value
2. **Convergence verification**: loss curve plateaued, not early-stopped at spike
3. **Cross-seed independence**: MD5 hash of metrics across seeds to detect duplication
4. **Output file audit**: every planned experiment has a result file
5. **Numerical provenance**: paper numbers traceable to single source file

### 4.7 Statistical Methods

- **Effect size**: Cohen's d (pooled std, formula pre-registered)
- **Significance**: Welch's t-test (unequal variances)
- **Multiple comparison**: Bonferroni correction for family-wise error
- **Correlation**: Pearson r on individual replicates (n >= 15), NOT aggregated means
- **Power**: With 5 seeds x 4 variants = 20 data points per condition, detectable effect size d >= 0.9 at 80% power

---

## 5. Expected Contributions

1. **First causal evidence** linking absorption rate to downstream interpretability failure (or lack thereof), resolving the contrarian challenge.
2. **L0-matched cross-architecture comparison** with statistical rigor (5 seeds, effect sizes, pre-registration), separating true architectural effects from sparsity confounds.
3. **Theoretically grounded absorption predictor** based on decoder mutual coherence, connecting SAEs to compressed sensing theory.
4. **Generalization test** of first-letter absorption metrics to real semantic features, validating (or invalidating) the community's primary evaluation protocol.
5. **Open-source evaluation framework** with data integrity pipeline and ground-truth synthetic benchmarks.

---

## 6. Revisions from Prior Feedback

### Addressing Iter 006-007 Lessons Learned

1. **Scope control**: Reduced from 7 variants to 4 per iteration, with explicit go/no-go criteria. Full design spans 2 iterations if needed.
2. **Honest reporting**: All claims include scope notes ("based on X of Y variants"). No definitive rankings from incomplete data.
3. **Pilot-first**: Each RQ begins with a 15-minute pilot (1 seed, 1024 features) before scaling to full experiments.
4. **Control delivery**: L0-matched comparisons, Random controls, and shuffled controls are mandatory---not optional.
5. **Correlation discipline**: Small-n correlations (n < 10) reported as "exploratory observations," not primary contributions.

### Addressing Contrarian Concerns

- **"Absorption may be a feature"**: RQ2 directly tests this. If downstream performance is unaffected by absorption, the contrarian is correct and we report it.
- **"Toy-task limitation"**: RQ4 tests generalization to semantic features.
- **"Architecture comparison is confounded"**: L0-matching protocol explicitly controls for reconstruction quality confounds.

### Addressing Empiricist Concerns

- **Seed independence**: 5 seeds minimum, mean +/- std reported.
- **Convergence**: Loss curves and final values mandatory.
- **Pre-registration**: Analysis plan written before data collection.
- **Provenance**: Every number traceable through paper -> analysis.json -> variant_result.json -> raw_log.csv.

---

## 7. Novelty Assessment

| Direction | Prior Art | Our Differentiation | Risk |
|-----------|-----------|---------------------|------|
| Cross-architecture L0-matched comparison | SAEBench (2025) compares architectures but not at matched L0 with full statistical rigor | 5 seeds, Cohen's d, explicit L0-matching protocol, ground-truth synthetic data | Medium: SAEBench may release similar analysis |
| Absorption-downstream causality | Kantamneni et al. (2025) show SAEs fail downstream but don't isolate absorption | First study to systematically vary absorption and measure dose-response on multiple downstream tasks | Low: genuine gap |
| Mutual coherence predictor | OrtSAE uses orthogonality; OSAE uses ordering; no absorption predictor derived | First to derive and test mu < 1/(2k-1) as absorption predictor | Medium: theory may not empirically hold |
| Task generalization | Chanin et al. validate only on first-letter tasks | First systematic comparison of first-letter vs. semantic absorption | Low: genuine gap |

**Overall novelty assessment**: The combination of causal dose-response design + L0-matched cross-architecture comparison + theoretical predictor + generalization test is novel. Individual components have partial precedents, but the integrated package addresses a genuine and important gap.

---

## 8. Risk Assessment and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| No causal link found (absorption doesn't harm downstream) | Medium | High (paper becomes null result) | Frame as important negative result. Contrarian was right. Still publishable as it resolves a critical gap. |
| Mutual coherence theory doesn't empirically hold | Medium | Medium | Report negative result. Theory may need refinement for nonlinear encoders. |
| Semantic feature absorption detection is unreliable | Medium | High | Use multiple detection methods (probe-based + synthetic ground-truth). Report method-dependent variance. |
| Training time exceeds budget | Low | Medium | Use pretrained SAELens SAEs where possible. SynthSAEBench is training-free. |
| First-letter vs. semantic correlation is weak | Medium | Medium | This is itself a finding. Community needs to know if primary metric is unrepresentative. |

---

## 9. Experiment Timeline

| Phase | Content | Time | Go/No-Go Criteria |
|-------|---------|------|-------------------|
| Pilot RQ1 | 4 variants x 1 seed x 1024 features, L0=50 | ~15 min | All variants converge; absorption rates span >2x range |
| Full RQ1 | 4 variants x 5 seeds x 16k features, L0=50 + L0=200 | ~60 min | Pilot go criteria met |
| Pilot RQ2 | Dose-response with 3 absorption levels x 1 seed | ~15 min | Downstream metric shows monotonic trend with absorption |
| Full RQ2 | 5 absorption levels x 5 seeds, 3 downstream metrics | ~45 min | Pilot shows monotonic trend |
| Pilot RQ3 | Compute mu vs. absorption on pilot data | ~5 min | |r| > 0.5 between mu and absorption |
| Full RQ3 | Full correlation with confidence interval | ~10 min | Pilot correlation significant |
| RQ4 | Semantic task absorption (3 categories) | ~30 min | Qualitative comparison with first-letter |
| Analysis | Statistical analysis + numerical audit | ~20 min | All checks pass |

**Total estimated GPU time**: ~3 hours (parallelizable across multiple GPUs)

---

## 10. Relation to Prior Work

- **Chanin et al. (2024)**: Extends their absorption detection protocol from first-letter tasks to semantic features and downstream causality.
- **SAEBench (2025)**: Complements their cross-architecture comparison with L0-matched controls and statistical rigor.
- **Matryoshka/OrtSAE/GBA**: Independent validation of their absorption reduction claims under controlled conditions.
- **Kantamneni et al. (2025)**: Isolates absorption as a specific cause of downstream failure, which they did not do.
- **Cui et al. (2025) / Bussmann et al. (2025)**: Connects their theoretical frameworks to empirical absorption prediction.

---

## 11. Weighted Perspective Analysis

**Most influential perspectives in this synthesis**:

1. **Contrarian (highest weight)**: The challenge that "absorption might be a feature, not a bug" fundamentally reframed the research question. Instead of "how do we reduce absorption?" we now ask "does absorption cause harm?" This is a stronger, more defensible research question.

2. **Empiricist (high weight)**: The demand for causal evidence, pre-registration, seed independence, and data provenance shaped the entire experimental design. The dose-response study is a direct response to empiricist concerns.

3. **Theoretical (medium-high weight)**: The mutual coherence predictor bridges theory and experiment. Even if it fails empirically, testing it advances the field. The sigmoid absorption model provides a testable framework.

4. **Pragmatist (medium weight)**: Scope constraints (4 variants per iteration, pilot-first) and resource estimates keep the project feasible. The training-free synthetic data approach is pragmatic.

5. **Interdisciplinary (medium weight)**: The compressed sensing analogy directly inspired the mutual coherence predictor. The crystal defect analogy informed the temporal dynamics thinking (though this was deprioritized due to ATM's prior coverage).

6. **Innovator (lower weight in this round)**: The temporal dynamics idea (tracking absorption emergence during training) is genuinely novel but was deprioritized because "Time-Aware Feature Selection" (Li & Ren, 2025) already covers similar ground with adaptive temporal masking and curriculum warmup. This direction is retained as a backup idea.

**Why the front-runner was selected**: The causal dose-response design directly addresses the most important unanswered question in the field (does absorption matter?) while building on the project's existing strengths (ground-truth synthetic data, statistical rigor, honest negative results). It also provides a clear path to publication regardless of outcome: a positive causal link is actionable; a negative link is equally important as it redirects community effort.


## 当前可检验假设
# Testable Hypotheses with Expected Outcomes

## H1: L0-Matched Architecture Effects (RQ1)

### H1a: Matryoshka maintains lower absorption at matched L0
- **Directional prediction**: Matryoshka absorption rate < Baseline absorption rate at same L0
- **Expected effect size**: Cohen's d > 1.0 (large effect, replicating Bussmann et al. claim)
- **Expected values**: Matryoshka ~0.05, Baseline ~0.49 (at L0=50)
- **Test**: Welch's t-test, 5 seeds per variant
- **Falsification threshold**: If d < 0.5 or p > 0.05, claim is not supported under our conditions

### H1b: OrtSAE shows no benefit over L0-matched Baseline (untuned lambda)
- **Directional prediction**: OrtSAE absorption rate == Baseline absorption rate at matched L0
- **Expected effect size**: Cohen's d < 0.2 (negligible)
- **Rationale**: Iter 006 found null result; lambda_ortho was not tuned
- **Test**: Welch's t-test
- **Falsification threshold**: If d > 0.8, prior null result was specific to our hyperparameters

### H1c: TopK shows higher absorption than L0-matched Baseline
- **Directional prediction**: TopK absorption rate > Baseline absorption rate at same L0
- **Expected effect size**: Cohen's d > 0.8 (consistent with SAEBench finding)
- **Rationale**: TopK optimizes reconstruction, which trades off against absorption
- **Test**: Welch's t-test
- **Falsification threshold**: If d < 0.2, SAEBench finding may be specific to their evaluation protocol

---

## H2: Absorption-Downstream Causality (RQ2)

### H2a: Absorption negatively correlates with sparse probing F1
- **Directional prediction**: Pearson r < -0.5 between absorption rate and sparse probing F1
- **Expected trend**: Linear or sublinear decrease in F1 as absorption increases
- **Test**: Pearson correlation on individual replicates (n >= 20)
- **Falsification threshold**: If r > -0.3 or p > 0.05, no reliable correlation

### H2b: Absorption negatively correlates with steering efficacy
- **Directional prediction**: Pearson r < -0.5 between absorption rate and steering effect size
- **Expected mechanism**: Absorbed parent features cannot be independently steered
- **Test**: Pearson correlation
- **Falsification threshold**: If r > -0.3, absorption does not impair steering

### H2c: Causality via sparsity manipulation
- **Directional prediction**: Increasing lambda_L1 (increasing absorption) degrades downstream metrics
- **Expected trend**: Monotonic degradation with increasing sparsity/absorption
- **Test**: Linear regression of downstream metric on absorption rate, controlling for architecture
- **Falsification threshold**: If slope is not significantly negative (p > 0.05), correlation is not causal

---

## H3: Mutual Coherence Predictor (RQ3)

### H3a: Mutual coherence positively correlates with absorption
- **Directional prediction**: Pearson r > 0.5 between mu(W_dec) and absorption rate
- **Expected mechanism**: Higher decoder coherence enables feature sharing -> absorption
- **Test**: Pearson correlation across all variants and seeds
- **Falsification threshold**: If r < 0.3, decoder coherence is not a useful absorption predictor

### H3b: Theoretical threshold predicts absorption onset
- **Directional prediction**: Absorption rate increases sharply when mu > 1/(2k-1)
- **Expected pattern**: Sigmoid-like relationship with threshold at mu = 1/(2k-1)
- **Test**: Logistic regression or piecewise linear model with breakpoint at 1/(2k-1)
- **Falsification threshold**: If AUC < 0.7 for threshold-based classification, theory does not empirically hold

---

## H4: Task Generalization (RQ4)

### H4a: First-letter absorption correlates with semantic absorption
- **Directional prediction**: Pearson r > 0.5 between first-letter absorption rate and semantic absorption rate
- **Expected implication**: First-letter tasks are valid proxies for real features
- **Test**: Pearson correlation across architectures
- **Falsification threshold**: If r < 0.3, first-letter metrics are unrepresentative

### H4b: Semantic absorption rates differ by feature category
- **Directional prediction**: Absorption rates differ significantly across syntactic/factual/safety features
- **Expected pattern**: Safety features may show higher absorption (more abstract parent concepts)
- **Test**: One-way ANOVA across feature categories
- **Falsification threshold**: If F-test p > 0.05, absorption is uniform across semantic domains

---

## Pre-registered Analysis Plan

### Primary Analyses (must be reported regardless of outcome)

1. **H1a-c**: Welch's t-test for each architecture vs. L0-matched Baseline
2. **H2a-b**: Pearson correlation with 95% CI between absorption and downstream metrics
3. **H3a**: Pearson correlation between mutual coherence and absorption

### Secondary Analyses (report if significant, note if not)

4. **H2c**: Linear regression of downstream metrics on absorption, controlling for architecture
5. **H3b**: Threshold model fit for mu = 1/(2k-1)
6. **H4a-b**: Cross-task correlation and ANOVA

### Exploratory Analyses (report with appropriate caveats)

7. Interaction effects: Does architecture moderate the absorption-downstream relationship?
8. Layer-wise patterns: Does absorption vary across model layers?
9. Dead latent confounding: Does dead latent rate correlate with absorption?

---

## Decision Rules for Pivot

| Hypothesis outcome | Decision |
|-------------------|----------|
| H2a AND H2b supported (r < -0.5, p < 0.05) | PROCEED with front-runner; consider Alternative A for next iteration |
| H2a OR H2b null (r > -0.3 or p > 0.05) | PIVOT to Alternative D (absorption as compositional semantics) |
| H3a AND H3b supported | PROCEED; consider Alternative C (architecture design) |
| H3a supported but H3b null | Report as partial validation; theory needs refinement |
| H4a null (first-letter doesn't generalize) | Flag as major methodological concern; prioritize semantic evaluation |
| All H1-H4 null or weak | PIVOT to Alternative D; absorption may not be the right framing |


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_a",
      "title": "Causal Link Between Feature Absorption and Downstream Interpretability Failure",
      "status": "front_runner",
      "summary": "Systematically establish whether absorption rate causally predicts downstream interpretability performance via dose-response design. Cross-architecture L0-matched comparison + mutual coherence theory + task generalization test.",
      "hypotheses": [
        "H1a: Matryoshka maintains lower absorption than Baseline at matched L0",
        "H1b: OrtSAE shows no benefit over L0-matched Baseline (untuned)",
        "H1c: TopK shows higher absorption than L0-matched Baseline",
        "H2a: Absorption negatively correlates with sparse probing F1",
        "H2b: Absorption negatively correlates with steering efficacy",
        "H2c: Causality via sparsity manipulation (monotonic degradation)",
        "H3a: Decoder mutual coherence positively correlates with absorption",
        "H3b: Theoretical threshold mu < 1/(2k-1) predicts absorption onset",
        "H4a: First-letter absorption correlates with semantic absorption",
        "H4b: Semantic absorption rates differ by feature category"
      ],
      "pilot_focus": "4 variants x 1 seed x 1024 features, L0=50. Verify absorption rates span >2x range and downstream metrics show monotonic trend.",
      "go_criteria": "Pilot shows absorption rates span >2x range AND downstream metrics show monotonic trend with absorption",
      "no_go_criteria": "Pilot shows all variants have similar absorption rates (no variation to study dose-response)",
      "expected_contribution": "First causal evidence linking absorption to downstream harm (or lack thereof); L0-matched cross-architecture comparison with statistical rigor; theoretically grounded absorption predictor",
      "novelty_assessment": "Medium-high. Individual components have partial precedents (SAEBench cross-architecture, Kantamneni downstream failure) but the integrated causal dose-response design + L0-matching + theory + generalization is novel.",
      "risk_level": "Medium",
      "estimated_gpu_hours": 3,
      "perspectives_weighted": {
        "contrarian": "highest - reframed research question from 'reduce absorption' to 'does absorption cause harm?'",
        "empiricist": "high - shaped experimental design with pre-registration, seed independence, data provenance",
        "theoretical": "medium-high - mutual coherence predictor bridges theory and experiment",
        "pragmatist": "medium - scope constraints and pilot-first approach keep project feasible",
        "interdisciplinary": "medium - compressed sensing analogy inspired mutual coherence direction",
        "innovator": "lower - temporal dynamics deprioritized due to ATM prior coverage"
      }
    },
    {
      "candidate_id": "cand_b",
      "title": "Temporal Dynamics of Absorption Emergence and Curriculum Prevention",
      "status": "backup",
      "summary": "Track when absorption emerges during SAE training, identify early warning signals, and test curriculum learning strategies (sparsity annealing, early stopping) to prevent absorption.",
      "hypotheses": [
        "Absorption onset occurs at a predictable training step (phase transition)",
        "Feature hedging precedes absorption (hedging is early warning signal)",
        "Sparsity annealing (gradual increase) reduces absorption vs. fixed sparsity",
        "Early stopping at pre-onset point maintains interpretability without full convergence"
      ],
      "pilot_focus": "Train Baseline SAE on GPT-2 small, log absorption every 1000 steps. Identify onset point.",
      "go_criteria": "Clear onset point identified; absorption increases >3x from early training to convergence",
      "no_go_criteria": "Absorption is present from step 1 (no emergence dynamics)",
      "expected_contribution": "First characterization of absorption emergence dynamics; practical curriculum learning strategy",
      "novelty_assessment": "Medium. Time-Aware Feature Selection (Li & Ren 2025) covers adaptive temporal masking but does not track emergence or test curriculum strategies.",
      "risk_level": "Medium",
      "estimated_gpu_hours": 4,
      "pivot_trigger": "RQ2 positive (absorption causes harm) - natural progression to prevention"
    },
    {
      "candidate_id": "cand_c",
      "title": "Cross-Layer Absorption Propagation",
      "status": "backup",
      "summary": "Study whether absorption in early-layer SAEs propagates to deeper layers, compounding interpretability degradation in multi-layer workflows.",
      "hypotheses": [
        "Absorption rate increases with layer depth",
        "Absorption at layer L predicts absorption at layer L+1 (propagation)",
        "Fixing absorption at early layers (via Matryoshka) reduces downstream absorption"
      ],
      "pilot_focus": "Measure absorption on layers 0, 4, 8 of GPT-2 small using pretrained SAELens SAEs.",
      "go_criteria": "Absorption rates differ significantly across layers (ANOVA p < 0.05)",
      "no_go_criteria": "Absorption rates are uniform across layers",
      "expected_contribution": "First cross-layer absorption study; practical guidance for multi-layer SAE deployment",
      "novelty_assessment": "High. No prior work on cross-layer absorption propagation identified.",
      "risk_level": "Low",
      "estimated_gpu_hours": 2,
      "pivot_trigger": "RQ4 shows layer-dependent absorption patterns"
    },
    {
      "candidate_id": "cand_d",
      "title": "Absorption as Compositional Semantics: Reframing the Pathology",
      "status": "backup",
      "summary": "If absorption does not cause downstream harm, investigate whether absorbed parent features remain causally active through child features---reframing absorption as a form of compositional feature representation.",
      "hypotheses": [
        "Absorbed parent features can be recovered by linear combination of child features",
        "Compositional recoverability is higher in absorbed features than in non-absorbed features",
        "Explicit hierarchical architectures (HSAE, Matryoshka) achieve similar compositionality without absorption"
      ],
      "pilot_focus": "Test compositional recoverability on 10 absorbed parent features from pilot data.",
      "go_criteria": "Compositional recoverability > 0.7 for absorbed features",
      "no_go_criteria": "Compositional recoverability < 0.3 (absorbed features are truly lost)",
      "expected_contribution": "Reframing of absorption from pathology to compositional representation; connection to hierarchical topic models",
      "novelty_assessment": "High. Directly challenges established framing. Connects to cognitive science literature on hierarchical concepts.",
      "risk_level": "Medium",
      "estimated_gpu_hours": 2,
      "pivot_trigger": "RQ2 null (absorption does not cause harm) - contrarian was right"
    }
  ],
  "synthesis_notes": {
    "front_runner_selection_rationale": "Candidate A was selected because it directly addresses the most important unanswered question in the field: does absorption actually cause downstream harm? This question is stronger than 'how do we reduce absorption?' because it establishes whether the entire mitigation research direction is warranted. The dose-response design provides causal evidence regardless of outcome (positive or null), making it publishable either way. It also builds on the project's existing strengths: ground-truth synthetic data (SynthSAEBench), statistical rigor, and honest negative result reporting.",
    "deprioritized_directions": [
      "Temporal dynamics (Innovator): Partially covered by Li & Ren (2025) Time-Aware Feature Selection with adaptive temporal masking and curriculum warmup. Retained as backup (Candidate B).",
      "New architecture proposal (Innovator/Interdisciplinary): The field is crowded with Matryoshka, OrtSAE, HSAE, GBA, ATM. A new architecture would need strong differentiation. Retained as conditional backup (Alternative C in alternatives.md)."
    ],
    "key_conflicts_resolved": [
      "Contrarian vs. Mainstream: Instead of assuming absorption is harmful, we test it. If harmful -> proceed with mitigation. If benign -> pivot to compositional semantics (Candidate D).",
      "Innovator vs. Pragmatist: Temporal dynamics idea is novel but expensive. Pilot-first approach keeps it feasible if pursued.",
      "Theoretical vs. Empiricist: Mutual coherence predictor is tested empirically; theory is validated or falsified, not assumed."
    ],
    "critical_concerns_addressed": [
      "Contrarian's 'absorption may be a feature': RQ2 directly tests this via dose-response causality.",
      "Contrarian's 'toy-task limitation': RQ4 tests generalization to semantic features.",
      "Contrarian's 'architecture comparison is confounded': L0-matching protocol explicitly controls for sparsity.",
      "Empiricist's 'causal evidence needed': Dose-response design with two independent manipulations (architectural + sparsity-induced).",
      "Empiricist's 'seed independence': 5 seeds minimum, mean +/- std reported.",
      "Theoretical's 'no predictive model': Mutual coherence predictor (H3a-b) provides testable prediction."
    ]
  }
}


## 上一轮 validation 决策意见
# Idea Validation Decision

## Pilot Evidence Summary

### RQ1: Cross-Architecture Comparison (Pilot + Full, 5 seeds each)

| Variant | Absorption Rate (mean ± std) | L0 Sparsity | vs Baseline Delta |
|---------|------------------------------|-------------|-------------------|
| Baseline ReLU | 0.254 ± 0.047 | 964 | — |
| TopK (k=50) | 0.056 ± 0.021 | 50.0 | -0.198 (-78%) |
| Matryoshka | 0.057 ± 0.023 | 50.0 | -0.197 (-77%) |
| Orthogonality SAE | 0.247 ± 0.048 | 550 | -0.007 (-3%) |
| Gated SAE | 0.257 ± 0.052 | 962 | +0.003 (+1%) |
| Random Control | 0.495 ± 0.035 | 1030 | +0.241 (+95%) |

**Key observation**: Absorption spans ~10x from TopK (0.056) to Random (0.495), exceeding the >2x go criteria. However, **this is NOT an L0-matched comparison** — TopK and Matryoshka run at L0=50 while Baseline runs at L0=964. The apparent architectural advantage is confounded by sparsity.

### RQ2: Dose-Response Causality (Full, 5 seeds x 5 lambda levels)

| Lambda_L1 | Absorption Range | feature_recovery_mcc |
|-----------|------------------|----------------------|
| 5e-05 | 0.146 – 0.238 | 0.220 – 0.222 |
| 0.0002 | 0.141 – 0.256 | 0.219 – 0.221 |
| 0.0005 | 0.147 – 0.258 | 0.219 – 0.221 |
| 0.001 | 0.153 – 0.289 | 0.218 – 0.220 |
| 0.002 | 0.176 – 0.319 | 0.217 – 0.219 |

**Critical finding**: Absorption varies ~2.3x across lambda levels (0.141 to 0.319), but **feature_recovery_mcc is essentially flat** (~0.22 with std ~0.001). There is NO monotonic trend, NO dose-response relationship, and NO correlation between absorption and downstream feature recovery.

### RQ3: Mutual Coherence (Pilot only)

- Only Baseline SAE mutual coherence computed: mu_max = 0.305, mu_mean = 0.0499
- No cross-variant correlation possible — insufficient data to test H3a or H3b

### RQ4: Semantic Generalization (Pilot only)

- Single semantic absorption rate computed: 0.204 (baseline)
- No cross-category comparison, no correlation with first-letter absorption
- H4a/b remain untested

### Ablation Results (Full, 5 seeds)

| Ablation | Absorption (mean) | Notes |
|----------|-------------------|-------|
| Matryoshka flat (no nesting) | 0.056 | Same as nested — nesting is not the driver |
| OrtSAE without penalty | 0.230 | Same as with penalty — orthogonality lambda had no effect |
| TopK as ReLU+L1 | 0.180 | Lower than Baseline but higher than TopK — explicit k matters |

---

## Decision Matrix

### Candidate A: Causal Link Between Feature Absorption and Downstream Interpretability Failure

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | Absorption varies 10x across architectures (strong signal for RQ1), but causal hypothesis H2 is falsified — no correlation with downstream metrics |
| Hypothesis survival | 0.25 | 2 | H1a partially supported (Matryoshka lower but not L0-matched), H1b supported, H1c falsified (TopK LOWER not higher), H2a/b FALSIFIED, H3/H4 untested |
| Path to full result | 0.20 | 3 | Negative result is publishable per pre-registered risk plan, but L0-confound undermines RQ1 claims; needs proper L0-matching to salvage |
| Novelty | 0.15 | 4 | Medium-high novelty per novelty assessment; negative causal result is genuinely novel and important for the field |
| Resource efficiency | 0.10 | 3 | ~3 GPU hours spent; need additional L0-matched experiments (~30 min) to fix confound |
| **Weighted Score** | **1.00** | **2.90** | |

### Candidate B: Temporal Dynamics of Absorption Emergence

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data collected; no evidence |
| Hypothesis survival | 0.25 | 1 | No hypotheses tested |
| Path to full result | 0.20 | 2 | Would require new experiments from scratch; estimated 4 GPU hours |
| Novelty | 0.15 | 3 | Medium novelty; partially covered by Li & Ren (2025) |
| Resource efficiency | 0.10 | 2 | High cost, no existing data to build on |
| **Weighted Score** | **1.00** | **1.55** | |

### Candidate C: Cross-Layer Absorption Propagation

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot data collected |
| Hypothesis survival | 0.25 | 1 | No hypotheses tested |
| Path to full result | 0.20 | 2 | Would require pretrained multi-layer SAEs; estimated 2 GPU hours |
| Novelty | 0.15 | 4 | High novelty; no prior work identified |
| Resource efficiency | 0.10 | 2 | New direction, no leverage from existing experiments |
| **Weighted Score** | **1.00** | **1.65** | |

### Candidate D: Absorption as Compositional Semantics

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No direct pilot, but the null causal result (H2 falsified) is indirect support — if absorption doesn't harm downstream, compositional recovery is plausible |
| Hypothesis survival | 0.25 | 2 | No hypotheses directly tested, but pivot trigger (RQ2 null) is met |
| Path to full result | 0.20 | 2 | Would require new experiments testing compositional recoverability; estimated 2 GPU hours |
| Novelty | 0.15 | 4 | High novelty; directly challenges established framing |
| Resource efficiency | 0.10 | 2 | New direction, limited leverage from existing data |
| **Weighted Score** | **1.00** | **2.05** | |

---

## Decision Rationale

**Candidate A scores 2.90 — REFINE.**

The pilot and full experiments produced a **strong negative result** that is itself scientifically valuable: **absorption rate does not predict downstream interpretability failure** (feature recovery MCC is flat across a 2.3x absorption range). This directly addresses the most important unanswered question in the field and validates the contrarian challenge.

However, the experiments also exposed **critical methodology issues** that must be fixed before the paper is defensible:

1. **L0-confound in RQ1**: The apparent architectural advantage of TopK/Matryoshka (absorption ~0.056 vs Baseline ~0.254) is confounded by L0 sparsity (50 vs 964). This is not a fair comparison. The pre-registered L0-matching protocol was not executed.

2. **H1c falsified**: TopK shows LOWER absorption than Baseline, opposite of the SAEBench-based prediction. This may be because the prediction applies at matched L0, which we did not achieve.

3. **H3/H4 untested**: Mutual coherence and semantic generalization lack sufficient data.

The pre-registered decision rules state: "H2a OR H2b null → PIVOT to Alternative D." However, the proposal **explicitly anticipated this outcome** in the risk assessment: "Frame as important negative result. Contrarian was right. Still publishable as it resolves a critical gap." The negative result was always a valid path forward.

**Why REFINE rather than PIVOT:**
- The negative result on causality is solid and important — abandoning it wastes the GPU investment
- With proper L0-matching, RQ1 can still yield a clean architectural comparison
- The paper can be reframed: "Architecture Comparison at Matched L0 + Absorption Does Not Predict Downstream Failure"
- This is a stronger, more honest paper than persisting with a falsified causal claim

**Why not ADVANCE:** The methodology issues (L0-confound) are too severe to proceed without fixing. The causal hypothesis is dead. The paper needs structural changes.

---

## Next Actions

1. **Fix L0-matching for RQ1**: Train Baseline L1 SAE with lambda sweep to achieve L0=50, enabling fair comparison with TopK/Matryoshka. Target: 1 seed pilot, then 5 seeds if signal is clear. (~15 min GPU)

2. **Reframe paper contribution**: Shift from "causal link" to:
   - (a) First L0-matched cross-architecture comparison with statistical rigor (5 seeds, effect sizes)
   - (b) First systematic test of absorption-downstream causality — **negative result**
   - (c) Implications: community focus on absorption reduction may be misdirected

3. **Drop H3 (mutual coherence) and H4 (semantic generalization)** from primary contributions — insufficient data, report as exploratory if at all.

4. **Add contrarian discussion section**: Address the "absorption as feature" perspective directly. The data support this view.

5. **Consider Candidate D integration**: If L0-matched comparison confirms TopK/Matryoshka advantage AND absorption still doesn't predict downstream failure, the compositional semantics framing (Candidate D) becomes a natural discussion point — absorbed parent features may remain recoverable through children.

6. **Do NOT pivot fully to Candidate D** — the existing data and reframed Candidate A provide a clearer, more complete paper.

---

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.55
DECISION: REFINE


## 上一轮 validation 结构化决策
{
  "decision": "REFINE",
  "selected_candidate_id": "cand_a",
  "confidence": 0.55,
  "candidate_scores": {
    "cand_a": {
      "weighted_score": 2.90,
      "verdict": "REFINE",
      "rationale": "Strong negative result on causality (H2 falsified) is valuable but L0-confound undermines RQ1. Needs proper L0-matching and paper reframing."
    },
    "cand_b": {
      "weighted_score": 1.55,
      "verdict": "DROP",
      "rationale": "No pilot data; high cost; partially covered by prior work."
    },
    "cand_c": {
      "weighted_score": 1.65,
      "verdict": "DROP",
      "rationale": "No pilot data; new direction with limited leverage from existing experiments."
    },
    "cand_d": {
      "weighted_score": 2.05,
      "verdict": "DROP",
      "rationale": "Pivot trigger met (H2 null) but insufficient evidence to justify abandoning existing data. Better integrated as discussion point in reframed Candidate A paper."
    }
  },
  "reasons": [
    "H2a/b FALSIFIED: Absorption varies 2.3x across lambda levels (0.141-0.319) but feature_recovery_mcc is flat (~0.22, std ~0.001). No dose-response relationship.",
    "L0-confound in RQ1: TopK/Matryoshka at L0=50 show absorption ~0.056 vs Baseline at L0=964 with absorption ~0.254. Not a fair comparison.",
    "H1c FALSIFIED: TopK shows LOWER absorption than Baseline, opposite of prediction. May be due to L0 mismatch.",
    "H1b SUPPORTED: Orthogonality SAE (0.247) ≈ Baseline (0.254) — no absorption benefit from orthogonality penalty.",
    "H3/H4 untested: Insufficient data for mutual coherence correlation or semantic generalization.",
    "Pre-registered risk plan anticipated this: 'Frame as important negative result. Contrarian was right. Still publishable.'",
    "Ablation: Matryoshka nesting has no effect (flat vs nested same absorption). OrtSAE penalty has no effect (with/without same). TopK explicit k-selection matters."
  ],
  "next_actions": [
    "Fix L0-matching: Train Baseline L1 with lambda sweep to achieve L0=50 for fair comparison with TopK/Matryoshka (1 seed pilot, then 5 seeds)",
    "Reframe paper contribution from 'causal link' to 'L0-matched architecture comparison + negative causal result'",
    "Drop H3 (mutual coherence) and H4 (semantic generalization) from primary contributions — report as exploratory only",
    "Add contrarian discussion: absorption may be a feature not a bug; community focus on reduction may be misdirected",
    "Integrate Candidate D (compositional semantics) as discussion point if L0-matched results confirm pattern",
    "Do NOT fully pivot to Candidate D — existing data supports reframed paper"
  ],
  "dropped_candidates": ["cand_b", "cand_c", "cand_d"],
  "pilot_evidence": {
    "absorption_range_across_architectures": {
      "min": 0.056,
      "max": 0.495,
      "ratio": 8.8
    },
    "absorption_range_dose_response": {
      "min": 0.141,
      "max": 0.319,
      "ratio": 2.3
    },
    "feature_recovery_mcc_range_dose_response": {
      "min": 0.217,
      "max": 0.222,
      "ratio": 1.02
    },
    "h1a_status": "partially_supported_confounded",
    "h1b_status": "supported",
    "h1c_status": "falsified",
    "h2a_status": "falsified",
    "h2b_status": "falsified",
    "h2c_status": "not_tested",
    "h3a_status": "insufficient_data",
    "h3b_status": "not_tested",
    "h4a_status": "insufficient_data",
    "h4b_status": "insufficient_data"
  },
  "critical_issues": [
    "L0-matching not executed: TopK/Matryoshka at L0=50 vs Baseline at L0=964 creates sparsity confound",
    "Core causal hypothesis (H2) falsified: absorption does not predict downstream feature recovery",
    "Full experiments used 1024 features (same as pilot), not planned 16k — may limit generalizability",
    "Only Baseline mutual coherence computed — cannot test H3a cross-variant correlation"
  ],
  "positive_findings": [
    "Absorption varies 10x across architectures — metric has discrimination power (Random 0.495 vs TopK 0.056)",
    "TopK and Matryoshka consistently show low absorption (~0.05-0.06) across 5 seeds",
    "Orthogonality penalty has NO effect on absorption — important negative result for OrtSAE claims",
    "Matryoshka nesting structure has NO effect — flat dictionary performs identically",
    "The null causal result itself is a genuine contribution: answers the field's most important open question"
  ]
}
