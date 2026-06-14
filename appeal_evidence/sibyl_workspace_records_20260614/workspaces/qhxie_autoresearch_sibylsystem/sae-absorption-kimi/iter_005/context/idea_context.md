

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
**Survey Date**: 2026-04-16
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

- **Absorption Evaluation Protocol** (from SAEBench, based on Chanin et al. 2024):
  - Train ground-truth probes for token properties (e.g., starting letter).
  - Use k-sparse probing (k=1..10) to identify main latents for each property.
  - Detect feature splitting via F1 threshold tau_fs = 0.03.
  - Compute absorption score as the fraction of ground-truth probe projection captured by "absorbing" latents vs. "main" latents.
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

---

## 4. Identified Research Gaps

- **Gap 1: Theoretical understanding of absorption dynamics.** While Chanin et al. (2024) identified absorption and Bussmann et al. (2025) mitigated it, there is no unified theoretical framework predicting *when* absorption will occur for a given feature hierarchy, SAE width, and sparsity level.

- **Gap 2: Scalable causal validation of absorbed features.** Existing metrics are correlational (probe-based). There is limited work on causal interventions that definitively establish whether a latent "knows about" vs. "uses" a parent feature, especially under absorption.

- **Gap 3: Cross-architecture absorption patterns.** Most absorption studies focus on residual-stream SAEs. How absorption manifests in attention-output SAEs, MLP SAEs, or multimodal (vision-language) SAEs remains underexplored.

- **Gap 4: Training dynamics and emergence time.** It is unclear at what point during SAE training absorption emerges, whether it is reversible, and how curriculum learning or temporal masking might prevent it.

- **Gap 5: Unified objective trade-offs.** No single training objective dominates across all interpretability goals. Methods that reduce absorption (Matryoshka, OrtSAE) introduce new pathologies (hedging, overhead) or trade off reconstruction fidelity.

- **Gap 6: Real-world downstream impact.** Kantamneni et al. (2025) showed SAEs often fail to outperform baselines on sparse probing. The causal link between absorption rates and downstream task underperformance has not been systematically quantified.

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
