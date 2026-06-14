

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


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Research Proposal

## Title
**Construct Validity of the SAEBench Feature Absorption Metric: Does First-Letter Absorption Predict Semantic-Hierarchy Absorption?**

## Abstract

Feature absorption is one of the most consequential failure modes identified in sparse autoencoders (SAEs). The dominant metric for measuring absorption, introduced by Chanin et al. (2024) and adopted by SAEBench (Karvonen et al., 2025), relies on first-letter classification tasks with ground-truth logistic probes. While this metric has shaped architectural development (Matryoshka SAEs, OrtSAE, HSAE), it has never been validated on real semantic hierarchies. We propose the first systematic construct-validity study of the SAEBench absorption metric, testing whether first-letter absorption scores generalize to matched-frequency semantic hierarchies drawn from WordNet. If the metric fails to generalize, it undermines a large body of follow-up work that optimizes for it. If it generalizes, the community gains confidence in a widely used benchmark. The study is entirely training-free, uses existing pretrained SAEs, and can be completed within 1 GPU-hour.

## Motivation

The SAE community has rapidly coalesced around feature absorption as a central pathology. Chanin et al. (2024) proved analytically that absorption is incentivized by sparsity loss for hierarchical features, and introduced a rigorous detection protocol based on first-letter probes. SAEBench (Karvonen et al., 2025) standardized this protocol, making it one of eight canonical SAE evaluations. Follow-up architectures—Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), Hierarchical SAEs (Zhan et al., 2026)—all report absorption reductions as primary contributions.

Yet a critical methodological question remains unasked: **Does the first-letter absorption metric actually measure a general phenomenon, or is it an artifact of a narrow, artificial task?** First-letter hierarchies (e.g., "starts with S" ⊃ "short") are convenient because they have ground-truth labels and causal ablations are tractable. But real-world semantic hierarchies (e.g., "animal" ⊃ "mammal" ⊃ "dog") differ in frequency structure, semantic coherence, and representational geometry. If first-letter absorption is uncorrelated with semantic-hierarchy absorption, then architectures optimized for the former may not improve behavior on the latter.

This question is urgent because:
1. **Benchmark validity shapes research direction.** If the metric lacks construct validity, the community may be optimizing the wrong target.
2. **Training-free constraints align perfectly.** We can answer this using existing pretrained SAEs and the SAEBench codebase.
3. **The stakes are high either way.** A positive result validates a cornerstone of the field; a negative result reveals a methodological blind spot with immediate implications for benchmark design.

## Research Questions

1. Do SAEs with low first-letter absorption rates (as measured by SAEBench) also exhibit low absorption rates on matched-frequency semantic hierarchies from WordNet?
2. Is the SAEBench absorption metric specific to hierarchical features, or does it also detect absorption-like behavior in non-hierarchical correlated features?
3. How robust is the first-letter vs. semantic-hierarchy correlation across different SAE architectures, layers, and feature-splitting thresholds?

## Hypotheses

**H1 (Construct Validity):** The Pearson correlation between first-letter absorption scores and semantic-hierarchy absorption scores across a diverse set of 6–8 SAEs will be greater than 0.6. A correlation below 0.6 would falsify the hypothesis and suggest the metric lacks construct validity as a general measure of feature absorption.

**H2 (Hierarchy Specificity):** Semantic-hierarchy absorption scores will be significantly higher than non-hierarchy correlated-feature absorption scores, indicating the metric is specific to hierarchical rather than merely correlated features.

**H3 (Robustness):** The correlation between first-letter and semantic-hierarchy absorption will be stable across τ_fs values (0.01, 0.03, 0.05) and across architectures.

## Method

### SAE Selection
Select 6–8 publicly available pretrained SAEs that span the absorption-rate spectrum:
- Matryoshka SAE (very low absorption)
- OrtSAE (low absorption)
- BatchTopK SAE (moderate absorption)
- Standard ReLU SAE (moderate-high absorption)
- TopK SAE (moderate)
- Gated SAE (moderate)
- JumpReLU SAE (moderate-high)
- Optionally a narrow SAE (high absorption / hedging)

Source: SAELens releases for GPT-2 small and Gemma-2-2B.

### First-Letter Absorption
Compute absorption scores using the standard SAEBench protocol (ground-truth logistic probes, k-sparse probing with τ_fs = 0.03, absorption formula with τ_pa = 0, τ_ps = -1).

### Semantic-Hierarchy Construction
1. Extract 8–10 parent-child pairs from WordNet where the parent is a direct hypernym of the child (e.g., animal → mammal, mammal → dog, vehicle → car).
2. Ensure tokens are single tokens in the model vocabulary.
3. **Frequency matching:** For each hierarchy, create a synthetic balanced dataset where parent and child tokens appear with equal frequency, drawn from a background corpus. This controls the frequency-confound identified by the empiricist and contrarian perspectives.

### Semantic-Hierarchy Absorption Measurement
1. For each parent concept, train a logistic regression ground-truth probe on the base model's residual-stream activations.
2. Apply the exact SAEBench absorption formula to the SAE latents, using the same k-sparse probing protocol but on the semantic parent-child pairs.
3. Compute absorption scores per parent, then average across parents.

### Control Condition (Non-Hierarchical Correlated Features)
1. Select 4–5 pairs of semantically related but non-hierarchical concepts (e.g., synonyms like "happy/joyful", or co-occurring attributes like "doctor/hospital").
2. Match frequencies and compute "absorption" scores using the same formula.
3. If the metric is hierarchy-specific, these scores should be near-zero.

### Random-SAE Control
Compute absorption on an SAE with randomly permuted decoder directions. Should yield near-zero absorption on all tasks; if not, the metric is not specific to learned structure.

## Evaluation Protocol

**Primary benchmarks:**
- SAEBench Feature Absorption (first-letter) — established public benchmark.
- Custom Semantic-Hierarchy Absorption — novel construct-validity test.
- Custom Non-Hierarchy Correlated-Feature Absorption — control benchmark.

**Metrics:**
- Mean absorption score (first-letter)
- Mean absorption score (semantic hierarchy)
- Mean absorption score (non-hierarchy control)
- Pearson correlation r (first-letter vs. semantic hierarchy) with bootstrap 95% CI (B = 10,000).
- Pearson correlation r (first-letter vs. non-hierarchy control) with bootstrap 95% CI.
- Paired t-test comparing semantic-hierarchy absorption to non-hierarchy control absorption.

**Random seeds:** 3 random seeds for probe training and data sampling; report mean ± std.

**Statistical test plan:**
- Bootstrap CI for correlation to handle small-n SAE sampling.
- If correlation CI excludes 0.6, hypothesis is supported; if it includes values < 0.3, hypothesis is rejected.
- Report all raw scores in an appendix table for full transparency.

## Ablation Schedule

| Ablation | What It Tests | Expected Outcome |
|----------|---------------|------------------|
| **Frequency-matched vs. natural-frequency hierarchies** | Whether frequency imbalance drives absorption-like effects | Frequency-matched should show lower variance and clearer hierarchy-specificity |
| **Single-token vs. multi-token concepts** | Whether tokenization artifacts confound the metric | Single-token should be cleaner; multi-token may show inflated absorption |
| **Different base models (Gemma-2-2B vs. GPT-2 small)** | Whether construct validity is model-specific | Correlation pattern should replicate across models if the metric is general |
| **Varying τ_fs (0.01, 0.03, 0.05)** | Whether feature-splitting threshold changes the correlation | Correlation should be robust to τ_fs if the metric is stable |

## Pilot Design
- **Scope:** 2 SAEs (Matryoshka + BatchTopK) × 3 semantic parent-child pairs × 1 control pair.
- **Target runtime:** 10–15 minutes on a single GPU.
- **Success criterion:** Pilot successfully computes semantic-hierarchy absorption scores that are numerically stable and show the expected ordering (Matryoshka < BatchTopK). If scores are noisy or inverted, refine the probe training or concept selection before scaling.

## Resource Estimate
- **GPU-hours:** ~0.5–1.0 GPU-hour for the full experiment (6–8 SAEs, probe training is lightweight).
- **Model sizes:** Gemma-2-2B (primary), GPT-2 small (replication control).
- **All tasks well under the 1-hour limit.**

## Risk Assessment

| Threat | Mitigation |
|--------|------------|
| **Probe quality is poor for semantic concepts** | Filter concepts to those with probe AUROC > 0.7; report probe-quality table |
| **WordNet concepts are not single tokens** | Pre-filter using model tokenizer; exclude multi-token concepts |
| **Frequency matching is imperfect** | Use synthetic balanced datasets; report token frequencies |
| **Small-n correlation is noisy** | Report bootstrap CIs; interpret cautiously; emphasize effect-size bounds |
| **First-letter and semantic tasks use different probe difficulties** | Standardize absorption scores by probe AUROC before correlating |

## Novelty Assessment

This would be the **first systematic construct-validity study of the dominant feature-absorption metric in the SAE literature**. We searched arXiv and Google Scholar for papers extending the Chanin/SAEBench absorption metric beyond first-letter tasks to semantic hierarchies. Key findings:

- **Chanin et al. (2024)** explicitly note the limitation: *"Our metric cannot capture absorption past layer 17... [and] requires ground-truth knowledge of true labels."* They call for *"finding examples of feature absorption unrelated to character identification"* as future work.
- **SAEBench (2025)** adapted the metric technically by replacing ablation with probe-projection criteria, enabling all-layer evaluation. However, the underlying evaluation task remains **first-letter classification**.
- **Hierarchical SAEs (2025–2026)** and **Matryoshka SAEs (2025)** report absorption improvements, but all use the first-letter benchmark.
- **RAVEL (2024)** evaluates factual disentanglement (country/continent) but is not used as an absorption metric.

**No prior work has systematically tested whether first-letter absorption predicts semantic-hierarchy absorption.** This makes our proposal both novel and timely.

## Revisions from Prior Feedback

This is the first synthesis round for this project; no prior proposal, novelty report, or Codex feedback exists to revise from. The proposal synthesizes six fresh perspectives generated in the current `idea_debate` stage.

## Synthesis Rationale

### How the Perspectives Were Weighted

**Highest weight: Empiricist + Contrarian.** The empiricist identified the core methodological gap (nobody has validated the absorption metric on real hierarchies) and designed a rigorous, falsifiable experiment. The contrarian reinforced this by challenging the assumption that lower absorption scores mean better features, and by highlighting the metric's blind spots (conservative underestimate, inapplicability past layer 17, ground-truth scarcity).

**Strong weight: Pragmatist.** The pragmatist confirmed that SAEBench and SAELens provide ready-to-use infrastructure, making the experiment feasible within tight constraints. The screening-tool idea (FastProbe-Absorb) is folded in as a potential follow-up rather than the main contribution.

**Moderate weight: Theoretical + Interdisciplinary.** Both perspectives provide conceptual framing: absorption is structurally inevitable under sparsity (theoretical), and it can be understood as a phase-transition phenomenon (interdisciplinary). These insights strengthen the introduction and motivation but are not the empirical core of the proposal.

**Lower weight: Innovator.** The innovator's cross-layer dose-response framework is ambitious and valuable, but its scope (25–30 SAE cohort + causal validation + cross-layer modeling) exceeds the project's 1-hour-per-experiment constraint. Its core insight—distinguishing "truly missing" from "merely hidden" absorbed features—is retained as a conceptual thread but not as the primary methodology.

### Why This Idea Was Selected

The empiricist's front-runner was selected because it:
1. **Addresses a genuine field-wide blind spot** with high stakes for both positive and negative results.
2. **Is perfectly aligned with training-free constraints** and existing infrastructure.
3. **Has clear falsification criteria** and a tight statistical protocol.
4. **Can be completed within 1 GPU-hour**, making it ideal for rapid iteration.
5. **Generates actionable implications**: either validate a cornerstone benchmark or reveal that architectures have been optimizing the wrong target.

### What Was Dropped or Deferred

- **Innovator's dose-response framework:** Deferred to a future, larger-scale follow-up study.
- **Pragmatist's FastProbe-Absorb tool:** Retained as a backup idea and potential follow-up contribution.
- **Contrarian's label-free geometric proxy:** Interesting but requires substantial validation; deferred.
- **Theoretical's combinatorial bound:** A strong standalone theory contribution; retained as Backup Idea B.
- **Interdisciplinary's phase-transition experiment:** Requires training SAEs with varying λ; deferred due to training-free constraint.

## Expected Contributions

1. **First construct-validity test** of the dominant SAE absorption metric on real semantic hierarchies.
2. **Empirical evidence** on whether first-letter absorption generalizes, with direct implications for benchmark design.
3. **Open-source replication materials** (WordNet hierarchy dataset, frequency-matching protocol, evaluation code).
4. **Guidance for the community** on whether absorption-reducing architectures should be adopted as default.


## 当前可检验假设
# Testable Hypotheses

## Primary Hypothesis (H1): Construct Validity

**Statement:** SAEs with low first-letter absorption rates (as measured by SAEBench) will also exhibit low absorption rates on matched-frequency semantic hierarchies drawn from WordNet. The Pearson correlation between first-letter absorption scores and semantic-hierarchy absorption scores across a diverse set of 6–8 SAEs will be greater than 0.6.

**Falsification criterion:** If the Pearson correlation is below 0.6 (or non-significant at p > 0.05), the hypothesis is falsified. This would imply that optimizing first-letter absorption may not improve SAE behavior on real-world hierarchical features.

**Expected outcome if true:** r > 0.6 with a 95% bootstrap CI that excludes 0.
**Expected outcome if false:** r < 0.6 or CI includes 0.

---

## Secondary Hypothesis (H2): Hierarchy Specificity

**Statement:** The SAEBench absorption metric is specific to hierarchical features rather than merely detecting correlated-feature co-occurrence. Semantic-hierarchy absorption scores will be significantly higher than non-hierarchy correlated-feature absorption scores.

**Falsification criterion:** If the paired t-test between semantic-hierarchy and non-hierarchy control absorption scores is non-significant (p > 0.05), the hypothesis is falsified. This would suggest the metric detects general correlation rather than hierarchy.

**Expected outcome if true:** Semantic-hierarchy absorption > non-hierarchy control absorption, p < 0.05.
**Expected outcome if false:** No significant difference between the two conditions.

---

## Tertiary Hypothesis (H3): Robustness Across Thresholds

**Statement:** The correlation between first-letter and semantic-hierarchy absorption is robust to the choice of feature-splitting threshold τ_fs.

**Falsification criterion:** If the correlation sign flips or drops below 0.3 when τ_fs is varied from 0.01 to 0.05, the hypothesis is falsified.

**Expected outcome if true:** r remains positive and > 0.5 across τ_fs ∈ {0.01, 0.03, 0.05}.
**Expected outcome if false:** r becomes negative or near-zero at one or more threshold values.

---

## Exploratory Hypothesis (H4): Model Generalization

**Statement:** The correlation pattern between first-letter and semantic-hierarchy absorption replicates across different base models (Gemma-2-2B and GPT-2 small).

**Falsification criterion:** If the correlation signs differ between models, or if one model shows r > 0.6 while the other shows r < 0.3, the hypothesis is falsified.

**Expected outcome if true:** Both models show positive correlations in the same direction, with overlapping bootstrap CIs.
**Expected outcome if false:** Divergent or opposite correlation patterns across models.

---

## Exploratory Hypothesis (H5): Architecture Ordering

**Statement:** The relative ranking of SAEs by absorption rate is preserved across first-letter and semantic-hierarchy tasks. Low-absorption architectures (Matryoshka, OrtSAE) remain low-absorption on semantic hierarchies, and high-absorption architectures (Standard ReLU, JumpReLU) remain high-absorption.

**Falsification criterion:** If the architecture ordering by absorption rate inverts between first-letter and semantic-hierarchy tasks (e.g., Matryoshka shows higher semantic-hierarchy absorption than Standard ReLU), the hypothesis is falsified.

**Expected outcome if true:** Kendall's τ_rank > 0.5 between first-letter and semantic-hierarchy architecture rankings.
**Expected outcome if false:** τ_rank ≤ 0 or statistically indistinguishable from 0.


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_a",
      "title": "Construct Validity of the SAEBench Feature Absorption Metric: Does First-Letter Absorption Predict Semantic-Hierarchy Absorption?",
      "status": "front_runner",
      "summary": "Systematically test whether the dominant first-letter absorption metric generalizes to matched-frequency semantic hierarchies from WordNet. A training-free study using existing pretrained SAEs and SAEBench infrastructure.",
      "hypotheses": [
        "H1: Pearson correlation between first-letter and semantic-hierarchy absorption > 0.6",
        "H2: Semantic-hierarchy absorption > non-hierarchy control absorption (hierarchy specificity)",
        "H3: Correlation is robust across feature-splitting thresholds (tau_fs = 0.01, 0.03, 0.05)"
      ],
      "pilot_focus": "2 SAEs (Matryoshka + BatchTopK) x 3 WordNet parent-child pairs x 1 control pair; target 10-15 min on 1 GPU"
    },
    {
      "candidate_id": "cand_b",
      "title": "FastProbe-Absorb: Automated Probe-Based Screening for Feature Absorption",
      "status": "backup",
      "summary": "Develop a lightweight, automated probe-projection screening method that detects absorption-like behavior in seconds per SAE, validated against SAEBench gold-standard scores.",
      "hypotheses": [
        "FastProbe-Absorb scores correlate with SAEBench absorption scores at r > 0.7",
        "Runtime is < 1 minute per SAE",
        "Flagged latents show worse downstream sparse-probing performance than unflagged latents"
      ],
      "pilot_focus": "5 GPT-2 Small SAEs x 5 simple probes; validate against SAEBench on 1-2 SAEs; target 30-45 min"
    },
    {
      "candidate_id": "cand_c",
      "title": "The Rate-Distortion Origin of Feature Absorption: A Combinatorial Bound",
      "status": "backup",
      "summary": "Prove a general theorem that absorption is inevitable under sparsity for tree-structured hierarchies, with explicit depth bound. Validate on synthetic hierarchical data.",
      "hypotheses": [
        "Absorption rate increases monotonically with sparsity penalty lambda",
        "Deeper tree nodes are absorbed before shallower nodes",
        "Real SAEs with higher L0 (lower sparsity) show lower absorption rates"
      ],
      "pilot_focus": "Synthetic 3-level binary tree (7 features) x 3 lambda values; target 15 min"
    },
    {
      "candidate_id": "cand_d",
      "title": "Rethinking Feature Absorption: A Validity-Aware Analysis of SAE Architectures",
      "status": "backup",
      "summary": "Challenge the assumption that reducing absorption scores improves feature quality. Test low-absorption architectures (Matryoshka, OrtSAE) against random-baseline-corrected metrics and causal actionability proxies.",
      "hypotheses": [
        "Low-absorption architectures do not outperform high-absorption ones on random-baseline margins",
        "Activation patching shows null or weak causal actionability for all architectures regardless of absorption rate",
        "A label-free geometric absorption proxy reveals higher absorption than the Chanin metric in deeper layers"
      ],
      "pilot_focus": "3 architectures (Matryoshka, BatchTopK, Standard) x random-baseline comparison x 50 activation-patching prompts; target 45 min"
    }
  ],
  "synthesis_metadata": {
    "front_runner_selected_from": "empiricist_phase_5_candidate_c",
    "perspectives_weighted_highest": ["empiricist", "contrarian"],
    "perspectives_weighted_strong": ["pragmatist"],
    "perspectives_weighted_moderate": ["theoretical", "interdisciplinary"],
    "perspectives_weighted_lower": ["innovator"],
    "dropped_or_deferred": [
      "innovator_cross_layer_dose_response (scope too large for 1-hour constraint)",
      "interdisciplinary_phase_transition_experiment (requires training SAEs with varying lambda)",
      "pragmatist_fastprobe_as_main (folded into backup A)"
    ],
    "novelty_verification_summary": "No prior work systematically tests whether first-letter absorption generalizes to semantic hierarchies. SAEBench adapted the metric technically (probe projection for all layers) but retained the first-letter task. RAVEL evaluates factual disentanglement but is not used as an absorption metric."
  }
}


## 上一轮 validation 决策意见
# Idea Validation Decision

## Pilot Evidence Summary

The pilot (`pilot_semantic_probe`) on 2 SAEs (MatryoshkaBatchTopK and TopK) with 3 WordNet hierarchies and 2 control pairs completed successfully and met all pass criteria:
- Numerically stable absorption scores
- All probe AUROCs = 1.0 (> 0.6 threshold)
- Expected ordering: Matryoshka (0.283) < TopK (0.339)
- Hierarchy > control for at least 1 SAE: True

However, the **full experimental suite** revealed critical anomalies that undermine confidence in the current implementation:

1. **Random-SAE Red Flag**: The Random SAE's semantic-hierarchy absorption (0.352) is *identical* to the Standard SAE's score (0.352) across all 10 hierarchies. This is statistically implausible and strongly suggests a data-handling bug or unintended sharing of results between the two configurations.

2. **H2 Reversed**: The paired t-test shows semantic-hierarchy absorption (mean = 0.235) is *significantly lower* than non-hierarchy control absorption (mean = 0.331), t = -4.748, p = 0.0032. This is the opposite of the hypothesized direction and raises serious questions about whether the custom pipeline is measuring the intended construct.

3. **H1 Inconclusive**: Pearson r between first-letter and semantic-hierarchy absorption = 0.463 (95% CI [-0.389, 0.981]), failing to reach the > 0.6 threshold and spanning zero.

4. **GPT-2 Replication Divergence**: GPT-2 small showed near-zero hierarchy absorption (Standard = 0.0, TopK = 0.003) versus substantial values on Pythia-160M. This model-specific divergence is unexpected and warrants methodological scrutiny.

## Decision Matrix

### Candidate A: Construct Validity of SAEBench Absorption Metric

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | Pilot passed, but full results show severe anomalies (Random = Standard, H2 reversed) |
| Hypothesis survival | 0.25 | 2 | H1 inconclusive, H2 rejected (reversed), H3 inconclusive |
| Path to full result | 0.20 | 2 | Major implementation red flags; cannot trust current pipeline without debugging |
| Novelty | 0.15 | 4 | First systematic construct-validity study of the dominant absorption metric |
| Resource efficiency | 0.10 | 2 | ~1 GPU-hour already spent; continuing on a buggy pipeline would be wasteful |

**Weighted Score**: 0.30×3 + 0.25×2 + 0.20×2 + 0.15×4 + 0.10×2 = **2.60**

### Candidate B: FastProbe-Absorb

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | No pilot run; feasibility inferred from existing infrastructure |
| Hypothesis survival | 0.25 | 3 | Hypotheses are plausible but untested |
| Path to full result | 0.20 | 3 | Could leverage SAEBench codebase, but scope is moderate |
| Novelty | 0.15 | 3 | Useful tool, but less novel than a construct-validity study |
| Resource efficiency | 0.10 | 4 | Likely 30-45 min pilot; well within budget |

**Weighted Score**: 0.30×3 + 0.25×3 + 0.20×3 + 0.15×3 + 0.10×4 = **3.10**

### Candidate C: Rate-Distortion Origin (Theory)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No empirical pilot; theoretical only |
| Hypothesis survival | 0.25 | 3 | Structurally plausible but no data yet |
| Path to full result | 0.20 | 2 | Proving a general theorem is high-risk within current constraints |
| Novelty | 0.15 | 4 | Strong standalone theory contribution |
| Resource efficiency | 0.10 | 3 | Low GPU cost, but high intellectual risk |

**Weighted Score**: 0.30×2 + 0.25×3 + 0.20×2 + 0.15×4 + 0.10×3 = **2.55**

### Candidate D: Validity-Aware Analysis (Contrarian)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No pilot run; would require causal patching infrastructure |
| Hypothesis survival | 0.25 | 2 | Ambitious and potentially confounded |
| Path to full result | 0.20 | 2 | 45 min pilot + activation patching is nontrivial |
| Novelty | 0.15 | 3 | Challenging assumptions is valuable but not unprecedented |
| Resource efficiency | 0.10 | 2 | Higher resource cost for uncertain payoff |

**Weighted Score**: 0.30×2 + 0.25×2 + 0.20×2 + 0.15×3 + 0.10×2 = **2.15**

## Decision Rationale

No candidate scores ≥ 3.5, so **ADVANCE is ruled out**. The front-runner (Candidate A) scores 2.60, placing it squarely in the **REFINE** zone. However, the rationale for refinement is not "minor methodological tweaks" — it is **fundamental pipeline debugging**.

The specific evidence triggering REFINE rather than PIVOT:
- The pilot *did* produce numerically stable, correctly ordered results.
- The research question remains important and novel.
- The anomalies in the full experiment are severe enough that they likely stem from **implementation bugs rather than a fundamentally flawed idea**. A Random SAE cannot plausibly produce *exactly* the same scores as a Standard SAE on 10 hierarchies unless there is a data-handling error.

If the bugs are fixed and the results still show no correlation or reversed hierarchy specificity, then a **PIVOT** would be warranted in the next validation round.

## Next Actions

1. **Debug the custom absorption pipeline**: Investigate why Random SAE and Standard SAE produce identical semantic-hierarchy and non-hierarchy control scores. Check for cached results, shared file paths, or incorrect SAE loading logic.
2. **Re-run the full semantic-hierarchy and non-hierarchy experiments** on the corrected pipeline for all 8 SAEs.
3. **Re-run the GPT-2 replication** with the fixed code to verify cross-model consistency.
4. **Re-compute statistical analysis** only after the above anomalies are resolved.
5. If the fixed pipeline still fails H1/H2, pivot to **Candidate B (FastProbe-Absorb)** as the backup direction.

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.40
DECISION: REFINE


## 上一轮 validation 结构化决策
{
  "decision": "REFINE",
  "selected_candidate_id": "cand_a",
  "confidence": 0.40,
  "candidate_scores": {
    "cand_a": {
      "weighted_score": 2.60,
      "verdict": "REFINE"
    },
    "cand_b": {
      "weighted_score": 3.10,
      "verdict": "REFINE"
    },
    "cand_c": {
      "weighted_score": 2.55,
      "verdict": "REFINE"
    },
    "cand_d": {
      "weighted_score": 2.15,
      "verdict": "PIVOT"
    }
  },
  "reasons": [
    "Pilot passed with stable scores and correct ordering (Matryoshka 0.283 < TopK 0.339)",
    "Full experiment reveals critical anomaly: Random SAE semantic-hierarchy absorption (0.352) is identical to Standard SAE, indicating a likely data-handling bug",
    "H2 (hierarchy specificity) is rejected and reversed: semantic-hierarchy absorption (0.235) is significantly lower than non-hierarchy control (0.331), t = -4.748, p = 0.0032",
    "H1 (construct validity) is inconclusive: Pearson r = 0.463, 95% CI [-0.389, 0.981], failing the > 0.6 threshold",
    "GPT-2 replication diverges sharply from Pythia (near-zero vs. substantial absorption), suggesting pipeline or model-specific issues",
    "No candidate scores >= 3.5; front-runner at 2.60 falls in REFINE zone"
  ],
  "next_actions": [
    "Debug custom absorption pipeline: investigate why Random and Standard SAEs produce identical scores across all hierarchies",
    "Fix any cached-result or SAE-loading bugs in the semantic-hierarchy and non-hierarchy evaluation scripts",
    "Re-run full semantic-hierarchy and non-hierarchy experiments on corrected pipeline for all 8 SAEs",
    "Re-run GPT-2 replication with fixed code",
    "Re-compute statistical analysis only after anomalies are resolved",
    "If fixed pipeline still fails H1/H2, pivot to Candidate B (FastProbe-Absorb)"
  ],
  "dropped_candidates": ["cand_d"],
  "critical_red_flags": [
    "Random_SAE_equals_Standard_SAE_on_all_hierarchies",
    "H2_reversed_semantic_lower_than_control",
    "GPT2_Pythia_divergence"
  ]
}
