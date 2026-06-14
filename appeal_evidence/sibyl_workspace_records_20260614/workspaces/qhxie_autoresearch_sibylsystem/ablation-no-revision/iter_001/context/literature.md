# Literature Survey Report

**Research Topic**: Feature Absorption in Sparse Autoencoders (SAEs)
**Survey Date**: 2026-04-29
**arXiv Search Keywords**: sparse autoencoder feature absorption, SAE hierarchical features, SAE interpretability failure modes, sparse autoencoder superposition, mechanistic interpretability SAE
**Web Search Keywords**: SAELens GemmaScope feature absorption, sparse autoencoder open source 2025, SAE circuit tracing steering 2025, SAEBench evaluation benchmark

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the primary tool for mechanistic interpretability of large language models, enabling researchers to decompose neural network activations into human-interpretable features. Since the foundational work on superposition (Elhage et al., 2022), the field has rapidly evolved from theoretical toy models to large-scale deployed systems like GemmaScope and SAELens.

Current research focuses on three interconnected challenges:

### 1.1 SAE Training and Architecture
- **Standard SAEs** (TopK, Gated, JumpReLU) optimize reconstruction with sparsity constraints
- **GemmaScope** (Lieberum et al., 2024) and **GemmaScope 2** (2025) provide open SAEs for Gemma 2/3 at all scales (270M to 27B)
- **SAELens** (Bloom et al.) is the de facto standard open-source library for training and analyzing SAEs
- New architectures in 2025 include **Matryoshka SAEs**, **Hierarchical SAEs (HSAE)**, **Orthogonal SAEs (OrtSAE)**, and **AdaptiveK SAEs**

### 1.2 Feature Quality and Failure Modes
- **Feature absorption** (Chanin et al., 2024): hierarchical parent features get silently absorbed into child features due to sparsity pressure
- **Feature hedging** (Chanin et al., 2025): correlated features break narrow SAEs through reconstruction-driven noise
- **Polysemanticity** and **superposition** remain fundamental theoretical frameworks
- **SAEBench** (Karvonen et al., 2025) provides standardized evaluation including absorption metrics

### 1.3 Downstream Applications
- **Circuit discovery**: SAE features as nodes in sparse computation graphs (Belinkov et al., ICLR 2025)
- **Feature steering**: Causal intervention via SAE latents (EMNLP 2025; CorrSteer, 2025)
- **Model verification**: Committed SAE-feature traces for detecting model substitution (2025)
- **Cross-layer tracing**: Cross-Layer Transcoders (CLTs) for multi-step computation analysis

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 | 2024 | First systematic study of feature absorption; proves sparsity objective encourages absorption; introduces quantification metrics | Limited to first-letter classification task; no downstream impact analysis |
| 2 | Toy Models of Superposition | Elhage et al., Transformer Circuits | 2022 | Foundational theory: neural networks represent more features than dimensions | Toy model only; real LLM behavior more complex |
| 3 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | arXiv:2408.05147 | 2024 | Large-scale open SAE suite for Gemma 2; JumpReLU architecture | No absorption-specific analysis |
| 4 | Gemma Scope 2: Suite of SAEs and Transcoders for Gemma 3 | Technical Paper / Neuronpedia | 2025 | Expanded to Gemma 3 (270M-27B); CLTs, crosscoders, affine skip connections | Focus on coverage, not absorption mitigation |
| 5 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders | arXiv:2503.09532 | 2025 | Standardized evaluation: absorption, SCR, TPP, sparse probing, RAVEL | Metrics may not capture all absorption variants |
| 6 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | arXiv:2505.11756 | 2025 | Identifies hedging (MSE-driven) vs absorption (sparsity-driven); compound multiplier analysis | Narrow SAE focus; different mechanism from absorption |
| 7 | Building a Structured Feature Forest with Hierarchical Sparse Autoencoders | arXiv:2602.11881 | 2026 | HSAE with explicit parent-child trees; reduces absorption at larger dict sizes | Requires training new SAEs; not applicable to pretrained |
| 8 | Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | OrtSAE reduces absorption by 0.17 at L0=70 vs BatchTopK; MetaSAE for atomicity measurement | Orthogonality constraints may limit expressiveness |
| 9 | From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit | arXiv:2506.03093 | 2025 | Hierarchical matching pursuit for feature extraction | Different paradigm from standard SAEs |
| 10 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | arXiv:2506.01197 | 2025 | Explicit hierarchical semantics in SAE design | Architectural change, not analysis |
| 11 | Train One Sparse Autoencoder Across Multiple Sparsity Budgets | arXiv:2505.24473 | 2025 | HierarchicalTopK: single SAE with progressive sparsity; mirrors feature hierarchies | Training-based, not analysis of existing SAEs |
| 12 | Matryoshka SAE / Interpreting CLIP with Hierarchical Sparse Autoencoders | ICML 2025 | 2025 | Nested hierarchical groups; SOTA Pareto frontier | CLIP focus; different domain |
| 13 | Sparse Feature Circuits: Discovering... | Belinkov et al., ICLR 2025 | 2025 | SAE-based circuit discovery via indirect effects; attribution patching through SAE features | Absorption may create missing circuit nodes |
| 14 | SAEs Are Good for Steering -- If You Select the Right Features | EMNLP 2025 | 2025 | Steering requires output-score filtering; unfiltered SAEs perform poorly (0.191 vs 0.546) | Absorbed features would have misleading S_out |
| 15 | CorrSteer: Generation-Time LLM Steering via Correlated SAE Features | arXiv:2508.12535 | 2025 | Correlation-based feature selection + causal intervention validation | Absorption creates spurious correlations |
| 16 | Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? | arXiv:2602.14111 | 2026 | SAEs show poor generalization; absorption and hedging are "corrupted" learning patterns | Critical assessment of SAE reliability |
| 17 | Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data | arXiv:2602.14687 | 2026 | Realistic synthetic data for SAE evaluation; scalable benchmark generation | Synthetic data may not capture real LLM behavior |
| 18 | A Unified Theory of Sparse Dictionary Learning in MI | arXiv:2512.05534 | 2025 | Piecewise biconvexity; spurious minima encode absorption/hedging/dead neurons | Theoretical; limited practical guidance |
| 19 | Pando: Do Interpretability Methods Work When Models Won't Explain Themselves? | arXiv:2604.11061 | 2026 | Evaluates sae_gradient vs circuit_tracer with GemmaScope transcoders | Focus on method comparison, not absorption |
| 20 | Committed SAE-Feature Traces for Audited-Session Substitution Detection | arXiv:2604.18179 | 2025 | SAE-based model verification; 96 probes across 8 circuit classes | Security application; not absorption analysis |

## 3. SOTA Methods and Benchmarks

### Benchmarks

| Benchmark | Source | Focus | Relevance |
|-----------|--------|-------|-----------|
| **SAEBench** | Karvonen et al., 2025 | Comprehensive SAE evaluation: absorption, SCR, TPP, sparse probing, RAVEL | Directly applicable; includes absorption metric |
| **First-Letter Classification** | Chanin et al., 2024 | Primary absorption detection task | Established methodology |
| **CE-Bench** | arXiv:2509.00691 | Contrastive evaluation of SAE interpretability | Complementary evaluation |
| **Synthetic Realistic Data** | arXiv:2602.14687 | Scalable realistic synthetic data generation | For controlled experiments |

### State-of-the-Art Methods

| Category | Methods | Performance |
|----------|---------|-------------|
| Absorption mitigation | HSAE, OrtSAE, Matryoshka SAE, GBA | 0.10-0.17 absorption reduction |
| Evaluation | SAEBench (absorption fraction, full absorption rate) | Standardized across architectures |
| Circuit discovery | Sparse Feature Circuits (Belinkov et al.) | Attribution patching through SAE features |
| Steering | CorrSteer, S_out filtering | Requires careful feature selection |

### Key Findings from Literature

1. **Absorption is fundamental**: Caused by sparsity objective when features form hierarchies; not solved by scaling SAE size or adjusting sparsity (Chanin et al., 2024)

2. **Absorption rate increases with sparsity**: Lower L0 (more sparse) pushes more absorption; observed across all tested LLM SAEs (Chanin et al., 2024; SAEBench, 2025)

3. **Hierarchical structure helps**: HSAE and Matryoshka SAEs show reduced absorption by explicitly modeling hierarchies (Luo et al., 2026; Bussmann et al., 2025)

4. **Feature hedging is distinct**: Caused by MSE loss (not sparsity), affects narrow SAEs, creates encoder/decoder asymmetry (Chanin et al., 2025)

5. **Downstream impact is underexplored**: Absorption may cause missing circuit nodes, misleading steering signals, and false confidence in safety monitoring -- but systematic quantification is lacking

## 4. Identified Research Gaps

### Gap 1: Systematic Downstream Impact Quantification
- Chanin et al. (2024) measures absorption but does NOT quantify its impact on:
  - Circuit discovery completeness (missing nodes/edges)
  - Feature steering effectiveness (misleading activation patterns)
  - Concept probing reliability (false negatives in feature detection)
- **This is a major gap**: We know absorption exists, but not how much it hurts downstream tasks

### Gap 2: Cross-Layer Absorption Propagation
- Most absorption studies focus on single-layer SAEs
- How does absorption in early layers affect downstream layer features and circuits?
- Cross-Layer Transcoders (CLTs) may reveal propagation patterns but no absorption-specific analysis exists

### Gap 3: Feature Absorption vs Model Scale
- GemmaScope 2 covers 270M to 27B, but absorption analysis has not been systematically compared across scales
- Does absorption worsen or improve with model scale?
- Different layer depths within the same model also unexplored

### Gap 4: Quantitative Impact on Interpretability Reliability
- "Sanity Checks" (2026) questions SAE generalization but does not isolate absorption's contribution
- No metric connects absorption rate to interpretability task performance degradation

### Gap 5: Training-Free Detection Beyond First-Letter Task
- Current absorption detection relies on first-letter classification (limited semantic scope)
- Need broader semantic categories (mathematical reasoning, factual knowledge, syntactic features)
- Need methods that work on pretrained SAEs without retraining

## 5. Prior Art Collision Analysis

### CRITICAL: Chanin et al. (2024) - "A is for Absorption"

**Paper**: arXiv:2409.14507, NeurIPS 2024/2025
**Authors**: David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Joseph Bloom

| Our Proposal | Chanin et al. |
|--------------|---------------|
| Feature absorption quantification | Already quantified with absorption fraction, full absorption rate |
| First-letter classification task | Primary evaluation methodology |
| Sparsity causes absorption | Proven mathematically and empirically |
| Hierarchical features most affected | Core finding |

**Collision severity**: The phenomenon itself is already discovered and named. We CANNOT claim discovery of feature absorption.

**What remains novel**:
1. Systematic downstream impact quantification (not done in Chanin et al.)
2. Cross-layer absorption propagation analysis (not done)
3. Broader semantic categories beyond first-letter (not done)
4. Connection between absorption rate and circuit/steering/probing reliability (not done)

### RELATED WORK: SAEBench (2025)
- Provides standardized absorption metrics
- Our work would USE SAEBench metrics, not compete with them

### RELATED WORK: HSAE / OrtSAE / Matryoshka SAE (2025-2026)
- These are SOLUTIONS to absorption (new architectures)
- Our work is ANALYSIS of absorption in existing pretrained SAEs
- Complementary, not competing

### RELATED WORK: Feature Hedging (2025)
- Different mechanism (MSE-driven vs sparsity-driven)
- Different effect (encoder/decoder noise vs missing activations)
- Both are failure modes but distinct phenomena

## 6. Available Resources

### Open-source Code

| Repo | URL | Description | License |
|------|-----|-------------|---------|
| **SAELens** | https://github.com/jbloomAus/SAELens | Primary SAE training/analysis library | MIT |
| **OpenAI Sparse Autoencoder** | https://github.com/openai/sparse_autoencoder | Official OpenAI implementation; Triton kernels | MIT |
| **SAEBench** | https://www.neuronpedia.org/sae-bench/info | Comprehensive SAE benchmarking | Open |
| **GemmaScope** | https://www.neuronpedia.org/gemma-scope | Pretrained SAEs for Gemma 2 | Open |
| **GemmaScope 2** | https://www.neuronpedia.org/gemma-scope-2 | Pretrained SAEs for Gemma 3 | Open |
| **MSAE** | https://github.com/WolodjaZ/MSAE | Hierarchical SAE for CLIP (ICML 2025) | Open |
| **TransformerLens** | https://github.com/neelnanda-io/TransformerLens | Mechanistic interpretability toolkit | MIT |

### Datasets

| Dataset | Source | Notes |
|---------|--------|-------|
| **GemmaScope SAEs** | Neuronpedia / HuggingFace | Pretrained SAEs for Gemma-2B, 4B, 9B, 27B |
| **SAELens pretrained SAEs** | SAELens releases | GPT2-small, various Gemma models |
| **First-letter classification data** | Chanin et al. (2024) | Established absorption detection benchmark |
| **OpenWebText / Pile** | Standard | For general activation analysis |

### Pretrained Models

| Model | Size | SAE Availability | Notes |
|-------|------|------------------|-------|
| Gemma 2 | 2B, 4B, 9B, 27B | GemmaScope | Well-studied; SAELens compatible |
| Gemma 3 | 270M-27B | GemmaScope 2 | Latest; CLTs available |
| GPT-2 | small (124M) | SAELens | Classic benchmark; smallest scale |

## 7. Implications for Idea Generation

### Directions Worth Exploring
1. **Downstream impact quantification**: Systematically measure how absorption affects circuit discovery, steering, and probing. This is the biggest gap.
2. **Cross-layer propagation**: Analyze how absorption in layer N affects features in layer N+k.
3. **Scale-dependent absorption**: Compare absorption patterns across model scales (270M to 27B).
4. **Semantic breadth**: Extend absorption detection beyond first-letter to mathematical, factual, syntactic features.
5. **Absorption-aware circuit discovery**: Modify circuit tracing methods to account for potentially absorbed features.

### Directions to Avoid
1. Claiming discovery of feature absorption (Chanin et al. already discovered it)
2. Proposing new SAE architectures (HSAE, OrtSAE already exist)
3. Generic SAE evaluation (SAEBench already exists)
4. Toy model analysis only (Elhage et al. established the theory)

### Saturated Directions
- Generic SAE training improvements (many 2025 papers)
- Confidence-based feature interpretation (well-explored)
- Simple sparsity/reconstruction trade-off analysis (standard)

## 8. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens | High | MIT | **Adopt** | De facto standard; handles loading pretrained SAEs, activation analysis, feature inspection |
| TransformerLens | High | MIT | **Adopt** | Essential for hooking into model forward passes and extracting activations |
| GemmaScope/GemmaScope 2 SAEs | High | Open | **Adopt** | Pretrained SAEs eliminate training need; compatible with SAELens |
| SAEBench metrics | High | Open | **Adopt/Extend** | Use absorption fraction metric; extend with downstream impact metrics |
| Chanin et al. first-letter code | Medium | - | **Reference** | Reference implementation for absorption detection; adapt for broader semantic categories |
| Sparse Feature Circuits (Belinkov) | Medium | - | **Reference** | Circuit discovery methodology; adapt to measure absorption impact on circuit completeness |

### Key Insights

1. **Training-free analysis is viable**: GemmaScope + SAELens + TransformerLens provide everything needed without training new SAEs
2. **Focus on impact, not detection**: Chanin et al. detects absorption; our contribution is quantifying its consequences
3. **Small models sufficient**: GPT-2-small or Gemma-2B are adequate for pilot experiments; scale up if signal is strong
4. **Established metrics exist**: SAEBench absorption fraction + Chanin et al. methodology provide solid foundation
5. **Clear novelty path**: Downstream impact quantification is genuinely unexplored

## 9. Updated Project Status

### Current State

| Aspect | Status | Notes |
|--------|--------|-------|
| Feature absorption discovery | PRIOR ART | Chanin et al. (2024) discovered and named it |
| Absorption quantification metric | AVAILABLE | SAEBench + Chanin et al. methodology |
| Downstream impact analysis | **GAP** | Not systematically studied |
| Pretrained SAEs | AVAILABLE | GemmaScope, GemmaScope 2, SAELens releases |
| Tools | AVAILABLE | SAELens, TransformerLens, Neuronpedia |
| Hardware | AVAILABLE | Local GPU with PyTorch 2.11.0 + Blackwell support |

### Recommended Next Steps

1. **Pilot: Reproduce Chanin et al. absorption detection** on Gemma-2B with SAELens (15 min)
2. **Pilot: Measure circuit completeness** with/without absorbed features (30 min)
3. **Pilot: Test steering effectiveness** on absorbed vs non-absorbed features (30 min)
4. **If signal strong**: Scale to multiple layers, model sizes, semantic categories
5. **If signal weak**: Pivot to cross-layer propagation or scale-dependent analysis

## 10. References

### Key Papers

1. Chanin et al. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507. https://arxiv.org/abs/2409.14507
2. Elhage et al. (2022). Toy Models of Superposition. Transformer Circuits Thread.
3. Lieberum et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2. arXiv:2408.05147. https://arxiv.org/abs/2408.05147
4. Gemma Scope 2 Technical Paper (2025). https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf
5. Karvonen et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders. arXiv:2503.09532. https://www.neuronpedia.org/sae-bench/info
6. Chanin et al. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756. https://arxiv.org/abs/2505.11756
7. Luo et al. (2026). Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. arXiv:2602.11881. https://arxiv.org/abs/2602.11881
8. OrtSAE (2025). Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033. https://arxiv.org/abs/2509.22033
9. Belinkov et al. (2025). Sparse Feature Circuits. ICLR 2025. https://belinkov.com/assets/pdf/iclr2025-sfc.pdf
10. SAEs Are Good for Steering (EMNLP 2025). https://aclanthology.org/2025.emnlp-main.519.pdf
11. CorrSteer (2025). Generation-Time LLM Steering via Correlated SAE Features. arXiv:2508.12535. https://arxiv.org/abs/2508.12535
12. Sanity Checks for SAEs (2026). arXiv:2602.14111. https://arxiv.org/abs/2602.14111
13. Evaluating SAEs on Synthetic Data (2026). arXiv:2602.14687. https://arxiv.org/abs/2602.14687
14. Unified Theory of Sparse Dictionary Learning (2025). arXiv:2512.05534. https://arxiv.org/abs/2512.05534
15. Pando (2026). Do Interpretability Methods Work When Models Won't Explain Themselves? arXiv:2604.11061. https://arxiv.org/abs/2604.11061
16. Committed SAE-Feature Traces (2025). arXiv:2604.18179. https://arxiv.org/abs/2604.18179

### Open Source Resources

- SAELens: https://github.com/jbloomAus/SAELens
- TransformerLens: https://github.com/neelnanda-io/TransformerLens
- Neuronpedia (GemmaScope): https://www.neuronpedia.org/gemma-scope
- Neuronpedia (GemmaScope 2): https://www.neuronpedia.org/gemma-scope-2
- SAEBench: https://www.neuronpedia.org/sae-bench/info
