# Literature Survey Report

**Research Topic**: Systematic analysis and quantification of feature absorption in Sparse Autoencoders (SAEs): causes, patterns, and impact on interpretability.

**Survey Date**: 2026-04-29

**arXiv Search Keywords**: ["sparse autoencoder feature absorption", "SAE feature splitting absorption", "SAE superposition dictionary learning", "mechanistic interpretability autoencoder", "sparse autoencoder hierarchical feature", "SAE benchmark evaluation SAEBench"]

**Web Search Keywords**: ["sparse autoencoder feature absorption phenomenon 2025 2026", "SAE superposition feature splitting absorption", "SAEBench feature absorption metric architecture comparison", "SAELens GemmaScope open source 2025", "feature sensitivity consistency sparse autoencoder 2025", "Matryoshka SAE hierarchical feature 2025", "orthogonal sparse autoencoder OrtSAE absorption 2025"]

## 1. Field Overview

Sparse Autoencoders (SAEs) have become the central tool for mechanistic interpretability of Large Language Models (LLMs), decomposing dense, polysemantic activations into human-interpretable sparse feature directions. The theoretical foundation rests on the Linear Representation Hypothesis and the phenomenon of superposition---where neural networks represent more features than dimensions by using non-orthogonal directions.

However, a fundamental limitation has emerged: **feature absorption**. When underlying features form hierarchical structures (e.g., "starts with S" and "short"), SAEs optimized for sparsity drive more specific features to "steal" the direction of more general features. This causes seemingly monosemantic latents to develop blind spots where they should activate, undermining their reliability as interpretable classifiers.

The field in 2025-2026 shows several critical trends:
- **From visualization to quantitative metrics**: The community has moved beyond maximum-activating examples toward precision/recall, absorption rate, and feature sensitivity as rigorous evaluation criteria.
- **Architecture innovation explosion**: TopK, JumpReLU, Matryoshka, HSAE, OrtSAE, and AdaptiveK SAEs represent a rapid evolution of architectures targeting absorption and related pathologies.
- **Benchmark maturation**: SAEBench (2025) and SynthSAEBench (2026) now provide standardized, multi-metric evaluation frameworks that have revealed proxy metrics (reconstruction, L0) do not reliably predict practical interpretability.
- **Hierarchical structure as solution**: Multiple independent lines of work (Matryoshka, HSAE, Feature Forest) converge on the insight that explicitly modeling hierarchical relationships mitigates absorption.

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 (NeurIPS 2025) | 2024/2025 | **First systematic definition of feature absorption**; introduced absorption rate metric; proved via toy models that hierarchy + sparsity optimization inevitably causes absorption; validated on hundreds of LLM SAEs | Ablation-dependent metric limited to layers 0-17; conservative underestimate (misses multi-latent absorption); only first-letter classification task |
| 2 | Measuring Sparse Autoencoder Feature Sensitivity | arXiv:2509.23717 | 2025 | Proposed **feature sensitivity** as a new evaluation dimension; found many interpretable features have poor sensitivity; sensitivity **declines with SAE width** across 7 variants up to 1M features | Requires LLM-generated similar text; computationally expensive; human validation needed |
| 3 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability | arXiv:2503.09532 (ICML 2025) | 2025 | **Eight-metric benchmark** (absorption, sparse probing, auto-interpretability, RAVEL, unlearning, SCR, TPP, CE loss) on 200+ SAEs; revealed **proxy metrics do not predict practical performance** | High variance on some metrics; indirect proxies for true feature recovery |
| 4 | SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data | arXiv:2602.14687 | 2026 | **Ground-truth synthetic benchmark** (16k features, hierarchy, correlation, superposition); identified MP-SAEs exploit superposition noise without learning true features; no architecture achieves perfect performance | Synthetic data may not capture all LLM representation complexity |
| 5 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | arXiv:2503.17547 | 2025 | **Nested dictionaries** trained simultaneously; smaller dicts learn general features, larger dicts specialize; reduced absorption from 0.49 to 0.05; reduced splitting from 3 latents to 1 | ~50% computational overhead; slightly worse reconstruction; feature hedging in narrow inner levels |
| 6 | Building a Structured Feature Forest with Hierarchical Sparse Autoencoders | arXiv:2602.11881 | 2026 | **HSAE**: jointly trains series of SAEs with explicit parent-child relationships; structural constraint loss + random perturbation; **substantially outperforms baselines on absorption**, especially at larger sizes | Newer work, less community validation; training complexity |
| 7 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | **Orthogonality penalty** on latents; **65% absorption reduction**, 15% composition reduction, +9% distinct features; chunk-wise strategy for linear scaling | Only training-time modification; orthogonality may be too strong for correlated features |
| 8 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | DeepMind (arXiv:2408.05147) | 2024 | Released **Gemma 2 full-layer SAEs** (16k/65k/1m width, 27 layers); TopK activation; Apache 2.0 license; became the dominant experimental platform for SAE research | Feature absorption observed across all layers and sizes |
| 9 | Scaling and Evaluating Sparse Autoencoders | Anthropic (ICLR 2025) | 2024/2025 | Large-scale SAE training on Claude 3 Sonnet; precision/recall analysis; demonstrated feasibility of million-feature SAEs | Precision/recall imbalance; absorption not directly addressed |
| 10 | Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders | arXiv:2508.09363 | 2025 | Argued broad-domain training produces representations vulnerable to nonlinear error and feature absorption; proposed **domain-specific SAEs** as mitigation | Domain-specific approach limits generalization |
| 11 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | arXiv:2505.11756 | 2025 | Identified **feature hedging**: correlated features merge in narrow SAEs; related failure mode to absorption; compound multiplier can balance Matryoshka SAE performance | Narrow SAE focus; not directly about hierarchy |
| 12 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | Emphasized **feature consistency** across training runs; proposed PW-MCC metric; different runs learn different feature sets, hurting reproducibility | Consistency is different from absorption, but related to reliability |
| 13 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | arXiv:2506.01197 | 2025 | Early hierarchical SAE architecture with explicit parent-child constraints; showed decreased absorption rate as width increases | Smaller scale than HSAE; less comprehensive evaluation |
| 14 | AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations | arXiv:2508.17320 | 2025 | **Dynamic per-input sparsity**; superior absorption results on SAEBench despite training on 2,000x less data (250K vs 500M tokens) | Newer architecture, limited community adoption |
| 15 | A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability | arXiv:2512.05534 | 2025/2026 | **Theoretical framework**: piecewise biconvexity and spurious minima; feature anchoring improves recovery across architectures; explains why simple linear encoders are hard to outperform | Theoretical focus; limited empirical guidance for practitioners |

## 3. SOTA Methods and Benchmarks

### Current Best Methods

| Method | Description | Key Advantage | Absorption Impact |
|--------|-------------|---------------|-------------------|
| **Matryoshka SAE** | Nested dictionaries trained simultaneously | Best disentanglement, growing advantage with scale | Absorption: 0.49 -> 0.05 |
| **HSAE (Feature Forest)** | Jointly trained series of SAEs with parent-child links | Explicit hierarchical structure | Substantially outperforms baselines |
| **OrtSAE** | Orthogonality penalty on latents | 65% absorption reduction, minimal overhead | Strong at L0=70 |
| **AdaptiveK SAE** | Dynamic per-input sparsity allocation | Superior absorption, data-efficient | Best absorption on SAEBench |
| **JumpReLU SAE** | Learnable per-feature thresholds | Better sparsity-fidelity trade-off, lower absorption than TopK | 0.0114 vs TopK 0.1402 |
| **TopK SAE** | Exact k features per input | Avoids L1 tuning, but rigid | Higher absorption, severe dead features |
| **BatchTopK SAE** | Cross-batch top-k*B selection | Better average reconstruction | Moderate absorption |
| **Matching Pursuit SAE** | Iterative feature selection | Good reconstruction | Exploits superposition noise; poor feature quality |

### Mainstream Datasets and Evaluation Resources

| Resource | Description | Access |
|----------|-------------|--------|
| **Gemma Scope SAEs** | Gemma 2 full-layer SAEs (16k/65k/1m, 27 layers) | HuggingFace / DeepMind (Apache 2.0) |
| **SAELens** | SAE training, analysis, and evaluation library | [GitHub](https://github.com/jbloomAus/SAELens) (Apache 2.0) |
| **Neuronpedia** | Online SAE feature dashboard (50+ models) | [neuronpedia.org](https://neuronpedia.org) |
| **SAEBench** | Comprehensive 8-metric benchmark | [neuronpedia.org/sae-bench](https://www.neuronpedia.org/sae-bench) |
| **SynthSAEBench-16k** | Ground-truth synthetic benchmark | HuggingFace (decoderesearch/synth-sae-bench-16k-v1) |
| **SAE Spelling Dataset** | First-letter classification test set | [GitHub](https://github.com/lasr-spelling/sae-spelling) (MIT) |

### Evaluation Metrics

| Metric | Definition | Key 2025 Source |
|--------|-----------|-----------------|
| **Absorption Rate** | num_absorptions / lr_probe_true_positives | Chanin et al. (2024/2025) |
| **Mean Absorption Fraction** | P_compensated_by_absorbers / (P_compensated + P_main) | SAEBench |
| **Full Absorption Rate** | Proportion where main features inactive, single dominant latent represents GT | SAEBench |
| **Feature Sensitivity** | Fraction of LLM-generated similar texts that activate the feature | Hu et al. (2025) |
| **MCC (Mean Correlation Coefficient)** | Optimal bipartite matching between learned and ground-truth features | O'Neill et al. (2025) |
| **PW-MCC** | Pairwise Mean Correlation Coefficient for cross-run consistency | Song et al. (2025) |
| **Explained Variance (R^2)** | Reconstruction quality | Standard |
| **L0 Sparsity** | Average active latent count | Standard |
| **Feature Uniqueness** | Fraction of latents tracking unique features | SynthSAEBench |

## 4. Identified Research Gaps

### Gap 1: Absorption Detection Beyond Ablation-Dependent Methods
**Problem**: The primary absorption rate metric (Chanin et al.) requires causal ablation experiments, which only work for early layers (<= layer 17 in Gemma 2 2B) where mediation is verifiable. Deep attention layers have moved information to final token positions, making ablation effects disappear.

**Exploration Directions**:
- Encoder-decoder asymmetry as an absorption signal (absorbed features may show mismatched encoder/decoder behavior)
- Attention-flow-based inference of deep-layer absorption patterns
- Sensitivity-based proxy: features with low sensitivity may indicate absorption

### Gap 2: Cross-Architecture Absorption Comparison at Scale
**Problem**: While individual papers report absorption metrics for their proposed architectures, there is no systematic, controlled comparison across all major architectures (ReLU, TopK, JumpReLU, Matryoshka, HSAE, OrtSAE, AdaptiveK) on the same models, layers, and sparsity levels using both SAEBench and SynthSAEBench.

**Exploration Directions**:
- Large-scale controlled ablation study across 6+ architectures on Gemma-2-2B
- Disentangle architecture effects from training data size and hyperparameter choices
- Analyze absorption-layer interaction: does absorption increase monotonically with depth?

### Gap 3: Quantifying Absorption's Impact on Downstream Interpretability
**Problem**: Most absorption research treats it as an intrinsic SAE property. The downstream consequences---how absorption affects sparse feature circuits, model editing, unlearning, and concept erasure---are underexplored.

**Exploration Directions**:
- Measure circuit completeness with/without absorbed features
- Quantify how absorption degrades targeted model editing success rates
- Evaluate whether absorbed features cause false negatives in safety-relevant concept detection

### Gap 4: Training-Free Analysis of Pretrained SAEs
**Problem**: Many proposed solutions (OrtSAE, HSAE, Matryoshka) require training new SAEs. For researchers with limited compute, training-free methods to detect, measure, and potentially mitigate absorption in existing pretrained SAEs (like Gemma Scope) are valuable but underdeveloped.

**Exploration Directions**:
- Post-hoc absorption detection in pretrained SAEs using encoder-decoder analysis
- Steering-based mitigation: can targeted interventions recover absorbed feature directions?
- Feature sensitivity screening as a training-free quality filter

### Gap 5: Theoretical Understanding of Absorption-Sensitivity Relationship
**Problem**: Feature sensitivity (Hu et al., 2025) and feature absorption (Chanin et al., 2024) are related but distinct concepts. Sensitivity measures inconsistent activation; absorption measures directional stealing by child features. Their quantitative relationship is unclear.

**Exploration Directions**:
- Joint measurement of sensitivity and absorption on the same features
- Determine whether low-sensitivity features are disproportionately absorbed
- Develop a unified theoretical framework connecting sparsity, hierarchy, sensitivity, and absorption

## 5. Available Resources

### Open-source Code

| Repository | Description | Language | License |
|------------|-------------|----------|---------|
| [SAELens](https://github.com/jbloomAus/SAELens) | SAE training, analysis, evaluation; supports Gemma Scope, GPT-2, Pythia | Python | Apache 2.0 |
| [sae-spelling](https://github.com/lasr-spelling/sae-spelling) | Feature absorption paper official code; absorption rate metric implementation | Python | MIT |
| [SAEBench](https://github.com/ai-safety-institute/SAEBench) | Comprehensive benchmark suite with 8 metrics | Python | - |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Mechanistic interpretability analysis framework | Python | BSD |
| [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) | Feature dashboard and browsing tools | Python/TS | - |
| [multimodal-sae](https://github.com/EvolvingLMMs-Lab/multimodal-sae) | Multimodal SAE analysis (ICCV 2025) | Python | - |
| [mlsae](https://github.com/tim-lawson/mlsae) | Multi-Layer SAEs (ICLR 2025) | Python | - |
| [RouteSAE](https://github.com/swei2001/RouteSAEs) | Cross-layer routing SAE | Python | - |

### Pretrained Models & Weights

| Resource | Description | Access |
|----------|-------------|--------|
| Gemma Scope SAEs | Gemma 2 27-layer SAEs (16k/65k/1m) | HuggingFace / DeepMind |
| GPT-2 SAEs | SAELens pretrained SAEs for GPT-2 small | HuggingFace |
| Pythia SAEs | SAELens pretrained SAEs for Pythia family | HuggingFace |
| E2E SAE | Function-importance-oriented SAE | ApolloResearch GitHub |

### Datasets

| Dataset | Description | Task |
|---------|-------------|------|
| First-letter ICL prompts | "{token} has the first letter: {letter}" format | Letter classification / absorption detection |
| SynthSAEBench-16k | 16k ground-truth features with hierarchy, correlation | Controlled SAE evaluation |
| OpenWebText | Standard language modeling corpus | General SAE training/evaluation |

## 6. Implications for Idea Generation

### Worth Exploring

1. **Training-free absorption quantification**: Develop methods to measure absorption in pretrained SAEs without ablation, using encoder-decoder asymmetry or sensitivity analysis. This directly addresses Gap 4 and aligns with the project's training-free constraint.

2. **Absorption-sensitivity joint analysis**: Systematically measure both metrics on the same feature set to determine their correlation and build a unified quality score. This addresses Gap 5.

3. **Cross-layer absorption propagation**: Study how absorption in early layers affects downstream layer representations and circuit completeness. This addresses Gap 3.

4. **Steering-based absorption mitigation**: Test whether targeted activation steering can recover absorbed feature directions without retraining SAEs.

5. **Hierarchical feature inference from absorption patterns**: Use the encoding-decoding asymmetry of absorbed features to reverse-engineer the underlying hierarchical structure of LLM representations.

### Saturated Directions

1. Simply expanding SAE width or adjusting L0 sparsity (shown insufficient by Chanin et al.)
2. Pure visualization analysis (maximum-activating examples)
3. Single-architecture, single-model studies without benchmark comparison
4. Proposing new SAE architectures without SAEBench or SynthSAEBench validation

### Cross-domain Potential

- **Code models**: API call hierarchies (general "API call" vs specific "HTTP GET") may exhibit similar absorption patterns
- **Multimodal models**: Vision-language SAEs may show cross-modal absorption (visual features absorbed into text-aligned features)
- **Safety applications**: Absorbed safety-relevant features (e.g., deception indicators) may evade detection, creating hidden risks

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens library | High | Apache 2.0 | **Adopt** | Complete SAE training/analysis framework; supports Gemma Scope, GPT-2; essential infrastructure |
| Gemma Scope SAEs | High | Apache 2.0 | **Adopt** | High-quality pretrained SAEs; full-layer coverage; ideal for training-free analysis |
| SAEBench evaluation suite | High | - | **Adopt** | Standardized 8-metric benchmark; includes absorption metric; essential for validation |
| sae-spelling (absorption paper code) | High | MIT | **Extend** | Provides absorption rate metric and toy model; can extend to new layers/architectures |
| TransformerLens | High | BSD | **Adopt** | Hook-based activation extraction; integrates with SAELens |
| SynthSAEBench | Medium | - | **Compose** | Ground-truth validation for methodology; use to verify training-free metrics |
| Matryoshka SAE code | Medium | - | **Reference** | Hierarchical design patterns; useful for understanding absorption mitigation strategies |
| OrtSAE code | Medium | - | **Reference** | Orthogonality-based mitigation; design patterns for post-hoc analysis |

### Recommended Research Pipeline for This Project

```
1. Foundation (Training-Free)
   ├── Load Gemma Scope / GPT-2 SAEs via SAELens
   ├── Reproduce absorption rate baselines (layers 0-17)
   ├── Measure feature sensitivity on same features
   └── Cross-validate metrics on SynthSAEBench ground truth

2. Novel Analysis
   ├── Encoder-decoder asymmetry as absorption signal
   ├── Absorption-sensitivity correlation analysis
   ├── Cross-layer absorption propagation study
   └── Steering-based mitigation experiments

3. Quantification & Impact
   ├── Downstream circuit completeness with/without absorbed features
   ├── Absorption frequency across model layers and SAE sizes
   └── Comparative analysis across architectures (TopK, JumpReLU, ReLU)

4. Validation
   ├── SAEBench absorption metric validation
   ├── Human evaluation of absorption detection accuracy
   └── Reproducibility package for community use
```

---

## References

### Key Papers
- Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2024/2025). "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507. NeurIPS 2025.
- Hu, N., et al. (2025). "Measuring Sparse Autoencoder Feature Sensitivity." arXiv:2509.23717.
- Karvonen, A., et al. (2025). "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability." arXiv:2503.09532. ICML 2025.
- Chanin, D., et al. (2026). "SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data." arXiv:2602.14687.
- Bussmann, B., et al. (2025). "Learning Multi-Level Features with Matryoshka Sparse Autoencoders." arXiv:2503.17547.
- Luo, Y., et al. (2026). "Building a Structured Feature Forest with Hierarchical Sparse Autoencoders." arXiv:2602.11881.
- Korznikov, A., et al. (2025). "OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features." arXiv:2509.22033.
- Lieberum, T., et al. (2024). "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2." arXiv:2408.05147.
- Bricken, T., et al. (2023/2025). "Scaling and Evaluating Sparse Autoencoders." ICLR 2025.
- Chanin, D., & Garriga-Alonso, A. (2025). "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders." arXiv:2505.11756.
- Song, Y., et al. (2025). "Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs." arXiv:2505.20254.
- Muchane, V., et al. (2025). "Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures." arXiv:2506.01197.
- Till, D. (2025). "Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders." arXiv:2508.09363.
- Marks, S., et al. (2025/2026). "A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534.

### Tools & Libraries
- SAELens: https://github.com/jbloomAus/SAELens
- SAEBench: https://www.neuronpedia.org/sae-bench
- SAE Spelling: https://github.com/lasr-spelling/sae-spelling
- Neuronpedia: https://neuronpedia.org
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
