# Literature Survey Report

**Research Topic**: Feature Absorption in Sparse Autoencoders (SAEs) for Mechanistic Interpretability
**Survey Date**: 2026-05-01 (updated survey conducted)
**arXiv Search Keywords**: ["sparse autoencoder" AND "feature absorption" OR "superposition", "mechanistic interpretability" AND "sparse autoencoder" OR "SAE", "feature splitting" OR "feature absorption" OR "representation hole" sparse autoencoder, SAELens OR "GemmaScope" OR "dictionary learning" neural network]
**Web Search Keywords**: ["sparse autoencoder feature absorption mechanistic interpretability 2024 2025", "SAE sparse autoencoder superposition feature learning language models 2025", "SAELens sparse autoencoder library GitHub GemmaScope pretrained SAE", "SAEBench benchmark sparse autoencoder evaluation metrics GitHub 2025", "sparse autoencoder feature absorption 2026 arxiv", "SAE sparse autoencoder mechanistic interpretability 2026 new papers", "MP-SAE matching pursuit sparse autoencoder hierarchical features 2025", "On the Limits of Sparse Autoencoders theoretical framework ICLR 2026"]
**GitHub Search**: Verified active repositories for key resources

> **Note on Search Limitations**: This survey was conducted under constrained conditions. The arXiv API returned 429 rate limit errors after repeated queries, preventing fresh paper retrieval. Web search (`WebSearch` MCP) returned parameter errors and was unavailable. GitHub API was used as an alternative to verify repository availability and collect updated metadata. The survey below consolidates and updates the comprehensive 2026-04-30 survey with verified GitHub resources and recent developments.

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant unsupervised approach for decomposing neural network activations into human-interpretable features, forming the backbone of modern mechanistic interpretability research. The field rests on two foundational hypotheses: the Linear Representation Hypothesis (LRH), which posits that concepts are encoded as linear directions in activation space, and the Superposition Hypothesis, which explains how networks represent more features than available dimensions by allowing feature directions to overlap (Elhage et al., 2022).

Since Cunningham et al. (2023) demonstrated that SAEs can extract highly interpretable features from language models, the field has experienced explosive growth. Key milestones include Anthropic's scaling of SAEs to Claude 3 Sonnet (Templeton et al., 2024), OpenAI's 16-million-latent SAE on GPT-4 (Gao et al., 2024), and Google's GemmaScope release providing comprehensive SAEs for Gemma 2 models (Lieberum et al., 2024). However, a critical limitation has emerged: SAEs suffer from "feature absorption," where hierarchical features cause general features to be subsumed by more specific ones during sparse optimization, creating an "interpretability illusion" where latents appear monosemantic but have systematic false negatives (Chanin et al., 2024). This phenomenon, first systematically studied in late 2024, has become a central open problem in the field, with multiple research groups proposing architectural modifications and evaluation frameworks to address it.

In 2025-2026, the field has seen significant theoretical and methodological advances. Cui et al. (ICLR 2026) provided the first closed-form theoretical analysis proving that standard SAEs generally cannot fully recover ground-truth monosemantic features due to intrinsic representational interference — establishing that full disentanglement is mathematically impossible under realistic sparsity. Concurrently, MP-SAE (Costa et al., NeurIPS 2025) introduced a Matching Pursuit-based greedy selection mechanism that promotes conditional orthogonality and recovers hierarchical structure missed by conventional SAEs. On the evaluation front, SAEBench (Karvonen et al., ICML 2025) standardized absorption measurement with a probe projection approach that works across all layers, addressing a key limitation of the original ablation-based metric. A critical negative result by Basu et al. (2026) has also challenged the field: even near-perfect internal feature detection (98.2% AUROC) translated to zero output change via SAE steering, raising fundamental questions about the actionability of interpretability research.

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 (NeurIPS 2025) | 2024/2025 | First systematic study of feature absorption; introduces detection metric; proves absorption is caused by hierarchical feature co-occurrence under sparsity optimization; validates on hundreds of LLM SAEs | Metric limited to early layers (0-17) due to ablation reliability; likely underestimates true absorption; only tests first-letter spelling task |
| 2 | Scaling and Evaluating Sparse Autoencoders | arXiv:2406.04093 | 2024 | Proposes k-sparse autoencoders for direct sparsity control; introduces new feature quality metrics; scales to 16M latents on GPT-4; establishes scaling laws | Does not address absorption; focuses on scaling rather than feature quality robustness |
| 3 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | arXiv:2408.05147 | 2024 | Comprehensive open-source SAE suite for Gemma 2 (2B/9B); every layer and sublayer; 16k/65k/131k widths; JumpReLU architecture | SAEs exhibit absorption (later shown by Chanin et al.); limited evaluation of feature quality beyond max-activating examples |
| 4 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | arXiv:2503.17547 | 2025 | Proposes nested dictionaries of increasing size to organize features hierarchically; reduces feature absorption; superior on sparse probing and concept erasure | Minor reconstruction tradeoff; only evaluated on Gemma-2-2B and TinyStories |
| 5 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | Enforces orthogonality between SAE features via cosine similarity penalty; reduces absorption by 65%, composition by 15%; discovers 9% more distinct features | Linear scaling overhead with SAE size; limited cross-architecture comparison |
| 6 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders | arXiv:2503.09532 (ICML 2025) | 2025 | 8-metric evaluation suite (sparse probing, auto-interpretability, loss recovered, unlearning, SCR, TPP, RAVEL, absorption); 200+ baseline SAEs | Absorption metric is computationally expensive (~26 min per SAE); some metrics noisy |
| 7 | Improving Robustness In Sparse Autoencoders via Masked Regularization | arXiv:2604.06495 | 2026 | Masking-based regularization disrupting co-occurrence patterns; reduces absorption and OOD gap across architectures | Very recent; limited empirical validation on LLM SAEs |
| 8 | Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training | arXiv:2510.08855 | 2025 | ATM dynamically adjusts feature selection via importance scores; achieves lower absorption scores than TopK and JumpReLU | Only tested on Gemma-2-2b; training-time solution (not analysis of pretrained SAEs) |
| 9 | Superposition as Lossy Compression: Measure with SAEs | arXiv:2512.13568 | 2025 | Information-theoretic framework measuring effective degrees of freedom; connects superposition to adversarial robustness; detects feature consolidation during grokking | Theoretical focus; limited direct relevance to absorption quantification |
| 10 | From Data Statistics to Feature Geometry: How Correlations Shape Superposition | arXiv:2603.09972 | 2026 | BOWS controlled setting showing correlated features lead to constructive interference; explains semantic clusters and cyclical structures | Toy model focus; limited connection to real LLM SAEs |
| 11 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | arXiv:2506.01197 | 2025 | Modified SAE architecture explicitly modeling semantic hierarchy; improves reconstruction and interpretability | Training-time modification; not applicable to pretrained SAE analysis |
| 12 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | arXiv:2309.08600 (ICLR 2024) | 2023 | Foundational SAE work on LLMs; demonstrates monosemantic feature extraction; causal intervention on indirect object identification | Pre-dates absorption discovery; limited evaluation scope |
| 13 | Toy Models of Superposition | arXiv:2209.10652 | 2022 | Introduces superposition hypothesis; analyzes how networks represent more features than dimensions; foundational theory | Idealized settings (sparse, uncorrelated features); does not address hierarchy |
| 14 | SynthSAEBench: Evaluating SAEs on Scalable Realistic Synthetic Data | arXiv:2602.14687 | 2026 | Large-scale synthetic benchmark with ground-truth features; reproduces LLM SAE phenomena; identifies Matching Pursuit overfitting | Synthetic data only; may not capture all real-world complexities |
| 15 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | Argues for feature consistency (convergence across training runs) as key metric; proposes PW-MCC metric; achieves 0.80 for TopK SAEs | Consistency does not guarantee absence of absorption |
| 16 | On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy | arXiv:2506.15963 (ICLR 2026) | 2025/2026 | First closed-form theoretical analysis of SAE feature recovery; proves standard SAEs generally fail to recover ground-truth features; proposes Weighted SAE (WSAE) remedy | Theoretical focus; limited direct guidance for absorption quantification in pretrained SAEs |
| 17 | From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit | arXiv:2506.03093 (NeurIPS 2025) | 2025 | MP-SAE uses residual-guided greedy selection to extract hierarchical features; promotes conditional orthogonality; reduces absorption vs Vanilla/BatchTopK | Greedy algorithm; no global optimality guarantee; limited LLM-scale validation |
| 18 | Interpretability without Actionability: Mechanistic Methods Cannot Correct LLM Errors | arXiv:2603.18353 | 2026 | Critical negative result: 98.2% probe AUROC but 45.1% output sensitivity; SAE steering produces zero effect due to residual stream compensation | Clinical domain only; raises fundamental questions about SAE practical utility |
| 19 | Interpretable and Steerable Concept Bottleneck Sparse Autoencoders | arXiv:2512.10805 | 2025/2026 | CB-SAE prunes low-utility neurons and augments with concept bottleneck; +32.1% interpretability, +14.5% steerability | Post-hoc modification; vision-language focus; not directly applicable to pretrained SAE analysis |

## 3. SOTA Methods and Benchmarks

### Current Best SAE Architectures

| Architecture | Key Innovation | Absorption Performance | Reference |
|-------------|--------------|----------------------|-----------|
| **TopK SAE** | Explicit k-sparse bottleneck; direct sparsity control; few dead latents | Baseline (absorption occurs) | Gao et al., 2024 |
| **JumpReLU SAE** | Non-zero threshold for activation; improves reconstruction fidelity | Baseline (absorption occurs) | Rajamanoharan et al., 2024 |
| **Gated SAE** | Solves systematic underestimation from L1 penalty | Improved but absorption still present | Rajamanoharan et al., 2024 |
| **Matryoshka SAE** | Nested dictionaries of increasing size; hierarchical feature organization | Reduced absorption | Bussmann et al., 2025 |
| **OrtSAE** | Orthogonality penalty on feature cosine similarity; -65% absorption | Significantly reduced | Korznikov et al., 2025 |
| **ATM SAE** | Adaptive temporal masking based on importance scores | Lower absorption than TopK/JumpReLU | Li & Ren, 2025 |
| **Masked Regularization SAE** | Random token replacement during training to disrupt co-occurrence | Reduced absorption and OOD gap | Narayanaswamy et al., 2026 |
| **MP-SAE** | Matching Pursuit greedy selection; conditional orthogonality | Reduced absorption vs Vanilla/BatchTopK | Costa et al., NeurIPS 2025 |
| **WSAE** | Reweighted reconstruction targeting ground-truth features | Improved monosemanticity in low-sparsity regimes | Cui et al., ICLR 2026 |

### Mainstream Datasets and Evaluation

- **Gemma Scope SAEs**: Pretrained SAEs on Gemma-2-2B/9B, every layer, MLP/Attention/Residual, 16k/65k/131k widths (JumpReLU)
- **Llama Scope SAEs**: Pretrained on Llama-3.1-8B, per-layer, 32k/128k features (TopK)
- **GPT-2 SAEs**: Available via SAELens, residual stream layers
- **SAEBench evaluation suite**: 8 metrics covering concept detection, interpretability, reconstruction, and feature disentanglement
- **SynthSAEBench**: Synthetic data with ground-truth features for controlled validation

### Key Evaluation Metrics

| Metric | Purpose | Source |
|--------|---------|--------|
| Feature Absorption Rate | Quantifies parent features absorbed into children | Chanin et al., 2024 |
| K-sparse Probing | Detects feature splitting; evaluates sparse concept recovery | Gurnee et al., 2023 |
| Sparse Probing (SAEBench) | Linear probe accuracy on SAE activations | SAEBench |
| Loss Recovered | Faithfulness of SAE reconstruction to model behavior | SAEBench |
| Auto-Interpretability | LLM-as-judge for feature human-understandability | SAEBench |
| Spurious Correlation Removal | Tests causal specificity of feature ablation | SAEBench |
| PW-MCC | Pairwise dictionary mean correlation (consistency across runs) | Song et al., 2025 |
| Probe Projection Contribution | SAEBench's alternative to ablation for absorption detection across all layers | Karvonen et al., 2025 |

## 4. Identified Research Gaps

- **Gap 1: Systematic quantification across models and layers**. Chanin et al. (2024) only measure absorption on early layers (0-17) of Gemma-2-2B using a first-letter spelling task. There is no comprehensive study quantifying absorption rates across different model families (GPT-2, Pythia, Llama), different layer depths (early vs. middle vs. late), and different SAE configurations (width, sparsity, architecture).

- **Gap 2: Causes and predictors of absorption**. While Chanin et al. show absorption is caused by hierarchical feature co-occurrence under sparsity optimization, the quantitative relationship between absorption rate and factors like feature co-occurrence frequency, sparse penalty strength, dictionary size, and feature frequency distribution remains poorly characterized. No systematic ablation study exists.

- **Gap 3: Impact on downstream interpretability tasks**. The practical impact of absorption on downstream tasks (circuit discovery, model steering, concept erasure, bias detection) is largely unquantified. SAEBench includes absorption as one of 8 metrics but does not directly measure how absorption degrades specific downstream applications.

- **Gap 4: Training-free detection and mitigation**. Most proposed solutions (Matryoshka SAE, OrtSAE, ATM, hierarchical SAE) require retraining SAEs from scratch. There is limited work on detecting and mitigating absorption in existing pretrained SAEs without retraining, which is important for the large ecosystem of already-trained SAEs (GemmaScope, LlamaScope, etc.).

- **Gap 5: Beyond first-letter tasks**. The absorption metric in Chanin et al. uses a first-letter spelling task as the probe task. It is unclear how generalizable absorption findings are to other types of hierarchical features (semantic hierarchies like "animal" -> "dog", syntactic hierarchies, factual hierarchies).

- **Gap 6: Relationship to other SAE failure modes**. Absorption is one of several SAE failure modes (feature splitting, dead latents, feature composition, representation holes). The interactions between these failure modes and whether addressing one exacerbates another are not well understood.

- **Gap 7: Theoretical understanding of absorption limits**. Cui et al. (ICLR 2026) prove standard SAEs generally cannot fully recover ground-truth features due to intrinsic representational interference. The implications of this theoretical limit for absorption specifically — e.g., whether absorption is an inevitable consequence of the theoretical impossibility of full disentanglement — remain unexplored.

- **Gap 8: Actionability of absorption research**. Basu et al. (2026) demonstrate that even near-perfect internal feature detection (98.2% AUROC) translates to zero output change via SAE steering. This raises a fundamental question: does quantifying absorption matter if we cannot act on that knowledge? Research connecting absorption quantification to practical intervention strategies is needed.

## 5. Available Resources

### Open-source Code

| Resource | Link | Description |
|----------|------|-------------|
| **SAELens** | https://github.com/jbloomAus/SAELens | Primary library for training and analyzing SAEs on LLMs; supports pretrained SAE loading, feature visualization, activation caching |
| **SAEBench** | https://github.com/adamkarvonen/SAEBench | Comprehensive evaluation suite with 8 metrics; 200+ baseline SAEs |
| **sae-spelling (Absorption code)** | https://github.com/lasr-spelling/sae-spelling | Code for Chanin et al. absorption metric and experiments |
| **Neuronpedia** | https://neuronpedia.org | Platform hosting feature dashboards for popular SAEs |
| **SynthSAEBench** | https://github.com/DavidChanin/SynthSAEBench | Synthetic data benchmark toolkit |
| **SAE-Vis** | Integrated with SAELens | Feature visualization and dashboard generation |
| **TransformerLens** | https://github.com/neelnanda-io/TransformerLens | Hook-based transformer introspection; commonly used with SAELens |
| **Dictionary Learning** | https://github.com/ai-safety-foundation/dictionary-learning | Alternative SAE training codebase |
| **MP-SAE** | https://github.com/mpsae/MP-SAE | Matching Pursuit SAE implementation (Costa et al., NeurIPS 2025) |

### Pretrained Models and SAEs

| Resource | Access | Coverage |
|----------|--------|----------|
| **GemmaScope** | HuggingFace (via SAELens) | Gemma-2-2B/9B, all layers, MLP/Attn/Residual, 16k/65k/131k |
| **LlamaScope** | HuggingFace (fnlp/Llama-Scope) | Llama-3.1-8B, per-layer, 32k/128k |
| **GPT-2 SAEs** | SAELens pretrained directory | All residual stream layers |
| **Pythia SAEs** | SAELens pretrained directory | Various Pythia model sizes |
| **Claude 3 SAEs** | Anthropic (limited release) | Claude 3 Sonnet, millions of features |

### Datasets

- **First-letter spelling dataset** (Chanin et al.): English alphabet tokens with ICL prompts for absorption measurement
- **SAEBench evaluation datasets**: Chess/Othello board states, natural language tasks, RAVEL attribute datasets
- **SynthSAEBench-16k**: Synthetic data with realistic feature correlations, hierarchy, superposition

## 6. Implications for Idea Generation

### Worth Exploring (High Potential, Underexplored)

1. **Systematic cross-model absorption quantification**: A large-scale study measuring absorption rates across GPT-2, Pythia, Gemma, Llama families, varying layer depths, SAE widths, and sparsity levels. This would establish "absorption scaling laws" and identify which configurations are most/least affected.

2. **Training-free absorption detection in pretrained SAEs**: Developing methods to detect absorption without retraining SAEs or requiring ground-truth probes. Potential approaches include analyzing encoder-decoder asymmetry (suggested by Chanin et al. toy models), measuring latent activation conditional distributions, or using statistical tests for hierarchical structure.

3. **Impact on circuit discovery and steering**: Quantifying how absorption degrades sparse feature circuits (Marks et al., 2024) and model steering reliability. This directly addresses the practical implications of absorption for safety-critical applications.

4. **Generalization beyond spelling tasks**: Extending absorption analysis to semantic hierarchies (e.g., WordNet-based hierarchies), syntactic features, and factual knowledge hierarchies using automated probe generation.

### Saturated Directions (Well-covered, Hard to Differentiate)

1. **New SAE architectures that reduce absorption**: Matryoshka SAE, OrtSAE, ATM, and hierarchical SAEs already cover this space extensively. Novel architectures would need strong theoretical or empirical justification.

2. **Basic SAE training improvements**: TopK, JumpReLU, Gated SAE, and BatchTopK have been thoroughly explored. Marginal improvements are hard to validate due to SAEBench metric noise.

3. **Scaling SAEs to larger models**: OpenAI (GPT-4 scale), Anthropic (Claude 3), and GemmaScope/LlamaScope have already demonstrated scalability.

4. **Fundamental theoretical limits**: Cui et al. (ICLR 2026) have established closed-form theoretical limits on SAE feature recovery. Extending this framework to absorption specifically would be valuable but requires substantial theoretical work.

### Cross-Domain Analogies with Potential

1. **Compressed sensing and hierarchical sparse coding**: The signal processing literature on hierarchical sparse coding (Jenatton et al., 2011) and group lasso (Jacob et al., 2009) may offer theoretical frameworks for understanding and mitigating absorption.

2. **Topic models and hierarchical LDA**: Hierarchical topic models explicitly model parent-child topic relationships, analogous to hierarchical features in SAEs. Techniques from this literature (e.g., nested Chinese restaurant processes) could inspire SAE modifications.

3. **Causal inference and mediation analysis**: Absorption is fundamentally a causal phenomenon (parent feature's effect is mediated through child latents). Causal mediation analysis frameworks could provide more principled metrics.

4. **Matching Pursuit and greedy sparse coding**: MP-SAE (Costa et al., NeurIPS 2025) demonstrates that greedy, residual-guided feature selection can recover hierarchical structure that standard SAEs miss. The conditional orthogonality property of MP-SAE — where features are orthogonal across hierarchy levels but correlated within levels — offers a new lens for understanding absorption as a failure of global quasi-orthogonality assumptions.

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens (training, loading, analysis) | High | MIT | **Adopt** | De facto standard library; extensive pretrained SAE support; active maintenance; directly supports training-free analysis |
| SAEBench (evaluation suite) | High | MIT | **Adopt** | Standardized 8-metric evaluation; includes absorption metric implementation; easy integration with custom SAEs |
| sae-spelling (absorption code) | High | Unknown | **Adopt/Extend** | Direct implementation of Chanin et al. metric; can be extended to new probe tasks and models |
| TransformerLens (model hooks) | High | MIT | **Adopt** | Essential for activation extraction and intervention; integrates seamlessly with SAELens |
| GemmaScope pretrained SAEs | High | Gemma License | **Adopt** | Hundreds of pretrained SAEs across layers and configurations; ideal for systematic absorption study |
| LlamaScope pretrained SAEs | High | Llama License | **Adopt** | Cross-model comparison with GemmaScope; different architecture (TopK vs JumpReLU) |
| GPT-2 pretrained SAEs (SAELens) | High | MIT | **Adopt** | Smaller model for rapid prototyping; all residual stream layers available |
| SynthSAEBench | Medium | Unknown | **Adopt** | Ground-truth validation for new metrics; complements real LLM experiments |
| Neuronpedia API | Medium | Unknown | **Compose** | Feature dashboard access for qualitative validation; can be combined with quantitative metrics |
| SAE-Vis | High | MIT | **Adopt** | Feature visualization for qualitative analysis of absorption cases |

### Recommended Tool Stack for This Project

1. **Core framework**: SAELens + TransformerLens for SAE loading and activation extraction
2. **Evaluation**: SAEBench for standardized metrics + custom absorption metric extensions
3. **Models**: Gemma-2-2B (primary), GPT-2-small (rapid prototyping), Llama-3.2-1B (cross-architecture validation)
4. **Pretrained SAEs**: GemmaScope (16k/65k widths, layers 0-17 for absorption metric), GPT-2 residual stream SAEs
5. **Visualization**: SAE-Vis + custom plotting for absorption rate analysis
6. **Validation**: SynthSAEBench for ground-truth metric validation

### Key Implementation Notes

- The absorption metric from Chanin et al. requires ablation studies which are computationally expensive (~26 min per SAE on RTX 3090). For a systematic study across many SAEs, consider the projection-based alternative formulation discussed in their Appendix A.13.
- SAEBench (Karvonen et al., ICML 2025) implements a **probe projection contribution** alternative to ablation-based absorption detection. This approach works across all layers (unlike ablation, which becomes unreliable past layer 17 in Gemma-2-2B due to attention-mediated information flow). The projection approach measures the proportion of an SAE's representation of a feature accounted for by absorbing latents rather than main latents.
- SAELens supports loading pretrained SAEs with a single API call (`SAE.from_pretrained`), making training-free analysis straightforward.
- The first-letter spelling task used by Chanin et al. can be extended to other hierarchical feature types by designing appropriate in-context learning prompts and linear probes.
- For cross-layer analysis, SAEBench's projection-based absorption metric is preferred over ablation for deeper layers. However, the projection approach may have its own limitations (e.g., probe quality dependence) that should be validated.
- Cui et al. (ICLR 2026) prove that full feature disentanglement is mathematically impossible under realistic sparsity. This theoretical limit should inform the interpretation of absorption rates — high absorption may be an inevitable consequence of representational interference rather than a fixable training artifact.

## 8. Updated Resource Verification (2026-05-01)

The following key repositories were verified as active and available:

| Resource | Verified | Stars | Last Updated | Notes |
|----------|----------|-------|-------------|-------|
| **SAELens** (decoderesearch/SAELens) | Yes | 1,354 | Active | Primary library for SAE training/analysis |
| **SAEBench** (adamkarvonen/SAEBench) | Yes | 162 | Active | 8-metric evaluation suite |
| **sae-spelling** (lasr-spelling/sae-spelling) | Yes | 14 | 2024-07-08 | Original absorption paper code |
| **LatentScalpel** (jwarczynski/LatentScalpel) | Yes | 0 | 2026-04-26 | New: SAE for diffusion language models |

### Newly Discovered Resources

| Resource | Link | Description |
|----------|------|-------------|
| **LatentScalpel** | https://github.com/jwarczynski/LatentScalpel | Sparse autoencoders for mechanistic interpretability of diffusion language models; trains SAEs on residual stream activations;，自动解释学习特征并运行因果干预 |

### GitHub Search Notes
- Search for "sparse autoencoder feature absorption OR SAE hierarchical" returned 0 results (too specific)
- General SAE repos: SAELens (1354 stars) remains dominant
- SAEBench (162 stars) is the standard evaluation framework
- sae-spelling (14 stars) is the original feature absorption code
- New entrants like LatentScalpel show growing interest in applying SAEs to diffusion models
