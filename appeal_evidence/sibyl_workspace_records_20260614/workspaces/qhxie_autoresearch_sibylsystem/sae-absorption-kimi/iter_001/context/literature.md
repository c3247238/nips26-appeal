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
