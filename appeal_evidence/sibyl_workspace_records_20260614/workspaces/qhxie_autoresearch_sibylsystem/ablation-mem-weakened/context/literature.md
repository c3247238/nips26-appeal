# Literature Survey Report

**Research Topic**: Feature Absorption in Sparse Autoencoders (SAEs): Systematic Analysis and Quantification
**Survey Date**: 2026-04-30
**arXiv Search Keywords**: sparse autoencoder feature absorption, SAE superposition, SAE interpretability, feature splitting sparse autoencoder, hierarchical features SAE
**Web Search Keywords**: SAE feature absorption SOTA 2025, sparse autoencoder evaluation benchmark, SAELens GemmaScope, feature hedging correlated features SAE, SAE circuit tracing steering 2025, SAE theoretical foundation dictionary learning 2025

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant tool for mechanistic interpretability of large language models (LLMs), aiming to decompose polysemantic neural activations into sparse, monosemantic features. The field traces its theoretical foundation to the superposition hypothesis proposed by Elhage et al. (2022), which posits that neural networks represent more features than they have dimensions by encoding features in overlapping, approximately orthogonal directions. SAEs attack this problem through overcomplete dictionary learning with sparsity constraints.

The field has seen explosive growth from 2023 to 2025. OpenAI's scaling work (Gao et al., 2024) demonstrated that SAEs can scale to 16 million latents on GPT-4 with predictable scaling laws. Google DeepMind released GemmaScope (Lieberum et al., 2024), providing comprehensive pretrained JumpReLU SAEs for Gemma 2 models. The open-source ecosystem has matured around SAELens, SAEBench, and Neuronpedia, making SAE research accessible to the broader community.

However, a critical limitation has emerged: **feature absorption**. First systematically identified by Chanin et al. (2024), feature absorption occurs when hierarchical features co-occur and the SAE's sparsity objective incentivizes merging parent feature directions into child latents, creating "interpretability illusions" where latents appear monosemantic but have arbitrary false negatives. This phenomenon has been validated across hundreds of open-source SAEs and represents a fundamental challenge to the reliability of SAE-based interpretability.

---

## 2. Core References

| # | Title | Authors | Source | Year | Key Contribution | Limitations |
|---|-------|---------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | Chanin et al. | arXiv:2409.14507 | 2024/2025 | **Foundational work**: Defines feature absorption, develops detection metric, proves it is a logical consequence of sparsity loss under hierarchical features. Validates across hundreds of SAEs. | No general solution proposed; metric requires manual inspection |
| 2 | Scaling and Evaluating Sparse Autoencoders | Gao et al. (OpenAI) | arXiv:2406.04093 / ICLR 2025 | 2024/2025 | Introduces TopK SAEs; scales to 16M latents on GPT-4; establishes scaling laws; proposes comprehensive evaluation metrics | Does not address absorption specifically; reconstruction-focused |
| 3 | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | Lieberum et al. (DeepMind) | arXiv:2408.05147 / BlackboxNLP 2024 | 2024 | Releases comprehensive JumpReLU SAE suite for Gemma 2 (2B/9B/27B); enables large-scale community research | SAEs still exhibit absorption; no absorption-specific analysis |
| 4 | A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of LLMs | Various | arXiv:2503.05613 / ACL Findings 2025 | 2025 | Comprehensive survey covering SAE architectures, training, evaluation; includes absorption as functional metric | Survey-level; no new methodology |
| 5 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | Bussmann et al. | arXiv:2503.17547 / ICML 2025 | 2025 | Proposes nested SAE architecture to address splitting/absorption; reduces absorption from 0.49 to 0.05 | Introduces hedging trade-off; reconstruction fidelity cost |
| 6 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | Chanin et al. | arXiv:2505.11756 | 2025 | Identifies hedging as distinct failure mode from absorption; shows Matryoshka exacerbates hedging; proposes balanced Matryoshka | Solution requires tuning; not universally applicable |
| 7 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | Korznikov et al. | arXiv:2509.22033 / OpenReview | 2025 | Enforces decoder orthogonality; reduces absorption by 65%, composition by 15%; discovers 9% more unique features | Slightly lower explained variance; chunk-wise approximation |
| 8 | On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy | Cui et al. | arXiv:2506.15963 | 2025 | Formal identifiability analysis; proves conditions for recovering ground-truth features; proposes reweighted remedy | Theoretical; limited empirical validation on LLM SAEs |
| 9 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders | Karvonen et al. | arXiv:2503.09532 / ICML 2025 | 2025 | Standardized benchmark with 8 metrics including absorption; evaluates 200+ SAEs across 7 architectures | Proxy metrics may not correlate with practical utility |
| 10 | Transcoders Beat Sparse Autoencoders for Interpretability | Paulo et al. | arXiv:2501.18823 | 2025 | Shows transcoders achieve Pareto dominance over SAEs on interpretability metrics; proposes skip transcoders | Different objective (cross-layer vs. self-reconstruction) |
| 11 | Are Sparse Autoencoders Useful? A Case Study in Sparse Probing | Kantamneni et al. | ICML 2025 | 2025 | **Negative results**: SAEs do not consistently outperform strong non-SAE baselines on downstream probing tasks | Limited to probing tasks; other applications may differ |
| 12 | Toy Models of Superposition | Elhage et al. (Anthropic) | Anthropic Blog | 2022 | Foundational theory: superposition hypothesis, polysemanticity, toy model analysis | Toy models only; LLM behavior more complex |
| 13 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs | Rajamanoharan et al. | arXiv:2407.14435 | 2024 | JumpReLU activation improves reconstruction; basis for GemmaScope | No interpretability improvement claim |
| 14 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical SAEs | Various | arXiv:2602.11881 | 2026 | Jointly learns SAEs and parent-child relationships; recovers semantic hierarchies | Requires hierarchical structure assumption |
| 15 | Interpretability Illusions with Sparse Autoencoders | Various | arXiv:2505.16004 | 2025 | Shows SAE interpretations are vulnerable to minimal input changes; raises reliability concerns | Focus on adversarial robustness, not absorption |
| 16 | Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? | Various | arXiv:2602.14111 | 2026 | Frozen/random baselines achieve comparable performance to trained SAEs on Gemma-2-2B and Llama-3-8B; challenges whether SAEs learn meaningful features | Questions fundamental value of SAE training; may not generalize to all metrics |
| 17 | Towards Understanding the Robustness of Sparse Autoencoders | Various | arXiv:2604.18756 | 2026 | Training-free SAE insertion at inference time; 5x reduction in jailbreak success rate; sparsity-robustness tradeoff | Focus on safety/robustness, not absorption specifically |
| 18 | Low-rank Adapting Models for Sparse Autoencoders | Various | ICLR 2025 | 2025 | LoRA adapts LM around pretrained SAE (training-free for SAE); 30-55% loss reduction, 3-20x faster than e2e training | SAE itself is fixed; absorption not addressed |
| 19 | Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models | Pach et al. | NeurIPS 2025 | 2025 | First comprehensive VLM SAE framework (CLIP, LLaVA); SAE interventions on vision encoder steer multimodal outputs | Cross-modal; absorption in VLMs unexplored |
| 20 | Step-Level Sparse Autoencoder for Reasoning Process Interpretation | Various | arXiv:2603.03031 | 2026 | Step-level (not token-level) SAEs for multi-step reasoning interpretation; addresses fragmentation | Early work; limited empirical validation |
| 21 | **Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training** | Li et al. | arXiv:2510.08855 / ICLR-W 2025 | 2025 | **~40% reduction in feature absorption** via temporal EMA tracking of activation magnitudes/frequencies; maintains reconstruction quality | Limited to Gemma-2-2B; single GPU; no theoretical analysis |
| 22 | **SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data** | Chanin et al. | arXiv:2602.14687 | 2026 | Large-scale synthetic benchmark (16K features, 768 dims); reproduces LLM SAE phenomena; discovers Matching Pursuit overfitting | Synthetic data may not fully capture LLM complexity |
| 23 | **Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders** | Wang et al. | arXiv:2510.03659 / ICLR 2026 | 2025 | Weak correlation (tau_b ~ 0.3) between interpretability and steering utility; proposes Delta Token Confidence selection | Correlation analysis, not causal; limited model coverage |
| 24 | **Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures** | Various | arXiv:2506.01197 | 2025 | H-SAE improves probing and decreases absorption rate increase with width; systematic first-letter benchmark tests | Limited architectural variants tested |
| 25 | **Evaluating SAE Interpretability Without Explanations** | Various | arXiv:2507.08473 | 2025 | Explanation-free interpretability evaluation; addresses reliability concerns of LLM-generated explanations | Does not directly measure absorption |
| 26 | **Interpretable and Steerable Concept Bottleneck Sparse Autoencoders** | Various | arXiv:2512.10805 | 2025 | CB-SAE: +32.1% interpretability, +14.5% steerability; finds most SAE neurons have low interpretability or steerability | Concept bottleneck requires predefined concepts |
| 27 | **Self-Ablating Transformers: More Interpretability, Less Sparsity** | Various | arXiv:2505.00509 | 2025 | Alternative to SAEs: built-in interpretability via self-ablation; challenges need for external sparse decomposition | Different paradigm; not directly comparable |
| 28 | **Sparse Autoencoder Features for Classifications and Transferability** | Gao et al. | EMNLP 2025 / arXiv:2502.11367 | 2025 | Systematic study of SAE feature transferability across layers and tasks | Transferability does not imply absence of absorption |
| 29 | **On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability** | Tang et al. | arXiv:2512.05534 | 2025 | Unified theoretical framework; first theoretical explanation for absorption as spurious local minima; introduces "feature anchoring" | Theoretical; limited empirical validation on real LLMs |
| 30 | **Sparse Autoencoders Enable Scalable and Reliable Circuit Identification in Language Models** | O'Neill et al. | arXiv:2405.12522 | 2024 | Discrete SAEs for efficient circuit discovery; higher precision/recall; runtime from hours to seconds | Limited to token-to-token tasks |
| 31 | **Route Sparse Autoencoder to Interpret Large Language Models** | Shi et al. | arXiv:2503.08200 | 2025 | Multi-layer routing SAE; extracts 22.5% more features with 22.3% higher interpretability score | Complex architecture; limited absorption analysis |
| 32 | **SplInterp: Improving our Understanding and Training of Sparse Autoencoders** | Budd et al. | arXiv:2505.11836 | 2025 | Theoretical framework using spline theory; characterizes TopK SAE geometry with power diagrams | Theoretical focus; limited empirical absorption study |
| 33 | **OpenAI Scaling SAE** | Adamek et al. | arXiv:2508.15841 | 2025 | Trained 16M feature autoencoder on GPT-4 with 40B tokens; discovered clean scaling laws | Reconstruction-focused; limited interpretability analysis |
| 34 | **CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features** | Arad et al. | arXiv:2508.12535 | 2025 | Correlation-based SAE feature selection for improved task performance and safety | Steering-focused; does not address absorption |
| 35 | **From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit** | Various | arXiv:2506.03093 | 2025 | MP-SAE with matching pursuit; finds Matryoshka preserves hierarchy but loses flat structure | Limited absorption analysis |
| 36 | **Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs** | Various | arXiv:2505.20254 | 2025 | Argues for feature consistency as primary evaluation criterion; critiques current metrics | Position paper; no new experiments |
| 37 | **Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?** | Various | arXiv:2602.14111 | 2026 | Frozen/random baselines achieve comparable performance to trained SAEs on Gemma-2-2B and Llama-3-8B | Questions fundamental value of SAE training |
| 38 | **Understanding SAE Scaling in the Presence of Feature Manifolds** | Various | arXiv:2509.02565 | 2025 | Analyzes how feature manifolds affect SAE scaling behavior | Theoretical; limited real-world validation |

---

## 3. SOTA Methods and Benchmarks

### 3.1 SAE Architectures (2024-2025)

| Architecture | Key Innovation | Absorption Impact | Year |
|-------------|---------------|-------------------|------|
| **ReLU + L1** (Standard) | L1 sparsity penalty | High absorption | 2023 |
| **TopK SAE** (Gao et al.) | Direct k-sparsity control | Moderate absorption | 2024 |
| **JumpReLU SAE** (Rajamanoharan et al.) | Learnable activation threshold | Moderate absorption | 2024 |
| **Gated SAE** (Rajamanoharan et al.) | Decouples detection from magnitude | Moderate absorption | 2024 |
| **BatchTopK SAE** (Bussmann et al.) | Batch-level top-k selection | Moderate absorption | 2024 |
| **Matryoshka SAE** (Bussmann et al.) | Nested multi-level dictionaries | Reduced absorption (0.05 vs 0.49) | 2025 |
| **OrtSAE** (Korznikov et al.) | Decoder orthogonality constraint | Reduced absorption (-65%) | 2025 |
| **Balance Matryoshka** (Chanin et al.) | Tunable level coefficients | Cancels hedging + absorption | 2025 |
| **CB-SAE** (Various) | Concept bottleneck constraint | +32.1% interpretability, +14.5% steerability | 2025 |
| **ATM** (Li et al.) | Adaptive Temporal Masking with EMA tracking | ~40% reduction in absorption | 2025 |
| **H-SAE** (Various) | Explicit hierarchical semantics incorporation | Decreases absorption rate increase with width | 2025 |
| **Discrete SAE** (O'Neill et al.) | Hard discrete codes for circuit discovery | Enables efficient circuit tracing; limited absorption analysis | 2024 |
| **RouteSAE** (Shi et al.) | Multi-layer routing across layers | Improved interpretability; limited absorption analysis | 2025 |

### 3.2 Evaluation Benchmarks

**SAEBench** (Karvonen et al., ICML 2025) is the leading comprehensive benchmark:
- 8 metrics: Feature Absorption, AutoInterp, L0/Loss Recovered, RAVEL, SCR, TPP, Sparse Probing, Unlearning
- 200+ SAEs across 7 architectures
- GitHub: https://github.com/adamkarvonen/SAEBench

**CE-Bench** (EMNLP-W 2025): LLM-free contrastive evaluation; 77.3% CRPR alignment with SAEBench.

**Feature Absorption Metric** (Chanin et al.): Measures whether a latent captures multiple independent concepts. Two variants:
- **Mean absorption**: Average across all latents
- **Full absorption**: Fraction of latents with any absorption

**Circuit Tracing Metrics** (O'Neill et al., 2024; Anthropic 2025): Precision and recall for circuit discovery using discrete SAEs and cross-layer transcoders (CLTs).

**Concept Separability Score** (Fereidouni et al., 2025): Jensen-Shannon distance-based metric for monosemanticity evaluation.

### 3.3 Key Datasets and Models

| Resource | Description | Access |
|----------|-------------|---------|
| **GemmaScope** | JumpReLU SAEs for Gemma 2 (2B/9B/27B) | HuggingFace: google/gemma-scope |
| **LlamaScope** | SAEs for Llama 3.1 | SAELens / Neuronpedia |
| **OpenAI SAEs** | GPT-2 small SAEs | GitHub: openai/sparse_autoencoder |
| **SAELens** | Training/loading library | GitHub: decoderesearch/SAELens |
| **Neuronpedia** | Interactive feature explorer | neuronpedia.org |

---

## 4. Identified Research Gaps

- **Gap 1: Quantification of absorption prevalence across model scales and layers.** While Chanin et al. validated absorption exists, there is no systematic cross-model, cross-layer quantification of how absorption rates vary with model size, layer depth, or SAE configuration.

- **Gap 2: Relationship between absorption and downstream interpretability tasks.** No work has systematically measured how absorption affects circuit discovery, concept erasure, or steering tasks. Kantamneni et al.'s negative results on probing do not directly address absorption's impact.

- **Gap 3: Theoretical understanding of absorption in real LLM feature hierarchies.** Current theory (Chanin et al.'s toy model, Cui et al.'s identifiability) assumes simplified feature structures. Real LLM features have complex correlation structures that are not well-captured.

- **Gap 4: Absorption-aware SAE training objectives.** Existing solutions (Matryoshka, OrtSAE) are architectural modifications. No work has developed training objectives that explicitly penalize absorption while maintaining reconstruction quality.

- **Gap 5: Absorption in non-language domains.** Most absorption research focuses on LLMs. Its prevalence in vision models, multimodal models, or scientific applications is unexplored. The NeurIPS 2025 VLM SAE work (Pach et al.) opens this direction but does not address absorption.

- **Gap 6: Temporal dynamics of absorption.** SAE features may absorb/desorb during training or as model capacity increases. ATM (Li et al., 2025) tracks temporal dynamics but only during training. No longitudinal studies exist for pretrained SAEs.

- **Gap 7: Fundamental validity of SAE features.** The "Sanity Checks" paper (arXiv:2602.14111, 2026) shows frozen/random SAE baselines match trained SAEs on multiple metrics, raising questions about whether absorption is a meaningful phenomenon or an artifact of massive dictionary sizes. A rigorous response to this challenge is needed for any absorption study to be credible.

- **Gap 8: Interpretability-utility disconnect.** Wang et al. (ICLR 2026) show weak correlation (~0.3) between interpretability and steering utility. It is unknown whether reducing absorption improves practical utility or merely improves interpretability scores.

- **Gap 9: Absorption on synthetic vs. real data.** SynthSAEBench (Chanin et al., 2026) enables controlled synthetic experiments, but the correspondence between synthetic absorption rates and real LLM absorption rates is unvalidated.

- **Gap 10: Cross-architecture absorption comparison.** Most absorption studies focus on a single architecture family. No systematic comparison exists across ReLU, TopK, JumpReLU, Gated, and Matryoshka SAEs using identical evaluation protocols.

---

## 5. Available Resources

### Open-source Code

| Repository | Description | License |
|-----------|-------------|---------|
| [SAELens](https://github.com/decoderesearch/SAELens) | Training and analyzing SAEs on language models | MIT |
| [SAEBench](https://github.com/adamkarvonen/SAEBench) | Comprehensive SAE evaluation benchmark | MIT |
| [OpenAI SAE](https://github.com/openai/sparse_autoencoder) | OpenAI's SAE training code + GPT-2 SAEs | MIT |
| [feature-hedging-paper](https://github.com/chanind/feature-hedging-paper) | Code for Feature Hedging paper | Unknown |
| [TransformerLens](https://github.com/neelnanda-io/TransformerLens) | Hook-based model introspection | GPL-3.0 |
| [nnsight](https://github.com/ndif-team/nnsight) | Framework-agnostic interventions | Apache-2.0 |
| [synth-sae-bench-experiments](https://github.com/decoderesearch/synth-sae-bench-experiments) | SynthSAEBench toolkit and experiments | Unknown |
| [sae-spelling](https://github.com/lasr-spelling/sae-spelling) | Official implementation of "A is for Absorption" paper | MIT |
| [sae_vis](https://github.com/callummcdougall/sae_vis) | Feature-centric and prompt-centric visualizations for SAEs | MIT |
| [Llamascopium](https://github.com/OpenMOSS/Llamascopium) | Framework for training, analyzing, visualizing SAEs | MIT |

### Datasets

- **OpenWebText**: Standard corpus for SAE training (used by GemmaScope, OrtSAE)
- **SAEBench evaluation datasets**: Contrastive stories, polysemous word pairs, spurious correlation datasets
- **Neuronpedia feature annotations**: Community-contributed feature descriptions

### Pretrained Models

- **GemmaScope**: 16K-1M latents, all Gemma 2 variants, all layers/sub-layers
- **LlamaScope**: Llama 3.1 8B SAEs
- **OpenAI GPT-2 SAEs**: Multiple scales available
- **Pythia SAEs**: Community-trained via SAELens

---

## 6. Implications for Idea Generation

### Directions Worth Exploring

1. **Systematic absorption quantification**: A large-scale study measuring absorption rates across model families (GPT-2, Gemma, Llama), layer depths, and SAE configurations. This directly addresses Gap 1 and would provide the community with reference baselines.

2. **Absorption impact on downstream tasks**: Design experiments that explicitly manipulate absorption levels and measure effects on circuit discovery accuracy, steering fidelity, or concept erasure success. This bridges the gap between absorption as a metric and absorption as a practical concern.

3. **Feature hierarchy recovery**: Building on HSAE (arXiv:2602.11881) and Matryoshka SAEs, develop methods that not only reduce absorption but explicitly recover the hierarchical structure of features. This is a natural extension with high interpretability value.

4. **Training-free absorption mitigation**: Given the project's training-free constraint, explore post-hoc methods to detect and correct absorbed features without retraining SAEs. This could involve decoder weight analysis or activation pattern clustering.

### Saturated Directions

- **New SAE architectures**: The space is crowded (TopK, JumpReLU, Gated, Matryoshka, OrtSAE, Balance Matryoshka). Incremental architectural improvements face diminishing returns.
- **Scaling studies**: OpenAI (16M latents on GPT-4) and GemmaScope have covered large-scale training. Smaller-scale scaling is less impactful.
- **Automated interpretability scores**: SAEBench and CE-Bench provide standardized metrics. New metrics need strong justification.
- **Training-free SAE analysis without addressing the "Sanity Checks" challenge**: Any training-free study must explicitly address whether its findings hold against random/frozen baselines, or risk being dismissed as artifacts.
- **Absorption mitigation without utility validation**: Wang et al. (ICLR 2026) show that better interpretability scores do not imply better steering utility. Any absorption reduction must be validated on downstream tasks, not just metrics.

### Cross-Domain Analogies with Potential

- **Matrix factorization in recommendation systems**: Similar "absorption" phenomena exist where popular items dominate latent factors. Solutions from that domain (e.g., regularization, negative sampling) may transfer.
- **Topic modeling (LDA)**: Hierarchical topic models face analogous parent-child concept merging. Hierarchical LDA extensions could inspire SAE solutions.
- **Source separation in signal processing**: Independent Component Analysis (ICA) faces similar non-identifiability challenges when sources are correlated.

### Key Risks to Monitor

- **Negative results on SAE utility** (Kantamneni et al.; Wang et al., ICLR 2026): The field is increasingly skeptical of SAEs' practical value. The weak interpretability-utility correlation (~0.3) means absorption studies must demonstrate real downstream impact, not just metric improvements.
- **Transcoder competition**: Transcoders may supersede SAEs for interpretability. Consider whether absorption is specific to SAEs or general to sparse dictionary learning.
- **Interpretability illusions** (arXiv:2505.16004): SAE features may be fragile. Absorption could be one manifestation of a broader reliability problem.
- **Sanity check challenge** (arXiv:2602.14111): Frozen/random SAE baselines match trained SAEs on key metrics. Any absorption study must include random baseline comparisons to rule out dictionary-size artifacts.
- **Synthetic-real gap** (SynthSAEBench): Controlled synthetic experiments may not generalize to real LLMs. Findings need validation on pretrained SAEs.
- **Self-ablation alternative** (arXiv:2505.00509): Built-in interpretability via self-ablation challenges the fundamental need for external SAE decomposition.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens (training/loading) | High | MIT | **Adopt** | Mature library with GemmaScope/LlamaScope support; active maintenance |
| SAEBench (evaluation) | High | MIT | **Adopt** | Standardized benchmark including absorption metric; pip installable |
| TransformerLens (model hooks) | High | GPL-3.0 | **Adopt** | Essential for activation extraction; integrates with SAELens |
| OpenAI SAE code | Medium | MIT | **Reference** | Good reference for TopK implementation; not actively maintained |
| GemmaScope pretrained SAEs | High | Apache-2.0 | **Adopt** | Best available pretrained SAEs; comprehensive layer coverage |
| LlamaScope pretrained SAEs | High | Unknown | **Adopt** | Alternative model family for cross-model comparison |
| feature-hedging-paper code | Medium | Unknown | **Reference** | Reference for Matryoshka implementation details |
| Neuronpedia API | Medium | N/A | **Compose** | Useful for feature annotation lookup; not essential |
| auto-gemmascope | Medium | Unknown | **Reference** | Autonomous vs human-guided SAE feature finding in Gemma 3 VLM |

### Recommended Toolchain

```
SAELens (load pretrained SAEs)
    + TransformerLens (extract activations)
    + SAEBench (run absorption metric)
    + Custom analysis scripts (quantification study)
```

### Key Reusable Components

1. **SAEBench's absorption metric**: Directly reusable for measuring absorption rates
2. **SAELens's pretrained SAE loader**: One-line loading of GemmaScope/LlamaScope SAEs
3. **TransformerLens's HookPoint system**: Clean activation extraction at arbitrary layers
4. **GemmaScope SAE weights**: Eliminates need for SAE training; covers all layers of Gemma 2 2B/9B

---

## Sources

- [A is for Absorption](https://arxiv.org/abs/2409.14507) - Chanin et al., 2024/2025
- [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093) - Gao et al., OpenAI, 2024
- [Gemma Scope](https://arxiv.org/abs/2408.05147) - Lieberum et al., DeepMind, 2024
- [A Survey on Sparse Autoencoders](https://arxiv.org/abs/2503.05613) - ACL Findings 2025
- [Matryoshka Sparse Autoencoders](https://arxiv.org/abs/2503.17547) - Bussmann et al., ICML 2025
- [Feature Hedging](https://arxiv.org/abs/2505.11756) - Chanin et al., 2025
- [OrtSAE](https://arxiv.org/abs/2509.22033) - Korznikov et al., 2025
- [On the Limits of SAEs](https://arxiv.org/abs/2506.15963) - Cui et al., 2025
- [SAEBench](https://arxiv.org/abs/2503.09532) - Karvonen et al., ICML 2025
- [Transcoders Beat SAEs](https://arxiv.org/abs/2501.18823) - Paulo et al., 2025
- [Are Sparse Autoencoders Useful?](https://proceedings.mlr.press/v267/kantamneni25a.html) - Kantamneni et al., ICML 2025
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) - Elhage et al., Anthropic, 2022
- [JumpReLU SAEs](https://arxiv.org/abs/2407.14435) - Rajamanoharan et al., 2024
- [Hierarchical SAEs](https://arxiv.org/abs/2602.11881) - 2026
- [Interpretability Illusions](https://arxiv.org/abs/2505.16004) - 2025
- [CE-Bench](https://arxiv.org/abs/2509.00691) - EMNLP-W 2025
- [Sanity Checks for SAEs](https://arxiv.org/abs/2602.14111) - 2026
- [Robustness of SAEs](https://arxiv.org/abs/2604.18756) - 2026
- [Low-rank Adapting Models for SAEs](https://openreview.net/forum?id=lDPtsCYTwr) - ICLR 2025
- [SAE for VLMs](https://github.com/ExplainableML/sae-for-vlm) - NeurIPS 2025
- [Step-Level SAE](https://arxiv.org/abs/2603.03031) - 2026
- [ATM: Adaptive Temporal Masking](https://arxiv.org/abs/2510.08855) - Li et al., ICLR-W 2025
- [SynthSAEBench](https://arxiv.org/abs/2602.14687) - Chanin et al., 2026
- [Does Higher Interpretability Imply Better Utility?](https://arxiv.org/abs/2510.03659) - Wang et al., ICLR 2026
- [Hierarchical Semantics in SAEs](https://arxiv.org/abs/2506.01197) - 2025
- [Evaluating SAE Interpretability Without Explanations](https://arxiv.org/abs/2507.08473) - 2025
- [CB-SAE: Concept Bottleneck SAEs](https://arxiv.org/abs/2512.10805) - 2025
- [Self-Ablating Transformers](https://arxiv.org/abs/2505.00509) - 2025
- [SAE Features for Classification and Transferability](https://arxiv.org/abs/2502.11367) - Gao et al., EMNLP 2025
- [On the Theoretical Foundation of SDL](https://arxiv.org/abs/2512.05534) - Tang et al., 2025
- [Circuit Identification with SAEs](https://arxiv.org/abs/2405.12522) - O'Neill et al., 2024
- [RouteSAE](https://arxiv.org/abs/2503.08200) - Shi et al., 2025
- [SplInterp](https://arxiv.org/abs/2505.11836) - Budd et al., 2025
- [OpenAI Scaling SAE](https://arxiv.org/abs/2508.15841) - Adamek et al., 2025
- [CorrSteer](https://arxiv.org/abs/2508.12535) - Arad et al., 2025
- [From Flat to Hierarchical (MP-SAE)](https://arxiv.org/abs/2506.03093) - 2025
- [Position: Feature Consistency in SAEs](https://arxiv.org/abs/2505.20254) - 2025
- [Understanding SAE Scaling with Feature Manifolds](https://arxiv.org/abs/2509.02565) - 2025
- [SAELens GitHub](https://github.com/decoderesearch/SAELens)
- [SAEBench GitHub](https://github.com/adamkarvonen/SAEBench)
- [OpenAI SAE GitHub](https://github.com/openai/sparse_autoencoder)
- [sae-spelling GitHub](https://github.com/lasr-spelling/sae-spelling)
- [sae_vis GitHub](https://github.com/callummcdougall/sae_vis)
- [Llamascopium GitHub](https://github.com/OpenMOSS/Llamascopium)
- [Neuronpedia](https://www.neuronpedia.org)
