# Literature Survey Report

**Research Topic**: Systematic Analysis and Quantification of Feature Absorption in Sparse Autoencoders (SAEs): Causes, Patterns, and Impact on Interpretability
**Survey Date**: 2026-04-23 (Updated)
**arXiv Search Keywords**: ["sparse autoencoder feature absorption", "SAE feature splitting absorption", "Matryoshka sparse autoencoder", "hierarchical sparse autoencoder", "orthogonal sparse autoencoder OrtSAE", "TopK SAE interpretability", "superposition polysemanticity", "JumpReLU Gated SAE"]
**Web Search Keywords**: ["SAE feature absorption mechanism 2025", "SAEBench benchmark absorption evaluation", "Matryoshka SAE absorption reduction", "sparse autoencoder GitHub open source", "feature hedging correlated features", "crosscoder multi-layer SAE", "Gated SAE Rajamanoharan", "Elhage superposition 2022"]

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant paradigm for mechanistic interpretability of large language models (LLMs). The theoretical foundation traces to the **superposition hypothesis** proposed by Elhage et al. (2022) at Anthropic, which posits that neural networks represent more independent features than they have neurons by encoding features as linear combinations of activations. This superposition leads to **polysemanticity**---individual neurons responding to multiple unrelated concepts simultaneously---making direct neuron-level interpretation intractable.

Bricken et al. (2023) at Anthropic demonstrated that SAEs could successfully extract monosemantic features from a one-layer transformer, sparking explosive growth in the field. Cunningham et al. (2023) at OpenAI further scaled SAE training, and by 2024-2025, SAE research spans multiple architectures (ReLU, TopK, JumpReLU, Gated, Matryoshka), comprehensive benchmarks (SAEBench), and applications beyond language models (vision, proteins, RNA, recommender systems).

**Feature absorption** stands out as one of the most theoretically consequential failure modes. Identified by Chanin et al. (2024) in "A is for Absorption," it describes how sparsity incentives cause parent features in a semantic hierarchy to be subsumed by their child features---creating "holes" in feature coverage and undermining the reliability of SAE-based interpretability. For example, a "starts with S" feature may fail to fire on "short" because that specific case has been absorbed into a child "short" feature. This discovery has catalyzed a wave of follow-up work: benchmark integration (SAEBench), hierarchical architectures (Matryoshka SAEs, HSAEs), orthogonality constraints (OrtSAE), and theoretical analyses of training dynamics (feature hedging, bias adaptation, unified theory).

The current state of the field is characterized by an active tension between improving reconstruction/sparsity trade-offs and ensuring that learned features are not merely interpretable-looking but causally faithful and structurally sound. A critical insight from SAEBench (Karvonen et al., 2025) is that **gains on proxy metrics do not reliably translate to better practical performance**---the sparsity-fidelity frontier does not predict downstream interpretability outcomes.

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | Toy Models of Superposition | Anthropic (Transformer Circuits) | 2022 | Formalized superposition hypothesis; showed networks encode more features than neurons via linear combinations; established theoretical basis for SAEs. | Toy models only; limited to synthetic settings. |
| 2 | Towards Monosemanticity: Decomposing Language Models with Dictionary Learning | Bricken et al., Anthropic | 2023 | Foundational SAE work; demonstrated monosemantic feature recovery in a 512-neuron MLP; showed features are significantly more monosemantic than neurons. | Single-layer transformer; limited scale; did not identify absorption. |
| 3 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | Cunningham et al., OpenAI | 2023 | Scaled SAE training; demonstrated interpretable features at larger scale; further validated monosemanticity hypothesis. | Limited evaluation of failure modes. |
| 4 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | Chanin et al. (MATS) | 2024 | First systematic study of feature absorption; defined absorption metric based on first-letter classification; showed sparsity regularization causes absorption. | Toy-model focus (first-letter tasks); no general theoretical solution proposed. |
| 5 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | Rajamanoharan et al. | 2024 | Introduced JumpReLU activation with learnable per-feature thresholds; prevents dead features; improves reconstruction fidelity. | Does not address absorption directly. |
| 6 | Improving Dictionary Learning with Gated Sparse Autoencoders | Rajamanoharan et al., NeurIPS 2024 | 2024 | Decoupled feature detection (gating path) from magnitude estimation; achieved Pareto improvements in sparsity-reconstruction trade-off; overcomes shrinkage bias. | Human interpretability not significantly better than baseline. |
| 7 | BatchTopK Sparse Autoencoders | Bussmann et al. | 2024 | Introduced batch-level TopK sparsity enforcement; outperforms L1 regularization; enables better feature disentanglement. | Can still suffer from absorption in wide dictionaries. |
| 8 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Templeton et al., Anthropic | 2024 | Scaled SAEs to frontier models; extracted millions of features; demonstrated safety-relevant feature detection. | Limited discussion of absorption at scale. |
| 9 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders | Karvonen et al. | 2025 | Standardized 8-evaluation benchmark across 200+ SAEs; adapted absorption metric for all layers; interactive leaderboard at neuronpedia.org/sae-bench. | Limited model coverage (mostly Gemma-2, Pythia); absorption eval expensive (~26 min per SAE). |
| 10 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | Bussmann et al., ICML 2025 | 2025 | Nested dictionaries of increasing size trained simultaneously; dramatically reduced absorption rates (0.05 vs. 0.49); learns hierarchical features. | ~50% computational overhead; inner levels suffer feature hedging. |
| 11 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | Chanin | 2025 | Identified feature hedging as opposite failure mode to absorption (reconstruction-loss-driven); showed Matryoshka SAEs trade absorption for hedging; proposed balanced variant. | Single model (Gemma-2-2b); focuses on narrow-width regime. |
| 12 | A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of LLMs | arXiv:2503.05613 | 2025 | Comprehensive survey covering architectures, evaluation metrics (structural, functional, monosemanticity), applications, and limitations. | Survey format; limited novel contributions. |
| 13 | A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability | arXiv:2512.05534 | 2025 | Identified polysemanticity, dead neurons, and feature absorption as fundamental structural properties of SDL optimization; piecewise biconvexity analysis. | Theoretical focus; limited empirical validation. |
| 14 | Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | Enforced orthogonality between SAE latents; reduced absorption by 65% and composition by 15%; avoids Matryoshka overhead. | Limited scale evaluation; adds 4-11% compute overhead. |
| 15 | Provable Feature Recovery via Sparse Autoencoders | arXiv:2506.14002 | 2025 | First SAE algorithm with theoretical recovery guarantees; introduced bias adaptation and Group Bias Adaptation (GBA). | Theory assumes specific generative model; validated up to 1.5B params. |
| 16 | Are Sparse Autoencoders Useful? A Case Study in Sparse Probing | arXiv:2502.16681 | 2025 | Critical evaluation showing SAEs do not consistently outperform strong non-SAE baselines (linear probes) on downstream probing tasks. | Does not isolate absorption as the cause of underperformance. |
| 17 | Measuring Sparse Autoencoder Feature Sensitivity | arXiv:2509.23717 | 2025 | Introduced feature sensitivity as a new evaluation dimension; found many "interpretable" features have poor sensitivity. | Complementary to absorption; not directly addressing hierarchy. |
| 18 | Signal in the Noise: Polysemantic Interference Transfers and Predicts Cross-Model Influence | arXiv:2505.11611 | 2025 | Showed polysemantic interference transfers across models and predicts cross-model influence; connects superposition to model behavior. | Focuses on polysemanticity rather than absorption specifically. |

---

## 3. SOTA Methods and Benchmarks

### Current Best SAE Architectures (per SAEBench 2025)

| Architecture | Best For | Key Mechanism | Absorption Performance |
|-------------|----------|---------------|----------------------|
| **Matryoshka BatchTopK** | Overall (5 of 8 metrics) | Nested dictionaries with independent reconstruction per level | Best (lowest absorption) |
| **BatchTopK** | Low-L0 sparsity (<100) | Batch-level TopK activation | Good |
| **JumpReLU** | Reconstruction fidelity | Learnable per-feature thresholds | Moderate |
| **Gated SAE** | Sparsity-reconstruction Pareto | Decoupled gating path (detection) + magnitude path (estimation) | Moderate |
| **Standard ReLU** | 1-sparse probing (L0 > 200) | L1 regularization | Worst overall |
| **OrtSAE** | Feature orthogonality | Orthogonal constraints on latents | Promising (early results) |

### Architecture Details

**Gated SAE** (Rajamanoharan et al., NeurIPS 2024): Decouples feature detection from magnitude estimation:
- Gating path: `1[(W_gate(x - b_dec) + b_gate) > 0]` (binary activation decision)
- Magnitude path: `ReLU(W_mag(x - b_dec) + b_mag)` (activation strength)
- Sparsity penalty applied only to gating path, overcoming shrinkage bias
- Achieves Pareto improvements; comparable interpretability to baseline SAEs

**JumpReLU SAE** (Rajamanoharan et al., 2024): Uses learnable per-feature thresholds instead of fixed zero threshold:
- Prevents "dead" features that never activate
- Improves reconstruction fidelity across sparsity levels

**Matryoshka SAE** (Bussmann et al., ICML 2025): Nested dictionaries of increasing size:
- Loss sums reconstruction across multiple prefix sizes: `L = sum_m (||a - a_hat_m||^2 + lambda * S_m) + alpha * L_aux`
- Early latents learn general concepts; later latents learn specific concepts
- Dramatically reduces absorption but introduces feature hedging in inner levels

### Current Best Methods (Absorption-Specific)

| Method | Core Innovation | Absorption Impact | Trade-off |
|--------|----------------|-------------------|-----------|
| **BatchTopK SAE** | Explicit k-sparsity selection per batch | Lower absorption than ReLU SAEs | Reconstruction fidelity |
| **Matryoshka SAE** | Nested dictionaries with multi-scale reconstruction | Absorption rate ~0.05 vs. 0.49 (BatchTopK) | Feature hedging in inner levels |
| **Balanced Matryoshka SAE** | Tuned loss coefficients across hierarchy levels | Better absorption-hedging Pareto frontier | Additional hyperparameter tuning |
| **OrtSAE** | Chunk-wise orthogonality penalty on decoder weights | -65% absorption, -15% composition | ~4-11% compute overhead |
| **HSAE** | Explicit tree-structured parent-child constraints | Reduced absorption in hierarchical settings | Complex architecture, early validation |
| **GBA (Group Bias Adaptation)** | Adaptive bias for frequency-matched activation sparsity | Theoretically avoids absorption under model assumptions | Empirical validation up to 1.5B params |

### Mainstream Benchmarks

- **SAEBench** (Karvonen et al., 2025): The dominant community benchmark. Includes 8 evaluations organized by capability:
  1. **Feature Absorption** (Concept Detection): Measures compensation by non-primary latents
  2. **Sparse Probing** (Concept Detection): How precisely individual latents correspond to meaningful concepts
  3. **LLM Automated Interpretability** (Interpretability): Human-understandability using LLM-as-judge
  4. **Loss Recovered** (Reconstruction): Faithfulness of SAE in preserving original model behavior
  5. **Spurious Correlation Removal (SCR)** (Feature Disentanglement): Removing unwanted correlations
  6. **Targeted Probe Perturbation (TPP)** (Feature Disentanglement): Whether concepts are encoded by distinct latents
  7. **RAVEL** (Feature Disentanglement): Resolving Attribute-Value Entanglements in Language Models
  8. **Unlearning** (Feature Disentanglement): Selective knowledge removal while preserving capabilities

- **Absorption Evaluation Protocol** (from SAEBench, based on Chanin et al. 2024):
  - Uses a **first-letter classification task** with prompts like "{word} has the first letter:" for all 26 letters
  - Train ground-truth linear probes for each letter
  - Use **k-sparse probing** (k=1..10) to identify "main" latents for each letter
  - Detect feature splitting via F1 threshold tau_fs = 0.03
  - Compute absorption score as the fraction of ground-truth probe projection captured by "absorbing" latents vs. "main" latents
  - Formula: `Absorption = (sum_{i in S_abs} a_i * d_i . p) / (sum_{i in S_abs} a_i * d_i . p + sum_{i in S_main} a_i * d_i . p)`
  - Report `1 - absorption_score` (higher is better); lower raw absorption is better

- **Key SAEBench Findings**:
  - **Best Overall**: Matryoshka BatchTopK outperforms on 5 of 8 metrics, especially L0 range 40-200
  - **Best Low-L0**: BatchTopK is clear winner in high-sparsity regime
  - **Worst Overall**: ReLU SAE outperformed on 5 of 8 metrics; worst loss recovered
  - **ReLU Exception**: Best at 1-sparse probing when L0 > 200 (above typical range)
  - Interactive leaderboard: [neuronpedia.org/sae-bench](https://neuronpedia.org/sae-bench)

### Evaluation Metrics

| Metric | What It Measures | Typical Target |
|--------|------------------|----------------|
| L0 | Average active features per token | 50-200 |
| CE Loss Score | Cross-entropy recovered vs. original | 80-95% |
| Explained Variance | Reconstruction quality | >90% |
| Absorption Score | Degree of parent-feature subsumption | As low as possible |
| Absorption Fraction | Detects partial absorption where parent still fires weakly | Lower is better |
| Feature Splitting Rate | Fragmentation of concepts | Context-dependent |
| PW-MCC | Pairwise dictionary mean correlation (consistency) | ~0.80+ |

### Recommended Sparsity Settings (L0)

| Use Case | Recommended L0 | Rationale |
|----------|---------------|-----------|
| Human Interpretability | Lower L0 (higher sparsity) | Cleaner individual features |
| Reconstruction & RAVEL | Higher L0 | Better fidelity and disentanglement |
| Balanced Compromise | 50-150 | Best across most metrics |

---

## 4. Identified Research Gaps

- **Gap 1: Theoretical understanding of absorption dynamics.** While Chanin et al. (2024) identified absorption and Bussmann et al. (2025) mitigated it, there is no unified theoretical framework predicting *when* absorption will occur for a given feature hierarchy, SAE width, and sparsity level. The unified theory paper (arXiv:2512.05534) identifies absorption as a fundamental property but does not provide predictive models.

- **Gap 2: Absorption metric construct validity.** The SAEBench absorption metric uses a first-letter classification task as a proxy for general feature absorption. Whether this task generalizes to other types of hierarchical features (semantic, syntactic, factual) is unknown. No study has validated whether lower absorption scores on the first-letter task correlate with better interpretability in real-world use cases.

- **Gap 3: Scalable causal validation of absorbed features.** Existing metrics are correlational (probe-based). There is limited work on causal interventions (activation patching, ablation) that definitively establish whether a latent "knows about" vs. "uses" a parent feature, especially under absorption.

- **Gap 4: Cross-architecture absorption patterns.** Most absorption studies focus on residual-stream SAEs. How absorption manifests in attention-output SAEs, MLP SAEs, crosscoders, or multimodal (vision-language) SAEs remains underexplored.

- **Gap 5: Layer-wise absorption patterns.** Absorption may vary systematically across model layers (early vs. middle vs. late), but this has not been systematically documented. SAEBench's extension of the absorption metric to all layers is a step forward, but layer-specific patterns remain uncharacterized.

- **Gap 6: Training dynamics and emergence time.** It is unclear at what point during SAE training absorption emerges, whether it is reversible, and how curriculum learning or temporal masking might prevent it.

- **Gap 7: Interaction between absorption and other failure modes.** Feature absorption, feature hedging, feature splitting, and dead neurons are studied largely in isolation. The trade-offs between these failure modes (e.g., Matryoshka SAEs reduce absorption but increase hedging) are not systematically characterized across diverse feature types and model layers.

- **Gap 8: Unified objective trade-offs.** No single training objective dominates across all interpretability goals. Methods that reduce absorption (Matryoshka, OrtSAE) introduce new pathologies (hedging, overhead) or trade off reconstruction fidelity.

- **Gap 9: Real-world downstream impact.** Kantamneni et al. (2025) showed SAEs often fail to outperform baselines on sparse probing. The causal link between absorption rates and downstream task underperformance has not been systematically quantified.

- **Gap 10: Absorption in non-language domains.** While SAEs are increasingly applied to vision, proteins, and other domains, absorption has only been studied in language models. Whether absorption manifests similarly in other modalities is unknown.

---

## 5. Available Resources

### Open-source Code

| Resource | Description | License | Link |
|----------|-------------|---------|------|
| **SAELens** | Primary PyTorch library for training and analyzing SAEs on LLMs. Supports standard, gated, TopK, JumpReLU, Matryoshka architectures. Integrates with TransformerLens and Neuronpedia. | Open Source | [github.com/jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) |
| **SAEBench** | Comprehensive benchmarking suite with absorption evaluation, AutoInterp, sparse probing, SCR, RAVEL, TPP, unlearning. PyPI: `sae-bench`. | Open Source | [github.com/adamkarvonen/SAEBench](https://github.com/adamkarvonen/SAEBench) |
| **TransformerLens** | Essential for extracting activations and running causal interventions on transformer models. | MIT | [github.com/neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens) |
| **Matryoshka SAE** | Official implementation of Matryoshka SAEs by Bussmann et al. | Open Source | [github.com/bartbussmann/matryoshka_sae](https://github.com/bartbussmann/matryoshka_sae) |
| **Sparsify** | Lean TopK SAE training library (EleutherAI). | Open Source | [github.com/EleutherAI/sparsify](https://github.com/EleutherAI/sparsify) |
| **OpenAI sparse_autoencoder** | GPT-2 small SAEs + visualizer. | Open Source | [github.com/openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder) |
| **crosscode** | Multi-layer/model crosscoders, transcoders, TopK/BatchTopK/GroupMax, JumpReLU. | Open Source | [github.com/oli-clive-griffin/crosscode](https://github.com/oli-clive-griffin/crosscode) |
| **mlsae** | Multi-Layer Sparse Autoencoders (ICLR 2025) by Tim Lawson et al. | Open Source | [github.com/tim-lawson/mlsae](https://github.com/tim-lawson/mlsae) |
| **SAELens-V** | Extension for vision-language models (LLaVA-NeXT, Chameleon). | Open Source | [github.com/saev-2025/SAELens-V](https://github.com/saev-2025/SAELens-V) |

### Datasets

- **The Pile / pile-uncopyrighted** (`monology/pile-uncopyrighted`): Standard training corpus for SAEs.
- **OpenWebText**: Common alternative for GPT-2 scale models.
- **First-letter classification tokens** (from Chanin et al. 2024): Controlled synthetic evaluation set for absorption measurement. Prompts like "{word} has the first letter:" for all 26 letters.
- **RAVEL dataset**: For evaluating attribute-value entanglements in language models.
- **SAEBench evaluation suite**: Pre-configured evaluation pipelines for all 8 metrics.

### Pretrained Models and SAEs

- **GPT-2 Small SAEs** (`gpt2-small-res-jb` release on SAELens): Widely used baseline for absorption and interpretability research.
- **Gemma-2-2B SAEs** (`gemma-2b-res` release): Used in Matryoshka SAE and OrtSAE papers.
- **Gemma Scope** (Lieberum et al., 2024): Large-scale open SAEs across layers and model sizes.
- **Pythia SAEs**: Available through SAELens; used in SAEBench and multi-layer SAE research.
- **SAEBench leaderboard**: 200+ SAEs across 7-8 architectures browsable at [neuronpedia.org/sae-bench](https://neuronpedia.org/sae-bench)

---

## 6. Implications for Idea Generation

**Directions worth exploring:**
- **Training-dynamic analysis of absorption:** Designing experiments that track when and how absorption emerges during SAE training, and whether early-stopping or curriculum strategies can prevent it.
- **Causal intervention benchmarks for absorption:** Moving beyond probe-based metrics to establish causal criteria (e.g., activation patching, ablation) that verify whether absorbed parent features are truly "missing" or merely "hidden."
- **Cross-layer absorption propagation:** Studying whether absorption in early residual-stream SAEs propagates to and compounds in deeper layers.
- **Quantifying the downstream cost of absorption:** Systematically varying absorption rates (via architecture or hyperparameters) and measuring sparse probing, steering, and safety-monitoring performance to establish a dose-response relationship.
- **Construct validity of absorption metric:** The first-letter task may not generalize. A study systematically validating the absorption metric against ground-truth hierarchical features (semantic, syntactic, factual) would be valuable.
- **Cross-architecture controlled analysis:** Varying only one architectural component at a time (activation function, dictionary size, sparsity mechanism) to isolate which components most affect absorption.
- **Layer-wise absorption patterns:** Systematic documentation of how absorption varies across model layers could reveal whether certain layers are inherently more prone to absorption.
- **Theoretical model of absorption:** Developing a predictive model for absorption severity based on feature correlation structure, dictionary size, and sparsity pressure.
- **Alternative mitigation strategies:** Beyond architectural changes, exploring training objective modifications or post-processing approaches.

**Saturated or crowded directions:**
- Simply proposing a new SAE architecture and showing lower absorption on first-letter tasks is becoming crowded (Matryoshka, OrtSAE, HSAE, GBA all exist).
- Purely descriptive studies of absorption in a single model/layer are unlikely to be novel.

**Cross-domain analogies with potential:**
- **Dictionary learning in signal processing:** The concept of "incoherence" in compressed sensing parallels OrtSAE's orthogonality penalty; more principled incoherence constraints could be imported.
- **Hierarchical topic models (LDA, HDP):** The explicit tree-structured priors in topic modeling could inspire probabilistic SAE variants with built-in hierarchical structure.
- **Sparse coding in neuroscience:** Olshausen & Field (1996/1997) sparse coding work may provide theoretical insights for feature decomposition.
- **Non-negative matrix factorization:** Faces similar part-whole decomposition challenges that may inform SAE hierarchical feature learning.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens (training + analysis) | High | MIT | Adopt | Dominant community library; supports multiple architectures out of the box. |
| SAEBench (absorption eval) | High | Open source | Adopt | Standardized benchmark; directly provides the absorption metric and evaluation protocol from Chanin et al. (2024). |
| TransformerLens (activation extraction) | High | MIT | Adopt | De facto standard for mechanistic interpretability on transformers. |
| Matryoshka SAE (Bussmann et al.) | High | Open Source | Adopt | Official implementation available; best-performing architecture on absorption metrics. |
| crosscode (Griffin) | Medium | Open Source | Compose | For cross-layer/cross-model experiments if extending to multi-layer absorption analysis. |
| mlsae (Lawson et al.) | Medium | Open Source | Compose | For multi-layer SAE experiments studying absorption propagation across layers. |
| Neuronpedia (feature browser) | Low | N/A | Compose | Use for qualitative validation and feature inspection, not for quantitative experiments. |

### Recommended Tool Stack

For studying feature absorption systematically:

1. **Base framework**: SAELens + SAEBench (provides training, evaluation, and benchmarking infrastructure)
2. **Models**: Pythia-160M/410M (fast iteration) + Gemma-2-2B (standard benchmark model)
3. **Architectures to compare**: Standard ReLU, TopK, BatchTopK, JumpReLU, Gated, Matryoshka BatchTopK, OrtSAE
4. **Evaluation**: SAEBench absorption metric + custom validation tasks
5. **Visualization**: SAEBench plotting utilities + custom analysis scripts

**Highlight:**
- **Reusable evaluation framework:** SAEBench's absorption evaluation (`sae_bench/evals/absorption/`) is the highest-priority resource to adopt. It provides ground-truth probe training, k-sparse probing, and absorption scoring.
- **Reusable data pipeline:** SAELens's `ActivationsStore` and `LanguageModelSAERunnerConfig` provide standardized activation buffering and training loops.
- **Reusable pretrained models:** GPT-2 small and Gemma-2-2B SAEs from SAELens/HuggingFace provide immediate baselines without requiring expensive training from scratch.

---

## Sources

- [Toy Models of Superposition (Elhage et al., 2022)](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Towards Monosemanticity (Bricken et al., 2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Sparse Autoencoders Find Highly Interpretable Features (Cunningham et al., 2023)](https://arxiv.org/abs/2309.08600)
- [A is for Absorption (Chanin et al., 2024)](https://arxiv.org/abs/2409.14507)
- [JumpReLU SAE (Rajamanoharan et al., 2024)](https://arxiv.org/abs/2407.14435)
- [Gated SAE (Rajamanoharan et al., 2024)](https://arxiv.org/abs/2404.16014)
- [BatchTopK SAE (Bussmann et al., 2024)](https://arxiv.org/abs/2412.06410)
- [Scaling Monosemanticity (Templeton et al., 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [SAEBench (Karvonen et al., 2025)](https://arxiv.org/abs/2503.09532)
- [Matryoshka SAE (Bussmann et al., 2025)](https://arxiv.org/abs/2503.17547)
- [Feature Hedging (Chanin, 2025)](https://arxiv.org/abs/2505.11756)
- [SAE Survey (arXiv:2503.05613)](https://arxiv.org/abs/2503.05613)
- [Unified Theory of SDL (arXiv:2512.05534)](https://arxiv.org/abs/2512.05534)
- [OrtSAE (arXiv:2509.22033)](https://arxiv.org/abs/2509.22033)
- [Provable Feature Recovery (arXiv:2506.14002)](https://arxiv.org/abs/2506.14002)
- [Feature Sensitivity (arXiv:2509.23717)](https://arxiv.org/abs/2509.23717)
- [Are SAEs Useful? (arXiv:2502.16681)](https://arxiv.org/abs/2502.16681)
- [SAELens GitHub](https://github.com/jbloomAus/SAELens)
- [SAEBench GitHub](https://github.com/adamkarvonen/SAEBench)
- [Neuronpedia SAE-Bench Leaderboard](https://neuronpedia.org/sae-bench)
