# Literature Survey Report

**Research Topic**: Systematic Analysis and Quantification of Feature Absorption in Sparse Autoencoders (SAEs): Causes, Patterns, and Impact on Interpretability
**Survey Date**: 2026-04-25 (Updated)
**arXiv Search Keywords**: ["feature absorption sparse autoencoder", "Matryoshka sparse autoencoder", "SAEBench benchmark", "monosemanticity", "sparse autoencoder interpretability", "OrtSAE orthogonal", "feature hedging", "sparse autoencoder scaling laws", "RouteSAE", "transcoder"]
**Web Search Keywords**: ["SAE feature absorption mechanism 2025", "SAEBench benchmark absorption evaluation", "Matryoshka SAE absorption reduction", "sparse autoencoder GitHub open source", "TopK JumpReLU BatchTopK comparison 2025", "RAVEL factual disentanglement SAE", "sparse autoencoder circuit tracing steering 2025", "sparse autoencoder polysemanticity metric 2025", "GemmaScope SAELens pretrained SAE"]

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant paradigm for mechanistic interpretability of large language models (LLMs), following Anthropic's foundational work on monosemanticity (Bricken et al., 2023). The core premise is that SAEs can decompose polysemantic neural activations into sparse, human-interpretable feature directions. Over 2024-2026, the field has shifted from demonstrating feasibility to rigorously diagnosing failure modes and building standardized benchmarks.

Feature absorption stands out as one of the most theoretically consequential failure modes. Identified by Chanin et al. (2024) in "A is for Absorption," it describes how sparsity incentives cause parent features in a semantic hierarchy to be subsumed by their child features---creating "holes" in feature coverage and undermining the reliability of SAE-based interpretability. This discovery has catalyzed a wave of follow-up work: benchmark integration (SAEBench), hierarchical architectures (Matryoshka SAEs, HSAEs), orthogonality constraints (OrtSAE), and theoretical analyses of training dynamics (feature hedging, bias adaptation). The current state of the field is characterized by an active tension between improving reconstruction/sparsity trade-offs and ensuring that learned features are not merely interpretable-looking but causally faithful and structurally sound.

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 (NeurIPS 2025 Oral) | 2024 | Introduced feature absorption as a sparsity-driven failure mode; proposed a detection metric based on first-letter probes; validated on hundreds of LLM SAEs. | Toy-model focus (first-letter tasks); no general theoretical solution proposed; metric requires ground-truth labels. |
| 2 | Towards Monosemanticity: Decomposing Language Models with Dictionary Learning | Transformer Circuits Thread | 2023 | Foundational SAE work; demonstrated monosemantic feature recovery in a 512-neuron MLP; introduced feature splitting concept. | Did not identify or address absorption; small scale. |
| 3 | Scaling and Evaluating Sparse Autoencoders | arXiv:2406.04093 (ICLR 2025) | 2024 | Large-scale SAE scaling study on GPT-2/GPT-4; established TopK SAEs and training best practices; 16M latent autoencoder on GPT-4. | Focused on aggregate metrics; absorption not a central concern. |
| 4 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders | arXiv:2503.09532 | 2025 | Standardized 8-evaluation benchmark including Feature Absorption, AutoInterp, Sparse Probing, SCR, RAVEL, TPP, Unlearning, L0/Loss Recovered. | Absorption evaluation is computationally expensive (~26 min per SAE); uses first-letter tasks exclusively. |
| 5 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | arXiv:2503.17547 | 2025 | Proposed nested dictionary SAEs that learn features at multiple scales; dramatically reduced absorption rates (0.05 vs. 0.49). | Inner levels act as narrow SAEs, exacerbating feature hedging (Chanin et al. 2025); +50% compute overhead. |
| 6 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | arXiv:2505.11756 | 2025 | Identified feature hedging as a reconstruction-loss-driven pathology in narrow SAEs; proposed balanced Matryoshka variant. | Focuses on narrow-width regime; less relevant to very wide SAEs. |
| 7 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders | arXiv:2602.11881 | 2026 | Proposed HSAE with explicit parent-child relationships and structural constraint loss to learn feature hierarchies. | Very recent preprint; limited community validation so far. |
| 8 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | Enforced decoder orthogonality via chunk-wise penalty; reduced absorption by 65% and composition by 15%; only ~4% compute overhead. | Adds ~4-11% compute overhead; chunk size is a new hyperparameter. |
| 9 | Are Sparse Autoencoders Useful? A Case Study in Sparse Probing | arXiv:2502.16681 (ICML 2025) | 2025 | Critical evaluation showing SAEs do not consistently outperform strong non-SAE baselines on downstream probing tasks (won only 2.2% of 113 tasks). | Does not isolate absorption as the cause of underperformance. |
| 10 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | Argued for feature consistency (PW-MCC metric) as a community priority; showed high consistency is achievable with TopK SAEs. | Consistency does not guarantee absence of absorption or causal validity. |
| 11 | Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders | arXiv:2506.14002 | 2025 | First SAE algorithm with theoretical feature recovery guarantees; introduced Group Bias Adaptation (GBA). | Theory assumes a specific generative model; real-world LLM features may not fit. |
| 12 | Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders | arXiv:2508.16560 | 2025 | Showed that incorrect L0 estimation leads to learning incorrect features; "sparsity-reconstruction tradeoff" plots are unsound evaluation. | Focuses on L0 estimation rather than absorption specifically. |
| 13 | RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations | arXiv:2402.17700 (ACL 2024) | 2024 | Introduced causal intervention benchmark for factual disentanglement; SAEs achieved lowest disentanglement scores among unsupervised methods. | Not specifically designed for absorption measurement. |
| 14 | Transcoders Beat Sparse Autoencoders for Interpretability | arXiv:2501.18823 | 2025 | Showed transcoders (mapping between layers) outperform SAEs on interpretability metrics; challenges SAE dominance. | Different paradigm; not directly comparable on absorption. |
| 15 | A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models | arXiv:2503.05613 (EMNLP 2025 Findings) | 2025 | Comprehensive survey covering SAE architectures, training, evaluation, and applications; identifies open challenges. | Survey paper; no novel experimental contributions. |
| 16 | Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds | arXiv:2509.02565 | 2025 | Theoretical framework for SAE capacity allocation; shows SAEs may over-allocate to common features at expense of rare ones. | Theoretical; limited empirical validation on absorption. |
| 17 | On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy | arXiv:2506.15963 | 2025 | Analyzes identifiability conditions for feature recovery from polysemantic inputs; proposes reweighted training remedy. | Focuses on identifiability theory; limited practical evaluation. |
| 18 | Sparse Autoencoders Can Interpret Randomly Initialized Transformers | arXiv:2501.17727 | 2025 | Shows SAEs extract similarly "interpretable" features from random models, challenging whether features reflect learned computations. | Raises fundamental questions about SAE validity; absorption may be exacerbated in random models. |
| 19 | Route Sparse Autoencoder to Interpret Large Language Models | arXiv:2503.08200 | 2025 | Multi-layer routing with shared SAE; +22.5% more features, +22.3% higher interpretability. | Does not explicitly evaluate absorption; routing may introduce new failure modes. |
| 20 | Evaluating SAE Interpretability without Explanations | arXiv:2507.08473 | 2025 | References absorption in broader SAE evaluation context; proposes explanation-free evaluation methods. | Does not focus on absorption specifically. |

---

## 3. SOTA Methods and Benchmarks

### Current Best Methods

| Method | Core Innovation | Absorption Impact | Trade-off |
|--------|----------------|-------------------|-----------|
| **BatchTopK SAE** | Explicit k-sparsity selection per batch | Lower absorption than ReLU SAEs | Reconstruction fidelity |
| **Matryoshka SAE** | Nested dictionaries with multi-scale reconstruction | Absorption rate ~0.05 vs. 0.49 (BatchTopK) | Feature hedging in inner levels; +50% compute |
| **Balanced Matryoshka SAE** | Tuned loss coefficients across hierarchy levels | Better absorption-hedging Pareto frontier | Additional hyperparameter tuning |
| **OrtSAE** | Chunk-wise orthogonality penalty on decoder weights | -65% absorption, -15% composition | ~4-11% compute overhead |
| **HSAE** | Explicit tree-structured parent-child constraints | Reduced absorption in hierarchical settings | Complex architecture, early validation |
| **GBA (Group Bias Adaptation)** | Adaptive bias for frequency-matched activation sparsity | Theoretically avoids absorption under model assumptions | Empirical validation up to 1.5B params |
| **TopK SAE** | Exactly K active features per token | Worse absorption than BatchTopK | Better reconstruction, consistent sparsity |
| **JumpReLU SAE** | Learned per-latent threshold | Better at high L0 than BatchTopK | More complex training dynamics |

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

- **CE-Bench** (Gulko et al., 2025): LLM-free contrastive evaluation using story pairs; >70% Spearman correlation with SAEBench.

- **RAVEL** (Huang et al., 2024): Causal intervention benchmark for factual disentanglement using activation patching.

- **SynthSAEBench-16k** (2026): Synthetic benchmark with known ground-truth features for fine-grained controlled evaluation.

### Evaluation Metrics

| Metric | What It Measures | Typical Target |
|--------|------------------|----------------|
| L0 | Average active features per token | 50-200 |
| CE Loss Score | Cross-entropy recovered vs. original | 80-95% |
| Explained Variance | Reconstruction quality | >90% |
| Absorption Score | Degree of parent-feature subsumption | As low as possible |
| Feature Splitting Rate | Fragmentation of concepts | Context-dependent |
| PW-MCC | Pairwise dictionary mean correlation (consistency) | ~0.80+ |
| Probe Loss | Recovery of hypothesized features via 1D probes | Lower is better |
| Ablation Sparsity | Sparsity of downstream effects when ablating latents | Higher is better |
| Explainability Score | Precision/recall of LLM-generated feature descriptions | Higher is better |

**Critical Gap**: No existing metric directly detects absorption in an unsupervised manner. The SAEBench absorption metric requires ground-truth probes (first-letter tasks), limiting generalizability to real semantic hierarchies.

---

## 4. Identified Research Gaps

- **Gap 1: Theoretical understanding of absorption dynamics.** While Chanin et al. (2024) identified absorption and Bussmann et al. (2025) mitigated it, there is no unified theoretical framework predicting *when* absorption will occur for a given feature hierarchy, SAE width, and sparsity level.

- **Gap 2: Scalable causal validation of absorbed features.** Existing metrics are correlational (probe-based). There is limited work on causal interventions that definitively establish whether a latent "knows about" vs. "uses" a parent feature, especially under absorption.

- **Gap 3: Cross-architecture absorption patterns.** Most absorption studies focus on residual-stream SAEs. How absorption manifests in attention-output SAEs, MLP SAEs, or multimodal (vision-language) SAEs remains underexplored.

- **Gap 4: Training dynamics and emergence time.** It is unclear at what point during SAE training absorption emerges, whether it is reversible, and how curriculum learning or temporal masking might prevent it.

- **Gap 5: Unified objective trade-offs.** No single training objective dominates across all interpretability goals. Methods that reduce absorption (Matryoshka, OrtSAE) introduce new pathologies (hedging, overhead) or trade off reconstruction fidelity.

- **Gap 6: Real-world downstream impact.** Kantamneni et al. (2025) showed SAEs often fail to outperform baselines on sparse probing. The causal link between absorption rates and downstream task underperformance has not been systematically quantified.

- **Gap 7: Construct validity of absorption metrics.** The dominant SAEBench absorption metric relies on first-letter classification tasks. Whether this metric generalizes to real semantic hierarchies (e.g., WordNet hypernyms) remains untested.

- **Gap 8: Frequency-confound control.** Feature absorption is confounded by frequency differences between parent and child features. Systematic frequency-matching studies are lacking.

- **Gap 9: Cross-layer absorption propagation.** Whether absorption in early residual-stream SAEs propagates to and compounds in deeper layers remains unstudied.

- **Gap 10: Real-world semantic hierarchy generalization.** The dominant absorption metric (first-letter tasks) may not generalize to real semantic hierarchies (e.g., WordNet hypernyms, conceptual taxonomies). Whether absorption is equally prevalent in natural language semantic hierarchies is unknown.

---

## 5. Available Resources

### Open-source Code

- **SAELens** (`https://github.com/jbloomAus/SAELens`): The primary PyTorch library for training and analyzing SAEs on LLMs. Supports standard, gated, TopK, JumpReLU, and BatchTopK architectures. Integrates with TransformerLens and Neuronpedia. MIT License.
- **SAEBench** (`https://github.com/adamkarvonen/SAEBench`): Comprehensive benchmarking suite with absorption evaluation, AutoInterp, sparse probing, and more. PyPI: `sae-bench`.
- **TransformerLens** (`https://github.com/neelnanda-io/TransformerLens`): Essential for extracting activations and running causal interventions on transformer models. MIT License.
- **Neuronpedia** (`https://neuronpedia.org`): Browser for pre-trained SAE features (e.g., GPT-2 small, Gemma).
- **CE-Bench** (`https://github.com/Yusen-Peng/CE-Bench`): Lightweight contrastive evaluation benchmark.
- **OpenAI Sparse Autoencoder** (`https://github.com/openai/sparse_autoencoder`): OpenAI's implementation with pre-trained SAEs on GPT-2-small.
- **SAEVis** (`https://github.com/callummcdougall/sae_vis`): Original SAE visualization tool by Callum McDougall; SAEDashboard builds upon it.
- **SAEDashboard** (`https://github.com/jbloomAus/SAEDashboard`): Feature visualization and dashboard generation for SAEs.
- **Feature-Interp** (`https://github.com/GeorgeMLP/feature-interp`): Code for "Revising and Falsifying SAE Feature Explanations" (NeurIPS 2025).

### Datasets

- **The Pile / pile-uncopyrighted** (`monology/pile-uncopyrighted`): Standard training corpus for SAEs.
- **OpenWebText**: Common alternative for GPT-2 scale models.
- **First-letter classification tokens** (from Chanin et al. 2024): Controlled synthetic evaluation set for absorption measurement.
- **RAVEL datasets**: Cities, Nobel laureates, verbs, physical objects, occupations with 4-6 attributes each.
- **SynthSAEBench-16k** (2026): Synthetic benchmark with known ground-truth features for controlled evaluation of absorption and other SAE properties.

### Pretrained Models and SAEs

- **GPT-2 Small SAEs** (`gpt2-small-res-jb` release on SAELens): Widely used baseline for absorption and interpretability research.
- **Gemma-2-2B SAEs** (`gemma-2b-res` release): Used in Matryoshka SAE and OrtSAE papers.
- **Gemma Scope** (Lieberum et al., 2024): Large-scale open SAEs across layers and model sizes.
- **Pythia-160M SAEs**: Available via SAELens; used in SAEBench evaluations.
- **Llama Scope**: Growing collection of SAEs for Llama models.

---

## 6. Implications for Idea Generation

**Directions worth exploring:**
- **Construct-validity of absorption metrics:** Testing whether first-letter absorption scores generalize to matched-frequency semantic hierarchies from WordNet. This is a genuine field-wide blind spot with high stakes.
- **Training-dynamic analysis of absorption:** Designing experiments that track when and how absorption emerges during SAE training, and whether early-stopping or curriculum strategies can prevent it.
- **Causal intervention benchmarks for absorption:** Moving beyond probe-based metrics to establish causal criteria (e.g., activation patching, ablation) that verify whether absorbed parent features are truly "missing" or merely "hidden."
- **Cross-layer absorption propagation:** Studying whether absorption in early residual-stream SAEs propagates to and compounds in deeper layers.
- **Quantifying the downstream cost of absorption:** Systematically varying absorption rates (via architecture or hyperparameters) and measuring sparse probing, steering, and safety-monitoring performance to establish a dose-response relationship.

**Saturated or crowded directions:**
- Simply proposing a new SAE architecture and showing lower absorption on first-letter tasks is becoming crowded (Matryoshka, OrtSAE, HSAE, GBA all exist).
- Purely descriptive studies of absorption in a single model/layer are unlikely to be novel.
- General SAE surveys and benchmarking papers are now well-covered (SAEBench, CE-Bench, Shu et al. survey).

**Cross-domain analogies with potential:**
- **Dictionary learning in signal processing:** The concept of "incoherence" in compressed sensing parallels OrtSAE's orthogonality penalty; more principled incoherence constraints could be imported.
- **Hierarchical topic models (LDA, HDP):** The explicit tree-structured priors in topic modeling could inspire probabilistic SAE variants with built-in hierarchical structure.
- **Rate-distortion theory:** The trade-off between sparsity (rate) and reconstruction (distortion) has formal parallels that could yield theoretical bounds on absorption inevitability.
- **Neuroscience receptive fields:** Visual cortex "surround suppression" is analogous to absorption; understanding neural mechanisms may inspire computational remedies.
- **Topic modeling absorption:** Document-topic distributions in LDA face analogous "topic absorption" issues; solutions like asymmetric Dirichlet priors may inspire SAE training modifications.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens (training + analysis) | High | MIT | Adopt | Dominant community library; supports multiple architectures out of the box; integrates with TransformerLens. |
| SAEBench (absorption eval) | High | Open source | Adopt | Standardized benchmark; directly provides the absorption metric and evaluation protocol from Chanin et al. (2024). |
| TransformerLens (activation extraction) | High | MIT | Adopt | De facto standard for mechanistic interpretability on transformers; essential for activation caching and interventions. |
| CE-Bench (contrastive eval) | Medium | Open source | Extend | Lightweight alternative to SAEBench; useful for quick validation but less comprehensive. |
| Matryoshka SAE code (from paper authors) | Medium | TBD | Extend | If publicly released, fork to study inner-level hedging vs. absorption trade-offs. |
| OrtSAE code (from paper authors) | Medium | TBD | Extend | If released, useful for comparing orthogonality-based absorption reduction against hierarchical methods. |
| RouteSAE code (from paper authors) | Medium | TBD | Extend | Multi-layer routing SAE; useful for cross-layer absorption analysis. |
| Neuronpedia (feature browser) | Low | N/A | Compose | Use for qualitative validation and feature inspection, not for quantitative experiments. |
| OpenAI sparse_autoencoder | Low | MIT | Reference | Simpler API but less active development; useful for reference implementations only. |
| Cross-Layer Transcoder (CLT) | Medium | TBD | Reference | Alternative to SAEs; may have different absorption-like failure modes. |

**Highlight:**
- **Reusable evaluation framework:** SAEBench's absorption evaluation (`sae_bench/evals/absorption/`) is the highest-priority resource to adopt. It provides ground-truth probe training, k-sparse probing, and absorption scoring.
- **Reusable data pipeline:** SAELens's `ActivationsStore` and `LanguageModelSAERunnerConfig` provide standardized activation buffering and training loops.
- **Reusable pretrained models:** GPT-2 small and Gemma-2-2B SAEs from SAELens/HuggingFace provide immediate baselines without requiring expensive training from scratch.
- **WordNet integration:** The NLTK WordNet API provides ready access to semantic hierarchies for constructing parent-child concept pairs.

---

## Sources

- [A is for Absorption (Chanin et al., NeurIPS 2025)](https://arxiv.org/abs/2409.14507)
- [SAEBench (Karvonen et al., 2025)](https://arxiv.org/abs/2503.09532)
- [Matryoshka SAE (Bussmann et al., 2025)](https://arxiv.org/abs/2503.17547)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [Feature Hedging (Chanin et al., 2025)](https://arxiv.org/abs/2505.11756)
- [Are SAEs Useful? (Kantamneni et al., ICML 2025)](https://arxiv.org/abs/2502.16681)
- [JumpReLU SAE (Rajamanoharan et al., 2024)](https://arxiv.org/abs/2407.14435)
- [BatchTopK SAE (Bussmann et al., 2024)](https://arxiv.org/abs/2412.06410)
- [HSAE (Luo et al., 2026)](https://arxiv.org/abs/2602.11881)
- [Taming Polysemanticity (Chen et al., 2025)](https://arxiv.org/abs/2506.14002)
- [Sparse but Wrong (Chanin et al., 2025)](https://arxiv.org/abs/2508.16560)
- [RAVEL (Huang et al., ACL 2024)](https://arxiv.org/abs/2402.17700)
- [Transcoders (Paulo et al., 2025)](https://arxiv.org/abs/2501.18823)
- [SAE Survey (Shu et al., EMNLP 2025)](https://arxiv.org/abs/2503.05613)
- [CE-Bench (Gulko et al., 2025)](https://arxiv.org/abs/2509.00691)
- [Feature Consistency Position (Song et al., 2025)](https://arxiv.org/abs/2505.20254)
- [Understanding SAE Scaling (2025)](https://arxiv.org/abs/2509.02565)
- [On the Limits of SAEs (2025)](https://arxiv.org/abs/2506.15963)
- [Sparse Autoencoders Can Interpret Random Models (2025)](https://arxiv.org/abs/2501.17727)
- [RouteSAE (Shi et al., 2025)](https://arxiv.org/abs/2503.08200)
- [Evaluating SAE without Explanations (2025)](https://arxiv.org/abs/2507.08473)
- [SAELens GitHub](https://github.com/jbloomAus/SAELens)
- [SAEBench GitHub](https://github.com/adamkarvonen/SAEBench)
- [TransformerLens GitHub](https://github.com/neelnanda-io/TransformerLens)
- [Neuronpedia](https://neuronpedia.org)
- [SAEDashboard GitHub](https://github.com/jbloomAus/SAEDashboard)
- [Feature-Interp GitHub](https://github.com/GeorgeMLP/feature-interp)
- [Towards Monosemanticity (Anthropic, 2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Scaling Monosemanticity (Anthropic, 2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [OpenAI SAE Scaling (Gao et al., 2024)](https://cdn.openai.com/papers/sparse-autoencoders.pdf)
- [GemmaScope (Lieberum et al., 2024)](https://huggingface.co/google/gemma-scope)
