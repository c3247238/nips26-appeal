# Literature Survey Report

**Research Topic**: Systematic Analysis and Quantification of Feature Absorption in Sparse Autoencoders (SAEs): Causes, Patterns, and Impact on Interpretability
**Survey Date**: 2026-04-26
**Last Updated**: 2026-04-26 (supplemented with Feb-Aug 2025 and 2026 papers)
**arXiv Search Keywords**: ["sparse autoencoder feature absorption", "SAE feature splitting absorption", "Matryoshka sparse autoencoder", "hierarchical sparse autoencoder", "orthogonal sparse autoencoder OrtSAE", "TopK SAE interpretability"]
**Web Search Keywords**: ["SAE feature absorption mechanism 2025", "SAEBench benchmark absorption evaluation", "Matryoshka SAE absorption reduction", "sparse autoencoder GitHub open source", "sparse autoencoder feature absorption 2025 2026 new research arxiv", "SAE superposition feature splitting absorption mechanistic interpretability 2025", "sparse autoencoder absorption quantification systematic analysis 2025 2026", "SAE feature absorption causality intervention ablation steering 2025", "sparse autoencoder training dynamics feature emergence absorption evolution 2025", "sparse autoencoder theoretical framework reweighted remedy limits 2025", "Sanity Checks for Sparse Autoencoders 2026", "SynthSAEBench 2026", "Sparse but Wrong 2025", "SplInterp 2025"]

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant paradigm for mechanistic interpretability of large language models (LLMs), following Anthropic's foundational work on monosemanticity (Bricken et al., 2023). The core premise is that SAEs can decompose polysemantic neural activations into sparse, human-interpretable feature directions. Over 2024-2025, the field has shifted from demonstrating feasibility to rigorously diagnosing failure modes and building standardized benchmarks. By early 2026, a wave of critical evaluation papers (Sanity Checks, SynthSAEBench) has challenged the fundamental assumption that SAEs reliably recover ground-truth features, even in controlled synthetic settings.

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
| 21 | Sparse but Wrong: Incorrect L0 Leads to Incorrect Features in Sparse Autoencoders | arXiv:2508.16560 | 2025 | Shows that wrong L0 sparsity targets systematically produce incorrect features; related to feature hedging pathology. | Focuses on narrow-width regime; less relevant to very wide SAEs. |
| 22 | SplInterp: Improving our Understanding and Training of Sparse Autoencoders | arXiv:2505.11836 | 2025 | Uses spline theory to characterize SAE geometry (power diagrams); proposes PAM-SGD training algorithm. | Theoretical focus; limited empirical validation on LLMs. |
| 23 | Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? | arXiv:2602.14111 | 2026 | Scathing critique: random baselines match trained SAEs on downstream tasks; SAEs recover only 9% of synthetic ground-truth features. | Synthetic evaluation may not fully transfer to real LLMs. |
| 24 | SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data | arXiv:2602.14687 | 2026 | Synthetic benchmark with 16k ground-truth features; discovers MP-SAEs exploit superposition noise without learning true features. | Best-case scenario for SAEs; real LLM features may be harder. |
| 25 | Evaluating Sparse Autoencoders for Monosemantic Representation | EACL 2026 Findings | 2026 | First systematic quantitative evaluation of SAE monosemanticity; introduces concept separability score and APP intervention method. | Limited to Gemma-2-2B and DeepSeek-R1. |
| 26 | Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders | arXiv:2506.14002 (ICLR 2026) | 2025 | First SAE algorithm with theoretical feature recovery guarantees; introduced Group Bias Adaptation (GBA). | Theory assumes a specific generative model; real-world LLM features may not fit. |
| 27 | On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy | arXiv:2506.15963 (ICLR 2026) | 2025 | Closed-form theoretical solution for SAEs; proves extreme sparsity needed for full recovery; proposes WSAE with adaptive weighting. | Simplified theoretical model; limited empirical validation. |
| 28 | Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering | arXiv:2505.15038 | 2025 | Proposes denoising SAE concept vectors for more reliable steering; addresses feature composition issues. | Focuses on steering application; limited absorption analysis. |
| 29 | CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features | arXiv:2508.12535 | 2025 | Uses correlation for feature selection and intervention for causality validation; multi-layer steering with correlated SAE features. | Steering-focused; does not directly quantify absorption. |
| 30 | Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders | arXiv:2508.09363 | 2025 | Argues for domain-specific SAE training to improve causal fidelity and reduce absorption. | Limited to specific domains; generalization unclear. |

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
| **WSAE** | Reweighted reconstruction loss based on polysemanticity | Improved monosemanticity; reduced feature shrinking/vanishing | Simplified theoretical model |
| **PAM-SGD** | Proximal Alternating Method SGD from spline theory | Better sample efficiency and sparsity | Limited LLM validation |
| **Denoising SAE** | Denoising concept vectors for steering | Reduced feature composition | Steering-focused |

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

- **SynthSAEBench** (Chanin & Garriga-Alonso, 2026): Synthetic benchmark with 16,384 ground-truth features and 768 hidden dimensions. Features realistic correlation, hierarchy, and superposition. Enables precise measurement of feature recovery rates (SAEs recover only ~9% of true features).

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

- **Gap 11: Absorption in synthetic vs. real settings.** SynthSAEBench (Chanin & Garriga-Alonso, 2026) shows SAEs fail even on synthetic ground-truth features, but the relationship between synthetic and real-LLM absorption patterns is unexplored.

- **Gap 12: Random baseline comparison for absorption.** Korznikov et al. (2026) show random baselines match trained SAEs on downstream tasks, but no work has evaluated whether random dictionaries exhibit absorption-like patterns.

- **Gap 13: Theoretical prediction of absorption severity.** While WSAE (Cui et al., 2025) provides theoretical bounds on feature recovery, no theory predicts *which specific features* will be absorbed or the severity of absorption for a given feature hierarchy.

- **Gap 14: L0 sparsity-absorption dose-response.** Chanin & Garriga-Alonso (2025) show incorrect L0 leads to incorrect features, but the precise dose-response relationship between L0 target and absorption rate remains unquantified.

---

## 5. Available Resources

### Open-source Code

- **SAELens** (`https://github.com/jbloomAus/SAELens`): The primary PyTorch library for training and analyzing SAEs on LLMs. Supports standard, gated, TopK, and JumpReLU architectures. Integrates with TransformerLens and Neuronpedia.
- **SAEBench** (`https://github.com/adamkarvonen/SAEBench`): Comprehensive benchmarking suite with absorption evaluation, AutoInterp, sparse probing, and more. PyPI: `sae-bench`.
- **TransformerLens** (`https://github.com/neelnanda-io/TransformerLens`): Essential for extracting activations and running causal interventions on transformer models.
- **Neuronpedia** (`https://neuronpedia.org`): Browser for pre-trained SAE features (e.g., GPT-2 small, Gemma).
- **SAELens-V** (`https://github.com/saev-2025/SAELens-V`): Vision/multimodal extension of SAELens for LLaVA-NeXT and Chameleon SAE training.
- **SynthSAEBench Toolkit** (`https://github.com/decoderesearch/synth-sae-bench-experiments`): Synthetic data generation and benchmark for controlled SAE evaluation.
- **SARM** (`https://github.com/schrieffer-z/sarm`): Sparse Autoencoder Reward Model (AAAI 2026 Oral).

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
- Simply proposing a new SAE architecture and showing lower absorption on first-letter tasks is becoming crowded (Matryoshka, OrtSAE, HSAE, GBA, WSAE all exist).
- Purely descriptive studies of absorption in a single model/layer are unlikely to be novel.
- Papers showing SAEs underperform baselines without isolating absorption as the causal factor are now well-established (Kantamneni et al. 2025; Korznikov et al. 2026).

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
| SynthSAEBench toolkit | High | Open source | Adopt | Essential for controlled experiments with ground-truth features; enables precise absorption quantification. |
| Neuronpedia (feature browser) | Low | N/A | Compose | Use for qualitative validation and feature inspection, not for quantitative experiments. |

**Highlight:**
- **Reusable evaluation framework:** SAEBench's absorption evaluation (`sae_bench/evals/absorption/`) is the highest-priority resource to adopt. It provides ground-truth probe training, k-sparse probing, and absorption scoring. SynthSAEBench complements this with synthetic ground-truth for controlled experiments.
- **Reusable data pipeline:** SAELens's `ActivationsStore` and `LanguageModelSAERunnerConfig` provide standardized activation buffering and training loops.
- **Reusable pretrained models:** GPT-2 small and Gemma-2-2B SAEs from SAELens/HuggingFace provide immediate baselines without requiring expensive training from scratch.
