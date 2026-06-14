# Literature Survey Report

**Research Topic**: Systematic Analysis and Quantification of Feature Absorption in Sparse Autoencoders (SAEs): Causes, Patterns, and Impact on Interpretability
**Survey Date**: 2026-04-28 (Updated)
**arXiv Search Keywords**: ["sparse autoencoder" AND "feature absorption", "sparse autoencoder" AND (superposition OR polysemanticity OR monosemanticity), "mechanistic interpretability" AND (SAE OR "sparse autoencoder"), "feature splitting" OR "feature absorption" OR "dead neurons" AND "sparse autoencoder", "JumpReLU" OR "TopK" OR "Gated SAE" AND "sparse autoencoder"]
**Web Search Keywords**: ["sparse autoencoder feature absorption SAE interpretability 2025", "SAE sparse autoencoder benchmark leaderboard 2025 mechanistic interpretability", "sparse autoencoder github open source SAELens transformerlens 2025", "feature hedging correlated features SAE 2025", "hierarchical semantics sparse autoencoder 2025", "SAE scaling feature manifolds 2025", "CE-Bench contrastive evaluation SAE 2025"]

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant paradigm in mechanistic interpretability for decomposing neural network activations into human-interpretable features. The foundational insight, formalized as the Linear Representation Hypothesis (LRH), posits that neural networks encode meaningful concepts as linear directions in superposition---allowing far more features than available dimensions (Elhage et al., 2022; Park et al., 2024). SAEs attempt to recover these superposed features through sparse dictionary learning, training auxiliary models with sparsity constraints to disentangle polysemantic representations into monosemantic ones (Cunningham et al., 2023; Bricken et al., 2023).

Despite remarkable empirical success---from circuit analysis (Marks et al., 2024) to feature steering (Wang et al., 2025) and model diffing (Lindsey et al., 2024)---SAEs consistently exhibit persistent failure modes that threaten their reliability. The most critical of these is **feature absorption**, first formally identified by Chanin et al. (2024), where seemingly monosemantic SAE latents fail to fire on seemingly arbitrary positive examples, instead having their activation "absorbed" into more specific child features. This creates an **interpretability illusion**: features appear clean and interpretable but harbor systematic false negatives. The phenomenon is caused by the interaction between the SAE's sparsity penalty and the inherent hierarchical structure of real-world features, and it persists across all tested LLM SAEs regardless of architecture or hyperparameters.

The field has responded along three fronts: (1) theoretical analysis seeking principled explanations for why absorption occurs; (2) architectural innovations (Matryoshka SAEs, Orthogonal SAEs, Hierarchical SAEs) attempting to mitigate absorption through structural constraints; and (3) evaluation frameworks (SAEBench, CE-Bench, Linear Representation Bench) establishing standardized metrics to measure and compare absorption rates across methods. However, a comprehensive, systematic study that quantifies absorption's prevalence, characterizes its patterns across model scales and layer depths, and rigorously evaluates proposed solutions remains a significant research gap.

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders | arXiv:2409.14507 (NeurIPS 2025) | 2024 | **First formal identification and metric for feature absorption**; proves hierarchical features cause absorption; validates on hundreds of LLM SAEs (Gemma Scope, Llama 3.2, Qwen2) | Metric is conservative underestimate; limited to first-letter task for causal validation; no systematic quantification across model scales |
| 2 | A Unified Theory of Sparse Dictionary Learning: Piecewise Biconvexity and Spurious Minima | arXiv:2512.05534 | 2025 | **First unified theoretical framework** for all SDL variants (SAEs, transcoders, crosscoders); proves biconvexity; explains absorption and dead neurons via non-identifiability; proposes feature anchoring | Theory assumes linear representation hypothesis; feature anchoring requires known anchor directions |
| 3 | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | arXiv:2503.17547 | 2025 | **Hierarchical SAE architecture** that reduces absorption by training nested dictionaries; superior sparse probing and concept erasure; maintains features at multiple abstraction levels | Minor reconstruction trade-off; evaluated only on Gemma-2-2B and TinyStories |
| 4 | OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features | arXiv:2509.22033 | 2025 | **Orthogonality constraint** reduces absorption by 65% and composition by 15%; discovers 9% more distinct features; scales linearly with SAE size | Limited evaluation scope; orthogonality may be too restrictive for some feature structures |
| 5 | Improving Robustness In Sparse Autoencoders via Masked Regularization | arXiv:2604.06495 | 2026 | **Masked regularization** disrupts co-occurrence patterns during training; reduces absorption, enhances probing, narrows OOD gap | Very recent; limited empirical validation on LLMs |
| 6 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders | arXiv:2602.11881 | 2026 | **HSAE** jointly learns SAEs at multiple levels with explicit parent-child relationships; structural constraint loss + random perturbation | Complex architecture; scalability to very large models untested |
| 7 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders | ICML 2025 | 2025 | **Standardized evaluation suite** with 8 metrics including feature absorption; 200+ pre-trained SAEs across 7 architectures; interactive visualization | Proxy metrics don't always translate to practical performance; absorption metric based on Chanin et al. |
| 8 | Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders | arXiv:2506.14002 | 2025 | **First SAE algorithm with theoretical recovery guarantees**; Group Bias Adaptation (GBA) with proven correct feature recovery | Assumes specific statistical model for features; limited to 1.5B parameter models |
| 9 | On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy | arXiv:2506.15963 | 2025 | **Closed-form theoretical analysis** showing SAEs generally fail to recover ground truth; proposes weighted SAE (WSAE) with principled weight selection | Ground truth features must be extremely sparse for perfect recovery |
| 10 | Superposition as Lossy Compression: Measure with Sparse Autoencoders | arXiv:2512.13568 | 2025 | **Information-theoretic framework** measuring effective degrees of freedom; Shannon entropy on SAE activations; connects superposition to adversarial robustness | Metric is indirect; doesn't directly address absorption |
| 11 | Stop Probing, Start Coding: Why Linear Probes and SAEs Fail at Compositional Generalisation | arXiv:2603.28744 | 2026 | **Reframes SAE failure as dictionary learning challenge** (not amortization); shows SAE-learned dictionaries point in wrong directions; oracle baseline proves problem is solvable | Focuses on compositional generalization rather than absorption specifically |
| 12 | Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training | arXiv:2510.08855 | 2025 | **ATM** dynamically adjusts feature selection via importance scores; achieves substantially lower absorption scores than TopK/JumpReLU | Evaluated only on Gemma-2-2b; temporal masking adds complexity |
| 13 | Interpretability as Compression: Reconsidering SAE Explanations with MDL-SAEs | arXiv:2410.11179 | 2024 | **MDL-inspired framework**; argues sparsity is imperfect proxy for interpretability; suggests hierarchical SAE architectures naturally follow from MDL principle | Theoretical framework; limited empirical validation on MNIST |
| 14 | Towards Principled Evaluations of Sparse Autoencoders | arXiv:2405.08366 | 2024 | **Supervised dictionary baseline** for evaluating unsupervised SAEs; identifies feature occlusion and over-splitting | Limited to IOI task on GPT-2 Small |
| 15 | SynthSAEBench: Evaluating SAEs on Scalable Realistic Synthetic Data | arXiv:2602.14687 | 2026 | **Synthetic benchmark** with ground-truth features; reproduces LLM SAE phenomena; identifies Matching Pursuit SAE overfitting | Synthetic data may not capture all real-world complexities |
| 16 | Evaluating Sparse Autoencoders for Monosemantic Representation | arXiv:2508.15094 | 2025 | **First systematic evaluation** of SAEs vs base models via activation distribution; introduces Jensen-Shannon-based concept separability score; APP intervention method | Focused on monosemanticity evaluation rather than absorption specifically |
| 17 | Measuring Sparse Autoencoder Feature Sensitivity | arXiv:2509.23717 | 2025 | **Feature sensitivity** as new evaluation dimension; many interpretable features have poor sensitivity; sensitivity declines with SAE width | Sensitivity is related to but distinct from absorption |
| 18 | Signal in the Noise: Polysemantic Interference Transfers and Predicts Cross-Model Influence | arXiv:2505.11611 | 2025 | **Polysemantic interference** is systematic not stochastic; interventions transfer across model scales; reveals convergent higher-order organization | Focuses on polysemanticity rather than absorption |
| 19 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | arXiv:2505.20254 | 2025 | **Feature consistency** across training runs; proposes PW-MCC metric; achieves 0.80 correlation for TopK SAEs | Consistency is complementary to absorption analysis |
| 20 | Fundamental Limits of Neural Network Sparsification: Evidence from Catastrophic Interpretability Collapse | arXiv:2603.18056 | 2026 | **Interpretability collapse** under extreme sparsification; dead neuron rates up to 90.6%; collapse is intrinsic to compression | Focuses on dead neurons rather than absorption |
| 21 | BatchTopK Sparse Autoencoders | arXiv:2412.06410 | 2024 | **Batch-level top-k** selection; adaptive latent allocation per sample; outperforms TopK, comparable to JumpReLU | Does not directly address absorption |
| 22 | AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features | arXiv:2510.00404 | 2025 | **Preserves negative activations** via magnitude thresholding; uncovers bidirectional concepts; matches supervised Difference-in-Mean method | Focuses on bidirectionality, not absorption |
| 23 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | arXiv:2506.01197 | 2025 | **H-SAE** uses mixture-of-experts with top-level parent SAE + down-projectors + child-specific low-level SAEs; respects hierarchical constraints | Limited to categorical hierarchies; scalability to large models untested |
| 24 | Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds | arXiv:2509.02565 | 2025 | **Capacity-allocation model** for SAE scaling; reveals pathological regime where SAEs tile common manifolds at expense of rare features; connects to "dark matter" | Model assumes power-law feature frequencies; limited empirical validation on LLMs |
| 25 | Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders | arXiv:2505.11756 | 2025 | **Feature hedging** phenomenon where SAEs merge correlated features; proposes Snap Loss and Balanced Matryoshka SAEs; compound multiplier ~0.75 improves absorption | Snap Loss fails when hedging is severe; no easy metric to detect hedging |
| 26 | Are Sparse Autoencoders Useful? A Case Study in Sparse Probing | arXiv:2502.16681 | 2025 | **Negative result**: SAEs underperform supervised baselines on probing across data scarcity, class imbalance, label noise, and covariate shift regimes | Limited probing tasks; does not test all SAE architectures |
| 27 | Route Sparse Autoencoder to Interpret Large Language Models | arXiv:2503.08200 | 2025 | **Dynamic routing** across multiple layers with shared SAE; +22.5% more features, +22.3% higher interpretability score at same L0 | Only evaluated on Llama-3.2-1B-Instruct; routing complexity |
| 28 | Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering | arXiv:2505.15038 | 2025 | **SDCV** uses SAEs to denoise concept vectors by scaling top-k differentiating latents; +4-16% steering success rate on complex concepts | Limited to steering tasks; concept-specific hyperparameter tuning |
| 29 | CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of General Interpretability of Sparse Autoencoders | arXiv:2509.00691 (BlackboxNLP 2025) | 2025 | **LLM-free benchmark** with 5K contrastive story pairs; contrastive + independence scores; 77.3% ranking agreement with SAEBench | Dataset generated with GPT-4; limited domain coverage; excludes Matryoshka |

---

## 3. SOTA Methods and Benchmarks

### Current Best Methods for Mitigating Feature Absorption

| Method | Approach | Absorption Reduction | Key Metric | Code Available |
|--------|----------|---------------------|------------|----------------|
| **Matryoshka SAE** | Nested dictionaries with prefix reconstruction losses | Significant (qualitative + quantitative) | Absorption score, sparse probing | Yes (SAEBench) |
| **OrtSAE** | Orthogonality penalty on decoder vectors | 65% absorption reduction | Absorption score, distinct features | Yes |
| **HSAE** | Explicit parent-child relationships with structural constraints | Substantial (especially at large dict sizes) | Absorption score, hierarchy recovery | Yes |
| **ATM** | Dynamic probabilistic masking based on activation statistics | Lower than TopK/JumpReLU | Absorption score | Not confirmed |
| **Masked Regularization** | Random token replacement to disrupt co-occurrence | Reduces absorption, improves OOD | Absorption, probing, OOD gap | Not confirmed |
| **Feature Anchoring** | Constrain learned features to known anchor directions | Substantial improvement on synthetic | Feature recovery rate | Yes (theory paper) |
| **WSAE** | Reweighted reconstruction targeting ground-truth features | Significant on tested settings | Monosemanticity, interpretability | Not confirmed |
| **AbsTopK** | Magnitude thresholding preserving negative activations | Improved reconstruction and interpretability | Reconstruction fidelity, bidirectional concepts | Yes |
| **Balanced Matryoshka** | Compound multiplier ~0.75 across nested levels | Improves absorption, TPP, SCR, and sparse probing | Absorption score, TPP, SCR | Yes |
| **H-SAE (Muchane)** | Mixture-of-experts with parent SAE + child-specific low-level SAEs | Dramatically improved reconstruction; moderate interpretability gains | Reconstruction, interpretability | Yes (GitHub) |
| **RouteSAE** | Dynamic routing across multiple layers with shared SAE | +22.5% more features, +22.3% higher interpretability | Interpretability score | Yes |

### Mainstream Datasets and Evaluation Metrics

**Datasets:**
- **First-letter task** (Chanin et al.): In-context learning prompts for token first-character prediction; primary absorption detection task
- **TinyStories** (Eldan & Li): Simple English children stories for controlled feature analysis
- **OpenWebText**: Standard language modeling corpus for SAE training
- **SynthSAEBench-16k**: Synthetic data with realistic feature characteristics (correlation, hierarchy, superposition)
- **Linear Representation Bench** (Tang et al.): Synthetic benchmark with fully accessible ground-truth features

**Key Metrics:**
- **Absorption Score** (Chanin et al.): Detects when main latents are disabled and single absorbing latent has disproportionate ablation effect
- **Feature Monosemanticity Score (FMS)** (Harle et al.): Quantifies feature monosemanticity in latent representations
- **L0 / Loss Recovered**: Standard reconstruction and sparsity metrics
- **Sparse Probing**: k-sparse linear probe performance on downstream tasks
- **Targeted Probe Perturbation (TPP)**: Feature manipulation testing
- **Spurious Correlation Removal (SCR)**: Measuring disentanglement
- **RAVEL**: Causal intervention evaluation
- **AutoInterp**: Automated interpretability scoring

### Benchmark Suites

| Benchmark | Coverage | Key Feature |
|-----------|----------|-------------|
| **SAEBench** | 8 metrics, 200+ SAEs, 7 architectures | Standardized absorption metric; interactive visualization at neuronpedia.org/sae-bench |
| **CE-Bench** | Contrastive evaluation | More reliable interpretability metrics |
| **SynthSAEBench** | Synthetic ground-truth features | Controlled ablations; diagnoses failure modes |
| **Linear Representation Bench** | Full ground-truth access | Theoretical validation of SDL methods |

---

## 4. Identified Research Gaps

- **Gap 1: Lack of systematic cross-scale quantification**. No existing work systematically measures absorption rates across model scales (from 100M to 70B+ parameters), layer depths, and SAE dictionary sizes with consistent methodology. Chanin et al. focus on Gemma-2-2B; OrtSAE and Matryoshka SAEs evaluate on limited model families.

- **Gap 2: Absorption dynamics under scaling**. While it is known that absorption increases with SAE width and sparsity, the precise scaling laws---how absorption rate scales with dictionary size, model depth, and feature hierarchy complexity---remain uncharacterized. The "representation holes" phenomenon (Bussmann et al., 2025) suggests larger SAEs fragment concepts, but quantitative scaling relationships are missing.

- **Gap 3: Causal impact on downstream interpretability tasks**. Most absorption studies measure the phenomenon itself but do not rigorously quantify how absorption degrades practical interpretability tasks: circuit finding, feature steering, model editing, and bias detection. The "interpretability illusion" is described qualitatively but not quantified in terms of task performance degradation.

- **Gap 4: Unified theoretical-empirical characterization**. Tang et al. (2025) provide a unified theory but limited empirical validation. Chanin et al. provide rich empirical data but no theoretical grounding. A study that bridges both---using theory to predict absorption patterns and experiments to validate---is missing.

- **Gap 5: Comparative evaluation of proposed solutions**. Multiple architectural solutions exist (Matryoshka, OrtSAE, HSAE, ATM, WSAE) but they have not been evaluated head-to-head on identical benchmarks with consistent metrics. SAEBench provides a partial solution but does not include all proposed methods.

- **Gap 6: Feature composition vs. absorption interaction**. Feature composition (merging independent features like "red" + "triangle" into "red triangle") and feature absorption are related but distinct phenomena. Their interaction---whether they compound or counteract---has not been studied systematically.

- **Gap 7: Recovery and repair of absorbed features**. Once absorption is detected, can it be repaired? Meta-SAEs (Leask et al., 2025) suggest decomposing latents into more fundamental "meta-latents," but systematic repair strategies remain unexplored.

- **Gap 8: Absorption in multimodal and non-language domains**. While SAEs have been extended to vision-language models (SAE-V), diffusion models (DLM-Scope), and protein models, absorption has only been studied in language models. Whether absorption manifests differently in other modalities---and whether multimodal SAEs exhibit cross-modal absorption---is completely unexplored.

- **Gap 9: Feature hedging and absorption interaction**. Chanin et al. (2025) identified feature hedging as a distinct failure mode where correlated features are merged in narrow SAEs. The relationship between hedging and absorption---whether they are manifestations of the same underlying issue or independent failure modes---has not been systematically studied. A unified framework explaining both phenomena is missing.

- **Gap 10: SAE scaling pathology and absorption**. Michaud et al. (2025) showed that SAEs can enter a pathological scaling regime where they tile common feature manifolds at the expense of rare features. The connection between this capacity-allocation pathology and feature absorption---whether absorption is a symptom of the SAE optimally allocating capacity to common features---remains unexplored.

- **Gap 11: Practical utility of SAEs under absorption**. Kantamneni et al. (2025) demonstrated that SAEs underperform supervised baselines on sparse probing tasks. The extent to which absorption contributes to this practical utility gap---as opposed to other failure modes---has not been isolated and quantified.

---

## 5. Available Resources

### Open-Source Code

| Repository | Description | License | Relevance |
|------------|-------------|---------|-----------|
| [SAELens](https://github.com/jbloomAus/SAELens) | Primary SAE training/analysis library (1100+ stars) | MIT | **Essential**: Training pipelines, pre-trained SAEs, evaluation tools |
| [SAEBench](https://github.com/adamkarvonen/SAEBench) | Comprehensive benchmark suite (ICML 2025) | MIT | **Essential**: Standardized evaluation including absorption metrics |
| [sae-spelling](https://github.com/lasr-spelling/sae-spelling) | Chanin et al. absorption study code | Not specified | **High**: Absorption detection metric implementation |
| [Language-Model-SAEs](https://github.com/OpenMOSS/Language-Model-SAEs) | Llama Scope training and visualization | MIT | **High**: Scalable SAE training, interpretation tools |
| [dictionary_learning](https://github.com/ai2es/dictionary_learning) | Various SAE architectures | MIT | **Medium**: Alternative SAE implementations |
| [sae-icm](https://github.com/mrtzmllr/sae-icm) | Orthogonality regularization SAEs | MIT | **Medium**: Identifiable features via orthogonality |
| [SAELens-V](https://github.com/saev-2025/SAELens-V) | Multimodal SAE extension | MIT | **Medium**: Vision-language SAE training |
| [Prisma](https://github.com/Prisma-Multimodal/ViT-Prisma) | Vision MI toolkit | MIT | **Medium**: 80+ pre-trained vision SAEs |
| [OpenAI/sparse_autoencoder](https://github.com/openai/sparse_autoencoder) | Official OpenAI SAE release | MIT | **Low**: GPT-2 SAEs; less active than SAELens |

### Pre-trained Models and Checkpoints

- **Gemma Scope** (DeepMind): 16K and 65K width SAEs on Gemma-2-2B, all layers and sublayers; publicly available on HuggingFace
- **Llama Scope** (FNLP): 32K and 128K features on Llama-3.1-8B-Base, 256 SAEs covering all layers and sublayers
- **Pythia SAEs** (EleutherAI): Standard reference SAEs on Pythia-160M
- **SAEBench Collection**: 200+ pre-trained SAEs across 7 architectures on Pythia-160M and Gemma-2-2B

### Datasets

- **OpenWebText**: Standard corpus for SAE training (derived from Reddit outbound links)
- **TinyStories**: Controlled dataset for feature hierarchy analysis
- **First-letter ICL prompts** (Chanin et al.): Custom dataset for absorption detection
- **SynthSAEBench synthetic data**: Configurable synthetic features with hierarchy, correlation, superposition

### Interactive Tools

- **Neuronpedia** (neuronpedia.org): Feature visualization and exploration platform
- **SAEBench visualization** (neuronpedia.org/sae-bench): Interactive metric comparison across SAEs
- **Feature Absorption Explorer** (feature-absorption.streamlit.app): Chanin et al. results explorer
- **SparseLatents Tree View** (sparselatents.com/treeview): Hierarchical feature visualization across SAE sizes

---

## 6. Implications for Idea Generation

### Directions Worth Exploring (High Potential)

1. **Systematic absorption quantification study**: A large-scale empirical study measuring absorption rates across model scales (Pythia-160M to Llama-70B), layer depths, and dictionary sizes with consistent methodology. This would fill Gap 1 and provide the field with definitive scaling relationships.

2. **Theoretical-empirical bridge**: Using Tang et al.'s unified framework to predict absorption patterns and validating with experiments. This could yield principled design guidelines for SAE architectures that avoid absorption.

3. **Absorption-aware SAE training objectives**: Designing training objectives that explicitly penalize absorption (e.g., via hierarchy-preserving regularization, causal consistency constraints, or adversarial training against absorption patterns).

4. **Cross-phenomenon interaction study**: Systematically studying how feature absorption, feature composition, feature splitting, and dead neurons interact and compound. This could reveal whether addressing one phenomenon exacerbates others.

5. **Task-oriented absorption impact**: Quantifying how absorption degrades specific downstream interpretability tasks (circuit finding accuracy, steering effectiveness, model editing precision). This would make the "interpretability illusion" concrete and measurable.

6. **Unified failure mode framework**: Developing a unified theoretical framework that explains feature absorption, feature hedging, and feature splitting as manifestations of a common underlying mechanism (e.g., SAE capacity allocation under correlated/hierarchical features). This would bridge the gap between Chanin et al.'s empirical findings and Michaud et al.'s scaling theory.

7. **Absorption-aware evaluation integration**: Incorporating absorption metrics into lightweight benchmarks like CE-Bench, enabling rapid development cycles where absorption is monitored alongside interpretability. Currently CE-Bench does not include absorption metrics.

### Saturated Directions (Lower Novelty)

- Simple detection of absorption in new model families (already done for Gemma, Llama, Qwen)
- Proposing yet another SAE architecture without theoretical grounding or comprehensive benchmarking
- Toy model demonstrations of absorption (well-established by Chanin et al.)
- Pure reconstruction-quality optimization without considering interpretability
- Proposing SAE architectures evaluated only on single model family or limited dictionary sizes
- Absorption detection in yet another model family without systematic cross-scale analysis

### Cross-Domain Analogies with Potential

- **Compressed sensing**: Classical sparse coding with per-sample iterative inference (FISTA) achieves better recovery than amortized SAE encoders (Pacela et al., 2026). The dictionary learning gap in SAEs mirrors the gap between iterative and amortized inference.
- **Non-negative matrix factorization (NMF)**: NMF's identifiability conditions under separability assumptions may provide theoretical tools for SAE feature recovery guarantees.
- **Causal representation learning**: The Independent Causal Mechanisms (ICM) principle (Muller et al., 2026) suggests orthogonality promotes modular representations amenable to causal intervention---directly applicable to absorption mitigation.
- **Information theory**: The MDL framework (Ayonrinde et al., 2024) suggests hierarchical SAE architectures naturally follow from compression principles, providing a principled foundation for multi-level feature learning.
- **Neural scaling theory**: Brill (2024) and Michaud et al. (2025)'s capacity-allocation model provides a mathematical framework for understanding when SAEs enter pathological scaling regimes. This can be extended to predict absorption rates as a function of feature frequency distribution and SAE capacity.
- **Dictionary learning theory**: The non-identifiability results from Tang et al. (2025) and the reweighted remedy from Cui et al. (2025) suggest principled ways to modify SAE training to avoid spurious minima that cause absorption.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens | High | MIT | **Adopt** | De facto standard; comprehensive training pipelines; pre-trained SAEs; integrates with TransformerLens |
| SAEBench | High | MIT | **Adopt** | Standardized evaluation including absorption metrics; 200+ pre-trained SAEs; PyPI package available |
| TransformerLens | High | MIT | **Adopt** | Essential for activation extraction and hook-based intervention; SAELens depends on it |
| Gemma Scope SAEs | High | Open | **Adopt** | High-quality pre-trained SAEs on Gemma-2-2B; primary dataset for absorption studies |
| Llama Scope SAEs | High | Open | **Adopt** | Scalable to larger models; 256 SAEs covering all layers |
| SynthSAEBench | Medium | Open | **Extend** | Synthetic ground-truth features for controlled experiments; can be extended with custom feature hierarchies |
| sae-spelling (Chanin et al.) | High | Unknown | **Adopt/Extend** | Contains absorption metric implementation; may need adaptation for broader model coverage |
| dictionary_learning repo | Medium | MIT | **Compose** | Alternative architectures (JumpReLU, TopK, Gated); can be composed with custom training objectives |
| CE-Bench | High | Open | **Adopt** | LLM-free lightweight benchmark for rapid SAE development iteration; 77.3% agreement with SAEBench |
| feature-hedging-paper (Chanin et al.) | High | MIT | **Adopt/Extend** | Snap Loss and Balanced Matryoshka implementations; compound multiplier approach for absorption reduction |
| hierarchical-sparse-autoencoders (Muchane et al.) | Medium | MIT | **Extend** | H-SAE with mixture-of-experts architecture; can be extended with absorption-aware routing |
| RouteSAE code | Medium | Open | **Compose** | Multi-layer dynamic routing; can be composed with absorption detection for cross-layer feature analysis |

### Recommended Technical Stack

```
Core Framework:    SAELens + TransformerLens + PyTorch
Evaluation:        SAEBench (pip install sae-bench)
Models:            Gemma-2-2B (primary), Pythia-160M (fast iteration), Llama-3.1-8B (scale)
Data:              OpenWebText (training), custom ICL prompts (absorption detection)
Compute:           24GB+ VRAM for full SAEBench evaluation; smaller GPUs for toy models
Visualization:     Neuronpedia API + custom matplotlib/seaborn plots
```

### Key Implementation Notes

1. **SAELens is the foundation**: All SAE training, loading, and analysis should use SAELens. It provides consistent APIs across architectures and integrates seamlessly with TransformerLens for activation extraction.

2. **SAEBench for evaluation**: The absorption metric in SAEBench is based on Chanin et al.'s work and provides standardized, comparable results. Running the full suite takes ~110 minutes per SAE on a 24GB GPU.

3. **Gemma Scope as primary data source**: These pre-trained SAEs have been most extensively studied for absorption and provide a solid baseline for comparison.

4. **Synthetic benchmarks for controlled experiments**: SynthSAEBench and custom toy models (tree-structured hierarchical features) enable ground-truth validation of absorption detection and mitigation strategies.

5. **First-letter task for causal validation**: Chanin et al.'s in-context learning setup provides a clean causal testbed for verifying absorption effects on model behavior.

---

## Bibliography (Key Papers)

- Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. *NeurIPS 2025*. arXiv:2409.14507.
- Tang, Y., Saini, H., Yao, Z., Lin, Z., Liao, Y., Li, Q., Du, M., & Liu, D. (2025). A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima. arXiv:2512.05534.
- Bussmann, B., Nabeshima, N., Karvonen, A., & Nanda, N. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547.
- Korznikov, A., Galichin, A., Dontsov, A., Rogov, O., Tutubalina, E., & Oseledets, I. (2025). OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033.
- Karvonen, A., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. *ICML 2025*.
- Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *ICLR 2024*. arXiv:2309.08600.
- Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *Transformer Circuits Thread*.
- Elhage, N., et al. (2022). Superposition, Memorization, and Double Descent. *Transformer Circuits Thread*.
- Marks, S., et al. (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. arXiv:2403.19647.
- Luo, Y., et al. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. arXiv:2602.11881.
- Narayanaswamy, V., et al. (2026). Improving Robustness In Sparse Autoencoders via Masked Regularization. arXiv:2604.06495.
- Li, T.E. & Ren, J. (2025). Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training. arXiv:2510.08855.
- Chen, S., et al. (2025). Taming Polysemanticity in LLMs: Provable Feature Recovery via Sparse Autoencoders. arXiv:2506.14002.
- Cui, J., Zhang, Q., Wang, Y., & Wang, Y. (2025). On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963.
- Pacela, V.B., et al. (2026). Stop Probing, Start Coding: Why Linear Probes and Sparse Autoencoders Fail at Compositional Generalisation. arXiv:2603.28744.
- Ayonrinde, K., Pearce, M.T., & Sharkey, L. (2024). Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs. arXiv:2410.11179.
- Chanin, D. & Garriga-Alonso, A. (2026). SynthSAEBench: Evaluating Sparse Autoencoders on Scalable Realistic Synthetic Data. arXiv:2602.14687.
- Muller, M., Draye, F., & Scholkopf, B. (2026). Identifying Intervenable and Interpretable Features via Orthogonality Regularization. arXiv:2602.04718.
- Muchane, M., Richardson, S., Park, K., & Veitch, V. (2025). Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. arXiv:2506.01197.
- Michaud, E.J., Gorton, L., & McGrath, T. (2025). Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds. arXiv:2509.02565.
- Chanin, D., Dulka, T., & Garriga-Alonso, A. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756.
- Kantamneni, S., Engels, J., Rajamanoharan, S., Tegmark, M., & Nanda, N. (2025). Are Sparse Autoencoders Useful? A Case Study in Sparse Probing. arXiv:2502.16681.
- Shi, W., Li, S., Liang, T., Wan, M., Ma, G., Wang, X., & He, X. (2025). Route Sparse Autoencoder to Interpret Large Language Models. arXiv:2503.08200.
- Zhao, H., Wu, X., Yang, F., Shen, B., Liu, N., & Du, M. (2025). Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering. arXiv:2505.15038.
- CE-Bench Authors. (2025). CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark of General Interpretability of Sparse Autoencoders. arXiv:2509.00691.
