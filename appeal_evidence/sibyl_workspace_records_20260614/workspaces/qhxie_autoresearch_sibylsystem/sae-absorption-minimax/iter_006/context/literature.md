# Literature Survey Report

**Research Topic**: Feature Absorption in Sparse Autoencoders: Systematic Analysis and Quantification of Its Causes, Patterns, and Impact on Interpretability
**Survey Date**: 2026-05-01
**arXiv Search Keywords**: Not available (arxiv-mcp-server not running)
**Web Search Keywords**: Not available (WebSearch API returned 400 error)

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as a dominant tool in mechanistic interpretability for decomposing the polysemantic activations of large language models into sparse, human-interpretable features. The field has progressed rapidly since Anthropic's seminal "Towards Monosemanticity" (2023), with multiple SAE architectures (standard ReLU, TopK, JumpReLU, Gated, Switch) and training algorithms now available. SAELens has become the standard library for training and analyzing SAEs.

The field now recognizes several fundamental limitations that undermine SAE reliability for robust interpretability. **Feature absorption** is arguably the most critical: when underlying features form hierarchical relationships (e.g., "India" implies "Asia"), SAEs trained with sparsity penalties tend to absorb parent features into child features, causing seemingly monosemantic features to fail to fire where they should. This was first systematically studied by Chanin et al. (2024) in "A is for Absorption."

Beyond absorption, the field has identified **feature composition** (independent features merging into composite representations), **dead features** (features that never activate), **poor feature sensitivity** (features failing to generalize to semantically similar contexts), and scaling pathologies related to **feature manifolds**. SAEBench (Karvonen et al., 2025) provides a comprehensive benchmark framework to evaluate these issues, revealing that gains on proxy metrics do not reliably translate to better practical performance.

The dominant paradigm is now shifting from single-level SAEs to hierarchical approaches (HSAE, hierarchical semantics architectures) that explicitly model parent-child feature relationships, and to orthogonal SAE variants (OrtSAE) that enforce disentanglement between learned features.

### Critical Recent Development: Null Result in Absorption-Steering Relationship

**Project Iteration 6 Findings** (2026-04-29): A controlled experiment testing whether feature absorption predicts steering effectiveness found **no significant correlation** at most steering magnitudes (p=0.299 for aggregated null result). This contradicts the assumed relationship between absorption and steering sensitivity. Key findings:

- Original uncontrolled analysis suggested r=+0.35, p<0.001
- After controlling for activation frequency and decoder L2 norm confounds: p=0.299 (null result)
- At high steering magnitude (beta=20): low-absorption features showed *higher* steering sensitivity than high-absorption features (p=0.015, does not survive Bonferroni correction)
- This directional reversal at beta=20 may indicate saturation effects

**Implication**: The assumed causal chain "absorption -> peripheralization -> reduced steering effectiveness" was not validated. This requires **new research angles** beyond the absorption-steering relationship.

### CRITICAL Iteration 7 Finding: UAS Metric Saturation at Layer 8

**Project Iteration 7 Discoveries** (2026-04-29): When applying the Chanin protocol to measure absorption using the UAS (Universal Absorption Score) metric on GPT-2 Small layer 8 residual stream SAE, a critical issue was discovered:

- **Layer 8 saturation**: All features at layer 8 show UAS = 1.0 (maximum absorption score)
- **Indistinguishability problem**: With all features showing identical absorption scores, it becomes impossible to distinguish absorbed from non-absorbed features
- **Metric failure**: The UAS-based Chanin protocol fails to provide meaningful absorption differentiation at this layer
- **Root cause hypothesis**: Layer 8 may have a specific architectural property (deeper in the transformer hierarchy) where feature hierarchies collapse, or where the residual stream representations encode hierarchical relationships differently

**Implications for Research**:
1. **Alternative absorption metrics needed**: UAS-based detection may saturate at certain layers; need architecture-agnostic methods
2. **Layer-dependent behavior**: Absorption measurement may require layer-specific calibration rather than universal thresholds
3. **Multi-metric approach required**: Combining UAS with other absorption detection methods (probe-based, causal intervention consistency, cross-layer information preservation) may be necessary
4. **New research angles**: Given the measurement limitations, consider pivoting to:
   - Feature sensitivity as a proxy for absorption (Tian et al., 2025)
   - Cross-layer information migration as absorption indicator
   - Compositional feature recovery from absorbed directions
   - Causal intervention consistency as ground truth for feature validity

**Key Question Opened**: If UAS=1.0 for all features at layer 8, does this mean all features are absorbed, or does the metric itself saturate? This question fundamentally challenges the Chanin protocol's universal applicability.

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders* (Chanin et al.) | arXiv:2409.14507 | 2024 | First systematic study of feature absorption; defines the phenomenon, introduces detection metrics, validates on hundreds of LLMs | Varying SAE sizes/sparsity insufficient to solve absorption; no complete solution provided |
| 2 | *Adaptive Temporal Masking for Stable SAE Training* (ATM, Li & Ren) | arXiv:2510.08855 | 2025 | ATM method with temporal importance tracking; 40% absorption reduction vs best prior method | Only evaluated on Gemma-2-2B layer 12; single architecture comparison |
| 3 | *OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features* (Korznikov et al.) | arXiv:2509.22033 | 2025 | Enforces feature orthogonality to reduce absorption (65% reduction) and composition (15% reduction); linear-scaling training | Orthogonality constraints may limit reconstruction quality; evaluated on limited model/layer combinations |
| 4 | *From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders* (Luo et al.) | arXiv:2602.11881 | 2026 | Jointly learns SAEs with parent-child relationships; structural constraint loss and random perturbation mechanism | Computational overhead of hierarchical learning; limited to residual stream layers |
| 5 | *SAEBench: A Comprehensive Benchmark for Sparse Autoencoders* (Karvonen et al.) | arXiv:2503.09532 | 2025 | 8-metric evaluation suite; 200+ SAEs across 8 architectures; reveals proxy metrics do not predict practical performance | Metrics still under active development; many practical applications require further validation |
| 6 | *Measuring Sparse Autoencoder Feature Sensitivity* (Tian et al.) | arXiv:2509.23717 | 2025 | New evaluation dimension: feature sensitivity (reliability on semantically similar texts); finds many interpretable features have poor sensitivity; sensitivity declines with SAE width | LLM-based evaluation may introduce its own biases; only tested on GPT-2 variants |
| 7 | *The Geometry of Concepts: Sparse Autoencoder Feature Structure* (Li et al.) | arXiv:2410.19750 | 2024 | Three-level structure analysis: atomic (crystals/parallelograms), brain (spatial modularity/lobes), galaxy (power-law eigenvalue distribution) | Primarily observational; does not directly address absorption |
| 8 | *Incorporating Hierarchical Semantics in SAE Architectures* (Muchane et al.) | arXiv:2506.01197 | 2025 | Explicitly models semantic hierarchy between concepts; improves reconstruction and interpretability; computational efficiency gains | Hierarchical structure must be known a priori or discovered separately |
| 9 | *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs* (Rajamanoharan et al.) | arXiv:2407.14435 | 2024 | SOTA reconstruction at given sparsity; JumpReLU activation with STE training; direct L0 optimization | Still suffers from feature absorption; discontinuous activations add complexity |
| 10 | *Understanding SAE Scaling with Feature Manifolds* (Michaud et al.) | arXiv:2509.02565 | 2025 | Capacity-allocation model for SAE scaling; identifies pathological regime where SAEs learn far fewer features than latents | Theoretical framework; empirical validation on limited settings |
| 11 | *Superposition as Lossy Compression* (Bereska et al.) | arXiv:2512.13568 | 2025 | Information-theoretic framework measuring effective degrees of freedom; Shannon entropy on SAE activations | Focuses on superposition measurement rather than absorption specifically |
| 12 | *Sparse Autoencoder Features for Classifications and Transferability* (Gallifant et al.) | arXiv:2502.11367 | 2025 | SAE features for safety-critical classification; cross-model transfer; binarization strategies | Does not address absorption; focuses on applied classification tasks |
| 13 | *PURE: Turning Polysemantic Neurons Into Pure Features* (Dreyer et al.) | arXiv:2404.06453 | 2024 | Disentangles polysemantic channels by identifying relevant circuits; applicable to CNNs | Designed for CNNs, not LLMs; circuit identification is computationally expensive |
| 14 | *Challenges in Mechanistically Interpreting Model Representations* (Golechha & Dao) | arXiv:2402.03855 | 2024 | Formalizes feature representation challenges; exploratory study on dishonesty representations; highlights insufficiency of current MI methods | Broad scope, limited depth on specific SAE issues |
| 15 | *Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?* (Korznikov et al.) | arXiv:2602.14111 | 2026 | **CRITICAL**: Tests whether SAEs recover meaningful features; SAEs recover only 9% of ground-truth features despite 71% explained variance; random baselines match trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), causal editing (0.73 vs 0.72) | Most alarming negative result in SAE literature; suggests SAEs may not reliably decompose model internals |
| 16 | *Variational Sparse Autoencoders* (Baker & Li) | arXiv:2509.22994 | 2025 | Introduces vSAE with stochastic sampling and KL divergence; finds vSAE features demonstrate improved independence but worse dead feature ratio; KL regularization creates excessive pressure | Demonstrates naive probabilistic approaches don't solve SAE limitations |
| 17 | *Group Equivariance Meets Mechanistic Interpretability: Equivariant Sparse Autoencoders* (Erdogan & Lucic) | arXiv:2511.09432 | 2025 | Incorporates group symmetries into SAEs; adaptive equivariant SAEs discover features with superior probing performance | Addresses domains beyond language with inherent symmetries |
| 18 | *Evaluating SAE Interpretability Without Explanations* (Paulo & Belrose) | arXiv:2507.08473 | 2025 | Alternative interpretability evaluation that doesn't require LLM-generated explanations; compares with human evaluation | Addresses evaluation methodology concerns |
| 19 | *SOSAE: Self-Organizing Sparse AutoEncoder* (Modi et al.) | arXiv:2507.04644 | 2025 | Dynamic feature space dimensionality adaptation via physics-inspired regularization; 130x FLOP reduction in tuning | Doesn't address absorption; focuses on efficiency |
| 20 | *Route Sparse Autoencoder to Interpret Large Language Models* (Shi et al.) | arXiv:2503.08200 | 2025 | RouteSAE with routing mechanism for multi-layer feature extraction; 22.5% more features than baseline under same sparsity | Multi-layer approach may interact with absorption differently |

---

## 3. SOTA Methods and Benchmarks

### Critical Caveat: Do SAEs Actually Work?

**The most important recent finding** (Korznikov et al., 2026) challenges the fundamental premise of SAE-based interpretability:

1. **Synthetic experiments**: SAEs recover only **9% of ground-truth features** despite achieving **71% explained variance**
2. **Real activations**: Random baseline features match trained SAEs on:
   - Interpretability scores: 0.87 vs 0.90
   - Sparse probing: 0.69 vs 0.72
   - Causal editing: 0.73 vs 0.72

**Implication**: Current SAEs may not reliably decompose model internal mechanisms. The strong reconstruction quality does not guarantee meaningful feature decomposition. This is a crisis for the entire field and directly motivates research on absorption as a root cause.

### Project Iteration 6 Result: Absorption Does Not Predict Steering Effectiveness

A controlled matched experiment (N=50 feature pairs, controlling for activation frequency and decoder L2 norm) found:

| Analysis | Result | Interpretation |
|----------|--------|----------------|
| Aggregated (all beta values) | p=0.299 | Null result - no absorption-steering relationship |
| Beta=5 | p>0.05 | No significant difference |
| Beta=10 | p>0.05 | No significant difference |
| Beta=15 | p>0.05 | No significant difference |
| Beta=20 | p=0.015 | Significant, but **direction reversal** (low-absorption > high-absorption) |
| Beta=25 | p>0.05 | No significant difference |

**Confound explanation for beta=20 reversal**: High-absorption features have higher decoder L2 norms by construction. At high steering magnitudes, these features may saturate the residual stream faster, producing smaller incremental effects.

### Current SAE Architectures (by training approach)

| Architecture | Key Mechanism | Absorption Resistance | Reconstruction | Reference |
|---|---|---|---|---|
| **Standard SAE** | ReLU + L1 penalty | Poor | Excellent | Cunningham et al. (2023) |
| **TopK SAE** | Hard threshold to K active features | Moderate | Good | Gao et al. (2024) |
| **JumpReLU SAE** | Discontinuous activation + STE for L0 | Moderate | Very Good | Rajamanoharan et al. (2024) |
| **Gated SAE** | Learned gating mechanism | Moderate | Good | Mudide et al. (2024) |
| **Switch SAE** | Learned routing mechanism | Moderate | Good | Mudide et al. (2024) |
| **OrtSAE** | Orthogonality penalty | **High (65% reduction)** | Good | Korznikov et al. (2025) |
| **ATM SAE** | Temporal importance masking | **High (40% reduction vs JumpReLU)** | Very Good | Li & Ren (2025) |
| **HSAE** | Hierarchical structure learning | High (by design) | Good | Luo et al. (2026) |

### Benchmark Framework (SAEBench)

SAEBench evaluates across **8 metrics**:
1. **L0 Sparsity**: Average active features per token
2. **CE Loss Recovered**: Cross-entropy recovered vs original model
3. **Explained Variance**: Reconstruction quality
4. **Feature Interpretability**: LLM-based and human evaluation of feature descriptions
5. **Feature Disentanglement**: Independence between feature representations
6. **Downstream Task Performance**: Feature-based probing accuracy
7. **Unlearning**: Ability to remove targeted knowledge
8. **Feature Sensitivity**: Reliability across semantically similar inputs

Key finding: **Gains on proxy metrics do not reliably translate to practical performance.** For example, Matryoshka SAEs slightly underperform on proxy metrics but substantially outperform on feature disentanglement, with advantage growing at scale.

### Feature Sensitivity Findings (Tian et al., 2025)

- Many interpretable features have **poor sensitivity** (fail to activate on semantically similar texts)
- Human evaluation confirms generated texts genuinely resemble original activating examples
- **Sensitivity declines with SAE width** across 7 SAE variants
- This represents a new failure mode orthogonal to absorption

### Feature Absorption Detection Protocol

Chanin et al. (2024) define absorption detection using:
- **First-letter classification task**: Tokens split into train/test sets
- **K-sparse probing**: Identify latents relevant to classification
- **Absorption criterion**: A feature is "absorbed" if feature-split latents fail classification (threshold tau_fs = 0.03) but a different latent with cosine similarity >= tau_ps = 0.025 explains >= tau_pa = 0.4 of probe projection

### Available Pre-trained SAEs

| Release | Model | Layers | Location |
|---------|-------|--------|----------|
| `gpt2-small-res-jb` | GPT-2 Small | Multiple residual streams | SAELens |
| `gemma-2b-res` | Gemma 2B | Residual streams | SAELens |
| `gemma-2-9b-res` | Gemma 2 9B | Residual streams | SAELens |
| Various on HuggingFace | Various | Various | Tag: `saelens` |

**Neuronpedia** (neuronpedia.org) provides an interactive feature browser with pre-trained SAE weights and human-interpretable feature descriptions.

---

## 4. Identified Research Gaps

### CRITICAL Gap 0: Validating that SAEs Actually Decompose Model Internals

The 2026 "Sanity Checks" paper reveals that SAEs recover only 9% of ground-truth features in synthetic settings and match random baselines on real tasks. **This is the most important gap**: understanding why SAEs fail to decompose model mechanisms and whether absorption is the root cause. Without solving this, all other SAE research is on unstable ground.

### CRITICAL Gap 0b: New Angles After Null Result (Iteration 6)

The iteration 6 finding that absorption does NOT predict steering effectiveness (p=0.299) invalidates the assumed absorption -> peripheralization -> reduced steering causal chain. This opens new questions:

1. **Why do absorbed features still appear interpretable?** If absorption doesn't affect steering, why do absorbed features show interpretable activation patterns? Are interpretations artifacts of selection bias?
2. **What explains the beta=20 reversal?** The saturation confound explanation is post-hoc; systematic study needed.
3. **Is absorption actually harmful?** If absorbed features still work for steering, is absorption actually a problem for interpretability?
4. **Alternative absorption definitions needed**: The UAS-based absorption metric may not capture the right phenomenon. What definition actually correlates with meaningful feature decomposition?

### CRITICAL Gap 0c: UAS Metric Saturation (Iteration 7)

Iteration 7 discovered that the UAS-based Chanin protocol produces UAS=1.0 for **all features** at layer 8, making absorption distinction impossible. Key questions:

1. **Metric saturation vs. genuine absorption**: Does UAS=1.0 mean all features are absorbed, or does the metric saturate at certain layers?
2. **Layer-dependent calibration**: Is universal UAS threshold inappropriate? Should absorption metrics be layer-specific?
3. **Architecture-agnostic alternatives**: What metrics can detect absorption without the saturation failure (causal intervention consistency, feature sensitivity, cross-layer information preservation)?
4. **Which layers can reliably measure absorption?** Understanding the saturation boundary across layers could provide insight into where and how absorption manifests.

### Gap 1: Theoretical Understanding of Absorption Root Causes
The field has documented feature absorption empirically but lacks a formal theoretical framework explaining *why* it emerges from sparsity optimization. Existing explanations (L1 penalty minimization, hierarchical feature relationships) are descriptive rather than predictive. A principled information-theoretic or optimization-theoretic account is needed.

### Gap 2: Comprehensive Absorption Quantification Metrics
Current absorption detection relies on specific probe tasks (first-letter classification). A **general, architecture-agnostic metric** for quantifying absorption across arbitrary feature hierarchies does not exist. SAEBench does not yet include absorption-specific metrics. The project iteration 6 found that UAS-based absorption does not predict steering, suggesting the metric may not capture the right phenomenon.

### Gap 3: Interaction Between Absorption and Other SAE Pathologies
Feature absorption likely interacts with dead features, poor sensitivity, and feature composition in complex ways. The **joint distribution** of these failure modes and their compounded effects on interpretability** is poorly understood.

### Gap 4: Scale Dependence of Absorption
Most absorption studies are on small-to-medium models (GPT-2, Gemma-2B, Gemma-2-9B). Whether absorption becomes more or less severe at frontier model scales (GPT-4 class, Claude class) is unknown. Larger models may have more complex feature hierarchies, potentially worsening absorption.

### Gap 5: Cross-Layer Absorption Dynamics
Features do not exist in isolation across layers. The **migration of feature information across layers** during absorption (e.g., if a parent feature is absorbed in layer L, does its information migrate to layer L+1?) is not well studied.

### Gap 6: Robust Interpretability Evaluation Without Ground Truth
Feature absorption undermines the reliability of SAE-based interpretability: if absorbed features do not fire consistently, human interpretations may be based on incomplete or misleading activation patterns. A framework for **calibrating interpretability claims in the presence of absorption** is needed.

### Gap 7: Causal Interventions on Absorbed Features
If a feature is absorbed, intervening on it (e.g., steering, ablation) may not produce the expected behavioral effect. The **validity of SAE-based causal interventions** under absorption is not systematically evaluated.

### Gap 8: Automatic Hierarchical Structure Discovery
HSAE and hierarchical semantic SAEs require explicit hierarchical structure input or produce post-hoc hierarchies. **Automatic, unsupervised discovery of feature hierarchies** that can then inform absorption-robust training is underexplored.

### Gap 9: Causal Validity of Features Under Absorption
The sanity check paper shows random features can match SAEs on interpretability/probing but their causal validity (ability to steer model behavior) is unclear. **Understanding the relationship between absorption, interpretability, and causal validity** is critical for determining which features are actually useful for mechanistic understanding.

---

## 5. Available Resources

### Open-source Code

| Repository | Description | Language | License | Relevance |
|---|---|---|---|---|
| [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens) | Primary library for SAE training, analysis, and feature steering | Python | MIT | **Essential** - foundation for all SAE work |
| [jbloomAus/SAELens tutorials](https://github.com/jbloomAus/SAELens/tree/main/tutorials) | Basic loading, training, and steering tutorials | Python | MIT | High |
| [Neuronpedia/sae-bench](https://github.com/Neuronpedia/sae-bench) | SAEBench evaluation framework; 200+ pre-trained SAEs | Python | Apache 2.0 | **Essential** - benchmark infrastructure |
| [neuronpedia.org/sae-bench](https://www.neuronpedia.org/sae-bench/info) | Interactive visualization of SAE metrics | Web | - | High |
| [shamakhn/SAE-FiRE](https://github.com/shan23chen/MOSAIC) | SAE for financial classification; feature selection pipeline | Python | - | Medium |
| [maxdreyer/PURE](https://github.com/maxdreyer/PURE) | Polysemantic neuron disentanglement via circuits | Python | - | Low (CNN-focused) |
| [Prisma-Multimodal/ViT-Prisma](https://github.com/Prisma-Multimodal/ViT-Prisma) | Vision mechanistic interpretability toolkit; 75+ vision models | Python | - | Low (vision focus) |
| [jbloomaus/SAELens/docs](https://jbloomaus.github.io/SAELens/) | Official SAELens documentation | - | MIT | High |

### Datasets

| Dataset | Description | Used By |
|---|---|---|
| WikiText-103 | Wikipedia articles, ~100M tokens | Standard SAE training (Gao et al., Li & Ren) |
| monology/pile-uncopyrighted | The Pile, copyright-filtered | SAELens default training dataset |
| BIAS_IN_BIOS | Gender bias in biographical data | Sparse probing benchmarks |
| WinoMT | Gender coreference (WinoBias) | Gender bias evaluation |
| Custom first-letter datasets | Token classification for absorption detection | Chanin et al. (2024) |

### Pretrained Models

| Model | SAE Release | Where to Access |
|---|---|---|
| GPT-2 Small | `gpt2-small-res-jb` | SAELens / HuggingFace |
| Gemma 2B | `gemma-2b-res` | SAELens / HuggingFace |
| Gemma 2 9B | `gemma-2-9b-res` | SAELens / HuggingFace |
| Llama models | Various | Neuronpedia |

### Tools and Platforms

| Tool | Purpose |
|---|---|
| [Neuronpedia](https://www.neuronpedia.org) | Interactive feature browser, SAE visualization, SAEBench results |
| TransformerLens | Model loading and activation extraction |
| SAELens HookedSAETransformer | Integrated TransformerLens + SAE pipeline |

---

## 6. Implications for Idea Generation

### Context: A Crisis in SAE Research + Null Result

The 2026 "Sanity Checks" paper fundamentally challenges SAE interpretability research. If random baselines match trained SAEs, the entire research program requires rethinking. **Feature absorption is likely a primary mechanism** explaining this failure: absorbed features may appear interpretable by coincidence (activated examples look meaningful) but do not correspond to genuine model computations.

However, **iteration 6's null result complicates this narrative**: absorption does NOT predict steering effectiveness in a controlled study. This means:
1. The assumed "absorption causes peripheralization" causal story may be wrong
2. Absorbed features may still be causally valid for steering
3. The interpretation community may need to reconsider what absorption means for reliability

### New Research Angles (After Null Result)

Given that all original hypotheses were rejected (p=0.299 aggregated null), the following new directions are more promising:

1. **What does absorption actually affect?** If not steering effectiveness, does absorption affect: feature stability over time, feature sensitivity, cross-layer consistency, or downstream task transfer?

2. **Saturation dynamics at high steering magnitudes**: The beta=20 reversal (low-absorption > high-absorption) suggests high-norm decoder directions saturate. Systematic study of steering saturation as a function of decoder norm could yield insights.

3. **Reconsidering absorption metrics**: UAS may not capture the right phenomenon. Alternative metrics based on: (a) causal intervention consistency, (b) feature stability across semantically similar inputs, (c) cross-layer information preservation.

4. **Absorption as epiphenomenon**: Perhaps absorption is a symptom rather than a cause of SAE failure. The real problem might be: (a) insufficient training data to disambiguate hierarchical features, (b) the L1 penalty inherently encourages feature merging, (c) the hierarchical structure of features doesn't match the linear structure SAEs can represent.

5. **Compositional recovery**: If absorption merges features, can we decompose absorbed features through compositional analysis? E.g., can the "India" direction be recovered from a weighted combination of "Asia" + other child features?

### Directions Worth Exploring

1. **Information-theoretic absorption bounds**: Derive theoretical limits on feature absorption given the statistical structure of hierarchical features and sparsity constraints. This connects to the superposition-as-compression literature (Bereska et al., 2025) and capacity-allocation models (Michaud et al., 2025).

2. **Hierarchical-robust SAE training**: Build on HSAE's structural constraint loss but make it absorption-aware -- penalize absorption during training rather than only discovering hierarchies post-hoc. The ATM approach of temporal importance tracking combined with hierarchical constraints is promising.

3. **Multi-task absorption detection**: Generalize the first-letter probe to arbitrary feature hierarchies (e.g., using ConceptNet or WordNet as ground truth) to create a scalable absorption benchmark. Integrate this into SAEBench as a dedicated metric.

4. **Absorption-aware causal interventions**: Study how feature absorption affects the reliability of steering/ablation experiments. Develop correction methods or confidence estimates for intervention effects under absorption.

5. **Scale-up studies of absorption**: Evaluate absorption severity across model families (GPT-2 -> Llama -> Claude-scale), layer types (attention vs. MLP), and training stages (pre-training vs. fine-tuning).

6. **Cross-modal absorption**: Extend absorption analysis to vision (ViT-SAEs via Prisma), audio (DiffRhythm-VAE SAEs), and diffusion models (DLM-Scope). Does the phenomenon persist across modalities?

7. **Compositional feature decomposition**: Rather than treating absorption as a bug, study whether absorbed features can be recovered through compositional analysis -- e.g., if "India" is absorbed into "Asia", can the original direction be reconstructed by combining "India"-related child features?

### New Research Directions (Post-Iteration 7 UAS Saturation)

With UAS=1.0 for all features at layer 8 making the Chanin protocol unusable at this layer, the following directions become critical:

1. **Feature sensitivity as absorption proxy**: Tian et al. (2025) demonstrate that many interpretable features have poor sensitivity (fail to activate on semantically similar texts). If absorbed features systematically show poorer sensitivity, this metric provides an absorption indicator without the saturation problem.

2. **Cross-layer information migration analysis**: Track whether parent feature information migrates to child features across layers during absorption. The migration pattern itself becomes the absorption signature.

3. **Causal intervention consistency**: Test features across multiple semantically varied contexts. Inconsistent behavioral effects may indicate absorption-induced peripheralization.

4. **Layer-wise saturation boundary mapping**: Systematically measure at which layers UAS saturates and whether this correlates with model depth/architecture properties. This boundary itself provides diagnostic information.

5. **Compositional recovery of absorbed features**: If absorption merges parent features into child features linearly, can we decompose absorbed directions using sparse weighted combinations of related child features?

6. **Alternative absorption metrics**: Move beyond UAS to architecture-agnostic metrics measuring: (a) decoder direction overlap with parent concept directions, (b) activation pattern similarity between hierarchical feature pairs, (c) information-theoretic measures of feature disentanglement.

### Saturated Directions

- **Standard SAE architecture improvements** (different activation functions, L1/L2 regularization ratios) without addressing absorption explicitly -- marginal returns are expected.
- **Single-feature interpretation studies** that do not account for absorption -- individual feature descriptions may be unreliable if absorption is prevalent.
- **Pure proxy-metric optimization** (L0, CE loss recovered) without validating practical interpretability -- SAEBench has already demonstrated this disconnect.
- **Any SAE method without random baseline comparison** -- the field now requires demonstrating that trained SAEs actually outperform random baselines on meaningful tasks.
- **Absorption-steering relationship (original hypothesis)**: This direction is now closed by the null result -- absorption does not predict steering effectiveness.
- **UAS-based absorption measurement at certain layers**: The Chanin protocol's UAS metric saturates to 1.0 at layer 8, making this layer's absorption unmeasurable by this method. The protocol requires layer-specific calibration or alternative metrics.

---

## 7. Implementation Strategy Recommendations

### Critical Path Forward: Addressing the Sanity Check Crisis + Null Result

The 2026 "Sanity Checks" paper (Korznikov et al.) reveals that current SAEs may not reliably decompose model internals. This, combined with the iteration 6 null result showing absorption does not predict steering effectiveness, fundamentally changes the research agenda:

1. **Absorption may not be the root cause of SAE failure**: The causal story "absorption -> peripheralization -> reduced steering effectiveness" was not validated
2. **Absorption may be an epiphenomenon**: Rather than causing problems, it may be a symptom of deeper issues (training data limitations, hierarchical feature structure incompatibility with linear decomposition)
3. **Causal validity is paramount**: Features that can be steered/ablated with consistent behavioral effects should be prioritized over those that merely appear interpretable

### Implementation Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|---|---|---|---|---|
| SAELens (jbloomAus/SAELens) | High | MIT | **Adopt** | Comprehensive SAE training, loading, analysis; essential foundation |
| SAEBench (Neuronpedia/sae-bench) | High | Apache 2.0 | **Adopt** | Benchmark infrastructure with 200+ pre-trained SAEs; essential for evaluation |
| Neuronpedia (neuronpedia.org) | Medium | - | **Adopt** | Interactive visualization and pre-trained weights; good for exploratory analysis |
| Chanin et al. absorption code | High | - | **Adopt** | First-letter absorption detection; direct baseline to compare against |
| OrtSAE training procedure | High | - | **Extend** | Orthogonality penalty shows 65% absorption reduction; extend to hierarchical setting |
| ATM training procedure | High | - | **Extend** | Temporal masking achieves best absorption scores; combine with orthogonality for hybrid |
| HSAE hierarchical learning | Medium | - | **Extend** | Structural constraint loss is novel; adapt to absorption-robust training objective |
| TransformerLens HookedSAETransformer | High | MIT | **Adopt** | Integrated model+SAE pipeline; required for activation extraction and analysis |
| Feature Sensitivity measurement | Medium | - | **Compose** | Novel metric; combine with absorption detection to get multi-dimensional feature quality |
| Sanity Check baselines | High | - | **Adopt** | Must include random baselines in all evaluations to validate SAE value-add |
| SAELens tutorials | High | MIT | **Adopt** | Well-documented training and steering pipelines; accelerates experiment setup |
| MOSAIC (SAE-FiRE) | Low | - | **Observe** | Application-focused; less relevant to absorption mechanisms |

### Priority Implementation Path (Post-Iteration 7 UAS Saturation)

Given the iteration 7 finding that UAS=1.0 for all features at layer 8, the following revised path addresses measurement limitations:

1. **Layer-wise metric validation**: Before applying UAS-based absorption measurement, validate metric behavior across layers using ground-truth synthetic features where absorption is known
2. **Multi-metric framework**: Combine multiple absorption indicators (UAS, feature sensitivity, cross-layer information preservation, causal consistency) to triangulate absorption severity
3. **Saturation boundary mapping**: Systematically test which layers exhibit UAS saturation and characterize the boundary; this itself provides diagnostic value
4. **Feature sensitivity baseline**: Implement Tian et al. (2025) sensitivity metric as absorption proxy, validated on ground-truth hierarchical features
5. **Compositional recovery experiments**: Design experiments to test whether absorbed parent features can be reconstructed from child feature combinations
6. **Ground-truth validation**: Use synthetic SAEs with known ground-truth feature hierarchies to calibrate absorption metrics before applying to real models

---

## References (Key Papers)

1. Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., & Bloom, J. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. *arXiv:2409.14507*.
2. Li, T.E. & Ren, J. (2025). Time-Aware Feature Selection: Adaptive Temporal Masking for Stable Sparse Autoencoder Training. *arXiv:2510.08855*.
3. Korznikov, A., Galichin, A., Dontsov, A., Rogov, O., Tutubalina, E., & Oseledets, I. (2025). OrtSAE: Orthogonal Sparse Autoencoders Uncover Atomic Features. *arXiv:2509.22033*.
4. Luo, Y., Zhan, Y., Jiang, J., Liu, T., Wu, M., Zhou, Z., & Dong, B. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. *arXiv:2602.11881*.
5. Karvonen, A., Rager, C., Lin, J., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. *arXiv:2503.09532*.
6. Tian, C., Tian, K., & Hu, N. (2025). Measuring Sparse Autoencoder Feature Sensitivity. *arXiv:2509.23717*.
7. Li, Y., Michaud, E.J., Baek, D.D., Engels, J., Sun, X., & Tegmark, M. (2024). The Geometry of Concepts: Sparse Autoencoder Feature Structure. *arXiv:2410.19750*.
8. Muchane, M., Richardson, S., Park, K., & Veitch, V. (2025). Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. *arXiv:2506.01197*.
9. Rajamanoharan, S., Lieberum, T., Sonnerat, N., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. *arXiv:2407.14435*.
10. Gao, L., Dupre la Tour, T., Tillman, H., et al. (2024). Scaling and Evaluating Sparse Autoencoders. *arXiv:2406.04093*.
11. Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. *arXiv:2309.08600*.
12. Michaud, E.J., Gorton, L., & McGrath, T. (2025). Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds. *arXiv:2509.02565*.
13. Bereska, L., Tzifa-Kratira, Z., Samavi, R., & Gavves, E. (2025). Superposition as Lossy Compression: Measure with Sparse Autoencoders and Connect to Adversarial Vulnerability. *arXiv:2512.13568*.
14. Gurnee, W., Nanda, N., Pauly, M., et al. (2023). Finding Neurons in a Haystack: Case Studies with Sparse Probing.
15. Mudide, A., Engels, J., Michaud, E.J., et al. (2024). Efficient Dictionary Learning with Switch Sparse Autoencoders. *arXiv:2410.19999*.
16. Korznikov, A., Galichin, A., Dontsov, A., Rogov, O., Oseledets, I., & Tutubalina, E. (2026). Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? *arXiv:2602.14111*.
17. Baker, Z. & Li, Y. (2025). Analysis of Variational Sparse Autoencoders. *arXiv:2509.22994*.
18. Erdogan, E. & Lucic, A. (2025). Group Equivariance Meets Mechanistic Interpretability: Equivariant Sparse Autoencoders. *arXiv:2511.09432*.
19. Paulo, G. & Belrose, N. (2025). Evaluating SAE Interpretability Without Explanations. *arXiv:2507.08473*.
20. Modi, S.K., Lim, Z.P., Cao, Y., et al. (2025). SOSAE: Self-Organizing Sparse AutoEncoder. *arXiv:2507.04644*.
21. Shi, W., Li, S., Liang, T., et al. (2025). Route Sparse Autoencoder to Interpret Large Language Models. *arXiv:2503.08200*.