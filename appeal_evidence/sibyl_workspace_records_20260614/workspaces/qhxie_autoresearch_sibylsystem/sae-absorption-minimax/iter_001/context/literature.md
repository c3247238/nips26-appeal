# Literature Survey Report

**Research Topic**: Feature Absorption in Sparse Autoencoders: Systematic Analysis and Quantification of Its Causes, Patterns, and Impact on Interpretability
**Survey Date**: 2026-04-27
**arXiv Search Keywords**: ["sparse autoencoder feature absorption superposition", "SAE feature absorption dead features polysemantic interpretability", "mechanistic interpretability sparse autoencoder limitation bias artifact", "SAEBench sparse autoencoder benchmark comprehensive evaluation", "JumpReLU sparse autoencoder interpretability reconstruction", "superposition lossy compression sparse autoencoder", "SAE scaling feature manifolds capacity allocation"]
**Web Search Keywords**: ["SAE sparse autoencoder feature absorption limitation mechanistic interpretability 2025", "sparse autoencoder dead features superposition neural network interpretability", "SAELens sparse autoencoder GitHub benchmark state of the art 2025", "SAE feature absorption JumpReLU TopK comparison benchmark Neuronpedia", "sparse autoencoder feature sensitivity evaluation benchmark 2025 GitHub"]

---

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as a dominant tool in mechanistic interpretability for decomposing the polysemantic activations of large language models into sparse, human-interpretable features. The field has progressed rapidly since Anthropic's seminal "Towards Monosemanticity" (2023), with multiple SAE architectures (standard ReLU, TopK, JumpReLU, Gated, Switch) and training algorithms now available. SAELens has become the standard library for training and analyzing SAEs.

The field now recognizes several fundamental limitations that undermine SAE reliability for robust interpretability. **Feature absorption** is arguably the most critical: when underlying features form hierarchical relationships (e.g., "India" implies "Asia"), SAEs trained with sparsity penalties tend to absorb parent features into child features, causing seemingly monosemantic features to fail to fire where they should. This was first systematically studied by Chanin et al. (2024) in "A is for Absorption."

Beyond absorption, the field has identified **feature composition** (independent features merging into composite representations), **dead features** (features that never activate), **poor feature sensitivity** (features failing to generalize to semantically similar contexts), and scaling pathologies related to **feature manifolds**. SAEBench (Karvonen et al., 2025) provides a comprehensive benchmark framework to evaluate these issues, revealing that gains on proxy metrics do not reliably translate to better practical performance.

The dominant paradigm is now shifting from single-level SAEs to hierarchical approaches (HSAE, hierarchical semantics architectures) that explicitly model parent-child feature relationships, and to orthogonal SAE variants (OrtSAE) that enforce disentanglement between learned features.

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

---

## 3. SOTA Methods and Benchmarks

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

### Gap 1: Theoretical Understanding of Absorption Root Causes
The field has documented feature absorption empirically but lacks a formal theoretical framework explaining *why* it emerges from sparsity optimization. Existing explanations (L1 penalty minimization, hierarchical feature relationships) are descriptive rather than predictive. A principled information-theoretic or optimization-theoretic account is needed.

### Gap 2: Comprehensive Absorption Quantification Metrics
Current absorption detection relies on specific probe tasks (first-letter classification). A **general, architecture-agnostic metric** for quantifying absorption across arbitrary feature hierarchies does not exist. SAEBench does not yet include absorption-specific metrics.

### Gap 3: Interaction Between Absorption and Other SAE Pathologies
Feature absorption likely interacts with dead features, poor sensitivity, and feature composition in complex ways. The **joint distribution** of these failure modes and their compounded effects on interpretability is poorly understood.

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

### Directions Worth Exploring

1. **Information-theoretic absorption bounds**: Derive theoretical limits on feature absorption given the statistical structure of hierarchical features and sparsity constraints. This connects to the superposition-as-compression literature (Bereska et al., 2025) and capacity-allocation models (Michaud et al., 2025).

2. **Hierarchical-robust SAE training**: Build on HSAE's structural constraint loss but make it absorption-aware -- penalize absorption during training rather than only discovering hierarchies post-hoc. The ATM approach of temporal importance tracking combined with hierarchical constraints is promising.

3. **Multi-task absorption detection**: Generalize the first-letter probe to arbitrary feature hierarchies (e.g., using ConceptNet or WordNet as ground truth) to create a scalable absorption benchmark. Integrate this into SAEBench as a dedicated metric.

4. **Absorption-aware causal interventions**: Study how feature absorption affects the reliability of steering/ablation experiments. Develop correction methods or confidence estimates for intervention effects under absorption.

5. **Scale-up studies of absorption**: Evaluate absorption severity across model families (GPT-2 -> Llama -> Claude-scale), layer types (attention vs. MLP), and training stages (pre-training vs. fine-tuning).

6. **Cross-modal absorption**: Extend absorption analysis to vision (ViT-SAEs via Prisma), audio (DiffRhythm-VAE SAEs), and diffusion models (DLM-Scope). Does the phenomenon persist across modalities?

7. **Compositional feature decomposition**: Rather than treating absorption as a bug, study whether absorbed features can be recovered through compositional analysis -- e.g., if "India" is absorbed into "Asia", can the original direction be reconstructed by combining "India"-related child features?

### Saturated Directions

- **Standard SAE architecture improvements** (different activation functions, L1/L2 regularization ratios) without addressing absorption explicitly -- marginal returns are expected.
- **Single-feature interpretation studies** that do not account for absorption -- individual feature descriptions may be unreliable if absorption is prevalent.
- **Pure proxy-metric optimization** (L0, CE loss recovered) without validating practical interpretability -- SAEBench has already demonstrated this disconnect.

### Cross-Domain Analogies with Potential

1. **Sparse coding in neuroscience**: The problem of "redundant coding" and "receptive field merging" in visual cortex has parallels to feature absorption. Insights from neuroscience sparse coding theory may transfer to SAE training.

2. **Dictionary learning in signal processing**: Classical compressed sensing and dictionary learning have well-established theoretical results on uniqueness conditions and atom interference. These could inform absorption-free SAE design.

3. **Concept bottleneck models**: Hierarchical concept learning in CBMs addresses similar problems of semantic hierarchy representation. Techniques from CBM hierarchical training may apply to SAEs.

4. **Molecular representation learning**: Feature absorption resembles "functional grouping" in molecular fingerprints, where substructures are absorbed into larger functional groups. Countermeasures developed in cheminformatics may transfer.

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|---|---|---|---|---|
| SAELens (jbloomAus/SAELens) | High | MIT | **Adopt** | Comprehensive SAE training, loading, analysis; essential foundation |
| SAEBench (Neuronpedia/sae-bench) | High | Apache 2.0 | **Adopt** | Benchmark infrastructure with 200+ pre-trained SAEs; essential for evaluation |
| Neuronpedia (neuronpedia.org) | Medium | - | **Adopt** | Interactive visualization and pre-trained weights; good for exploratory analysis |
| Chanin et al. absorption code | High | - | **Adopt** | First-letter absorption detection; direct baseline to compare against |
| OrtSAE training procedure | High | - | **Extend** | Orthogonality penalty shows strong absorption reduction; extend to hierarchical setting |
| ATM training procedure | High | - | **Extend** | Temporal masking achieves best absorption scores; combine with orthogonality for hybrid approach |
| HSAE hierarchical learning | Medium | - | **Extend** | Structural constraint loss is novel; adapt to absorption-robust training objective |
| TransformerLens HookedSAETransformer | High | MIT | **Adopt** | Integrated model+SAE pipeline; required for activation extraction and analysis |
| Feature Sensitivity measurement | Medium | - | **Compose** | Novel metric; combine with absorption detection to get multi-dimensional feature quality |
| SAELens tutorials | High | MIT | **Adopt** | Well-documented training and steering pipelines; accelerates experiment setup |
| MOSAIC (SAE-FiRE) | Low | - | **Observe** | Application-focused; less relevant to absorption mechanisms |

### Priority Implementation Path

1. **Foundation**: Use SAELens for SAE loading and training infrastructure
2. **Baseline**: Implement Chanin et al. absorption detection protocol
3. **Baseline SAEs**: Load SAEBench pre-trained SAEs (200+ across 8 architectures) for immediate experiments
4. **Hybrid Method**: Combine OrtSAE orthogonality penalty with ATM temporal masking for absorption-robust training
5. **Evaluation**: Integrate absorption metrics into SAEBench framework
6. **Analysis**: Use Neuronpedia for exploratory visualization, SAELens for programmatic analysis

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
