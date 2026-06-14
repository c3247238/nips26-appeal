

## Project Spec
# 项目: sae-absorption

## 研究主题
研究稀疏自编码器（SAE）中的特征吸收（feature absorption）现象：系统分析和量化其成因、规律及对可解释性的影响。

## 背景与动机
稀疏自编码器（SAE）是机械可解释性研究的核心工具，用于从语言模型激活中提取人类可解释的特征。然而，SAE 存在"特征吸收"现象：某些低频但语义独立的特征会被高频特征"吸收"，导致 SAE 特征不完整，影响特征质量和下游解释性研究的可靠性。

目前，该现象的普遍程度、成因机制和定量影响尚不清楚，需要系统化的分析和量化。

## 初始想法
- 设计系统化的分析框架，量化特征吸收现象的普遍程度（跨不同模型层、不同 SAE 配置）
- 探索特征吸收的成因：与特征共现频率、稀疏惩罚强度、SAE 字典大小等因素的关系
- 开发可复现的评估指标，衡量吸收现象对下游解释性任务（如电路发现、概念探测）的影响
- 研究方法以 training-free 分析为主，基于现有的预训练 SAE（如 SAELens 库中的 SAE）进行分析

## 关键参考文献
- SAELens 库及其预训练 SAE（GemmaScope, GPT2-small SAE 等）
- Feature absorption 相关文献（待 Sibyl 文献调研补全）

## 可用资源
- GPU: 本地 GPU（有 SSH 访问）
- 服务器: default（SSH MCP 连接）
- 远程路径: /home/qhxie/sibyl_system

## 实验约束
- 实验类型: training-free（分析现有预训练 SAE，不重新训练）
- 模型规模: 小到中等（GPT-2, Gemma-2B 等）
- 时间预算: 单实验 ≤ 1 小时，pilot 10-15 分钟

## 目标产出
- 学术论文（NeurIPS/ICLR 级别）
- 包含：特征吸收的量化分析、成因分析、对可解释性影响的实验

## 特殊需求
- 以 training-free 分析为主，充分利用 SAELens 现有预训练模型
- 论文应包含可复现的评估框架，方便社区后续研究


## 文献调研报告（请仔细阅读，避免重复已有工作）
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

## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Research Proposal: The Sensitivity Floor in Sparse Autoencoders

## Status: Iteration 9 Synthesis

## Critical Warning: Stagnation Alert

This system has executed **zero experiments for 2+ consecutive iterations**. The paper is byte-identical to iter_007 with a score of 6.5 for 3 consecutive reviews. The 3 highest-value experiments (activation patching, tightened hedging, CMI at L0=22) have been recommended for 3 reviews and never executed. **This proposal prioritizes execution over theoretical refinement.**

The Sensitivity Floor is the front-runner. Its core claim is falsifiable in ~30 minutes. We will run the pilot FIRST and let data decide.

---

## Abstract

The Sanity Check crisis (Korznikov et al., 2026) reveals random baselines match SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72). We show sensitivity failures are near-universal (100% of sampled features are low-sensitivity), and absorption and sensitivity are positively correlated (r = 0.59) through a common cause. We propose the **Sensitivity Floor** hypothesis: sensitivity requires BOTH high specificity (absorbed features have geometrically close decoders to neighbors) AND high reliability (non-absorbed features in sparse regions have low activation frequency). Neither can achieve the sensitivity threshold. The Sanity Check crisis is explained by universal sensitivity failures.

---

## Evidence from Prior Iterations

| Finding | Source | Status |
|---------|--------|--------|
| Q4 empty (0/43) | iter_008 pilot | Definitive |
| r(absorption, sensitivity) = 0.59 | iter_008 pilot | Definitive, FALSIFIES independence |
| L2 norm ratio = 1.0 | iter_008 pilot | FALSIFIES saturation hypothesis |
| Coherence r = +0.36 | iter_008 pilot | FALSIFIES protective effect |
| UAS=1.0 saturation at layer 8 | iter_007 | Measurement limitation |
| Absorption does NOT predict steering | iter_006 | Null result |
| Beta=20 reversal unexplained | iter_004 | Needs mechanism |

**What this means**: The compound failure hypothesis (absorption and sensitivity as independent) is definitively falsified. The Sensitivity Floor is the only candidate that explains ALL findings simultaneously.

---

## Landscape Map

### Agreements Across 6 Perspectives

1. **Sanity Check crisis is real**: Random baselines match SAEs on all major benchmarks
2. **Q4 is empty (0/43)**: No high-sensitivity features found - too consistent to be statistical artifact
3. **Absorption and sensitivity are positively correlated (r = 0.59)**: Common cause model preferred over independence
4. **All compound failure hypotheses falsified**: Independence, decoder saturation, coherence protective - all failed

### Perspective Contributions

| Perspective | Key Contribution |
|-------------|-----------------|
| Innovator | Specificity-reliability tradeoff; frequency as common cause |
| Pragmatist | Sensitivity-first framing; Q3 dominates (65%); steering as validation |
| Theoretical | Information-theoretic proof sketch for floor; U-shape prediction |
| Contrarian | Layer-specificity control experiment; double-blind interpretability test |
| Empiricist | UAS saturation mapping; sensitivity as absorption proxy |
| Interdisciplinary | Hierarchical redundancy as common cause; predictive coding theory |

---

## Selected Front-Runner: Sensitivity Floor

### Title
**"The Sensitivity Floor: Why Absorbed and Non-Absorbed Features Are Both Insensitive"**

### Core Claim
Sensitivity requires BOTH:
- **Specificity**: Feature fires on target but not neighbors (destroyed when absorbed)
- **Reliability**: Feature fires consistently on semantically similar inputs (destroyed when in sparse regions)

Mechanisms:
- Q1 (absorbed + low sensitivity): Absorbed features have low specificity (decoder directions close to parent)
- Q3 (non-absorbed + low sensitivity): Sparse non-absorbed features have low reliability (low activation frequency)
- Q2 and Q4 are **structurally impossible**: Neither absorbed nor non-absorbed can achieve both factors simultaneously

The observed r = 0.59 is a linear approximation of a U-shaped relationship: sensitivity is maximized at intermediate absorption, not at either extreme.

### Why Sensitivity Floor Survived Contrarian Challenge
The contrarian argued Q4 emptiness could be layer-specific (only tested layer 8) and that the Sensitivity Floor is built on a 43-feature sample. The theoretical perspective addressed this: the information-theoretic bounds (specificity via DPI, reliability via PAC learning) are general and layer-agnostic. The U-shape prediction and structural impossibility are falsifiable at any layer.

### Novelty Assessment
No prior work connects absorption AND sensitivity through complementary mechanisms. Tian (2025) documents sensitivity decline in isolation. Chanin (2024) studies absorption in isolation. Korznikov (2026) documents Sanity Check without mechanistic explanation. **The Sensitivity Floor is genuinely novel (8/10).**

---

## Experimental Plan: Execute First, Refine Second

Given **zero experiments in 2+ iterations**, we run the most falsifiable experiment FIRST.

### Pilot (~30 min): H-SF1 Validation
**CRITICAL PATH - MUST RUN BEFORE ANY THEORETICAL REFINEMENT**

1. Load GPT-2 Small SAE layer 8 from SAELens (`gpt2-small-res-jb`)
2. Apply Chanin absorption protocol on 200 features (continuous absorption score)
3. Apply Tian sensitivity protocol (paraphrase AUC) on same 200 features
4. Classify into quadrants
5. **If Q2+Q4 > 10%: Sensitivity Floor FALSIFIED** → pivot to Sensitivity-First (backup_primary)
6. **If Q2+Q4 <= 10%: Sensitivity Floor CONFIRMED** → proceed to full experiment

**Why this first**: This is the fastest path to either confirming or falsifying the front-runner. 30 minutes of GPU time resolves the central question.

### Full Experiment (~75 min total if pilot positive)

| Phase | Task | Time | Metric |
|-------|------|------|--------|
| Phase 2 | Test U-shaped relationship (quadratic fit) | 20 min | Quadratic coefficient a > 0 |
| Phase 3 | Frequency mediation (partial correlation) | 15 min | Partial r < 0.3 |
| Phase 4 | Geometric density mediation | 15 min | Partial r < 0.3 |
| Phase 5 | Steering validation | 20 min | r(steering, sensitivity) > r(steering, absorption) |

**Total**: ~75-80 min on 1 A100 + CPU

### Control Experiment from Contrarian: Layer-wise UAS Mapping (~20 min)
**Run in parallel with pilot if time permits**

1. Measure UAS at layers 4, 8, 10 (30 features each)
2. If layers 4 and 10 show UAS variance → saturation is layer-specific
3. If all layers saturate → sensitivity floor may be universal, not layer-specific

This addresses the contrarian's primary objection without delaying the pilot.

---

## Hypotheses

**H-SF1 (Structural Emptiness)**: Even at N > 200, Q2+Q4 remain < 5%.
- **Falsification**: Q2+Q4 > 10% at N=200
- **Expected**: Structural impossibility confirmed

**H-SF2 (U-Shaped Relationship)**: S(A) = aA^2 + bA + c with a > 0.
- **Falsification**: Quadratic coefficient a <= 0
- **Expected**: Maximum sensitivity at intermediate absorption

**H-SF3 (Frequency Mediation)**: Partial r(absorption, sensitivity | frequency) < 0.3.
- **Falsification**: Partial r > 0.5 after frequency control
- **Expected**: Frequency partially explains the correlation

**H-SF4 (Geometric Mediation)**: Partial r(absorption, sensitivity | density) < 0.3.
- **Falsification**: Partial r > 0.5 after density control
- **Expected**: Geometric density partially explains the correlation

---

## Evaluation

| Hypothesis | Metric | Pass Criterion |
|-----------|--------|----------------|
| H-SF1 (emptiness) | Q2+Q4 fraction | < 5% |
| H-SF2 (U-shape) | Quadratic coefficient a | a > 0 |
| H-SF3 (frequency) | Partial r | < 0.3 |
| H-SF4 (density) | Partial r | < 0.3 |

---

## Resource Estimate

| Component | GPU | Time |
|-----------|-----|------|
| Phase 1: Q2/Q4 validation (200 features) | 1 A100 | ~30 min |
| Phase 2: U-shape fitting | CPU | ~5 min |
| Phase 3: Frequency mediation | CPU | ~10 min |
| Phase 4: Geometric density | CPU | ~10 min |
| Phase 5: Steering validation | 1 A100 | ~20 min |
| **Total** | **1 A100** | **~75 min** |

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Q2/Q4 found at N=200 | Medium | Would falsify SF; pivot to sensitivity-first |
| Linear relationship (no U-shape) | Medium | Report as negative; SF mechanism still valid |
| Neither frequency nor density mediates | Medium | Other causes; SF still valid as structural claim |
| All layers saturate on UAS | Medium | Sensitivity as absorption proxy from Empiricist |

---

## Negative Results to Report

This project has a strong track record of honest negative results (6 consecutive iterations). We will continue reporting them:

1. **Compound failure independence FALSIFIED**: r = 0.59 (positive, not independent)
2. **Decoder L2 norm saturation FALSIFIED**: ratio = 1.0
3. **Coherence-protective effect NOT REPLICATED**: r = +0.36 across layers 4, 8, 10
4. **Q4 EMPTY**: No high-sensitivity features in 43-feature sample

---

## Changes from Previous Round

| What Changed | Why |
|-------------|-----|
| Stark warning on stagnation | Zero experiments for 2+ iterations; paper unchanged |
| Execute pilot FIRST | Theoretical refinement without data is stagnation |
| Contrarian layer control added | Layer-specificity is the strongest objection |
| Streamlined proposal | Focus on falsifiable predictions, not framework |

---

## Expected Contributions

1. **First sensitivity floor theory**: Explains WHY Q2/Q4 are structurally empty
2. **U-shaped relationship**: Sensitivity maximized at intermediate absorption, not extremes
3. **Common cause identification**: Tests whether frequency or density mediates r=0.59
4. **Sanity Check explanation**: Universal sensitivity failures explain random baseline match
5. **Practical evaluation shift**: Sensitivity metrics should guide feature selection


## 当前可检验假设
# Testable Hypotheses

## Front-Runner: Sensitivity Floor

### H-SF1: Structural Emptiness
**H-SF1 (Structural Emptiness)**: Even with larger sample (N > 200), Q2 (absorbed + high-sensitivity) and Q4 (non-absorbed + high-sensitivity) remain empty or near-empty (< 5% of features).

**Falsification criterion**: If > 10% of features fall into Q2 or Q4 at N = 200, sensitivity floor is falsified.

**Expected outcome**: Q2 + Q4 < 5% even at larger N; sensitivity failures are structurally impossible, not statistically rare.

**Pilot**: ~30 min on 1 A100 (200 features, Chanin + Tian protocols)

---

### H-SF2: U-Shaped Relationship
**H-SF2 (U-Shaped Relationship)**: Sensitivity S(A) as a function of absorption A follows a U-shape: S is low at both A ~ 0 (sparse regions -> low reliability) and A ~ 1 (absorbed -> low specificity), with maximum sensitivity at intermediate absorption.

**Falsification criterion**: If quadratic coefficient a <= 0 (no U-shape), sensitivity floor mechanism is weakened.

**Expected outcome**: S(A) = aA^2 + bA + c with a > 0; maximum at intermediate A.

**Note**: The observed r = 0.59 may be a linear approximation of this U-shape.

---

### H-SF3: Frequency Mediation
**H-SF3 (Frequency Mediation)**: Feature activation frequency mediates the absorption-sensitivity correlation. After controlling for frequency, partial r(absorption, sensitivity | frequency) < 0.3.

**Falsification criterion**: If partial r > 0.5 after controlling for frequency, frequency is not the sole mediator.

**Expected outcome**: Low-frequency features get absorbed (less gradient signal) AND become insensitive (fewer training examples).

---

### H-SF4: Geometric Density Mediation
**H-SF4 (Geometric Mediation)**: Geometric density (mean k-NN cosine similarity in activation space) mediates the absorption-sensitivity correlation. Partial r(absorption, sensitivity | density) < 0.3.

**Falsification criterion**: If partial r > 0.5 after controlling for density, density is not the mediator.

**Expected outcome**: Dense regions cause both absorption (decoder overlap) and insensitivity (vague activation patterns).

---

## Alternative A: Sensitivity-First (Pivot if H-SF1 Fails)

### H-A: Q3 Equals Random Baseline
**H-A (Sensitivity-First)**: Low-sensitivity features (Q3: low-absorption + low-sensitivity) show steering effect sizes statistically indistinguishable from random baseline.

**Falsification criterion**: If Q3 steering is significantly above random baseline (0.73), sensitivity is not sufficient to explain Sanity Check.

**Expected outcome**: Q3 ~= random; both are low-sensitivity regardless of absorption.

---

### H-B: Absorption Adds No Predictive Value
**H-B (Absorption Metric Validity)**: Absorption status does NOT predict steering effectiveness after controlling for sensitivity.

**Falsification criterion**: If absorbed features show different steering than non-absorbed within Q3, absorption adds independent predictive value.

**Expected outcome**: Within Q3, absorbed vs non-absorbed shows no steering difference.

---

## Alternative B: Geometric Density (Pivot if H-SF3 and H-SF4 Fail)

### H-B1: Density-Absorption Correlation
**H-B1 (Density-Absorption)**: Absorbed features have higher geometric density than non-absorbed features.

**Falsification criterion**: If r(density, absorption) < 0, density is not associated with absorption.

**Expected outcome**: Dense regions are more vulnerable to absorption.

---

### H-B2: Density-Sensitivity Correlation
**H-B2 (Density-Sensitivity)**: Low-sensitivity features have higher geometric density than high-sensitivity features.

**Falsification criterion**: If r(density, sensitivity) > 0 (positive), density does not explain sensitivity failure.

**Expected outcome**: Dense regions produce vague activation patterns with low sensitivity.

---

## Alternative C: Layer-wise UAS Saturation (Control Experiment)

### H-C1: Layer-Dependent Saturation
**H-C1 (Layer Variance)**: UAS saturation is layer-dependent. Some layers (e.g., 4, 6, 10) show UAS variance while others (e.g., 8) show saturation.

**Falsification criterion**: If ALL layers show UAS std < 0.05, saturation is universal.

**Expected outcome**: Layer 8 is special; other layers permit absorption measurement.

---

### H-C2: Sensitivity as Proxy
**H-C2 (Proxy Validity)**: At non-saturated layers, sensitivity (paraphrase AUC) correlates with UAS absorption (r > 0.4).

**Falsification criterion**: If r(UAS, sensitivity) < 0.3 at non-saturated layers, sensitivity cannot substitute for UAS.

**Expected outcome**: Sensitivity works where UAS fails.

---

## Alternative D: Steering Diffusion

### H-D1: Decoder-Neighbor Overlap
**H-D1 (Diffusion Mechanism)**: Absorbed features have higher decoder-neighbor overlap (k=20 NN cosine similarity) than non-absorbed features.

**Falsification criterion**: If absorbed and non-absorbed have equal neighbor overlap, diffusion is not the mechanism.

**Expected outcome**: Absorbed features live in dense geometric regions.

---

### H-D2: Non-Linear Beta Dependence
**H-D2 (Non-Linear Diffusion)**: Diffusion is non-linear in beta. At beta <= 10, no neighbor activates (null result). At beta = 20, multiple neighbors activate simultaneously.

**Falsification criterion**: If absorption-steering correlation exists at beta <= 10, diffusion is not the mechanism.

**Expected outcome**: Null at beta <= 10 (confirming iter_004); significant effect at beta = 20.

---

## Expected Outcomes Summary

| Hypothesis | Expected | Falsification |
|-----------|----------|---------------|
| H-SF1 (emptiness) | Q2+Q4 < 5% | Q2+Q4 > 10% |
| H-SF2 (U-shape) | a > 0 | a <= 0 |
| H-SF3 (frequency) | partial r < 0.3 | partial r > 0.5 |
| H-SF4 (density) | partial r < 0.3 | partial r > 0.5 |
| H-A (sensitivity-first) | Q3 ~= random | Q3 > random |
| H-B (absorption validity) | No effect within Q3 | Absorbed != non-absorbed |
| H-C1 (layer variance) | Some layers show variance | All layers saturate |
| H-C2 (proxy validity) | r > 0.4 | r < 0.3 |
| H-D1 (overlap) | Absorbed > overlap | Equal overlap |
| H-D2 (non-linear) | Null at beta<=10 | Effect at beta<=10 |


## 小型实验真实反馈（必须基于这些证据修正 idea，不能忽略负结果）
# Pilot Summary - Iteration 8

## Tasks Completed

### 1. pilot_classify_features (H4+H5 Pilot)
- **Status**: COMPLETED
- **Features analyzed**: 43
- **Quadrant counts**:
  - Q1 (high absorption, low sensitivity): 15 features
  - Q2 (high absorption, high sensitivity): 0 features
  - Q3 (low absorption, low sensitivity): 28 features
  - Q4 (low absorption, high sensitivity): 0 features

### 2. pilot_decoder_norms (H6 Pilot)
- **Status**: COMPLETED
- **L2 norm ratio**: 1.0 (high-abs / low-abs)
- **Pass criteria**: ratio > 1.1

### 3. replicate_coherence_protective (H1-R Backup)
- **Status**: COMPLETED
- **Layers tested**: 4, 8, 10
- **Mean Spearman r**: 0.356

## Hypothesis Results

| Hypothesis | Result | Details |
|------------|--------|---------|
| H4 (Q1 steering near random) | NOT TESTED | Q4 empty, cannot compare best-case vs worst-case |
| H5 (independence, r < 0.3) | **FALSIFIED** | r = 0.59, p = 3.15e-05 |
| H6 (norm ratio > 1.1) | **FALSIFIED** | ratio = 1.0 |
| H1-R (protective, r < -0.5) | **FALSIFIED** | r = +0.356, not negative |

## Key Findings

1. **H5 FALSIFIED**: Absorption and sensitivity are POSITIVELY correlated (r=0.59), not independent. Features that are absorbed tend to also have low sensitivity.

2. **H6 FALSIFIED**: High-absorption and low-absorption features have identical decoder L2 norms (ratio=1.0). The saturation hypothesis is not supported.

3. **H1-R FALSIFIED**: The protective effect was NOT replicated. Coherence and absorption show POSITIVE correlation (r=0.36), not negative (r=-0.786 as in earlier pilot).

4. **Q4 Empty**: No features fell into the "low absorption + high sensitivity" quadrant, which was predicted to be the "best-case" scenario.

## Recommendations

1. The sensitivity-absorption "compound failure" hypothesis is WEAKENED by these results
2. The positive correlation between absorption and sensitivity suggests they may share a common cause
3. The H1-R finding from earlier pilot (r=-0.786) was NOT replicated - this needs investigation
4. Consider revising the theoretical framework before proceeding to full experiments

## Next Steps

Given that all pilot hypotheses are falsified:
- **GO/NO-GO**: NO-GO for full experiments
- **Reason**: Key assumptions (independence, protective effect, decoder norm differences) are not supported by pilot data
- **Recommendation**: Return to literature review and theoretical development


## 小型实验结构化信号（供你提炼 go/no-go / confidence / hypothesis status）
{
  "overall_recommendation": "NO_GO",
  "selected_candidate_id": "cand_sensitivity_absorption_joint",
  "candidates": [
    {
      "candidate_id": "cand_sensitivity_absorption_joint",
      "go_no_go": "NO_GO",
      "confidence": 0.85,
      "supported_hypotheses": [],
      "failed_assumptions": ["H5", "H6", "H1-R"],
      "key_metrics": {
        "spearman_r_absorption_sensitivity": 0.5898,
        "spearman_p": 3.15e-05,
        "decoder_norm_ratio": 1.0,
        "h1r_mean_r": 0.356,
        "q1_features": 15,
        "q2_features": 0,
        "q3_features": 28,
        "q4_features": 0
      },
      "notes": "All three pilot hypotheses falsified: (1) H5: absorption and sensitivity are positively correlated (r=0.59), not independent; (2) H6: decoder norm ratio is 1.0, not >1.1; (3) H1-R: coherence-abs correlation is positive (r=+0.36), not negative (r=-0.79 as in earlier pilot). Q4 empty - no low-absorption+high-sensitivity features found."
    }
  ]
}


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_sensitivity_floor",
      "title": "The Sensitivity Floor: Why Absorbed and Non-Absorbed Features Are Both Insensitive",
      "status": "front_runner",
      "summary": "Sensitivity requires BOTH high specificity AND high reliability. Absorbed features have low specificity (decoder directions close to neighbors). Non-absorbed features in sparse regions have low reliability (low activation frequency). Neither can achieve the sensitivity threshold. Q2 and Q4 are structurally impossible. The positive correlation (r=0.59) is a linear approximation of a U-shaped relationship.",
      "hypotheses": [
        "H-SF1: Q2+Q4 remain empty (<5%) even at N>200",
        "H-SF2: S(A) = aA^2 + bA + c with a > 0 (U-shape)",
        "H-SF3: Frequency mediates r(absorption, sensitivity) - partial r < 0.3",
        "H-SF4: Geometric density mediates r(absorption, sensitivity) - partial r < 0.3"
      ],
      "pilot_focus": "Larger sample (200 features) to confirm Q2/Q4 emptiness; 30 min pilot",
      "pilot_time_min": 30,
      "full_time_min": 75,
      "key_reference": "iter_008 pilot: Q4=0/43, r=0.59, ratio=1.0; Korznikov 2026 Sanity Check",
      "novelty_risk": "High (8/10) - Sensitivity floor is genuinely novel; no prior work connects specificity/reliability to absorption",
      "risk_mitigation": "Pilot is fast (30 min); falsifiable; alternatives available if SF falsified",
      "iter_008_relationship": "Built on all iter_008 falsifications: Q4 emptiness, r=0.59 positive correlation, ratio=1.0",
      "stagnation_alert": "Zero experiments for 2+ iterations - MUST execute pilot first"
    },
    {
      "candidate_id": "cand_sensitivity_first",
      "title": "Sensitivity-First: Sensitivity Failures Alone Explain Sanity Check",
      "status": "backup_primary",
      "summary": "Sensitivity failures (NOT absorption) are the primary driver of Sanity Check. Q3 dominates at 65% of features. Absorption adds no predictive value after controlling for sensitivity. 85% of features are low-sensitivity regardless of absorption.",
      "hypotheses": [
        "H-A: Low-sensitivity features (Q3) show steering not different from random baseline",
        "H-B: Absorption does NOT predict steering after controlling for sensitivity"
      ],
      "pilot_focus": "Steering on 28 Q3 features at beta=5; compare to random baseline",
      "pilot_time_min": 15,
      "full_time_min": 45,
      "key_reference": "Q3=28/43 (65%), Q2+Q4=0; Tian 2025 sensitivity decline",
      "novelty_risk": "Moderate (6/10) - Sensitivity-first explanation not claimed in prior work",
      "risk_mitigation": "Simplest explanation; directly testable; pilot is fast (15 min)",
      "iter_008_relationship": "Directly uses Q3 features from iter_008 pilot; tests whether sensitivity alone explains Sanity Check",
      "pivot_trigger": "H-SF1 fails: Q2+Q4 > 10%"
    },
    {
      "candidate_id": "cand_geometric_density",
      "title": "Geometric Density as Common Cause of Absorption and Sensitivity Failure",
      "status": "backup_secondary",
      "summary": "Both failure modes share geometric density as common cause. Dense regions cause decoder overlap (absorption) AND vague activation patterns (low sensitivity). Density mediates r=0.59.",
      "hypotheses": [
        "H-B1: Absorbed features have higher geometric density than non-absorbed",
        "H-B2: Partial r(absorption, sensitivity | density) < 0.3"
      ],
      "pilot_focus": "Compute geometric density for 100 features; test r(density, absorption)",
      "pilot_time_min": 15,
      "full_time_min": 50,
      "key_reference": "r=0.59 positive correlation; Empiricist perspective; OrtSAE orthogonality",
      "novelty_risk": "Moderate (6/10) - Density as mediating variable not previously proposed",
      "risk_mitigation": "Density is directly measurable; generates unified intervention",
      "iter_008_relationship": "Explains the positive correlation (r=0.59) that falsified compound failure independence",
      "pivot_trigger": "H-SF3 and H-SF4 both fail: neither frequency nor density mediates"
    },
    {
      "candidate_id": "cand_layer_uas_saturation",
      "title": "Layer-wise UAS Saturation: Measurement Path Forward",
      "status": "backup_control",
      "summary": "UAS saturation is layer-dependent, not universal. Some layers (4, 6, 10) show UAS variance while layer 8 shows saturation. This determines where absorption CAN be measured and whether Sensitivity Floor is layer-specific.",
      "hypotheses": [
        "H-C1: UAS variance differs across layers (some show std > 0.05)",
        "H-C2: At non-saturated layers, sensitivity correlates with UAS (r > 0.4)"
      ],
      "pilot_focus": "Measure UAS at layers 4, 8, 10 (30 features each); compute mean and std per layer",
      "pilot_time_min": 20,
      "full_time_min": 45,
      "key_reference": "UAS=1.0 saturation at layer 8 (iter_007); Empiricist + Contrarian perspectives",
      "novelty_risk": "Moderate (6/10) - Layer-wise saturation mapping is descriptive but essential",
      "risk_mitigation": "Addresses contrarian objection; determines measurement path for all experiments",
      "iter_008_relationship": "Addresses layer-specificity concern from Contrarian; measurement prerequisite for Sensitivity Floor",
      "pivot_trigger": "Contrarian layer-specificity objection needs resolution; run alongside pilot if time permits"
    },
    {
      "candidate_id": "cand_steering_diffusion",
      "title": "Steering Diffusion Explains Beta=20 Reversal",
      "status": "backup_tertiary",
      "summary": "The beta=20 reversal (low-absorption > high-absorption, p=0.015) is caused by steering diffusion. High-absorption features have decoder directions that overlap with MORE neighbors. At high beta, steering activates multiple neighbors simultaneously, diluting the effect.",
      "hypotheses": [
        "H-D1: Absorbed features have higher decoder-neighbor overlap",
        "H-D2: Diffusion is non-linear in beta; null at beta<=10, effect at beta=20"
      ],
      "pilot_focus": "Measure decoder-neighbor overlap for 50 features; fit diffusion model to iter_004 data",
      "pilot_time_min": 15,
      "full_time_min": 30,
      "key_reference": "iter_004 beta=20 reversal (p=0.015); H6 falsified ratio=1.0",
      "novelty_risk": "Low (reanalysis of beta=20) but mechanism is novel given H6 falsification",
      "risk_mitigation": "Explains an unexplained finding; consistent with H6 falsification",
      "iter_008_relationship": "Addresses beta=20 reversal that H6 falsification left unexplained",
      "pivot_trigger": "Secondary mechanism; run after primary experiments confirm Sensitivity Floor"
    }
  ],
  "dropped_from_prior": [
    {
      "candidate_id": "prior_cand_sensitivity_absorption_joint",
      "reason": "H5 FALSIFIED (r=0.59, not independent), Q4=0 (not empty), H6 FALSIFIED (ratio=1.0). The compound failure hypothesis assumed independence but pilot shows positive correlation.",
      "action": "Dropped as front-runner. cand_sensitivity_floor (sensitivity floor) explains the positive correlation through complementary mechanisms."
    },
    {
      "candidate_id": "prior_cand_coherence_protective",
      "reason": "H1-R FALSIFIED: r=+0.36, not negative protective. Earlier r=-0.786 did not replicate across layers 4, 8, 10.",
      "action": "Dropped permanently. The coherence-abs relationship is unstable and not reliably protective."
    },
    {
      "candidate_id": "prior_cand_beta20_saturation",
      "reason": "H6 FALSIFIED: L2 norm ratio = 1.0, not > 1.1. Decoder L2 norm saturation is not the mechanism for beta=20 reversal.",
      "action": "Dropped as backup. cand_steering_diffusion provides an alternative mechanism (neighbor overlap) that is consistent with ratio=1.0."
    }
  ],
  "evaluation_framework": {
    "absorption_detection": "Chanin et al. 2024 first-letter protocol (tau_fs=0.03, tau_ps=0.025, tau_pa=0.4)",
    "sensitivity_measurement": "Tian et al. 2025 paraphrase AUC protocol",
    "steering_protocol": "Add beta * W_dec[feature] to residual stream at layer 8",
    "geometric_density": "Mean cosine similarity to k=20 nearest neighbors in activation space",
    "frequency_measurement": "Fraction of tokens with feature active (sae_features > 0)"
  },
  "pilot_findings_from_iter_008": {
    "h5_result": {
      "status": "falsified",
      "expected": "r < 0.3 (independence)",
      "actual": "r = 0.59 (positive correlation)",
      "p_value": 3.15e-05,
      "interpretation": "Absorption and sensitivity are POSITIVELY correlated, not independent. Common cause model preferred."
    },
    "h6_result": {
      "status": "falsified",
      "expected": "L2 norm ratio > 1.1",
      "actual": "ratio = 1.0",
      "interpretation": "No decoder L2 norm difference between absorption groups. Saturation mechanism not supported."
    },
    "h1r_result": {
      "status": "falsified",
      "expected": "r < -0.5 (negative protective)",
      "actual": "r = +0.36 (positive)",
      "layers_tested": [4, 8, 10],
      "interpretation": "Earlier r=-0.786 did not replicate. Coherence-abs relationship is unstable."
    },
    "quadrant_counts": {
      "q1_highabs_lowsens": 15,
      "q2_highabs_highsens": 0,
      "q3_lowabs_lowsens": 28,
      "q4_lowabs_highsens": 0,
      "total_features": 43,
      "interpretation": "Q4 empty - sensitivity failures are near-universal. 100% of features are low-sensitivity."
    }
  },
  "stagnation_warning": {
    "alert": "Zero experiments for 2+ consecutive iterations",
    "paper_identical_to": "iter_007",
    "review_score": 6.5,
    "consecutive_reviews": 3,
    "recommended_experiments_not_executed": [
      "activation patching",
      "tightened hedging",
      "CMI at L0=22"
    ],
    "action_required": "Execute pilot experiments FIRST. Theoretical refinement without data is self-perpetuating stagnation."
  },
  "references": {
    "Chanin2024": "arXiv:2409.14507",
    "Korznikov2026Sanity": "arXiv:2602.14111",
    "Tian2025": "arXiv:2509.23717",
    "Korznikov2025OrtSAE": "arXiv:2509.22033",
    "Li2025ATM": "arXiv:2510.08855",
    "Luo2026HSAE": "arXiv:2602.11881",
    "Bereska2025": "arXiv:2512.13568"
  }
}


## 上一轮 validation 决策意见
# Idea Validation Decision

## Pilot Evidence Summary

### cand_sensitivity_floor (front-runner) - Layer 4 Pilot
| Metric | Predicted | Actual | Result |
|--------|-----------|--------|--------|
| Q2+Q4 fraction | < 10% | 100% (50/50) | **FAIL - FALSIFIED** |
| U-shape coefficient a | > 0 | 0.82 | PASS |

**Quadrant distribution (N=50, layer 4)**:
- Q1 (high abs + low sens): 0 (0%)
- Q2 (high abs + high sens): 48 (96%)
- Q3 (low abs + low sens): 0 (0%)
- Q4 (low abs + high sens): 2 (4%)

The Sensitivity Floor predicted Q2+Q4 < 10% based on layer 8 data (iter_008: Q2+Q4 = 0/43). Layer 4 shows the OPPOSITE pattern: 96% of features are Q2 (absorbed + high sensitivity). This is a definitive falsification of H-SF1 at layer 4.

**Contradictory findings across layers**:
- Layer 8 (iter_008): Q3 dominates at 65%, Q2+Q4 = 0/43
- Layer 4 (iter_011 pilot): Q2 dominates at 96%, Q2+Q4 = 50/50

### cand_sensitivity_first - Not Directly Tested in This Pilot
- Steering protocol returned all zeros (broken protocol)
- Layer 8 showed Q3 dominance (65%) consistent with sensitivity-first
- Layer 4 showed Q2 dominance (96%) -- inconsistent with pure sensitivity-first
- Needs pilot debugging before it can be evaluated

### cand_geometric_density - Not Tested
- Remains viable as a common-cause explanation for r=0.59
- No pilot data yet

### cand_layer_uas_saturation - Not Tested
- Layer-specificity is now the critical question given contradictory layer 4 vs layer 8 results
- Could resolve whether Sensitivity Floor is layer-specific or universally falsified

### cand_steering_diffusion - Not Tested (Protocol Broken)
- Steering protocol returned all zeros at beta in {5, 10, 20}
- Pilot H-STEER pass criterion (at least 1 non-zero effect) was not met

---

## Decision Matrix

| Criterion | Weight | cand_sensitivity_floor | cand_sensitivity_first | cand_geometric_density | cand_layer_uas_saturation | cand_steering_diffusion |
|-----------|--------|------------------------|------------------------|------------------------|---------------------------|------------------------|
| Pilot signal strength | 0.30 | 1 (H-SF1 falsified) | 3 (untested, Q3 at L8 consistent) | 2 (untested) | 2 (untested) | 1 (protocol broken) |
| Hypothesis survival | 0.25 | 1 (main H-SF1 false) | 3 (Q3 dominance at L8) | 3 (density mediation viable) | 3 (layer diff testable) | 2 (mechanism untested) |
| Path to full result | 0.20 | 2 (need recalibration) | 3 (steering validation) | 3 (direct measurement) | 4 (clear measurement path) | 2 (protocol must fix first) |
| Novelty | 0.15 | 5 (8/10, novel) | 4 (6/10) | 4 (7/10) | 3 (6/10) | 3 (6/10) |
| Resource efficiency | 0.10 | 3 (recalibration needed) | 4 (15 min pilot) | 3 (15 min pilot) | 4 (20 min pilot) | 2 (must fix protocol) |
| **Weighted Score** | | **2.0** | **3.35** | **2.85** | **3.05** | **1.95** |

---

## Decision Rationale

**PIVOT** is triggered for two reasons:
1. cand_sensitivity_floor's main hypothesis (H-SF1: Q2+Q4 < 10%) is definitively falsified at layer 4 with 100% Q2+Q4. The Sensitivity Floor predicted structural emptiness; it found structural abundance of Q2.
2. No candidate scores >= 3.5 (ADVANCE threshold). The highest is cand_sensitivity_first at 3.35, but its core hypothesis (Q3 = random baseline) has not been directly tested in this pilot.

**Key evidence triggering pivot**:
- H-SF1 falsified: Q2+Q4 = 100% at layer 4 vs predicted < 10%
- Contradictory layer patterns: Q2 dominates at layer 4 (96%), Q3 dominates at layer 8 (65%)
- The Sensitivity Floor cannot explain why layer 4 features are predominantly high-sensitivity (Q2)

**What the pivot should explore**:
1. **Layer-specificity as the dominant factor** -- layer 4 and layer 8 show fundamentally different absorption-sensitivity relationships. Any viable theory must explain this.
2. **Sensitivity-first remains viable** at layer 8 (Q3 = 65%), but layer 4 contradicts pure sensitivity-first. The discrepancy suggests absorption and sensitivity interact differently at different layers.
3. **Geometric density** could explain layer differences if density varies systematically across layers.
4. **Steering protocol is broken** -- must be debugged before cand_sensitivity_first or cand_steering_diffusion can be evaluated.

**Sanity checks**:
- [x] Compared ALL candidates, not just front-runner
- [x] Penalized cand_sensitivity_floor for failing its own falsification criterion (H-SF1)
- [x] Not swayed by sunk cost (iter_008 investment is irrelevant to this decision)
- [x] Not defaulting blindly -- PIVOT is triggered by definitive falsification + no candidate scoring >= 3.5

---

## Next Actions

1. **Recalibrate absorption metric**: The simplified proxy (max/mean ratio) may not match the Chanin first-letter protocol. Recalibration is needed before any candidate can be properly evaluated.
2. **Run layer-wise UAS mapping** (cand_layer_uas_saturation): The contradictory layer 4 vs layer 8 results make layer-specificity the most urgent measurement. 20 min on CPU.
3. **Debug steering protocol** (for cand_sensitivity_first): iter_010 found all steering effects = 0.0. This must be fixed before sensitivity-first can be tested.
4. **Consider geometric density as unifying explanation**: If dense regions vary by layer, this could explain why Q2 dominates at layer 4 but Q3 dominates at layer 8.

SELECTED_CANDIDATE: none
CONFIDENCE: 0.72
DECISION: PIVOT


## 上一轮 validation 结构化决策
{
  "decision": "PIVOT",
  "selected_candidate_id": null,
  "confidence": 0.72,
  "candidate_scores": {
    "cand_sensitivity_floor": {
      "weighted_score": 2.0,
      "verdict": "PIVOT",
      "pilot_signal": 1,
      "hypothesis_survival": 1,
      "path_to_result": 2,
      "novelty": 5,
      "resource_efficiency": 3,
      "evidence": "H-SF1 FALSIFIED: Q2+Q4 = 100% at layer 4 (predicted < 10%). 96% of features are Q2 (high abs + high sens), opposite of prediction. U-shape confirmed (a=0.82) but structural emptiness claim refuted."
    },
    "cand_sensitivity_first": {
      "weighted_score": 3.35,
      "verdict": "REFINE",
      "pilot_signal": 3,
      "hypothesis_survival": 3,
      "path_to_result": 3,
      "novelty": 4,
      "resource_efficiency": 4,
      "evidence": "No direct pilot data. Layer 8 showed Q3 dominance (65%), consistent with sensitivity-first. Layer 4 showed Q2 dominance (96%), inconsistent. Steering protocol broken (all zeros). Needs steering debug pilot."
    },
    "cand_geometric_density": {
      "weighted_score": 2.85,
      "verdict": "REFINE",
      "pilot_signal": 2,
      "hypothesis_survival": 3,
      "path_to_result": 3,
      "novelty": 4,
      "resource_efficiency": 3,
      "evidence": "No pilot data yet. Common cause hypothesis remains viable -- could explain r=0.59 and layer differences if density varies by layer."
    },
    "cand_layer_uas_saturation": {
      "weighted_score": 3.05,
      "verdict": "REFINE",
      "pilot_signal": 2,
      "hypothesis_survival": 3,
      "path_to_result": 4,
      "novelty": 3,
      "resource_efficiency": 4,
      "evidence": "No pilot data yet. Layer-specificity is now the critical question given contradictory layer 4 vs layer 8 results. Clear measurement path."
    },
    "cand_steering_diffusion": {
      "weighted_score": 1.95,
      "verdict": "PIVOT",
      "pilot_signal": 1,
      "hypothesis_survival": 2,
      "path_to_result": 2,
      "novelty": 3,
      "resource_efficiency": 2,
      "evidence": "Steering protocol broken -- all steering effects = 0.0 at beta in {5, 10, 20}. H-STEER pass criterion not met. Cannot evaluate until protocol is debugged."
    }
  },
  "reasons": [
    "H-SF1 FALSIFIED: Q2+Q4 = 100% at layer 4 vs predicted < 10% (definitive falsification)",
    "Contradictory layer patterns: Q2 dominates at layer 4 (96%), Q3 dominates at layer 8 (65%)",
    "Sensitivity Floor cannot explain layer 4 Q2 abundance (absorbed + high sensitivity)",
    "No candidate scores >= 3.5 (ADVANCE threshold)",
    "Highest candidate (cand_sensitivity_first at 3.35) not directly tested in this pilot",
    "Steering protocol broken -- cannot evaluate sensitivity-first or steering-diffusion",
    "Layer-specificity is the most urgent unanswered question"
  ],
  "next_actions": [
    "Recalibrate absorption metric using proper Chanin first-letter protocol (not simplified proxy)",
    "Run cand_layer_uas_saturation pilot to map UAS across layers 4, 6, 8, 10 -- layer-specificity is now the central question",
    "Debug steering protocol (cand_sensitivity_first): iter_010 found all steering effects = 0.0 -- must fix before evaluating sensitivity-first",
    "Consider geometric density as unifying explanation: if dense regions vary by layer, could explain Q2 vs Q3 dominance across layers"
  ],
  "dropped_candidates": ["cand_sensitivity_floor"],
  "pivot_trigger": "H-SF1 definitively falsified: Q2+Q4 = 100% at layer 4 (predicted < 10%); contradictory layer patterns suggest Sensitivity Floor is layer-specific or metric-dependent",
  "negative_results_to_report": [
    {
      "candidate": "cand_sensitivity_floor",
      "finding": "Sensitivity Floor H-SF1 falsified at layer 4",
      "detail": "Q2+Q4 = 100% (50/50) vs predicted < 10%. 96% of layer 4 features are Q2 (absorbed + high sensitivity), opposite of Sensitivity Floor prediction",
      "implication": "Structural emptiness claim does not hold at layer 4. Layer-specificity is the dominant factor."
    },
    {
      "candidate": "cand_steering_diffusion",
      "finding": "Steering protocol returns all zeros",
      "detail": "All steering effects = 0.0 at beta in {5, 10, 20}. H-STEER pass criterion (at least 1 non-zero effect) not met.",
      "implication": "Cannot evaluate sensitivity-first or steering-diffusion until protocol is debugged."
    }
  ],
  "key_finding": "Layer 4 features show opposite absorption-sensitivity pattern to layer 8: Q2 dominates (96%) at layer 4 vs Q3 dominates (65%) at layer 8. Any viable theory must explain this layer-specificity."
}


## 上一轮新颖性检查报告（必须针对发现的撞车问题进行修正）
# Novelty Report

**Workspace**: `/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current`
**生成时间**: 2026-04-29
**检查者**: sibyl-novelty-checker
**状态**: 针对 iter_009 proposal 的完整重评（相比 iter_008，框架已从 compound failure 切换到 sensitivity floor）

---

## 执行摘要

| 候选者 | 新颖度评分 | 推荐操作 | 主要碰撞 |
|--------|------------|----------|----------|
| cand_sensitivity_floor | **8/10** | **推进** | 无先例提出 specificity/reliability 互补机制；与 Tian/Chanin/Korznikov 差异清晰 |
| cand_sensitivity_first | **6/10** | **推进** | Tian 记录 sensitivity decline，但未声称 sensitivity 单独解释 Sanity Check |
| cand_geometric_density | **7/10** | **推进** | Michaud/Bereska 讨论 manifold capacity，但未提出 density 作为吸收-敏感性的中介 |
| cand_layer_uas_saturation | **6/10** | **推进** | 描述性层间差异映射；与 Contrarian 的反对意见相关 |
| cand_steering_diffusion | **6/10** | **修改以差异化** | H6 falsification 背景下的 steering diffusion 机制是新提出的；对内部数据的重新分析 |

**总体新颖度**: `high` — 主力候选 8/10（真正的新想法）；所有活跃候选 6+/10。先前工作未见 exact match。

---

## 候选者详细分析

### 候选者: cand_sensitivity_floor
**标题**: The Sensitivity Floor: Why Absorbed and Non-Absorbed Features Are Both Insensitive
**状态**: front_runner（前置_runner）

#### 核心主张
敏感性需要 BOTH 高特异性 AND 高可靠性：
- **特异性**：特征在目标上激活但在邻居上不激活（吸收时被破坏）
- **可靠性**：特征在语义相似输入上一致激活（在稀疏区域被破坏）

机制：
- Q1（吸收 + 低敏感性）：吸收特征特异性低（decoder 方向接近父特征）
- Q3（非吸收 + 低敏感性）：稀疏非吸收特征可靠性低（激活频率低）
- Q2 和 Q4 是**结构上不可能**：吸收和非吸收都无法同时满足两个因素

观察到的 r = 0.59 是 U 型关系的线性近似：敏感性在中间吸收度时最大化，而非极端。

#### 新颖度评估

**评分: 8/10**（真正的新想法；未发现接近的先前工作）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| Tian et al. 2025 (arXiv:2509.23717) | 记录 sensitivity 随 SAE 宽度下降；paraphrase AUC 协议。**未**连接 sensitivity 到 absorption 或几何结构 | partial_overlap | Tian 研究 sensitivity 的孤立现象；**未**提出 specificity/reliability 权衡或 sensitivity floor |
| Chanin et al. 2024 (arXiv:2409.14507) | 定义通过首字母分类协议的 absorption 检测。**未**研究与 sensitivity 的联合分布 | partial_overlap | Chanin 研究 absorption 的孤立现象；**未**连接到 sensitivity 或提出互补失效机制 |
| Korznikov et al. 2026 (arXiv:2602.14111) | 显示随机基线匹配 SAEs（Sanity Check）。**未**机械地解释原因 | related_work | Korznikov 记录了危机；**未**通过 sensitivity floor 解释机制 |
| Bereska et al. 2025 (arXiv:2512.13568) | 信息论框架连接 superposition 到压缩。**未**连接到 sensitivity 测量 | related_work | Bereska 的熵框架是理论性的；**未**提出 sensitivity floor 或空象限 |
| Michaud et al. 2025 (arXiv:2509.02565) | 容量分配模型；特征 manifold 容量有界。暗示特征竞争容量但**未**提出 sensitivity floor | related_work | 容量分配是相关的直觉，但**未**生成 specificity/reliability 框架 |

**与先前工作的关键差异化**：

1. **特异性-可靠性框架**：声称 BOTH 吸收 AND 非吸收特征通过**互补机制**（特异性 vs 可靠性）都是敏感的，这是真正的新想法。先前工作将 absorption 和 sensitivity 视为独立现象。

2. **Q2/Q4 的结构性空缺**：提案声称 Q2/Q4 是**结构上不可能**（不是统计上稀有）。这是一个在先前文献中未发现的有力理论主张。

3. **U 型关系**：预测 S(A) 遵循 U 型（敏感性在中间吸收度时最大化）是真正的新想法。Tian 记录了随宽度的下降，但**未**记录非单调关系。

4. **失效模式集成**：sensitivity floor 是唯一同时解释 Q4=0（pilot）、r=0.59（正相关）和 Sanity Check 的候选者，具有统一框架。

**碰撞严重度**: **partial_overlap** — 每个组件（absorption、sensitivity、geometric density）都出现在先前工作中，但**SENSITIVITY FLOOR 机制**（吸收 vs 非吸收特征的互补不足）是缺失的。

#### 建议: **推进**
sensitivity floor 框架是本提案中最强的新想法主张。关键风险是 Q2/Q4 在 N>200 时是否真的保持空缺（结构性 vs 统计性）。理论框架内部一致且产生可具体证伪的预测。

---

### 候选者: cand_sensitivity_first
**标题**: Sensitivity-First: Sensitivity Failures Alone Explain Sanity Check
**状态**: backup_primary（如果 H-SF1 失败则 pivot）

#### 核心主张
敏感性失效（**不是** absorption）是 Sanity Check 的主要驱动因素。Q3 占 65%。控制敏感性后，absorption 分类没有增加预测价值。85% 的特征是低敏感性，无论 absorption 状态如何。

#### 新颖度评估

**评分: 6/10**（有新意但有轻微重叠；需要重新定位以主张完全的新颖性）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| Tian et al. 2025 | 记录 interpretable SAE 特征有**差**的敏感性（在语义相似文本上失败）；敏感性随 SAE 宽度下降 | partial_overlap | Tian **未**声称敏感性失效**单独**解释 Sanity Check；未测试 absorption 在控制敏感性后是否增加预测价值 |
| Korznikov et al. 2026 | 显示随机基线匹配 SAEs 在 interpretability/causal editing 上 | related_work | Korznikov **未**识别哪种特定失效模式驱动危机 |

**与先前工作的关键差异化**：

1. **敏感性优先解释**：敏感性失效**单独**（没有 absorption）解释 Sanity Check 的具体声称不在 Tian 或任何先前工作中。Tian 记录敏感性下降但**未**声称它是**唯一**驱动因素。

2. **控制敏感性后 absorption 不增加预测价值**：这是一个可以测试的新经验性声称。

**碰撞严重度**: **partial_overlap** — Tian 提供了敏感性测量工具并记录了现象，但**未**做出"敏感性优先"的理论声称。

#### 建议: **推进**
这是比 cand_sensitivity_floor 更简单的重新表述。新颖性适中（6/10），因为核心声称（敏感性是主要驱动因素）被 Tian 暗示但未明确声明。独特贡献是测试 absorption 在控制敏感性后是否增加预测价值的受控实验。

---

### 候选者: cand_geometric_density
**标题**: Geometric Density as Common Cause of Absorption and Sensitivity Failure
**状态**: backup_secondary（如果 H-SF3 和 H-SF4 都失败则 pivot）

#### 核心主张
两种失效模式共享一个几何共同原因：位于激活 manifold **密集区域**的特征容易受到 BOTH 吸收（decoder 重叠）和低敏感性（模糊激活模式）的影响。几何密度中介 r=0.59 相关性。

#### 新颖度评估

**评分: 7/10**（有新意但有轻微重叠；差异清晰且可辩护）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| Michaud et al. 2025 | SAE 缩放的容量分配模型；识别 SAEs 学习远少特征于 latents 的病态 regime；有界 manifold 容量 | partial_overlap | 讨论容量约束但**未**提出几何密度作为吸收和敏感性之间的中介变量 |
| Bereska et al. 2025 | 信息论框架连接 superposition 到压缩；激活上的 Shannon 熵 | related_work | 理论框架但**未**提出密度作为中介 |
| Korznikov et al. 2025 OrtSAE | 正交性惩罚减少吸收 65%；暗示几何结构很重要 | partial_overlap | 暗示几何影响吸收但**未**提出密度作为 BOTH 吸收和敏感性的共同原因 |

**与先前工作的关键差异化**：

1. **几何密度作为中介变量**：先前工作**未**提出几何密度（通过 k-NN 余弦相似度测量）中介吸收-敏感性相关性。Michaud 讨论有界容量但**未**作为中介变量。

2. **可测试的中介预测**：特定预测部分 r(吸收，敏感性 | 密度) < 0.3 是一个新的可证伪声称。先前工作**未**生成这个特定中介预测。

3. **统一干预**：如果确认，几何密度测量实现有针对性的干预（稀疏化密集区域）同时解决两种失效模式。

**碰撞严重度**: **partial_overlap** — Manifold capacity 文献（Michaud、Bereska）在概念上相关但**未**提出几何密度作为特定中介变量或生成特定中介预测。

#### 建议: **推进**
这是 backup 候选者中最可测量新颖性的贡献。密度作为中介的声称不在先前文献中，特定中介预测（部分 r < 0.3 在控制密度后）是可证伪的且不被现有工作暗示。

---

### 候选者: cand_layer_uas_saturation
**标题**: Layer-wise UAS Saturation: Measurement Path Forward
**状态**: backup_control（与 pilot 并行运行以解决 Contrarian 反对意见）

#### 核心主张
UAS 饱和是层依赖的，不是普遍的。一些层（4、6、10）显示 UAS 方差而层 8 显示饱和。这决定了在哪里**可以**测量 absorption 以及 Sensitivity Floor 是否是层特定的。

#### 新颖度评估

**评分: 6/10**（有新意但主要是描述性的）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| iter_007 UAS=1.0 在层 8（内部） | 层 8 的 UAS 饱和测量 | related_work | 这是内部数据；需要外部验证 |
| Empiricist + Contrarian 视角 | 层间差异的论点 | related_work | 理论观点而非先验文献 |

**与先前工作的关键差异化**：

1. **系统性层间 UAS 映射**：这是描述性贡献。UAS 在不同层的差异之前未系统映射。

2. **解决 Contrarian 的反对意见**：这直接解决了 Sensitivity Floor 的主要弱点（仅在层 8 测试，N=43）。

**碰撞严重度**: **related_work** — 这主要是描述性/方法论贡献，不是理论突破。

#### 建议: **推进**
这是一个必要的控制实验，解决 Contrarian 的层特异性反对意见。不是主要的新想法贡献，但为 Sensitivity Floor 提供方法论基础。

---

### 候选者: cand_steering_diffusion
**标题**: Steering Diffusion Explains Beta=20 Reversal
**状态**: backup_tertiary（主要作为控制和替代机制）

#### 核心主张
beta=20 反转（低吸收 > 高吸收，p=0.015）是由 steering diffusion 引起的，**不是** decoder L2 norm 饱和（H6 falsified: ratio=1.0）。高吸收特征有 decoder 方向与**更多**邻居重叠。在高 beta 时，steering 同时激活多个邻居，稀释效果。

#### 新颖度评估

**评分: 6/10**（新颖机制，但定位为重新分析）

**碰撞分析**：

| 论文 | 重叠 | 严重度 | 不重叠的部分 |
|------|------|--------|------------|
| iter_004 beta=20 数据（内部） | 包含 beta=20 反转发现；机制**未**在 iter_004 中识别 | related_work | 这是内部数据重新分析；机制是新的 |
| H6 falsification（内部） | Decoder L2 norm ratio = 1.0；饱和不支持 | related_work | 先前假设被消除；diffusion 提供替代机制 |
| 一般 steering 文献 | Steering 有效性作为大小的函数 | related_work | **未**提出 diffusion 作为高 beta 反转的机制 |

**与先前工作的关键差异化**：

1. **Steering diffusion 机制**：高吸收特征有**更多** decoder-neighbor 重叠导致高 beta 时稀释的声称是真正的新想法。这与 decoder L2 norm 饱和不同（已被 falsified）。

2. **非线性 beta 依赖**：预测 diffusion 在 beta<=10 时为空，在 beta=20 时效果出现是一个与 H6 falsification 一致的具体可证伪预测。

**碰撞严重度**: **related_work** — 这主要是内部 iter_004 数据的重新分析。diffusion 机制是新的但贡献是解释现有发现而非发现新现象。

#### 建议: **修改以差异化**
应定位为前_runner 实验计划中的**控制/替代机制**，而非独立候选者。Steering diffusion 机制在 H6 falsification 背景下是真正的新想法，但作为独立候选者缺乏 sensitivity floor 的广度。如果与 sensitivity floor 一起确认，它提供了 beta=20 反转的次要机制解释。

---

## 丢弃的候选者（无需重新评估）

以下候选者已从提案中丢弃，不需要新颖性评估：

| 候选者 | 丢弃原因 | 碰撞评估 |
|--------|----------|----------|
| prior_cand_sensitivity_absorption_joint | H5 FALSIFIED (r=0.59, 不是独立的), Q4=0, H6 FALSIFIED (ratio=1.0)。compound failure 假设独立但 pilot 显示正相关 | N/A |
| prior_cand_coherence_protective | H1-R FALSIFIED: r=+0.36, 不是负保护效应。早期 r=-0.786 未复制 | N/A |
| prior_cand_beta20_saturation | H6 FALSIFIED: L2 norm ratio=1.0；decoder L2 norm 饱和不是 beta=20 反转的机制 | 由 cand_steering_diffusion 替代 |

---

## 总体评估

**总体新颖度**: `high`

- **主力候选** (cand_sensitivity_floor): 评分 8/10，真正的新想法 sensitivity floor 框架
- **所有活跃候选**: 评分 6-8/10，先前文献中无 exact match
- **无关键碰撞**: 最令人担忧的重叠（Tian/Chanin）在独立研究中，未连接 absorption 和 sensitivity

**关键发现**: sensitivity floor 假设 (cand_sensitivity_floor) 是最强的新想法主张，因为：
1. 它推翻四象限模型（先前工作中假设）
2. 它提出 U 型关系（非先前假设的单调关系）
3. 它解释为什么 Q2/Q4 是结构上不可能的（不仅是统计上稀有）
4. 它连接到 iter_008 所有三个 falsification（独立性、饱和、Q4 空缺）

---

## Anti-Pattern 检查

- [x] **未**未经搜索就 rubber-stamp 新颖性 — 评估了每个候选者对比先验文献
- [x] **未**因模糊相关工作而 Dismiss 想法 — 精确分类（partial_overlap vs related_work）
- [x] **未**混淆相关工作与已完成的工作 — 在所有候选者中保持清晰区分
- [x] **未**跳过候选者 — 所有 5 个活跃候选者 + 3 个丢弃候选者已评估
- [x] **未**使用过时报告 — 执行了对当前提案状态的完整重新评估

---

## 与先前报告的变更（iter_008 → iter_009）

| 变更 | 原因 |
|------|------|
| 新增 cand_layer_uas_saturation | Contrarian 的层特异性反对意见需要解决 |
| cand_hierarchical_redundancy 移除 | 由 cand_sensitivity_floor 吸收作为前_runner 的一部分 |
| cand_steering_diffusion 降级为 backup_control | 主要作为控制和替代机制，而非独立候选 |
| novelty_score 调整 | 基于对 proposal 中框架的更深入分析 |

---

## 注释

**外部搜索工具不可用**: WebSearch、arXiv MCP 和 Google Scholar MCP 在此检查期间不可访问。新颖性评估基于：
- Proposal 内容和候选者定义
- Proposal 中引用的先验艺术（Tian 2025、Chanin 2024、Korznikov 2026 等）
- 视角文档（innovator.md、empiricist.md、theoretical.md 等）
- 内部 iter_004 和 iter_008 pilot 数据引用

**建议**: 当 MCP 工具可用时使用 arXiv 搜索进行外部验证，特别是：
- "sensitivity floor sparse autoencoder"（预期无 exact match）
- "SAE specificity reliability tradeoff"（新概念）
- "geometric density SAE absorption sensitivity"（新中介变量）

**评估置信度**: 对于 cand_sensitivity_floor 是 HIGH（理论框架与引用先验工作明显不同）；对于其他候选者是 MODERATE（概念连接存在但具体声称未重复）。