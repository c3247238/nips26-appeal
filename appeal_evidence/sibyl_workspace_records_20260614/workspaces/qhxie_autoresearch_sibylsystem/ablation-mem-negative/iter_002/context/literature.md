# Literature Survey Report

**Research Topic**: SAE Absorption, Dead Features, Feature Resuscitation, and Training Stability in Sparse Autoencoders
**Survey Date**: 2026-04-28
**arXiv Search Keywords**: ["feature absorption" "sparse autoencoder", "feature splitting" "sparse autoencoder", "SAE benchmark" "sparse autoencoder evaluation", "JumpReLU" "Gated SAE" "TopK SAE", "Matryoshka SAE" "hierarchical sparse autoencoder", "provable feature recovery" "sparse autoencoder", "dead features" "sparse autoencoder", "feature resuscitation" "sparse autoencoder", "training stability" "sparse autoencoder"]
**Web Search Keywords**: ["sparse autoencoder feature absorption SAE mechanistic interpretability 2024 2025", "SAEBench sparse autoencoder benchmark evaluation", "SAELens sparse autoencoder library GemmaScope pretrained SAEs", "Matryoshka SAE hierarchical absorption solution 2025", "transcoder vs sparse autoencoder interpretability 2025", "provable feature recovery sparse autoencoder theoretical guarantee 2025", "ghost gradients dead neurons sparse autoencoder Anthropic 2024", "TopK SAE JumpReLU training stability 2025", "adaptive temporal masking stable SAE training", "fundamental limits neural network sparsification dead neurons"]

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant technique for mechanistic interpretability of Large Language Models (LLMs), enabling researchers to decompose dense model activations into sparse, human-interpretable features. The theoretical foundation was laid by Elhage et al.'s "Toy Models of Superposition" (2022), which established that neural networks represent more features than they have dimensions by encoding features as approximately orthogonal directions in activation space.

The seminal application of SAEs to language models came with Bricken et al.'s "Towards Monosemanticity" (2023) from Anthropic's Transformer Circuits Thread, which demonstrated that dictionary learning via sparse autoencoders could decompose MLP activations into monosemantic features. Cunningham et al. (2023) showed SAEs outperform PCA/ICA on automated interpretability scores. This was followed by Templeton et al.'s scaling work (2024) on Claude 3 Sonnet, extracting millions of interpretable features.

However, critical limitations were identified in 2024-2025 across three interconnected dimensions:

1. **Feature Absorption** --- A phenomenon where broad, interpretable features get "absorbed" into more specific, token-aligned latents, creating interpretability illusions with arbitrary false negatives. Discovered by Chanin et al. (2024) in "A is for Absorption", this has sparked vigorous research on architectural solutions.

2. **Dead Features/Neurons** --- Latent dimensions that permanently stop activating during training, wasting capacity and reducing effective dictionary size. Anthropic's ghost gradients (2024) and OpenAI's AuxK loss (2024) were early responses, but fundamental limits exist.

3. **Training Instability** --- Sensitivity to initialization, learning rates, and hyperparameters causing inconsistent feature discovery across runs. The field has moved from L1-sparsity to TopK and JumpReLU architectures to decouple sparsity control from feature learning.

By 2025, the field has expanded to include architectural solutions (Matryoshka SAEs, Orthogonal SAEs, ATM), theoretical analyses (provable feature recovery guarantees, closed-form optimal solutions), comprehensive benchmarks (SAEBench, CE-Bench), and critical reassessments (SAEs interpreting randomly initialized transformers). The tension between reconstruction fidelity, sparsity, feature quality, and training stability remains the central challenge.

## 2. Core References

| # | Title | Authors | Source | Year | Key Contribution | Limitations |
|---|-------|---------|--------|------|-----------------|-------------|
| 1 | Toy Models of Superposition | Elhage et al. | Transformer Circuits Thread / arXiv:2209.10652 | 2022 | Established theoretical foundation of superposition; showed networks learn to represent sparse features in superposition | Toy models only; not applied to real LLMs |
| 2 | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | Bricken et al. | Transformer Circuits Thread | 2023 | First application of SAEs to real transformers; demonstrated monosemantic feature recovery; introduced feature splitting | Single-layer model only; limited scale |
| 3 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | Cunningham et al. | arXiv:2309.08600 | 2023 | Showed SAEs outperform PCA/ICA on automated interpretability scores | Did not identify absorption or dead features as failure modes |
| 4 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Templeton et al. | Anthropic / arXiv | 2024 | Scaled SAE methods to production LLM; millions of features; safety-relevant feature discovery | Did not systematically study absorption or dead features at scale |
| 5 | Scaling and evaluating sparse autoencoders | Gao et al. | arXiv:2406.04093 / ICLR 2025 | 2024 | Introduced k-sparse autoencoders (TopK), scaling laws, new quality metrics; trained 16M latent SAE; AuxK loss eliminates almost all dead latents | Does not address absorption specifically |
| 6 | Improving Dictionary Learning with Gated Sparse Autoencoders | Rajamanoharan et al. | arXiv:2404.16014 | 2024 | Gated SAE: separates gating from magnitude estimation; solves shrinkage; Pareto improvement | Absorption and dead features not studied |
| 7 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | Rajamanoharan et al. | arXiv:2407.14435 / ICLR 2025 | 2024 | JumpReLU activation: SOTA reconstruction fidelity at given sparsity; trains L0 directly; minimal dead features via threshold-only gradients | Later shown to have absorption issues |
| 8 | **A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders** | Chanin, Wilken-Smith, Dulka, Bhatnagar, Bloom | arXiv:2409.14507 / OpenReview | 2024 | **Foundational paper**: Identified and named feature absorption; linked cause to sparsity loss + hierarchical co-occurrence; showed varying size/sparsity insufficient | Metric relies on ablation (limited to early layers); conservative underestimate; focused on GPT-2 |
| 9 | Ghost Grads: An improvement on resampling | Jermyn, Templeton | Transformer Circuits Thread (Jan 2024) | 2024 | Introduced auxiliary loss using exponential activation on dead neurons; often achieves zero dead neurons | ~2x training cost; abandoned by Anthropic due to loss spikes; bug in original implementation |
| 10 | Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders | Makkuva et al. | arXiv:2411.13117 | 2024 | Theoretical analysis of SAE inference optimality; identified amortization gap | Not directly about absorption or dead features |
| 11 | BatchTopK Sparse Autoencoders | Bussmann et al. | arXiv:2412.06410 | 2024 | Batch-level top-k: adaptive latent allocation; outperforms TopK, comparable to JumpReLU | Absorption not evaluated |
| 12 | **SAEBench: A Comprehensive Benchmark for Sparse Autoencoders** | Karvonen et al. | arXiv:2503.09532 | 2025 | **Standardized evaluation**: 8+ metrics across interpretability, disentanglement, applications; 200+ SAEs evaluated; moved beyond sparsity-fidelity | Proxy metrics may not fully capture absorption |
| 13 | A Survey on Sparse Autoencoders: Interpreting the Internal Representations of LLMs | Various | arXiv:2503.05613 / EMNLP 2025 | 2025 | Comprehensive survey covering SAE architectures, training, evaluation, and applications | Survey paper; limited novel contributions |
| 14 | **Learning Multi-Level Features with Matryoshka Sparse Autoencoders** | Bussmann et al. | arXiv:2503.17547 / ICML 2025 | 2025 | Proposed hierarchical nested SAEs; achieved ~90% reduction in absorption (0.49 -> 0.05 at L0=40) | Introduces feature hedging in inner levels; higher computational cost |
| 15 | **Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders** | Chanin et al. | arXiv:2505.11756 | 2025 | Identified hedging as the complement to absorption; showed Matryoshka trades absorption for hedging; proposed balanced loss coefficients (beta_m ~ 0.75) | Limited to empirical analysis; no theoretical characterization of trade-off |
| 16 | **Orthogonal Sparse Autoencoders Uncover Atomic Features** | Korznikov et al. | arXiv:2509.22033 | 2025 | Alternative solution via orthogonality constraints (cosine similarity penalty); ~65% absorption reduction; ~50% less compute than Matryoshka | Slightly lower explained variance |
| 17 | **Provable Feature Recovery via Sparse Autoencoders** | Chen et al. | arXiv:2506.14002 / ICLR 2025 | 2025 | **First SAE algorithm with theoretical recovery guarantees**; introduced Group Bias Adaptation (GBA); validated on 2B parameter LLMs | Guarantees require specific statistical model assumptions |
| 18 | **On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy** | Cui et al. | arXiv:2506.15963 | 2025 | **First closed-form optimal solution analysis**; proved standard SAEs fail unless features are extremely sparse; identified feature shrinking/vanishing; proposed Weighted SAE (WSAE) | Theoretical analysis; limited empirical validation |
| 19 | **Fundamental Limits of Neural Network Sparsification** | Dip Roy et al. | arXiv:2603.18056 | 2025 | Showed dead neuron recovery is severely limited under extreme sparsification; intrinsic to compression process; zero recovery on dSprites after 100 epochs | Extreme sparsification regime; may not generalize to moderate sparsity |
| 20 | **Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training** | T. Ed Li et al. | arXiv:2510.08855 / ICLR 2025 | 2025 | ATM achieves absorption score 0.0068 vs TopK's 0.1402 via dynamic probabilistic masking with EMA tracking | Newer method; less community validation |
| 21 | **Stable and Steerable Sparse Autoencoders with Weight Regularization** | Oliver Crook et al. | arXiv:2603.04198 | 2026 | L2 regularization doubles steering success rates; increases cross-seed consistency; tied init + unit-norm constraints | Aggressively kills many features, creating bimodal structure |
| 22 | Transcoders Beat Sparse Autoencoders for Interpretability | Paulo, Shabalin, Belrose | arXiv:2501.18823 | 2025 | Proposed transcoders as superior alternative; skip transcoders Pareto-dominate SAEs; higher automated interpretability scores across Pythia, Llama, Gemma | Different objective (input-output vs. self-reconstruction); not directly comparable for all use cases |
| 23 | Sparse Autoencoders Can Interpret Randomly Initialized Transformers | Various | arXiv:2501.17727 | 2025 | Showed SAEs find similar features in trained and untrained models; raises questions about what SAEs actually capture | Challenges the assumption that SAE features reflect learned computations |
| 24 | Measuring Sparse Autoencoder Feature Sensitivity | Tian et al. | arXiv:2509.23717 | 2025 | Introduces feature sensitivity: reliability of feature activation on semantically similar texts | Complementary to absorption; not causal |
| 25 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | Muchane et al. | arXiv:2506.01197 | 2025 | HSAE: explicitly models semantic hierarchy; improves reconstruction and interpretability | Requires hierarchical structure assumption |
| 26 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical SAEs | Luo et al. | arXiv:2602.11881 | 2026 | HSAE jointly learns SAEs and parent-child relationships; recovers meaningful hierarchies | Very recent; limited validation |
| 27 | Data Whitening Improves Sparse Autoencoder Learning | Saraswatula & Klindt | arXiv:2511.13981 | 2025 | PCA whitening improves interpretability metrics on SAEBench; challenges sparsity-fidelity tradeoff | Minor reconstruction drops |
| 28 | Resurrecting the Salmon: Domain-Specific Sparse Autoencoders | O'Neill et al. | arXiv:2508.09363 | 2025 | Domain confinement mitigates fragmentation/absorption; 20% more variance explained | Limited to medical domain |
| 29 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | Song et al. | arXiv:2505.20254 | 2025 | Argues for feature consistency (PW-MCC metric); high consistency achievable with right architectures | Consistency != absence of absorption |
| 30 | Kronecker Factorization Improves Efficiency and Interpretability of SAEs | Various | arXiv:2505.22255 | 2025 | KronSAE: Kronecker factorization + mAND gating; reduces absorption via AND-like behavior | Architectural complexity |
| 31 | Distribution-Aware Feature Selection for SAEs | Various | arXiv:2508.21324 | 2025 | L2-norm/Squared-l scoring reduces absorption vs BatchTopK; no single config dominates all metrics | Trade-offs remain |
| 32 | CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark | Gulko et al. | arXiv:2509.00691 | 2025 | LLM-free contrastive evaluation using story pairs; 70%+ Spearman correlation with SAEBench | Limited metric coverage |
| 33 | Taming Polysemanticity in LLMs: Theory-Grounded Feature Recovery | Chen et al. | ICLR 2025 | 2025 | "Neuron resonance" phenomenon; neurons learn monosemantic features when activation frequency matches feature occurrence | Theoretical; limited scope |
| 34 | Train One Sparse Autoencoder Across Multiple Sparsity Levels | Various | EMNLP 2025 | 2025 | HierarchicalTopK enables post-hoc sparsity selection without retraining; BatchTopK mixes k values within batch | Limited sparsity range (k <= 128) |
| 35 | Interpreting CLIP with Hierarchical Sparse Autoencoders | Zaigrajew et al. | arXiv:2502.20578 / ICML 2025 | 2025 | Soft-capping reduces dead neurons (6 vs 491 at scale); multi-granularity TopK for vision-language | Vision-language specific; may not transfer to pure LLMs |
| 36 | Deep sparse autoencoders yield interpretable features too | Various | Alignment Forum | 2025 | Activation decay as alternative to dead neuron resampling for deep SAEs | Deep SAE specific |
| 37 | Sparse Autoencoders Do Not Find Canonical Units of Analysis | Leask et al. | ICLR 2025 / arXiv | 2025 | Challenged canonical feature hypothesis via SAE stitching and Meta-SAEs; introduced BatchTopK | Shows SAEs are incomplete/non-atomic, raising fundamental concerns |
| 38 | Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models | Various | ICLR 2025 | 2025 | Investigates feature universality across different LLMs via activation correlation and SVCCA | Focus on universality, not absorption |
| 39 | Binary Sparse Coding for Interpretability | Various | arXiv:2509.25596 | 2025 | Binarizes activations to prevent information smuggling through continuous activation strengths | Binary constraints may limit expressiveness |
| 40 | Self-Ablating Transformers: More Interpretability, Less Sparsity | Various | arXiv:2505.00509 | 2025 | Self-ablating mechanism for improved interpretability with reduced sparsity requirements | Alternative to SAEs entirely |

## 3. SOTA Methods and Benchmarks

### Current Best Methods for Addressing Absorption

| Method | Absorption Reduction | Key Innovation | Computational Overhead | Trade-off |
|--------|---------------------|----------------|----------------------|-----------|
| **Matryoshka SAE** | ~90% (0.49 -> 0.05 at L0=40) | Hierarchical nested dictionaries with independent reconstruction constraints | High (multiple dictionary sizes) | Introduces hedging in inner levels |
| **Balanced Matryoshka SAE** | Optimized trade-off | Tuned relative loss coefficients (beta_m ~ 0.75) | High | Balances absorption vs. hedging |
| **Orthogonal SAE (OrtSAE)** | ~65% | Cosine similarity penalty between latents | Low (~50% less than Matryoshka) | Slightly lower explained variance; 4-11% slower inference |
| **ATM (Adaptive Temporal Masking)** | ~95% (0.1402 -> 0.0068) | Temporal dynamics + probabilistic masking with dual EMAs | Medium | Very recent; limited validation |
| **Weighted SAE (WSAE)** | Improved low-sparsity recovery | Theoretically-grounded reweighting strategy | Low | Limited empirical validation |
| **Group Bias Adaptation (GBA)** | Strong empirical results | Adaptive bias adjustment for feature identifiability; theoretical guarantees | Medium | Requires specific statistical assumptions; tested up to 2B params |
| **KronSAE** | Lower absorption | Kronecker factorization + mAND gating | Medium | Architectural complexity |
| **JumpReLU** | ~92% (0.1402 -> 0.0114) | Threshold-only sparsity gradients; minimal dead features | Low | Requires careful gradient routing |
| **Binary Sparse Coding** | Moderate | Binarized activations prevent information smuggling | Low | Binary constraints may limit expressiveness |

### Current Best Methods for Dead Feature Resuscitation/Prevention

| Method | Mechanism | Effectiveness | Trade-off |
|--------|-----------|---------------|-----------|
| **AuxK Loss** (OpenAI) | Auxiliary reconstruction using dead latents; L_aux = ||e - e_hat||^2 with top-k_aux dead latents | "Eliminates almost all dead latents by end of training" | Low overhead; combined with tied initialization |
| **Ghost Gradients** (Anthropic) | Second forward pass with exponential activation on dead neurons; gradient pushes dead neurons toward explaining residual | Often achieves zero dead neurons | ~2x training cost; abandoned due to loss spikes; bug in original impl |
| **Soft-capping** | Bounds activation magnitudes | 6 vs 491 dead neurons at scale (latent size 12288) | Minimal fidelity impact |
| **L2 Weight Regularization** | Penalizes large weights; creates core of highly aligned features | Roughly doubles steering success rates; better cross-seed consistency | Aggressively kills many features; bimodal structure |
| **Tied Initialization** (W_enc = W_dec^T) | Better gradient flow at start | "Important for preventing dead latents" (OpenAI) | Standard practice |
| **Geometric Median Initialization** | Initialize decoder bias at geometric median of activation distribution | Avoids both dense and dead features; reduces hyperparameter sensitivity | Good initialization practice |
| **Bias Adaptation (GBA)** | Adaptively adjusts bias parameters; "neuron resonance" principle | Outperforms benchmarks on LLMs up to 2B; theoretical guarantees | Moderate overhead |
| **Activation Decay** | Penalize mean of squared sparse feature activations | Alternative to resampling for deep SAEs | Deep SAE specific |
| **JumpReLU (threshold-only gradients)** | Sparsity gradient only to threshold theta, not encoder/decoder | Hardly any dead features | Best reconstruction fidelity |
| **Lower Learning Rates** | Slower, more stable optimization | Anecdotally helps decrease dead latents | Longer training |
| **LR Warmup** | Gradual LR increase at start | Keeps features alive before ghost gradients activate | Minimal overhead |

### The Absorption-Hedging Trade-off (Key 2025 Insight)

| Problem | Cause | Affected Architecture | Solution Approach |
|---------|-------|----------------------|-------------------|
| **Feature Absorption** | Sparsity loss + hierarchical co-occurrence | Wide SAEs | Matryoshka hierarchy; orthogonality constraints |
| **Feature Hedging** | Reconstruction (MSE) loss + correlations | Narrow SAEs | Balanced loss coefficients; wider SAEs |

Chanin et al. (2025) showed these are complementary problems: Matryoshka SAEs trade absorption for hedging in inner (narrow) levels.

### Training Stability: Architecture Comparison

| Architecture | Stability Strength | Stability Weakness | Dead Features | Absorption Score |
|-------------|-------------------|-------------------|---------------|------------------|
| **Standard SAE (L1)** | Simple | Worst feature absorption; unstable training | High | 0.0161 |
| **TopK SAE** | Fast, efficient; fixed sparsity | Feature absorption (0.1402); dead features without AuxK | High (without AuxK) | 0.1402 |
| **JumpReLU** | Best reconstruction fidelity; minimal dead features | Requires careful gradient routing (only to theta) | Very low | 0.0114 |
| **BatchTopK** | Adaptive per-sample sparsity | Moderate absorption | Moderate | Moderate |
| **ATM (2025)** | Best feature absorption (0.0068); adaptive masking | Newer, less validated | Low | 0.0068 |
| **L2-Regularized TopK** | Excellent cross-seed consistency; better steering | Aggressive feature death (many dead latents) | High (bimodal) | Improved |
| **Matryoshka SAE** | Hierarchical feature organization | Higher compute; hedging in inner levels | Low | 0.005 (outer) |

### Mainstream Datasets and Evaluation Standards

- **Training data**: OpenWebText (most common), The Pile
- **Models**: GPT-2 Small (most common for research), Pythia series (70M-2.8B), Llama 3.2, Gemma 2, Claude 3 Sonnet
- **Standard evaluation metrics**:
  - **L0 Sparsity**: Average number of active features per input
  - **MSE / Normalized MSE / FVU**: Reconstruction error
  - **Loss Recovered**: Cross-entropy preservation when substituting SAE reconstructions; formula: (H* - H0) / (H_orig - H0)
  - **R^2 (Explained Variance)**: Proportion of variance captured
  - **Feature Absorption**: Fraction of cases where correct latents fail to activate (mean and full absorption)
  - **Dead Latent Percentage**: Fraction of features that never activate
  - **Cross-seed Consistency**: Feature alignment across different random initializations
  - **Steering Success Rate**: Ability to modify model behavior via feature intervention
  - **Sparse Probing**: k=1,2,5 probing accuracy for concept detection
  - **Auto-Interp**: LLM-as-judge automated interpretability scores

### SAEBench (2025) --- The Emerging Standard

SAEBench organizes evaluation around four capability dimensions:

| Capability | Metrics Used |
|------------|-------------|
| **Concept Detection** | Sparse Probing, Feature Absorption |
| **Interpretability** | LLM-as-judge automated interpretability |
| **Reconstruction** | Loss Recovered, KL Divergence, MSE |
| **Feature Disentanglement** | Unlearning, SCR, TPP, RAVEL |

Novel metrics introduced:
- **SCR (Spurious Correlation Removal)**: Tests if zero-ablating small latent sets removes unwanted correlations
- **TPP (Targeted Probe Perturbation)**: Measures completeness and isolation of concepts in small latent groups
- **RAVEL**: Evaluates clean separation of related attributes via interventions

Critical insight: "Small gains in sparsity-fidelity trade-off do not necessarily translate into qualitatively better representations."

### Transcoders as Alternative Paradigm (2025)

Paulo et al. (2025) proposed transcoders as a superior alternative to SAEs:

| Aspect | SAEs | Transcoders |
|--------|------|-------------|
| Objective | Self-reconstruction | Input-output function approximation |
| Training target | x_r = x_p (same layer) | x_p = NN_l(s), x_r = NN_{l+1}(s) |
| What they capture | Polysemantic activations | Layer-to-layer transformations |
| Key advantage | Simple | More interpretable; better generalization |

Skip transcoders (with affine skip connection) Pareto-dominate SAEs on reconstruction vs. interpretability. The authors recommend shifting focus from SAEs to (skip) transcoders.

## 4. Identified Research Gaps

### Absorption-Specific Gaps

- **Gap 1: Theoretical understanding of absorption-hedging trade-off**. While Matryoshka SAEs reduce absorption, they introduce hedging in inner levels. The fundamental Pareto frontier of this trade-off is not well-characterized theoretically.
- **Gap 2: Absorption-aware training objectives**. Current solutions modify architecture; absorption-aware loss terms that directly penalize absorption during training are underexplored.
- **Gap 3: Unsupervised absorption detection**. Current metrics require knowing the "parent" feature a priori. Detecting absorption without ground truth is unsolved.
- **Gap 4: Cross-architecture comparison on absorption**. Most absorption studies focus on JumpReLU/GemmaScope. Systematic comparison across all major architectures using standardized absorption metrics is lacking.
- **Gap 5: Absorption in non-text modalities**. SAEs are being applied to vision-language models. Whether absorption manifests differently in cross-modal features is unknown.

### Dead Feature-Specific Gaps

- **Gap 6: Theoretical understanding of feature death**. While empirical methods (AuxK, ghost gradients) work well, the theoretical conditions under which features die and can be recovered remain poorly understood. The "neuron resonance" principle from GBA is a start but needs extension.
- **Gap 7: Fundamental limits of recovery**. Dip Roy et al. (2025) showed zero recovery on dSprites even after 100 epochs. Whether this holds for LLM SAEs at moderate sparsity is unknown.
- **Gap 8: Dynamic feature resuscitation during inference**. Current resuscitation methods operate during training. Real-time revival of dead features during model deployment is unexplored.
- **Gap 9: Interaction between absorption and dead features**. These two problems are often studied separately, but they likely interact --- absorbed features may be more prone to dying, and dead features may contribute to absorption in surviving features.

### Training Stability Gaps

- **Gap 10: Unified stability metric**. The field uses many metrics (absorption score, dead latent %, cross-seed consistency) but lacks a unified framework that captures the full stability landscape.
- **Gap 11: Scaling to larger models**. Most dead feature/absorption research is validated on models <= 2B parameters. Behavior at 7B+ and especially 70B+ remains largely unstudied.
- **Gap 12: Training stability under data distribution shift**. SAE stability when trained on different data distributions or when deployed on out-of-distribution inputs is underexplored.
- **Gap 13: What do SAEs actually capture?** The finding that SAEs interpret randomly initialized transformers raises fundamental questions about whether absorption and dead features stem from data statistics and architecture rather than learned computations.

## 5. Available Resources

### Open-source Code

| Resource | URL | Description | Relevance |
|----------|-----|-------------|-----------|
| **SAELens** | https://github.com/jbloomAus/SAELens | Training, loading, analyzing SAEs; integrates GemmaScope; standard tool in MI community; supports ReLU, Gated, TopK, JumpReLU | **High** --- foundational library |
| **sae-spelling** | https://github.com/lasr-spelling/sae-spelling | Official code for "A is for Absorption"; FeatureAbsorptionCalculator, k-sparse probing | **High** --- direct absorption tooling |
| **matryoshka_sae** | https://github.com/bartbussmann/matryoshka_sae | Matryoshka SAE implementation; hierarchical dictionary learning | **High** --- primary absorption solution |
| **SAEBench** | https://github.com/adamkarvonen/SAEBench | Comprehensive benchmark framework for SAE evaluation | **High** --- standard evaluation suite |
| **TransformerLens** | https://github.com/neelnanda-io/TransformerLens | Mechanistic interpretability library for transformers | **High** --- required for interventions |
| **Sparsify** | https://github.com/EleutherAI/sparsify | EleutherAI's lean alternative focused on TopK SAEs | **Medium** --- alternative implementation |
| **SAETrainer** | https://github.com/sionic-ai/SAETrainer | Another training framework with different architectural choices | **Medium** --- alternative |
| **SAELens-V** | https://github.com/PKU-Alignment/SAELens-V | Extension for multimodal (vision + language) SAEs | **Medium** --- for cross-modal research |
| **MSAE** | https://github.com/WolodjaZ/MSAE | Hierarchical SAE for CLIP/SigLIP | **Medium** --- vision-language specific |
| **HSAE** | https://github.com/nqgl/HSAE | Hierarchical Sparse Autoencoder (early work) | **Medium** --- precursor to Matryoshka |

### Datasets and Pretrained Models

- **GemmaScope** (Google, 2024): Pre-trained JumpReLU SAEs for Gemma-2-2B and Gemma-2-9B, every layer, widths 16k-1M
- **GPT-2 Small SAEs**: Available via SAELens for all residual stream layers
- **Pythia SAEs**: Available via SAELens
- **OpenWebText / The Pile**: Standard training corpora for SAEs
- **SAEBench evaluation suite**: Standardized datasets for all 8+ metrics
- **CE-Bench dataset**: Contrastive evaluation pairs for SAE interpretability assessment

### Key Libraries

```python
# SAELens installation
pip install sae-lens

# Loading GemmaScope SAEs
from sae_lens import SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_0/width_16k/canonical",
)
```

## 6. Implications for Idea Generation

### Saturated Directions

- Pure architectural innovations without absorption analysis (many 2024 papers)
- Single-metric optimization (sparsity or reconstruction alone)
- Small-scale validation (single model, single layer)
- Basic feature splitting observation (well-documented)
- Simple sparsity-fidelity trade-off analysis (well-covered by SAEBench)
- L1-sparsity SAEs with resampling --- superseded by TopK/JumpReLU
- Pure empirical scaling studies without theoretical insight

### Promising Directions

1. **Joint absorption-death modeling**: Design a unified objective that simultaneously addresses feature absorption and dead features, exploiting their likely interaction. Current methods treat these as separate problems.

2. **Absorption-aware training objectives**: Rather than architectural changes, design loss terms that directly penalize absorption during training. This could combine with any SAE architecture (TopK, JumpReLU, etc.).

3. **Adaptive resuscitation schedules**: Instead of fixed thresholds (e.g., 2M tokens for ghost gradients, k_aux=512 for AuxK), learn dynamic resuscitation triggers based on feature importance and training dynamics.

4. **Hierarchical resuscitation**: Extend Matryoshka's hierarchical structure with explicit resuscitation pathways between levels --- when a feature dies at one level, attempt to revive it using information from adjacent levels.

5. **Stability-aware architecture search**: Automatically discover SAE architectures that optimize the stability-reconstruction-sparsity Pareto frontier, considering absorption, dead features, and cross-seed consistency jointly.

6. **Theoretical characterization of absorption**: Develop mathematical conditions under which absorption occurs, analogous to identifiability theory for feature recovery. The closed-form analysis by Cui et al. (2025) provides a starting framework.

7. **Unsupervised absorption detection**: Methods to identify absorption without ground-truth parent features. This would enable absorption auditing at scale across all SAE features.

8. **Scale-dependent analysis**: Systematically study how absorption and dead features change with model scale (100M -> 1B -> 10B+ parameters). Most research stops at 2B.

9. **Training-free mitigation**: Post-hoc methods to recover absorbed features from trained SAEs without retraining.

10. **Cross-modal absorption**: Study whether absorption manifests differently in vision-language models. Cross-modal hierarchical features may exhibit novel patterns.

11. **What do SAEs actually capture?**: Follow-up to the randomly-initialized transformers finding; distinguish architecture/data artifacts from learned computations.

### Cross-Domain Analogies with Potential

- **Signal processing**: Source separation problems (e.g., ICA with dependent sources) face similar "absorption" issues where correlated signals merge. Techniques from blind source separation may transfer.
- **Topic modeling**: Hierarchical topic models (e.g., hLDA) handle nested concepts without absorption-like failures. Their probabilistic approach to hierarchy may inspire SAE variants.
- **Neuroscience**: The "grandmother cell" debate (whether neurons encode specific vs. distributed representations) parallels the monosemanticity debate. Insights from sparse coding in visual cortex may apply.
- **Mixture of Experts**: Load balancing techniques from MoE training could address feature utilization imbalance (dead features).
- **Dictionary learning in signal processing**: OMP and CoSaMP algorithms have sophisticated atom selection criteria that could inspire SAE feature selection.

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens | High | MIT | **Adopt** | Mature library with GemmaScope integration; essential infrastructure for all SAE experiments; supports all major variants |
| sae-spelling | High | Unknown | **Adopt/Extend** | Official absorption measurement code; directly implements FeatureAbsorptionCalculator and k-sparse probing |
| matryoshka_sae | High | Unknown | **Adopt/Extend** | Primary solution to absorption; based on SAELens; can be extended with custom loss coefficients |
| SAEBench | High | MIT | **Adopt** | Standardized evaluation; necessary for rigorous benchmarking beyond sparsity-fidelity |
| TransformerLens | High | MIT | **Adopt** | Required for activation extraction and interventions |
| GemmaScope SAEs | High | Apache 2.0 | **Adopt** | Pre-trained SAEs eliminate training cost; established absorption test cases |
| Sparsify | Medium | MIT | **Extend** | Lean TopK-focused; good for experiments needing minimal dependencies |
| Chanin et al. absorption metric | High | N/A (paper) | **Extend** | Good foundation but layer-limited; extend to all layers |
| HSAE (Muchane et al.) | Medium | N/A | **Reference** | Hierarchical modeling approach informs design |
| KronSAE | Medium | N/A | **Reference** | mAND gating mechanism may inspire solutions |

### Recommended Stack

```
Base: SAELens + TransformerLens + PyTorch
Evaluation: SAEBench metrics + custom absorption metrics + CE-Bench
Models: Gemma-2-2B (primary), Pythia-70M-deduped (fast iteration), GPT-2 Small (historical baseline)
SAEs: GemmaScope (pre-trained), train custom variants if needed
```

### Key Reusable Components

1. **SAELens SAE loading**: Direct access to 1000+ pre-trained SAEs
2. **SAEBench evaluation pipeline**: Standardized metrics computation
3. **GemmaScope first-letter features**: Chanin et al. established these as absorption test cases
4. **k-sparse probing**: Feature splitting detection from absorption paper
5. **Integrated gradients ablation**: Causal verification from absorption paper
6. **FeatureAbsorptionCalculator**: Direct measurement tool from sae-spelling repo
7. **AuxK loss implementation**: From OpenAI's scaling paper; standard in SAELens
8. **Ghost gradients reference**: From Anthropic's Transformer Circuits Thread (for historical comparison)

### Critical Implementation Notes

- **Use TopK or JumpReLU instead of L1 sparsity** for new experiments --- L1 is now considered deprecated for SAEs.
- **Always include AuxK or equivalent dead feature mitigation** --- training without it yields unreliable results.
- **Use tied initialization (W_enc = W_dec^T)** --- shown to be important for preventing dead latents.
- **Track cross-seed consistency** --- this is becoming a standard evaluation criterion alongside reconstruction and sparsity.
- **Monitor absorption score explicitly** --- many papers now report this alongside traditional metrics.
- **Consider geometric median initialization for decoder bias** --- avoids both dense and dead features.
- **For JumpReLU**: Ensure sparsity gradient flows ONLY to threshold theta, not encoder/decoder --- this is critical for minimizing dead features.
- **For Matryoshka SAEs**: Tune beta_m ~ 0.75 for balanced absorption-hedging trade-off.

---

## Bibliography

1. Elhage, N., et al. (2022). Toy Models of Superposition. Transformer Circuits Thread. arXiv:2209.10652.
2. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread.
3. Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600.
4. Templeton, A., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Anthropic.
5. Gao, L., et al. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093.
6. Rajamanoharan, S., et al. (2024). Improving Dictionary Learning with Gated Sparse Autoencoders. arXiv:2404.16014.
7. Rajamanoharan, S., et al. (2024). Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders. arXiv:2407.14435.
8. Chanin, D., et al. (2024). A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders. arXiv:2409.14507.
9. Jermyn, A., & Templeton, A. (2024). Ghost Grads: An improvement on resampling. Transformer Circuits Thread (January 2024 Update).
10. Makkuva, A., et al. (2024). Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders. arXiv:2411.13117.
11. Bussmann, B., et al. (2024). BatchTopK Sparse Autoencoders. arXiv:2412.06410.
12. Karvonen, A., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. arXiv:2503.09532.
13. Bussmann, B., et al. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547.
14. Chanin, D., et al. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756.
15. Korznikov, et al. (2025). Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033.
16. Chen, S., et al. (2025). Provable Feature Recovery via Sparse Autoencoders. arXiv:2506.14002.
17. Cui, J., et al. (2025). On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963.
18. Dip Roy, et al. (2025). Fundamental Limits of Neural Network Sparsification. arXiv:2603.18056.
19. Li, T. E., et al. (2025). Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training. arXiv:2510.08855.
20. Crook, O., et al. (2026). Stable and Steerable Sparse Autoencoders with Weight Regularization. arXiv:2603.04198.
21. Paulo, G., Shabalin, S., & Belrose, N. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv:2501.18823.
22. Various. (2025). Sparse Autoencoders Can Interpret Randomly Initialized Transformers. arXiv:2501.17727.
23. Tian, C., et al. (2025). Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717.
24. Muchane, M., et al. (2025). Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. arXiv:2506.01197.
25. Luo, Y., et al. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. arXiv:2602.11881.
26. Saraswatula, A., & Klindt, D. (2025). Data Whitening Improves Sparse Autoencoder Learning. arXiv:2511.13981.
27. O'Neill, C., et al. (2025). Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders. arXiv:2508.09363.
28. Song, X., et al. (2025). Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs. arXiv:2505.20254.
29. Various. (2025). Kronecker Factorization Improves Efficiency and Interpretability of SAEs. arXiv:2505.22255.
30. Various. (2025). Distribution-Aware Feature Selection for SAEs. arXiv:2508.21324.
31. Gulko, A., et al. (2025). CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark. arXiv:2509.00691.
32. Chen, S., et al. (2025). Taming Polysemanticity in LLMs: Theory-Grounded Feature Recovery. ICLR 2025.
33. Various. (2025). Train One Sparse Autoencoder Across Multiple Sparsity Levels. EMNLP 2025.
34. Zaigrajew, V., et al. (2025). Interpreting CLIP with Hierarchical Sparse Autoencoders. arXiv:2502.20578.
35. Various. (2025). Deep sparse autoencoders yield interpretable features too. Alignment Forum.
36. Leask, P., et al. (2025). Sparse Autoencoders Do Not Find Canonical Units of Analysis. ICLR 2025.
37. Various. (2025). Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models. ICLR 2025.
38. Various. (2025). Binary Sparse Coding for Interpretability. arXiv:2509.25596.
39. Various. (2025). Self-Ablating Transformers: More Interpretability, Less Sparsity. arXiv:2505.00509.
40. Lieberum, T., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.
41. Bloom, J., et al. (2024). SAELens: Training Sparse Autoencoders on Language Models. https://github.com/jbloomAus/SAELens
