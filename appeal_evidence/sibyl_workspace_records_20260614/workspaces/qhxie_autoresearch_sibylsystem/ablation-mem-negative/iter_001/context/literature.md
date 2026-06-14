# Literature Survey Report

**Research Topic**: Feature Absorption in Sparse Autoencoders (SAEs) for Mechanistic Interpretability
**Survey Date**: 2026-04-28
**arXiv Search Keywords**: ["feature absorption" "sparse autoencoder", "feature splitting" "sparse autoencoder", "SAE benchmark" "sparse autoencoder evaluation", "JumpReLU" "Gated SAE" "TopK SAE", "Matryoshka SAE" "hierarchical sparse autoencoder", "provable feature recovery" "sparse autoencoder"]
**Web Search Keywords**: ["sparse autoencoder feature absorption SAE mechanistic interpretability 2024 2025", "SAEBench sparse autoencoder benchmark evaluation", "SAELens sparse autoencoder library GemmaScope pretrained SAEs", "Matryoshka SAE hierarchical absorption solution 2025", "transcoder vs sparse autoencoder interpretability 2025", "provable feature recovery sparse autoencoder theoretical guarantee 2025"]

## 1. Field Overview

Sparse Autoencoders (SAEs) have emerged as the dominant technique for mechanistic interpretability of Large Language Models (LLMs), enabling researchers to decompose dense model activations into sparse, human-interpretable features. The theoretical foundation was laid by Elhage et al.'s "Toy Models of Superposition" (2022), which established that neural networks represent more features than they have dimensions by encoding features as approximately orthogonal directions in activation space.

The seminal application of SAEs to language models came with Bricken et al.'s "Towards Monosemanticity" (2023) from Anthropic's Transformer Circuits Thread, which demonstrated that dictionary learning via sparse autoencoders could decompose MLP activations into monosemantic features. Cunningham et al. (2023) showed SAEs outperform PCA/ICA on automated interpretability scores. This was followed by Templeton et al.'s scaling work (2024) on Claude 3 Sonnet, extracting millions of interpretable features.

However, a critical limitation was identified in 2024: **feature absorption** — a phenomenon where broad, interpretable features get "absorbed" into more specific, token-aligned latents, creating interpretability illusions with arbitrary false negatives. This discovery, formalized in "A is for Absorption" (Chanin et al., 2024), has sparked a vigorous research direction focused on understanding, measuring, and mitigating absorption.

By 2025, the field has expanded dramatically to include:
- **Architectural solutions**: Matryoshka SAEs (hierarchical nesting), Orthogonal SAEs (orthogonality constraints), KronSAE, ATM, HSAE
- **Theoretical analyses**: First provable feature recovery guarantees (Chen et al., 2025), closed-form optimal solutions revealing fundamental limits (Cui et al., 2025)
- **Comprehensive benchmarks**: SAEBench (2025) with 8+ metrics beyond sparsity-fidelity
- **Alternative paradigms**: Transcoders (Paulo et al., 2025) challenging SAEs' primacy for interpretability
- **Critical reassessments**: Findings that SAEs interpret even randomly initialized transformers, raising questions about what SAEs actually capture

The tension between reconstruction fidelity, sparsity, and feature quality — with the added complexity of the absorption-hedging trade-off — remains the central challenge.

## 2. Core References

| # | Title | Authors | Source | Year | Key Contribution | Limitations |
|---|-------|---------|--------|------|-----------------|-------------|
| 1 | Toy Models of Superposition | Elhage et al. | Transformer Circuits Thread / arXiv:2209.10652 | 2022 | Established theoretical foundation of superposition; showed networks learn to represent sparse features in superposition | Toy models only; not applied to real LLMs |
| 2 | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | Bricken et al. | Transformer Circuits Thread | 2023 | First application of SAEs to real transformers; demonstrated monosemantic feature recovery; introduced feature splitting | Single-layer model only; limited scale |
| 3 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | Cunningham et al. | arXiv:2309.08600 | 2023 | Showed SAEs outperform PCA/ICA on automated interpretability scores | Did not identify absorption as a failure mode |
| 4 | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Templeton et al. | Anthropic / arXiv | 2024 | Scaled SAE methods to production LLM; millions of features; safety-relevant feature discovery | Did not systematically study absorption at scale |
| 5 | Scaling and evaluating sparse autoencoders | Gao et al. | arXiv:2406.04093 | 2024 | Introduced k-sparse autoencoders (TopK), scaling laws, new quality metrics; trained 16M latent SAE | Does not address absorption specifically |
| 6 | Improving Dictionary Learning with Gated Sparse Autoencoders | Rajamanoharan et al. | arXiv:2404.16014 | 2024 | Gated SAE: separates gating from magnitude estimation; solves shrinkage; Pareto improvement | Absorption not studied |
| 7 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | Rajamanoharan et al. | arXiv:2407.14435 | 2024 | JumpReLU activation: SOTA reconstruction fidelity at given sparsity; trains L0 directly | Later shown to have absorption issues |
| 8 | **A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders** | Chanin, Wilken-Smith, Dulka, Bhatnagar, Bloom | arXiv:2409.14507 | 2024 | **Foundational paper**: Identified and named feature absorption; linked cause to sparsity loss + hierarchical co-occurrence; showed varying size/sparsity insufficient | Metric relies on ablation (limited to early layers); conservative underestimate; focused on GPT-2 |
| 9 | Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders | Makkuva et al. | arXiv:2411.13117 | 2024 | Theoretical analysis of SAE inference optimality; identified amortization gap | Not directly about absorption |
| 10 | BatchTopK Sparse Autoencoders | Bussmann et al. | arXiv:2412.06410 | 2024 | Batch-level top-k: adaptive latent allocation; outperforms TopK, comparable to JumpReLU | Absorption not evaluated |
| 11 | **SAEBench: A Comprehensive Benchmark for Sparse Autoencoders** | Karvonen et al. | arXiv:2503.09532 | 2025 | **Standardized evaluation**: 8+ metrics across interpretability, disentanglement, applications; 200+ SAEs evaluated; moved beyond sparsity-fidelity | Proxy metrics may not fully capture absorption |
| 12 | **A Survey on Sparse Autoencoders: Interpreting the Internal Representations of LLMs** | Various | arXiv:2503.05613 / EMNLP 2025 | 2025 | Comprehensive survey covering SAE architectures, training, evaluation, and applications | Survey paper; limited novel contributions |
| 13 | **Learning Multi-Level Features with Matryoshka Sparse Autoencoders** | Bussmann et al. | arXiv:2503.17547 | 2025 | Proposed hierarchical nested SAEs; achieved ~90% reduction in absorption (0.49 -> 0.05 at L0=40) | Introduces feature hedging in inner levels; higher computational cost |
| 14 | **Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders** | Chanin et al. | arXiv:2505.11756 | 2025 | Identified hedging as the complement to absorption; showed Matryoshka trades absorption for hedging; proposed balanced loss coefficients (beta_m ~ 0.75) | Limited to empirical analysis; no theoretical characterization of trade-off |
| 15 | **Orthogonal Sparse Autoencoders Uncover Atomic Features** | Korznikov et al. | arXiv:2509.22033 | 2025 | Alternative solution via orthogonality constraints (cosine similarity penalty); ~65% absorption reduction; ~50% less compute than Matryoshka | Slightly lower explained variance |
| 16 | **Provable Feature Recovery via Sparse Autoencoders** | Chen et al. | arXiv:2506.14002 / ICLR 2025 | 2025 | **First SAE algorithm with theoretical recovery guarantees**; introduced Group Bias Adaptation (GBA); validated on 2B parameter LLMs | Guarantees require specific statistical model assumptions |
| 17 | **On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy** | Cui et al. | arXiv:2506.15963 | 2025 | **First closed-form optimal solution analysis**; proved standard SAEs fail unless features are extremely sparse; identified feature shrinking/vanishing; proposed Weighted SAE (WSAE) | Theoretical analysis; limited empirical validation |
| 18 | **Transcoders Beat Sparse Autoencoders for Interpretability** | Paulo, Shabalin, Belrose | arXiv:2501.18823 | 2025 | Proposed transcoders as superior alternative; skip transcoders Pareto-dominate SAEs; higher automated interpretability scores across Pythia, Llama, Gemma | Different objective (input-output vs. self-reconstruction); not directly comparable for all use cases |
| 19 | Sparse Autoencoders Can Interpret Randomly Initialized Transformers | Various | arXiv:2501.17727 | 2025 | Showed SAEs find similar features in trained and untrained models; raises questions about what SAEs actually capture | Challenges the assumption that SAE features reflect learned computations |
| 20 | Measuring Sparse Autoencoder Feature Sensitivity | Tian et al. | arXiv:2509.23717 | 2025 | Introduces feature sensitivity: reliability of feature activation on semantically similar texts | Complementary to absorption; not causal |
| 21 | Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures | Muchane et al. | arXiv:2506.01197 | 2025 | HSAE: explicitly models semantic hierarchy; improves reconstruction and interpretability | Requires hierarchical structure assumption |
| 22 | From Atoms to Trees: Building a Structured Feature Forest with Hierarchical SAEs | Luo et al. | arXiv:2602.11881 | 2026 | HSAE jointly learns SAEs and parent-child relationships; recovers meaningful hierarchies | Very recent; limited validation |
| 23 | Data Whitening Improves Sparse Autoencoder Learning | Saraswatula & Klindt | arXiv:2511.13981 | 2025 | PCA whitening improves interpretability metrics on SAEBench; challenges sparsity-fidelity tradeoff | Minor reconstruction drops |
| 24 | Resurrecting the Salmon: Domain-Specific Sparse Autoencoders | O'Neill et al. | arXiv:2508.09363 | 2025 | Domain confinement mitigates fragmentation/absorption; 20% more variance explained | Limited to medical domain |
| 25 | Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs | Song et al. | arXiv:2505.20254 | 2025 | Argues for feature consistency (PW-MCC metric); high consistency achievable with right architectures | Consistency != absence of absorption |
| 26 | Time-Aware Feature Selection: Adaptive Temporal Masking for Stable SAE Training | Various | arXiv:2510.08855 | 2025 | ATM: temporal dynamics + probabilistic masking; substantially lower absorption scores | Very recent; limited validation |
| 27 | Kronecker Factorization Improves Efficiency and Interpretability of SAEs | Various | arXiv:2505.22255 | 2025 | KronSAE: Kronecker factorization + mAND gating; reduces absorption via AND-like behavior | Architectural complexity |
| 28 | Distribution-Aware Feature Selection for SAEs | Various | arXiv:2508.21324 | 2025 | L2-norm/Squared-l scoring reduces absorption vs BatchTopK; no single config dominates all metrics | Trade-offs remain |
| 29 | CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark | Gulko et al. | arXiv:2509.00691 | 2025 | LLM-free contrastive evaluation using story pairs; 70%+ Spearman correlation with SAEBench | Limited metric coverage |
| 30 | Taming Polysemanticity in LLMs: Theory-Grounded Feature Recovery | Chen et al. | ICLR 2025 | 2025 | "Neuron resonance" phenomenon; neurons learn monosemantic features when activation frequency matches feature occurrence | Theoretical; limited scope |

## 3. SOTA Methods and Benchmarks

### Current Best Methods for Addressing Absorption

| Method | Absorption Reduction | Key Innovation | Computational Overhead | Trade-off |
|--------|---------------------|----------------|----------------------|-----------|
| **Matryoshka SAE** | ~90% (0.49 -> 0.05 at L0=40) | Hierarchical nested dictionaries with independent reconstruction constraints | High (multiple dictionary sizes) | Introduces hedging in inner levels |
| **Balanced Matryoshka SAE** | Optimized trade-off | Tuned relative loss coefficients (beta_m ~ 0.75) | High | Balances absorption vs. hedging |
| **Orthogonal SAE (OrtSAE)** | ~65% | Cosine similarity penalty between latents | Low (~50% less than Matryoshka) | Slightly lower explained variance |
| **Weighted SAE (WSAE)** | Improved low-sparsity recovery | Theoretically-grounded reweighting strategy | Low | Limited empirical validation |
| **Group Bias Adaptation (GBA)** | Strong empirical results | Adaptive bias adjustment for feature identifiability | Medium | Requires specific statistical assumptions |
| **KronSAE** | Lower absorption | Kronecker factorization + mAND gating | Medium | Architectural complexity |
| **ATM** | Substantially lower absorption | Temporal masking | Medium | Very recent; limited validation |
| **HSAE** | Recovers hierarchies | Explicit hierarchy modeling | Medium | Assumes hierarchical structure |

### The Absorption-Hedging Trade-off (Key 2025 Insight)

| Problem | Cause | Affected Architecture | Solution Approach |
|---------|-------|----------------------|-------------------|
| **Feature Absorption** | Sparsity loss + hierarchical co-occurrence | Wide SAEs | Matryoshka hierarchy; orthogonality constraints |
| **Feature Hedging** | Reconstruction (MSE) loss + correlations | Narrow SAEs | Balanced loss coefficients; wider SAEs |

Chanin et al. (2025) showed these are complementary problems: Matryoshka SAEs trade absorption for hedging in inner (narrow) levels.

### Mainstream Datasets and Evaluation Standards

- **Training data**: OpenWebText (most common), The Pile
- **Models**: GPT-2 Small (most common for research), Pythia series, Llama 3.2, Gemma 2, Claude 3 Sonnet
- **Standard evaluation metrics**:
  - **L0 Sparsity**: Average number of active features per input
  - **MSE / Normalized MSE**: Reconstruction error
  - **Loss Recovered**: Cross-entropy preservation when substituting SAE reconstructions; formula: (H* - H0) / (H_orig - H0)
  - **R^2 (Explained Variance)**: Proportion of variance captured
  - **Feature Absorption**: Fraction of cases where correct latents fail to activate (mean and full absorption)
  - **Sparse Probing**: k=1,2,5 probing accuracy for concept detection
  - **Auto-Interp**: LLM-as-judge automated interpretability scores

### SAEBench (2025) — The Emerging Standard

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

- **Gap 1: Theoretical understanding of absorption-hedging trade-off**. While Matryoshka SAEs reduce absorption, they introduce hedging in inner levels. The fundamental Pareto frontier of this trade-off is not well-characterized theoretically.

- **Gap 2: Absorption in non-English and multilingual settings**. Most absorption research focuses on English text. Whether absorption patterns differ across languages remains unexplored.

- **Gap 3: Dynamic/feature evolution perspective**. Absorption is typically studied in static, trained SAEs. How absorption emerges during training and whether it can be prevented dynamically is underexplored.

- **Gap 4: Absorption in non-text modalities**. SAEs are being applied to vision-language models and multimodal settings. Whether absorption manifests differently in cross-modal features is unknown.

- **Gap 5: Causal impact of absorption on downstream tasks**. While absorption is known to create interpretability illusions, its causal impact on specific downstream capabilities (e.g., safety classification, circuit tracing, steering) is not well-quantified.

- **Gap 6: Relationship between absorption and model scale**. Most absorption studies use GPT-2-scale models. Whether absorption worsens or improves with model scale is an open question.

- **Gap 7: Alternative architectures beyond hierarchy and orthogonality**. Matryoshka and OrtSAE represent two solution paradigms (hierarchy vs. constraint). Other architectural innovations (e.g., attention-based routing, learned sparsity patterns) remain unexplored.

- **Gap 8: Absorption-aware training objectives**. Current solutions modify architecture; absorption-aware loss terms that directly penalize absorption during training are underexplored.

- **Gap 9: Unsupervised absorption detection**. Current metrics require knowing the "parent" feature a priori. Detecting absorption without ground truth is unsolved.

- **Gap 10: Cross-architecture comparison on absorption**. Most absorption studies focus on JumpReLU/GemmaScope. Systematic comparison across all major architectures (Gated, TopK, BatchTopK, Matryoshka, OrtSAE) using standardized absorption metrics is lacking.

- **Gap 11: What do SAEs actually capture?** The finding that SAEs interpret randomly initialized transformers raises fundamental questions about whether absorption and other artifacts stem from data statistics and architecture rather than learned computations.

## 5. Available Resources

### Open-source Code

| Resource | URL | Description | Relevance |
|----------|-----|-------------|-----------|
| **SAELens** | https://github.com/jbloomAus/SAELens | Training, loading, analyzing SAEs; integrates GemmaScope; standard tool in MI community | **High** — foundational library |
| **sae-spelling** | https://github.com/lasr-spelling/sae-spelling | Official code for "A is for Absorption"; FeatureAbsorptionCalculator, k-sparse probing | **High** — direct absorption tooling |
| **matryoshka_sae** | https://github.com/bartbussmann/matryoshka_sae | Matryoshka SAE implementation; hierarchical dictionary learning | **High** — primary absorption solution |
| **SAEBench** | https://github.com/adamkarvonen/SAEBench | Comprehensive benchmark framework for SAE evaluation | **High** — standard evaluation suite |
| **TransformerLens** | https://github.com/neelnanda-io/TransformerLens | Mechanistic interpretability library for transformers | **High** — required for interventions |
| **EleutherAI/sae** | https://github.com/EleutherAI/sae | General sparse autoencoder repository | **Medium** — alternative implementation |
| **HSAE** | https://github.com/nqgl/HSAE | Hierarchical Sparse Autoencoder (early work) | **Medium** — precursor to Matryoshka |

### Datasets and Pretrained Models

- **GemmaScope** (Google, 2024): Pre-trained JumpReLU SAEs for Gemma-2-2B and Gemma-2-9B, every layer, widths 16k-1M
- **GPT-2 Small SAEs**: Available via SAELens for all residual stream layers
- **Pythia SAEs**: Available via SAELens
- **OpenWebText**: Standard dataset for SAE training
- **SAEBench evaluation suite**: Standardized datasets for all 8+ metrics

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

### Promising Directions

1. **Absorption-aware training objectives**: Rather than architectural changes, design loss terms that directly penalize absorption during training. This could combine with any SAE architecture.

2. **Systematic absorption quantification**: Cross-architecture, cross-model, cross-layer absorption analysis using standardized metrics. No study has comprehensively compared all major architectures on absorption.

3. **Unsupervised absorption detection**: Methods to identify absorption without ground-truth parent features. This would enable absorption auditing at scale.

4. **Absorption impact on downstream tasks**: Quantify how absorption affects circuit discovery, steering efficacy, and model editing reliability.

5. **Theoretical characterization of absorption-hedging trade-off**: Toy models or closed-form analysis that predicts the Pareto frontier.

6. **Cross-modal absorption**: Study whether absorption manifests differently in vision-language models. Cross-modal hierarchical features may exhibit novel patterns.

7. **Scale-dependent absorption analysis**: Systematically study how absorption changes with model scale (100M -> 1B -> 10B+ parameters).

8. **Training-free mitigation**: Post-hoc methods to recover absorbed features from trained SAEs without retraining.

9. **What do SAEs actually capture?**: Follow-up to the randomly-initialized transformers finding; distinguish architecture/data artifacts from learned computations.

### Cross-Domain Analogies with Potential

- **Signal processing**: Source separation problems (e.g., ICA with dependent sources) face similar "absorption" issues where correlated signals merge. Techniques from blind source separation may transfer.
- **Topic modeling**: Hierarchical topic models (e.g., hLDA) handle nested concepts without absorption-like failures. Their probabilistic approach to hierarchy may inspire SAE variants.
- **Neuroscience**: The "grandmother cell" debate (whether neurons encode specific vs. distributed representations) parallels the monosemanticity debate. Insights from sparse coding in visual cortex may apply.

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| SAELens | High | MIT | **Adopt** | Mature library with GemmaScope integration; essential infrastructure for all SAE experiments |
| sae-spelling | High | Unknown | **Adopt/Extend** | Official absorption measurement code; directly implements FeatureAbsorptionCalculator and k-sparse probing |
| matryoshka_sae | High | Unknown | **Adopt/Extend** | Primary solution to absorption; based on SAELens; can be extended with custom loss coefficients |
| SAEBench | High | MIT | **Adopt** | Standardized evaluation; necessary for rigorous benchmarking beyond sparsity-fidelity |
| TransformerLens | High | MIT | **Adopt** | Required for activation extraction and interventions |
| GemmaScope SAEs | High | Apache 2.0 | **Adopt** | Pre-trained SAEs eliminate training cost; established absorption test cases |
| Chanin et al. absorption metric | High | N/A (paper) | **Extend** | Good foundation but layer-limited; extend to all layers |
| HSAE (Muchane et al.) | Medium | N/A | **Reference** | Hierarchical modeling approach informs design |
| KronSAE | Medium | N/A | **Reference** | mAND gating mechanism may inspire solutions |

### Recommended Stack

```
Base: SAELens + TransformerLens + PyTorch
Evaluation: SAEBench metrics + custom absorption metrics
Models: Gemma-2-2B (primary), GPT-2 Small (secondary)
SAEs: GemmaScope (pre-trained), train custom variants if needed
```

### Key Reusable Components

1. **SAELens SAE loading**: Direct access to 1000+ pre-trained SAEs
2. **SAEBench evaluation pipeline**: Standardized metrics computation
3. **GemmaScope first-letter features**: Chanin et al. established these as absorption test cases
4. **k-sparse probing**: Feature splitting detection from absorption paper
5. **Integrated gradients ablation**: Causal verification from absorption paper
6. **FeatureAbsorptionCalculator**: Direct measurement tool from sae-spelling repo

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
9. Makkuva, A., et al. (2024). Compute Optimal Inference and Provable Amortisation Gap in Sparse Autoencoders. arXiv:2411.13117.
10. Bussmann, B., et al. (2024). BatchTopK Sparse Autoencoders. arXiv:2412.06410.
11. Karvonen, A., et al. (2025). SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability. arXiv:2503.09532.
12. Bussmann, B., et al. (2025). Learning Multi-Level Features with Matryoshka Sparse Autoencoders. arXiv:2503.17547.
13. Chanin, D., et al. (2025). Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders. arXiv:2505.11756.
14. Korznikov, et al. (2025). Orthogonal Sparse Autoencoders Uncover Atomic Features. arXiv:2509.22033.
15. Chen, S., et al. (2025). Provable Feature Recovery via Sparse Autoencoders. arXiv:2506.14002.
16. Cui, J., et al. (2025). On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy. arXiv:2506.15963.
17. Paulo, G., Shabalin, S., & Belrose, N. (2025). Transcoders Beat Sparse Autoencoders for Interpretability. arXiv:2501.18823.
18. Various. (2025). Sparse Autoencoders Can Interpret Randomly Initialized Transformers. arXiv:2501.17727.
19. Tian, C., et al. (2025). Measuring Sparse Autoencoder Feature Sensitivity. arXiv:2509.23717.
20. Muchane, M., et al. (2025). Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures. arXiv:2506.01197.
21. Luo, Y., et al. (2026). From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders. arXiv:2602.11881.
22. Saraswatula, A., & Klindt, D. (2025). Data Whitening Improves Sparse Autoencoder Learning. arXiv:2511.13981.
23. O'Neill, C., et al. (2025). Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders. arXiv:2508.09363.
24. Song, X., et al. (2025). Position: Mechanistic Interpretability Should Prioritize Feature Consistency in SAEs. arXiv:2505.20254.
25. Gulko, A., et al. (2025). CE-Bench: Towards a Reliable Contrastive Evaluation Benchmark. arXiv:2509.00691.
26. Lieberum, T., et al. (2024). Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.
27. Bloom, J., et al. (2024). SAELens: Training Sparse Autoencoders on Language Models. https://github.com/jbloomAus/SAELens
