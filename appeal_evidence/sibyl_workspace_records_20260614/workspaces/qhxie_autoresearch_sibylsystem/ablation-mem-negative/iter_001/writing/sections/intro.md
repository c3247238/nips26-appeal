# Introduction

Sparse Autoencoders (SAEs) have emerged as the dominant unsupervised tool for mechanistic interpretability, decomposing neural network activations into sparse, human-interpretable features [Bricken et al., 2023; Cunningham et al., 2023]. The core promise is that each SAE dictionary element corresponds to a semantically coherent concept---what has been termed *monosemanticity*. Yet a persistent concern undermines this promise: *feature absorption*, where distinct semantic concepts are mapped to shared dictionary elements, potentially degrading the reliability of interpretability analyses.

Feature absorption was first formally characterized by Chanin et al. [2024], who demonstrated that parent features (e.g., "starts with a vowel") can suppress child features (e.g., "starts with 'a'") when both co-occur, causing the SAE to fire only the more general feature. This creates systematic "holes" in feature coverage: the SAE appears to have learned a clean monosemantic representation, but silently fails to activate for subsets of its semantic domain. Absorption is connected to the broader phenomenon of *superposition* [Elhage et al., 2022], where neural networks represent more features than dimensions by encoding features in non-orthogonal directions. While superposition has been extensively studied at the neuron level, its manifestation in SAE dictionaries remains poorly understood.

Prior work has documented absorption in single architectures [Chanin et al., 2024], but it has never been systematically quantified across SAE architectures, and its causal impact on downstream interpretability tasks remains unproven. We use GPT-2 Small as a canonical testbed due to its widespread adoption in interpretability research, comparing pretrained JumpReLU (GemmaScope) and trained TopK architectures. We evaluate six hypotheses (H1--H6, defined in Section 3.2).

**Research Questions.** This work addresses four open questions:
- **RQ1**: How do collision rates compare across SAE architectures under standardized conditions?
- **RQ2**: Does absorption causally impair downstream interpretability tasks?
- **RQ3**: Can absorption be detected without ground-truth parent features?
- **RQ4**: Can absorbed features be recovered at inference time without SAE retraining?

**Contributions.** We make four contributions:
1. **CAAB**: The first cross-architecture absorption benchmark, comparing JumpReLU and TopK SAEs with standardized metrics.
2. **Causal assessment**: Controlled experiments linking collision rate to sparse probing accuracy.
3. **UAD**: An unsupervised detection method achieving 54.3% precision at 100% recall, requiring no labeled hierarchies.
4. **DFDA**: An SAE-retraining-free mitigation via lightweight residual compensation (11.1% per-pair residual MSE improvement with 388 parameters).

**Key Finding.** Collision rates differ dramatically by architecture (15.4% vs. 3.8%), yet the correlation with downstream task performance is near-zero ($\rho_S$ = 0.10, $p$ = 0.870)---suggesting the community may be over-indexing on collision rate as a quality metric.

---
