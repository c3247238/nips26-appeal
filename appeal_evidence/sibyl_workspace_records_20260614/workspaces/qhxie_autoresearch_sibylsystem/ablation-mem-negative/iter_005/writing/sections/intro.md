# 1 Introduction

Sparse Autoencoders (SAEs) have emerged as the dominant tool for extracting interpretable features from neural network activations [Bricken et al., 2023; Cunningham et al., 2023]. By learning an overcomplete dictionary of sparse, semantically meaningful features, SAEs enable researchers to identify what specific directions in a model's representation space encode. However, a critical failure mode---feature absorption---threatens the reliability of SAE-based interpretability.

**Feature absorption** occurs when a parent feature, which should represent a broad concept, is suppressed by child features that represent more specific sub-concepts [Chanin et al., 2024]. For example, a feature encoding "animal" may be suppressed by features encoding "dog," "cat," and "bird," making the parent feature appear non-existent or misleadingly weak. This creates dangerous interpretability illusions: researchers may conclude that a model lacks a concept when in fact it is simply absorbed by more specific features.

Detecting absorption is essential for accurate SAE interpretation. All existing detection methods, however, require **supervised ground truth**---known parent features and trained probes [Chanin et al., 2024; Karvonen et al., 2025]. This supervised requirement means absorption is only detectable for concepts where we already have labels, severely limiting scalability. An unsupervised detection method would unlock absorption analysis at scale, across the full SAE dictionary.

Unsupervised Absorption Detection (UAD) [Chanin et al., 2024] proposes co-occurrence clustering as a training-free solution. The core hypothesis is that features with hierarchical relationships will frequently co-occur on the same tokens. UAD's pipeline filters dead features, computes phi coefficients between feature pairs, hierarchically clusters features, and extracts candidate absorption pairs from within clusters---all without any labeled data.

**Our central question:** Does co-occurrence clustering detect feature absorption on pre-trained SAEs?

**Our answer:** No. UAD achieves F1 = 0.00048---detecting exactly 1 true positive out of 4,155 candidates---a performance numerically identical to random sampling within clusters. We identify the root cause: absorption features are **mutually exclusive at the token level**. A feature that activates on "three" never activates on "four," because these tokens represent different child concepts. Co-occurrence clustering detects features that fire *together*, but absorption features fire on *different* tokens. This is a structural mismatch, not a parameter-tuning problem.

Our investigation also reveals a positive finding. We demonstrate that **collision rate**---the Jaccard overlap of top-$K$ activating features---exhibits strong internal consistency across concept pairs (Spearman $r = 0.869$, $n = 56$, 95% CI $[0.780, 0.938]$). This indicates that our operationalization of absorption via top-$K$ feature overlap is structurally coherent: concept pairs with known hierarchical relationships show systematically higher overlap than unrelated pairs. We emphasize that this is an internal consistency check of our operationalization, not an independent proxy validation, as both metrics are computed from the same top-$K$ feature sets.

**Our contributions:**
1. **Empirical falsification:** We demonstrate that UAD achieves F1 = 0.0005, identical to a random baseline, on pre-trained SAEs.
2. **Root cause identification:** We show that token-level mutual exclusivity makes co-occurrence clustering fundamentally incompatible with hierarchical absorption detection.
3. **Operationalization consistency:** We demonstrate that collision rate exhibits strong internal consistency ($r = 0.87$), validating the structural coherence of our absorption operationalization.
4. **Constructive forward look:** We identify decoder weight similarity and causal intervention as theoretically sound alternative approaches.

The remainder of this paper is organized as follows. Section 2 reviews background on feature absorption and existing detection methods. Section 3 describes our experimental methods. Section 4 presents our results, documenting UAD's failure, baseline comparisons, ablations, collision rate consistency, and root cause analysis. Section 5 discusses the theoretical implications and proposes alternative approaches. Section 6 concludes with a summary and call to action.
