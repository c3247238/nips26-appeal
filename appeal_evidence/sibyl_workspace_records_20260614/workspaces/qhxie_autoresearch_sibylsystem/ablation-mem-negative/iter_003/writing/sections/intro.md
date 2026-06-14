# 1 Introduction

Sparse Autoencoders (SAEs) have emerged as the dominant tool for extracting interpretable features from neural network activations [Bricken et al., 2023; Cunningham et al., 2023]. By learning an overcomplete dictionary of sparse, semantically meaningful features, SAEs enable researchers to identify what specific neurons or directions in a model's representation space encode. However, a critical failure mode---feature absorption---threatens the reliability of SAE-based interpretability.

**Feature absorption** occurs when a parent feature, which should represent a broad concept, is suppressed by child features that represent more specific sub-concepts [Chanin et al., 2024]. For example, a feature encoding "animal" may be suppressed by features encoding "dog," "cat," and "bird," making the parent feature appear non-existent or misleadingly weak. This creates dangerous interpretability illusions: researchers may conclude that a model lacks a concept when in fact it is simply absorbed by more specific features.

Detecting absorption is essential for accurate SAE interpretation. All existing detection methods, however, require **supervised ground truth**---known parent features and trained probes [Chanin et al., 2024; Karvonen et al., 2025]. This supervised requirement means absorption is only detectable for concepts where we already have labels, severely limiting scalability. An unsupervised detection method would unlock absorption analysis at scale, across the full SAE dictionary.

Unsupervised Absorption Detection (UAD) [Chanin et al., 2024] proposes co-occurrence clustering as a training-free solution. The core hypothesis is straightforward: features that frequently co-occur on the same tokens may have hierarchical relationships, with parent features being "absorbed" by their children. UAD's pipeline filters dead features, computes phi coefficients between feature pairs, hierarchically clusters features, and extracts candidate absorption pairs from within clusters---all without any labeled data.

**Our central question:** Does co-occurrence clustering actually detect feature absorption?

**Our answer:** No. UAD fails catastrophically, achieving an F1 score of 0.00048---statistically indistinguishable from random sampling within clusters. We identify the root cause: absorption features are **mutually exclusive at the token level**. A feature that activates on "three" never activates on "four," because these tokens represent different child concepts. Co-occurrence clustering detects features that fire *together*, but absorption features fire on *different* tokens. This is a structural mismatch, not a parameter-tuning problem.

Our investigation is not purely negative. We validate **collision rate**---the Jaccard overlap of top-K activating features---as a robust proxy for true absorption rate, achieving Spearman $\rho = 0.869$ ($n = 56$ pairs, 95% CI $[0.780, 0.938]$). This positive result provides a validated metric for future absorption research.

**Our contributions:**
1. **Empirical falsification:** We demonstrate that UAD achieves F1 = 0.0005, identical to a random baseline, on pre-trained SAEs.
2. **Root cause identification:** We prove that token-level mutual exclusivity makes co-occurrence clustering fundamentally incompatible with hierarchical absorption detection.
3. **Validated proxy:** We show that collision rate correlates strongly with true absorption rate ($\rho = 0.87$), providing a reliable metric for screening candidate absorption pairs.
4. **Constructive forward look:** We identify decoder weight similarity and causal intervention as theoretically sound alternative approaches.

The remainder of this paper is organized as follows. Section 2 reviews background on feature absorption and existing detection methods. Section 3 describes our experimental methods, including UAD's pipeline, ground truth definition, and collision rate computation. Section 4 presents our results, documenting UAD's failure, baseline comparisons, ablations, collision rate validation, and root cause analysis. Section 5 discusses the theoretical implications and proposes alternative approaches. Section 6 concludes with a summary and call to action.
