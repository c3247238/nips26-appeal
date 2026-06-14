# 1. Introduction

Sparse autoencoders (SAEs) have become the de facto standard for mechanistic interpretability of language models, decomposing high-dimensional residual stream activations into sparse, interpretable features [Bricken et al., 2023; Cunningham et al., 2023]. Yet a critical failure mode undermines their reliability: feature absorption, where broad parent features are suppressed by specific child features when both co-occur, creating arbitrary false negatives that escape detection [Chanin et al., 2024].

The core barrier is methodological. All existing absorption detection methods require ground-truth parent feature labels and supervised probe directions [Chanin et al., 2024; Karvonen et al., 2025]. This supervised requirement creates a paradox: absorption can only be detected for concepts we already know, precisely where SAEs are least needed. For unknown features---the vast majority of any SAE dictionary---no detection mechanism exists.

We address this gap with UAD (Unsupervised Absorption Detection), the first method to detect absorbed feature pairs without any labeled data or probe directions. UAD operates on a simple structural insight: absorbed parents exhibit anomalous co-occurrence patterns. They fire primarily when their child features fire, but rarely independently. This signature is detectable via hierarchical clustering on phi coefficient co-occurrence matrices, requiring only unlabeled text and a pre-trained SAE.

**Research Questions.** This paper investigates three questions:
- **RQ1**: Can feature absorption be detected without ground-truth parent features or supervised probe directions?
- **RQ2**: Does a co-occurrence clustering approach generalize across layers and model scales?
- **RQ3** (preliminary): Can absorbed parent activations be recovered at inference time without retraining the SAE?

**Contributions.** Our contributions are:
1. **UAD**: The first unsupervised absorption detection method, achieving F1 = 0.725 with perfect recall (precision = 0.569) on GPT-2 Small layer 8, requiring no labeled hierarchies or probe directions.
2. **Cross-layer validation**: UAD achieves mean F1 = 0.561 across layers 4, 8, 10, demonstrating layer-dependent performance with layer 8 optimal.
3. **Multi-seed reproducibility**: Identical F1 = 0.725 across seeds 42, 123, 456, demonstrating deterministic behavior on fixed SAEs.
4. **DFDA** (preliminary): A training-free compensation method using 193 parameters per pair, reported with explicit metric limitations.

UAD detects all 29 Chanin-supervised collision cases among 51 same-cluster pairs on GPT-2 Small layer 8. The 43% false positive rate reflects a detection tool requiring post-hoc filtering, not a finished solution. Nevertheless, UAD eliminates the supervision bottleneck that has constrained all prior absorption detection methods, opening absorption detection to any SAE without prior knowledge of feature hierarchies.

<!-- FIGURES
- None
-->
