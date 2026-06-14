# Why Co-occurrence Clustering Cannot Detect Feature Absorption in Sparse Autoencoders

## Abstract

Sparse Autoencoders (SAEs) have emerged as the dominant tool for extracting interpretable features from neural network activations, but a critical failure mode---feature absorption---threatens their reliability. Feature absorption occurs when parent features representing broad concepts are suppressed by child features representing more specific sub-concepts. All existing detection methods require supervised ground truth, limiting scalability. Unsupervised Absorption Detection (UAD) proposes co-occurrence clustering as a training-free alternative. We present the first systematic evaluation of UAD on pre-trained SAEs. Our finding: UAD fails catastrophically, achieving F1 = 0.00048---numerically identical to random sampling within clusters. We identify the root cause: absorption features are mutually exclusive at the token level. Co-occurrence clustering detects features that fire together, but absorption features fire on different tokens. This is a structural mismatch, not a parameter-tuning problem. On the positive side, we demonstrate that collision rate---the Jaccard overlap of top-K activating features---exhibits strong internal consistency as an operationalization of absorption (Spearman r = 0.869, n = 56 pairs, 95% CI [0.780, 0.938]). We conclude with concrete proposals for alternative approaches based on decoder weight similarity and causal intervention.

## Full Paper

The complete paper is available as a compiled LaTeX PDF at `writing/latex/main.pdf`.

The LaTeX source is at `writing/latex/main.tex`.

## Key Results

- **UAD F1**: 0.00048 (1 TP / 4,155 detected pairs)
- **Same-cluster random F1**: 0.00048 (identical to UAD)
- **Collision rate Spearman r**: 0.869 (n=56, 95% CI [0.780, 0.938])
- **K-means best variant**: F1 = 0.0037 (85.7% recall, 0.185% precision)

## Target Venue

ICBINB ("I Can't Believe It's Not Better") Workshop or NeurIPS/ICML Workshop on Mechanistic Interpretability.
