# 6 Conclusion

## 6.1 Summary

We presented the first systematic evaluation of unsupervised absorption detection (UAD) on pre-trained SAEs. Our results are unambiguous: UAD fails catastrophically, achieving F1 = 0.00048---statistically indistinguishable from random sampling within clusters. Through ablation experiments, we showed that this failure is not due to any specific component of UAD's pipeline but to a fundamental structural mismatch: UAD's co-occurrence clustering detects features that fire on the same tokens, while absorption features are mutually exclusive at the token level by the logic of hierarchical concepts.

On the positive side, we validated collision rate---the Jaccard overlap of top-K activating features---as a robust proxy for true absorption rate, achieving Spearman $\rho = 0.869$ ($n = 56$, 95% CI $[0.780, 0.938]$). This provides the SAE interpretability community with a validated, computationally cheap metric for screening candidate absorption pairs.

## 6.2 Call to Action

We call on the SAE interpretability community to **abandon co-occurrence-based approaches for absorption detection**. The structural mismatch identified in this paper is not fixable by parameter tuning, larger datasets, or more sophisticated clustering algorithms. It is a category error: co-occurrence clustering asks "which features fire together?" when absorption requires answering "which features represent the same concept at different granularities?"

We propose three concrete next steps:

1. **Test decoder weight similarity.** The most immediate and computationally feasible alternative is to cluster features by decoder weight cosine similarity rather than co-occurrence. A pilot study on 100 feature pairs would cost minutes of GPU time and could rapidly establish whether this direction is promising.

2. **Develop hybrid pipelines.** Combine collision rate (for screening), decoder weight similarity (for refinement), and causal intervention (for validation) into a cascading detection system. This balances the strengths of each approach while managing computational cost.

3. **Expand ground truth.** Construct larger, more diverse ground truth datasets for absorption, spanning abstract concepts, visual hierarchies, and multimodal settings. A standardized benchmark would accelerate progress and enable fair comparison across methods.

Our work demonstrates that negative results, when accompanied by rigorous analysis and constructive forward looks, can be as valuable as positive findings. By identifying what does not work and why, we hope to direct the community's efforts toward approaches that stand a genuine chance of success.
