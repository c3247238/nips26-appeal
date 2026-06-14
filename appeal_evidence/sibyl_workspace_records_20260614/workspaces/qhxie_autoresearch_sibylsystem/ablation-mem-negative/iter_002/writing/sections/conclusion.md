# 6. Conclusion

We presented the first systematic analysis of why unsupervised absorption detection fails in Sparse Autoencoders. Our key findings are:

1. **Co-occurrence clustering is no better than random**: UAD achieves F1 = 0.007, indistinguishable from random selection (F1 = 0.0075) on GPT-2 Small.

2. **The failure is conceptual, not implementational**: Ablations across clustering methods, feature filters, and layers all produce near-zero precision. The problem is that co-occurrence detects correlation, while absorption requires detecting suppression---fundamentally different statistical phenomena.

3. **Mitigation remains feasible**: DFDA improves per-pair residual MSE by 21.2\% when absorbed pairs are known, suggesting inference-time compensation is viable even when detection is not.

These findings have implications for the SAE community: rather than pursuing unsupervised detection of absorption after training, research should focus on either (1) supervised detection with validated ground truth, or (2) preventive architectures that eliminate absorption during training. The distinction between correlation and suppression is not merely semantic---it determines whether a detection method can possibly work.

Our conclusion: **absorption detection is a causal inference problem, not a clustering problem.** The path forward requires interventions, not observations.
