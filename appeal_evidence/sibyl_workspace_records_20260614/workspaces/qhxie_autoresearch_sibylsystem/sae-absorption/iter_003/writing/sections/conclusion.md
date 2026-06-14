# 6. Conclusion

We present three contributions toward understanding and detecting feature absorption in Sparse Autoencoders.

**Mechanistic.** A controlled OMP oracle experiment shows that replacing the feedforward SAE encoder with an optimal $K$-sparse solver at matched sparsity produces zero reduction in feature absorption rates across all tested configurations. This decisively falsifies the amortization gap hypothesis as a dominant cause of absorption and provides the first direct empirical support for the sparsity landscape account \citep{tang2025partial}. The practical implication is clear: practitioners seeking to reduce absorption should focus on training-time interventions, not inference-time encoder improvements.

**Methodological.** Encoder weight norm ($\|\mathbf{w}_{e,j}\|_2$) predicts absorbed latents with AUROC $= 0.757$--$0.837$ across Standard and TopK SAE architectures, significantly outperforming EDA (DeLong $p = 0.0012$). Jaccard co-occurrence score provides an independent signal (AUROC $= 0.721$, $\rho = 0.044$ with EncNorm). Both metrics require only SAE weights or activation statistics — no probe directions or integrated gradients — making them scalable to large SAE dictionaries.

**Practical.** 67\% of absorbed features in a 24k Standard SAE have geometric counterparts in a wider 32k TopK SAE, suggesting dictionary expansion partially addresses absorption. The remaining 33\% require training-time structural changes to the dictionary or objective. EncNorm does not distinguish which absorbed features benefit from width expansion, motivating the combination of EncNorm with structural taxonomy analysis for targeted remediation.

**Future work.** The most pressing open issues are: (1) resolving the hook confound in cross-architecture comparison with a matched-hook experiment; (2) obtaining a larger gold-label dataset ($n_\text{pos} \gg 18$) to narrow detection AUROC confidence intervals; (3) testing whether absorption generalizes to non-spelling-task feature hierarchies using probes trained on the target model; and (4) empirically validating whether masked regularization \citep{narayanaswamy2026masked} achieves measurable absorption reduction consistent with our H2 falsification.

The core message is that feature absorption in SAEs is primarily a training problem, and the interpretability community's diagnostic and remediation tooling should be oriented accordingly.
