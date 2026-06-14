# Experiment Critique

## Overall Assessment: 6/10

The experimental design is methodologically rigorous in its controlled comparison protocol, but execution gaps---missing NoBN controls, absent ImageNet validation, and figure-data mismatches---significantly weaken the empirical claims.

## Strengths

### Controlled Comparison Protocol

All methods share identical base hyperparameters, optimizer internals, and training infrastructure. The pluggable WDModulator interface ensures differences are isolated to the phi function. This is exactly how systematic comparisons should be done. The paper correctly notes the tradeoff (fixed hyperparameters may disadvantage some methods) in Section 6.4.

### Multi-Seed with Proper Statistics

Three seeds with paired t-tests, Bonferroni correction, effect sizes, and TOST equivalence testing. This is rigorous. The explicit power analysis (80% power to detect 0.7% effects) is refreshingly honest.

### Two Optimizers as Boundary Test

Testing both AdamW and SGD is well-motivated: it provides a natural control where the conjecture predicts different outcomes. The SGD results confirming the boundary condition strengthen the AdamW null result.

## Critical Issues

### 1. Missing NoBN Control (Confounding)

Both architectures use batch normalization. The VGG-16-BN results show invariance UNDER SGD, suggesting BN---not AdamW---may be the dominant mechanism. Without NoBN experiments, the paper cannot distinguish its claimed mechanism (AdamW adaptive scaling) from an alternative (BN implicit norm control). This is a classic confound.

The lessons_learned.md confirms NoBN data exists from iter_005 but is not in the paper. This is zero-compute work that resolves the paper's central ambiguity.

### 2. CIFAR-Only Scale

CIFAR-10 at 90% accuracy and CIFAR-100 at 63% accuracy are not compelling settings for evaluating WD methods. The network may simply be near the architecture's performance ceiling on CIFAR-10, compressing all methods into a narrow band. The methods that claim improvements (CWD, AlphaDecay) were evaluated at LLM/ImageNet scale. Testing the conjecture only at CIFAR scale is testing it where it is most likely to hold trivially.

### 3. BEM Value Inconsistency

Table 6 shows half_lambda BEM = 0.500 (correct). Figure 4 (fig3_bem_vs_accuracy.png) visually plots half_lambda at BEM approximately 0.0, clustered with constant. If the figure is correct and the table is wrong, half_lambda is actually applying the same effective WD as constant, meaning the implementation has a bug. If the table is correct and the figure is wrong, the figure needs regeneration.

### 4. AdamW + VGG-16-BN Results Missing

The paper claims 105 experiments (7 methods x 3 seeds x [2 datasets x 2 optimizers for ResNet-20 + 1 dataset x 1 optimizer for VGG-16-BN]). VGG-16-BN is only shown under SGD. Were AdamW + VGG-16-BN runs conducted? If yes, they must be reported. If no, the experiment count needs correction. A complete 2x2 factorial (optimizer x architecture) would strengthen the mechanistic analysis.

### 5. Single Architecture Per Optimizer for Full Comparison

ResNet-20 gets both AdamW and SGD results on both datasets. VGG-16-BN gets only SGD on CIFAR-10. This is an incomplete factorial design. The missing cells (VGG-16-BN + AdamW, VGG-16-BN + CIFAR-100) limit the conclusions about optimizer vs architecture interactions.

## Major Issues

### 6. PMP-WD Data Provenance

PMP-WD appears in Figures 3, 8, and 9 but not in any results table. Was PMP-WD actually run? If yes, its results should be in Table 2 (or a separate table). If no, the figures use fabricated data points. The lessons_learned.md mentions "PMP-WD as efficiency demonstration" and "CIFAR-100 provenance mismatch: PMP-WD uses iter_006." This suggests PMP-WD WAS run but was dropped from the paper text without updating figures.

### 7. No Computational Cost Comparison

The paper evaluates accuracy but not training cost. Some WD methods (SWD with gradient-norm computation, CWD with sign-alignment check) have overhead. A wall-clock time comparison would complete the practical evaluation. If all methods are equivalent in accuracy AND some are more expensive, that strengthens the "just use constant WD" recommendation.

### 8. TOST Partial Equivalence

TOST equivalence confirmed for only 6 of 12 method-dataset comparisons at delta=1.0%. The remaining 6 are inconclusive (neither significantly different nor confirmed equivalent). The paper should report which specific comparisons achieved equivalence vs which are inconclusive.

## Minor Issues

- No learning curve plots (train loss vs epoch). These would show whether methods differ in optimization trajectory even if final accuracy converges.
- Gradient-weight cosine similarity logged per-100-steps but not visualized in the paper.
- No per-layer analysis of phi values: does CWD modulate different layers differently?
