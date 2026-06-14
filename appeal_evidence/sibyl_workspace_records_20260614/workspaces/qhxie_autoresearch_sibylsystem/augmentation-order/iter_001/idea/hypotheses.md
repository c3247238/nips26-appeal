# Testable Hypotheses (Revised — Iteration 1)

**Last updated**: 2026-04-03 (post-pilot evidence integration)  
**Status**: Hypotheses H1–H4 revised based on Tier 0 and full Tier 1 pilot data. H5 replaced by H5b and H6.

---

## H1 — Primary Factorial Effect

**Statement**: For a fixed set of K augmentation operations applied per image, the permutation of operations produces a statistically significant effect on classification accuracy.

**Specific prediction**: The accuracy gap between the best and worst ordering exceeds 0.3% on at least 2 of 4 architecture-dataset combinations (ResNet-18/CIFAR-10, ResNet-18/CIFAR-100, ViT-Small/CIFAR-10, ViT-Small/CIFAR-100), confirmed by paired t-test at p < 0.05 after Bonferroni correction.

**Pilot evidence**: 3/4 blocks exceed 0.5% threshold at preliminary scale (1 seed, 10 epochs, 100-sample subsets). Maximum spread: 2.32% (ViT/CIFAR-10). These are preliminary indications only; 5 seeds and 200 epochs required for statistical confirmation.

**Measurement**: Max-min accuracy spread across K! orderings; paired t-test between best and worst ordering (same seed, different ordering eliminates between-seed variance).

**Expected outcome (full scale)**: Effect persists in 3–4 blocks, with ViT/CIFAR-10 showing the largest effect. Full-training effects may be smaller than pilot (optimization absorbs some ordering sensitivity) or larger (effects compound over more epochs).

**Falsification**: Spread < 0.2% on all 4 blocks with 95% CI of paired difference including zero, in the 5-seed full experiment.

---

## H2 — Architecture Differential (CNN vs. ViT Sensitivity)

**Statement**: ViTs exhibit larger absolute ordering sensitivity than CNNs across matched arch-dataset combinations, due to the interaction between patchification and spatial transform ordering.

**Specific prediction**:
- ViT-Small/CIFAR-10 spread > ResNet-18/CIFAR-10 spread (matched dataset, different architecture)
- ViT-Small/CIFAR-100 spread > ResNet-18/CIFAR-100 spread
- The ordering × architecture interaction is statistically significant (two-way ANOVA, p < 0.05, partial eta-squared > 0.05)

**Mechanistic prediction**: When geometric transforms (crop) are applied to an image before patch tokenization, the resulting patches contain consistent spatial content. When crop is applied after other operations, patches boundary-effects interact with those prior transforms, creating higher variance in patch-level representations. CNNs with local receptive fields are less sensitive to this because they aggregate spatially at each layer.

**Pilot evidence**: ViT/CIFAR-10 spread (2.32%) = 2.4× ResNet-18/CIFAR-10 spread (0.96%). ViT/CIFAR-100 spread (0.25%) compared to ResNet-18/CIFAR-100 (0.88%) — unexpectedly reversed in the low-signal block. Full-scale data needed.

**Measurement**: Two-way ANOVA on accuracy across orderings and architectures. Report interaction effect size (partial eta-squared). Requires n ≥ 2 per cell (5 seeds ensures this).

**Falsification**: Both architectures show the same sign and similar magnitude of ordering preference across all 4 blocks. Interaction ANOVA p > 0.1.

**NOTE (hypothesis consistency)**: H2 refers strictly to the architecture sensitivity differential, not to the "reversibility-sorted wins" finding (which is H4). This distinction was blurred in the prior iteration's analysis scripts.

---

## H3 — Feature-Space NC Predicts Effect Size (Revised from Pixel-Space)

**Statement**: The Wasserstein Non-Commutativity NC_2 between transform pairs, computed in the learned feature space of a trained ResNet-18, predicts the magnitude of accuracy sensitivity to their relative ordering.

**Pilot falsification (Tier 4a)**: Pixel-space NC_2 (SWD proxy, 100 samples, 100 projections, 3072-D space) yields rho = -0.20, p = 0.68 — falsified. Root cause: 100 samples in 3072-D is statistically meaningless (curse of dimensionality). Moreover, pixel-space divergence does not capture how ordering alters the gradient landscape seen by the optimizer.

**Revised prediction**: Feature-space NC_2 — computed as SWD between P_AB and P_BA projected through the penultimate layer of a pretrained ResNet-18 (512-D, 10k samples, 1000 projections) — will show:
- Spearman rho > 0.5 with accuracy ranking across orderings (p < 0.05 via permutation test)
- Cross-category pairs (Crop-CJ, Crop-Flip) will have higher feature-space NC_2 than within-category pairs if the 3-operation set is extended to include a second photometric operation

**Pilot alignment**: Individual transform SWD vs. identity (Crop: 0.096, CJ: 0.027, Flip: 0.022) correctly ranks Crop as most destructive. Feature-space NC_2 should amplify this signal by filtering through task-relevant representations.

**Measurement**: SWD computed in 512-D feature space with 10k augmented image pairs, 1000 random projections. Spearman rho between NC_2 ranking and accuracy-difference ranking. Also: compute pixel-space NC_2 with proper sample size (10k, 1000 projections) for direct comparison.

**Falsification**: Feature-space NC_2 Spearman rho < 0.3. If this is falsified, the theoretical framework requires fundamental revision: ordering effects are not captured at any fixed representation level and must be analyzed dynamically (e.g., via gradient flow analysis).

---

## H4 — DPI Reversibility Principle (Revised and Weakened)

**Statement**: Orderings that place high-reversibility transforms before low-reversibility transforms achieve equal or higher accuracy than orderings that place low-reversibility transforms first.

**Corrected reversibility ranking**:
- **Highest reversibility (zero info cost)**: RandomHorizontalFlip — strictly bijective (its own inverse), eta = 1.0 by DPI
- **Medium reversibility (low info cost)**: ColorJitter — pointwise, approximately invertible if parameters are known; eta < 1 but close
- **Lowest reversibility (high info cost)**: RandomCrop — discards pixels irreversibly, eta << 1

**Corrected DPI prediction**: The optimal ordering places Flip first (zero info cost), then CJ (low cost), then Crop (high cost). This gives Flip→CJ→Crop or Flip→Crop→CJ. **This is consistent with the pilot winner**: Flip→Crop→CJ wins in 2/4 blocks; Flip→CJ→Crop wins in 1/4 blocks. The prior proposal's prediction of CJ-first was based on an incorrect reversibility classification of Flip as "medium."

**Pilot evidence**: Flip→Crop→CJ wins in 2/4 blocks (including the highest-spread ViT/CIFAR-10 block). CJ→Flip→Crop (incorrectly predicted by prior proposal) wins in 0/4 blocks as front-runner in Tier 1. The corrected prediction (Flip-first) is supported.

**Measurement**: Paired comparison between reversibility-sorted ordering (Flip→CJ→Crop or Flip→Crop→CJ) and conventional (Crop→Flip→CJ) across 5 seeds. Also: Spearman correlation between InfoNCE MI estimate (using properly trained encoder) and accuracy ranking.

**Falsification**: Neither Flip-first ordering outperforms conventional (Crop→Flip→CJ) on any of the 4 blocks in the full experiment. If this is falsified, the DPI principle is inapplicable to this context (optimization dynamics dominate information-theoretic considerations).

---

## H5b — Category-Level Interleaved Ordering Dominates

**Statement**: Interleaved photometric-geometric ordering (P→G→P→G→P→G) outperforms block-level ordering (all-geometric-first or all-photometric-first) on the 6-operation pipeline.

**Pilot evidence (Tier 2, single pilot)**: Interleaved P→G achieves 0.2939 vs. all-geometric-first at 0.2038 on CIFAR-10/ResNet-18 pilot (5k samples, 10 epochs, 1 seed). Spread of 9.01% is the largest observed effect across all experiments. This is a striking finding that requires confirmation.

**Mechanism**: Interleaved ordering prevents any transform category from dominating the distribution shift. Specifically:
- All-geometric-first applies 3 geometric distortions before any color normalization, potentially destroying spatial structure the model relies on for color-sensitive classes
- All-photometric-first applies 3 color distortions before any spatial normalization, potentially distorting color statistics the model relies on for spatially-sensitive classes
- Interleaved ordering distributes information loss across both channels alternately, yielding a more uniform effective data distribution

**Measurement**: Accuracy of 5 canonical orderings across 2 architectures × 2 datasets × 5 seeds. Paired t-test (interleaved P→G vs. all-geometric-first, all-photometric-first separately). Effect size Cohen's d.

**Priority**: Requires a Tier 2 confirmation pilot (30 epochs, 2 seeds, 2 arch-dataset blocks) before full Tier 2 investment. If the 9.01% pilot effect does not survive even modest training, the Tier 2 budget may be redirected.

**Falsification**: Interleaved P→G ordering is not significantly better than all-geometric-first on any of the 4 blocks in the full experiment (p > 0.05, |d| < 0.2 for all).

---

## H6 — Inverted-U Magnitude Interaction (New Hypothesis)

**Statement**: Ordering sensitivity (best-minus-worst spread across orderings) follows an inverted-U shape with augmentation magnitude: low magnitude yields small spread, medium magnitude yields the largest spread, and high magnitude yields near-zero spread due to augmentation saturation.

**Pilot evidence (Tier 3, ResNet-18/CIFAR-100, 1 seed, 10 epochs)**:
- M5 (low): spread = 0.35%
- M9 (medium): spread = 0.88%
- M14 (high): spread = 0.00% (both orderings converge to 0.245)

**Mechanism**: At low magnitude, both orderings cause minimal distortion and the distributions P_AB, P_BA are nearly identical. At medium magnitude, non-commutativity effects are large enough to matter but small enough that some orderings are clearly better. At high magnitude, both orderings cause such severe distortion that the training signal is approximately equivalent — the model adapts to the hard augmentation regardless of sequence.

**Measurement**: Spread across orderings at M=5, 9, 14 for best/worst orderings from Tier 1, across 2 architectures × CIFAR-100 × 5 seeds. Quadratic regression of spread on magnitude, testing for negative quadratic term.

**Falsification**: Spread is monotonically increasing with magnitude (no inverted-U). Or: M14 spread is larger than M9 spread in the 5-seed full experiment.

---

## Null Hypothesis (N0)

**Statement**: Augmentation ordering effects are real at the pixel level (transforms are non-commutative) but are absorbed by SGD stochasticity and do not produce meaningful accuracy differences in practice.

**Current pilot assessment**: N0 is *unlikely* given the pilot evidence (2.32% and 9.01% spreads), but cannot be formally rejected without full-scale statistical tests.

**Value if confirmed**: Formally validates the implicit assumption in RandAugment and TrivialAugment. Provides evidence-based justification for the community's convention. Closes a documented research gap with a principled negative result.

---

## Measurement Summary

| Hypothesis | Primary Measurement | Statistical Test | Threshold | Pilot Status |
|---|---|---|---|---|
| H1 | Max-min accuracy spread | Paired t-test, Bonferroni | Spread > 0.3%, p < 0.05 in ≥2/4 blocks | Preliminary: 3/4 blocks above 0.5% |
| H2 | ViT spread > CNN spread | Two-way ANOVA | Interaction p < 0.05, eta-sq > 0.05 | Preliminary: ViT 2.32% > CNN 0.96% (CIFAR-10) |
| H3 | Feature-space NC_2 vs. accuracy | Spearman + permutation test | rho > 0.5, p < 0.05 | Pixel-space FALSIFIED; feature-space pending |
| H4 | Flip-first wins vs. conventional | Paired t-test across 5 seeds | Flip-first better in ≥2/4 blocks | Preliminary: Flip→Crop→CJ wins 2/4 blocks |
| H5b | Interleaved P→G vs. block ordering | Paired t-test | Interleaved better than all-geo in ≥3/4 blocks | Preliminary: 9.01% spread (1 seed, needs confirmation) |
| H6 | Inverted-U spread vs. magnitude | Quadratic regression | Negative quadratic term, p < 0.05 | Preliminary: M9 > M5 and M9 > M14 (1 seed) |
| N0 | All spreads < noise floor | 95% CI of paired diff includes 0 | All p > 0.1 | Inconsistent with pilot — unlikely |

**Critical note**: All "pilot status" entries reflect 1 seed, 10 epochs, 100-sample subsets and CANNOT be used for final hypothesis verdicts. Full-scale runs with n=5 seeds and 200 epochs are required.
