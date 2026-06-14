# Testable Hypotheses (Updated Iteration 9)

## Confirmed Hypotheses

### H1: Metric Non-Transfer (CONFIRMED)

**Statement**: The Chanin absorption metric does not transfer from L1-ReLU SAEs on GPT-2 Small to JumpReLU SAEs on Gemma 2 2B. Shuffled-label controls will produce higher "absorption" than true labels.

**Expected outcome**: Shuffled/measured ratio > 1.5 for at least 3/5 domains.

**Actual outcome**: Shuffled exceeds measured in ALL 5 domains. Ratios: first-letter 4.7x, city-continent 6.9x, city-language 2.7x, animal-class 27.5x, city-country infinity (measured=0%, shuffled=10.3%).

**Status**: CONFIRMED. This is the paper's headline finding. The metric's thresholds are miscalibrated for JumpReLU activation geometry.

**Evidence**: Section 4.1 of paper; Table 2; Figure 1.

---

### H3: L0 Phase Transition (CONFIRMED)

**Statement**: Absorption rate declines monotonically with L0, exhibiting a phase transition between L0=40-80 where the rate drops by >50%.

**Expected outcome**: Monotonic decline with sharp drop in the L0=40-80 region.

**Actual outcome**: 42.85% (L0=22) -> 37.49% (L0=41) -> 14.39% (L0=82) -> 0.84% (L0=176). Spearman rho_s = -1.0. Cross-layer stable: L10 13.88%, L12 15.96%, L20 13.55% at L0=82 (CV < 10%). The sharpest decline is between L0=41 and L0=82, centered in the predicted 40-80 range.

**Status**: CONFIRMED. The most robust finding in the paper. L0 operating point -- not architecture -- is the primary control parameter for absorption severity.

**Evidence**: Section 5; Figure 3.

---

## Partially Supported Hypotheses

### H4: CMI Diagnostic (PARTIALLY SUPPORTED)

**Statement**: Features with lower conditional mutual information I(X; f_parent | f_child) are more susceptible to absorption. Specifically, CMI separates absorbed from non-absorbed letters.

**Expected outcome**: Significant negative correlation between CMI and absorption rate; absorbed letters have lower mean CMI.

**Actual outcome**: At d'=10 -- absorbed mean CMI 0.649 +/- 0.187 vs non-absorbed 0.861 +/- 0.258; Mann-Whitney U=28, p=0.045; Cohen's d=-0.924 (large effect). Spearman rho_s=-0.383 (p=0.059 uncorrected, p=0.236 Bonferroni-corrected). HOWEVER: sign reversal at d'>=20 (signal collapses), and rho=-0.67 between absorption rate and probe F1 creates a probe quality confound.

**Status**: PARTIALLY SUPPORTED -- exploratory only. The effect is genuine at d'=10 but marginal, does not survive multiple comparison correction, and collapses at higher subspace dimensions. Must be downgraded from "predicts" to "shows a directionally consistent negative association."

**Evidence**: Section 6. Limitation: Section 7.

**Critical next step**: CMI replication at L0=22 (where all probes F1=1.0, eliminating the confound) with pre-registered d'=10 as primary dimension.

---

## Falsified Hypotheses

### H2: Hierarchy-Driven Dominance (FALSIFIED)

**Statement**: At L0=22 (high sparsity), >80% of detected false negatives will be hierarchy-driven absorption rather than hedging.

**Expected outcome**: Hierarchy-driven fraction >80%.

**Actual outcome**: 1.4% hierarchy-driven (9/657), 98.6% hedging (648/657), 0% reconstruction error. At L0=176: 90% hierarchy-driven, 10% hedging -- but only 10 tokens remain FN at all.

**Status**: FALSIFIED. The most decisive negative result. Hedging dominates overwhelmingly. The 9 persistent core words (eight, lower, liked, offer, often, + 4 unnamed) are candidates for genuine competitive exclusion but represent only 0.75% of the vocabulary.

**Evidence**: Section 4.2; Figure 2.

**Critical implication**: The hedging classification is an upper bound (permissive definition). The tightened hedging experiment (Gate 1, Experiment 2) will check whether the SPECIFIC parent-associated latent fires at L0=176, distinguishing genuine hedging from compensatory resolution.

---

### H5: Cross-Domain Absorption Rates (FALSIFIED)

**Statement**: Absorption is measurable (rate > 10%) on at least 2 of 4 knowledge-domain hierarchies (city-country, city-continent, city-language, animal-class).

**Expected outcome**: Absorption rate 10-25% on city-country, lower on other domains.

**Actual outcome**: city-continent 6.49%, city-language 6.56%, animal-class 1.43%, city-country 0.0%. But ALL rates fall below their own shuffled controls (Table 2), making absolute rates uninterpretable as genuine absorption. The cross-domain measurements are the first ever reported, but serve as evidence of universal control failure rather than evidence of cross-domain absorption.

**Status**: FALSIFIED. The rates cannot be interpreted because the metric produces higher scores for randomized labels than for true hierarchical labels in every domain tested.

**Evidence**: Section 4.4; Table 2.

---

### H6: Cross-Domain Entity Matching (FALSIFIED)

**Statement**: Entity features identified in SAE latents will match between city and country representations.

**Expected outcome**: Positive matches between entity-level SAE features.

**Actual outcome**: Zero matching pairs found.

**Status**: FALSIFIED.

---

### H7: L1 vs JumpReLU Distribution Shape (FALSIFIED)

**Statement**: L1-ReLU SAEs will show a unimodal absorption distribution while JumpReLU SAEs will show bimodal distribution (due to hard thresholds).

**Expected outcome**: Different distribution shapes between architectures.

**Actual outcome**: Both architectures show bimodal absorption distributions. The hard-threshold mechanism does not produce qualitatively different distributional shapes.

**Status**: FALSIFIED.

**Note**: Section 5.3 presents formal tests (Hartigan dip, bimodality coefficients) comparing JumpReLU (Gemma 2B) vs L1-ReLU (GPT-2 Small), but these are confounded (different model sizes, architectures, training data). These formal tests should be compressed or moved to appendix.

---

## Pending Hypotheses (Iteration 9)

### H8: Activation Patching Recovers Parent Features (PENDING)

**Statement**: For at least 7 of 9 persistent core words, zeroing the child feature's activation will cause the parent-associated latent to recover (activation > 0), confirming genuine competitive exclusion.

**Expected outcome**: If 7+/9 recover: competitive exclusion confirmed at small scale. If <3: "all hedging" narrative validated.

**Falsification criterion**: <3 of 9 words showing parent recovery after child zeroing.

**Status**: PENDING -- the highest-value unexecuted experiment. Recommended for 3 consecutive reviews.

---

### H9: Tightened Hedging Rate (PENDING)

**Statement**: When hedging is classified strictly (requiring the SPECIFIC parent-associated latent to fire at L0=176, not just any latent), the hedging rate will remain >80% of the permissive rate.

**Expected outcome**: Strict rate >80% validates the 98.6% headline. Strict rate <50% requires narrative revision.

**Falsification criterion**: Strict hedging rate <50%.

**Status**: PENDING.

---

### H10: CMI at L0=22 (PENDING)

**Statement**: At L0=22 where all probes achieve F1=1.0, the CMI-absorption association (at pre-registered d'=10) will be significant at p<0.05 with Spearman rho < -0.3.

**Expected outcome**: rho < -0.3, p < 0.05 at d'=10. The probe quality confound (rho=-0.67) is eliminated because F1=1.0 for all 25 letters.

**Falsification criterion**: rho > -0.2 or p > 0.10.

**Status**: PENDING.

---

## Summary Table

| Hypothesis | Prediction | Actual | Status |
|---|---|---|---|
| H1: Metric non-transfer | Shuffled > measured in 3+/5 domains | ALL 5 domains, up to 4.7x | **CONFIRMED** |
| H2: Hierarchy dominance | >80% hierarchy-driven at L0=22 | 1.4% hierarchy, 98.6% hedging | **FALSIFIED** |
| H3: L0 phase transition | Monotonic decline, sharp in L0=40-80 | 42.9% -> 0.8%, CV<10% cross-layer | **CONFIRMED** |
| H4: CMI diagnostic | CMI separates absorbed/non-absorbed | d=-0.924 but p=0.236 Bonferroni | **PARTIAL** |
| H5: Cross-domain rates | >10% in 2+/4 domains | All below shuffled controls | **FALSIFIED** |
| H6: Entity matching | Positive entity matches | Zero matches | **FALSIFIED** |
| H7: Distribution shape | Unimodal L1 vs bimodal JumpReLU | Both bimodal | **FALSIFIED** |
| H8: Activation patching | 7+/9 parent recovery | -- | **PENDING** |
| H9: Strict hedging | Strict >80% of permissive | -- | **PENDING** |
| H10: CMI at L0=22 | rho < -0.3, p < 0.05 | -- | **PENDING** |
