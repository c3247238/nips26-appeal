# Experiment Critique -- Iteration 10

## Overall Assessment

The experimental program is mature and covers considerable ground: cross-domain absorption measurement across 4 hierarchies, activation patching for causal validation, probe degradation ablation, decoder entanglement analysis, and rate-distortion predictor testing. The probe degradation ablation (H10) is the standout experiment of this iteration, producing a statistically significant curve (R^2=0.777, p=0.009) with perfect monotonicity. First-letter causal evidence (d=1.33) is rock-solid across iterations.

However, five experimental issues threaten the paper's credibility.

---

## Critical Issue 1: Table 3 Confidence Intervals Are Mathematically Impossible

In 5 of 7 rows of Table 3, the 95% CI lower bound exceeds the point estimate. Example:

- F1=0.70: absorption=36.1%, CI=[37.9%, 42.1%]
- F1=0.75: absorption=35.3%, CI=[39.2%, 43.4%]
- F1=0.80: absorption=34.4%, CI=[37.8%, 41.8%]

A confidence interval that does not contain its point estimate is a statistical impossibility. This occurs because the point estimates use per-token aggregation (11,725 tokens) while the CIs appear to use per-word bootstrap resampling (2,345 words). The per-word rates are systematically higher than per-token rates because words with more prompt contexts get downweighted in per-word aggregation.

**Impact:** A reviewer will flag this within 5 minutes of reading Table 3. It immediately undermines trust in ALL numbers in the paper.

**Fix:** Recompute CIs using per-token bootstrap, or present matched per-word point estimates with per-word CIs.

---

## Critical Issue 2: Cross-Domain Patching Data Provenance

The sign reversal from d=-0.91 (FAILED pilot) to d=+1.50 (corrected FULL) for city-continent activation patching is the paper's most dangerous vulnerability. The data comes from `iter_009/exp/results/full/activation_patching_crossdomain_full.json`, but the phase0_data_integrity task that would have verified it was "skipped for speed."

This matters because:
1. The pilot was classified as FAILED with a plausible explanation (negative recovery = city-continent patching makes things worse).
2. The correction to d=+1.50 was attributed to a code bug, but no independent verification was performed.
3. The paper's entire "universal mechanism" claim (Contribution 2) rests on this corrected data.
4. City-language patching (d=0.75) was similarly corrected from buggy pilot data.

**Impact:** If a reviewer discovers the iteration history, the unverified correction from a failed experiment to a strong positive is devastating.

**Fix:** Run a 20-entity spot-check for city-continent: randomly sample 20 entities, re-run the full patching pipeline, verify recovery rate is consistent with 61.9%. Takes 0.5 GPU-hours. This is the single highest-ROI experiment remaining.

---

## Critical Issue 3: Probe Degradation Extrapolation

The probe degradation curve injects Gaussian noise into first-letter binary probe weights, then extrapolates to predict absorption rates for RAVEL multi-class probes. These are fundamentally different:

| Property | First-letter probes | RAVEL probes |
|----------|-------------------|-------------|
| Type | 26 binary (one-vs-rest) | Multi-class logistic |
| Classes | 25 (balanced) | 6-77 (imbalanced) |
| F1 range | 1.0 (all layers) | 0.41-0.87 |
| Degradation mode | Gaussian noise on weights | Real training difficulty |
| Failure mode | Random misclassification | Systematic class confusion |

The claim that city-continent "matches the curve within 0.6 pp" implicitly assumes that the functional relationship between probe F1 and absorption rate is domain-independent. This assumption is not tested.

**Impact:** The skeptic (result debate) correctly identified this as the paper's methodological Achilles' heel. A domain expert reviewer will raise this immediately.

**Fix:** Either (a) run city-continent probe degradation (1 GPU-hour) to validate that the slope is similar, or (b) reframe Section 4.6 as "probe quality is a confound in the right direction and of plausible magnitude" rather than "probe quality quantitatively explains city-continent absorption."

---

## Major Issue 4: Three Different First-Letter Absorption Rates

The workspace contains three different values for the same measurement (first-letter, L24, 16k JumpReLU):
- 27.1% (iter_009, per-word aggregation -- used as paper's canonical value)
- 21.6% (iter_010 FULL probe degradation F1=1.0 control, per-token aggregation)
- 34.5% (referenced in earlier iterations, per-unique-word aggregation)

The 21.6% control falls BELOW the 95% CI [26.3%, 34.7%] of the 27.1% baseline. This means the probe degradation experiment's internal control does not reproduce the paper's headline number for first-letter absorption.

**Impact:** A reader comparing Table 2 (27.1%) with Table 3 (21.6% at F1=1.0) will wonder why the same measurement gives different results in the same paper.

**Fix:** Use one aggregation method everywhere. Add a footnote to Table 3 if the aggregation method differs from Table 2, explaining why and reporting the magnitude of the difference.

---

## Major Issue 5: City-Country Excluded from Patching

The paper presents activation patching for first-letter, city-continent, and city-language, but not city-country -- the hierarchy with the HIGHEST absorption rate (45.1%). The exclusion is never acknowledged or explained.

Possible reasons (not stated in the paper):
- Probe quality too low (F1=0.73) to trust patching results
- Not enough entities per country for reliable per-entity recovery rates
- Time/resource constraints

**Impact:** A reviewer will ask: "If competitive exclusion is 'universal,' why wasn't it tested on the hierarchy with the most absorption?"

**Fix:** Add an explicit paragraph explaining the exclusion, or run city-country patching (0.5 GPU-hours).

---

## Strengths

1. **H10 probe degradation (R^2=0.777, rho=-1.0)** is a genuinely strong experiment. The 7-level curve with perfect monotonicity is convincing. The improvement from pilot (R^2=0.077) to FULL (R^2=0.777) demonstrates that additional data points (F1=0.75 and 0.95) were critical. This is the paper's best new experiment.

2. **First-letter activation patching (d=1.33)** is reproducible across iterations and constitutes the first interventional evidence for competitive exclusion. This is bulletproof.

3. **Decoder entanglement cross-hierarchy consistency** (6.16 nats first-letter vs 3.98 nats city-continent, both 100% above all thresholds) is a clean result with acknowledged circularity.

4. **Rate-distortion at scale (131 pairs)** definitively closes the door on this approach: all individual predictors reversed direction. This is a valuable negative result with high confidence.

5. **Quadruple correlational failure** (GAS, CMI, T(G), rate-distortion) is a coherent story that establishes a methodological boundary.
