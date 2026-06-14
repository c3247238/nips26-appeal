# Supervisor Review: Phi Invariance Conjecture (Iteration 6/Current)

## Score: 6.5 / 10 (Borderline Reject)

**Verdict: CONTINUE** -- the research direction has genuine merit but critical soundness and evidence gaps prevent acceptance.

**Previous score: 7.0 (iter_005).** This iteration scores LOWER despite several improvements because cross-validation against raw data revealed a previously undetected data provenance violation.

---

## Executive Summary

The paper makes a genuine contribution to the weight decay literature: the Phi Modulator Framework provides a clean organizational abstraction, the diagnostic metrics (BEM, CSI, AIS) are useful, and the "weight decay illusion" finding is compelling and well-framed. The addition of PMP-WD implementation, VGG-16-BN integration, certified band visualization, and Theorem 2 validation scatter plot all represent real progress from iter_005.

However, four critical issues prevent acceptance:

1. **Data provenance mismatch**: Table 2 uses iter_003 data (constant=90.13) but PMP-WD uses iter_006 data (90.29). The same method (constant AdamW CIFAR-10) yields 89.80 in iter_006 -- a 0.33pp cross-iteration shift that EXCEEDS the paper's claimed 0.25pp inter-method spread.

2. **Lyapunov certificate is empirically vacuous**: V_t increases throughout training, contradicting the V_{t+1} <= V_t guarantee of Theorem 1.

3. **Theorem 2 validation fails statistical significance**: rho=-0.379, p=0.121, yet the paper claims this "validates the cumulative alignment bound's predictive power."

4. **No ImageNet experiments**: Project constraints require ImageNet; all major WD papers include it.

---

## Dimension Scores

### Novelty & Significance: 7/10

The Phi Modulator Framework is a clean organizational contribution. The four modulation axes (temporal, directional, spatial, target-norm) provide a genuinely useful taxonomy. The "weight decay illusion" framing is compelling and publishable.

However, the framework is ultimately notational -- expressing methods as lambda_eff = phi * lambda_base is straightforward once stated. The deeper novelty should come from the theorems, but the Lyapunov certificate is empirically vacuous, Theorem 2 validation fails significance, and PMP-WD reduces to random binary masking on BN architectures.

The diagnostic metrics are mixed: BEM is clean and useful; AIS has a clear interpretation; CSI has unjustified weights and no within-architecture predictive power (Spearman rho=0.03).

### Technical Soundness: 6/10

Multiple soundness issues:

- **V_t increasing**: The Lyapunov function should decrease per Theorem 1 but empirically increases. This is not a minor issue -- it means the convergence certificate does not apply to actual training trajectories.

- **Theory-experiment mismatch**: All theorems assume SGD dynamics; primary experiments use AdamW. The PMP-WD costate derivation assumes SGD momentum structure.

- **Theorem 2 spin**: Claiming rho=-0.379, p=0.121 "validates" a bound is misleading. The evidence does not support the claim.

- **BEM formula inconsistency**: half_lambda shows BEM=0.000 but should show BEM=0.5 under the stated formula. Raw data confirms mean_wd_actual=0.0005, identical to constant.

- **Data mixing**: iter_003 and iter_006 data mixed without provenance tracking. The 0.33pp cross-iteration shift undermines the paper's core narrative about a 0.25pp inter-method spread.

### Experimental Rigor: 6/10

**Strengths**:
- Hyperparameter fairness protocol (identical base hyperparameters) is methodologically correct
- Inclusion of random_mask and half_lambda controls is excellent
- Statistical reporting (paired t-tests, Bonferroni, Cohen's d, TOST) is above community norm
- VGG-16-BN cross-architecture validation (0.16pp spread) is compelling

**Weaknesses**:
- No ImageNet experiments (mandatory per project constraints, standard for top-venue WD papers)
- Data provenance violation (mixing iter_003 and iter_006 data)
- N=3 seeds insufficient for null-result claims (TOST power ~15-20% at delta=0.5%)
- NoBN ablation data exists but excluded from paper
- PMP-WD absent from Table 3 (statistical tests), hiding its non-significance
- No proofs in appendix despite referencing "Appendix A" four times

### Reproducibility: 6/10

- L (smoothness constant) estimation method undocumented; certified band depends critically on L
- PMP-WD implementation details insufficient: is the momentum buffer raw or bias-corrected?
- Backward recursion for mu_t requires terminal condition and offline computation, but this is not clarified
- No code release mentioned
- No appendix with proofs

---

## Cross-Validation Results

| Method | Paper (Table 2) | iter_003 raw | iter_006 raw | Match? |
|--------|----------------|-------------|-------------|--------|
| constant (AdamW CIFAR-10) | 90.13 +/- 0.31 | 90.13 +/- 0.31 | 89.80 +/- 0.31 | Paper = iter_003 |
| VGG-16-BN constant (SGD) | 92.05 +/- 0.05 | -- | 92.05 +/- 0.06 | MATCH |
| PMP-WD (text ref) | 90.29 +/- 0.12 | N/A | 90.29 +/- 0.12 | iter_006 only |

The Table 2 AdamW data is from iter_003. PMP-WD data is from iter_006. These are not the same experiment batch. The constant baseline shifted by 0.33pp between batches. This is the paper's most dangerous hidden inconsistency.

---

## What Would Raise the Score

From 6.5 to 8.0 requires:

1. **Data consistency** (~8 GPU-hours): Rerun all methods with iter_006 instrumented code for both CIFAR-10 and CIFAR-100. This resolves the provenance issue and provides instrumented data for all diagnostic metrics.

2. **Appendix with proofs** (~6 hours writing): Write complete proofs for Theorems 1-4. Non-negotiable.

3. **ImageNet-100** (~6 GPU-hours): 3 methods (constant, CWD, PMP-WD) x 3 seeds on ImageNet-100 subset. Even partial ImageNet results substantially strengthen the paper.

4. **V_t decomposition** (~2 hours): New figure showing f(w_t) and mu_t*||w_t||^2 separately. Honest discussion of when the Lyapunov guarantee applies.

5. **Additional seeds** (~2 GPU-hours): 2 more seeds for 4 key AdamW comparisons to enable TOST at delta=0.5%.

Total: ~16 GPU-hours + 8 hours writing.

---

## Calibration Note

This score (6.5) reflects a genuine regression from iter_005 (7.0) due to the newly discovered data provenance violation. While the paper has more content (PMP-WD, certified band figure, Theorem 2 validation), the cross-validation against raw data reveals that the core empirical narrative (0.25pp spread) is built on mixed-provenance data where the inter-batch variance (0.33pp) exceeds the inter-method variance (0.25pp). This is a fundamental soundness issue that was not present (or detected) in iter_005's narrower scope.
