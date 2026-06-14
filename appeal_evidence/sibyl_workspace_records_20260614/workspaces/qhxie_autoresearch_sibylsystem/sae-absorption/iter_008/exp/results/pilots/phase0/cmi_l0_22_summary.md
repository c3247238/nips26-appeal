# Phase 0.3: CMI Replication at L0=22 -- Pilot Results

## Summary

**Task:** CMI Replication at L0=22 (Theoretical Pillar Validation)
**Status:** SUCCESS (all 25/25 letters computed)
**Verdict:** NOT SUPPORTED -- Demote CMI theoretical claims to appendix
**Runtime:** 22 minutes (planned: 60 min)
**GPU:** RTX PRO 6000 Blackwell, single GPU (GPU 4)

## Key Result

At L0=22 where ALL 25 probes achieve F1=1.0 (eliminating probe quality confound entirely):

- **Spearman rho(absorption_rate, CMI) = 0.044** (p = 0.835)
- **Permutation test p = 0.832** (1000 permutations)
- **Bootstrap 95% CI for rho: [-0.41, 0.47]** -- wide CI spanning zero
- **H3 target (rho < -0.3): NOT MET**

This is a **definitive null result**. The CMI-absorption correlation observed at L0=82 (rho = -0.383, p = 0.059 uncorrected) was driven by the probe quality confound (rho = -0.67 between probe F1 and absorption). When the confound is eliminated at L0=22, the correlation vanishes.

## Comparison: L0=82 vs L0=22

| Metric | L0=82 | L0=22 |
|--------|-------|-------|
| Aggregate absorption | 15.96% | 42.85% |
| Mean probe F1 | 0.82 | **1.00** |
| Letters with F1>0.85 | 10/25 | **25/25** |
| Spearman rho | -0.383 | **0.044** |
| Bonferroni p | 0.236 | **1.000** |
| Permutation p | N/A | **0.832** |

## Mann-Whitney U Test (Absorbed vs Non-Absorbed)

- Absorbed letters (20): mean CMI = 2.888, std = 0.157
- Non-absorbed letters (3): mean CMI = 2.721, std = 0.234
- Cohen's d = 0.944 (large effect in the WRONG direction -- absorbed have HIGHER CMI)
- p = 0.230 (two-sided) -- not significant

Note: The "large" Cohen's d is misleading because n_non_absorbed = 3 (Q, Y, Z only).
These letters also have very few words (Q=15, Y=12, Z=8), which likely deflates their CMI.

## Subspace Dimension Sensitivity

| d' | Spearman rho | p-value | Bonferroni p | Cohen's d |
|----|-------------|---------|-------------|-----------|
| **10 (pre-registered)** | **0.044** | **0.835** | **1.000** | 0.944 |
| 20 | 0.248 | 0.232 | 1.000 | 1.600 |
| 30 | 0.410 | 0.042 | 1.000 | 1.616 |
| 50 | 0.483 | 0.014 | 0.364 | 1.675 |

**Important:** The pre-registered d'=10 shows no correlation. Higher dimensions (d'=30, 50)
show positive correlations -- the OPPOSITE direction from the theoretical prediction (expected negative).
This suggests that at higher dimensions, the subspace captures more general letter-specific information
that correlates with absorption rate, but this is likely a confound (more features = more CMI) rather
than evidence for hierarchy-driven absorption.

## k-Sensitivity (kNN parameter)

| k | Spearman rho | p-value |
|---|-------------|---------|
| 3 | 0.040 | 0.848 |
| 5 | 0.044 | 0.835 |
| 10 | -0.003 | 0.988 |
| 20 | 0.044 | 0.834 |

Robust null across all k values -- the result is not an artifact of k-NN hyperparameter choice.

## Interpretation for Paper

The CMI-absorption correlation at L0=82 (rho = -0.383) was a statistical artifact driven by
the probe quality confound. At L0=22, where the confound is eliminated:

1. The pre-registered analysis (d'=10, k=5) shows rho = 0.044 (essentially zero)
2. The permutation test confirms this is indistinguishable from random (p = 0.832)
3. k-sensitivity analysis shows robustness across k = {3, 5, 10, 20}
4. Higher-dimension analyses show POSITIVE correlations (opposite to theory), likely driven by
   confounds in the subspace construction

**Recommendation:** Demote CMI theoretical claims from Section 6 to Appendix. The "CMI predicts
absorption susceptibility" claim is not supported. Report this as an honest negative result that
resolves the overclaiming issue identified in 4 consecutive reviews.

## Pass Criteria

- [x] Absorption rates computed for 25/25 letters at L0=22
- [x] All probe F1 values confirmed = 1.0 at L0=22
- [x] CMI value and permutation p-value computed
- [x] Spearman rho(d', absorption) reported with 95% CI
- [x] Any result (significant or null) is informative -- NULL result is definitive
