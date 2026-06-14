# Supervisor Review: Phi Invariance Conjecture Paper

**Score: 6.5 / 10** | **Verdict: CONTINUE** | **Iteration: current (post iter_007)**

---

## Executive Summary

The paper presents a compelling empirical finding (weight decay method choice is irrelevant under AdamW on CIFAR) with a well-organized taxonomy (Phi Modulator Framework) and exemplary statistical methodology. The AdamW-vs-SGD contrast is scientifically interesting, and the honest treatment of negative results (Theorem 2, TOST partial confirmation) is commendable.

However, the score remains at 6.5 -- unchanged from the prior review -- because **none of the four critical issues identified in iter_007 have been resolved**:

1. Data provenance mismatch (0.33pp inter-batch shift > 0.25pp inter-method spread)
2. No appendix with proofs (5th consecutive iteration)
3. No ImageNet experiments
4. Lyapunov certificate empirically contradicted

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|:-----:|---------------|
| Novelty | 7 | Phi Invariance Conjecture is a genuinely useful empirical finding. The framework is primarily notational but the taxonomy has organizational value. |
| Soundness | 5.5 | Data provenance mismatch is fatal: 0.33pp batch variance > 0.25pp claimed signal. V_t contradiction unresolved. Theory-experiment optimizer mismatch undiscussed. |
| Experiments | 6 | 105 experiments with rigorous statistical tests, but CIFAR-only, N=3, no ImageNet, PMP-WD inconsistently integrated, NoBN data unused. |
| Reproducibility | 5 | No proofs, no code release, undocumented L estimation, CSI implementation details missing. |

---

## Cross-Validation of Paper Claims Against Raw Data

### Table 2 (AdamW + ResNet-20, CIFAR-10)

Paper reports constant = 90.13 +/- 0.31. Cross-validated against:
- **iter_003**: seeds 90.48, 90.03, 89.89 -> mean = 90.13. **MATCHES Table 2.**
- **iter_006 instrumented rerun**: seeds 89.72, 90.15, 89.54 -> mean = 89.80. **DOES NOT MATCH.**

The 0.33pp discrepancy between iter_003 and iter_006 for the identical configuration (constant, AdamW, CIFAR-10, ResNet-20, same hyperparameters) demonstrates significant inter-batch variance. The paper's narrative of "0.25pp spread across methods" is built on iter_003 data, but PMP-WD (90.29 from iter_006) is from a different batch. Mixing these creates an internally inconsistent dataset.

### PMP-WD

iter_006 PMP-WD seeds: 90.16, 90.34, 90.38 -> mean = 90.29 +/- 0.12. These numbers appear in the text but PMP-WD is absent from Tables 1-3. PMP-WD's BEM values (0.505, 0.485, 0.535) confirm it uses approximately half the WD budget, consistent with the optimal control derivation.

### Table 6 Diagnostic Metrics

BEM values cross-validated against iter_006 instrumented data:
- constant: BEM=0.0 (paper: 0.000) -- **MATCHES**
- cwd_hard: mean_wd_actual=0.000244 -> BEM ~= 0.51 (paper: 0.503) -- **APPROXIMATELY MATCHES**
- half_lambda: Table 6 reports BEM=0.500 -- **CORRECT** (prior review flagged BEM=0.000, which appears to have been fixed in the paper text but may persist in figures)

---

## Critical Issues (Must Fix Before Score Can Improve)

### 1. Data Provenance (Critical -- Soundness)

The paper's central claim is that inter-method accuracy spread is 0.25pp under AdamW. But the inter-BATCH spread for a single method (constant) is 0.33pp. The signal-to-noise ratio is < 1. Either:
- All methods must be rerun from a single consistent codebase, or
- The provenance must be documented and the inter-batch variance factored into the analysis

### 2. Missing Appendix (Critical -- Reproducibility)

5th consecutive iteration. The paper states 4 theorems and references "Appendix A" in at least 2 places. Zero proofs exist. This is a desk-reject issue at any top venue.

### 3. No ImageNet (Critical -- Experiments)

CIFAR-only experiments limit the paper's impact and relevance. The observation that WD doesn't matter on CIFAR with AdamW is informally known. The paper needs larger-scale validation to make a novel empirical contribution.

### 4. Lyapunov Contradiction (Critical -- Soundness)

V_t increases empirically, contradicting Theorem 1. Section 5.7's claim that "the band narrows rapidly" is contradicted by the figure. This needs honest treatment.

---

## Major Issues

1. **Theory-experiment optimizer mismatch**: All theorems derived for SGD; all primary experiments use AdamW. Never discussed.
2. **PMP-WD integration**: Appears in figures but not tables. Creates method set inconsistency.
3. **Theorem 2 overclaiming**: Now reported honestly (p=0.121) but still listed as a numbered contribution.
4. **Statistical power**: N=3, TOST passes only 50% of comparisons at delta=1.0%.
5. **NoBN ablation missing**: Data exists from iter_005, would test the BN mechanism hypothesis.
6. **CSI predictive failure**: Arbitrary weights, rho < 0.3, zero predictive value.
7. **Abstract misleading factorial framing**: Implies 168 runs, actual is 105.

---

## What Is Working Well

1. **Statistical methodology** is exemplary: paired t-tests, Bonferroni correction, TOST equivalence testing, Cohen's d, explicit power analysis. This sets a standard for empirical ML papers.

2. **Honest negative-result framing**: Theorem 2 validation, TOST partial confirmation, and Section 6.4 limitations are written with scientific integrity.

3. **Phi taxonomy (Table 1)** is genuinely useful for the community. The four-axis organization and programmatic interface provide real organizational value.

4. **AdamW vs. SGD contrast** is the paper's strongest empirical finding. The 1.2% vs. 97% weight norm variation provides clear mechanistic evidence.

5. **Writing quality** is generally high. The paper is well-organized and readable.

---

## Path to Score >= 8.0

| Action | Effort | Score Impact |
|--------|--------|-------------|
| Rerun all methods with consistent codebase | ~8 GPU-hrs | Soundness 5.5 -> 7.0 |
| Write Appendix A with proofs | ~6 hrs writing | Reproducibility 5 -> 7 |
| ImageNet-100 experiments | ~8 GPU-hrs | Experiments 6 -> 7.5 |
| V_t decomposition + Section 5.7 fix | ~2 hrs analysis | Soundness +0.5 |
| Add 2 seeds for key comparisons | ~2 GPU-hrs | Experiments +0.5 |
| **Total** | **~18 GPU-hrs + ~8 hrs writing** | **~8.0** |

---

*Review generated by Supervisor Agent. Score 6.5 directly controls quality gate: score < 8.0 triggers another iteration.*
