# Planning Critique: Activation Energy Theory Methodology

## Overall Assessment

The methodology is well-structured and appropriate for the research questions. The experimental design follows a clear group structure (G0 baseline, G1 saturation, G2 consistency, G3 routing) with explicit hypotheses, metrics, and thresholds. However, the methodology does not adequately address several critical issues that emerged during execution: the Ea computation discrepancy, the post-hoc threshold problem, and the lack of cross-validation.

**Score: 6/10**

---

## Critical Issues

### 1. Methodology Does Not Match Actual Computation

The methodology.md defines Ea = -ln(c0) where c0 is "the answer consistency at k = 16 samples (fraction of samples agreeing on the majority answer)." However, the actual data shows Ea = 9.465 with c0 = 0.10565, which is inconsistent with this formula.

**Impact**: The methodology describes a computation that was not actually performed. This is a methodological integrity issue.

**Fix**: Update methodology.md to reflect the actual Ea computation used in the experiment code, or correct the experiment code to match the stated methodology.

### 2. No Cross-Validation Protocol for Threshold Selection

The methodology states: "Use Ea threshold to classify problems as 'easy' vs. 'hard'. Validate single-pass accuracy on easy problems >75%." But it does not specify:
- How the threshold is selected
- Whether threshold selection is done on a held-out set
- Whether cross-validation is used

The actual execution used post-hoc threshold optimization on the full dataset, which is a form of data leakage.

**Fix**: Add a cross-validation protocol to the methodology. Specify that threshold selection must be done on a training fold and evaluation on a held-out test fold. If the sample size is too small for cross-validation, state this limitation explicitly.

---

## Major Issues

### 3. Per-Problem Fit Failure Is Not Addressed in Methodology

The methodology states: "Fit Arrhenius model: P_k = P_∞ × (1 - exp(-k/k₀)). Extract per-problem k₀." But it does not specify:
- What constitutes a "valid" fit
- How to handle problems where the fit fails
- Whether to report per-problem statistics

The actual execution found that 80% of per-problem fits fail and only 10/50 problems yield valid k0 estimates. The methodology should have anticipated this and specified a protocol.

**Fix**: Add criteria for valid fits (e.g., R² > 0.5, successful convergence) and a protocol for handling failures (e.g., exclude from H5 analysis, report failure rate).

### 4. No Protocol for Error Classification (H4)

The methodology does not describe how errors would be classified into execution, conceptual, and extraction categories. The outline mentions H4 error analysis but the methodology.md does not specify:
- Who performs the classification (human annotator, automated script)
- Inter-annotator agreement procedures
- Classification criteria and examples

**Fix**: Add a detailed error classification protocol to the methodology, or remove H4 from the methodology if it is not part of the core experiment.

### 5. Missing Controls and Baselines

The methodology compares Ea-based routing to a 75% threshold but does not include:
- A random baseline (coin flip routing)
- A majority-voting baseline (always use multi-sample)
- A difficulty-level baseline (route by MATH level)

The AUC = 0.436 comparison to random (AUC = 0.5) is mentioned in the results but not planned in the methodology.

**Fix**: Add explicit baseline comparisons to the methodology.

### 6. Sample Size Justification Is Missing

The methodology uses n = 50 problems but does not justify this sample size. With 5 difficulty levels, this yields only 10 problems per level on average (and only 2 at Level 1). No power analysis is provided.

**Fix**: Add a power analysis or sample size justification. For the stated effect sizes (Spearman > 0.4, AUC > 0.5), compute the minimum sample size needed for 80% power.

---

## Minor Issues

### 7. Temperature Choice Is Not Justified

Temperature T = 0.7 is specified without justification. Why not T = 0.5 or T = 1.0? Different temperatures could affect consistency patterns.

**Fix**: Add a brief justification for T = 0.7 (e.g., "balanced diversity and repeatability based on pilot testing").

### 8. k Values Are Arbitrarily Spaced

The k values {1, 2, 4, 8, 16} are powers of 2, but there is no justification for this spacing. Would linear spacing (1, 4, 7, 10, 13, 16) provide better curve fits?

**Fix**: Justify the exponential spacing or note it as a design choice.

### 9. Answer Extraction Audit Is Under-Specified

The methodology mentions "manual audit for 10% of cases" but does not specify:
- How the 10% is selected (random, stratified)
- Who performs the audit
- What constitutes a correct extraction
- How disagreements are resolved

**Fix**: Specify the audit protocol in detail.

### 10. No Protocol for Handling Ties in Majority Voting

If k = 2 and the two samples disagree, there is no majority. The methodology does not specify how ties are handled.

**Fix**: Add a tie-breaking rule (e.g., random selection, lower-index sample, or exclude from analysis).

---

## Summary

The methodology is sound in broad strokes but lacks critical details that emerged as problems during execution. The most important fixes are:

1. Align the stated Ea formula with the actual computation
2. Add cross-validation for threshold selection
3. Specify per-problem fit criteria and failure handling
4. Add error classification protocol or remove H4
5. Include baseline comparisons
6. Justify sample size with power analysis

Without these additions, the methodology does not provide sufficient detail for reproducibility.
