# Experiment Critique: Activation Energy Theory

## Overall Assessment

The experimental design is straightforward and appropriate for the research questions. The execution is mostly clean, with data properly saved in JSON format. However, critical issues---a formula-data inconsistency, contradictory hypothesis labeling, missing quantitative error classification, and a small sample size---undermine the reliability of the results.

**Score: 5/10**

---

## Critical Issues

### 1. Ea Formula-Data Inconsistency (Showstopper)

The paper defines Ea = -ln(c0). The data shows:
- Ea = 9.465, c0 = 0.10565 (problem idx 0)
- But -ln(0.10565) = 2.247, not 9.465

This inconsistency appears across all 50 problems. Either:
- (a) The formula is wrong and should be something like Ea = -ln(c0/c_ref)
- (b) c0 is not the consistency fraction but a transformed value
- (c) The data computation is completely different from the stated formula

**Evidence**: In analysis_h3.json, every problem with Ea ≈ 9.465 has c0 = 0.10565106800135829. Every problem with Ea ≈ 10.0 has c0 ≈ 0.1. The relationship is NOT Ea = -ln(c0).

If we solve for the actual formula: for c0 = 0.10565, Ea = 9.465, so the transformation might be Ea = -ln(c0) + 7.218, or c0 might be something other than consistency fraction.

**Impact**: The entire theoretical framework depends on Ea being derived from c0. If the derivation is different from what is stated, the framework's validity is compromised.

**Fix**: Inspect analysis_h3.py and analysis_h2.py to determine the actual Ea computation. Correct the paper's formula or the data.

### 2. H3 Source Data Mislabeled as "CONFIRMED"

analysis_h3.json contains:
```json
"h3_evaluation": {
  "status": "CONFIRMED",
  "pass": true,
  "notes": "Low-Ea accuracy (75.0%) meets threshold (75%). Ea is a useful routing signal."
}
```

This directly contradicts the paper's narrative. The same file also contains:
```json
"predictor_quality": {
  "auc": 0.4356617647058823,
  "useful_predictor": false,
  "spearman_ea_accuracy": -0.06341151396912596,
  "p_value_ea_accuracy": 0.6617582904647242
}
```

The predictor_quality section correctly identifies that Ea is NOT a useful predictor, but the h3_evaluation section ignores this and reports "CONFIRMED."

**Root cause**: The H3 evaluation logic appears to only check the threshold test (low-Ea accuracy > 75%) and ignores the AUC and Spearman evidence.

**Fix**: Update the evaluation logic to require BOTH threshold pass AND AUC > 0.5 for H3 confirmation. Update analysis_h3.json and analysis_h3_summary.md.

---

## Major Issues

### 3. Sample Size Is Inadequate

n = 50 problems with only 2 at Level 1 and 13 at Level 5. Key implications:
- Level 1 mean Ea is based on 2 problems---statistically meaningless
- Level 5 near-zero variance (σ ≈ 1.9e-6) may be an artifact of small sample
- Spearman correlation confidence intervals are wide and not reported

**Evidence**: The pilot used n=30; the "full" experiment expanded to n=50. But 50 is still pilot-scale for statistical claims.

**Fix**: Report confidence intervals for all correlations. Acknowledge the small sample more prominently. Consider expanding to n=200 if feasible.

### 4. Per-Problem Fit Statistics Undermine H1

analysis_h1.json reveals:
- Median R² = 0.000 for all three models (exponential, power-law, logarithmic)
- 80% of problems "fail to fit" (exact criterion unclear)
- Only 8/50 problems prefer exponential by AICc; 20 prefer power-law; 22 prefer logarithmic
- By BIC: 4 prefer exponential, 24 prefer power-law, 22 prefer logarithmic

The aggregate R² = 0.924 is driven by averaging over 50 problems with heterogeneous dynamics. The exponential model is NOT the best fit for most individual problems.

**Implication**: H1 is confirmed only at the aggregate level. The Arrhenius framework may be a statistical artifact of averaging rather than a genuine mechanism.

**Fix**: Report bootstrap confidence intervals for the aggregate R². Discuss whether the exponential form is simply the best group-level curve fit or reflects a genuine mechanism.

### 5. H5 Falsification Is Under-Reported

H5 tests whether consistency-Ea matches saturation-k0. Results:
- Spearman = -0.219, p = 0.54
- Only 10/50 problems have valid k0 estimates
- Status: FALSIFIED

This is a significant theoretical problem: the two "difficulty" measures derived from the same framework do not agree. The paper mentions this briefly in Section 5.4 but does not discuss the implication for the theory's internal consistency.

**Fix**: Expand discussion of H5 in the Discussion section. Explain why the two measures diverge.

### 6. Missing Quantitative Error Classification (H4)

The paper claims execution errors "dominate" low-Ea failures but provides no data. The outline planned quantitative error classification, but:
- No error classification experiment was run
- Table 8 only has qualitative descriptions
- The claim is unsupported

**Fix**: Perform manual error classification on low-Ea failures, or remove/soften the claim.

### 7. Post-Hoc Threshold Is Data Leakage

The optimal threshold (9.9999999999) is optimized on the evaluation data. This is acknowledged but the 75% figure is still used to compute the "irreducible error floor."

**Fix**: Report a cross-validated threshold or clearly label the 25pp as an optimistic upper bound.

### 8. Level 5 Ea Shows Algorithmic Saturation

All 13 Level 5 problems have Ea ≈ 10.0 with σ ≈ 1.9e-6. This is not biological---it suggests:
- All Level 5 problems hit the same numerical floor
- The consistency computation has limited precision
- The Ea transformation amplifies small differences near the floor

**Fix**: Report raw c0 values for Level 5. Investigate whether this is a computational artifact.

---

## Minor Issues

### 9. Accuracy at k=8 and k=16 Decreases

Table 4 shows:
- k=4: 86.0%
- k=8: 84.0%
- k=16: 82.0%

Accuracy decreases with more samples, which is counterintuitive. The exponential fit (R²=0.924) smooths over this non-monotonicity. Possible explanations:
- Sampling variance (n=50 is small)
- Majority voting can degrade if wrong answers become more consistent
- Random seed effects

**Fix**: Acknowledge the non-monotonicity and discuss whether it affects the exponential fit interpretation.

### 10. Experiment State Contains Stale/Failed Tasks

experiment_state.json contains many stale tasks from prior rounds (train_g1_stepdpo, eval_g3_adaptive_routing, setup_api_environment) that are irrelevant to the current paper. This does not affect the paper but indicates cleanup is needed.

### 11. No Confidence Intervals or Bootstrap Estimates

Key statistics (R²=0.924, Spearman=0.448, AUC=0.436) are reported as point estimates without confidence intervals. With n=50, the uncertainty is substantial.

**Fix**: Add bootstrap confidence intervals for all key statistics.

---

## Summary

The experiments are competently executed but suffer from:
1. A critical formula-data inconsistency that must be resolved
2. Contradictory hypothesis labeling in source data
3. Small sample size limiting statistical power
4. Missing quantitative evidence for key claims (H4)
5. Under-reporting of H5 falsification

**Priority fixes**:
1. Resolve Ea formula inconsistency
2. Fix H3 labeling in source data
3. Add confidence intervals
4. Perform or remove H4 quantitative analysis
5. Explain Level 5 Ea artifact
