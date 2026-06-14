# Supervisor Review: The Limits of Consistency-Based Activation Energy for Problem-Level Routing

**Reviewer**: Independent Senior Supervisor (NeurIPS/ICML/ICLR Calibrated)
**Date**: 2026-04-29
**Score**: 5.5 / 10
**Verdict**: REVISE

---

## Executive Summary

This paper investigates whether consistency-derived "activation energy" (Ea) can predict single-pass solveability for LLM mathematical reasoning, framing the investigation as a systematic falsification. The central negative result---that Ea has zero predictive power (AUC = 0.436, Spearman = -0.063)---is honest and potentially valuable. However, the paper is critically undermined by a formula-data inconsistency that makes the entire Ea computation unreproducible, contradictory source data labeling, a tiny sample size (n=50), and an aggregate H1 fit that masks catastrophic per-problem failure. The novelty claim is narrow and fragile, resting entirely on a single negative result since H1/H2 collide with prior work. In its current form, this paper would receive a borderline-to-clear reject at a top venue.

---

## Dimension Scores

### 1. Novelty & Significance: 5/10

The paper's novelty rests almost entirely on the H3 negative result (Ea cannot predict single-pass solveability). H1 (exponential saturation, R2=0.924) and H2 (Ea correlates with difficulty, Spearman=0.448) collide directly with Yang et al. (2025), whose exponential saturation formula is mathematically equivalent. The novelty_report.json correctly scores overall novelty at 6/10 ("medium"), which is accurate.

The H3 falsification is defensible but not unprecedented. ACAR (2026) reports a related "agreement-but-wrong" ceiling of ~8pp for variance-based routing. The paper does not clearly differentiate its consistency-based finding from ACAR's variance-based finding. The diagnostic analysis (H4 error classification) is promised but not delivered with quantitative evidence, further narrowing the contribution.

**What would raise this score**: Complete the H4 quantitative error classification with counts/proportions, clearly differentiate from ACAR (consistency vs. variance, ~25pp vs. ~8pp ceiling), and demonstrate generalizability across multiple models.

### 2. Technical Soundness: 4/10

The technical soundness is severely compromised by multiple issues:

**Critical**: The formula Ea = -ln(c0) is inconsistent with all reported data. For c0 = 0.10565, -ln(c0) ≈ 2.247, not 9.465. This is not a rounding error; it is a fundamental discrepancy. Either the formula is wrong, the data computation is wrong, or c0 is not what the paper claims. Until this is resolved, readers cannot reproduce the analysis, and the entire theoretical framework is suspect.

**Major**: The per-problem fit statistics undermine the H1 claim. While aggregate R2=0.924 looks good, the median per-problem R2=0.000, 80% of problems fail to fit, and only 8/50 prefer the exponential model by AICc. The aggregate "success" is likely an ensemble averaging artifact, not evidence of a genuine physical law. The paper mentions this as a "critical caveat" but does not adequately address the implication.

**Major**: H5 falsification (consistency-Ea vs. saturation-k0, Spearman=-0.219) challenges the internal consistency of the theoretical framework. The two "difficulty" measures are uncorrelated, suggesting the theory lacks coherence. This is under-reported.

**What would raise this score**: Resolve the Ea formula inconsistency, add bootstrap confidence intervals for aggregate R2, and expand the discussion of why the theoretical framework lacks internal consistency (H5).

### 3. Experimental Rigor: 6/10

**Strengths**: The paper honestly reports negative results (AUC=0.436, per-problem fit failure) without spin. The experimental protocol is clearly described (model, dataset, sampling parameters, temperature, seed). The cross-validation of paper claims against raw data is possible because the result files are well-structured.

**Weaknesses**:
- **n=50 is pilot-scale**, not "full." With only 2 problems at Level 1, per-level statistics are unstable. Confidence intervals are not reported for any key statistic.
- **Post-hoc threshold optimization** (threshold=9.9999999999) constitutes data leakage. The 75% low-Ea accuracy and the 25pp "irreducible error floor" derived from it are optimistic upper bounds, not measured quantities.
- **Table 8 lacks quantitative data**. The claim that "execution errors dominate" has no counts, proportions, or statistical support.
- **Single model** (Qwen2.5-Math-7B-Instruct) with no cross-model validation.
- **Level 5 Ea shows near-zero variance** (sigma ≈ 1.9e-6), suggesting algorithmic saturation rather than genuine measurement.

**What would raise this score**: Expand to n=100-200 problems, add a second model, report confidence intervals for all statistics, add quantitative error classification to Table 8, and explain the Level 5 artifact.

### 4. Reproducibility: 4/10

**Critical gaps**:
- The Ea formula is inconsistent with data---readers cannot reproduce Ea values from the stated formula.
- No bibliography exists, preventing verification of cited works. Some citations ("Li 2026", "ACAR 2026") appear to be future-dated and may not exist.
- No code or data repository is referenced.
- The post-hoc threshold is not reproducible without the exact dataset and optimization procedure.
- Missing appendix content (full problem data table, extraction audit results).

**What would raise this score**: Resolve the formula inconsistency, add a complete bibliography with verified references, add the planned appendix with full problem data, and reference the experiment scripts.

---

## Critical Issues (Must Fix)

### 1. Ea Formula-Data Inconsistency [CRITICAL / soundness]
The paper defines Ea = -ln(c0). The data shows Ea = 9.465 with c0 = 0.10565. But -ln(0.10565) ≈ 2.247, not 9.465. This is mathematically impossible. The notation.md Usage Note 3 incorrectly states "Ea ≈ 9.47 corresponds to c0 ≈ 0.106"---false under the stated formula.

**Fix**: Verify the actual formula in analysis_h3.py and analysis_h2.py. Correct either the formula or the data throughout.

### 2. H3 Source Data Contradicts Paper [CRITICAL / experiment]
analysis_h3.json labels H3 as "CONFIRMED" with note "Ea is a useful routing signal," while the paper correctly interprets the same data as falsified due to AUC=0.436. The 75% threshold pass is a post-hoc artifact.

**Fix**: Update source data files to reflect the correct interpretation. Add explicit note explaining why threshold test is overridden by AUC/Spearman.

### 3. Missing Bibliography [CRITICAL / writing]
The paper cites [1] through [11] but has no References section. Some citations may not exist ("Li 2026", "ACAR 2026").

**Fix**: Add complete bibliographic entries. Verify all cited works exist.

---

## Major Issues (Should Fix)

### 4. Table 8 Lacks Quantitative Data [MAJOR / experiment]
Section 6.2 claims "Execution errors dominate low-Ea failures" but Table 8 only has qualitative descriptions. No counts or proportions support this central diagnostic claim.

**Fix**: Perform quantitative error classification and add Count/Proportion columns, or soften the claim.

### 5. Sample Size Too Small [MAJOR / experiment]
n=50 is pilot-scale. Per-level statistics are unstable. Confidence intervals are not reported.

**Fix**: Report confidence intervals. Acknowledge n=50 as a limitation. Expand to 100-200 if feasible.

### 6. Post-Hoc Threshold = Data Leakage [MAJOR / analysis]
The 75% figure and 25pp "error floor" are derived from post-hoc threshold optimization on the same data.

**Fix**: Clarify that 25pp is an optimistic upper bound, not a measured floor.

### 7. Novelty Claim Is Narrow [MAJOR / novelty]
Novelty rests entirely on H3 negative result. ACAR reports a conceptually related finding. Differentiation is weak.

**Fix**: Explicitly contrast consistency-based vs. variance-based routing. Quantify the larger error floor (~25pp vs. ~8pp).

### 8. Per-Problem Fit Failure [MAJOR / experiment]
Median R2=0.000, 80% fit failure. Only 8/50 prefer exponential by AICc. The aggregate fit may be an averaging artifact.

**Fix**: Expand discussion. Add bootstrap confidence intervals for aggregate R2.

### 9. Single Model Limits Generalizability [MAJOR / reproducibility]
All experiments use one model. Broad claims about consistency-based routing exceed the evidence.

**Fix**: Add second model, or temper claims to "on Qwen2.5-Math-7B-Instruct."

### 10. H5 Falsification Under-Reported [MAJOR / analysis]
Consistency-Ea and saturation-k0 are uncorrelated (Spearman=-0.219), challenging the theory's internal consistency.

**Fix**: Expand discussion in Section 6. Explain why the two measures capture different constructs.

---

## Minor Issues

11. **Inconsistent terminology**: "Arrhenius kinetics" vs. "Arrhenius-like kinetics"
12. **Misleading product interpretation**: P_infty * k0 conflates ceiling and rate
13. **"Prove" is too strong**: Replace with "demonstrate" or "show"
14. **Missing appendix**: Full problem data table, extraction audit
15. **Level 5 Ea artifact**: Near-zero variance unexplained
16. **Promised but undelivered**: H4/H5 experiments mentioned but not completed

---

## What Would Raise the Score

**To 6.5 (+1.0)**: Resolve Ea formula inconsistency, add complete verified bibliography, add quantitative data to Table 8.

**To 7.5 (+2.0)**: All above plus expand to n=100-200, add second model, report confidence intervals, expand H5 discussion.

**To 8.0+ (publication-worthy)**: All above plus either (a) complete H4 quantitative error classification and H5 entropy routing with positive diagnostic results, or (b) demonstrate generalizability across multiple models/datasets with rigorous statistical validation.

---

## Risks

1. If the Ea formula inconsistency reveals a different computation, the H3 interpretation may change.
2. Yang et al. collision eliminates H1/H2 novelty; reviewers may reject for insufficient contribution.
3. ACAR's related finding may lead reviewers to question whether H3 falsification is novel enough.
4. Single model + small sample makes generalizability claims untenable.
5. Per-problem fit failure (median R2=0.000) may cause reviewers to dismiss the Arrhenius analogy entirely.

---

## Evidence Gaps

- Actual Ea computation formula (stated formula inconsistent with all data)
- Quantitative error classification counts for Table 8
- Confidence intervals / bootstrap estimates for key statistics
- Cross-model validation
- Explanation for Level 5 Ea near-zero variance
- H5 entropy routing experiment (promised but not delivered)
- Verified bibliography (some citations may not exist)
- Appendix with full problem data and extraction audit

---

## Final Verdict

The paper has a defensible negative result at its core, but it is not ready for submission. The critical formula-data inconsistency must be resolved before any further progress. Even after resolution, the small sample size, single-model limitation, and narrow novelty claim mean the paper would likely receive a borderline reject at NeurIPS/ICML/ICLR. Substantial additional experimental work (larger sample, second model, quantitative diagnostics) is needed to reach publication quality.

**Recommendation**: REVISE. Do not proceed to writing or submission until the critical issues are resolved and at least the major issues are addressed.
