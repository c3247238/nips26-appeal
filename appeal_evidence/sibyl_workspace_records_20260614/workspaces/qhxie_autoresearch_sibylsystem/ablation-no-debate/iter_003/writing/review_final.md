# Final Review: Feature Absorption in Sparse Autoencoders

**Iteration**: 4
**Date**: 2026-05-01
**Overall Score**: 7/10
**Verdict**: CONTINUE with revisions

---

## Executive Summary

The paper presents a genuinely novel methodology (multi-child proportional ablation) that addresses a real saturation problem in feature absorption research. The writing is clear and the hypothesis structure is well-organized. However, three critical data narrative issues must be resolved before quality gate.

---

## Critical Issues (Must Fix)

### 1. H_Mech Decomposition Contradicts Source Data

**Problem**: Paper Section 4.4/5.1 claims "decoder contributes 0.299" based on synthetic pilot data, but the real SAE experiment (held_out_validation.json) shows:
- Condition A (Random/Random) = 218.52
- Condition D (Trained/Trained) = 218.52
- Condition C (Random/Trained) = 34.39

Condition A = Condition D and Condition C = Condition A. Decoder contribution is **ZERO** on real SAEs. The paper text was never updated from synthetic to real findings.

**Fix Required**: Rewrite H_Mech narrative to reflect the real SAE finding that absorption is encoder-driven, not decoder-geometric.

### 2. H_Comp Hypothesis Falsified But Not Reported

**Problem**: The paper presents H_Comp as "PASSED" but h_comp_full.json (6 levels x 3 seeds x 100 samples) shows:
- monotonic = false
- R² = 0.04 (threshold was > 0.8)
- p_value = 0.703 (not significant)

The full experiment contradicts both the hypothesis and the earlier pilot result.

**Fix Required**: Report H_Comp honestly. Add discussion of why monotonicity fails at 6 levels (saturation, noise at extremes, or genuine non-monotonicity).

### 3. H_Safe P-Value Discrepancy

**Problem**: Paper reports p=0.665 but source data shows p=0.345. Data integrity issue.

**Fix Required**: Reconcile the discrepancy or acknowledge measurement difference.

---

## Major Issues

### 4. Table 6 Degenerate Values

Table 6 contains "134,717,856%" and "1.52 billion%" from dividing by near-zero baselines. Remove percentage column.

### 5. Overlap Method Undefined

Section 3.7 introduces "overlap method" for H_Safe but never defines it. Align with multi-child proportional ablation methodology.

### 6. Deterministic Absorption

H1 absorption is exactly 0.50 with std=0.0 across all seeds - this is deterministic arithmetic, not statistical measurement. The t-test is mathematically valid but scientifically hollow.

**Fix**: Report geometric computation directly, acknowledge deterministic outcome.

### 7. Shuffled/Permuted = Trained SAE

H1 claims to differentiate trained SAEs from baselines, but Shuffled (0.487) and Permuted (0.484) nearly equal Trained (0.500). The differentiation is specifically about trained vs random encoder.

**Fix**: Reframe to "multi-child proportional ablation differentiates trained-encoder SAEs from random-encoder SAEs."

### 8. H2 Zero-Inflation

98.5% of features show zero absorption. The rho=+0.17 correlation is entirely driven by 15 outliers. Zero-inflated distributions require specialized analysis.

**Fix**: Report zero-inflation explicitly. Note correlation unreliable with this distribution.

---

## Minor Issues

9. **H_Downstream untested**: Listed as contribution but never executed. Remove or execute.
10. **Figure 1 lacks walkthrough**: Add 2-3 sentence prose description alongside caption.
11. **Abstract overloaded**: Single sentence contains 5 hypotheses. Split into 2-3 sentences.

---

## What Works

1. Multi-child proportional ablation is genuinely novel and solves a real problem
2. Clear hypothesis structure with pre-registered falsification criteria
3. Honest negative results (H2, H_Safe) reported without spin
4. H_Comp multi-seed validation provides robust evidence (even though it falsifies the hypothesis)
5. Encoder-driven finding on real SAEs is an important reframing

---

## Path to Quality Gate (8/10)

1. Fix H_Mech narrative to reflect real SAE data (encoder-driven, decoder effect = 0)
2. Report H_Comp honestly (monotonic = false)
3. Verify H_Safe p-value
4. Remove degenerate Table 6 percentages
5. Acknowledge deterministic H1 absorption
6. Reframe H1 to trained-encoder vs random-encoder
7. Define "overlap method" or align H_Safe with multi-child methodology

---

## Evidence Sources

- `/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/critic/critique_writing.md`
- `/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/supervisor/review_writing.md`
- `/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/critic/findings.json`
- `/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/supervisor/review.json`