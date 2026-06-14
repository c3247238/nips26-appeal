# Supervisor Review: Feature Absorption in Sparse Autoencoders

## Overall Score: 6.5 / 10 (Borderline Reject)

**Verdict: CONTINUE** -- The project has genuine methodological contributions (multi-child proportional ablation, encoder-driven absorption on real SAEs) but critical narrative issues must be resolved before publication.

---

## Dimension Scores

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| **Novelty & Significance** | 7 | Multi-child proportional ablation is genuinely novel and solves a real saturation problem. The encoder-driven finding on real SAEs is unexpected and reframes absorption research. |
| **Technical Soundness** | 6 | Core methodology is sound. However, paper narrative still claims decoder contributes 0.299 (from synthetic data) when real SAE data shows C=A (zero decoder effect). |
| **Experimental Rigor** | 6 | H_Comp multi-seed validation (6 levels x 3 seeds) is the strongest experiment. H_Mech confirmed encoder-driven on real SAEs. H_Safe p-value discrepancy needs resolution. H_Comp hypothesis falsified but paper only reports pilot. |
| **Reproducibility** | 6 | No experiment code in repository. Limited detail on H_Safe overlap method. H3 single-seed only. |

---

## Executive Summary

Multi-child proportional ablation is a genuine methodological advance that resolves the saturation problem in single-feature ablation. The multi-seed H_Comp validation (6 levels x 3 seeds) provides the most robust finding in the paper, though it contradicts the stated hypothesis: absorption does NOT increase monotonically with hierarchy strength. The real SAE H_Mech experiment (held_out_validation.json) confirms encoder-driven absorption (B=D, C=A), but the paper text was never updated from the earlier synthetic pilot.

Three critical issues require immediate attention:

1. **H_Mech narrative contradicts source data**: Paper claims decoder contributes 0.299, but held_out_validation.json shows Condition A = Condition D (218.52) and Condition C = Condition A (34.39). On real SAEs, decoder effect is ZERO. The paper text reflects synthetic pilot data, not the real SAE finding.

2. **H_Comp hypothesis is falsified but not reported**: h_comp_full.json shows monotonic=false with R²=0.04 (not R²>0.8). The full 6-level x 3-seed experiment contradicts the paper's hypothesis and the earlier pilot that claimed to confirm monotonic increase. The paper cites only the pilot result.

3. **H_Safe p-value discrepancy**: Paper claims p=0.665 but held_out_validation.json shows p=0.345. This data integrity issue must be resolved.

---

## Critical Issues

### 1. Paper Narrative Claims Decoder Contributes 0.299 But Real SAE Data Shows Zero Effect

**What the paper says (Sections 4.4, 5.1, Abstract):**
- "decoder geometry contributes 0.299 to absorption while encoder alignment adds 0.185"
- "The geometric contribution (0.299) exceeds the learned contribution (0.185)"
- "Absorption is primarily geometric (0.299 from decoder structure)"

**What the real SAE data actually shows (held_out_validation.json):**
| Condition | Encoder | Decoder | Value |
|-----------|---------|---------|-------|
| A | Random | Random | 218.52 |
| B | Trained | Random | (not reported in aggregate) |
| C | Random | Trained | 34.39 |
| D | Trained | Trained | 218.52 |

Condition A = Condition D. Condition C = Condition A. Decoder contribution is **ZERO** on real language model SAEs.

**Source of error**: The 0.299 figure comes from the synthetic pilot (h_mech_pilot_seed42.json) where d_model=128. The paper text was never updated after the real SAE experiment confirmed encoder-driven absorption.

**Fix**: Rewrite H_Mech sections to reflect the real SAE finding: "On real GPT-2 Small SAEs, encoder alignment is necessary and sufficient for absorption (B≈D), while decoder training has no measurable independent effect (C≈A)."

### 2. H_Comp Monotonic Hypothesis Is Falsified But Not Reported

**What the paper says (Section 4.7, Table 8):**
- H_Comp listed as "PASSED"
- Monotonic increase with hierarchy strength confirmed

**What the full experiment shows (h_comp_full.json):**
- 6 cosine levels × 3 seeds × 100 samples = 1800 total measurements
- monotonic = false
- R² = 0.04 (required R² > 0.8)
- p_value = 0.703 (not significant)

The pilot (3 levels, single seed) showed a monotonic trend. The full experiment (6 levels, 3 seeds) shows the trend does not hold.

**Fix**: Report the honest result: H_Comp hypothesis is NOT supported. Absorption does not increase monotonically with hierarchy strength at 6 levels. Add discussion of why this might be (saturation effects, measurement noise at extreme cosine values, or genuine non-monotonicity).

### 3. H_Safe P-Value Discrepancy

**Paper Abstract (Section 4.6.2)**:
- "Mann-Whitney U = 216.5, p = 0.665"

**Source data (held_out_validation.json)**:
- "u_statistic": 63.0, "p_value": 0.3447

These are not the same values. Verification required.

---

## Major Issues

### 4. Deterministic Absorption Undermines Statistical Claims

H1 trained SAE absorption is exactly 0.50 with std=0.0 across all samples. This is deterministic arithmetic from synthetic hierarchy geometry (cos=0.67), not measurement of a stochastic phenomenon. The t-test (t=63.2, p=3e-133) is statistically valid but scientifically meaningless: it compares a constant to a random variable.

**Fix**: Acknowledge that absorption under the current synthetic hierarchy design is deterministic. Report the geometric computation directly rather than framing as statistical significance test.

### 5. Shuffled/Permuted Baselines Match Trained SAE (H1 Overstatement)

Shuffled (0.487) and Permuted (0.484) baselines are nearly identical to Trained SAE (0.500) with effect sizes d=0.54 and d=0.61. Only Random Decoder (0.059) differs.

The paper claims H1 "differentiates trained SAEs from random baselines." But the differentiation is specifically about trained vs random **encoder**, not training per se. Shuffled features (broken identity) with trained encoder achieve ~0.48 absorption.

**Fix**: Reframe to: "Multi-child proportional ablation differentiates trained-encoder SAEs from random-encoder SAEs."

### 6. H2 Correlation Driven by 1.5% Non-Zero Outliers

Of 1024 features, only 15 show non-zero absorption (98.5% zeros). The rho=+0.17 correlation is entirely driven by these outliers. Zero-inflated distributions require specialized analysis methods.

**Fix**: Report zero-inflation explicitly. Note that correlation analysis is unreliable with 98.5% zeros.

### 7. Table 6 Contains Degenerate Percentage Values

"134,717,856%" and "1.52 billion%" are not meaningful metrics. Remove the percentage column.

---

## Minor Issues

8. **'Overlap method' undefined**: H_Safe uses an "overlap method" never defined in the paper.
9. **H_Downstream untested**: Listed as contribution but never executed.
10. **Figure 1 lacks textual walkthrough**: Caption only, no prose description.

---

## What Works

1. **Multi-child proportional ablation**: Genuinely novel methodology solving a real saturation problem.
2. **Honest negative results**: H2 and H_Safe failures reported without spin.
3. **H_Comp multi-seed validation**: 6 levels × 3 seeds provides robust evidence (even though it falsifies the hypothesis).
4. **Clear hypothesis structure**: Pre-registered falsification criteria.
5. **Encoder-driven finding**: Real SAE experiment confirms absorption is encoder-driven, not decoder-geometric.

---

## Path to 7.0 (Weak Accept)

1. Fix H_Mech narrative to reflect real SAE finding (B=D, C=A, decoder effect = 0)
2. Report H_Comp full experiment honestly (monotonic=falsified, R²=0.04)
3. Verify H_Safe p-value discrepancy and correct
4. Acknowledge deterministic absorption in H1

## Path to 8.0 (Accept)

1. Redesign H_Mech with independently trained conditions
2. Multi-seed validation for H3 steering
3. Include experiment code in repository
4. Execute H_Downstream or remove from contributions

---

## Risks

1. Paper narrative will be flagged by reviewers aware of the real SAE H_Mech data
2. Falsified hypothesis (H_Comp monotonic) discovered during review damages credibility
3. H_Safe p-value discrepancy raises questions about data integrity
4. Deterministic absorption undermines statistical significance claims

## Evidence Gaps

1. H_Mech text not updated from synthetic to real SAE finding
2. H_Comp full results contradict paper but not reported
3. H_Safe p-value needs verification
4. No multi-seed H3 validation
5. No experiment code for independent verification
