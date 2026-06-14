# Experiment Critique: Feature Absorption in SAEs

## Overview

The project conducted 8+ major analyses across 4 layers of GPT-2 Small, plus cross-model validation on Pythia-70M and two pilot experiments (H9, H10). The experimental design is generally sound, but there are significant methodological issues that undermine the validity of key conclusions.

## Critical Issues

### 1. The Chanin Absorption Metric Validity is in Question (H10)

**Finding:** Random SAE shows 8x HIGHER absorption than trained SAE (0.278 vs 0.034, p < 0.001).

**Implication:** The Chanin differential correlation metric is not specific to learned structure. It detects patterns that exist in random, untrained SAEs. This means:
- The absorption rates reported for trained SAEs may reflect structural artifacts of overcomplete dictionaries, not learned pathologies
- The downstream correlations (H1-H4) may be correlations with structural artifacts
- The precision-recall decomposition may decompose artifacts, not real phenomena

**Severity:** CRITICAL. This undermines the empirical foundation of the entire project.

**Evidence:** h10_random_sae_baseline.json shows random SAE mean absorption = 0.278, trained = 0.034. Paired t-test: t = -6.745, p = 4.55e-07. The correlation between trained and random absorption rates is r = 0.023, p = 0.913---no relationship.

**Recommendation:** The paper must integrate H10 and honestly report that the Chanin metric's validity is questionable. The honest conclusion: trained SAEs show LESS absorption-like patterns than random SAEs, suggesting training actually suppresses structural artifacts.

### 2. H9 Co-occurrence Analysis is Tautological

**Finding:** Perfect negative correlation (r = -1.0, p < 0.001) between p_11 and absorption_rate.

**Implication:** This is a mathematical artifact, not evidence. By construction: p_11 + absorption_rate = 1.0 (if parent fires, child is not absorbing; if parent does not fire, child is counted as absorbed).

**Severity:** CRITICAL. The operationalization is flawed and any conclusions based on it are invalid.

**Evidence:** h9_cooccurrence_analysis.json shows p_11 values ranging from 0.1 to 1.0, with absorption_rate = 1.0 - p_11 for all features. The correlation with existing Chanin absorption is r = -0.033, p = 0.874---no relationship.

**Recommendation:** Remove H9 from any claims. Acknowledge the operationalization flaw explicitly.

### 3. Zero Hypotheses Survive Multiple Comparison Correction

**Finding:** All 12 tests (H1, H1b, H2 across L4/L8 with Pearson and Spearman) fail Bonferroni correction (alpha = 0.00417). The closest is H1b L8 Spearman (p = 0.0089, Bonferroni p = 0.107, BH-FDR q = 0.107).

**Implication:** There is no statistically significant evidence for ANY absorption-downstream correlation after proper correction.

**Severity:** MAJOR. The paper prominently features H1b (r = -0.431, p = 0.028) as "the strongest signal" but this does not survive correction.

**Evidence:** correlation_report_full.json explicitly states:
- "n_rejected_bonferroni: 0"
- "n_rejected_bh: 0"
- "any_significant_after_bonferroni: false"
- "any_significant_after_bh: false"

**Recommendation:** State prominently that ZERO hypotheses survive correction. The H1b result should be framed as a non-significant trend.

### 4. Small Sample Size (n=26) Limits Statistical Power

**Finding:** With 26 first-letter features and only 4 high-absorption features at layer 8, the study has low power to detect small-to-medium effects.

**Implication:** The null results may reflect insufficient power rather than true null effects. The paper's conclusion that "absorption does not degrade downstream tasks" is underpowered.

**Severity:** MAJOR. The paper acknowledges this but does not quantify the power.

**Evidence:** With n=26, power to detect r = 0.5 at alpha = 0.05 is approximately 80%. But the expected effect size is unknown. If the true effect is r = 0.3, power is only ~40%.

**Recommendation:** Report a power analysis. State the minimum detectable effect size given n=26 and alpha=0.00417 (Bonferroni). This will likely show the study is underpowered for small effects.

### 5. H8 Claim Lacks Verifiable Data Source

**Finding:** Section 4.5 claims r = +0.12, p = 0.55 for total incoming inhibition vs. absorption rate, but no data file contains this statistic.

**Implication:** The claim is unverifiable.

**Severity:** MAJOR for reproducibility.

**Evidence:** h6_inhibition_graph.json contains top-k correlations but not per-feature total incoming inhibition.

**Recommendation:** Either generate the data file or remove specific numbers.

### 6. Single Model Family (GPT-2 Small)

**Finding:** All primary experiments use GPT-2 Small (124M parameters). Cross-model validation on Pythia-70M was inconclusive.

**Implication:** Results may not generalize to larger models or different architectures.

**Severity:** MEDIUM. Acknowledged in limitations but not mitigated.

**Evidence:** Pythia-70M cross-validation shows no significant correlation (r = -0.041, p = 0.841).

**Recommendation:** Frame all conclusions as specific to GPT-2 Small. Avoid general claims.

### 7. First-Letter Features Are Artificial

**Finding:** The feature set (A-Z first-letter features) is a shallow, orthographic hierarchy, not a natural semantic hierarchy.

**Implication:** Absorption dynamics may differ for semantic hierarchies (e.g., animal -> dog -> poodle).

**Severity:** MEDIUM. Acknowledged but not tested.

**Recommendation:** Discuss how shallow vs. deep hierarchies might differ. The competitive suppression framework predicts stronger effects for deeper hierarchies, but this is untested.

### 8. Table 3 Values Unverified

**Finding:** Graph statistics in Table 3 (clustering coefficient, std edge weight) are not verifiable from available data.

**Implication:** Reproducibility gap.

**Severity:** MINOR.

**Recommendation:** Add raw graph data to supplementary materials.

### 9. Homeostatic Rebalancing Deferred

**Finding:** H10 (homeostatic rebalancing) was deferred because the graph does not identify correct parent-child relationships.

**Implication:** A claimed contribution ("First training-free post-hoc repair for absorption") is unvalidated.

**Severity:** MINOR. Honestly reported as deferred.

**Recommendation:** Either execute the experiment with a corrected graph or remove the claim from contributions.

### 10. EC50 Analysis Uses Only 6 Strength Levels

**Finding:** EC50 is estimated from 6 steering strengths (1.0, 2.0, 5.0, 10.0, 20.0, 50.0).

**Implication:** EC50 estimation may be imprecise with sparse dose levels. The Hill equation fitting may be unstable.

**Severity:** MINOR. The EC50 result is null regardless, so imprecision does not change the conclusion.

**Evidence:** ec50_analysis.json shows some features with EC50 = null (could not be estimated from the sparse data).

## Positive Aspects

1. **Multiple comparison correction is applied** (Bonferroni and BH-FDR) and honestly reported.
2. **Cross-layer validation** (L4 and L8) tests consistency.
3. **Random baseline subtraction** for delta-corrected analysis is methodologically sound.
4. **Precision-recall decomposition** is a useful methodological innovation.
5. **H6 falsification is honestly reported** with exact statistics.

## Summary

The experimental design is sound in principle but undermined by three critical issues: (1) the Chanin metric's validity is questionable (H10), (2) the co-occurrence analysis is tautological (H9), and (3) zero hypotheses survive multiple comparison correction. The small sample size and single-model limitation further constrain the conclusions. The methodological contributions (baseline correction, precision-recall, EC50) are valuable and reusable, but the empirical findings are weaker than the paper's framing suggests.
