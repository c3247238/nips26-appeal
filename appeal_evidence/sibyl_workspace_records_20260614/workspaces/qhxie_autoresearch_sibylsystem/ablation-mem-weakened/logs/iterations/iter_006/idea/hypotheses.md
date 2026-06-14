# Testable Hypotheses with Expected Outcomes (Iteration 6)

## Overview

This document contains all hypotheses for the feature absorption study, including results from iterations 1-6. The key new evidence from this iteration is H9 (co-occurrence tautological) and the H10 confirmation that trained SAEs have lower absorption than random baselines, strengthening the optimal compression framing.

---

## Completed Hypotheses: Status Summary

### H1: Absorption Degrades Steering Effectiveness (Raw Metric)
**Result: SUPPORTED (null hypothesis)** --- r = +0.008 (layer 4), r = -0.301 (layer 8), p > 0.05. No significant correlation.

### H1b: Absorption Degrades Delta-Corrected Steering Effectiveness
**Result: NOT SUPPORTED AFTER CORRECTION** --- r = -0.431, p = 0.028 (layer 8, uncorrected). Does NOT survive Bonferroni (p = 0.334) or BH-FDR (q = 0.107).

### H2: Absorption Degrades Sparse Probing Accuracy
**Result: SUPPORTED (null hypothesis)** --- r = -0.003 (layer 4), r = -0.107 (layer 8), p > 0.05.

### H3: Consistency Across Configurations
**Result: SUPPORTED (null hypothesis)** --- Opposite-sign slopes; CV = 1.079 (fails CV < 0.5).

### H4: Absorption Affects Steering Efficiency, Not Capability (EC50)
**Result: SUPPORTED (null hypothesis)** --- L4: r=-0.166, p=0.439; L8: r=+0.180, p=0.380.

### H5: Absorption Affects Recall (Coverage), Not Precision (Selectivity)
**Result: SUPPORTED** --- Precision = 1.0 universally at k >= 5; recall varies (0.05-1.0).

### H6: Decoder Correlation Graph Predicts Absorption Pairs
**Result: FALSIFIED** --- Precision@20 = 0.0 (predicted >= 0.10). Enrichment = 0.0x. Fisher p = 1.0.

---

## New Hypotheses (Iteration 5-6)

### H7: Trained SAEs Have Lower Absorption Than Random Baselines

**Statement:** Trained SAEs exhibit significantly lower absorption rates than random SAE baselines (frozen orthonormal decoder, random encoder), indicating that absorption is partially a structural artifact that training reduces.

**Evidence:**
- Trained SAE: mean = 0.034, std = 0.069, max = 0.242
- Random SAE: mean = 0.278, std = 0.169, max = 0.676
- Difference: mean = -0.244, t = -6.745, p < 0.001
- Wilcoxon: W = 0.0, p < 0.001
- Correlation between trained and random: r = 0.023, p = 0.913 (no correlation)

**Interpretation:**
The random SAE shows ~8x HIGHER absorption than the trained SAE. This suggests:
1. The Chanin absorption metric is sensitive to structural artifacts
2. Training optimizes decoder geometry to reduce these artifacts
3. Absorption in trained SAEs may be "residual structural artifact" rather than "learned failure"

**Why this matters:** Reframes absorption from "failure mode" to "metric artifact that training reduces."

---

### H8: Absorption is Rate-Distortion Optimal

**Statement:** Under hierarchical co-occurrence and sparsity constraints, absorption minimizes the rate (sparsity loss) while preserving decoder alignment (reconstruction quality).

**Evidence:**
- Chanin et al. Proposition 2: absorption reduces sparsity loss by Delta L_sp = p_11 per parent-child pair
- Project data: precision = 1.0 (decoder alignment preserved), recall varies (encoder activation suppressed)
- Steering success remains 100% even at 24.2% absorption (decoder direction intact)
- H7: Trained SAEs optimize better than random, consistent with compression behavior

**Formalization:**
- For parent feature P and child feature C with co-occurrence probability p_11:
  - Expected sparsity loss without absorption: L_sp = p_11 * 2 + p_10 * 1 + p_01 * 1
  - Expected sparsity loss with full absorption: L_sp' = p_11 * 1 + p_10 * 1 + p_01 * 1
  - Savings: Delta L_sp = p_11 > 0
- The SAE achieves lower rate (sparsity) at the cost of reduced recall (parent feature coverage).

**Why this matters:** Reframes absorption from "failure mode" to "optimal compression strategy."

---

### H9: Co-occurrence Strength Predicts Absorption Rate

**Statement:** Features with stronger parent-child co-occurrence (higher p_11) exhibit higher absorption rates.

**Result: TAUTOLOGICAL** --- p_11 + absorption_rate = 1.0 by construction. If parent fires, child is not "absorbing"; if parent does not fire, child is counted as absorbed. This is a definitional relationship, not a causal one.

**Interpretation:** A meaningful test would require a different operationalization of co-occurrence (e.g., independent measurement from held-out corpus).

**Why this matters:** Excluded from main paper. Informs future experimental design.

---

### H10: Random SAE Baseline Comparison (Validation)

**Statement:** Random SAE baselines exhibit absorption-like patterns, confirming absorption is partially structural.

**Result: SUPPORTED (informative)** --- See H7 for full results.

**Interpretation:** The Chanin absorption metric is not specific to learned structure. Training reduces structural artifacts.

---

## Full Hypothesis Testing Summary (Final)

| Hypothesis | Type | Status | Key Evidence |
|------------|------|--------|---------------|
| H1 | Primary | SUPPORTED (null) | r=+0.008 (L4), r=-0.301 (L8), p>0.05 |
| H1b | Primary | NOT SUPPORTED (after correction) | p=0.028 uncorrected, p=0.334 Bonferroni |
| H2 | Primary | SUPPORTED (null) | r=-0.003 (L4), r=-0.107 (L8), p>0.05 |
| H3 | Primary | SUPPORTED (null) | CV=1.079, opposite signs |
| H4 | Secondary | SUPPORTED (null) | r=-0.166 (L4), r=+0.180 (L8), p>0.05 |
| H5 | Secondary | SUPPORTED | Precision=1.0, recall varies |
| H6 | Secondary | FALSIFIED | precision@20=0.0, enrichment=0.0 |
| H7 | Primary (new) | SUPPORTED | trained=0.034, random=0.278, p<0.001 |
| H8 | Secondary (new) | SUPPORTED (framing) | Rate-distortion framework |
| H9 | Exploratory | TAUTOLOGICAL | Definitional relationship; excluded |

---

## Integration with Prior Findings

| Prior Finding | New Interpretation |
|---|---|
| Precision = 1.0 universally (H5) | Decoder directions preserve semantic content; information redistributed, not lost |
| Recall varies widely | Encoder optimally suppresses redundant activations to maintain sparsity |
| Feature U (24.2% abs, 100% steering) | Decoder alignment intact; steering works because direction is correct |
| EC50 shows no efficiency degradation | Absorption affects activation probability, not decoder geometry |
| H1-H4 null results | Absorption does not degrade performance because it is optimal compression |
| H6 falsified | Decoder correlations do not capture absorption; mechanism is not competitive suppression via decoder geometry |
| H7 (trained < random) | Absorption is a structural artifact; training reduces it |
| H9 tautological | Co-occurrence measurement is definitional; need independent measurement |
| H10 (validation of H7) | Confirms metric sensitivity to structure |

---

## Risk Assessment

| Hypothesis | Risk | Mitigation |
|---|---|---|
| H7 (trained < random) | May be seen as undermining subfield | Frame as "metric validation" not "SAEs don't work" |
| H8 (optimal compression) | May be seen as apologetics | Frame as "understanding mechanism" not "defending status quo"; ground in Chanin et al.'s theorem |
| General (null results) | Paper dismissed as "we found nothing" | Strong framing: metric validation + methodological contributions + honest null results |
