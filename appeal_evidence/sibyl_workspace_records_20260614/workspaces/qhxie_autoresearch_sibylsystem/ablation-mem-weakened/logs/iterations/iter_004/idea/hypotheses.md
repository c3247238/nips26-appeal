# Testable Hypotheses with Expected Outcomes (Revised)

## Revision Notice

This document has been revised based on pilot evidence that decisively falsified H6 (the Local Inhibition Graph's core hypothesis). The inhibition graph framework (H6-H10) has been dropped. The remaining hypotheses are reframed around null-result reporting and the one robust finding (H5).

---

## Completed Hypotheses: Status Summary

### H1: Absorption Degrades Steering Effectiveness (Raw Metric)
**Result: FALSIFIED** --- r = +0.008 (layer 4), r = -0.301 (layer 8), p > 0.05. No significant correlation.

### H1b: Absorption Degrades Delta-Corrected Steering Effectiveness
**Result: NOT SUPPORTED AFTER CORRECTION** --- r = -0.431, p = 0.028 (layer 8, uncorrected). Does NOT survive Bonferroni (p = 0.334) or BH-FDR (q = 0.107).

### H2: Absorption Degrades Sparse Probing Accuracy
**Result: FALSIFIED** --- r = -0.003 (layer 4), r = -0.107 (layer 8), p > 0.05.

### H3: Consistency Across Configurations
**Result: FALSIFIED** --- Opposite-sign slopes; CV = 1.079 (fails CV < 0.5).

### H4: Absorption Affects Steering Efficiency, Not Capability (EC50)
**Result: NOT SUPPORTED** --- L4: r=-0.166, p=0.439; L8: r=+0.180, p=0.380.

### H5: Absorption Affects Recall (Coverage), Not Precision (Selectivity)
**Result: SUPPORTED** --- Precision = 1.0 universally at k >= 5; recall varies (0.05-1.0).

### H6: Decoder Correlation Graph Predicts Absorption Pairs
**Result: FALSIFIED** --- Precision@20 = 0.0 (predicted >= 0.10). Enrichment = 0.0x. Fisher p = 1.0.

---

## Reframed Hypotheses: Optimal Compression Framework

### H7 (Primary): Absorption is Rate-Distortion Optimal

**Statement:** Under hierarchical co-occurrence and sparsity constraints, absorption minimizes the rate (sparsity loss) while preserving decoder alignment (reconstruction quality).

**Evidence:**
- Chanin et al. Proposition 2: absorption reduces sparsity loss by Delta L_sp = p_11 per parent-child pair.
- Project data: precision = 1.0 (decoder alignment preserved), recall varies (encoder activation suppressed).
- Steering success remains 100% even at 24.2% absorption (decoder direction intact).

**Formalization:**
- For parent feature P and child feature C with co-occurrence probability p_11:
  - Expected sparsity loss without absorption: L_sp = p_11 * 2 + p_10 * 1 + p_01 * 1
  - Expected sparsity loss with full absorption: L_sp' = p_11 * 1 + p_10 * 1 + p_01 * 1
  - Savings: Delta L_sp = p_11 > 0
- The SAE achieves lower rate (sparsity) at the cost of reduced recall (parent feature coverage).

**Why this matters:** Reframes absorption from "failure mode" to "optimal compression strategy."

---

### H8 (Secondary): Precision-Recall Asymmetry Reflects Information Redistribution

**Statement:** The precision-recall asymmetry (precision invariant, recall variable) reflects that absorption redistributes parent feature information into child feature decoder directions, rather than destroying it.

**Evidence:**
- Precision = 1.0: decoder directions retain correct semantic content (selectivity preserved).
- Recall varies: encoder suppresses redundant activations (coverage reduced).
- Feature U (24.2% absorption) still steers 100%: information is redistributed, not lost.
- EC50 shows no efficiency degradation: decoder alignment unaffected.

**Formalization:**
- Let I(P; S) = mutual information between parent feature P and stimulus S.
- Absorption does not reduce I(P; S); it changes how I(P; S) is encoded:
  - Without absorption: I(P; S) encoded in both P and C activations.
  - With absorption: I(P; S) encoded primarily in C's decoder direction.
- Precision measures whether P's decoder direction is correct: unaffected.
- Recall measures whether P's encoder activates: reduced.

**Why this matters:** Explains the project's strongest finding (H5) through information theory.

---

### H9 (Exploratory): Absorption Rate Correlates with Hierarchical Co-occurrence Strength

**Statement:** Features with stronger parent-child co-occurrence (higher p_11) exhibit higher absorption rates.

**Evidence:**
- Chanin et al. proved absorption is a logical consequence of sparsity loss under hierarchical features.
- The proof shows dL_sp/ddelta = -p_11 < 0, so higher p_11 drives stronger absorption.
- Project data: high-absorption features (U: 24.2%, H: 19.0%, S: 16.0%) may have stronger hierarchical structure.

**Formalization:**
- Compute p_11 for each first-letter feature (fraction of child prompts where parent also fires).
- Test correlation: absorption_rate vs. p_11.
- Prediction: r > 0.3, p < 0.05.

**Falsification criterion:** If r < 0.2 or p > 0.05, co-occurrence strength does not explain absorption variance.

**Why this matters:** Tests whether absorption is driven by hierarchical structure (optimal compression) or other factors.

---

### H10 (Exploratory): Random SAE Baselines Show Similar Absorption Patterns

**Statement:** Random SAE baselines (frozen decoder weights) exhibit absorption-like patterns, confirming that absorption is partially a structural artifact of overcomplete dictionaries.

**Evidence:**
- Korznikov et al. (2026): random SAE baselines match trained SAEs on standard metrics.
- If random SAEs also show absorption, the phenomenon is not purely a learned failure.

**Formalization:**
- Construct random SAE baseline (frozen decoder, frozen encoder).
- Run Chanin absorption metric on same first-letter features.
- Compare absorption rates: trained vs. random.
- Prediction: Random SAE shows non-zero absorption rates (structural artifact).

**Why this matters:** Addresses the "Sanity Checks" challenge directly. If random SAEs absorb, absorption is structural, not learned.

---

## Hypothesis Testing Summary (Final)

| Hypothesis | Type | Status | Key Evidence |
|------------|------|--------|-------------|
| H1 | Primary | SUPPORTED (null) | r=+0.008 (L4), r=-0.301 (L8), p>0.05 |
| H1b | Primary | NOT SUPPORTED (after correction) | p=0.028 uncorrected, p=0.334 Bonferroni |
| H2 | Primary | SUPPORTED (null) | r=-0.003 (L4), r=-0.107 (L8), p>0.05 |
| H3 | Primary | SUPPORTED (null) | CV=1.079, opposite signs |
| H4 | Secondary | SUPPORTED (null) | r=-0.166 (L4), r=+0.180 (L8), p>0.05 |
| H5 | Secondary | SUPPORTED | Precision=1.0, recall varies |
| H6 | Secondary | FALSIFIED | precision@20=0.0, enrichment=0.0 |
| H7 | Primary (new) | To be analyzed | Rate-distortion framework |
| H8 | Secondary (new) | To be analyzed | Information redistribution |
| H9 | Exploratory | To be tested | p_11 vs. absorption_rate |
| H10 | Exploratory | To be tested | Random SAE baseline |

## Integration with Prior Findings

| Prior Finding | New Interpretation (Optimal Compression) |
|---|---|
| Precision = 1.0 universally (H5) | Decoder directions preserve semantic content; information redistributed, not lost |
| Recall varies widely | Encoder optimally suppresses redundant activations to maintain sparsity |
| Feature U (24.2% abs, 100% steering) | Decoder alignment intact; steering works because direction is correct |
| EC50 shows no efficiency degradation | Inhibition affects activation probability, not decoder geometry |
| H1-H4 null results | Absorption does not degrade performance because it is optimal compression |
| H6 falsified | Decoder correlations do not capture absorption; mechanism is not competitive suppression via decoder geometry |

## Risk Assessment

| Hypothesis | Risk | Mitigation |
|---|---|---|
| H7 (optimal compression) | May be seen as apologetics | Frame as "understanding mechanism" not "defending status quo"; ground in Chanin et al.'s theorem |
| H8 (information redistribution) | Hard to quantify formally | Use mutual information estimation; acknowledge limitations |
| H9 (co-occurrence correlation) | May be weak or non-significant | Report honestly; if null, absorption is not driven by co-occurrence |
| H10 (random baseline) | Random SAE may not have detectable features | Use synthetic data or feature-agnostic metric |
