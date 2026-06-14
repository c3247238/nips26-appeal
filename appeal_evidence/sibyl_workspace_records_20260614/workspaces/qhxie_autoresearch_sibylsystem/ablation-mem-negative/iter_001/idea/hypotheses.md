# Research Hypotheses: Unsupervised Feature Absorption Detection and Dynamic Compensation

## Primary Hypotheses (UAD + DFDA)

### H1: Unsupervised Detection Feasibility
**Statement**: A co-occurrence-based clustering method can detect absorbed feature pairs with F1 >= 0.6 compared to Chanin et al.'s supervised method, without requiring probe directions or ground-truth labels.

**Falsification Criteria**:
- If F1 < 0.5 on GPT-2 Small, H1 is falsified.
- If precision < 0.4 or recall < 0.5, H1 is falsified.
- If UAD outputs are no better than random feature pair selection (assessed by AUC-ROC against Chanin labels), H1 is falsified.

**Measurement**:
- Precision = TP / (TP + FP), where TP = correctly identified absorbed pairs
- Recall = TP / (TP + FN), where FN = missed absorbed pairs
- F1 = 2 * precision * recall / (precision + recall)
- Ground truth: Chanin et al. supervised labels on first-letter features

**Control**: Random feature pair selection baseline; co-occurrence thresholding without clustering.

---

### H2: Cross-Model Generalization
**Statement**: UAD achieves F1 >= 0.55 on models larger than GPT-2 Small (Gemma-2B, Pythia-2.8B), confirming that co-occurrence patterns are a general signature of absorption.

**Falsification Criteria**:
- If F1 < 0.5 on any of Gemma-2B or Pythia-2.8B, H2 is falsified.
- If F1 variance across models > 0.15, H2 is falsified (lack of generalizability).

**Measurement**:
- Same F1 metrics as H1, computed on each model independently
- Cross-model consistency: Spearman correlation of feature pair rankings across models

**Control**: Same UAD algorithm with fixed hyperparameters across all models.

---

### H3: Dynamic Compensation Efficacy
**Statement**: DFDA recovers >10% of absorbed parent feature activation (measured by MSE reduction) using <0.01% of SAE parameters, without increasing reconstruction error.

**Falsification Criteria**:
- If mean MSE improvement < 5%, H3 is falsified.
- If reconstruction MSE increases > 2%, H3 is falsified.
- If fewer than 60% of absorbed pairs show positive improvement, H3 is falsified.

**Measurement**:
- MSE improvement = (baseline_MSE - compensated_MSE) / baseline_MSE
- Reconstruction MSE change = (compensated_recon_MSE - original_recon_MSE) / original_recon_MSE
- Parameter count ratio = DFDA_params / SAE_params

**Control**: No-compensation baseline; random residual injection.

---

### H4: End-to-End Pipeline Validity
**Statement**: The UAD+DFDA pipeline achieves measurable improvement on downstream sparse probing accuracy for concepts identified as absorbed by UAD.

**Falsification Criteria**:
- If probe accuracy improvement on absorbed concepts < 5 percentage points, H4 is falsified.
- If improvement is not statistically significant (p > 0.05, paired t-test), H4 is falsified.

**Measurement**:
- Sparse probing accuracy before and after DFDA compensation
- Paired t-test across absorbed concept set
- Effect size (Cohen's d)

**Control**: Non-absorbed concept accuracy (should not change significantly).

---

## Supplementary Hypotheses (Retained from Iteration 1, Lower Priority)

### H-S1: Collision Rate vs True Absorption Correlation
**Statement**: The "collision rate" proxy metric (shared features across first-letter concepts) correlates with true absorption rate (Chanin protocol) at r > 0.5.

**Falsification Criteria**:
- If Spearman r < 0.3, the proxy is not useful.

**Note**: This validates whether the CAAB collision rate experiments from Iteration 1 can be reinterpreted meaningfully.

---

### H-S2: Absorption-Downstream Relationship (Revisited)
**Statement**: True absorption rate (not collision rate) negatively correlates with sparse probing accuracy at r < -0.3 when controlling for reconstruction quality and sparsity.

**Falsification Criteria**:
- If partial correlation r > -0.1 or p > 0.05, H-S2 is falsified.

**Note**: The original H2 (r=0.103, p=0.87) used collision rate, not true absorption. This revised hypothesis uses proper absorption detection.

---

## Exploratory Hypotheses (Future Work)

### H-E1: Semantic Hierarchy Generalization
UAD achieves comparable F1 on WordNet semantic hierarchies (animal -> mammal -> dog) as on first-letter hierarchies.

### H-E2: Multi-Concept Absorption
Some SAE features absorb multiple parent concepts simultaneously. UAD can detect these "super-absorber" features via cluster size analysis.

### H-E3: Absorption Severity Spectrum
Absorption is not binary. UAD confidence scores correlate with the severity of parent feature suppression (measured by activation ratio).

---

## Hypothesis Dependency Graph

```
H1 (UAD feasibility on GPT-2 Small)
  |
  +---> H2 (cross-model generalization) [depends on H1 success]
  |
  +---> H4 (end-to-end pipeline) [depends on H1 for detection]
          |
          +---> H3 (DFDA efficacy) [depends on H1 for pair identification]

H-S1 (collision vs true absorption) [independent, validation-only]
H-S2 (absorption-downstream) [depends on H-S1 if using CAAB data]
```

## Pre-registered Analysis Plan

1. **Primary Analysis**: For H1-H4, report mean F1 / MSE improvement / accuracy with bootstrap 95% CI across seeds.
2. **Multiple Comparisons**: Use Bonferroni correction when testing across multiple models (alpha = 0.05 / 3 = 0.017 per model).
3. **Negative Result Handling**: If any primary hypothesis is falsified, report prominently and discuss implications.
4. **Power Analysis**: Report achieved power for all primary hypotheses. If power < 0.8, note as limitation.
5. **Replication**: All key results replicated across 3 random seeds (42, 123, 456).

## Go/No-Go Gates

| Gate | Condition | If Passed | If Failed |
|------|-----------|-----------|-----------|
| G1 | H1: F1 >= 0.6 on GPT-2 Small | Proceed to H2 | Re-tune UAD hyperparameters; if still failing, PIVOT |
| G2 | H2: F1 >= 0.55 on >=2 larger models | Proceed to H3/H4 | If F1 >= 0.5 but < 0.55, proceed with caveat; if < 0.5, PIVOT |
| G3 | H3: Mean improvement >= 10% on >=8 pairs | Proceed to H4 | Downgrade DFDA to "pilot result"; paper focuses on UAD only |
| G4 | H4: Pipeline improves probing accuracy | Paper is complete | Report UAD+DFDA as separate contributions without end-to-end claim |
