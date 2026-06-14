# Testable Hypotheses

## Front-Runner: Sensitivity Floor

### H-SF1: Structural Emptiness
**H-SF1 (Structural Emptiness)**: Even with larger sample (N > 200), Q2 (absorbed + high-sensitivity) and Q4 (non-absorbed + high-sensitivity) remain empty or near-empty (< 5% of features).

**Falsification criterion**: If > 10% of features fall into Q2 or Q4 at N = 200, sensitivity floor is falsified.

**Expected outcome**: Q2 + Q4 < 5% even at larger N; sensitivity failures are structurally impossible, not statistically rare.

**Pilot**: ~30 min on 1 A100 (200 features, Chanin + Tian protocols)

---

### H-SF2: U-Shaped Relationship
**H-SF2 (U-Shaped Relationship)**: Sensitivity S(A) as a function of absorption A follows a U-shape: S is low at both A ~ 0 (sparse regions -> low reliability) and A ~ 1 (absorbed -> low specificity), with maximum sensitivity at intermediate absorption.

**Falsification criterion**: If quadratic coefficient a <= 0 (no U-shape), sensitivity floor mechanism is weakened.

**Expected outcome**: S(A) = aA^2 + bA + c with a > 0; maximum at intermediate A.

**Note**: The observed r = 0.59 may be a linear approximation of this U-shape.

---

### H-SF3: Frequency Mediation
**H-SF3 (Frequency Mediation)**: Feature activation frequency mediates the absorption-sensitivity correlation. After controlling for frequency, partial r(absorption, sensitivity | frequency) < 0.3.

**Falsification criterion**: If partial r > 0.5 after controlling for frequency, frequency is not the sole mediator.

**Expected outcome**: Low-frequency features get absorbed (less gradient signal) AND become insensitive (fewer training examples).

---

### H-SF4: Geometric Density Mediation
**H-SF4 (Geometric Mediation)**: Geometric density (mean k-NN cosine similarity in activation space) mediates the absorption-sensitivity correlation. Partial r(absorption, sensitivity | density) < 0.3.

**Falsification criterion**: If partial r > 0.5 after controlling for density, density is not the mediator.

**Expected outcome**: Dense regions cause both absorption (decoder overlap) and insensitivity (vague activation patterns).

---

## Alternative A: Sensitivity-First (Pivot if H-SF1 Fails)

### H-A: Q3 Equals Random Baseline
**H-A (Sensitivity-First)**: Low-sensitivity features (Q3: low-absorption + low-sensitivity) show steering effect sizes statistically indistinguishable from random baseline.

**Falsification criterion**: If Q3 steering is significantly above random baseline (0.73), sensitivity is not sufficient to explain Sanity Check.

**Expected outcome**: Q3 ~= random; both are low-sensitivity regardless of absorption.

---

### H-B: Absorption Adds No Predictive Value
**H-B (Absorption Metric Validity)**: Absorption status does NOT predict steering effectiveness after controlling for sensitivity.

**Falsification criterion**: If absorbed features show different steering than non-absorbed within Q3, absorption adds independent predictive value.

**Expected outcome**: Within Q3, absorbed vs non-absorbed shows no steering difference.

---

## Alternative B: Geometric Density (Pivot if H-SF3 and H-SF4 Fail)

### H-B1: Density-Absorption Correlation
**H-B1 (Density-Absorption)**: Absorbed features have higher geometric density than non-absorbed features.

**Falsification criterion**: If r(density, absorption) < 0, density is not associated with absorption.

**Expected outcome**: Dense regions are more vulnerable to absorption.

---

### H-B2: Density-Sensitivity Correlation
**H-B2 (Density-Sensitivity)**: Low-sensitivity features have higher geometric density than high-sensitivity features.

**Falsification criterion**: If r(density, sensitivity) > 0 (positive), density does not explain sensitivity failure.

**Expected outcome**: Dense regions produce vague activation patterns with low sensitivity.

---

## Alternative C: Layer-wise UAS Saturation (Control Experiment)

### H-C1: Layer-Dependent Saturation
**H-C1 (Layer Variance)**: UAS saturation is layer-dependent. Some layers (e.g., 4, 6, 10) show UAS variance while others (e.g., 8) show saturation.

**Falsification criterion**: If ALL layers show UAS std < 0.05, saturation is universal.

**Expected outcome**: Layer 8 is special; other layers permit absorption measurement.

---

### H-C2: Sensitivity as Proxy
**H-C2 (Proxy Validity)**: At non-saturated layers, sensitivity (paraphrase AUC) correlates with UAS absorption (r > 0.4).

**Falsification criterion**: If r(UAS, sensitivity) < 0.3 at non-saturated layers, sensitivity cannot substitute for UAS.

**Expected outcome**: Sensitivity works where UAS fails.

---

## Alternative D: Steering Diffusion

### H-D1: Decoder-Neighbor Overlap
**H-D1 (Diffusion Mechanism)**: Absorbed features have higher decoder-neighbor overlap (k=20 NN cosine similarity) than non-absorbed features.

**Falsification criterion**: If absorbed and non-absorbed have equal neighbor overlap, diffusion is not the mechanism.

**Expected outcome**: Absorbed features live in dense geometric regions.

---

### H-D2: Non-Linear Beta Dependence
**H-D2 (Non-Linear Diffusion)**: Diffusion is non-linear in beta. At beta <= 10, no neighbor activates (null result). At beta = 20, multiple neighbors activate simultaneously.

**Falsification criterion**: If absorption-steering correlation exists at beta <= 10, diffusion is not the mechanism.

**Expected outcome**: Null at beta <= 10 (confirming iter_004); significant effect at beta = 20.

---

## Expected Outcomes Summary

| Hypothesis | Expected | Falsification |
|-----------|----------|---------------|
| H-SF1 (emptiness) | Q2+Q4 < 5% | Q2+Q4 > 10% |
| H-SF2 (U-shape) | a > 0 | a <= 0 |
| H-SF3 (frequency) | partial r < 0.3 | partial r > 0.5 |
| H-SF4 (density) | partial r < 0.3 | partial r > 0.5 |
| H-A (sensitivity-first) | Q3 ~= random | Q3 > random |
| H-B (absorption validity) | No effect within Q3 | Absorbed != non-absorbed |
| H-C1 (layer variance) | Some layers show variance | All layers saturate |
| H-C2 (proxy validity) | r > 0.4 | r < 0.3 |
| H-D1 (overlap) | Absorbed > overlap | Equal overlap |
| H-D2 (non-linear) | Null at beta<=10 | Effect at beta<=10 |
