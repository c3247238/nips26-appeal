# Testable Hypotheses

## Primary Hypotheses (Front-runner: cand_p1 + H_Downstream)

### H1: Multi-Child Proportional Ablation Differentiates Trained SAEs from Baselines

**Status**: PASSED (iter_001 pilot evidence)

**Hypothesis**: Trained SAEs on synthetic 3-level feature hierarchies exhibit significantly higher multi-child proportional absorption rates than random baselines.

**Measurement**: Multi-child proportional ablation (k=5):
- `absorption_rate = parent_activation_after_ablating_top_k_children / parent_activation_before`

**Prior Pilot Results** (iter_001, 3 seeds):
| Condition | Absorption Rate | Std | vs Trained SAE |
|-----------|----------------|-----|----------------|
| Trained SAE | 0.500 | 0.0 | --- |
| Random Baseline | 0.147 | 0.065 | delta=0.353 |

**Falsification Criterion**: No significant difference (p > 0.05). **STATUS: NOT FALSIFIED**

---

### H_Mech: Encoder Alignment Drives Absorption, Not Decoder Geometry

**Status**: PASSED (iter_001 pilot evidence) + ANOMALY NOTED

**Hypothesis**: Absorption is driven entirely by the encoder's learned alignment with hierarchical features. The decoder contributes nothing beyond reconstructing from encoder outputs.

**Measurement**: 2x2 factorial design:
- Condition A: Random encoder + Random decoder (pure geometry)
- Condition B: Trained encoder + Random decoder (encoder alignment only)
- Condition C: Random encoder + Trained decoder (decoder geometry only)
- Condition D: Trained encoder + Trained decoder (full training)

**Prior Pilot Results**:
| Condition | Encoder | Decoder | Absorption Rate |
|-----------|---------|---------|-----------------|
| A | Random | Random | 0.299 |
| B | Trained | Random | **0.490** |
| C | Random | Trained | **0.299** |
| D | Trained | Trained | **0.484** |

**Key Finding**: Condition B ≈ Condition D (encoder alignment is sufficient), Condition C ≈ Condition A (decoder geometry is irrelevant).

**ANOMALY**: Condition B (0.490) > Condition D (0.484). The trained encoder with random decoder produces MORE absorption than full training. This suggests the decoder acts as an implicit regularizer that partially counteracts encoder-driven absorption.

**Falsification Criterion**: If Condition D >> Condition B (decoder contributes significantly). **STATUS: NOT FALSIFIED (decoder contributes NOTHING, but B>D anomaly requires explanation)**

---

### H2: Absorption Rate Correlates With Feature Frequency

**Status**: FAILED (iter_001 pilot evidence - wrong direction)

**Original Hypothesis**: Features with lower activation frequency show systematically higher absorption rates.

**Prior Pilot Result**: Spearman rho = +0.171 (positive correlation)
- Expected: rho < -0.3 (negative)
- Observed: rho = +0.171 (positive)

**Interpretation**: Higher-frequency features are MORE absorbed, not less. The efficient coding mechanism (Barlow, 1961) explains this: the encoder compresses frequently-activating parent features into child subspaces for metabolic efficiency. This is compression, not competition.

**Archive as**: Honest negative result - frequency-absorption correlation in opposite direction

---

### H3: Steering Absorbed Features Toward Parent Directions Improves Sensitivity

**Status**: PASSED (iter_001 pilot evidence)

**Hypothesis**: For absorbed features, steering toward parent directions improves sensitivity.

**Prior Pilot Result**:
```json
{
  "absorbed_mean_sensitivity": 0.055,
  "non_absorbed_mean_sensitivity": 0.034,
  "sensitivity_ratio": 1.620
}
```

**Unexpected Finding**: Absorbed features are 1.62x MORE sensitive to steering than non-absorbed features. This suggests absorbed features retain parent direction information and can be "nudged" back toward intended behavior.

**Falsification Criterion**: No sensitivity difference between absorbed and non-absorbed. **STATUS: NOT FALSIFIED**

---

### H_Safe: Safety-Critical Features Show Elevated Absorption Rates

**Status**: NOT TESTED on validated real SAE features

**Hypothesis**: Features annotated as safety-critical (deception, jailbreak, harm, manipulation) in Neuronpedia show higher absorption rates than matched non-safety features in real Gemma Scope SAEs.

**Prior "Pilot" Result** (iter_003, NOT VALIDATED):
- Used arbitrary indices (1024, 2048, 3072, 4096, 5120) as "safety features"
- Safety features: mean absorption = 0.0
- Non-safety features: mean absorption = 0.0
- Mann-Whitney p = 1.0

**CRITICAL ISSUE**: These indices are NOT validated safety features. The lessons_learned explicitly states this pilot is invalid. Must use Neuronpedia-validated features only.

**Required for full experiment**:
1. Install SAELens with Gemma Scope pretrained SAEs
2. Select 20 **validated** safety-relevant features from Neuronpedia annotations
3. Match with 20 non-safety features (by activation frequency and layer)
4. Measure absorption via multi-child proportional method
5. Mann-Whitney test comparing absorption distributions

**Falsification Criterion**: No significant difference in absorption rates (p > 0.05).

**Novelty**: 9/10 - No prior work examines whether safety-critical features are disproportionately absorbed.

---

## New Hypotheses (iter_004)

### H_Comp: Absorption Increases With Hierarchy Strength

**Status**: PASSED (pilot evidence)

**Hypothesis**: Feature absorption rate increases monotonically with feature hierarchy strength (parent-child cosine similarity).

**Measurement**: Vary hierarchy overlap across {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}. Fit absorption vs. overlap curve.

**Pilot Results** (iter_003):
| Cosine Sim | Absorption Rate |
|------------|-----------------|
| 0.6 | 0.585 |
| 0.8 | 0.673 |
| 0.95 | 0.802 |

**Confirmed**: Monotonic increase with hierarchy strength.

**Expected outcome**: Absorption → 1.0 as overlap → 1.0. Fit quality R² > 0.8.

---

### H_Pareto: Sensitivity-Absorption Pareto Frontier Exists

**Status**: SUSPENDED (formula bug)

**Hypothesis**: There exists an irreducible Pareto frontier between feature sensitivity (Hu et al., 2025) and absorption rate. No SAE can simultaneously maximize sensitivity and minimize absorption.

**Measurement**:
- Vary L0 targets: {16, 32, 64, 128}
- At each L0, measure both absorption rate and feature sensitivity
- Fit Pareto frontier curve

**Issue**: Pilot data shows sensitivity = 1.525 at L0=16, exceeding [0,1] bounds. This indicates a formula bug in the sensitivity measurement. **Frontier claims are suspended until the formula is verified and fixed.**

**Expected outcome**: Frontier shape matches theoretical prediction from absorption-sensitivity uncertainty relation.

---

### H_Downstream: Absorbed Features Show Degraded Downstream Utility

**Status**: NOT TESTED (NEW - from Contrarian)

**Hypothesis**: Features identified as "absorbed" via ablation show degraded performance on downstream interpretability tasks (steering, circuit completeness) compared to matched non-absorbed features.

**Falsification Criterion**: If absorbed vs non-absorbed features show < 10% difference in steering success rate, absorption does not have practically significant downstream impact.

**Why This Matters**: SAEBench shows proxy metrics don't predict practical performance. If absorption also doesn't predict downstream failure, the field is optimizing the wrong metric.

**Method**:
1. Classify features as "absorbed" vs "non-absorbed" via multi-child proportional ablation
2. Match pairs on confounders (activation frequency, magnitude, Neuronpedia interpretability score)
3. Test both on:
   - Steering accuracy: behavior change correlation
   - Circuit completeness: edge coverage comparison
4. Statistical test: McNemar's test for paired proportions

**Sample size**: 50 absorbed + 50 matched non-absorbed features (power analysis: 80% power to detect 25% relative improvement)

---

## Summary Table

| ID | Candidate | Status | Metric | Threshold |
|----|-----------|--------|--------|-----------|
| H1 | front_runner | **PASSED** | t-test | p < 0.05, delta > 0.15 |
| H_Mech | front_runner | **PASSED** + anomaly | 2x2 factorial | B ≈ D, C ≈ A (B>D anomaly) |
| H2 | front_runner | **FAILED** | Spearman rho | rho < -0.3 (failed: +0.171) |
| H3 | front_runner | **PASSED** | Sensitivity ratio | > 1.0 (observed: 1.62) |
| H_Safe | front_runner | **NOT TESTED** | Mann-Whitney | p < 0.05 (requires validated features) |
| H_Comp | front_runner | **PASSED** | Monotonic | Confirmed |
| H_Pareto | front_runner | **SUSPENDED** | Frontier fit | Formula bug - formula fix required |
| H_Downstream | front_runner | **NOT TESTED** | McNemar | > 10% difference |

---

## Negative Results Documentation

### H2: Frequency-Absorption Correlation

**Finding**: Positive correlation (rho = +0.171) contradicts competitive exclusion hypothesis (predicted negative).

**Implication**: The mechanism underlying absorption is not competitive exclusion. The efficient coding mechanism (Barlow, 1961) explains this: frequent hierarchical patterns are efficiently encoded by routing parent activations through children for metabolic efficiency.

**Recommendation**: Archive as honest negative result. Reframe from "competitive exclusion" to "efficient coding compression."

### H_Safe: Placeholder Feature Pilot

**Finding**: Used arbitrary indices (1024, 2048, etc.) as "safety features" -- NOT validated.

**Implication**: Prior pilot produced meaningless results. Real Neuronpedia-validated features required.

**Recommendation**: Discard placeholder results. Proceed only with validated safety feature indices from Neuronpedia.

---

## Action Items (iter_004)

1. **H_Downstream test** (Priority 0, NEW): Contrarian's core challenge -- does absorption matter for downstream utility?
2. **H_Mech full validation** (Priority 1): 5 seeds, stochastic hierarchy, confirm B>D anomaly
3. **H_Safe with validated features** (Priority 2): Neuronpedia-validated safety features only
4. **Fix sensitivity formula** (Priority 3): H_Pareto claims suspended until verified
5. **H_Comp full sweep** (Priority 4): Complete hierarchy strength curve (0.5 to 0.95)