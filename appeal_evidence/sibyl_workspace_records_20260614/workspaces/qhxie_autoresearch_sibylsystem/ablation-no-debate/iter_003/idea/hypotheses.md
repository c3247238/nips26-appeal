# Testable Hypotheses

## Primary Hypotheses (Front-runner: cand_p1)

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

**Status**: PASSED (iter_001 pilot evidence)

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

**Falsification Criterion**: If Condition D >> Condition B (decoder contributes significantly). **STATUS: NOT FALSIFIED**

---

### H2: Absorption Rate Correlates With Feature Frequency

**Status**: FAILED (iter_001 pilot evidence - wrong direction)

**Original Hypothesis**: Features with lower activation frequency show systematically higher absorption rates.

**Prior Pilot Result**: Spearman rho = +0.171 (positive correlation)
- Expected: rho < -0.3 (negative)
- Observed: rho = +0.171 (positive)

**Interpretation**: Higher-frequency features are MORE absorbed, not less. The encoder-driven mechanism suggests this is because frequent hierarchical patterns are efficiently encoded by routing parent activations through children.

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

**Unexpected Finding**: Absorbed features are 1.62x MORE sensitive to steering than non-absorbed features. This suggests absorbed features retain parent direction information and can be "nudged" back.

**Falsification Criterion**: No sensitivity difference between absorbed and non-absorbed. **STATUS: NOT FALSIFIED**

---

### H_Safe: Safety-Critical Features Show Elevated Absorption Rates

**Status**: NOT TESTED on real SAEs (synthetic pilot failed)

**Hypothesis**: Features annotated as safety-critical (deception, jailbreak, harm, manipulation) show higher absorption rates than matched non-safety features in real Gemma Scope SAEs.

**Synthetic Pilot Result** (iter_001, using random feature indices):
- Safety features: mean absorption = 0.907
- Non-safety features: mean absorption = 0.906
- Mann-Whitney p = 0.665 (no difference)

**Why Synthetic Failed**: Synthetic SAE features lack semantic content. Safety vs. non-safety is meaningless on random feature indices.

**Required for full experiment**:
1. Install SAELens with Gemma Scope pretrained SAEs
2. Select 20 safety-relevant features from Neuronpedia annotations
3. Match with 20 non-safety features (by activation frequency and layer)
4. Measure absorption via multi-child proportional method
5. Mann-Whitney test comparing absorption distributions

**Falsification**: No significant difference in absorption rates.

**Novelty**: 9/10 - No prior work examines whether safety-critical features are disproportionately absorbed.

---

## New Hypotheses (iter_003)

### H_Comp: Absorption Increases With Hierarchy Strength

**Status**: NOT TESTED

**Hypothesis**: Feature absorption rate increases monotonically with feature hierarchy strength (parent-child cosine similarity).

**Measurement**: Vary hierarchy overlap across {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}. Fit absorption vs. overlap curve.

**Expected outcome**: Absorption → 1.0 as overlap → 1.0. Fit quality R² > 0.8.

**Connection to H_Mech**: If absorption is encoder-driven, the encoder should learn stronger parent-child correlations as hierarchy strength increases.

---

### H_Pareto: Sensitivity-Absorption Pareto Frontier Exists

**Status**: NOT TESTED

**Hypothesis**: There exists an irreducible Pareto frontier between feature sensitivity (Hu et al., 2025) and absorption rate. No SAE can simultaneously maximize sensitivity and minimize absorption.

**Measurement**:
- Vary L0 targets: {16, 32, 64, 128}
- At each L0, measure both absorption rate and feature sensitivity
- Fit Pareto frontier curve

**Expected outcome**: Frontier shape matches theoretical prediction from absorption-sensitivity uncertainty relation.

**Connection to H_Mech**: The encoder's learned alignment creates the sensitivity-absorption trade-off.

---

## Summary Table

| ID | Candidate | Status | Metric | Threshold |
|----|-----------|--------|--------|-----------|
| H1 | front_runner | **PASSED** | t-test | p < 0.05, delta > 0.15 |
| H_Mech | front_runner | **PASSED** | 2x2 factorial | B ≈ D, C ≈ A |
| H2 | front_runner | **FAILED** | Spearman rho | rho < -0.3 (failed: +0.171) |
| H3 | front_runner | **PASSED** | Sensitivity ratio | > 1.0 (observed: 1.62) |
| H_Safe | front_runner | **NOT TESTED** | Mann-Whitney | p < 0.05 |
| H_Comp | backup | NOT TESTED | R² | R² > 0.8 |
| H_Pareto | backup | NOT TESTED | Frontier fit | Matches theory |

---

## Negative Results Documentation (iter_001)

### H2: Frequency-Absorption Correlation

**Finding**: Positive correlation (rho = +0.171) contradicts competitive exclusion hypothesis (predicted negative).

**Implication**: The mechanism underlying absorption is not competitive exclusion. The encoder-driven mechanism explains this: frequent hierarchical patterns are efficiently encoded by routing parent activations through children (efficient coding, not competitive exclusion).

**Recommendation**: Archive as honest negative result. Reframe from "competitive exclusion" to "efficient coding."

### H_Safe: Synthetic Pilot

**Finding**: No difference between safety and non-safety features on synthetic SAE (p = 0.665).

**Implication**: Synthetic SAE features lack semantic content. Real Gemma Scope SAEs required for meaningful safety analysis.

**Recommendation**: Do not report synthetic H_Safe result. Proceed with real SAE analysis only.

---

## Action Items (iter_003)

1. **H_Mech full validation** (Priority 1): 5 seeds, stochastic hierarchy, confirm encoder-driven mechanism
2. **H_Comp** (Priority 2): Hierarchy strength sweep, test monotonic prediction
3. **H_Pareto** (Priority 3): Sensitivity-absorption frontier measurement
4. **H_Safe on Gemma Scope** (Priority 4): Highest novelty, requires SAELens installation
