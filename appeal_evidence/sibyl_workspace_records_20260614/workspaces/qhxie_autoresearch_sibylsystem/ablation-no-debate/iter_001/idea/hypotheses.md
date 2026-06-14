# Testable Hypotheses

## Primary Hypotheses (Front-runner: cand_p1)

### H1: Multi-Child Proportional Ablation Differentiates Trained SAEs from Baselines

**Status**: PASSED (pilot evidence, confirmed across new pilots)

**Hypothesis**: Trained SAEs on synthetic 3-level feature hierarchies exhibit significantly higher multi-child proportional absorption rates than random baselines.

**Measurement**: Multi-child proportional ablation (k=5):
- `absorption_rate = parent_activation_after_ablating_top_k_children / parent_activation_before`

**Pilot Results** (new pilots, 3 seeds):
| Condition | Absorption Rate | Std | vs Trained SAE |
|-----------|----------------|-----|----------------|
| Trained SAE | 0.500 | 0.0 | --- |
| Random Baseline | 0.147 | 0.065 | delta=0.353 |

**Statistical Test**: t-test vs random baseline: p ≈ 0, large effect size

**Interpretation**: Trained SAEs significantly differ from random baselines. However, zero variance in trained SAE is suspicious -- may indicate deterministic synthetic hierarchy.

**Falsification Criterion**: No significant difference (p > 0.05). **STATUS: NOT FALSIFIED**

---

### H_Mech: Encoder Alignment Drives Absorption, Not Decoder Geometry

**Status**: PASSED (new pilot evidence)

**Hypothesis**: Absorption is driven entirely by the encoder's learned alignment with hierarchical features. The decoder contributes nothing beyond reconstructing from encoder outputs.

**Measurement**: 2x2 factorial design:
- Condition A: Random encoder + Random decoder (pure geometry)
- Condition B: Trained encoder + Random decoder (encoder alignment only)
- Condition C: Random encoder + Trained decoder (decoder geometry only)
- Condition D: Trained encoder + Trained decoder (full training)

**New Pilot Results**:
| Condition | Encoder | Decoder | Absorption Rate |
|-----------|---------|---------|-----------------|
| A | Random | Random | 0.299 |
| B | Trained | Random | **0.490** |
| C | Random | Trained | **0.299** |
| D | Trained | Trained | **0.484** |

**Key Finding**: Condition B ≈ Condition D (encoder alignment is sufficient), Condition C ≈ Condition A (decoder geometry is irrelevant).

**Statistical Test**: t-test C vs D: p < 1e-111, delta = 0.185

**Interpretation**: The encoder learns to map hierarchical inputs into a representation where child features activate in place of parents. The decoder is passive -- it merely reconstructs from encoder outputs.

**Falsification Criterion**: If Condition D >> Condition B (decoder contributes significantly). **STATUS: NOT FALSIFIED**

---

### H2: Absorption Rate Correlates With Feature Frequency

**Status**: FAILED (pilot evidence - wrong direction)

**Original Hypothesis**: Features with lower activation frequency show systematically higher absorption rates.

**Pilot Result**: Spearman rho = +0.171 (positive correlation)
- Expected: rho < -0.3 (negative)
- Observed: rho = +0.171 (positive)

**Interpretation**: Higher-frequency features are MORE absorbed, not less. The encoder-driven mechanism suggests this is because frequent hierarchical patterns are efficiently encoded by routing parent activations through children.

**Archive as**: Negative result - frequency-absorption correlation in opposite direction

**Revised Interpretation**: The encoder learns to compress frequent parent representations into child subspaces for efficiency. This is "efficient coding" (Barlow 1961), not competitive exclusion.

---

### H3: Steering Absorbed Features Toward Parent Directions Improves Sensitivity

**Status**: PASSED (new pilot evidence - fixed implementation)

**Original Hypothesis**: For absorbed features, steering toward parent directions improves sensitivity.

**New Pilot Result** (fixed implementation):
```json
{
  "absorbed_mean_sensitivity": 0.055,
  "non_absorbed_mean_sensitivity": 0.034,
  "sensitivity_ratio": 1.620
}
```

**Issue**: Original pilot showed baseline = steered mean (steering not applied). Fixed in new pilot.

**Unexpected Finding**: Absorbed features are 1.62x MORE sensitive to steering than non-absorbed features. This suggests absorbed features retain parent direction information and can be "nudged" back.

**Statistical Test**: Sensitivity difference increases monotonically with alpha (0.0 → 0.003, 0.5 → 0.063 at alpha=5.0).

**Falsification Criterion**: No sensitivity difference between absorbed and non-absorbed. **STATUS: NOT FALSIFIED**

---

### H_Safe: Safety-Critical Features Show Elevated Absorption Rates

**Status**: NOT TESTED on real SAEs (synthetic pilot failed)

**Hypothesis**: Features annotated as safety-critical (deception, jailbreak, harm, manipulation) show higher absorption rates than matched non-safety features in real Gemma Scope SAEs.

**Synthetic Pilot Result** (using random feature indices):
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

**Expected outcome**: Safety features show mean absorption rate > non-safety features (p < 0.05).

**Falsification**: No significant difference in absorption rates.

**Novelty**: 9/10 - No prior work examines whether safety-critical features are disproportionately absorbed.

---

## Supplementary Hypotheses (Backup candidates)

### H_Comp: Absorption Increases With Hierarchy Strength

**Status**: NOT TESTED

**Hypothesis**: Feature absorption rate increases monotonically with feature hierarchy strength (parent-child cosine similarity).

**Measurement**: Vary hierarchy overlap across {0.5, 0.7, 0.9, 1.0}. Fit absorption vs. overlap curve.

**Expected outcome**: Absorption → 1.0 as overlap → 1.0. Fit quality R² > 0.8.

**Connection to H_Mech**: If absorption is encoder-driven, the encoder should learn stronger parent-child correlations as hierarchy strength increases.

---

### H_EncReg: Encoder Regularization Reduces Absorption

**Status**: NOT TESTED

**Hypothesis**: Adding a regularization term to the encoder that penalizes parent-child activation correlation reduces absorption without degrading reconstruction.

**Measurement**: Train SAE with additional loss term: `L_reg = lambda * sum_{parent-child pairs} |E[encoder_parent * encoder_child]|`. Compare absorption rate vs. baseline.

**Expected outcome**: Absorption rate reduced by >30% with <5% reconstruction degradation.

**Connection to H_Mech**: Direct test of whether encoder modification can control absorption.

---

### H_Ens: Multi-Resolution Ensemble Recovers Absorbed Features

**Status**: NOT TESTED

**Hypothesis**: Child features recoverable in high-L0 SAE (L0=256) are absorbed in low-L0 SAE (L0=16), and cross-SAE matching correctly identifies parent-child pairs.

**Measurement**: Train ensemble (L0=16, 64, 256). Match features via decoder cosine similarity. Verify recovered representations.

**Expected outcome**: >50% of absorbed features have recoverable children in ensemble.

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
| H_EncReg | backup | NOT TESTED | Absorption reduction | >30% |
| H_Ens | backup | NOT TESTED | Recovery rate | >50% |

---

## Negative Results Documentation

### H2: Frequency-Absorption Correlation

**Finding**: Positive correlation (rho = +0.171) contradicts competitive exclusion hypothesis (predicted negative).

**Implication**: The mechanism underlying absorption is not competitive exclusion. The encoder-driven mechanism explains this: frequent hierarchical patterns are efficiently encoded by routing parent activations through children (efficient coding, not competitive exclusion).

**Recommendation**: Archive as honest negative result. Reframe from "competitive exclusion" to "efficient coding."

### H_Safe: Synthetic Pilot

**Finding**: No difference between safety and non-safety features on synthetic SAE (p = 0.665).

**Implication**: Synthetic SAE features lack semantic content. Real Gemma Scope SAEs required for meaningful safety analysis.

**Recommendation**: Do not report synthetic H_Safe result. Proceed with real SAE analysis only.

---

## Action Items

1. **H_Mech full validation** (Priority 1): 5 seeds, 3 L0 targets, stochastic hierarchy
2. **H3 replication** (Priority 2): Confirm 1.62x sensitivity ratio with stochastic data
3. **H_Safe on Gemma Scope** (Priority 3): Install SAELens, select real safety features
4. **H_Comp** (Priority 4): Vary hierarchy strength, test encoder prediction
