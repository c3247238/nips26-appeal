# Methodologist Report: Encoder-Driven Feature Absorption in SAEs

## Executive Summary

This report audits the methodology of experiments testing encoder-driven absorption in SAEs. **Three of four main hypotheses have failed or shown anomalous results**. The experimental design has significant validity threats, and several metrics appear broken.

**Overall Methodology Score: 2/5 (Poor)**

---

## 1. Baseline Fairness Audit

### H_Mech (Encoder-Driven Mechanism)

| Issue | Severity | Details |
|-------|----------|---------|
| Condition D is invariant across seeds | CRITICAL | All 5 seeds report identical absorption for Condition D: 0.01746... This suggests either (a) a shared underlying value not actually measured per seed, or (b) a bug in result aggregation |
| Condition C variance is extreme | CRITICAL | Mean = 12.28, std = 17.13, range [0, 43.84]. This is 2-3 orders of magnitude larger than Conditions A/B/D |
| Stochastic noise may have introduced floor effects | HIGH | Epsilon = N(0, 0.05) on deterministic hierarchies. The noise magnitude should be validated against actual feature scales |

**Verdict**: Baseline is NOT fair. The decoder-irrelevant conclusion (C ≈ A) from iter_001 does NOT replicate under stochastic conditions. The stochastic design revealed decoder contribution is non-zero under some conditions.

---

### H_Comp (Hierarchy Strength Dependence)

| Issue | Severity | Details |
|-------|----------|---------|
| Unequal variance across cosine levels | HIGH | cos_0.5 std=0.131 vs cos_0.9 std=0.397. Heteroscedasticity violates regression assumptions |
| Non-monotonic relationship directly contradicts theory | CRITICAL | cos_0.5: 0.814, cos_0.6: 0.989, cos_0.7: 0.972, cos_0.8: 0.607, cos_0.9: 1.201, cos_0.95: 0.512. The R² = 0.04 failure is not noise — the direction reverses multiple times |

**Verdict**: The experiment design correctly identified that the monotonic assumption is wrong. This is a legitimate negative result, not a design flaw.

---

### H_Pareto (Sensitivity-Absorption Frontier)

| Issue | Severity | Details |
|-------|----------|---------|
| **Sensitivity metric is BROKEN** | CRITICAL | ALL sensitivity values are identical: 3.0188171245077235 (std=0.0 for all L0 levels). This indicates either (a) bug in measurement, (b) wrong formula, or (c) sensitivity metric is not sensitive to L0 changes |
| Frontier fit R² = 0 | CRITICAL | With constant sensitivity, frontier fitting is mathematically meaningless |
| "full_pass": true is incorrect | HIGH | The pass criteria requires "detectable frontier shape" but with constant sensitivity no shape can be detected |

**Verdict**: The sensitivity measurement must be debugged before H_Pareto conclusions are valid.

---

### H_Safe (Safety-Critical Features)

| Issue | Severity | Details |
|-------|----------|---------|
| All absorption rates = 0 in pilot | CRITICAL | Gemma Scope pilot: all 10 features (5 safety + 5 non-safety) show exactly 0.0 absorption. This suggests either (a) measurement issue, (b) feature selection problem, or (c) SAE does not exhibit absorption for these features |
| Feature selection is unvalidated | HIGH | Safety features [1024, 2048, 3072, 4096, 5120] and non-safety [100, 200, 300, 400, 500] appear arbitrary/sequential, not based on actual Neuronpedia annotation matching |
| Held-out validation also failed | HIGH | GPT-2 Small: p = 0.345, safety_mean=233.13 vs non_safety_mean=221.70 (small actual difference, not significant) |

**Verdict**: The safety hypothesis is not supported, but the measurement has likely failed for Gemma Scope.

---

## 2. Metric-Claim Alignment

| Claim | Metric | Alignment | Gap |
|-------|--------|-----------|-----|
| "Encoder alignment drives absorption" | Multi-child proportional absorption | MISALIGNED | The metric measures post-ablation parent activation, but encoder alignment is a training property. The causal mechanism is not directly measured |
| "Absorption increases with hierarchy strength" | Spearman correlation | MISALIGNED | The metric measures correlation, not causation. A third factor could drive both |
| "Safety features disproportionately absorbed" | Mann-Whitney U | ALIGNED | But measurement yielded all zeros |
| "Sensitivity-absorption trade-off exists" | Steering coefficient variance | BROKEN | Metric produces constant values |

**Critical Gap**: No experiment directly measures the **encoder's learned alignment** as a variable. The 2x2 factorial manipulates encoder training status (trained vs random) but does not measure alignment quality itself.

---

## 3. Validity Threats Checklist

- [ ] **Data leakage**: NOT LIKELY. Synthetic hierarchies are generated fresh per experiment.
- [ ] **Contamination**: NOT LIKELY for synthetic experiments. Gemma Scope features could theoretically be in training data, but this is standard for published SAEs.
- [x] **Selection bias**: YES. Feature indices for H_Safe ([1024, 2048, ...] vs [100, 200, ...]) are clearly not matched by activation frequency. The "matching" described in methodology was not executed.
- [x] **Overfitting to evaluation**: YES. H_Pareto reports "full_pass": true despite R²=0. The pass criteria is not being applied correctly.
- [x] **Floor effect**: YES. H_Safe Gemma pilot shows 0.0 absorption for ALL features, suggesting measurement is below detectable range.
- [ ] **Citation accuracy**: NOT AUDITED. References to Hu et al. 2025, Tang et al. 2512.05534 need verification.

---

## 4. Ablation Gap Analysis

| Component | Ablation Present? | Quality |
|-----------|-------------------|---------|
| Encoder alignment | Yes (Conditions A vs B) | Valid |
| Decoder geometry | Yes (Conditions A vs C) | INVALID due to anomalous C values |
| Hierarchy noise | No | Missing: should have A' (random hierarchy, same encoder/decoder) |
| L0 target | Yes (H_Pareto) | INVALID due to broken sensitivity metric |
| Safety-critical status | Yes (H_Safe) | INVALID due to zero absorption readings |

**Missing ablations**:
1. Encoder alignment QUALITY: Current design only varies "trained vs random", not alignment strength
2. Hierarchy depth: All experiments use 3-level hierarchies; effect at 2 or 4 levels is unknown
3. Feature frequency interaction: H2 was tested and failed, but was not properly integrated into H_Safe design

---

## 5. Reproducibility Score: 2/5

| Factor | Present | Notes |
|--------|---------|-------|
| Random seeds | YES | Seeds specified (42, 123, 456, 789, 1024) |
| Hyperparameters | PARTIAL | n_samples specified, but hierarchy generation params not fully specified |
| Code availability | NO | `exp/results/code/` exists but contents unknown |
| Hardware requirements | NO | GPU model, CUDA version not documented |
| Data splits | N/A | Synthetic data generated fresh |

**Critical reproducibility barrier**: The sensitivity metric produces constant values across all conditions, suggesting it is either bugged or implementation differs from the cited Hu et al. 2025.

---

## 6. Top-3 Recommendations

### Priority 1: Debug Sensitivity Metric (H_Pareto)

**Problem**: Sensitivity returns identical values (3.018...) across all L0 levels, making Pareto frontier analysis impossible.

**Action**: Before any paper claims about sensitivity-absorption trade-offs, the sensitivity measurement code must be:
1. Tested on a synthetic case where sensitivity SHOULD vary
2. Compared against the Hu et al. 2025 implementation
3. Verified that the formula in the code matches the cited paper

**What changes if fixed**: Validates or invalidates H_Pareto.

---

### Priority 2: Fix H_Safe Feature Selection or Abandon the Hypothesis

**Problem**: Safety features [1024, 2048, 3072, 4096, 5120] are clearly arbitrary indices, not Neuronpedia-matched features. All absorption values are 0.0, suggesting measurement failure.

**Action**:
1. Actually retrieve Neuronpedia annotations for Gemma Scope layer 12
2. Match safety vs non-safety by TRUE activation frequency (not index proximity)
3. Verify absorption measurement works on non-zero features first (e.g., high-frequency features)

**What changes if fixed**: Either validates safety concern (if p < 0.05) or properly establishes it as a negative result.

---

### Priority 3: Re-examine H_Mech Conclusion

**Problem**: The stochastic hierarchy reveals C ≈ A does NOT hold (C mean = 12.28 vs A mean = 0.18). The original "encoder-driven only" conclusion from iter_001 (deterministic hierarchy) does not replicate.

**Action**:
1. Investigate why Condition C has extreme values (17.3, 43.8 in seeds 789, 1024)
2. Determine if this is a bug or a real phenomenon (decoder over-compensation)
3. Re-run iter_001's deterministic case to confirm the original finding
4. Update the conclusion to: "Under stochastic hierarchies, decoder geometry contributes non-zero absorption in some conditions"

**What changes if fixed**: The paper's main contribution (encoder-driven mechanism) requires either (a) qualification to "under deterministic hierarchies" or (b) a deeper investigation of when decoder contributes.

---

## 7. Additional Concerns

### Anomalous Condition D Invariance

All seeds report EXACTLY 0.01746168273373232 for Condition D. This value appears 5 times with zero variance. This suggests either:
- A shared constant being reported instead of measured values
- A bug in result aggregation
- The same hierarchy sample being used across seeds

This must be explained before Condition D results can be trusted.

### H_Comp Negative Result is Valid but Undermined

The H_Comp experiment correctly identified a non-monotonic relationship (R² = 0.04). However, the variance structure suggests real effects exist but are obscured. The paper should either:
- Report this as a genuine finding: "absorption does not monotonic increase with hierarchy strength"
- Investigate why cos_0.8 shows anomalously low absorption (0.607 vs surrounding 0.97-1.2)

### Falsification Criteria Are Not Being Applied

The experiment state shows:
- H_Mech: "factorial_checks.pass": false but experiment is marked "DONE"
- H_Comp: "full_pass": false but experiment is marked "DONE"  
- H_Pareto: "full_pass": true despite R² = 0 and constant sensitivity
- H_Safe: Pilot "NO_GO" but no subsequent full experiment appears to be planned

The go/no-go decision pipeline is not being followed.

---

## 8. Summary Table

| Hypothesis | Expected | Observed | Verdict | Validity Threat |
|------------|----------|----------|---------|-----------------|
| H_Mech (encoder-driven) | B≈D, C≈A | B≈D HOLD, C≠A FAIL | PARTIAL | Decoder contribution revealed under stochastic setting |
| H_Comp (monotonic) | R² > 0.8 | R² = 0.04 | FAILED | None — legitimate negative result |
| H_Pareto (frontier exists) | Shape detected | Constant sensitivity | UNINTERPRETABLE | Metric broken |
| H_Safe (safety absorption) | p < 0.05 | p = 1.0, p = 0.345 | FAILED | Measurement failure (zeros); feature selection flawed |

---

## 9. Conclusion

The methodology has **serious issues** that prevent strong conclusions:

1. **The main contribution (encoder-driven mechanism) is weakened** by stochastic hierarchy results showing decoder contribution
2. **The Pareto frontier claim is uninterpretable** due to broken sensitivity metric
3. **The safety analysis is likely invalid** due to arbitrary feature selection and zero readings
4. **Pass/fail criteria are not being applied** to experiment outcomes

**Recommendation**: Do not proceed to paper writing until:
- Sensitivity metric is debugged and validated
- H_Safe feature selection is redone with proper matching
- H_Mech anomalous values (Condition C, invariant Condition D) are explained
- Falsification criteria are consistently applied

The core phenomenon (absorption exists) is real and well-documented. But the mechanistic interpretations and safety implications require additional validation before publication.
