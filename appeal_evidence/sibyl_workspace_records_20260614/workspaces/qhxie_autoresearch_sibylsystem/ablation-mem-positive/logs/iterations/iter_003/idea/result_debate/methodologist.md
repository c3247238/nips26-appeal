# Methodologist Report: CV-Based Actionability Decomposition

## Executive Summary

This iteration's experiments test whether coefficient of variation (CV) predicts steering effectiveness for absorbed SAE features. While pilot results showed a promising 2x difference (High-CV: 0.153 vs Low-CV: 0.075), full experiment results reveal **inconsistent replication** and **methodological concerns** that warrant scrutiny before accepting the primary claim.

**Overall Assessment**: The evidence for CV as a predictor of steering heterogeneity is WEAK to MODERATE. Several flagship findings fail to replicate or have concerning effect sizes.

---

## 1. Baseline Fairness Audit

### 1.1 Comparison Groups

| Experiment | Control Group | Treatment Group | Balance Issue |
|------------|---------------|-----------------|---------------|
| Steering by CV (Pilot) | Low-CV absorbed (n=30) | High-CV absorbed (n=30) | Balanced |
| Steering by CV (Full) | Low-CV absorbed (n=30) | High-CV absorbed (n=30) | Balanced |
| Decoder Orthogonality | Low orthogonality (n=30) | High orthogonality (n=30) | Balanced |
| Non-absorbed Baseline | Non-absorbed (n=30) | Absorbed high-CV (n=30) | Balanced |

**Issue**: The non-absorbed baseline uses different feature selection criteria than absorbed groups. Non-absorbed features are selected by absorption_score < 0.5, while absorbed groups use CV threshold (CV > 1.0 or <= 1.0). This creates potential confounding - the baseline may include features with different underlying properties unrelated to absorption status.

### 1.2 Hyperparameter Asymmetry

- **Pilot**: 100 samples, seed 42, +5 strength
- **Full**: 100 samples, seed 42, +5 strength (consistent)

No hyperparameter asymmetry detected. The protocol is consistent across pilot and full experiments.

---

## 2. Metric-Claim Alignment

### 2.1 Primary Claim: "CV predicts steering heterogeneity"

| Metric | Claimed | Actual | Alignment Issue |
|--------|---------|--------|-----------------|
| Steering effect (logit change) | High-CV > Low-CV | See replication failure below | CLAIM OVERSTATED |

**Critical Finding**: Pilot results do NOT replicate in full experiment.

| Group | Pilot Mean | Full Mean | Direction Reversed? |
|-------|------------|-----------|---------------------|
| High-CV absorbed | 0.153 | 0.0975 | YES - 36% drop |
| Low-CV absorbed | 0.075 | 0.1266 | YES - 69% increase |
| High/Low ratio | 2.04x | 0.77x | REVERSED |

The pilot showed High-CV features 2x more steerable. The full experiment shows High-CV features *less* steerable than Low-CV (ratio 0.77x). This is a **severe replication failure**.

### 2.2 Secondary Claims

| Claim | Metric | Status |
|-------|--------|--------|
| H4: CV difference exists (absorbed vs non-absorbed) | CV ratio 733x | SUPPORTED (but reversed direction - absorbed have HIGHER CV) |
| H6: Decoder orthogonality predicts steering | Correlation r > 0.3 | NOT_SUPPORTED (r = -0.136) |
| Activation patching validates absorption | Recovery > 10% | SUPPORTED (67.3% mean) |

---

## 3. Validity Threats Checklist

### 3.1 Data Leakage
- [ ] **NOT LIKELY**: Feature selection uses CV computed on training data but steering tested on held-out prompts. No clear leakage path.

### 3.2 Contamination
- [ ] **MINOR CONCERN**: SAE features were trained on Pile data; steering prompts ("The movie was very", "The food was extremely") may appear in training data. However, this affects all features equally and is standard practice.

### 3.3 Selection Bias
- [x] **MAJOR CONCERN**: Feature selection (top 30 by CV) was done on the same dataset used for evaluation. The CV threshold of 1.0 was chosen post-hoc based on pilot results.
- [x] **CONCERN**: The full experiment's Low-CV group showed *higher* steering effect (0.1266) than High-CV (0.0975), directly contradicting pilot and contradicting the hypothesis that CV predicts actionability.

### 3.4 Overfitting to Evaluation
- [ ] **MINOR CONCERN**: 3 prompts for steering measurement. The semantic appropriateness of tokens for steering targets was not independently validated.

---

## 4. Ablation Gap Analysis

### 4.1 Completed Ablations

| Ablation | What was tested | Finding |
|----------|-----------------|---------|
| H1: CV threshold (1.0) | High-CV vs Low-CV steering | **FAILED TO REPLICATE** |
| H6: Decoder orthogonality | Orthogonal vs non-orthogonal decoder weights | NOT_SUPPORTED (r=-0.136) |
| Non-absorbed baseline | Absorbed vs non-absorbed steering | Absorbed slightly lower than non-absorbed (0.0975 vs 0.102) |

### 4.2 Missing Ablations

1. **Fano factor control**: The proposal mentions controlling for decoder magnitude using Fano factor (CV²/mean). This control was not reported in results.

2. **Cross-architecture validation (Gemma-2)**: Listed in task plan but no results found. This is a critical missing ablation for the claim that "CV generalizes across architectures."

3. **Multiple steering strengths**: Only +5 was used in non-absorbed baseline; full_steering_cv claims +3, +5, +7 but results file not found.

---

## 5. Reproducibility Score: 3/5

| Criterion | Status |
|-----------|--------|
| Random seeds fixed | YES (seed=42) |
| Hyperparameters specified | YES (samples=100, strength=5) |
| Code available | Not verified (no repo link) |
| Hardware documented | YES (GPU in output) |
| Independent reproduction likely | UNCERTAIN - pilot-to-full replication failed |

**Concern**: The discrepancy between pilot and full results is severe enough that an independent reproduction might yield different conclusions.

---

## 6. Top-3 Recommendations

### Recommendation 1: Resolve Pilot-Full Replication Failure (CRITICAL)
**Issue**: High-CV steering effect dropped from 0.153 (pilot) to 0.0975 (full); Low-CV increased from 0.075 to 0.1266. The direction reversed.

**Required actions**:
- Compute steering effects using the SAME feature selection criteria in pilot vs full
- Check whether different features were selected (potential selection drift)
- Report confidence intervals for both pilot and full results
- If replication fails consistently, the primary claim (CV predicts steering) must be downgraded

**This is the highest-priority issue** because it directly threatens the main finding.

### Recommendation 2: Complete Cross-Architecture Validation
**Issue**: Gemma-2-2B validation is listed in task plan as key evidence for generalization claim, but no results appear in exp/results/.

**Required actions**:
- Run full_cross_architecture experiment or explicitly report it as pending/omitted
- If omitted, remove generalization claim from paper abstract and title

### Recommendation 3: Add Fano Factor Control or Remove Claim
**Issue**: The proposal states "Control for decoder magnitude using Fano factor (CV²/mean)" but no Fano-controlled results were reported.

**Required actions**:
- Either compute Fano factor for all features and include as covariate, OR
- Explicitly acknowledge this control was not performed and qualify the CV-steering correlation claim

---

## 7. Specific Methodological Issues

### 7.1 chi_ratio Threshold Violation
The proposal acknowledges chi_ratio=1.88 is below the "sharp transition" threshold of 3.0. Yet the abstract and title imply a clear threshold effect. **This is a framing inconsistency** - either the threshold is meaningful (requires chi_ratio > 3.0) or it is not a requirement (in which case, why emphasize "quasi-critical"?).

### 7.2 H3 Falsified but Still Listed
H3 (layer depth as temperature) was falsified at λ=0.001 - all layers saturated at absorption_rate=1.0. The proposal acknowledges this but still lists it as needing "retesting at λ_c=5e-5." This is appropriate, but the paper should explicitly report the falsification rather than burying it in revision text.

### 7.3 Effect Size Magnitudes
The absolute steering effects are small (0.05-0.15 logit change). Even if the High/Low CV difference were reliable, the practical significance for interpretability applications is unclear. A logit change of 0.05 on a base logits of ~2-4 is ~1-2% probability shift - marginal for steering interventions.

### 7.4 Non-Absorbed Baseline Interpretation
The non-absorbed baseline (0.102) and absorbed high-CV (0.0975) are essentially equivalent (difference 0.004). This suggests absorbed features, even high-CV ones, do not have substantially different steering potential than non-absorbed features. This is a negative result for the "robust absorbed" narrative.

---

## 8. Summary Assessment

| Finding | Evidence Strength | Concern Level |
|---------|-------------------|---------------|
| Activation patching validates absorption | MODERATE (67.3% recovery) | LOW |
| H4: Absorbed have higher CV than non-absorbed | STRONG (733x ratio) | LOW |
| H6: Orthogonality predicts steering | NOT_SUPPORTED | MEDIUM (expected relationship absent) |
| CV threshold (1.0) predicts steering | WEAK (replication failed) | HIGH |
| Phase transition with critical λ | MODERATE (chi_ratio=1.88) | MEDIUM (below threshold) |
| Finite-size scaling (nu=3) | MODERATE (R²=0.951) | LOW |

**Overall Credibility**: The most impactful claim (CV predicts steering heterogeneity) has a **replication failure** that has not been resolved. Additional evidence (orthogonality, cross-architecture) is missing or fails to support the narrative.

**Suggested Framing**: The paper should present CV-based decomposition as an exploratory finding requiring validation, not as a confirmed predictor. The actionability paradox refinement should be framed cautiously given the weak CV-steering evidence.

---

## 9. Flags for Debate

1. **PILOT-FULL REPLICATION FAILURE**: The direction of High-CV vs Low-CV steering effect reversed between pilot and full experiments. This is a serious methodological concern.

2. **MISSING ABLATION**: Cross-architecture validation (Gemma-2) was not completed despite being a key piece of evidence for generalization.

3. **chi_ratio THRESHOLD**: chi_ratio=1.88 is below the self-imposed "sharp transition" threshold of 3.0. The paper should not claim a "sharp" or "quasi-critical" transition with this value.

4. **EFFECT SIZE MAGNITUDE**: Steering effects of 0.05-0.15 logit units are marginal for practical interpretability applications.

5. **NON-ABSORBED COMPARISON**: Absorbed high-CV (0.0975) is nearly identical to non-absorbed (0.102), suggesting absorption status does not substantially alter steering potential.