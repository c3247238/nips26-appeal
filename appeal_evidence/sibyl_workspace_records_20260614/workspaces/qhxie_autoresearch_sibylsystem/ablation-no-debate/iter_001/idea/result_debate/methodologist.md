# Methodologist Report: Feature Absorption in SAEs

**Date**: 2026-04-29
**Workspace**: ablation-no-debate/current
**Analyst Role**: Methodologist

---

## Executive Summary

The experiments demonstrate strong methodology with well-designed baselines and clear pass criteria. However, **critical issues** exist in the H_Safe feature selection (arbitrary index-based selection without validation) and the deterministic absorption finding (zero variance) warrants additional investigation. The H_Mech factorial is the most methodologically rigorous contribution.

---

## 1. Baseline Fairness Audit

### Overall Assessment: GOOD

| Condition | Hyperparameter Budget | Data Splits | Notes |
|-----------|----------------------|-------------|-------|
| Trained SAE | Full training (20K steps) | Same | Fair |
| Random Decoder | No training | Same | Fair baseline |
| Shuffled Features | Same activations | Same | Preserves decoder geometry |
| Permuted Encoder | Same trained SAE | Same | Tests encoder weight structure |

**Key Finding**: The Korznikov-style baselines are well-matched. Shuffled and permuted baselines preserve different aspects of trained SAE structure, enabling meaningful decomposition.

**Minor Asymmetry**: The shuffled/permuted baselines are derived from the trained SAE, meaning they inherit its training hyperparameters. However, this is intentional to isolate the contribution of specific components.

**Pass/Fail**: PASS - Baselines are fair and well-motivated.

---

## 2. Metric-Claim Alignment

### Primary Metric: Multi-Child Proportional Absorption Rate

```
absorption_k5 = parent_activation_after_ablating_top_k_children / parent_activation_before
```

**Does this metric capture absorption?** Yes. This measures how much the parent feature's activation "leaks" to child features - exactly the absorption phenomenon described in Chanin et al. (2024).

**Limitation**: This is a forward-pass metric. It measures activation-level absorption but does not directly measure downstream behavioral effects.

### Secondary Metrics and Claims

| Claim | Metric | Alignment | Gap |
|-------|--------|-----------|-----|
| H1: Trained > Random | Cohen's d = 8.94 | Strong | None |
| H3: Steering sensitivity | Sensitivity ratio 1.62x | Moderate | Directional, not causal |
| H_Safe: Safety > Non-safety | Mann-Whitney p = 0.665 | Failed | Feature selection invalid |
| H_Mech: Encoder vs Decoder | t-statistic = 48.5 | Strong | None |

**Critical Gap in H_Safe**: The claim tests whether **safety-critical** features are disproportionately absorbed. However, features were selected by **arbitrary index ranges** (500-519 for "safety", 100-119 for "non-safety") without validation that these indices correspond to actual safety-relevant neurons. This invalidates the H_Safe conclusion entirely.

**Pass/Fail**: FAIL on H_Safe feature selection, PASS on all other metrics.

---

## 3. Validity Threats Checklist

### Data Leakage
- [x] **NOT PRESENT**: Test data is synthetic hierarchy, not from real model training
- [x] Training uses separate synthetic hierarchies from evaluation

### Contamination
- [ ] **UNCERTAIN**: Gemma Scope SAEs were trained on The Stack/Pile
- [ ] If benchmark text is in pretraining data, absorption measurements on real features could be contaminated
- [ ] The H_Safe analysis on Gemma Scope features could be affected

### Selection Bias
- [x] **NOT PRESENT**: Hyperparameters tuned on synthetic data, not test set
- [x] All conditions use identical synthetic hierarchy generation

### Overfitting to Evaluation
- [ ] **PARTIAL RISK**: Results specific to single synthetic hierarchy configuration
- [ ] Hierarchy parameters (parent-children cosine=0.67, grandchild orthogonality~0) are fixed
- [ ] Generalization to different hierarchy structures not tested

### Statistical Power
- [ ] **CONCERN**: Pilot experiments use n=100 samples
- [ ] With std=0.0 in trained SAE condition, any sample size is sufficient
- [ ] Random baseline variance is high (std=0.065-0.069), suggesting need for more samples

**Summary**:
| Threat | Risk Level | Mitigation |
|--------|------------|------------|
| Data Leakage | LOW | Synthetic evaluation data |
| Contamination | MEDIUM | Gemma Scope not validated for H_Safe |
| Selection Bias | LOW | Synthetic hierarchies |
| Overfitting to Evaluation | MEDIUM | Single hierarchy config |
| Statistical Power | LOW | Effect sizes are large |

---

## 4. Ablation Gap Analysis

### What Was Ablated

| Component | Ablation Method | Adequate? |
|-----------|-----------------|-----------|
| H1: Trained vs Random | Full replacement of trained weights | YES |
| H2: Frequency correlation | N/A (observational) | YES |
| H3: Steering sensitivity | Alpha scaling (0.5-5.0) | PARTIAL - Logit-level not tested |
| H_Safe: Feature type | Index-based feature selection | **NO - Invalid selection** |
| H_Mech: Encoder vs Decoder | 2x2 factorial swapping | YES - Best design |

### Missing Ablations

1. **H3: Logit-level steering validation** - Basu et al. (2026) methodology not implemented
2. **H_Safe: Actual safety feature identification** - No Neuronpedia validation
3. **Hierarchy structure generalization** - Results only from one hierarchy configuration (cosine=0.67)
4. **SAE architecture ablation** - Only Config A tested in factorial; Configs B/C from proposal not compared

**Pass/Fail**: PARTIAL - Strong ablation for H1 and H_Mech; critical gaps for H3 and H_Safe.

---

## 5. Reproducibility Score: 3.5/5

### What Is Documented

| Element | Status | Notes |
|---------|--------|-------|
| Random seeds | YES | 42, 43, 44 documented |
| Hyperparameters | PARTIAL | Training steps specified; lr/batch partially documented |
| Hardware | NO | GPU requirements not stated |
| Code availability | NO | No repository link provided |
| Synthetic data generation | YES | d_model=512, d_sae=4096, L0=32 documented |
| Statistical tests | YES | t-test, Mann-Whitney, ANOVA documented |

### Missing for Full Reproduction

1. **Exact training hyperparameters**: Learning rate, optimizer, scheduler not in results
2. **Hardware requirements**: GPU memory, runtime not reported
3. **SAE implementation**: Framework (SAELens), version, configuration details missing
4. **Code repository**: No GitHub/link provided

**Credibility adjustment**: Given that results are derived from a running research system with documented code paths, reproducibility is moderate. The main gaps are documentation, not experimental design.

---

## 6. Top-3 Recommendations

### Priority 1: Fix H_Safe Feature Selection (Critical)

**Problem**: Features selected by arbitrary index ranges (500-519 vs 100-119) without validation.

**Required Fix**:
```python
# 1. Query Neuronpedia API for actual safety-relevant feature indices
# 2. Use Gemma Scope activation-based classification (if available)
# 3. Validate selected features show differential response to safety-related prompts
```

**Effort**: Medium (requires Neuronpedia API access or manual curation)
**Impact**: Without this, H_Safe is not a valid experiment and should be removed from paper.

---

### Priority 2: Investigate Deterministic Absorption (Medium)

**Problem**: Trained SAE absorption is exactly 0.500 with std=0.0 across ALL seeds (42, 43, 44). This is suspicious - suggests a deterministic mathematical relationship, not a learned behavior.

**Required Investigation**:
1. Test whether 0.50 absorption is a function of hierarchy parameters (parent-children cosine=0.67)
2. Vary hierarchy cosine similarity and measure effect on absorption
3. Derive theoretical expected absorption given hierarchy geometry

**Effort**: Low (analyze existing data + small new experiment)
**Impact**: If absorption is purely deterministic, the paper's narrative ("SAEs learn to absorb") is misleading.

---

### Priority 3: Implement Logit-Level Steering Validation (Medium)

**Problem**: H3 steering validation is at activation level, not behavioral level. Basu et al. (2026) methodology not implemented.

**Required Fix**:
```python
# 1. Apply steering at residual stream level
# 2. Measure logit change for target concept (e.g., refusal vs compliance)
# 3. Compare absorbed vs non-absorbed feature steering effects on behavior
```

**Effort**: Medium (requires understanding of residual stream steering)
**Impact**: Without logit-level validation, H3 only shows sensitivity, not causal effect on model behavior.

---

## Summary Ratings

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Baseline Fairness | 5/5 | Excellent Korznikov-style baselines |
| Metric Validity | 3/5 | H_Safe feature selection invalid |
| Validity Threats | 3/5 | Medium contamination risk on Gemma Scope |
| Ablation Completeness | 3.5/5 | Missing logit-level H3, valid H_Safe |
| Reproducibility | 3.5/5 | Core elements documented, missing HW/code |
| **Overall** | **3.6/5** | Strong design with critical H_Safe flaw |

---

## Conclusion

The experimental methodology is **generally sound** with well-motivated baselines and rigorous statistical tests. The **2x2 factorial design** for H_Mech is the strongest contribution.

**Critical action required**: The H_Safe feature selection must be fixed or the hypothesis must be removed from the paper. The current arbitrary index-based selection invalidates any conclusions about safety-critical feature absorption.

**Secondary concern**: The deterministic absorption finding (std=0.0) warrants theoretical investigation to understand whether absorption is truly learned or a geometric inevitability.

---

*Methodologist analysis complete. Report saved to: idea/result_debate/methodologist.md*
