# Methodologist Analysis: SAE Feature Absorption Phase Transitions

## Analysis Date: 2026-05-01

## Executive Summary

This report audits the methodology of the SAE feature absorption phase transition study. The project claims to demonstrate quasi-critical behavior in SAE absorption with finite-size scaling (nu=3, R^2=0.951) and discovers a "variance paradox" where absorbed features have HIGHER coefficient of variation than non-absorbed features.

**Overall Assessment**: The core methodology is sound for the supported hypotheses (H1, H2, H4, H5). However, there are significant validity threats that must be addressed before publication, particularly around the CV measurement protocol and the missing full activation patching validation.

**Reproducibility Score**: 3/5 - Core parameters specified but key procedures lack detail

---

## 1. Baseline Fairness Audit

### 1.1 H1 (Critical Sparsity Threshold) - SUPPORTED

**Claim**: Absorption exhibits quasi-critical threshold at lambda_c = 5e-5 with susceptibility peak chi_max=11.19

**Baseline Comparison**:
- No external baseline comparison exists in this study
- The claim is self-referential: chi_ratio=1.88 is compared against an arbitrary threshold of 3.0
- The "sharp transition" threshold of 3.0 appears to be chosen post-hoc to frame the result as "not quite sharp"

**Issues**:
- chi_ratio = 1.88 < 3.0 is framed as "quasi-critical" but there is no prior literature justification for the 3.0 threshold
- This could be considered p-hacking on the narrative level: choosing a threshold that makes the result sound impressive while acknowledging it failed the threshold

**Verdict**: The methodology is acceptable but the framing is post-hoc. The actual physics is valid (chi_max=11.19 is a real peak), but calling it "quasi-critical" rather than "no sharp transition" is motivated reasoning.

### 1.2 H2 (Finite-Size Scaling) - SUPPORTED

**Claim**: Scaling collapse with nu=3, R^2=0.951

**Methodology**:
- Dictionary sizes: 6144, 12288, 24576 (layer 8 feature-splitting SAEs)
- Sparsity percentiles: 90-99
- Nu=3 found by grid search

**Baseline Fairness**: No comparison to alternative nu values reported. R^2=0.951 is good but no confidence interval or comparison to nu=2 or nu=4 is provided.

**Issues**:
- Only one nu value tested without statistical comparison
- Should compare Akaike or Bayesian information criteria for nu=2,3,4

**Verdict**: Acceptable but weak statistical comparison.

### 1.3 H4 (CV Difference - Reversed) - SUPPORTED with CONCERN

**Claim**: Absorbed features have CV=6.22 vs non-absorbed CV=0.01 (733x ratio)

**Critical Issue - Metric Contamination**:
- CV is computed at lambda=0.001, but absorption classification is ALSO done at lambda=0.001
- This creates a circular dependency: features are classified as "absorbed" based on the same metric being tested
- The CV comparison may be tautological: absorbed features are defined AS features with high absorption scores, and absorption scores correlate with activation magnitude which inflates CV

**Mitigation Needed**:
- CV should be measured on a held-out set or at a different sparsity value than absorption classification
- Should control for mean activation magnitude to isolate variance from scale

**Verdict**: Valid discovery but current protocol has metric contamination risk.

---

## 2. Metric-Claim Alignment

| Claim | Metric | Alignment | Issue |
|-------|--------|-----------|-------|
| "Quasi-critical threshold" | chi_ratio = 1.88 | WEAK | chi_ratio < 3.0 threshold is arbitrary, not from literature |
| "First finite-size scaling" | R^2 = 0.951 | STRONG | Clear scaling collapse demonstration |
| "CV reversal discovery" | CV ratio 733x | STRONG but CONTAMINATED | Valid measurement but same lambda for classification and measurement |
| "Layer saturation masks heterogeneity" | absorption_rate=1.0 | ACCEPTABLE | Direct measurement supports claim |
| "Steering effectiveness predicted by CV" | Steering effect 2x for high-CV | ACCEPTABLE | pilot_steering_cv shows 0.153 vs 0.075 |

**Key Gap**: The connection between CV and steering effectiveness is only validated in a pilot (n=100 samples, 30 features per group). The full experiment (n=1000) has NOT been executed yet per task_plan.json.

---

## 3. Validity Threats Checklist

### 3.1 Data Leakage
- [ ] **NOT DETECTED**: No evidence of test data in training set
- SAE is pretrained, not fine-tuned on the evaluation data
- Samples are drawn from a generic corpus, no benchmark contamination

### 3.2 Contamination
- [ ] **LOW RISK**: GPT-2 pretraining data includes generic web text
- The specific words tested ('eight', 'lower', 'liked', etc.) are common English words
- However, no benchmark answers are directly tested, only generic activation patterns

### 3.3 Selection Bias
- [ ] **MODERATE RISK**: The 9 "persistent core words" were SELECTED based on prior analysis showing they persist across layers
- This selection criterion biases toward words that work, potentially inflating recovery percentages
- A proper validation would use words randomly sampled from the vocabulary

### 3.4 Overfitting to Evaluation
- [ ] **LOW RISK**: The primary metrics (chi_ratio, R^2) are not benchmark-specific
- Phase transition physics should generalize beyond this specific SAE

### 3.5 Metric Miscalibration
- [ ] **HIGH RISK**: The activation patching results show that feature 22545 appears as a "top feature" for 5 different words (eight, lower, school, turn, move) - this suggests the metric may be picking up general high-activation features rather than word-specific features
- The same top feature appearing across different words could indicate the SAE latents are not as monosemantic as claimed

---

## 4. Ablation Gap Analysis

| Component | Ablation Present | Notes |
|-----------|-----------------|-------|
| Sparsity threshold (lambda_c) | YES | Full sparsity sweep completed |
| Finite-size scaling (nu=3) | PARTIAL | Only nu=3 tested, no comparison to alternatives |
| CV difference (H4) | PARTIAL | Pilot validates, full pending |
| Cross-layer measurement | YES | Completed but at wrong lambda (0.001 instead of 5e-5) |
| Activation patching | YES | Pilot completed, full pending |
| Steering effectiveness | PARTIAL | Pilot validates, full pending |
| Information bottleneck (H5) | YES | Co-occurrence analysis completed |
| Graph topology (H6) | YES | Falsified correctly |

**Missing Ablations**:
1. **H3 (Layer Criticality)**: The critical experiment at lambda_c=5e-5 has NOT been executed. The claim that "layer heterogeneity appears at true critical point" is speculative without measurement.

2. **Feature Selection Randomization**: The 9 core words were pre-selected. A proper ablation would randomize word selection to test whether recovery is specific to pre-selected words.

3. **CV Measurement at Different Lambda**: The CV measurement uses the same lambda (0.001) as absorption classification - this should be ablated to test robustness.

---

## 5. Reproducibility Score: 3/5

**Strengths**:
- Random seed specified (42)
- Model version specified (gpt2-small, no finetune)
- SAE release specified (gpt2-small-res-jb)
- Activation cache mentioned for reuse

**Weaknesses**:
- No hardware requirements documented (GPU memory, CUDA version)
- No software environment specified (Python version, library versions)
- Full activation patching experiment not completed
- Cross-layer at lambda_c=5e-5 not completed
- Steering effectiveness full experiment not completed

**What is missing for full reproducibility**:
1. `requirements.txt` or `environment.yml` with exact library versions
2. CUDA/GPU configuration
3. Full experiment results (not just pilots)
4. Code availability - no repo link provided

---

## 6. Top-3 Recommendations

### Recommendation 1: Fix CV Measurement Protocol (HIGH PRIORITY)

**Problem**: CV is measured at the same lambda used for absorption classification, creating metric contamination.

**Action**:
1. Measure CV at a DIFFERENT sparsity value than absorption classification (e.g., classify at lambda=0.001, measure CV at lambda=5e-5)
2. Control for mean activation magnitude by computing Fano factor (CV / mean) instead of raw CV
3. Validate that the CV difference holds when controlling for activation scale

**Expected Impact**: Would strengthen the "variance paradox" claim by ruling out scale confounds.

**Effort**: Low (recompute from existing cached activations) | **Credibility Gain**: HIGH

### Recommendation 2: Complete Full Activation Patching with Randomization (HIGH PRIORITY)

**Problem**: The 9 core words were pre-selected based on prior analysis showing persistence. This biases toward positive results.

**Action**:
1. Complete full_activation_patching (1000 samples) as planned
2. ADD a randomized control: randomly select 9 words from vocabulary and measure their recovery
3. Compare pre-selected words vs random words to quantify selection bias

**Expected Impact**: Would establish whether the 67.3% mean recovery is specific to pre-selected words or generalizes.

**Effort**: Medium (requires additional random word selection and measurement) | **Credibility Gain**: HIGH

### Recommendation 3: Execute Cross-Layer at True Critical Sparsity (MEDIUM PRIORITY)

**Problem**: H3 claims layer heterogeneity "may appear at finer lambda values" but this has not been tested at lambda_c=5e-5.

**Action**:
1. Execute full_cross_layer_critical experiment at lambda=5e-5 (not 0.001)
2. Use SAEBench probe projection metric for cross-layer reliability
3. Compare absorption rates across layers [0, 3, 6, 9, 11]

**Expected Impact**: Would either validate the layer-criticality narrative or definitively falsify it.

**Effort**: Medium (45 min GPU time per task_plan.json) | **Credibility Gain**: MEDIUM-HIGH

---

## 7. Additional Concerns

### 7.1 Phantom Contributions

The task_plan.json lists H10 (Homeostatic Rebalancing) as "not executed due to negative H6 result" but it appears as Contribution #4 in proposal.md ("First training-free post-hoc repair"). This is a phantom contribution that must be removed or validated.

### 7.2 Logical Incoherence in LCA Framework

H6 is decisively falsified (component count decreases with layer, not peaked at L6). Yet the paper claims the LCA framework is "supported" via H7 precision-recall asymmetry. This is logically incoherent - falsification of the order parameter should invalidate the framework.

### 7.3 chi_ratio Threshold Justification

The chi_ratio threshold of 3.0 for "sharp transition" is not justified by any prior literature. This appears to be a post-hoc threshold chosen to frame chi_ratio=1.88 as impressive ("quasi-critical") rather than simply reporting "no sharp transition observed."

---

## 8. Summary

| Aspect | Score | Notes |
|--------|-------|-------|
| Baseline Fairness | 3/5 | Self-referential comparisons, arbitrary thresholds |
| Metric-Claim Alignment | 3/5 | Good alignment but CV has contamination risk |
| Validity Threats | 2/5 | Selection bias in word selection, metric contamination in CV |
| Ablation Completeness | 3/5 | Major experiments done, but critical H3 not executed |
| Reproducibility | 3/5 | Core parameters specified, but missing env/version docs |
| Overall Credibility | 3/5 | Solid empirical work with methodological gaps |

**Verdict**: Publishable at mid-tier (AAAI/EMNLP/Workshop) with the recommended fixes. The core discoveries (finite-size scaling, CV reversal) are genuine but require protocol fixes to rule out confounds.
