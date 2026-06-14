# Methodologist Audit: Competitive Geometry of Feature Absorption in SAEs

**Auditor**: Methodologist Agent  
**Date**: 2026-04-14  
**Scope**: Full experiment results for all components (P0, C1A-C1D, C2A-C2D, C3A-C3C, ablations)

---

## 1. Baseline Fairness Audit

### 1.1 Model Substitution (CRITICAL)

The proposal specifies **Gemma 2 2B with Gemma Scope SAEs** as the target model. All experiments were actually run on **GPT-2 Small** with JB/v5 SAEs due to "gated HuggingFace access" for Gemma. This is the single most consequential methodological deviation in the study.

**Impact assessment:**
- The entire LV competitive exclusion framework (H1) was calibrated and evaluated on GPT-2 Small, which has fundamentally different architecture (768-dim vs 2304-dim, 12 layers vs 26 layers, different tokenizer, different training data).
- SAE widths tested are 24k/49k/98k (GPT-2 JB feature-splitting SAEs) instead of 16k/65k/131k (Gemma Scope). These are not directly comparable.
- The proposal's theoretical predictions (e.g., alpha_ij threshold near 1.0) were derived with Gemma Scope geometry in mind. GPT-2 Small has different superposition and polysemanticity characteristics.
- **The C3A SAEBench correlation (the one genuinely positive result) used actual Gemma Scope data** from HuggingFace, making it the only component that operates on the intended model. This creates an asymmetry: the detector (H1) is calibrated on GPT-2, while the downstream correlation (H3) is measured on Gemma -- they cannot be directly related.

**Verdict**: The model substitution means no result from Components 1, 2, or the safety probe (C3C) can be directly attributed to the Gemma Scope ecosystem that the proposal targets. The paper cannot claim validation on Gemma Scope SAEs based on GPT-2 Small experiments.

### 1.2 Baseline Comparisons

- **Cosine-similarity baseline** (C1B, C1 ablations): Fairly implemented. The same data splits and evaluation protocol are used for both the LV detector and the cosine baseline. However, the cosine baseline **outperforms** the LV detector (test F1 = 0.165 vs 0.128), which is the opposite of the proposal's prediction. This is an honest negative result.
- **Chanin et al. ground truth**: Used as intended via `sae-spelling`. However, the ground truth labels are generated on the same model (GPT-2 Small, layer 8) and thus are internally consistent, even though the model differs from the proposal.
- **SAEBench absorption scores** (C3A): These are pre-computed on Gemma 2 2B -- a fair and appropriate baseline. The analysis correctly filters to Gemma 2 2B SAEs only and applies Bonferroni correction.
- **Dense linear probe** (C3C): Fairly implemented with the same 5-fold CV protocol across all three SAE configurations.

### 1.3 Hyperparameter Budget Asymmetry

The LV detector gets a 5-value tau sweep {0.5, 0.75, 1.0, 1.25, 1.5} plus a finer 17-value sweep in ablations, while the cosine baseline gets a 6-value threshold sweep {0.15, 0.2, 0.25, 0.3, 0.35, 0.4}. The search ranges are not equivalent in granularity. This mildly favors the LV detector, yet it still underperforms the cosine baseline, making this asymmetry inconsequential.

---

## 2. Metric-Claim Alignment

| Claimed Contribution | Evaluation Metric | Alignment |
|---|---|---|
| Probe-free unsupervised detector (H1) | F1, ROC-AUC against sae-spelling labels | ADEQUATE -- F1 directly measures detection quality |
| Corpus PMI predictor (H2) | Partial R-squared for PMI term in regression | ADEQUATE -- partial R-squared after controlling for SAE config is the right metric |
| Downstream causal chain (H3) | Pearson r between absorption and SAEBench tasks | PARTIALLY ADEQUATE -- correlation is not causation; the study correctly acknowledges this but the paper title uses "impact" language |
| Width paradox (H4) | Fraction of letters with positive DAS(k=3) slope | ADEQUATE but UNDERPOWERED -- only 3 width points per letter makes slope estimation unreliable |
| Absorption taxonomy (H5) | Fraction of letters in each type | PROBLEMATIC -- see Section 3 |

### Measurement Gap: H3 "Impact" vs. Correlation
The proposal claims to provide "the first controlled test of the downstream causal chain." However, C3A is purely correlational (Pearson/Spearman across SAEs). C3B (matched RAVEL comparison) gets closer to a causal test but uses only pre-computed SAEBench data without actual RAVEL task re-running. The claim of testing the "causal chain" is overstated for what is observational data analysis.

---

## 3. Validity Threats Checklist

### Data Leakage
- [PASS] Calibration/test split (letters A-M / N-Z) prevents leakage in C1B threshold calibration.
- [PASS] 5-fold CV in C3C prevents train-test contamination.
- [CONCERN] C2B absorption survey uses the same sae-spelling library for both generating labels (used in C1B ground truth) and measuring absorption rates. This is by design, but means the study is validating the LV detector against a specific operationalization of absorption (Chanin et al.) rather than absorption as a phenomenon.

### Contamination
- [N/A] No model training is performed; all experiments are inference-only. Pre-training contamination of GPT-2/Gemma is not relevant to the experimental protocol.

### Selection Bias
- [CONCERN] The pilot P0 uses a "lenient" gate criterion ("Final: GO" despite alpha gate FAIL). The alpha_ij sanity check -- the most LV-theory-specific check -- failed (no letter-feature pairs found in top-10 alpha_ij pairs). The decision to proceed anyway undermines the pilot's purpose as a validity gate.
- [CONCERN] C1B calibration selects the best tau from the sweep. The best tau is 0.5 (the minimum of the sweep range), meaning the "optimal" threshold is at the boundary. This suggests the true optimal may be below the searched range, or that the relationship between alpha_ij and absorption is not threshold-like.

### Overfitting to Evaluation
- [CONCERN] All H1 results are evaluated on a single domain (first-letter task) with a single model (GPT-2 Small). Generalizability to other feature hierarchies (cities, parts-of-speech, safety-relevant features) is entirely unvalidated.
- [MODERATE CONCERN] The C1C cross-architecture "validation" tests v5-32k and v5-128k SAEs, but these use different hook points (resid_post vs resid_pre), different training objectives, and the alpha_ij data was collected at a different layer (layer 9 for v5 vs layer 8 for JB). The F1 scores are 0.009 and 0.0 respectively -- catastrophic failure, but the failure may partly reflect architectural mismatch rather than LV theory failure.

---

## 4. Ablation Gap Analysis

| Proposed Component | Ablation Present? | Assessment |
|---|---|---|
| LV coefficient alpha_ij | YES (A1: tau sweep, A2: cosine pre-filter) | Thorough |
| Frequency ratio f_j/f_i | NO | MISSING -- no ablation separating the frequency ratio from the co-activation term sigma_ij |
| Decoder cosine pre-filter | YES (A2: thresholds 0.10, 0.15, 0.25) | Present but shows coverage at 0.15 is only 34%, far below the 80% criterion |
| PMI regression | YES (A3: PMI-only, A4: per-layer) | Thorough |
| DAS(k=3) vs DAS(k=1) | IMPLICIT -- compared in C1D | Present |
| Type II classification threshold (0.5 magnitude ratio) | NO | MISSING -- the 0.5 threshold for Type II is arbitrary, and the result (88.5% Type II rate) depends critically on it |
| Comparison set for Type II expected magnitude | NO | MISSING -- n_comparison_tokens = 0 for most letters, leading to global fallback; no ablation tests this |

**Critical missing ablation**: The LV competition coefficient alpha_ij = sigma_ij * (f_j / f_i) has two components. No experiment isolates whether sigma_ij alone (co-activation rate) or the frequency ratio alone explains any predictive power. The cosine baseline ablation (A2) tests a different quantity entirely (decoder geometry). The proper ablation would be: (a) sigma_ij only as detector, (b) f_j/f_i only as detector, (c) full alpha_ij. Without this, the claim that the LV framing adds value over simpler co-activation metrics is unsupported.

---

## 5. Reproducibility Score: 3/5

| Criterion | Score | Notes |
|---|---|---|
| Random seeds fixed | 1/1 | Seed 42 used throughout |
| Hyperparameters specified | 0.5/1 | Most specified; but model substitution means the reported hyperparameters (tau=0.5 on GPT-2) may not transfer to Gemma |
| Code/data available | 0.5/1 | Code exists in exp/code/; sae-spelling and SAELens are public; but Gemma Scope access blocked during experiments |
| Hardware requirements documented | 0.5/1 | A100 mentioned in proposal; actual hardware not recorded in result JSONs |
| Reproducible within 10% | 0.5/1 | GPT-2 Small results likely reproducible; but Gemma results cannot be reproduced from this data since they were never run |

**Key issue**: The entire experimental protocol needs to be re-run on Gemma 2 2B before any paper claim about Gemma Scope can be made. The GPT-2 Small results serve as a pilot at best.

---

## 6. Top-3 Recommendations (by effort-to-credibility ratio)

### Recommendation 1: Re-run C1B and C1 ablations on Gemma 2 2B (HIGH PRIORITY, MEDIUM EFFORT)

The LV detector is the paper's central contribution. Its complete failure (F1 = 0.128, worse than cosine baseline) on GPT-2 Small is ambiguous: it could reflect a genuine failure of LV theory, or it could reflect the model substitution. Before concluding that H1 is falsified, the experiment must be repeated on the intended model.

**Specific action**: Obtain gated HuggingFace access to `google/gemma-2-2b`, load Gemma Scope SAEs at layer 12 widths 16k/65k, and re-run C1B with the same protocol. This is the single experiment that would most change the paper's conclusions.

**What would change the conclusion**: If F1 > 0.50 on Gemma Scope (meeting the pre-specified criterion), H1 is supported and the GPT-2 results become "cross-model transfer is poor" rather than "the theory is wrong." If F1 remains < 0.50, then H1 is genuinely falsified across models.

### Recommendation 2: Fix the Type II taxonomy classification (HIGH PRIORITY, LOW EFFORT)

The Type II rate of 88.5% (23/26 letters) is self-flagged as "CRITICAL: likely inflated" in the results JSON. The taxonomy result (H5) is currently not credible because:
- Parent features were identified by a selectivity heuristic, not by sae-spelling ground truth
- The comparison set has n_comparison_tokens = 0 for most letters, forcing a fallback to global mean
- The magnitude ratio < 0.5 threshold systematically fires because the identified parents are not the true first-letter parents

**Specific action**: Use sae-spelling's actual parent feature IDs (already available from C1B output) as the parent features for taxonomy classification. Compute expected magnitude from a proper comparison set (tokens where the parent should fire but absorption is not expected). Re-classify all 26 letters.

**What would change the conclusion**: A properly calibrated taxonomy would likely show a lower Type II rate (perhaps 40-60%), which is still higher than the Type I rate and supports the claim that comprehensive absorption exceeds the Chanin metric. But 92.3% is not believable and would draw immediate reviewer skepticism.

### Recommendation 3: Report the C3A SAEBench result as the paper's lead finding (LOW EFFORT, HIGH IMPACT)

The C3A SAEBench correlation analysis is the experiment where the original hypothesis (H3: |r| < 0.2, absorption is disconnected from downstream tasks) is clearly **falsified**. The actual results show:
- Pearson r = -0.59 for sparse probing (partial r = -0.66 controlling for config)
- Pearson r = -0.43 for SCR (partial r = -0.68)
- Pearson r = -0.45 for RAVEL proxy (partial r = -0.49)
- All significant after Bonferroni correction

This is the strongest, most credible result in the entire study: it uses actual Gemma Scope data from SAEBench, applies proper statistical controls (Bonferroni, partial correlation), and the effect sizes are large. The falsification of H3 (absorption DOES correlate with downstream quality) is arguably more important than the proposed contributions. It directly addresses the DeepMind concern and validates the research motivation.

**Specific action**: Restructure the paper around this finding. Make the SAEBench downstream correlation the primary result (with the H3 falsification as the headline). Demote the LV detector to "exploratory analysis" or "preliminary methodology" given its current performance. This reframing is honest, impactful, and defensible.

---

## Evidence Samples

### Sample 1: LV Detector Underperformance
```
C1B test set: LV detector (tau=0.5) F1 = 0.128, AUC = 0.148
C1B test set: Cosine baseline (threshold=0.2) F1 = 0.165, AUC = 0.201
Delta: F1 = -0.037, AUC = -0.053 (LV WORSE than cosine)
```
The proposed LV competition coefficient does not outperform the simpler cosine-similarity baseline. The sharpness test confirms this: AIC favors linear over sigmoid (AIC_sigmoid = -60.95, AIC_linear = -61.05), meaning there is no sharp transition near alpha_ij = 1 as predicted by LV theory.

### Sample 2: PMI Non-Predictive
```
C2C regression: PMI coefficient = -0.006303, p = 0.593
Partial R^2 for PMI = 0.0006 (criterion: >= 0.10)
Per-letter Pearson (mean absorption vs log_PMI) = -0.080, p = 0.699
```
PMI has no meaningful predictive power for absorption rates, and the coefficient sign is opposite to the hypothesis. H2 is clearly not supported.

### Sample 3: Width Paradox Partial Failure
```
H4 prediction: DAS(k=3) positive slope for >= 80% of letters
H4 observed: DAS(k=3) positive slope for 42.3% of letters
H4 prediction: DAS(k=1) non-positive slope for >= 60% of letters  
H4 observed: DAS(k=1) non-positive slope for 57.7% of letters
```
Only the k=1 prediction marginally passes (57.7% vs 60% criterion). The k=3 prediction decisively fails (42.3% vs 80%). The width paradox explanation via distributed competitive exclusion is not supported.

---

## Summary Assessment

| Hypothesis | Verdict | Confidence |
|---|---|---|
| H1 (LV detector F1 > 0.65) | **FALSIFIED** on GPT-2 Small (F1 = 0.128) | Medium (model substitution creates ambiguity) |
| H2 (PMI predicts absorption, partial R^2 > 0.10) | **FALSIFIED** (partial R^2 = 0.0006) | High |
| H3 (Absorption disconnected from downstream, \|r\| < 0.2) | **FALSIFIED** (mean \|r\| = 0.41) -- this is GOOD news | High (uses actual Gemma Scope data) |
| H4 (DAS(k=3) monotonically increases with width) | **PARTIALLY FALSIFIED** (42.3% vs 80% criterion) | Medium (only 3 width points) |
| H5 (Comprehensive rate > Chanin Type I rate) | **SUPPORTED** but with inflated Type II rate | Low (taxonomy methodology is flawed) |

The study has one genuinely strong positive finding (C3A: absorption predicts downstream SAE quality) and several honest negative results. The methodology for the central contribution (LV detector) is sound in protocol design but critically undermined by the model substitution. The paper can be restructured around the SAEBench finding and the negative results as informative nulls, producing a credible if less ambitious contribution.
