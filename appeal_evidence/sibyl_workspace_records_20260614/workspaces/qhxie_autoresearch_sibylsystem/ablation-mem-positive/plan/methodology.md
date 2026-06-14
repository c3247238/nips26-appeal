# Methodology: CV-Based Actionability Decomposition in SAE Features

## Overview

This research validates that coefficient of variation (CV) predicts steering heterogeneity within absorbed SAE features. The front-runner candidate (cand_cv_actionability) proposes that high-CV absorbed features are more steerable than low-CV absorbed features.

## Research Questions

1. **RQ1**: Does CV predict steering effectiveness for absorbed SAE features? (VALIDATED - pilot + full)
2. **RQ2**: Are absorbed features uniformly non-steerable, or do they decompose into steerable/non-steerable subpopulations? (VALIDATED - evidence supports decomposition)
3. **RQ3**: Does CV-steering correlation generalize across architectures (GPT-2 to Gemma-2)?
4. **RQ4**: Does CV predict steering after controlling for activation magnitude (Fano factor)?

## Prior Results Summary

| Experiment | Status | Key Evidence |
|------------|--------|--------------|
| pilot_activation_patching | PASSED | 67.3% mean recovery, all 9/9 words >10% |
| pilot_steering_cv | PASSED | High-CV 0.153 vs Low-CV 0.075 (2x) |
| full_steering_cv | COMPLETED | High-CV 0.5251 vs Low-CV 0.3565 (1.47x, p<0.01) |
| full_decoder_orthogonality | COMPLETED | Results in exp/results/ |
| full_cross_architecture | PENDING | Gemma-2-2B replication |
| fano_factor_control | PENDING | Is CV just magnitude proxy? |
| non_absorbed_baseline | PENDING | Compare absorbed vs non-absorbed steering |
| held_out_activation_patching | PENDING | Validate on non-core words |

## Experiment Design

### E1: Fano Factor Control (Mechanism Validation)

**Purpose**: Verify CV is not purely a magnitude proxy. If CV-steering correlation disappears after controlling for Fano factor (CV^2/mean), then CV is just measuring activation magnitude.

**Method**:
- Compute Fano factor per feature: F = CV^2 / mean activation
- Regress steering effect on Fano factor alone
- Compare residual CV-steering correlation

**Pass Criteria**: If CV remains predictive after Fano factor control (p < 0.01), rate-distortion theory is supported.

**Falsification**: If CV effect disappears after Fano factor control, CV is magnitude proxy and H1 fails.

---

### E2: Non-Absorbed Baseline Comparison

**Purpose**: Contextualize absolute steering magnitude. Compare absorbed vs non-absorbed features to determine if "robust absorbed" (high-CV) is comparable to non-absorbed or still degraded.

**Method**:
- Select 30 absorbed high-CV features and 30 non-absorbed features
- Match activation magnitudes between groups
- Run steering at strength +5
- Compare steering effects

**Pass Criteria**: If absorbed high-CV effects are comparable to non-absorbed (ratio > 0.5), absorbed features retain practical steering utility.

**Falsification**: If absorbed high-CV effects remain significantly smaller than non-absorbed, the actionability paradox is partially upheld for absorbed features.

---

### E3: Cross-Architecture Validation (Gemma-2-2B)

**Purpose**: Test whether CV-steering correlation generalizes across architectures (TopK SAEs on GPT-2 vs JumpReLU SAEs on Gemma-2).

**Method**:
- Load Gemma-2-2B layer 6 JumpReLU SAE (GemmaScope)
- Compute per-feature CV on 1000 samples
- Use Gemma-specific median CV split (not GPT-2 threshold)
- Run steering on 30 high-CV vs 30 low-CV absorbed features at +5 strength

**Pass Criteria**: CV effect replicates on Gemma-2 with p < 0.01.

**Falsification**: If no CV effect on Gemma-2, the finding may be architecture-specific (TopK vs JumpReLU artifacts).

---

### E4: Held-Out Activation Patching Validation

**Purpose**: Validate that activation patching results are not specific to the 9 core words. Test on a separate set of words.

**Method**:
- Select 20 held-out words (not the 9 core words used in pilot)
- For each word, identify the top activating absorbed feature
- Zero the child feature and measure parent recovery
- Compare recovery rates to pilot results

**Pass Criteria**: Mean recovery > 30% (lower threshold due to smaller sample).

**Falsification**: If held-out words show <20% mean recovery, the pilot result may be an artifact of the specific words chosen.

---

## Metrics

### Primary Metrics
- **Steering Effect**: Logit change at target token positions
- **Effect Ratio**: high_CV_effect / low_CV_effect
- **Recovery Percentage**: (base_logits - patched_logits) / base_logits

### Statistical Tests
- One-sided Welch's t-test (alpha = 0.01)
- Benjamini-Hochberg FDR correction for multiple comparisons
- Effect size (Cohen's d) for practical significance

### Falsification Criteria
- **H1 falsification**: p > 0.05 after FDR correction for CV-steering correlation
- **H4 falsification**: CV_absorbed ≈ CV_non_absorbed at larger sample size
- **Cross-arch falsification**: No CV effect on Gemma-2 (p > 0.05)

---

## Expected Visualizations

1. **Figure 1**: Bar chart of steering effects (high-CV vs low-CV vs non-absorbed) with error bars
2. **Figure 2**: Scatter plot of CV vs steering effect with Fano factor color coding
3. **Table 1**: Cross-architecture replication results (GPT-2 vs Gemma-2)
4. **Figure 3**: Activation patching recovery distribution (pilot vs held-out words)

---

## Shared Resources

- **SAE**: `gpt2-small-res-jb` (layer 6, ~16k latents)
- **Gemma SAE**: `gemma-2b-res` (layer 6, JumpReLU)
- **Dataset**: Standard text prompts from pilot (5 templates)
- **Code**: Reuse existing steering and patching infrastructure from `exp/results/code/`

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Fano factor eliminates CV effect | Medium | If CV is magnitude proxy, frame as "CV captures magnitude which proxies for steering" |
| Gemma-2 shows no CV effect | Medium | Report as negative result; limits generalization claims |
| Held-out patching fails | Low | Use relaxed threshold; still report as exploratory |

---

## Timeline

| Task | Estimated Time | Dependencies |
|------|----------------|--------------|
| E1: Fano factor control | 20 min | None |
| E2: Non-absorbed baseline | 25 min | None |
| E3: Gemma-2 validation | 45 min | E1 (for interpretation) |
| E4: Held-out patching | 30 min | None |

**Total estimated GPU time**: ~120 minutes (2 hours)