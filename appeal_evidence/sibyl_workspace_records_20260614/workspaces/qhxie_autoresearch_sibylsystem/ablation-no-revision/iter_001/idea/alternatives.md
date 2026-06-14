# Alternative Research Directions

## Overview

This document outlines alternative approaches considered in the Round 5 synthesis. Based on the comprehensive result debate (6 perspectives), the primary focus is a negative-result paper on Ea-based routing failure. These alternatives provide pivot options if the main direction encounters obstacles.

---

## Alternative A: Expanded Negative Result (n=250)

**Status**: Immediate fallback if n=50 is criticized

**Perspective**: Empiricist + Methodologist

### Rationale

- The primary criticism of the current results is small sample size (n=50)
- Expanding to the full MATH test set (n=500, or at least n=250) would strengthen statistical power
- All core findings (H3 falsified, H5 falsified) are expected to hold at larger scale

### Hypothesis

H1: All Round 4 findings replicate at n=250 with stronger statistical power
H2: Per-problem fit failure rate remains >70% even with larger samples
H3: AUC remains <0.5 with pre-registered threshold

### Method

1. Run G0-G3 on full MATH test set (n=500) or stratified sample (n=250, 50 per level)
2. Use pre-registered threshold (median Ea) to eliminate data leakage
3. Report confidence intervals and effect sizes (Cohen's d, Cliff's delta)
4. Include cross-validation: optimize threshold on train set, report on test set

### Expected Timeline

- Pilot: 120 minutes (n=250)
- Full: 240 minutes (n=500)

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Findings reverse at larger scale | Low | Core patterns (AUC<0.5, per-problem fit failure) are structural |
| Compute cost | Medium | Batch inference on 250 problems x 16 samples = 4000 forward passes |
| Answer extraction errors scale | Medium | Fix extraction pipeline first |

---

## Alternative B: Cross-Model Validation

**Status**: Backup if reviewer questions model-specificity

**Perspective**: Comparativist + Empiricist

### Rationale

- Current results are on Qwen2.5-Math-7B-Instruct only
- Reviewers may ask whether Ea-routing failure is model-specific
- Testing on multiple models would strengthen generalization claims

### Hypothesis

H1: Ea-routing failure (AUC<0.5) replicates across model families
H2: Aggregate Arrhenius saturation holds across models (with different P_inf, k_0)
H3: Ea-difficulty correlation holds across models

### Method

1. Test on DeepSeek-Math-7B (26% baseline) -- already available
2. Test on Qwen2.5-Math-1.5B (smaller, faster)
3. Test on Qwen2.5-Math-14B or 72B if available (larger, stronger)
4. Compare saturation parameters (P_inf, k_0) across models
5. Compare Ea distributions and routing AUC across models

### Expected Timeline

- Per model: 90 minutes pilot, 180 minutes full
- 3 models: 270 minutes pilot, 540 minutes full

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DeepSeek too weak (26%) | High | Use for qualitative comparison only |
| Larger models not available | Medium | Focus on 1.5B and 7B |
| Results vary across models | Medium | Frame as "model-dependent boundary" rather than universal failure |

---

## Alternative C: Error Depth Targeted Training (EDTT)

**Status**: Long-term backup if negative result paper is rejected

**Perspective**: Innovator + Pragmatist

### Rationale

- Li (2026) shows L3 errors resist correction
- If routing cannot predict solveability, perhaps training can
- GPU is now available (PyTorch 2.11.0 + Blackwell support)
- Round 2 training failed due to weak model + hardware; both issues now resolved

### Hypothesis

H1: EDTT reduces L3 error rate more than standard GRPO
H2: Error depth targeting improves over generic training by >5%
H3: Depth-aware reward shaping in GRPO is effective

### Method

1. Classify MATH errors by depth (L1/L2/L3) using Python execution
2. Train GRPO with depth-aware rewards (higher penalty for L3)
3. Use selective token-level optimization (S-GRPO)
4. Compare to standard GRPO baseline on Qwen2.5-Math-7B

### Expected Timeline

- Pilot: 120 minutes
- Full: 240 minutes

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| L3 classification unreliable | Medium | Human validation subset |
| GRPO training unstable | Medium | Warm start, tune LR, small batch |
| EDTT doesn't outperform GRPO | Medium | Compare SwS baseline |
| Training cost | High | Use LoRA/QLoRA to reduce memory |

---

## Alternative D: Routing Signal Comparison Study

**Status**: Conditional backup (only if H5 entropy experiment shows promise)

**Perspective**: Pragmatist + Interdisciplinary

### Rationale

- ACAR found entropy routing has weak correlation
- Our H5 (Ea vs k_0) is falsified
- A comprehensive comparison of routing signals could be valuable
- But only if at least one signal shows meaningful predictive power

### Hypothesis

H1: No single routing signal (Ea, entropy, confidence, variance) achieves AUC > 0.6
H2: Hybrid signals outperform individual signals
H3: "No free lunch" -- different signals work for different problem types

### Method

1. Extract multiple signals per problem:
   - Ea (consistency convergence rate)
   - Token entropy (average per-sample)
   - Self-consistency variance (ACAR's sigma)
   - Confidence (logprob of final answer)
   - Problem features (length, symbol density)
2. Compare AUC for each signal
3. Test hybrid combinations
4. Analyze which signal works for which error type

### Expected Timeline

- Pilot: 90 minutes
- Full: 180 minutes

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| All signals fail | High | Frame as "comprehensive negative result" |
| Paper becomes unfocused | Medium | Keep tight scope; report concise comparison table |
| Computational overhead | Low | All signals computable from existing data |

---

## Alternative E: H1-H2 Validation Paper (Minimalist Fallback)

**Status**: Last resort if all else fails

**Perspective**: Empiricist

### Rationale

- H1 (Arrhenius kinetics) and H2 (Ea = difficulty) are solid findings
- Acknowledge Yang et al. (2508.16456) collision explicitly
- Focus on empirical validation and per-level analysis
- Publish as "independent validation + model-specific parameters"

### Hypothesis

H1: Arrhenius kinetics holds across multiple model families
H2: Ea-difficulty correlation generalizes

### Method

1. Test on multiple models (see Alternative B)
2. Report per-model saturation parameters
3. Compare k_0 and P_inf across model sizes
4. Acknowledge Yang et al. as prior work

### Expected Timeline

- Per model: 90 minutes
- 3 models: 270 minutes

### Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Reviewers ask "so what?" | High | Emphasize model-specific parameter estimates |
| Yang et al. collision | Critical | Acknowledge explicitly; frame as validation, not novelty |

---

## Decision Tree

```
START: Round 5 Pilot (H4 + P1)

|
v
Does H4 support "stability != correctness" narrative?
|
+-- Yes (any error mix) --> Continue to P1
|
+-- No (inconclusive) --> Expand to n=250 (Alt A)
|
v
Does P1 confirm AUC < 0.5 with pre-registered threshold?
|
+-- Yes --> PROCEED with negative-result paper
|
+-- No (AUC > 0.5) --> Re-evaluate; consider Alt D (signal comparison)
|
v
Paper written and submitted

|
v
Reviewer feedback:
|
+-- Accept / Minor revision --> DONE
|
+-- Sample size criticism --> Alt A (n=250 expansion)
|
+-- Model-specificity criticism --> Alt B (cross-model validation)
|
+-- Reject negative result --> Alt C (EDTT training) or Alt E (H1-H2 validation)
|
v
All alternatives exhausted --> Publish as tech report / arXiv
```

---

## Summary of Alternatives

| Alternative | Novelty | Complexity | Timeline | Risk | Trigger Condition |
|-------------|---------|------------|----------|------|-------------------|
| A: n=250 Expansion | 5/10 | Low | 120 min | Low | Sample size criticism |
| B: Cross-Model | 6/10 | Medium | 270 min | Medium | Model-specificity criticism |
| C: EDTT Training | 8/10 | High | 240 min | High | Negative result rejected |
| D: Signal Comparison | 6/10 | Medium | 180 min | Medium | H5 entropy shows promise |
| E: H1-H2 Validation | 4/10 | Low | 270 min | Medium | All else fails |

---

## Recommendation

**Primary path**: Negative-result paper with current n=50 data + H4 error classification.

**If criticized for sample size**: Execute Alternative A (n=250 expansion) as revision.

**If criticized for model-specificity**: Execute Alternative B (cross-model) as revision.

**If rejected entirely**: Execute Alternative C (EDTT training) as new direction.

The negative-result framing is the strongest contribution because:
1. It is honest about what the data shows
2. It saves community resources
3. It cross-validates ACAR's findings
4. It provides a clear methodological template for future routing signal evaluation
