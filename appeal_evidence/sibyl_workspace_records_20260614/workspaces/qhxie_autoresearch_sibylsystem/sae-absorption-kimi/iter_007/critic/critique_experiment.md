# Experiment Critique: Iteration 009

## Executive Summary

The experiments reveal a genuinely important methodological observation (L0 confounds absorption comparisons) but suffer from critical metric issues, computational anomalies, and insufficient statistical analysis that undermine the central claims.

## Critical Findings

### 1. MCC is Not a Valid Downstream Metric (CRITICAL)

**Evidence**:
- Random SAE (untrained): MCC = 0.222 ± 0.0004
- Baseline L1 (trained): MCC = 0.216 ± 0.0004
- TopK (trained): MCC = 0.213 ± 0.0013
- Matryoshka (trained): MCC = 0.220 ± 0.0004
- OrtSAE (trained): MCC = 0.222 ± 0.0003

**Analysis**: MCC is essentially identical across all variants including Random. This means either:
(a) All SAEs recover features at chance level, or
(b) MCC is not measuring what we think it measures.

The classification metrics tell a different story:
- Random: accuracy = 0.497 (chance), F1 = 0.0034
- Baseline: accuracy = 0.529, F1 = 0.0034
- OrtSAE: accuracy = 0.731, F1 = 0.0043
- TopK: accuracy = 0.974, F1 = 0.0012

OrtSAE shows genuinely better classification performance (73% vs 50% accuracy), but this does NOT correlate with absorption rate (OrtSAE absorption = 0.247, Baseline = 0.254).

**Implication**: The null causal result is uninterpretable because MCC is flat everywhere. A flat metric cannot detect causal relationships.

### 2. Explained Variance Anomaly (CRITICAL)

**Evidence**:
- Baseline: EV = -0.88 ± 0.16
- Gated: EV = -0.49 ± 0.12
- TopK: EV = -0.39 ± 0.06
- Matryoshka: EV = -0.28 ± 0.08
- OrtSAE: EV = +0.994 ± 0.0001

**Analysis**: Negative explained variance means the model predicts worse than the mean. This is possible for poorly trained models, but the 1.8-unit gap between OrtSAE (+0.994) and Baseline (-0.88) is implausible. OrtSAE achieves near-perfect reconstruction (MSE = 3.1e-5) while Baseline has MSE = 0.010. The reconstruction quality difference is real, but the explained variance computation may have a bug.

**Action**: Debug explained_variance before reporting.

### 3. Dead Latent Catastrophe (CRITICAL)

**Evidence**:
- TopK: 1677/2048 = 81.9% dead latents
- Matryoshka: 1151/2048 = 56.2% dead latents
- Baseline: 0/2048 = 0% dead latents
- Gated: 0/2048 = 0% dead latents

**Analysis**: TopK and Matryoshka are operating with only ~350-900 active features. This has two implications:

1. **Absorption rate may be artificially suppressed**: With fewer active features, there are fewer opportunities for parent-child co-activation. The absorption metric (fraction of parent firings where child latents also activate) could be lower simply because fewer latents fire overall.

2. **Dictionary underutilization**: A 2048-feature dictionary with 82% dead latents is effectively a ~350-feature dictionary. The comparison with Baseline (0% dead, ~964 active features) is not just about L0—it's about dictionary utilization.

**Action**: Recompute absorption rates using only non-dead latents. Compare effective dictionary sizes.

### 4. L0-Matching is Possible (MAJOR)

**Evidence from pilot data**:
- `pilot_rq1_l0_match_lambda_0.02.json`: L0 = 50.0, absorption = 0.495
- `pilot_rq1_l0_match_lambda_0.01.json`: L0 = 50.0, absorption = 0.495
- `pilot_rq1_l0_match_lambda_0.005.json`: L0 = 50.0, absorption = 0.495

**Analysis**: Baseline L1 CAN achieve L0=50 at lambda ≥ 0.005. The full experiment did not include these lambda values, creating a false impression of impossibility. At L0=50, Baseline absorption is ~0.495 (similar to Random), which is actually HIGHER than TopK/Matryoshka at the same L0. This suggests that at matched L0, TopK/Matryoshka DO have lower absorption—but the paper misses this because it claims matching is impossible.

**Action**: Include lambda ≥ 0.005 in the full experiment for true L0-matched comparison.

### 5. Training Time Anomaly (MAJOR)

**Evidence**:
- full_baseline: 13.98s for 5 seeds = 2.8s/seed
- full_topk: 16.48s for 5 seeds = 3.3s/seed
- full_matryoshka: 23.56s for 5 seeds = 4.7s/seed

**Analysis**: Training on 2M tokens with batch size 1024 = 1953 steps. At 2.8s total, that's 1.4ms per step. This is implausibly fast for GPU training of a 256→2048→256 autoencoder on 2M samples. Possible explanations:
1. Models loaded from cached checkpoints
2. Training loop has a bug (e.g., not actually iterating)
3. Data is trivially small (but 2M tokens should take longer)
4. The timing includes only evaluation, not training

**Action**: Verify training actually occurred. Check loss curves.

### 6. Missing Statistical Analysis (MAJOR)

No statistical tests are reported despite promises in the methodology. Key missing analyses:

1. **Architecture comparison**: Welch's t-test for each variant vs. Baseline
2. **Dose-response**: Pearson r between absorption and MCC across 25 measurements
3. **Ablation**: t-test for Matryoshka flat vs. nested, OrtSAE with/without penalty
4. **Effect sizes**: Cohen's d for all comparisons

**Action**: Compute and report all promised statistical tests.

### 7. Dose-Response Design Incomplete (MINOR)

The methodology promises two manipulations:
- Manipulation A (architectural): Use variants with naturally different absorption rates
- Manipulation B (sparsity-induced): Fix architecture, vary lambda

Only Manipulation B is reported. Manipulation A would compare absorption rates across architectures (0.056 to 0.495) against their MCC values—but MCC is flat, so this manipulation also shows null.

**Action**: Report Manipulation A results or remove the claim about two manipulations.

## Summary of Experimental Issues

| Issue | Severity | Status |
|-------|----------|--------|
| MCC at chance level | Critical | Unresolved |
| Explained variance anomaly | Critical | Unresolved |
| Dead latent rates misreported | Critical | Data error |
| L0-matching possible per pilot | Major | Contradicts claims |
| Training time suspicious | Major | Needs verification |
| No statistical tests | Major | Missing |
| Manipulation A not reported | Minor | Incomplete design |
| Mutual coherence no correlation | Minor | Needs statistical test |
