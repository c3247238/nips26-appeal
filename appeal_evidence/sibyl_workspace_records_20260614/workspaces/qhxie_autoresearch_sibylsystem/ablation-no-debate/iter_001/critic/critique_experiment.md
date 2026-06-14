# Experiment Critique: Feature Absorption Study

## Overall Assessment

The experiments suffer from critical methodological flaws: a deterministic measurement masquerading as a statistical result, a broken steering experiment presented as a negative result, and severe data sparsity undermining correlation analysis. Only the basic multi-child ablation comparison (trained vs random decoder) is methodologically sound, and even that is limited to a single seed.

---

## Critical Issues

### 1. Deterministic H1 Result -- Not a Statistical Measurement (CRITICAL)

**Evidence**: multichild_absorption.json shows:
- All 100 samples: absorption_k5 = exactly 0.5 for trained SAE
- All 100 samples: overlap_parent_child1 = exactly 0.4
- All 100 samples: overlap_parent_child2 = exactly 0.6
- Std = 0.0 across all metrics

**Analysis**: This is not "low variance" -- it is ZERO variance. The measurement appears to be a deterministic geometric computation that reduces to a constant given the synthetic hierarchy parameters (cos=0.67). The "statistical test" vs random decoder compares a constant (0.5) against a variable (0.059 +/- 0.069) and reports d=8.94. While mathematically correct, this inflates perceived significance because the trained SAE has no within-condition variance.

**Implication**: The paper presents this as evidence that "trained SAEs show absorption rates of 0.50" but does not acknowledge that this rate is invariant to sample variation. It is a geometric identity, not a measured property.

**Fix**: Investigate the formula. If absorption_k5 = (overlap_c1 + overlap_c2) / 2 = (0.4 + 0.6) / 2 = 0.5 always, then the metric is not measuring what the paper claims. Report this explicitly.

---

### 2. H3 Steering -- Broken, Not Negative (CRITICAL)

**Evidence**: h3_steering_results.json shows:
- absorbed.baseline_mean = 37.44512171672082
- absorbed.steered_mean = 37.44512171672082 (identical to 15 decimal places)
- non_absorbed.baseline_mean = 185.43392871520658
- non_absorbed.steered_mean = 185.43392871520658 (identical to 15 decimal places)
- t_statistic = NaN, p_value = NaN

**Analysis**: The steering code did not modify activations at all. The parent_direction vector may be zero, or the apply_steering function may be a no-op. This is a bug, not a null result.

**Implication**: The paper's conclusion that "absorption may be epistemic rather than causal" is drawn from broken code. This is a severe error.

**Fix**: Remove all H3 conclusions. State: "Steering implementation failed; causal claims cannot be evaluated."

---

### 3. H2 Correlation -- Driven by Zero-Inflation (MAJOR)

**Evidence**: h2_frequency_correlation.json shows:
- 1024 features total
- Only 15 features have non-zero absorption (0.0175, 0.5, 0.5, 0.5335, 0.321, 0.5865, 0.179, 0.5, 0.5, 0.5, 0.5, 0.321, 0.5865, 0.5, 0.179)
- 98.5% of absorption values are exactly 0.0

**Analysis**: Computing Spearman correlation on 98.5% zeros is statistically problematic. The "positive correlation" may simply reflect that the few non-zero absorption values happen to occur on features with moderate-to-high frequency. There is no meaningful pattern here.

**Implication**: The paper presents rho=+0.17 as falsifying competitive exclusion, but the data distribution makes any correlation interpretation unreliable.

**Fix**: Report zero-inflation explicitly. Do not present H2 as a meaningful finding.

---

### 4. Severely Underpowered H3 (MAJOR)

**Evidence**: h3_steering_results.json shows n_absorbed_features = 7 out of 1021 total.

**Analysis**: With n=7, power to detect a medium effect (Cohen's d=0.5) at alpha=0.01 is approximately 18%. Even if steering worked, the experiment could not reliably detect effects.

**Implication**: The paper should not have attempted H3 with this sample size.

**Fix**: Report power calculation. State: "With n=7, the experiment was underpowered and cannot distinguish null from small effects."

---

### 5. Single Seed, Config Discrepancy (MAJOR)

**Evidence**:
- Paper states: d_model=512, d_sae=4096, 5 seeds
- multichild_absorption.json shows: d_model=128, d_sae=1024, seed=42
- Only L0=32 reported; L0=16 and L0=64 not mentioned

**Analysis**: The paper overstates experimental breadth. Only 1 of 15 planned configurations was actually run.

**Fix**: Report actual configuration (d_model=128, d_sae=1024, L0=32, seed=42) and state that other configurations are planned.

---

### 6. Missing Full Experiments (MAJOR)

**Evidence**: task_plan.json lists 6 tasks (h_safe_pilot, h3_fix_pilot, multichild_multiseed_pilot, h_mech_factorial, h_safe_full, visualize_results). experiment_state.json shows only 4 tasks with minimal completion.

**Analysis**: The full experimental plan was never executed. The paper presents pilot results as if they were full experiments.

**Fix**: Clearly label all results as "pilot" and state which full experiments are planned.

---

## What Works Well

1. **Multi-child ablation concept**: The core idea of ablating top-k children simultaneously is sound and addresses a real saturation problem.
2. **Baseline design**: Three baselines (random decoder, shuffled features, permuted encoder) follow Korznikov et al. appropriately.
3. **Honest reporting of raw data**: JSON files contain complete raw data, enabling external verification.

---

## Score: 4/10

One sound methodological idea (multi-child ablation) undermined by deterministic measurement, broken steering, zero-inflated correlation, and severe underpowering.
