# Methodologist Analysis — Iteration 5 Result Debate

**Role**: Methodologist Agent
**Date**: 2026-03-18
**Scope**: Experimental methodology audit for "When Does Phi Modulation Matter? The ρ = λ/η Regime Boundary and Its Implications for Dynamic Weight Decay Design"
**Evidence base**: sgd_baseline_analysis.json, pilot summary.json files, experiment_plan.md, proposal.md, hypotheses.md, code review of optimizers.py and train_unified.py

---

## 1. Baseline Fairness Audit

### 1.1 AdamW vs SGD Comparison Asymmetry (HIGH SEVERITY)

**Finding**: The AdamW/SGD comparison is confounded by different hyperparameter regimes.

| Parameter | AdamW config | SGD config | Ratio |
|---|---|---|---|
| lr | 1e-3 | 0.1 | 100× |
| wd (base) | 5e-4 | 5e-3 | 10× |
| LR schedule | cosine annealing | multistep [80,120,160] | different |

The paper's central claim is that the 18.3× effect ratio between SGD and AdamW is attributable to the optimizer's mechanism (AdamW's implicit ℓ∞ constraint). However, this ratio conflates three independent sources:
- The optimizer mechanism itself (the intended signal)
- The 10× difference in absolute λ (ρ_SGD ≈ 5e-3/0.1 = 0.05 initially vs ρ_AdamW ≈ 0.5)
- The different LR schedule shapes (cosine vs multistep)

**Specifically**: The SGD ρ = λ/η at peak LR is 5e-3/0.1 = 0.05, whereas at later multistep stages (lr=0.001) it is 5e-3/1e-3 = 5.0. The AdamW ρ is approximately 5e-4/1e-3 = 0.5 throughout cosine decay. The SGD experiment is thus traversing Regime I → Regime III according to the paper's own Trichotomy. This makes the "18.3× effect ratio" a ratio of effects measured in different regime regimes, not a clean optimizer comparison.

**Required fix**: Add matched-λ SGD control: `wd=5e-4, lr=0.1, momentum=0.9, multistep schedule`. This isolates optimizer mechanism from WD magnitude. Without this, the 18.3× ratio cannot be attributed cleanly to AdamW's mechanism.

### 1.2 Baselines Within AdamW Experiments (LOW SEVERITY)

All AdamW variants (constant, cosine_schedule, cwd_hard, swd, half_lambda, random_mask, no_wd) share identical hyperparameters (lr=1e-3, wd=5e-4, batch=128, cosine LR, 200 epochs). The Phi Modulator architecture ensures only the WD computation differs. This is a strong and fair design: no baseline received privileged hyperparameter tuning.

**One concern**: The method-specific parameters (cwd_beta=100.0, swd_sensitivity=1.0) are fixed defaults and were not searched. If any method has a better-than-default parameter setting, the invariance claim might break at those settings. This is especially relevant for CWD (beta sensitivity is theoretically meaningful as beta → ∞ approaches perfect hard mask).

### 1.3 CIFAR-100 SGD no_wd: n=1 Data Point (CRITICAL)

From sgd_baseline_analysis.json, CIFAR-100 SGD no_wd has `"n": 1, "best_std": 0` — a single seed with fabricated standard deviation of zero. This data point cannot be used in any statistical test and should be excluded from all tables claiming n=3. The Cohen's d listed as 0.0 for this comparison is mathematically undefined with n=1 in the comparison group.

---

## 2. Metric-Claim Alignment

### 2.1 Primary Claim: "Phi Invariance under AdamW"

**Claim**: All budget-equivalent WD strategies under AdamW produce accuracy within ±0.5% of constant WD.
**Metric used**: Raw accuracy difference, paired t-test, Bonferroni-Holm correction.
**Gap**: The paper uses significance testing (H0: methods differ) when the claim actually requires equivalence testing (H0: methods are NOT equivalent within δ). Using a non-significant t-test to claim equivalence is a methodological error. The proposal acknowledges TOST is needed but iter_003 results rely on non-significant t-tests.

**Current n=3 TOST power**: With σ ≈ 0.30% (observed AdamW CIFAR-10 between-seed std), δ=0.5%, n=3:
- SE = 0.30/√3 ≈ 0.173%
- t_critical = 2.35 (one-sided, df=2, α=0.05)
- Power ≈ 0.15-0.20 (severely underpowered)

This means even if the methods ARE equivalent at δ=0.3%, n=3 has only ~15-20% chance of achieving TOST significance. The plan calls for n=5 (estimated power ~40%), still below the 80% minimum standard.

### 2.2 Secondary Claim: "18.3× Effect Ratio is the SGD/AdamW Contrast"

**Claim**: SGD shows 18.3× larger effect for WD strategy choice than AdamW.
**Metric**: Ratio of Cohen's d values for (constant vs no_wd) between SGD and AdamW.
**Gap**: The claim is valid but the reported Cohen's d of 10.29 (SGD CIFAR-10) vs an implied ~0.56 for AdamW is computed from different datasets with potentially different underlying variances. The correct ratio should be effect size ratio (Cohen's d ratio) or accuracy difference ratio, reported with bootstrap CIs. The current analysis does not provide CIs for the ratio.

### 2.3 ρ = λ/η as Order Parameter: Not Yet Tested

**Claim**: ρ = λ/η determines the regime boundary for WD strategy utility.
**Current evidence**: Zero direct evidence. The λ sweep (H3, P1-1 experiment) that would test this has not been conducted. The ρ framework is theoretical extrapolation from two data points (ρ=0.5 for AdamW showing invariance, ρ≫1 implicitly assumed for SGD). This is the paper's most novel claim but currently has no empirical support beyond construction.

### 2.4 BEM/AIS/CSI Diagnostic Metrics

**BEM**: After the fix (removing abs()), BEM correctly tracks budget deviation. The pilot confirms half_lambda=-0.5, cosine_schedule≈-0.6. However, BEM captures only the time-average WD; two schedules with identical BEM can have orthogonal temporal profiles. This limitation is not discussed in the proposal.

**AIS**: The entropy-of-alignment measure (AIS ∈ [0,1]) is reasonable but computed from the last batch of each epoch only — this is a single-point sample from a stochastic quantity. Observed values cluster in 0.25-0.50 (pilot data), with limited discriminative power between methods.

**CSI**: CV of weight norm changes over last 10 epochs. The observed clustering (0.80-1.01) suggests methods produce similar weight stability under AdamW. However, this metric is not falsifiable — if all methods converge to similar trajectories under AdamW (because of Phi Invariance), CSI will necessarily be similar regardless of method.

---

## 3. Validity Threats Checklist

- [x] **Data leakage**: No evidence. Test sets are standard CIFAR-10/100 splits. Not a concern here.
- [x] **Contamination**: N/A for supervised CIFAR training from scratch.
- [PARTIAL] **Selection bias via hyperparameter tuning**: The single (lr, wd) configuration was presumably chosen to produce "standard" results. If the AdamW configuration was iteratively tuned until methods appeared equivalent, the Phi Invariance finding would be circular. The proposal does not discuss how the final hyperparameter configuration was selected.
- [CRITICAL] **Overfitting to evaluation**: The ρ framework was constructed post-hoc to explain observed data (ResNet-20 CIFAR-10 AdamW invariance + SGD sensitivity). The λ sweep (P1-1) is the first prospective test of the framework's predictions. Until P1-1 is completed, the Trichotomy is a retroactive explanation, not a prospective theory. This is a significant threat to internal validity.
- [MINOR] **Seed selection**: Seeds {42, 123, 456} are conventional choices that may have been selected after observing preliminary results. True pre-registration would require committing to seeds before any data collection.

---

## 4. Ablation Gap Analysis

| Proposed Component | Ablation Exists? | Quality |
|---|---|---|
| AdamW mechanism (vs SGD) | YES | GOOD — direct comparison, but confounded by different λ |
| ρ = λ/η regime boundary | NO | Missing — P1-1 λ sweep not yet run |
| BN role (BN vs NoBN) | NO | Missing — P0-3 BN ablation not yet run |
| CWD alignment signal | Partial | random_mask provides a random alignment baseline |
| Half-λ budget equivalence | YES | Exists — half_lambda is the "same budget at constant" ablation |
| Dynamic vs constant (no budget matching) | YES | no_wd exists |
| Architecture effect | NO | VGG-16-BN experiments not yet completed |
| Scale effect | NO | ImageNet experiments not yet completed |

**Critical missing ablation**: The BN confound ablation (ResNet-20 without BN) is identified as P0-3 but not yet executed. Without this, the paper cannot distinguish between two competing mechanisms:
1. AdamW's implicit ℓ∞ norm constraint causes Phi Invariance (any architecture with AdamW would show it)
2. Batch normalization's scale-invariance causes the observed invariance (BN-free networks would not show it)

This distinction is not cosmetic — it changes the paper's scope from "AdamW mechanism" to "AdamW+BN interaction mechanism."

**CWD-specific ablation gap**: CWDHardPhi uses the Adam first moment (exp_avg) as the "momentum direction" proxy. This means in AdamW, CWD applies weight decay only where sign(Adam_moment) == sign(parameter). But the paper's CWD formulation in theory uses gradient direction, not momentum. No ablation tests the sensitivity of CWD's performance to this implementation choice.

---

## 5. Reproducibility Assessment

**Score: 3.5/5**

Strengths:
- All random seeds fixed with PyTorch deterministic mode (4-component standard protocol)
- All hyperparameters passed via CLI args or config dict, none hard-coded
- Structured output: config.json + epoch_metrics.jsonl + summary.json + _DONE per run
- Unified Phi Modulator framework ensures single training loop for all methods

Weaknesses:
- **DataLoader workers**: `worker_init_fn` not set. If `num_workers > 0`, data loading order across seeds may not be exactly reproducible. Not confirmed whether num_workers=0 is enforced.
- **SWD dead code bug**: In `SWDPhi.get_per_param_wd()`, `global_per_elem` is computed from an empty generator `(p.numel() for group in [] for p in group['params'])` which always evaluates to 0. This is dead code that does not affect the actual output (the ratio uses `global_grad_norm` directly), but it signals incomplete implementation and undermines confidence in SWD's correctness.
- **Hardware specification**: RTX PRO 6000 Blackwell (98GB) is documented in the memory file but not in the experiment config/output. Different GPU hardware could affect floating-point precision and reproducibility.
- **Code version control**: No git commit hash is saved in config.json or summary.json. If the training code changes between runs, there is no way to trace which code version produced which result.

**Would a competent ML engineer reproduce within 10% of reported numbers?** Yes for the core accuracy metrics, with the caveat about DataLoader worker seeding.

---

## 6. Top-3 Recommendations (Ordered by Effort-to-Credibility Ratio)

### Recommendation 1: Add Matched-λ SGD Control (HIGH PRIORITY)

**What**: Run ResNet-20 CIFAR-10 with SGD, lr=0.1, wd=5e-4 (same as AdamW), multistep schedule, methods: constant/no_wd/cosine_schedule/cwd_hard, seeds 42/123/456.
**Why critical**: This is the single most credibility-threatening gap. The 18.3× effect ratio is the paper's empirical anchor. Without isolating λ from optimizer type, a skeptical reviewer will correctly note that the ratio could be partially explained by different wd magnitudes (ρ_SGD vs ρ_AdamW). With matched λ, you can report two ratios: "matched-λ ratio" (isolates optimizer) and "full-setting ratio" (matches published practice).
**What it would show**: If matched-λ SGD still shows large effect (Cohen's d > 5), the optimizer mechanism claim is vindicated. If the effect disappears, the 10× λ difference is the actual driver.
**Cost**: ~6 GPU hours, 12 runs. Extremely favorable effort-to-credibility ratio.

### Recommendation 2: Execute BN Ablation Before Claiming AdamW Mechanism (BLOCKING)

**What**: ResNet-20 without BN (replace BN layers with Identity), CIFAR-10, AdamW + SGD, constant/cosine_schedule/no_wd, 3 seeds each.
**Why critical**: The BN confound is rated 25-35% probability by multiple agents. If the ablation reveals BN is necessary for invariance, the theoretical framing (AdamW ℓ∞ constraint as the mechanism) requires complete revision. The proposed Theorem T1 assumes BN-free stability properties that must be empirically validated first.
**Decision rule**: If NoBN spread > 0.5% while BN spread < 0.5% → reframe as "BN+AdamW joint mechanism." If NoBN spread < 0.5% → AdamW mechanism is the dominant driver.
**Cost**: ~1 GPU hour, 18 runs. This is identified as P0-3 but not yet run. It is a hard blocker for theoretical claims.

### Recommendation 3: Execute λ Sweep Before Asserting Regime Boundaries (BLOCKING)

**What**: ResNet-20 CIFAR-10 AdamW, λ ∈ {5e-5, 5e-4, 5e-3, 5e-2}, constant/cosine_schedule/cwd_hard, 3 seeds each.
**Why critical**: The entire Trichotomy Theorem (ρ₁ < ρ < ρ₂ regime transitions) rests on this single experiment. Without it, the paper has exactly one data point in Regime I (λ=5e-4, ρ=0.5) and no data in Regimes II or III. A theory with one supporting data point is not empirically validated.
**Falsification criterion**: If spread < 0.5% persists at λ=5e-3 (ρ=5), the regime boundary ρ₂ > 5, and the theoretical prediction is falsified. This would require revising the Trichotomy boundaries.
**Cost**: ~2-3 GPU hours, 36 runs. Identified as P1-1 but critical enough to be P0.

---

## 7. Summary Risk Register

| Issue | Severity | Status | Threat to Main Conclusion? |
|---|---|---|---|
| Matched-λ SGD missing | HIGH | Not planned as P0 | Partially: 18.3× ratio attribution is ambiguous |
| BN ablation not run | HIGH | P0-3 (planned) | Fully: mechanism claim could flip |
| λ sweep not run | HIGH | P1-1 (planned) | Fully: Trichotomy has no direct evidence |
| n=3 for TOST | HIGH | Partially addressed (n=5 in plan) | Partially: TOST confidence intervals are wide |
| CIFAR-100 SGD no_wd n=1 | MEDIUM | Known, plan to fix | Partially: CIFAR-100 SGD results are unreliable |
| SWD dead code | LOW | Identified | No: SWD results are likely correct, just undocumented |
| AIS single-batch sampling | LOW | Not addressed | No: diagnostic metric, not primary outcome |
| Hyperparameter sensitivity | MEDIUM | Not addressed | Potentially: if invariance is hyperparameter-specific |
| CWD uses Adam moment not gradient | LOW | Not addressed | Minor: implementation detail, theory-practice gap |

---

*Methodologist Agent (sibyl-light) — Iteration 5 Result Debate*
*Focus: How experiments were conducted, not just what results they produced.*
