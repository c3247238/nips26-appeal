# Skeptic Analysis — Iteration 4 Result Debate

**Role**: Skeptic (maximum skepticism, statistical statistician)
**Date**: 2026-03-18
**Scope**: iter_003 full experiments (42 AdamW runs + 40 SGD baseline runs), iter_004 Phase 0 pilots (5 runs), proposal.md theoretical claims

---

## Overall Verdict: Grossly Over-Theorized from Critically Underpowered Data

The current dataset supports exactly one defensible empirical claim: "In the specific setting of ResNet-20 + AdamW + CIFAR-10/100 + λ=5e-4 + η=1e-3, accuracy differences across 7 WD methods fall within ±0.3% and are not statistically distinguishable with n=3." Everything else — the Phi Invariance Trichotomy Theorem, the ρ = λ/η order parameter, the three-layer novelty framework, the regime boundary predictions — is theoretical architecture erected on a n=3 empirical foundation that cannot bear the load.

**Confidence in publishability at NeurIPS/ICML level: 2.5/10**

---

## Issue 1: The 18.3× Ratio is a Statistically Meaningless Ratio

**Severity: FATAL FLAW**

The paper calls the "18.3× SGD/AdamW effect ratio" its "empirical anchor" (proposal.md line 24). This number is numerically accurate but statistically meaningless, and presenting it as an anchor is actively misleading.

**The denominator problem:**
- SGD constant vs no_wd Δ = 0.913% (robust signal)
- AdamW constant vs no_wd Δ = **0.050%** (n=3, p=0.825, completely buried in noise)
- Dividing a real signal by noise produces a noise-amplified ratio, not scientific evidence

**Bootstrap evidence (10,000 resamples with replacement, n=3):**
- 37.3% of bootstrap samples yield a **negative** ratio (sign flip of AdamW effect)
- 90th-percentile CI: [-12.2×, +36.8×]
- Mean bootstrap ratio: 10.9×, median: 3.3×
- The point estimate of 18.3× is near the 70th percentile of the bootstrap distribution — the value is not a stable estimate

**What this means:** The ratio can be anywhere from -60× to +320× depending on which 3 seeds happen to be drawn. Reporting 18.3× as "confirmed with p=0.0022, Cohen's d=12.17" is technically accurate for the SGD *numerator* but creates a false impression of precision for the *ratio itself*.

**Concrete remediation:** Report the SGD effect (Δ=0.913%, 95% BCa CI [0.73%, 1.09%]) without computing a ratio to an underpowered AdamW effect. The ratio should either be omitted or presented with explicit bootstrap uncertainty showing it spans multiple sign changes.

---

## Issue 2: n=3 Cannot Confirm the Equivalence Claim

**Severity: FATAL FLAW**

The paper's central claim is Phi Invariance — that AdamW WD methods are equivalent. But the statistical tests cannot confirm equivalence with n=3.

**Power analysis:**
- For Δ=0.5%, σ≈0.3% (typical inter-seed std), n=3 paired t-test power = ~35%
- Minimum detectable effect at 80% power, n=3: ±0.77%
- This means an AdamW effect as large as 0.76% would go undetected 80% of the time

**Absence of evidence ≠ evidence of absence:** The paper reports "all comparisons non-significant (p > 0.05)" and lists this under "Key Findings" as if confirming equivalence. This is a logical fallacy. The same n=3 data cannot reject either:
- H0: all methods are equivalent
- H_a: methods differ by up to 0.76%

**TOST (equivalence testing) requirement:** The proposal itself acknowledges (Part II, Score Trajectory) that TOST at δ=0.5% requires n≥5 for >60% power. The current n=3 has ~40% power for TOST at this δ. No equivalence claim can be made.

**CIFAR-100 AdamW situation is worse:** The 0.76% spread (63.42% cosine vs 62.66% no_wd) is substantial relative to typical τ≈0.35% std. The analysis.md dismisses this as "within overlap of confidence intervals" without actually computing the CI. Compute them:
- 95% CI for no_wd mean (62.66, σ=0.38, n=3): [61.93%, 63.39%]
- 95% CI for cosine_schedule mean (63.42, σ=0.42, n=3): [62.60%, 64.24%]
- These intervals barely overlap — the claim "within noise margin" is unsubstantiated

**Concrete remediation:** Increase n to ≥5 for all core AdamW comparisons. Run formal TOST equivalence tests. Do not use "non-significant at n=3" as a synonym for "equivalent."

---

## Issue 3: BEM=0.000 for half_lambda is a Confirmed Metric Bug with Cascading Effects

**Severity: SERIOUS CONCERN**

The BEM bug has been acknowledged in the proposal (P0-2), but its implications are not fully confronted.

**The bug:** HalfLambdaPhi uses λ_effective = λ_base/2 at every step. BEM should be:
BEM(half_lambda) = (Σ_t λ/2 − Σ_t λ) / (Σ_t λ) = −0.5

Instead, iter_003 reports BEM=0.000 for half_lambda on ALL 42 runs — the same value as the constant baseline. This is because the implementation does not report `mean_wd_actual` (confirmed by comparing iter_003 vs iter_004 pilot summaries: iter_004 correctly reports `mean_wd_actual`).

**Cascading effects:**
1. The analysis.md conclusion "BEM 0.0 to 1.0, accuracy differences < 0.5%" artificially maps half_lambda to BEM=0.0, making it appear as if the same WD budget as constant produces comparable results — when in fact it used only HALF the budget. The "10× budget variation" claim is inflated.
2. The ranked metric sanity table is wrong for any column involving half_lambda.
3. Any correlation analysis between BEM and accuracy is corrupted for this method.

**iter_004 pilot confirms the fix direction:** The half_lambda pilot (5 epochs) correctly shows `"mean_wd_actual": 0.00025`, `"final_bem": -0.5`. The fix works, but the iter_003 data is not automatically corrected.

**CSI>1.0 anomaly in VGG pilot:** The VGG-16-BN cwd_hard pilot shows `"final_csi": 1.011125` — CSI exceeds 1.0. If CSI is supposed to be a normalized stability index in [0,1], a value >1.0 indicates a normalization bug for VGG-scale weight norms (weight_norm=185 vs ResNet-20 weight_norm≈96). This suggests CSI is architecture-dependent in a way that makes cross-architecture comparison invalid.

**Concrete remediation:** Re-run BEM computation for all iter_003 half_lambda runs using logged weight decay values. Investigate and fix the CSI normalization to be architecture-invariant before reporting cross-architecture comparisons.

---

## Issue 4: Cohen's d = 12.17 Uses the Paired Formula — This is Non-Standard and Inflated

**Severity: SERIOUS CONCERN**

The proposal reports "Cohen's d=12.17" for SGD constant vs no_wd. This number uses the **paired** formula: d = mean_diff / std_of_differences.

**Recomputation from raw data:**
- Paired diffs: [91.3−90.39, 91.18−90.19, 91.17−90.33] = [0.91, 0.99, 0.84]
- std(diffs, ddof=1) = 0.075%
- d_paired = 0.913 / 0.075 = **12.17** ✓ — matches the claimed value

But:
- d_pooled (standard unpaired formula) = 0.913 / 0.089 = **10.3**
- d_pooled is the standard for between-group comparison

The paired formula is appropriate when the same seeds are directly comparable (which is the case here, since the same seed controls different aspects of training). However, presenting d=12.17 without clearly stating it's the paired formula creates an impression of an even larger effect. The proposal should specify which formula was used. More importantly:

**With n=3, the standard error of any Cohen's d estimate is enormous:**
- SE(d) ≈ d × √(2/n) ≈ 12.17 × 0.816 ≈ 9.9
- 95% CI for d=12.17 with n=3: approximately [2.7, 21.7]
- The uncertainty spans one order of magnitude

**Concrete remediation:** Report both paired and pooled Cohen's d with explicit 95% CIs. The SGD effect is real and large; it does not need to be inflated by formula choice.

---

## Issue 5: Architecture Monoculture Makes Generalization Claims Unsupported

**Severity: SERIOUS CONCERN**

The complete experimental evidence base is:
- 42 runs: ResNet-20, AdamW, CIFAR-10/100
- 40 runs: ResNet-20, SGD, CIFAR-10/100 (several incomplete — SGD CIFAR-100 no_wd has n=1)
- 5 runs: VGG-16-BN, 10 epochs, seed_42 only (3 methods, CIFAR-10 only)
- 2 runs: ResNet-20, 5 epochs, seed_42 only (2 methods, CIFAR-10 only)

**The VGG pilot result is anomalous and ignored:**
- VGG no_wd (80.61%) > VGG constant (79.94%) > VGG cwd_hard (80.30%)
- At 10 epochs, no_wd slightly *outperforms* constant by 0.67%
- This is the OPPOSITE direction from what Phi Invariance with ρ<1 predicts
- The proposal does not discuss this result at all — it was not mentioned in the experimental analysis section

This anomaly has three possible interpretations:
- (a) 10-epoch results are meaningless (task not converged) — then the pilot proves nothing
- (b) VGG early training dynamics differ fundamentally from ResNet-20 — undermines universality
- (c) VGG uses AdamW but BEM=-1.0 for no_wd was reported correctly (mean_wd_actual=0), suggesting the fixed BEM might suppress WD enough to benefit from training instability at epoch 10

**SGD CIFAR-100 no_wd has n=1 (0.64% below constant):**
- Proposal lists constant CIFAR-100 SGD mean as 65.37% ± 0.16% (n=3)
- no_wd CIFAR-100 SGD: 63.59% (n=1 only — seed_42)
- Any statistical comparison involving this point is based on a single run
- The proposal acknowledges this ("needs 2 more seeds") but still includes it in the analysis

**Concrete remediation:** Complete n=3 for SGD CIFAR-100 no_wd before including it in any analysis. Report the VGG pilot anomaly explicitly and discuss whether it challenges the ρ < 1 invariance claim.

---

## Issue 6: The ρ = λ/η Regime Boundary Has No Experimental Support in Regimes II or III

**Severity: SERIOUS CONCERN**

The Phi Invariance Trichotomy (proposed as Theorem T1) predicts three regimes based on ρ = λ/η:
- Regime I (ρ ≤ ρ₁, approximately ρ ≤ 0.5): Invariance holds
- Regime II (ρ₁ < ρ < ρ₂): Alignment matters
- Regime III (ρ ≥ ρ₂): WD dominates gradient

**The entire experimental dataset sits at a single point: ρ = 0.5 (λ=5e-4, η=1e-3).**

A three-regime theory derived from one operating point is not a theory — it is extrapolation. Specifically:
- ρ₁ and ρ₂ are defined without empirical grounding
- The prediction "spread 1-3% for ρ = 5" (λ=5e-3) has zero experimental support
- Predicting a phase boundary without observing anything above it is unfalsifiable until tested

**The proposal correctly identifies this as P1-1 (Lambda regime sweep, "the single most falsifiable experiment in the entire paper"). But writing the Trichotomy as a formal theorem before P1-1 is complete inverts the scientific method.** A falsifiable prediction has scientific value; a claimed "theorem" presented in a paper before the key falsification test is run is a theorem in name only.

**Alternative explanation:** The data at ρ=0.5 is consistent with a much simpler hypothesis: "With AdamW, any WD value from 0 to 2× base produces similar accuracy on CIFAR-scale tasks." No regime boundary, no order parameter, just a flat accuracy landscape for these tasks/scales.

**Concrete remediation:** P1-1 (lambda sweep) must be completed before Theorem T1 can be presented as anything stronger than a conjecture. Phrase it as "Conjecture C1" until the boundary is empirically observed.

---

## Issue 7: BN Confound vs. AdamW Mechanism — Cannot Distinguish Without NoBN Data

**Severity: SERIOUS CONCERN**

Six independent agents (Optimist, Contrarian, Pragmatist, Empiricist, Innovator, Interdisciplinary) and the Codex reviewer all flagged the BN confound. The proposal acknowledges it requires P0-3 (NoBN ablation). The experiment has not been run.

**Why this matters structurally:**
- D'Angelo (2024) already showed BN scale-invariance causes WD ineffectiveness
- If BN is the mechanism: the Phi Invariance is a restatement of D'Angelo with more complicated notation
- If AdamW is the mechanism (ℓ∞ implicit constraint): the novelty is real
- Currently we cannot distinguish between these two explanations

**The existing data has a confound that has been identified but not resolved.** The paper cannot be written without this answer.

**Secondary concern — CSI measures BN-layer dynamics incorrectly:**
CSI is defined as a coupling stability between weight norm and gradient norm. In BN networks, BN layers are scale-invariant — their weight norm is meaningless as a regularization metric. The CSI values for BN layers are measuring an invariant quantity. If CSI is high because BN normalizes everything regardless of WD, CSI is not a useful diagnostic.

---

## Issue 8: CWD Phi Uses g_t vs u_t — AIS Measures the Wrong Quantity

**Severity: MINOR CAVEAT (with potential to escalate)**

The proposal notes (Evolution Lessons): "CWD phi formula uses u_t (preconditioned update) but framework signature takes g_t (raw gradient). The distinction matters but is never formally resolved."

**Consequence:** AIS (Alignment Informativeness Score) measures cos_sim(w_t, g_t). CWD's modulation is based on cos_sim(w_t, u_t) where u_t = m_t / (√v_t + ε).

For AdamW with Adam's preconditioner:
- u_t and g_t have the same sign pattern only when Adam's variance estimate is uniform
- In BN layers where gradients have highly non-uniform second moments, u_t direction can differ from g_t direction

This means:
- AIS values are measuring g_t-alignment but CWD responds to u_t-alignment
- The claim "alignment signal is moderate (AIS ≈ 0.28-0.41)" could reflect g_t-alignment, not the alignment signal CWD actually uses

This is a metric definition inconsistency that needs to be resolved before AIS can be used as evidence for or against CWD's mechanism.

---

## Severity Classification Summary

| Issue | Severity | Fix Required Before... |
|-------|----------|----------------------|
| 18.3× ratio statistically unstable (bootstrap spans sign changes) | **Fatal Flaw** | Any ratio claim in writing |
| n=3 cannot confirm equivalence (underpowered for TOST) | **Fatal Flaw** | Any "invariance confirmed" claim |
| BEM bug corrupts half_lambda data in all 42 iter_003 runs | **Serious Concern** | Any BEM-based analysis in writing |
| CSI>1.0 for VGG — normalization is architecture-dependent | **Serious Concern** | Cross-architecture CSI comparisons |
| Cohen's d=12.17 uses non-standard paired formula, SE spans order of magnitude | **Serious Concern** | Reporting effect sizes in writing |
| VGG pilot no_wd>constant anomaly unexplained | **Serious Concern** | Any VGG universality claim |
| ρ regime II/III have zero experimental data | **Serious Concern** | Writing Theorem T1 as "theorem" |
| BN confound vs AdamW mechanism unresolved | **Serious Concern** | Any mechanistic claim in abstract |
| CWD uses u_t but AIS measures g_t alignment | **Minor Caveat** | Any AIS-based CWD analysis |

---

## Concrete Remediation Experiments

1. **Bootstrap CI for 18.3× ratio (zero GPU, 2 hours):** Run 10,000-resample BCa bootstrap on the ratio SGD_Δ/AdamW_Δ. Report median and 5th-95th percentile. If CI spans sign changes, remove the ratio from the paper and describe the effects separately.

2. **TOST equivalence testing at n=5 (3-4 GPU hours, 14 runs):** Add seeds 789 and 999 for constant and no_wd AdamW on CIFAR-10. Run TOST at δ=0.5%, α=0.05. Only after this passes can "invariance" be stated.

3. **SGD CIFAR-100 no_wd completion (1 GPU hour, 2 runs):** Seeds 123 and 456 for SGD CIFAR-100 no_wd. Required before any SGD CIFAR-100 statistical claims.

4. **ResNet-20-NoBN ablation (1 GPU hour, 18 runs, already in P0-3):** This is the highest-priority unresolved scientific question. Without it, the mechanism chapter cannot be written.

5. **Lambda regime sweep (2-3 GPU hours, P1-1):** Test ρ ∈ {0.05, 0.5, 5, 50} (λ ∈ {5e-5, 5e-4, 5e-3, 5e-2}). If no inflection is observed near ρ≈1, the Trichotomy has failed its key falsification test.

6. **BEM recomputation for iter_003 half_lambda (zero GPU, 1 hour):** Use logged `mean_wd_actual` from checkpoints or re-derive from schedule formula. Update all tables and analyses that reference half_lambda BEM.

---

*This analysis is deliberately adversarial. The SGD effect (Δ=0.913%, p=0.0022) is real, large, and interesting. The AdamW invariance (at the single tested operating point) is a genuine observation. The problems are in the extrapolation, the statistical precision of the effect ratio, and the gap between "observed in one setting" and "theorem holding universally."*
