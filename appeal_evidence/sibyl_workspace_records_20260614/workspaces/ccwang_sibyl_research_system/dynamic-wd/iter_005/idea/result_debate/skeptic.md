# Skeptic Analysis: Iteration 7 Experiment Results

**Agent**: sibyl-skeptic
**Date**: 2026-03-18
**Iteration**: 7 (result debate for Iter 7 proposal)
**Scope**: Statistical validity, alternative explanations, proxy metric audit, and remediation for the "Stability-Optimal Control Theory of Dynamic Weight Decay" paper.

---

## 1. Statistical Risk Inventory

### Risk 1: VGG Phi Spread is Statistically Undetectable at N=3 — The "Null Result" May Simply Be Noise

The VGG-16-BN phi spread across 4 available methods is **0.157%** (constant=92.05%, cwd_hard=92.06%, half_lambda=92.15%, cosine_schedule=91.99%). The proposal describes this as "multi-architecture confirmation of WD method insensitivity."

**Why this is unreliable**: The within-method standard deviation for cwd_hard is **0.209%** and for cosine_schedule is **0.263%** — *both larger than the phi spread itself*. A power analysis confirms the study is severely underpowered:

- Cohen's d = phi/std = 0.157/0.2 = 0.785
- t_observed = 0.785 × √(3/2) = 0.96
- t_critical (α=0.05, df=4) = 2.78
- **Power < 20%** to detect phi=0.157% at the given std

The best-vs-worst comparison (half_lambda 92.147% vs cosine_schedule 91.990%) produces p=0.48 by two-sample t-test — entirely non-significant. TOST equivalence testing passes at delta=1% (upper 95% CI bound = 0.478% < 1%), meaning the methods are statistically equivalent within a 1% tolerance. However, this means only: "we cannot rule out that the true difference is less than 1%." The CI spans [-0.16%, +0.48%]; the true phi could be anywhere in this range.

**Specific concern**: The 95% CI upper bound is 0.478% — barely passing the 0.5% threshold cited in the proposal, with the true difference potentially as high as 0.478%. Three more seeds or three more methods could easily push the point estimate in either direction.

### Risk 2: VGG Experiment Coverage is 4/7 Methods — The 3 Missing Methods are the Highest-Risk Ones

The 4 completed VGG methods are constant, cwd_hard, half_lambda, cosine_schedule — all smooth-decay or moderate-masking methods. The **3 missing methods are no_wd, swd, and random_mask**, precisely the ones expected to diverge most from constant:

- **no_wd**: On SGD CIFAR-10, no_wd achieves mean 90.30% (vs constant 91.22%), a gap of 0.91%. On CIFAR-100 SGD (per Iter 3 data), no_wd gap is 1.29%. If no_wd on VGG follows even a fraction of this pattern (e.g., 0.5-0.8% gap), the full VGG phi spread could exceed 0.5%, directly contradicting the null result claim.
- **random_mask**: On SGD CIFAR-10, random_mask within-seed range is 0.83% (largest of all methods). On VGG, stochastic masking interacting with BN scale invariance could amplify this.
- **swd**: On SGD, swd achieves 90.71% mean vs constant 91.22% — a gap of 0.51% even in the SGD regime. VGG is a more complex architecture with more diverse layer sensitivities, potentially amplifying the swd gap.

**The 0.157% phi spread is computed from the 4 MOST CONSERVATIVE methods. The claim "VGG null result confirmed" is an artifact of selection bias in experiment completion order.**

### Risk 3: Matched-rho SGD Phi Spread is Uncomputed — The Paper's #1 Confound Remains Unresolved

Matched-rho SGD results show only the **constant method** with 3 seeds (seeds 123 and 456 complete at 90.88%/90.89%; seed 42 ran only 5 epochs and shows 76.12%). Zero comparison methods exist. The phi spread for matched-rho SGD is undefined.

**Why this critically matters**: The entire narrative rests on: "SGD shows 0.91% phi spread vs AdamW's 0.25% because SGD uses rho~0.005 vs AdamW rho~0.5 (100× difference)." Without matched-rho SGD having multiple WD methods, this hypothesis is **untestable from current data**. The experiment produces a stable baseline accuracy (90.88%) but zero evidence about method sensitivity. The proposal labels this "Gate 3" as testable — it is not.

**Specific numbers**: SGD phi = 0.91%, AdamW phi = 0.25%, ratio = 3.65×. Without matched-rho phi, the rho hypothesis is pure conjecture. The evolution lesson identifying this as "the paper's #1 weakness" remains accurate.

---

## 2. Alternative Explanations

### Alternative 1: The VGG "Null" Reflects AdamW Optimizer Property, Not BN Scale Invariance

The proposal attributes VGG WD insensitivity to BN scale invariance + low rho. But AdamW's ℓ∞ normalization already makes the effective WD relatively schedule-invariant regardless of BN — the adaptive learning rate absorbs the variance in WD schedules. VGG-16-BN uses AdamW with lr=0.001, same as ResNet-20.

**Critical test**: ResNet-20 AdamW CIFAR-10 phi = 0.25%, ResNet-20 SGD CIFAR-10 phi = 0.91%. Both use the same BN architecture; the difference is the optimizer. If BN scale invariance were the mechanism, both should show similar phi. The 3.65× difference implicates the optimizer, not BN, as the primary mechanism. **The VGG null result may confirm AdamW's schedule-robustness, not BN's role.**

### Alternative 2: The SGD Phi Spread is Inflated by Two Outlier Seeds

SGD CIFAR-10 phi = 0.91% (constant 91.22% vs no_wd 90.30%). But:
- cwd_hard seed_456 = 90.38% vs seed_42 = 91.20%: within-method range of **0.82%**
- random_mask seed_456 = 91.29% vs seed_42 = 90.46%: within-method range of **0.83%**

These two methods account for most of the phi spread. If these high-variance seeds reflect training stochasticity rather than method effects (e.g., lucky/unlucky initialization), the true SGD phi could be substantially lower. The proposal claims SGD phi is "rock-solid with N=3×7 methods" but two of seven methods have within-seed variance comparable to the entire between-method spread.

### Alternative 3: The Null Result at CIFAR-10 is a Task-Difficulty Artifact

ResNet-20 on CIFAR-10 achieves 90-91% accuracy with even no_wd (90.30%). CIFAR-10 is nearly saturated for this architecture; regularization is less critical because the model is capacity-limited, not overfitting-limited. The 3× amplification of phi_spread from CIFAR-10 (0.25%) to CIFAR-100 (0.75%) suggests that harder tasks reveal WD sensitivity. Without ImageNet data, the conclusion may be: "WD method invariance holds on easy benchmarks only."

---

## 3. Proxy Metric Audit

### BEM (Budget Equivalence Metric) — Computationally Inconsistent Across Methods and Experiments

**Critical bug confirmed by code audit (iter_005/exp/code/optimizers.py)**:

BEM is computed as `(mean_wd_actual - wd_nominal) / wd_nominal`. The `mean_wd_actual` comes from `effective_wd` in optimizer diagnostics, which is derived differently per method:

| Method | effective_wd source | BEM correctness |
|--------|--------------------|-----------------|
| constant | `wd_schedule = base_wd` | Correct (BEM=0.0) |
| half_lambda | `wd_schedule = base_wd * 0.5` | Correct for AdamW (BEM=-0.5) |
| cosine_schedule | `wd_schedule = wd(t)` | Correct (BEM=-0.5025) |
| cwd_hard | `mask_ratio` (last param) | **Biased**: single-parameter sample |
| swd | `wd_multiplier` (last param) | **Severely biased**: single-parameter sample |
| random_mask | `mask_ratio` (last param) | **Biased**: single-parameter sample |

**SGD logging bug**: SGD half_lambda shows BEM=0.0 (no `mean_wd_actual` field), but weight_norm is 83.45 vs constant 64.49 (52% higher) — confirming WD IS halved. BEM=0.0 is a logging failure in the SGD code path that does not exist in the AdamW path.

**SWD BEM = 0.9 is physically impossible**: If BEM=0.9 means mean_wd_actual = 0.0005×1.9 = 0.00095, SWD is applying 90% MORE WD than nominal. Yet SWD's weight_norm = 104.4 >> constant = 64.49, confirming LESS regularization. The BEM sign is wrong, and the magnitude is a sampling artifact of `wd_multiplier` from a single high-norm parameter.

**Impact on paper**: The prior evolution lesson "BEM = 0.000 for half_lambda is mathematically impossible" is partially resolved for AdamW (BEM=-0.5 is correct) but NOT for SGD (still BEM=0.0). Any paper figure or table comparing BEM across methods or across SGD/AdamW experiments is unreliable.

### AIS (Alignment Informativeness Score) — No Demonstrated Predictive Power

AIS is presented as the trigger for Theorem 1's regime ("CWD helps iff AIS > threshold"). Examining the actual AIS values:

| Experiment | Method | AIS range across seeds | Std |
|---|---|---|---|
| VGG constant | — | 0.175–0.242 | 0.034 |
| VGG cwd_hard | — | 0.229–0.364 | 0.069 |
| NoBN constant | — | 0.370–0.592 | 0.110 |

The within-method AIS seed variance for VGG cwd_hard (std=0.069) is larger than the between-method AIS difference (cwd_hard mean=0.311 vs constant mean=0.217 = 0.094 difference). The NoBN AIS variance is striking: 0.370 to 0.592, a range of 0.222% for the same method and architecture.

**No demonstration that AIS correlates with performance**: No correlation coefficient, no regression, no ANOVA has been reported. AIS is measured but not validated as a predictor. Theorem 1 states "CWD outperforms constant WD iff AIS > threshold" — this iff is not tested with the current data.

### CSI (Coupling Stability Index) — Theorem 2 Bound Not Computed

Theorem 2 states: `GenGap({λ_{i,t}}) − GenGap(λ̄) ≤ (2Lσ²/n) × CSI_param × T`. CSI values are recorded in all summary.json files (range 0.43–2.06 across all experiments). But:

- The RHS bound has never been computed with actual values of L, σ², n, T
- No GenGap vs CSI scatter plot exists
- No Pearson r between CSI and gen_gap has been reported

VGG cosine_schedule seed_42 has CSI=2.06 and gen_gap=8.43 (highest for VGG — consistent). But VGG constant seed_123 has CSI=1.91 (second highest) and gen_gap=8.04 (only slightly lower). These two data points suggest the CSI-GenGap relationship is noisy or confounded by other factors.

**Theorem 2 is currently a mathematical statement with no empirical validation.**

---

## 4. Severity Classification

### Fatal Flaws

**F1: VGG phi spread from 4/7 methods with systematic selection bias — "multi-architecture null" claim is premature.**
- The 3 missing methods (no_wd, swd, random_mask) are the highest-divergence methods in every other experiment. Their VGG results could push phi to 0.5-1.5%, directly refuting the null claim.
- Reporting "VGG null result confirmed" before all 7 methods are complete is a critical validity threat.
- Must complete the full 7-method VGG experiment before any multi-architecture claim.

**F2: Matched-rho SGD has zero comparison methods — the rho confound hypothesis is untestable.**
- The paper's self-identified "#1 weakness" (100× rho confound) remains unaddressed by data.
- A baseline accuracy of 90.88% with only the constant method tells us only that matched-rho SGD is stable. It tells us nothing about method sensitivity, which is the entire point.
- The paper cannot claim "rho explains the SGD/AdamW sensitivity ratio" with current data.

### Serious Concerns

**S1: BEM computation is buggy for SWD (all experiments) and half_lambda (SGD experiments).**
- SGD half_lambda BEM=0.0 is a logging failure; true value should be ~-0.5.
- SWD BEM reflects a single-parameter sample, not the true mean per-step WD.
- Any paper claim involving BEM as a diagnostic or predictor metric is unreliable for these methods.

**S2: rho_high (rho=5.0) has zero completed runs.**
- Theorem 1 Corollary ("higher rho → higher WD sensitivity") is entirely a theoretical prediction with no empirical support.
- The "rho-regime map" (Fig 1 in the proposal) cannot be constructed without rho=5.0 data.
- The Gate 2 decision tree is hypothetical and cannot yet be evaluated.

**S3: VGG seed variance (std up to 0.26%) exceeds phi spread (0.16%) — study is underpowered for the claimed effect size.**
- The null result cannot be distinguished from noise at N=3 with this effect size.
- TOST passes at delta=1% but barely. Equivalence within 0.5% cannot be established.

**S4: NoBN experiment has 1 WD method — BN-masking hypothesis is untested.**
- NoBN constant shows higher AIS (0.49 vs BN 0.34), consistent with theory, but phi spread requires multiple methods.
- The NoBN ablation also has an lr confound (lr=0.5e-3 vs BN lr=1e-3), which changes rho independently of BN removal.

### Minor Caveats

**M1: Proposition 1 (alignment noise at batch≤256) is cited from SimiGrad (NeurIPS 2021), not measured on our architecture.**
The CV>>1 claim is plausible but not directly verified on ResNet-20/VGG-16-BN with our specific training configuration. It is an extrapolated design principle, not a measured property of our system.

**M2: The PMP + RG "dual derivation" may not be as independent as claimed.**
Both assume linearized ρ-dynamics near steady state. The shared linearization assumption means they are two algebraic routes to the same approximation, not two independent theories arriving at the same conclusion.

**M3: cwd_hard seed_456 on SGD (90.38%) may be a training instability, not a method effect.**
Its within-method range of 0.82% is anomalously large. If excluded as an outlier, SGD cwd_hard mean rises from 90.87% to 91.11%, and the SGD phi spread drops from 0.91% to ~0.65%.

---

## 5. Concrete Remediation

### For F1 (VGG missing methods):
**Experiment**: Run swd, no_wd, random_mask on VGG-16-BN/CIFAR-10 × 3 seeds each (9 runs × ~13 min = 2 GPU-hours, parallelizable on 3 GPUs).
**Acceptance criterion**: Full 7-method VGG phi spread. If phi < 0.5%: null result confirmed with complete evidence. If no_wd diverges by >0.5%: null result is refuted — a new positive finding requiring reframing (architecture matters for no_wd, not for alignment-aware methods).
**Do NOT write the multi-architecture section until this completes.**

### For F2 (Matched-rho SGD):
**Experiment**: Run cwd_hard and no_wd at matched-rho SGD (lr=0.01, wd=0.005) × 3 seeds each (6 runs × ~70 min = 7 GPU-hours). Also rerun constant seed_42 at 200 epochs (current run stopped at epoch 5).
**Key gate**: phi_spread across {constant, cwd_hard, no_wd} at matched rho.
- phi < 0.25%: rho fully explains SGD/AdamW ratio → confirms Theorem 1 regime boundary
- phi > 0.5%: optimizer mechanism is additionally important → requires narrowing Theorem 1 scope

### For S1 (BEM bug):
**Code fix**: Add per-step mean WD accumulation to `train_sgd.py` (same as AdamW path). For SWD and random_mask, accumulate `mean(wd_tensor)` over all parameters rather than using the last-evaluated parameter's multiplier. Expected: ~2 hours of code change, then re-run 3 SGD experiments to verify.
**Minimum viable workaround**: Exclude SWD and half_lambda (SGD experiments only) from any BEM-based comparison table. Add footnote explaining the logging inconsistency.

### For S2 (rho_high missing):
**Experiment**: Diagnose why rho_high previously failed. Run a single-seed pilot at rho=2.0 (wd=0.002) before attempting rho=5.0, to verify numerical stability. Then run {constant, cwd_hard, no_wd} × 3 seeds at rho=5.0 if stable.
**Without this data**: Label Theorem 1 Corollary explicitly as "Conjecture 1 (empirically untested, predicted from theory)" rather than presenting it alongside confirmed theorems.

### For S3 (VGG statistical power):
**Analysis**: Report TOST results at both delta=1.0% AND delta=0.5% for all VGG method pairs. If delta=0.5% TOST fails, acknowledge in the paper: "VGG methods are statistically equivalent within 1% but equivalence within 0.5% is not established at N=3."
**Optional experiment**: Increase VGG seeds to N=5 (2 additional seeds per method × 4 complete methods = 8 additional runs, ~1.5 GPU-hours) to reduce SEM and tighten the CI.

---

## Summary Verdict

**What is defensible with current data:**
1. AdamW + ResNet-20 + CIFAR-10/100: phi < 0.3% across 7 methods × 3 seeds (N=42 per condition). This is the paper's unambiguous empirical foundation — solid, reproducible, well-powered.
2. SGD + ResNet-20 + CIFAR-10/100: phi < 1.0% across 7 methods × 3 seeds. High within-seed variance for 2 methods should be acknowledged.
3. VGG-16-BN CIFAR-10: phi = 0.16% across 4/7 methods × 3 seeds. Statistically equivalent within 1% (TOST). Not a null result claim — a partial result pending 3 critical missing methods.

**What is NOT defensible with current data:**
- "Multi-architecture null result confirmed" (VGG: 4/7 methods, missing the 3 highest-risk methods)
- "rho explains the SGD/AdamW sensitivity ratio" (matched-rho SGD has 1 method; phi undefined)
- "Theorem 1 Corollary: high rho → high sensitivity" (rho_high: zero data)
- "BN masking mechanism confirmed" (NoBN: 1 method, lr confounded)
- BEM as a predictor of WD performance (BEM is buggy for SWD and SGD half_lambda)

**Priority verdict**: The proposal's experimental priority ranking is correct — VGG parallelization (all 7 methods) and matched-rho SGD (cwd_hard + no_wd) are the two highest-value actions. Without them, the paper's multi-architecture and confound-resolution claims are premature. The BEM bug is a credibility risk that should be fixed regardless of experimental priority.
