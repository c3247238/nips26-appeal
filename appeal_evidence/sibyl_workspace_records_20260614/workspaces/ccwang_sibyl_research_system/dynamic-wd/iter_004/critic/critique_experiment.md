# Experiment Critique

**Date**: 2026-03-18
**Critic**: sibyl-heavy (critic role)
**Iteration**: 7 (updated)

---

## Overall Assessment: Insufficient Experimental Coverage for Claimed Contributions

The executed experiments are technically sound and correctly reported. The problem is scope: the paper claims to evaluate "all major dynamic WD strategies spanning four modulation axes" and proposes a three-regime Trichotomy Conjecture, yet experiments cover only two of four axes and only one of three regimes. All four Gate experiments declared blocking before writing are unexecuted.

---

## Critical Issue 1: The Conjecture Is Tested Only Where It Must Hold

The Phi Invariance Trichotomy Conjecture posits Regime I (rho < ~1, strategy-invariant), Regime II (rho 1-10, transitional), Regime III (rho >> 1, strategy-sensitive). At the tested operating point rho=0.5, AdamW's sign normalization produces near-uniform coordinate update magnitudes regardless of WD schedule. The WD perturbation amounts to scalar multiplication of parameter values — when every gradient coordinate is sign-normalized to ~±1 per parameter, different WD schedules (temporal axis) or masked application (directional axis) barely change the effective update. The null result at rho=0.5 is expected from basic AdamW mechanics, not from a novel trichotomy.

**The conjecture becomes scientifically interesting only at Regime II (rho=5, lambda=5e-3) or III (rho=50, lambda=5e-2).** At rho=5, the WD step is 10x larger relative to the standard AdamW setting; AdamW's absorption may saturate; alignment-conditioned strategies may gain leverage. Regimes II and III have zero empirical support.

The proposal correctly identifies P1-1 (lambda sweep: rho in {0.05, 0.5, 5, 50}) as the "central falsification experiment" and explicitly marks it as P1 priority. It has not been run. The paper presents a conjecture with exactly one supporting data point.

**Verdict**: The lambda-sweep (P1-1) must run before submission. Even a single pilot at lambda=5e-3 (rho=5, ~3 extra seeds, ~2 GPU-hours) would test the most interesting regime transition prediction.

---

## Critical Issue 2: Four-Axis Coverage Is False

Table 1 catalogs methods along four axes. Tested: temporal (SWD, cosine_schedule), directional (CWD). Untested: spatial (AlphaDecay), target-norm (AdamWN). Both untested methods carry dagger symbols and appear in no experiment.

The paper abstract and contribution list (Section 1.4) state the benchmark covers "seven methods spanning all four modulation axes." Counting the actual coverage: random_mask (budget control, temporal-stochastic hybrid), half_lambda (budget control), and constant (baseline) are not additional axis representatives. The four-axis claim is not supported. The null result — "all strategies equivalent under AdamW" — explicitly cannot extend to untested axes. If AdamWN's per-parameter target-norm modulation breaks invariance at rho=0.5 (a plausible hypothesis: it applies zero WD to parameters already near the target norm), the conjecture is falsified by an already-existing method without running any new experiment.

**Verdict**: Add AdamWN to the benchmark (6 runs: 3 seeds x 2 datasets, ~2 GPU-hours). If AdamWN shows similar null result, four-axis coverage is legitimate. If AdamWN breaks invariance, a much more interesting result is discovered. Either outcome justifies the 2 GPU-hours.

---

## Critical Issue 3: All Gate Experiments Are Unexecuted

The proposal explicitly states Gates 0-3 must complete before any writing. Current status:

| Gate | Description | Status | Compute |
|------|-------------|--------|---------|
| Gate 0 | Theorem provability (A3 assumption check) | UNRUN | 2-3 h CPU |
| Gate 1 | BN ablation (ResNet-20-NoBN) | UNRUN | ~1 GPU-h |
| Gate 2 | VGG-16-BN full (72 runs, 200 ep) | UNRUN (pilot: 3 runs, 10 ep) | 6-8 GPU-h |
| Gate 3 | CIFAR-100 SGD no_wd seed_123 | UNRUN | 20 min |

**Gate 1 consequence**: The paper cannot distinguish AdamW's l_inf constraint from BN's scale-invariance as the driver of observed invariance. Section 7.3 acknowledges: "We cannot distinguish (a) AdamW's l_inf constraint, (b) BN scale-invariance, (c) both jointly." Every reviewer familiar with D'Angelo et al. (2024) will demand the NoBN ablation. Running 18 NoBN runs (~1 GPU-hour) resolves this.

**Gate 3 consequence**: CIFAR-100 SGD no_wd has n=1 (seed_42 only). No cross-dataset SGD analysis involving no_wd can be trusted. Gate 3 is 20 minutes of compute.

---

## Major Issue: Power Insufficient for Null-Result Claim

| Metric | Value |
|--------|-------|
| Seeds per condition | 3 |
| Within-method sigma | ~0.25-0.32% |
| MDE at 80% power | ~0.77% |
| TOST power at delta=0.5% | ~15-20% |
| CIFAR-100 range | 0.76% (at MDE boundary) |

The paper claims "statistically equivalent" throughout, but N=3 non-significant t-tests are weak evidence for the null. TOST at delta=0.5% with N=3 has ~15-20% power — meaning 80% of the time, a real 0.5% effect would be missed. The CIFAR-100 range of 0.76% is exactly at the detection boundary and cannot be interpreted.

P0-2 (2 additional seeds for key comparisons: ~8 GPU-hours for 28 extra runs) is specified in the proposal. It has not been run. TOST power at N=5 rises to ~55% for delta=0.5%. The paper should not claim equivalence without either N>=5 TOST or explicit acknowledgment that it can only claim "consistent with equivalence, not proven."

(Note: The paper does include this acknowledgment in Section 5.3 and 7.3, which is honest. But the executive summary and abstract use language closer to "equivalent" than "consistent with equivalence.")

---

## SGD swd Significance Inconsistency

The experiments.md section (Table 3, Section 6.2) states: "only one pairwise comparison achieves significance after Holm correction: constant vs. no_wd (p_adj=0.002)." SWD is listed at p_adj=0.054 (not significant).

The paper.md abstract and Section 6.2 state: "two of six comparisons achieve significance after Holm correction... constant vs. swd (d=3.48, p_adj=0.004)." This uses a 11-comparison cross-dataset Holm family.

Both analyses are technically valid but answer different questions. They must not both appear in the same paper. The conservative analysis (6 within-dataset comparisons, p_adj=0.054 for swd) is more defensible for a single-architecture claim. Choose one and apply consistently.

---

## Minor Issues

**BEM pipeline bug**: half_lambda BEM pipeline computes 0.000 but paper reports -0.500. A single code fix (phi=0.5 rather than lambda=2.5e-4 with phi=1) and 6 re-runs eliminates the manual correction footnote entirely.

**CIFAR-100 SGD full table absent**: Complete CIFAR-100 SGD data exists in workspace (7 methods x 3 seeds, confirmed) but no full table appears in the paper. The SWD result (p=0.036, delta=-1.07%) is the paper's second-strongest SGD finding and deserves full tabular documentation.

**WD Stability Condition (H2) untested**: Section 4.3 derives a warmup stability condition. Five warmup ablation runs (K in {1, 10, 50, 200, 1000}) would validate this prediction at ~2 GPU-hours. Currently the paper's only nontrivial theoretical prediction is unverified.

---

## Data Integrity Status: Verified

Contrary to a prior critique that alleged fabricated p-values, all SGD CIFAR-10 Table 3 values verify correctly against sgd_baseline_analysis.json:
- constant mean: 91.217% (JSON: 91.21666...) ✓
- no_wd mean: 90.303% (JSON: 90.30333...) ✓
- swd mean: 90.710% (JSON: 90.71) ✓
- SWD cohens_d: 3.478 (JSON: 3.4784...) ✓
- no_wd cohens_d: 10.29 (JSON: 10.2866...) ✓

The prior SWD significance retraction (from 2-seed estimate to 3-seed confirmed n=3 result) is correctly handled. SWD seed_456=90.93% is confirmed from disk.
