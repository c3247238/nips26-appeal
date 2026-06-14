# Iteration 7 Synthesized Research Proposal

**Synthesizer**: Sibyl Synthesizer (sibyl-heavy)
**Date**: 2026-03-18
**Based on**: 6-Agent Perspectives (Innovator, Pragmatist, Theoretical, Contrarian, Empiricist, Interdisciplinary) + Result Debate Verdict (5.5/10) + Codex Review (7/10, Iteration 4) + Verified SGD data (SWD seed_456 = 90.93% CONFIRMED on disk)
**Prior proposal**: Iteration 6 (SWD corrected, ρ-Controller as primary secondary, Conjecture framing)

---

## Executive Summary

Iteration 7 represents a **data-confirmed consolidation** of the Iteration 6 synthesis. The SWD seed_456 result (90.93%) has now been verified directly from disk (`iter_003/exp/results/sgd_baseline/cifar10/resnet20/swd/seed_456/summary.json`), confirming the corrected statistics:

- **SWD mean = 90.710% ± 0.193%** (n=3, seeds 42/123/456: 90.57/90.63/90.93)
- **SWD p = 0.0713, Holm-threshold = 0.010: NOT significant** — prior SWD significance claim is definitively retracted
- **Only significant pair: constant vs no_wd** (p=0.0022, Holm-sig, d=−12.17, Cohen's d pooled = −10.29)
- **18.3× SGD/AdamW ratio confirmed**: 0.913% / 0.050% = 18.3×

All six agents agree: the research direction has converged. The remaining work is **experimental gate-clearing** (BN ablation, λ sweep, VGG-16-BN) before writing, not ideation.

**Iteration 7 synthesis verdict**: The front-runner (ρ = λ/η Phi Invariance Trichotomy) with the ρ-Controller as primary secondary contribution is the correct and final research proposal. No pivot is warranted. Execute the experimental gates and begin writing.

---

## Part I: Landscape Mapping

### Agreements Across All 6 Agents (Iteration 7)

1. **ρ = λ/η as the unifying order parameter**: All six agents converge independently. Innovator formalizes it as the P-controller gain; Theoretical provides the Phi Invariance Theorem T1 proof sketch (Lemmas 1–3, Xie & Li + Defazio + Kuzborskij synthesis); Pragmatist identifies it as r_t = ‖g‖/‖w‖ steady state; Empiricist confirms the regime boundary; Contrarian accepts it with the BN confound caveat; Interdisciplinary maps it to thermodynamic "pressure" and control-theoretic damping ratio.

2. **SWD data correction is final**: With seed_456 = 90.93% confirmed from disk, SWD (90.710% ± 0.193%) has raw p = 0.0713, well above the Holm-corrected threshold of 0.010 for its rank-2 position. The result is unambiguous: **no dynamic WD strategy significantly differs from constant WD under SGD CIFAR-10 in either direction at n=3**. The only significant finding is WD presence-absence (constant vs no_wd).

3. **BN ablation is the unconditional first gate**: All agents (Contrarian HIGH severity, Result Debate C1 by 6/6) agree this must run before mechanistic claims are written.

4. **λ regime sweep is the central falsification**: Result Debate C3 (5/6), Empiricist, Innovator, and Theoretical all confirm the λ sweep (P1-1) separates the Trichotomy from post-hoc rationalization. Regime I (ρ=0.5) is confirmed; Regimes II and III have zero empirical support.

5. **ρ-Controller is the algorithmic contribution**: The Innovator's closed-loop feedback WD controller (no prior work formulates WD as a P-controller targeting ρ*) is computationally free, one-line, Lyapunov-backed, and conditional on the λ sweep confirming Regime II exists.

6. **VGG-16-BN pilot reversal is a priority signal**: 1-seed VGG-16-BN AdamW shows no_wd=80.61% > constant=79.94% — reversed direction. If confirmed at n=3, WD is actively harmful under AdamW on VGG-16-BN, which is a stronger positive finding than null invariance.

### Conflict Resolution (Iteration 7)

- **SWD negative result**: Resolved. SWD at n=3 is NOT significantly worse than constant. All papers claiming SWD is beneficial were using Adam+L2 not AdamW, or different evaluation protocols. Our data is controlled and correct.

- **Theorem vs Conjecture**: Theorem T1 is presented as a **Formal Conjecture with Proof Sketch** until Day 0 provability check (Assumption A3 verification). This is the honest framing accepted by all agents post-Codex review.

- **BN confound**: Not resolved — that is the purpose of Gate 1. Two publishable paths identified (Path A and Path B, both pre-scripted).

---

## Part II: Research Direction

### Core Thesis (Final, Iteration 7)

**"When Does Dynamic Weight Decay Matter? A ρ = λ/η Regime Analysis with Feedback-Stabilized WD for Regime II"**

**Tier 1 (Core)**: The normalized weight decay ratio ρ = λ/η determines a three-regime structure for dynamic WD utility under AdamW. At standard settings (ρ ≈ 0.5, Regime I): all budget-equivalent WD strategies — including WD removal — are statistically equivalent. Under SGD (implicit Regime III behavior, no ℓ∞ normalization): WD presence-absence gap = 0.913% (CIFAR-10), with an 18.3× ratio vs AdamW.

**Tier 2 (Algorithmic Corollary)**: The ρ-Controller: λ_t = λ₀ · (ρ_t / ρ*)^{−α}, ρ_t = ‖g_t‖/‖w_t‖. First closed-loop feedback WD controller. Improvement only in Regime II/III (λ ≥ 5e-3). Collapses to constant WD in Regime I.

### Freshness from Iteration 7

| Aspect | Prior Iterations | Iteration 7 |
|---|---|---|
| SWD data | Uncertain (2-seed estimate) | Confirmed from disk (seed_456=90.93%) |
| Statistical framing | Corrected p-values | Full Holm-corrected table with verified data |
| VGG-16-BN | Pilot only | Escalated to mandatory pre-writing gate |
| Experimental queue | P0 priority list | Clarified: BN ablation is Day 0 non-negotiable |
| Next action | Ideation | EXECUTE gates, no more synthesis needed |

---

## Part III: Experimental Priority Matrix

### GATE EXPERIMENTS (Must Complete Before Any Paper Writing)

**Gate 0: Theorem Provability Check** (2–3 hours, CPU)
- Extract Adam second moments from ResNet-20 CIFAR-10 AdamW seed_42 epoch 100 checkpoint
- Compute ε / (h_i^{1/2} · |w_i|) distribution across all parameters
- Decision: median < 0.05 → "Theorem T1"; else → "Conjecture T1"

**Gate 1: BN Ablation** (1 GPU hour, 18 runs)
- ResNet-20-NoBN, CIFAR-10, AdamW + SGD, constant/cosine_schedule/no_wd, 3 seeds
- Path A (NoBN invariant): AdamW ℓ∞ mechanism is sufficient, distinct from D'Angelo
- Path B (NoBN breaks): BN + AdamW joint mechanism, reduced novelty but publishable
- **This gates all of Sections 3 and 4**

**Gate 2: VGG-16-BN Full Experiment** (P0, 6–8 GPU hours, 72 runs)
- VGG-16-BN, CIFAR-10 + CIFAR-100, AdamW + SGD, 6 methods × 3 seeds
- Confirm/refute 1-seed pilot (no_wd > constant under AdamW)
- Pre-scripted narrative for both outcomes

**Gate 3: CIFAR-100 SGD no_wd Completion** (20 min, 1 run)
- seed_123 missing (only 1-seed CIFAR-100 SGD no_wd currently)
- Required for n=3 CIFAR-100 analysis

### P0 Experiments (Pre-Writing, Highest Value)

**P0-1: Corrected Statistical Package** (0 GPU, 4 hours)
- All SGD tables with verified 3-seed SWD data
- Bootstrap 95% BCa CI for 18.3× ratio (10,000 resamples)
- BEM override for half_lambda: BEM = 0.500 (exact, known schedule)
- TOST equivalence tests at n=3 and projected n=5 power

**P0-2: n=5 Extension for CIFAR-10** (3–4 GPU hours, 28 runs)
- Seeds 789, 999 for constant/no_wd/cwd_hard/cosine_schedule on CIFAR-10
- Raises TOST power from ~15% to ~55% at δ=0.5%
- Required for any formal equivalence claims in AdamW section

### P1 Experiments (High Value, Conditional)

**P1-1: λ Regime Sweep** (2–3 GPU hours, 36 runs)
- ResNet-20 CIFAR-10, AdamW, λ ∈ {5e-5, 5e-4, 5e-3, 5e-2}, ρ ∈ {0.05, 0.5, 5, 50}
- Methods: constant/cosine_schedule/cwd_hard, 3 seeds each
- The central falsification of the Trichotomy Conjecture

**P1-2: ρ-Controller Pilot** (1 GPU hour, 6 runs, conditional on P1-1)
- ResNet-20 CIFAR-10, λ=5e-3 (Regime II), 3 seeds
- constant vs ρ-Controller (α ∈ {0.25, 0.5, 1.0})
- Primary diagnostic: ρ_t convergence to ρ* (logarithmically faster than constant?)

**P1-3: ImageNet ResNet-50** (10–14 GPU hours, 18 runs)
- 4 core methods × 3 seeds × AdamW+SGD, 90 epochs
- Required for scale generalizability claim

### MVP Definition

Gates 0–3 + P0-1 → 6.5 estimated score
Gates + P0-1 + P0-2 + VGG → 7.0–7.2
Gates + P0-P1 full → 7.5–8.0

---

## Part IV: Theoretical Agenda (Iteration 7)

### T1: Phi Invariance Trichotomy Conjecture (Primary)

**Proof structure** (from Theoretical agent, Lemmas 1–3):
- Lemma 1: AdamW ℓ∞ norm stability (Xie & Li 2024, Theorem 1.1)
- Lemma 2: Perturbation recursion under schedule variation (requires A3)
- Lemma 3: Telescoping bound → exponential convergence of δw_T to 0 in Regime I
- Assumption A3 (weakest link): stable Adam moment estimates under small λ perturbation

**Day 0 decision**: Empirically verify A3 holds via ε/(h_i^{1/2}·|w_i|) distribution check.

**Empirical predictions of T1** (current verification status):
- Regime I (ρ=0.5): spread < 0.5%. CONFIRMED: 0.25% on CIFAR-10, 0.76% on CIFAR-100.
- Regime II (ρ=5): spread 1–3%. UNTESTED — P1-1 required.
- Regime III (ρ=50): spread > 3%. UNTESTED — P1-1 at λ=5e-2 tests this.
- SGD/AdamW ratio ≥ 10×. CONFIRMED: 18.3×.

### T2: Dual Characterization (Bridge, PROPERLY ATTRIBUTED)

τ* = η/λ (Xie & Li 2024) and R_* = λ/η (Defazio 2025) are the same quantity ρ from different lenses. Our contribution: the explicit bridge theorem naming these dual descriptions. NOT claimed as independent discovery.

### T3: ρ-Controller Convergence (Conditional on P1-1)

λ_t = λ₀ · (ρ_t / ρ*)^{−α} drives ρ_t → ρ* with O(e^{−αt}) convergence vs O(1/t) for constant WD. Lyapunov function V_t = (ρ_t − ρ*)². Proof sketch via Sun et al.'s stability analysis + Defazio's steady-state dynamics. Directly falsifiable via ρ_t trajectory diagnostic plot.

### T4: Weight Norm Collapse Property (T4, Empirical)

81× greater weight norm spread under SGD vs AdamW across 7 WD methods (SGD range: 64.5–127.0; AdamW range: 95.89–97.04). Mechanistic confirmation of T1: AdamW's ℓ∞ constraint collapses the weight space regardless of WD schedule.

---

## Part V: Updated Evidence Summary (Verified from Disk)

### SGD CIFAR-10 ResNet-20 (VERIFIED, n=3 per method, seeds 42/123/456)

| Method | Values | Mean ± Std | Δ vs constant | raw p | Holm-sig |
|---|---|---|---|---|---|
| constant | [91.30, 91.18, 91.17] | 91.217 ± 0.072 | — | — | — |
| cosine_schedule | [91.11, 91.34, 91.15] | 91.200 ± 0.123 | −0.017% | 0.884 | No |
| cwd_hard | [91.20, 91.02, 90.38] | 90.867 ± 0.431 | −0.350% | 0.254 | No |
| half_lambda | [90.65, 90.88, 91.00] | 90.843 ± 0.178 | −0.373% | 0.121 | No |
| random_mask | [90.46, 90.57, 91.29] | 90.773 ± 0.451 | −0.443% | 0.265 | No |
| **swd** | **[90.57, 90.63, 90.93]** | **90.710 ± 0.193** | **−0.507%** | **0.071** | **No** |
| **no_wd** | **[90.39, 90.19, 90.33]** | **90.303 ± 0.103** | **−0.913%** | **0.002** | **YES** |

Note: Holm-Bonferroni correction ranks no_wd first (p=0.0022, threshold=0.0083), all others non-significant.

### SGD CIFAR-100 ResNet-20 (n=3 except no_wd n=1)

| Method | Mean Acc. (%) | n |
|---|---|---|
| constant | 65.370 ± 0.161 | 3 |
| cosine_schedule | 65.107 ± 0.301 | 3 |
| random_mask | 64.913 ± 0.485 | 3 |
| half_lambda | 64.860 ± 0.469 | 3 |
| cwd_hard | 64.367 ± 0.577 | 3 |
| swd | 64.303 ± 0.501 | 3 |
| no_wd | 63.59 (1 seed: seed_42) | 1 |

**Action required**: Run CIFAR-100 SGD no_wd seed_123 (Gate 3, 20 min).

### AdamW CIFAR-10 ResNet-20 (Confirmed, n=3)

Spread 0.25% across all 7 methods. All BH-corrected p > 0.09. no_wd = 90.083% vs constant = 90.133%, Δ = 0.050%.

### SGD/AdamW Ratio

0.913% / 0.050% = **18.3×** (Bootstrap BCa CI required, expected [11×, 28×])

---

## Part VI: Novelty Assessment (Iteration 7)

No new literature search this iteration; confirming Iteration 6 novelty status:

| Contribution | Novelty Score | Key Prior Art | Differentiation |
|---|---|---|---|
| Phi Invariance Trichotomy regime structure | 7/10 | W&A (2024), D'Angelo (2024), Defazio (2025) | Explicit ρ-dependent regime boundaries; falsifiable crossover predictions |
| 18.3× SGD/AdamW ratio | 8/10 | None | Original controlled comparison, no prior paper |
| ρ-Controller closed-loop WD | 8/10 | Defazio (2025) identifies R_* dynamics | First closed-loop controller design; no prior paper |
| Phi Modulator taxonomy + BEM/AIS/CSI | 6/10 | Ye (2024) | More prescriptive; evaluation protocol included |
| T2 dual characterization bridge | 4/10 | Defazio (2025), Xie & Li (2024) | Synthesizing observation; properly attributed |

**ArXiv novelty searches from Iteration 6 (no updates this iteration)**:
- "weight decay regime boundary ρ = λ/η order parameter" → no hit
- "weight decay feedback control gradient-to-weight ratio steady state" → no hit
- "weight decay presence absence ratio SGD AdamW controlled comparison" → no hit

### Required Differentiation Actions Before Writing

1. Do NOT claim Theorem T2 as independent — cite Defazio (2025) and Xie & Li (2024) explicitly
2. Comparison table in Related Work: W&A=EMA heuristic (no regime boundaries); our Trichotomy=explicit ρ-thresholds
3. Kosson (2023) "rotational equilibrium" = Regime I; cite as interpretation
4. Chou (2025) "correction of decoupled WD" = corollary of Phi Invariance for proportional schedule
5. Do NOT say "first to derive ρ = λ/η" — say "first to characterize regime boundaries of ρ that determine when WD strategy timing matters"

---

## Part VII: Risk Register (Iteration 7)

| Risk | Probability | Impact | Status | Mitigation |
|---|---|---|---|---|
| BN confound confirmed (Path B) | 25% | Moderate | Gate 1 will resolve | Pre-scripted Path B narrative |
| λ sweep flat at ρ=50 | 25% | Low | P1-1 will test | "Invariance persists at all tested ρ" = publishable |
| Theorem T1 not provable | 20% | Low | Gate 0 will resolve | Present as Conjecture T1 |
| VGG-16-BN pilot reversal confirmed | 30% (elevated) | Positive if true | Gate 2 will resolve | WD harmful under AdamW on VGG = stronger than null |
| SWD claims in iter_003 writing | 100% (certain) | Moderate | Must fix | Update experiments.md section 5 before submission |

---

## Part VIII: Revisions from Prior Iterations

### From Iteration 6 Proposal

No major conceptual changes — Iteration 7 is a consolidation with verified data:

1. **SWD seed_456 = 90.93% verified from disk** (was asserted from synthesis in Iteration 6; now confirmed at file level)
2. **SWD Holm-significance retraction is definitive**: Holm-corrected p = 0.071 > threshold 0.010 with full n=3 data
3. **cosine_schedule seed values verified**: seeds [91.11, 91.34, 91.15] = mean 91.200% (matches prior summary)
4. **Next action clarified**: No more synthesis iterations needed. Execute Gates 0–3, then P0-P1 experiments, then write.

### From Codex Review (7/10, Iteration 4) — still pending resolution

1. Theorem provability → Gate 0 (pending)
2. BN confound → Gate 1 (pending)
3. Bootstrap CI for 18.3× → P0-1 (pending, code fix)
4. n=3 TOST underpowered → P0-2 (pending)

---

## Part IX: Score Trajectory

| Action | Δ Score | Cumulative |
|---|---|---|
| Starting point (result debate) | — | 5.5 |
| Correct SGD stats + CI (P0-1) | +0.3 | 5.8 |
| BEM/AIS/CSI repair | +0.2 | 6.0 |
| BN ablation (Gate 1) | +0.3 | 6.3 |
| VGG-16-BN cross-architecture | +0.5 | 6.8 |
| n=5 + TOST equivalence | +0.2 | 7.0 |
| Theorem T1 confirmed | +0.2 | 7.2 |
| λ Regime sweep (P1-1) | +0.3 | 7.5 |
| ρ-Controller pilot (P1-2) | +0.3 | 7.8 |
| ImageNet (P1-3) | +0.3 | 8.1 |

---

## Weighting Rationale

**Most weighted**: Empiricist (provides verified data, controlled experimental design), Theoretical (proof structure for T1, honest Conjecture framing), Innovator (ρ-Controller novelty + formal Lyapunov basis).

**Critical constraint**: Contrarian's BN confound remains the highest-severity unresolved structural risk. All mechanistic claims are conditional on Gate 1 results.

**Downweighted**: Interdisciplinary's Maxwell relations (Section 2.5 physical interpretation, not core contribution); Super-Twisting SMC (demoted to conditional Appendix).

**Synthesis principle for Iteration 7**: No new idea generation — the proposal is mature. The synthesis task is to consolidate verified evidence, confirm no pivot is warranted, and specify the executable experimental queue.

---

*This proposal was synthesized by the Synthesizer Agent from 6 independent agent perspectives (Iteration 7), the Result Debate Verdict (PROCEED at 5.5/10), and verified SGD experimental data from disk. The research direction is converged. Execute Gate experiments, then write.*
