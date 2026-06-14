# Pragmatist Perspective

**Agent**: Sibyl Pragmatist (sibyl-standard)
**Date**: 2026-03-18
**Iteration**: 6 (idea_debate stage)
**Topic**: Unified Dynamic Weight Decay Framework — Stability-Optimal Control Theory

---

## Phase 1: Literature Survey

### Key Resources Found

1. **`tml-epfl/why-weight-decay`** (GitHub, MIT) — D'Angelo et al. NeurIPS 2024. Ready-to-run experimental framework: ResNet/VGG/ViT on CIFAR/ImageNet with per-epoch weight norm, gradient norm, BN tracking. **Primary scaffolding repo.** Has code.

2. **`zeke-xie/stable-weight-decay-regularization`** (GitHub, MIT) — SWD / AdamS optimizer. One-file implementation of gradient-norm-aware dynamic WD from NeurIPS 2023. Direct SWD baseline. Has code.

3. **`dyunis/spectral_dynamics`** (GitHub, MIT) — Yunis et al. 2024. Full codebase for singular value tracking during training across architectures. Directly usable for rank-based WD visualization. Has code.

4. **CWD (Cautious Weight Decay)** (arXiv:2510.12402, ICLR 2026) — Binary sign-alignment mask, one-line drop-in for any optimizer. Baseline alignment-aware method. Trivially implementable from paper description.

5. **Sun et al. CVPR 2025** — First nonconvex SGD convergence proof for SGDW. Alignment quantity δ_T as key theoretical variable. The foundational theory this project builds on. No public code, but the math is the scaffold.

6. **Defazio arXiv:2506.02285** — WD drives ‖g‖/‖w‖ ratio of normalized layers to a common steady state ("layer balancing"). The cleanest mechanistic WD explanation in the literature. ~10 lines to implement. No public code.

7. **NOVAK arXiv:2601.07876** — Critical warning: coupling WD with α_eff (not base α) degrades CIFAR-100 by 4-8pp. Directly relevant to PMP-WD design: the control variable must be based on ρ = λ/η, not λ alone.

8. **AdamO arXiv:2602.05136** — Radial/tangential dynamics decoupling. Not yet publicly released. Provides the "Radial Tug-of-War" framing that motivates PMP-WD's norm-targeting control law.

9. **`facebookresearch/schedule_free`** (GitHub, MIT) — WD at interpolated point y (not w) improves results. Practical comparison point for schedule-independent WD.

10. **Kosson et al. arXiv:2305.17212** — Rotational equilibrium: ρ = λ/η is the fundamental invariant for normalized networks. Explains why our data shows method insensitivity — all methods operated at ρ ≈ 0.5 under AdamW.

11. **Wang & Aitchison arXiv:2405.13698** — WD as EMA timescale. Optimal timescale ≈ constant in epochs across model/dataset scales. Provides theoretical backing for ρ scale-invariance.

12. **`kozistr/pytorch_optimizer`** (GitHub) — 100+ optimizers and schedulers, including warmup_stable_decay. Useful survey of what practitioners use in production.

### Landscape Summary

**What the existing data (iter_003) definitively shows:**

- **AdamW regime (ρ ≈ 0.5)**: All 7 WD strategies span ≤0.25% accuracy on CIFAR-10, ≤0.76% on CIFAR-100. No method survives Bonferroni correction. The null result is **rock-solid** with 3 seeds × 7 methods × 2 datasets.
- **SGD regime (ρ ≈ 0.005)**: Method spread is 3.7× larger (0.91% CIFAR-10, larger CIFAR-100). Constant WD wins at 91.22% ± 0.07%; CWD underperforms at 90.87% ± 0.43%; SWD at 90.71% ± 0.19%; no_wd at 90.30% ± 0.10%.
- **Critical confound**: The SGD vs AdamW comparison is confounded by a 100× difference in ρ (0.005 vs 0.5). This is the single most important uncontrolled variable.
- **SWD consistency**: SWD has systematically high CSI (~1.16 for SGD, ~0.84+ for AdamW) and consistently underperforms constant WD on both optimizers at standard ρ. The high CSI = high stability cost interpretation is the cleanest theoretical explanation.

**What doesn't work in practice (empirically confirmed):**
- Binary sign masking (CWD) underperforms constant WD on SGD by 0.35% — a real, reproducible negative finding
- Gradient-norm-based scheduling (SWD) underperforms constant WD on SGD — increases stability cost without sufficient alignment benefit
- Random masking (equivalent WD budget by design) underperforms constant WD — the temporal distribution of WD matters, not just the total budget
- At standard ρ under AdamW: ALL dynamic strategies are statistically equivalent to constant WD

**The practical gap**: We have strong evidence that ρ = λ/η is the organizing control variable. The missing experiments are: (1) matched-ρ comparison to isolate the optimizer effect from the ρ confound, (2) high-ρ regime to find where methods diverge, (3) second architecture (VGG) at matched ρ.

---

## Phase 2: Initial Candidates

### Candidate A: Matched-ρ SGD Controls the Confound — The Most Important Missing Experiment

**Hypothesis**: The 3.7× larger method spread in SGD vs AdamW (0.91% vs 0.25%) is primarily explained by the 100× ρ difference (0.005 vs 0.5), not by optimizer class. At matched ρ = 0.5, SGD should show the same insensitivity as AdamW; at ρ = 5.0, both SGD and AdamW should show method divergence.

**Implementation sketch**:
- Run SGD (lr=0.1, weight_decay=0.05) to achieve ρ = λ/η = 0.5 — matching the AdamW ρ
- Run the same 7-method benchmark with SGD at this matched ρ
- Compare method spread: if spread(SGD, ρ=0.5) ≈ spread(AdamW, ρ=0.5) < 0.3%, then ρ is a sufficient explanation; if spread(SGD, ρ=0.5) >> spread(AdamW, ρ=0.5), then AdamW's ℓ∞ implicit bias is an additional independent cause
- This directly addresses the synthesizer's "uncontrolled for ρ" note in the current proposal

**Simplest possible version**:
- SGD, CIFAR-10, ResNet-20, ρ = 0.5 (lr=0.1, wd=0.05): {constant, no_wd, SWD} × 3 seeds = 9 runs
- Note: wd=0.05 is extremely high for SGD in standard practice but necessary for matched-ρ comparison
- Expected: ~20 min per run on a single GPU; 9 runs total = ~3 GPU-hours
- At ρ=0.5 with SGD, WD step is η × λ × ‖w‖ which equals learning-rate-sized steps — near Regime III boundary

**Time estimate**: 3 GPU-hours (9 runs at ~20 min each) for minimal version; 7 GPU-hours for all 7 methods

**Reusable components**:
- All current experiment infrastructure (zero new code, only config change: optimizer=sgd, lr=0.1, wd=0.05)
- Existing CSI/AIS/BEM logging
- Direct comparison with existing AdamW data

---

### Candidate B: ρ Sweep at Two Extreme Points — The Regime Map

**Hypothesis**: There exists ρ* ∈ [1, 5] such that at ρ > ρ*, alignment-aware methods (SWD, CWD) show Cohen's d > 0.5 improvement over uninformed methods (random_mask, half_lambda) on CIFAR-10 ResNet-20.

**Implementation sketch**:
- Add ρ = 5.0 level to existing benchmark: 7 methods × 3 seeds = 21 runs (under AdamW, η=1e-3, λ=5e-3)
- Add ρ = 0.05 level for under-regularization: 7 methods × 3 seeds = 21 runs (η=1e-3, λ=5e-5)
- The ρ = 0.5 data already exists; no re-running needed
- The only code change: modify the `weight_decay` config value

**Simplest possible version**:
- ρ = 5.0, ResNet-20 CIFAR-10, {constant, CWD, SWD, no_wd} × 3 seeds = 12 runs
- If constant and no_wd diverge at ρ=5.0: method spread has re-emerged at high ρ
- If CWD outperforms constant at ρ=5.0: Theorem 1 corollary (alignment benefit dominates stability cost at high ρ)
- ~2h on 4 parallel GPUs

**Time estimate**: 2 GPU-hours for minimal (12 runs); 7 GPU-hours for full 7-method sweep

**Reusable components**:
- All current infrastructure
- rho_sweep pilot already running on GPU 7 (as of current state)

---

### Candidate C: PMP-WD Implementation and Pilot — The Novel Algorithm

**Hypothesis**: The PMP-optimal control law λ*(t) = κ·(ρ* − ρ̂_t)^+ achieves accuracy ≥ constant WD on CIFAR-10 and > constant WD on high-ρ experiments, validating the stability-optimal control framing.

**Implementation sketch**:
- PMP-WD requires: (a) per-layer ‖g_l‖/‖w_l‖ ratio computation at each step (~10 lines), (b) EMA of this ratio as ρ̂_t, (c) target ratio ρ* set to Defazio's steady-state prediction (= η²/2λ for normalized layers under AdamW), (d) proportional feedback κ tuned to match CSI of constant WD
- Implementation fits in the existing optimizer wrapper (~30 lines total)
- Run: ResNet-20 CIFAR-10, 3 seeds, 200 epochs — compare vs constant, CWD, SWD
- Expected at low ρ: PMP-WD ≈ constant (per Theorem 3 low-costate prediction)
- Expected at high ρ: PMP-WD > constant (PMP exploits alignment signal when stability cost is low)

**Simplest possible version**:
- Seed=42 only, 200 epochs, compare to constant and CWD: 3 total runs
- This tests the baseline claim ("PMP-WD at least matches constant") at negligible cost
- ~2 GPU-hours (3 runs at 40 min each)

**Time estimate**: 2 GPU-hours for pilot; 6 GPU-hours for 3-seed comparison

**Reusable components**:
- Existing training loop
- Defazio's steady-state formula for ρ* (analytically derived from existing data: ρ* = η²/2λ ≈ 5×10^-7 / 10^-3 = 5×10^-4 — but this needs per-layer normalization context)

---

## Phase 3: Self-Critique

### Against Candidate A (Matched-ρ SGD)

**Implementation reality check**: Running SGD at ρ = 0.5 means wd=0.05 with lr=0.1. This is an unusual regime. At ρ = 0.5, the WD step per iteration is 0.05 × lr × ‖w‖, which is 50% of the learning rate step in magnitude. This is near the "aggressive WD" regime. Searching for SGD training at wd=0.05: this is unusual but not unheard of in continual learning or few-shot regimes. The training may converge slower or to a suboptimal point — but that's the point, it tests the theory.

**Reproducibility attack**: Setting ρ = 0.5 for SGD is well-defined. The experiment is exactly reproducible by changing two config values. Anyone can run it.

**Baseline sanity check**: The existing SGD data at ρ=0.005 shows constant WD at 91.22%. At ρ=0.5, we expect constant WD accuracy to decrease (WD is now very aggressive) — but the *relative* comparison across methods is what matters. The "baseline" is matched-ρ constant WD.

**Scope attack**: This tests a specific mechanism (ρ-driven insensitivity) that applies across all architectures and datasets where BN is present. The mechanism is not ResNet-specific. However, at ρ = 0.5 with SGD, training may be numerically unstable — this is an engineering risk.

**Verdict**: STRONG. Addresses the single biggest confound in the current paper. Zero new code. Direct test of a falsifiable mechanism. Engineering risk of instability at high ρ is manageable (start with ρ = 0.2 as a sanity check if ρ = 0.5 diverges).

---

### Against Candidate B (ρ Sweep)

**Implementation reality check**: Already being executed (pilot_rho_sweep_cifar10 running on GPU 7 as of this analysis). This is the safest possible experiment. Zero new code, only config changes. If the pilot confirms method divergence at high ρ, the full sweep is straightforward.

**Reproducibility attack**: Identical to the existing experiment suite; only ρ differs. Highly reproducible.

**Baseline sanity check**: At ρ = 5.0 (wd=5e-3, lr=1e-3 AdamW), expected accuracy decreases significantly for all methods due to over-regularization. The key question is relative ordering. Searching for AdamW experiments at wd=5e-3 with lr=1e-3: this is equivalent to wd_hat = λ/η_eff in normalized-learning-rate terms. With cosine annealing, late-stage wd/lr ratio is even higher. Instability risk is moderate.

**Scope attack**: CIFAR-10 results at unusual ρ values may not transfer to standard-practice experiments. The practical implication ("practitioners don't need complex WD at standard ρ") is the strongest takeaway, not the high-ρ behavior.

**Verdict**: STRONG. Already running. Direct test of the Phi trichotomy conjecture. Both positive and null results are publishable.

---

### Against Candidate C (PMP-WD Implementation)

**Implementation reality check**: The PMP-WD algorithm requires: (a) per-layer ‖g_l‖/‖w_l‖ ratio tracking, (b) EMA update, (c) proportional feedback. This is ~30 lines of PyTorch. However, the value of κ (the feedback gain) requires calibration. Defazio's steady-state formula gives ρ* analytically, but κ depends on the Riccati equation solution, which requires knowing the noise variance σ². In practice, κ would need to be tuned as a hyperparameter, making the experiment less clean than presented in Theorem 3.

**Reproducibility attack**: PMP-WD with a single hyperparameter κ is reproducible. But κ sensitivity is a concern — if PMP-WD requires κ tuning, it's less practical than constant WD. The paper needs to address this: either (a) provide an analytic formula for κ from training statistics, or (b) show that PMP-WD is robust to κ ∈ [κ_min, κ_max] and recommend a default.

**Baseline sanity check**: On CIFAR-10 at standard ρ, Theorem 3 predicts PMP-WD ≈ constant WD (low-costate regime). This means the "experiment validates the theory" framing requires PMP-WD to match constant — not beat it. This is a weak validation. The strong validation is at high ρ where PMP-WD should outperform constant. But Candidate B hasn't established high-ρ behavior yet.

**Scope attack**: PMP-WD is ResNet-20/CIFAR-10 validated. Without ImageNet results, the algorithm's practical relevance is hard to claim. Standard practice uses AdamW at ρ ≈ 0.5 — the regime where PMP-WD ≈ constant. The regime where PMP-WD helps (high ρ) is nonstandard.

**Verdict**: MODERATE. Theoretically compelling, but the pilot validation is weak (PMP-WD ≈ constant is the expected positive result). Priority: implement as a 3-run pilot *after* Candidate B confirms high-ρ method divergence, which establishes the setting where PMP-WD should help.

---

## Phase 4: Refinement

**Dropped ideas**: Candidate C is not dropped but demoted. It should run *only after* Candidate B confirms high-ρ method divergence. Running PMP-WD before knowing whether ρ=5.0 shows method sensitivity is premature.

**Refined priority order** based on evidence from the data and the proposal:

1. **Candidate B (ρ Sweep)** — already in-flight (GPU 7). Wait for pilot results. If pilot shows method divergence at high ρ: launch full 7-method sweep AND launch PMP-WD pilot simultaneously.

2. **Candidate A (Matched-ρ SGD)** — run immediately. This controls the most important confound in the paper. The ρ confound (100× difference) is the single most damaging critique the paper currently faces. Zero new code, 9 runs, ~3 GPU-hours. Should launch on GPU 0-2 in parallel with the ρ sweep continuation.

3. **Candidate C (PMP-WD)** — contingent on Candidate B results. If ρ=5.0 shows CWD > constant with Cohen's d > 0.5, then implement PMP-WD for that ρ regime. Expected ~30 lines of code + 3 seed runs.

**Additional search**: Searched for existing implementations of matched-ρ SGD comparisons in the WD literature — none found. The "budget-equivalent" comparisons in the WD literature (Chou 2025 gamma^2 scaling; Wang & Aitchison 2024 EMA timescale) normalize by model scale, not optimizer type. Our matched-ρ comparison is genuinely novel.

**Key simplification**: The most important experiment is not the most novel algorithm (PMP-WD) but the most direct confound control (matched-ρ SGD). The paper's theoretical contribution (Theorems 1-2) already has 7/7 empirical predictions confirmed. The remaining weakness is not theoretical — it's the uncontrolled confound in the SGD/AdamW comparison.

**Selected front-runner**: **Candidate A (Matched-ρ SGD) + Candidate B (ρ Sweep) as co-equal P0 priorities.** Both are zero-new-code experiments targeting the paper's most important empirical weakness. They must run in parallel.

---

## Phase 5: Final Proposal

### Title
Characterizing the Stability Cost of Dynamic Weight Decay: Empirical Falsification of the ρ-Confound and Regime Boundary

### Hypothesis (Falsifiable)
**H1 (Confound)**: The 3.7× larger method spread of SGD vs AdamW is explained primarily by the 100× ρ difference, not by optimizer class. At matched ρ = 0.5, SGD method spread will be ≤ 0.3% (matching AdamW), falsifying the "AdamW's ℓ∞ bias is the proximate cause" interpretation.

**H2 (Regime boundary)**: There exists ρ* ∈ [1, 5] such that for ρ > ρ*, alignment-aware methods (CWD, SWD) produce statistically distinguishable outcomes from uninformed methods (random_mask), with Cohen's d > 0.5.

**H3 (PMP-WD prediction)**: PMP-WD ≈ constant WD at ρ = 0.5 (low-ρ regime); PMP-WD > constant WD at ρ = 5.0 (high-ρ regime, if H2 is confirmed).

### Motivation
The current paper has a decisive weakness: the SGD/AdamW comparison conflates optimizer class with ρ level (100× difference). Reviewers will immediately flag this. The matched-ρ SGD experiment is the most direct possible fix — it runs in 3 GPU-hours with zero new code.

The ρ sweep directly validates the Phi trichotomy conjecture (Regime I: ρ low → methods equivalent; Regime II: ρ moderate → methods diverge under right conditions; Regime III: ρ very high → all methods degrade). This converts the null result from "we didn't find a winner" to "we identified the invariance boundary — and it's at ρ* ≈ X."

### Method

**Step 1: Matched-ρ SGD (Priority P0-3, fills the confound gap)**

Run SGD on CIFAR-10 ResNet-20 at three ρ levels:
- ρ = 0.005 (standard SGD: lr=0.1, wd=5e-4) — **already have data**
- ρ = 0.5 (matched AdamW: lr=0.1, wd=0.05) — **critical new experiment**

Methods: {constant, no_wd, SWD, CWD} × 3 seeds at ρ = 0.5 = 12 runs

Code: zero new code, change `optimizer=sgd, lr=0.1, wd=0.05` in config.

**Step 2: ρ Sweep Under AdamW (Priority P0-2, already in-flight)**

ρ levels under AdamW (lr=1e-3):
- ρ = 0.05: wd=5e-5 — **new**
- ρ = 0.5: wd=5e-4 — **have data**
- ρ = 5.0: wd=5e-3 — **pilot running**

Methods: {constant, CWD, SWD, random_mask, half_lambda, no_wd, cosine_schedule} × 3 seeds

**Step 3: VGG-16-BN at standard ρ, 200 epochs (Priority P0-4)**

3 methods {constant, SWD, no_wd} × 3 seeds at ρ = 0.5: 9 runs
Confirms multi-architecture invariance at standard ρ.

**Step 4: PMP-WD Pilot (Priority P0-5, contingent on ρ sweep result)**

If ρ = 5.0 sweep shows Cohen's d > 0.5 for at least one alignment-aware method:
- Implement PMP-WD: λ_t = clip(κ·(ρ* − ρ̂_t)^+, 0, λ_max) where ρ̂_t = EMA(‖g_l‖/‖w_l‖) per layer
- Run 3 seeds, 200 epochs, CIFAR-10 ResNet-20 at ρ = 5.0
- ~30 lines of implementation code in the optimizer wrapper

### Simplest Viable Experiment

**Maximum parsimony version**: Run only matched-ρ SGD (12 runs, ~4 GPU-hours) and wait for the pilot ρ sweep (already running). These two experiments together close the two biggest gaps in the current draft.

If both confirm theoretical predictions:
- H1 confirmed → paper can claim "ρ is the sufficient explanation for optimizer sensitivity"
- H2 confirmed → paper can claim "methods diverge at ρ* ≈ [measured value]"

Either outcome (positive or null) is publishable and theoretically interpretable.

### Baselines

1. **AdamW, ρ=0.5, constant WD** (existing data): 90.13% ± 0.31% CIFAR-10 AdamW; 91.22% ± 0.07% SGD
2. **AdamW, ρ=0.5, no WD** (existing data): 90.08% ± 0.32% — establishes the "WD irrelevance" baseline
3. **SGD, ρ=0.005, constant WD** (existing data): 91.22% ± 0.07% — current SGD anchor
4. **SGD, ρ=0.5, constant WD** (to be run): expected accuracy decrease (high WD), but relative spread is the key metric
5. **AdamW, ρ=5.0, constant WD** (partially in-flight): expect ~88-90% accuracy (strong over-regularization)

### Experimental Plan

**Wave 1 (~T+0 → T+2h): Matched-ρ SGD**
- 12 runs: SGD, CIFAR-10, ResNet-20, ρ=0.5, {constant, no_wd, SWD, CWD} × 3 seeds
- GPU 0-3 (3 parallel runs, ~20 min each = ~1h total)

**Wave 1 parallel: ρ sweep continuation (already running, GPU 7)**
- Wait for pilot_rho_sweep_cifar10 result
- If pilot shows divergence: launch full 7-method ρ=5.0 sweep (21 runs, GPU 4-6)
- If pilot shows null: record and plan ρ=10.0 as next level

**Wave 2 (~T+2h → T+6h): VGG full scale**
- 9 runs: VGG-16-BN, CIFAR-10, ρ=0.5, {constant, SWD, no_wd} × 3 seeds
- GPU 0-2 (~45 min per run)

**Wave 3 (~T+6h → T+8h, contingent): PMP-WD pilot**
- 3 runs: PMP-WD seed=42 at ρ=5.0 (if pilot confirmed divergence)
- GPU 0: ~40 min

**Ablations (priority order):**
1. Matched-ρ SGD: method spread at ρ=0.5 vs ρ=0.005 (confound falsification)
2. AdamW ρ sweep: invariance boundary (ρ* characterization)
3. VGG architecture: generality of CIFAR-10 ResNet-20 findings
4. PMP-WD pilot: algorithm validation at high-ρ regime

**Metrics**:
- Primary: Best test accuracy (3 seeds, mean ± std), Cohen's d vs constant baseline
- Secondary: CSI (coupling stability), AIS (alignment informativeness)
- Mechanism: ‖g_l‖/‖w_l‖ ratio per layer (logged free with existing hooks)

### Resource Estimate

| Experiment | Runs | GPU-hours (serial) | GPU-hours (parallel, 4 GPUs) |
|---|---|---|---|
| Matched-ρ SGD (4 methods × 3 seeds) | 12 | ~4h | ~1h |
| ρ=5.0 full sweep (7 methods × 3 seeds) | 21 | ~7h | ~1.75h |
| ρ=0.05 sweep (7 methods × 3 seeds) | 21 | ~7h | ~1.75h |
| VGG-16-BN (3 methods × 3 seeds) | 9 | ~7h | ~1.75h |
| PMP-WD pilot (contingent, 3 seeds) | 3 | ~2h | ~0.5h |
| **Total P0 CIFAR** | **66** | **~27h** | **~6.75h** |

Wall-clock estimate with 8× RTX PRO 6000: all P0 CIFAR experiments complete in **~4 hours running in parallel batches**. Well within project constraints.

### Risk Assessment

| Risk | Probability | Mitigation |
|---|---|---|
| SGD diverges at ρ=0.5 (wd=0.05 too aggressive) | 35% | First run seed=42 only as sanity check; reduce to ρ=0.2 if divergent |
| ρ=5.0 sweep shows null result (invariance persists) | 30% | This confirms "invariance is surprisingly robust"; increases the scope of the null finding; still publishable |
| PMP-WD implementation requires κ tuning (hyperparameter sensitivity) | 50% | Use κ=0.1 as default from theoretical derivation; ablate κ ∈ {0.01, 0.1, 1.0} in 1 run each |
| VGG shows opposite trend (invariance breaks at standard ρ) | 10% | Report it; the difference itself (architecture-dependent invariance) is informative |
| Matched-ρ SGD confirms H1 but H2 is null (no ρ* found up to ρ=5) | 20% | Extend to ρ=10 or ρ=20; if invariance persists, reframe paper as "invariance is structural, not ρ-dependent" |

**No risk is fatal**: every outcome maps onto a theoretically motivated prediction. The worst case is "invariance persists even at ρ=5 on CIFAR-10" — which would be an even stronger result ("WD strategies are equivalent across 100× ρ range").

### Novelty Claim

**What is new (conservative):**
1. **Matched-ρ confound control**: No prior paper has compared SGD vs AdamW at matched ρ levels. This is the first controlled comparison that isolates the optimizer effect from the ρ effect.
2. **Empirical ρ* characterization**: No paper has mapped out where WD strategy sensitivity emerges as a function of ρ. The ρ-regime boundary is an empirical finding with direct practical value.
3. **Theorem 1 empirical validation**: 7/7 binary masking stability cost predictions confirmed from iter_003 data; matched-ρ SGD provides 8th independent prediction test.
4. **PMP-WD algorithmic contribution**: First WD algorithm derived from Pontryagin's Maximum Principle; existing methods (CWD, SWD, cosine, constant) are formally shown to be costate approximations.

**Even if PMP-WD ≈ constant on all CIFAR experiments**: The negative result ("optimally controlled WD doesn't help at standard ρ") is theoretically explained by Theorem 3 (low-costate regime). This is the correct outcome under the theory, and its confirmation strengthens the theoretical framework.

---

## Summary Recommendation

**Execute now, in parallel:**

1. Launch matched-ρ SGD immediately (12 runs, GPU 0-3, ~1h): `optimizer=sgd, lr=0.1, wd=0.05` — this is the single highest-ROI experiment for fixing the paper's main weakness.

2. Wait for pilot_rho_sweep_cifar10 on GPU 7 (should complete in ~10 min per breadcrumb.json). Upon completion: if spread > 0.5%, launch full 7-method ρ=5.0 sweep (21 runs, GPU 4-6); record as Gate 1 decision point.

3. Run VGG-16-BN full scale (9 runs, GPU 0-2, after Wave 1 completes, ~2h).

4. Run PMP-WD pilot only after Gate 1 confirms ρ=5.0 method divergence.

**Do NOT do:**
- Start writing any results sections before matched-ρ SGD and ρ sweep data are in hand
- Run ImageNet before CIFAR ρ sweep is complete (wrong priority; the confound fix is more urgent than scale)
- Add more WD method variants (7 is sufficient for full characterization)
- Tune κ for PMP-WD before the high-ρ regime is confirmed to show method sensitivity

**The paper's strength depends on executing these in the right order**: matched-ρ SGD first (closes the confound), then ρ sweep (establishes the boundary), then VGG (multi-architecture), then PMP-WD (algorithm validation in the confirmed high-ρ regime). Any other order risks wasting compute on the algorithm before the regime is established.

The paper can be written in 2 days once these ~66 runs complete and Gate 1/2 decisions are made.
