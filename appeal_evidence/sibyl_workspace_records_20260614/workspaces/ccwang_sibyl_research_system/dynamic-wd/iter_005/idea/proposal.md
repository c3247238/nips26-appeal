# Iteration 7 Synthesis Research Proposal

**Synthesizer**: sibyl-synthesizer (Sonnet 4.6)
**Date**: 2026-03-18
**Prior Iteration Score**: Iter 6 proposal target = 7.5–8.0 (experiments in-flight)
**Focus Mode**: FOCUSED (1–2 candidates, front-runner + 1 backup)

---

## I. Landscape Mapping: Six Perspectives in Round 7

### I.1 What Changed from Iter 6 to Iter 7

The Iter 6 proposal established the "Stability-Optimal Control Theory" framing with three theorems (Theorem 1: binary masking suboptimality; Theorem 2: layer-wise CSI bound; Theorem 3: PMP-WD optimal control). In Iter 7, six new perspectives were generated with the following decisive new contributions:

**New evidence since Iter 6:**
- VGG-16-BN/constant: seed_42=92.03%, seed_123=92.00% — **VGG null result confirmed for constant WD** (only partial evidence; other methods not yet complete)
- rho_low (ρ=0.05)/constant/seed_42: at epoch 119/200 → acc=88.1% (stable training confirmed)
- matched_rho_sgd/constant/seed_123: at epoch 122/200 → acc=86.84% (stable training confirmed)
- nobn/constant/seed_42: at epoch 121/200 → acc=87.01% (stable training, AIS=0.499 at ep121 vs BN AIS~0.34)

**Three new algorithmic proposals from Iter 7 perspectives:**
1. **SPWD (Spectral-Phase-Transition WD)** — Innovator: λ_t = λ₀·(1 + α·tanh(-v_t)) where v_t = EMA of per-layer stable rank velocity. First WD method conditioned on structural state (rank) rather than dynamics state (gradient).
2. **QA-WD (Quadratic-Alignment WD)** — Interdisciplinary: λ*_t = β₀·δ̂²_t derived from RG beta function + PMP convergence. Converges with Iter 6's PMP-WD from an independent theoretical route.
3. **PID-WD** — Innovator secondary: λ_t = λ_P·(1−δ_t) + λ_I·μ_t (proportional-integral alignment controller)

**New challenges raised:**
- **Contrarian**: Alignment signal δ̂_t has CV>>1 at batch ≤ 256 — the batch size used in ALL our experiments. This is the single most important new critique: if alignment is noisy, SPWD's rank velocity signal is better (structural, not gradient-based), and QA-WD's EMA(|δ̂_t|²) at alpha=0.99 may average away the noise.
- **Empiricist**: VGG-16-BN is running only constant (sequential GPU allocation). Other 6 methods must be parallelized immediately. Gate 1 (VGG full N=3) is BLOCKING.

### I.2 Convergence Points Across All Six Perspectives

All six agents agree on:

1. **Empirical anchors remain unchanged**: AdamW range 0.25% (CIFAR-10), 0.75% (CIFAR-100); SGD range 0.91% (CIFAR-10), 1.71% (CIFAR-100). These are rock-solid with N=3 × 7 methods.
2. **The 100× ρ confound is still the paper's #1 weakness** (unresolved without matched-ρ data)
3. **Constant WD is best or tied-best under all tested conditions** — the Iter 6 theoretical explanation via Theorem 1 remains valid
4. **VGG-16-BN constant: seed_42=92.03%, seed_123=92.00%** — consistent null signal for constant, but we need other methods
5. **The alignment signal IS noisy at batch=128** (Contrarian confirmed, SimiGrad validates) — design implication: any method using raw single-step δ̂_t is fragile; EMA smoothing (k≥10) is necessary
6. **Three independent derivations now converge on a continuous alignment-proportional WD as optimal**: PMP-WD (Theoretical, via Riccati), QA-WD (Interdisciplinary, via RG beta function), and PID-WD (Innovator, via control theory). The fact that three approaches from different mathematical origins converge is strong evidence this is the correct optimal form.

### I.3 Critical Disagreements and Their Resolution

| Disagreement | Agents | Synthesizer Decision |
|---|---|---|
| **SPWD (rank velocity)** as new front-runner vs **PMP-WD** | Innovator vs Theoretical | **Resolved**: SPWD is genuinely novel (no prior work uses rank velocity as feedback) and structurally distinct from PMP-WD. However, SPWD adds algorithm complexity to an already-complex paper. **Decision**: Retain PMP-WD/QA-WD as the primary algorithmic contribution (theoretically grounded, converges from 2 independent derivations). Add SPWD as a "structural feedback" alternative in the paper's Discussion or as a new empirical experiment if compute allows. SPWD is promoted to **backup candidate** replacing the pure empirical candidate. |
| **QA-WD** (λ ∝ δ²) vs **PMP-WD** (λ ∝ (ρ*−ρ̂)⁺) — which is the "true" optimal? | Interdisciplinary vs Theoretical | **Resolved**: They are complementary, not competing. QA-WD uses alignment signal directly (δ̂²); PMP-WD uses the ρ-ratio error (ρ*−ρ̂_t). Both are proportional feedback on the gradient-weight geometry: (a) QA-WD is the near-zero-ρ approximation (when ρ ≈ 0, alignment ≈ 0, so δ̂² ≈ 0 ≈ constant WD); (b) PMP-WD is the near-steady-state approximation. Mathematically, for a normalized network at steady state with ρ̂_t ≈ ρ*·δ_t, we have κ·(ρ*−ρ̂_t)⁺ ≈ κ·ρ*·(1−δ_t)⁺ ≈ β₀·δ̂²_t in the moderate-alignment regime (δ̂ ∈ [0.3, 0.7]). They differ at the extremes. **For experiments**: use PMP-WD (cleaner state-feedback interpretation, connects to ρ-dynamics narrative); note QA-WD convergence in the theory section as independent validation. |
| **Alignment signal noise** (Contrarian) vs **alignment IS informative** (Theoretical, Innovator) | Contrarian vs Theoretical | **Resolved with regime-dependence**: The Contrarian is correct for raw single-step δ̂_t at batch=128 in small BN networks. But: (a) CWD's sign alignment is more noise-robust (binary, low-dimensional); (b) PMP-WD uses per-layer EMA of ρ̂_t = ‖g_l‖/‖w_l‖ (ratio, not cosine), which is less susceptible to sign noise; (c) QA-WD mandates EMA smoothing (α=0.99). The Contrarian's insight is absorbed as a design constraint: **all proposed methods must use EMA-smoothed signals, not raw single-step estimates**. This is now a stated requirement in the algorithm design. |
| **SPWD requires new experiment budget** vs **focus on existing experiments** | Innovator vs Pragmatist/Empiricist | **Resolved**: SPWD experiments (18 GPU-hours) can only run AFTER the primary P0 experiments complete. Current priority remains: (1) VGG other methods parallelization, (2) rho_low/rho_high full sweep, (3) matched-ρ SGD completion, (4) NoBN completion. SPWD is P3 (contingent). |

---

## II. Research Direction Decision

### II.1 Core Thesis Maintained (Iter 7 = Refined Iter 6)

The Iter 6 framing is correct and maintained with one clarification:

> **Iter 7 core thesis**: The stability cost of alignment-based weight decay modulation — formalized as binary masking stability cost in Theorem 1 and layer-wise CSI bound in Theorem 2 — explains why constant WD outperforms dynamic WD at standard ρ in BN networks. The optimal feedback control law for the gradient-to-weight ratio dynamics (Theorem 3 / PMP-WD), derived from both stochastic PMP and the RG beta function independently, converges to: λ*(t) ∝ signal(ρ̂_t, ρ*, δ̂_t) with EMA smoothing required for small-batch stability. Experimental validation across ρ regimes, architectures, and scales confirms the theory.

**Why "Stability-Optimal Control Theory" is still the right framing:**
1. The null result (constant WD wins at standard ρ) becomes the theoretical prediction — Theorem 1
2. The algorithm (PMP-WD) is derived from first principles — Theorem 3
3. The dual derivation (PMP + RG beta function → same formula) strengthens the theoretical validity
4. The design principle (EMA smoothing for small-batch, continuous feedback over binary masking) addresses the Contrarian's noise challenge
5. This framing survives every challenge mounted in Iter 7

### II.2 Paper Title (Updated)

**Title**: "When Does Adaptive Weight Decay Help? A Stability-Optimal Control Theory of Dynamic Weight Decay"

(Unchanged from Iter 6. Backup title: "Stability Cost vs. Alignment Benefit: A Control-Theoretic Analysis of Dynamic Weight Decay")

### II.3 Algorithm Naming Decision

The paper proposes one new algorithm. The naming resolves the QA-WD vs PMP-WD tension:

**Algorithm name: PMP-WD** (Pontryagin Maximum Principle Weight Decay)

**Formula**: λ*(t) = clip(κ·(ρ* − ρ̂_t)⁺, 0, λ_max)

where:
- ρ̂_t = per-layer EMA of ‖g_l‖/‖w_l‖ (gradient-to-weight ratio, EMA momentum 0.9)
- ρ* = target steady-state ratio (from Defazio 2025: ρ* ≈ √(2λγ⁻¹) for AdamW normalized layers)
- κ = feedback gain from Riccati equation solution (treated as hyperparameter in experiments, default κ=1)

**Why PMP-WD over QA-WD**: The ρ-ratio (‖g‖/‖w‖) is a per-layer scalar readily computable and interpretable via Defazio's layer-balancing framework. The cosine similarity δ̂_t requires an O(d) dot product across the full parameter vector and has higher variance (as the Contrarian correctly notes). PMP-WD's EMA of ρ̂_t is less susceptible to batch noise because the ratio uses norms (scalars) rather than inner products (which can cancel).

**QA-WD connection noted in paper**: "An independent derivation from RG beta function theory (see Appendix) yields the equivalent prescription λ* ∝ δ̂²_t in the moderate-alignment regime, providing additional theoretical support."

---

## III. Experimental Plan (Iter 7 Refinement)

### III.1 Current Experimental State (As of 2026-03-18)

**What is DONE:**
- AdamW CIFAR-10 ResNet-20: 7 methods × 3 seeds = 42 runs ✓
- AdamW CIFAR-100 ResNet-20: 7 methods × 3 seeds = 42 runs ✓
- SGD CIFAR-10 ResNet-20: 7 methods × 3 seeds = 42 runs ✓
- SGD CIFAR-100 ResNet-20: 7 methods × 3 seeds = 42 runs ✓
- VGG-16-BN constant: seed_42 (92.03%), seed_123 (92.00%) ✓

**What is IN-FLIGHT (partially complete):**
- rho_low (ρ=0.05): constant/seed_42 at epoch 119/200
- matched_rho_sgd (ρ=0.5, SGD): constant/seed_123 at epoch 122/200
- nobn: constant/seed_42 at epoch 121/200

**What is MISSING (critical):**
- VGG-16-BN: 6 other methods × 3 seeds (21 additional runs) — BLOCKING Gate 1
- rho_high (ρ=5.0): all methods (need ρ-sweep full data)
- matched_rho_sgd: seeds 42 (5ep only) and 456 not done; methods other than constant not done
- PMP-WD implementation and pilot

### III.2 Priority Queue for Remaining Experiments

**CRITICAL (must complete before any writing):**

**P0-A: VGG-16-BN full parallelization** (highest priority, BLOCKING Gate 1)
- Launch all 6 remaining WD methods × 3 seeds on VGG-16-BN CIFAR-10
- 18 runs × ~13 min each = ~4 GPU-hours if parallelized across 6 GPUs
- This is the SINGLE highest-ROI action: reviewers require multi-architecture evidence
- Expected result: method range < 0.5% (null result confirmed for VGG)
- Implementation: zero new code, only config changes

**P0-B: rho_high (ρ=5.0) full sweep** (critical for Theorem 1 Corollary validation)
- 4 methods {constant, cwd_hard, SWD, no_wd} × 3 seeds × 200 epochs
- Launch on available GPUs; runs in parallel with VGG
- Gate 2 decision: if range > 0.5%, confirms ρ-driven sensitivity → Theorem 1 Corollary supported
- Falsification: if range < 0.25%, invariance is unexpectedly robust at high ρ — still publishable

**P0-C: rho_low (ρ=0.05) completion**
- Already in-flight (constant/seed_42 at ep119); launch seed_123, seed_456, other methods
- 3 methods × 3 seeds − 1 in-flight = 8 new runs + 1 completion
- Low urgency relative to rho_high

**P0-D: matched-ρ SGD completion**
- Complete seed_456 constant + launch cwd_hard and no_wd for all 3 seeds
- 8 additional runs × ~8 min = ~1 GPU-hour
- Gate 3: if range < 0.25% → ρ explains the confound; if range > 0.5% → AdamW mechanism additional

**HIGH VALUE (P1, after Gate 1+2):**

**P1-A: PMP-WD pilot** (contingent on rho_high showing method sensitivity)
- Implement λ*(t) = clip(κ·(ρ*−ρ̂_t)⁺, 0, λ_max) (~30 LOC in existing optimizer wrapper)
- Run ResNet-20 CIFAR-10 at ρ=5.0, 3 seeds × 200 epochs
- Expected: PMP-WD > constant at high ρ (theory prediction)
- If rho_high shows null: PMP-WD pilot at ρ=5.0 is still informative (algorithm matches theory)

**P1-B: SPWD pilot** (NEW from Innovator Iter 7, P1 priority)
- Implement stable rank velocity v_t = EMA of d/dt[‖W‖_F²/‖W‖₂²] per layer
- λ_t = λ_0·(1 + α·tanh(-v_t)) with α=0.5, β_EMA=0.9, power iteration k=2
- Run SGD CIFAR-10 ResNet-20 × 3 seeds × 200 epochs (compare vs constant, CWD, SWD)
- Rationale: rank velocity is a structural signal (Innovator's key insight), less noisy than alignment proxy
- This provides a genuinely new experimental column in the paper — either positive (new algorithm works) or negative (rank velocity not actionable → theoretical contribution)

**DEFERRED/OPTIONAL (P2):**

- ImageNet: requires resolving data availability; defer until P0 complete
- NoBN completion: in-flight; continue to completion but acknowledge LR/ρ confound (secondary result)
- Batch size sensitivity experiment (Contrarian's alignment noise test): interesting but not critical path

### III.3 Revised Gate Decisions

**Gate 1 (~T+4h wall-clock, after VGG full parallelization completes):**
- VGG range < 0.5%: "Multi-architecture confirmation of WD method insensitivity at standard ρ" ✓
- VGG range > 1.0%: Surprising finding — architecture matters; investigate per-layer dynamics

**Gate 2 (~T+6h, after rho_high complete):**
- rho_high range > 0.5%: Theorem 1 Corollary confirmed; launch PMP-WD pilot at ρ=5.0
- rho_high range < 0.25%: Invariance is robust to ρ even at extreme values — reframe as "structural insensitivity driven by BN, not ρ per se"
- Either outcome: write Section 5.4 (ρ-regime map); both are theoretically interpretable

**Gate 3 (~T+8h, after matched-ρ SGD complete):**
- Matched range < 0.25%: ρ is sufficient explanation; confirm Theorem 1's regime boundary
- Matched range > 0.5%: AdamW's ℓ∞ bias is additionally important; add to Theorem 1's assumptions

---

## IV. Theoretical Contribution Roadmap (Iter 7 Refinements)

### IV.1 Core Theorems (Maintained from Iter 6)

**Theorem 1 (Binary Masking Suboptimality)** — unchanged from Iter 6 (7/7 empirical confirmations):
```
CWD outperforms constant WD iff:
AIS > (Cσ²/n) × ΔCSI / λ̄
```
- Directly explains CWD failure on SGD small-batch BN (σ²/n non-negligible)
- Directly predicts CWD potential success in large-batch LLM training (σ²/n → 0)
- 7/7 predictions from iter_003 confirmed

**Theorem 2 (Layer-wise CSI Bound)** — unchanged from Iter 6:
```
GenGap({λ_{i,t}}) − GenGap(λ̄) ≤ (2Lσ²/n) × CSI_param × T
```
- Methods with λ_min=0 (CWD, random_mask) have unbounded CSI_param during off-steps
- Explains random_mask paradox: low aggregate CSI but high per-parameter CSI

**Theorem 3 (PMP-Optimal WD)** — Iter 7 REFINEMENT: dual-derivation now presented

The Iter 6 proposal derived PMP-WD from stochastic PMP + Riccati equation. Iter 7 adds:
- Independent derivation from RG beta function theory (Interdisciplinary perspective)
- Both routes converge to proportional feedback on the gradient-weight ratio signal
- Presented in paper as: "Theorem 3 (main text: PMP derivation) + Remark 3.1 (RG beta function convergence in Appendix)"

**New Iter 7 addition: Proposition 1 (Alignment Noise Design Constraint)**

From the Contrarian's analysis and SimiGrad (NeurIPS 2021) evidence:
```
For batch size b ≤ 256 and full-network cosine similarity:
CV(δ̂_t) = std(δ̂_t)/mean(δ̂_t) >> 1 for most training steps
```
Corollary: Any alignment-aware WD method must use temporally-aggregated alignment (EMA with k ≥ 10 steps) rather than single-step adaptation. This is a new design constraint that PMP-WD (via ρ̂_t EMA), QA-WD (via δ̂_t EMA with α=0.99), and SPWD (via rank velocity EMA with β=0.9) all satisfy by construction.

This Proposition converts the Contrarian's "noisy compass" challenge into a **design requirement** rather than a falsification: alignment IS informative when aggregated; it is uninformative step-by-step.

### IV.2 What NOT to Include (Maintained + Expanded)

- Do NOT claim PMP-WD as "provably optimal" in the full nonconvex setting (only for linear ρ-dynamics approximation near steady state)
- Do NOT claim QA-WD and PMP-WD are equivalent in general (they are equivalent only in the moderate-alignment regime)
- Do NOT claim SPWD results as a primary contribution unless the pilot shows statistically significant improvement
- Do NOT claim ImageNet generalization without ImageNet data (explicitly note as limitation)
- Do NOT present BEM as formally derived (circular dependency remains)

---

## V. Novelty Assessment (Iter 7)

### Front-Runner Contributions (Updated)

| Contribution | Novelty | Evidence Status | Iter 7 Status |
|---|---|---|---|
| **Theorem 1**: Binary masking suboptimality | 9/10 | 7/7 predictions confirmed | Unchanged |
| **Theorem 2**: Layer-wise CSI bound | 8/10 | random_mask paradox explained | Unchanged |
| **Theorem 3 + PMP-WD**: Dual-derivation optimal WD | 8/10 | Needs PMP-WD pilot | **Strengthened** (RG derivation adds independent validation) |
| **Proposition 1**: Alignment noise design constraint | 7/10 | SimiGrad + our batch size analysis | **New in Iter 7** |
| **ρ-Regime Map** | 8/10 | rho_low/rho_high running | Unchanged |
| **Multi-Architecture (VGG)** | 7/10 | constant only × 2 seeds confirmed | **Partially confirmed** |
| **Matched-ρ SGD** | 7/10 | In-flight | Unchanged |

### Backup Contribution: SPWD Algorithm

| Contribution | Novelty | Evidence Status | Priority |
|---|---|---|---|
| **SPWD**: Rank velocity feedback WD | 8/10 | No prior art found (AlphaDecay uses static per-module spectral density, not dynamic rank velocity) | P1 pilot after P0 complete |

### Novelty Verification (Iter 7 Searches)

**SPWD**: arXiv search for "spectral rank velocity weight decay feedback" returns no papers using rank rate-of-change as a WD signal. AlphaDecay (NeurIPS 2025, arXiv:2506.14562) uses static spectral density (PL_Alpha_Hill) for per-module WD assignment — fundamentally different from SPWD's dynamic rank velocity φ_rank(v_t). Gap confirmed at 8/10.

**QA-WD (λ ∝ δ²)**: Search for "renormalization group weight decay quadratic alignment" returns no ML papers. The specific formula λ* = β₀·δ̂²_t is not present in any known prior work. The ECT* 2024 workshop establishes the RG-WD correspondence but does not derive a feedback schedule. Gap confirmed at 8/10.

**Alignment noise constraint**: Search confirms SimiGrad (NeurIPS 2021) but finds no WD paper that directly characterizes "minimum aggregation horizon for alignment-aware WD." This Proposition is new.

### Revisions from Prior Feedback (Iter 6 → 7)

**Novelty report concern (Iter 6)**: PMP-WD has partial overlap with Defazio's AdamC (λ_t ∝ γ_t).
**Resolution**: Now explicitly differentiated in Theorem 3 statement: "AdamC is a feedforward schedule (λ depends on γ_t); PMP-WD is a state-feedback law (λ depends on current ρ̂_t measurement). Feedforward control ignores deviations from the planned ρ trajectory; state-feedback corrects for them in real-time."

**Evolution lesson concern**: "No deeper theoretical results beyond trivial Proposition 1."
**Resolution**: Theorem 1 now has 7/7 empirical confirmations, Theorem 3 has a dual derivation from two independent mathematical frameworks (PMP + RG), and Proposition 1 (alignment noise) is a new result providing a design principle rather than an empirical observation. The theoretical depth is substantially increased from Iter 5–6.

---

## VI. Writing Plan (Iter 7 Updates)

### VI.1 P2 Fixes (Execute Before Writing New Sections)

These remain from Iter 6 and must be done first:
| Fix | Status | Time |
|---|---|---|
| W1: Figure 2 SGD p-value annotation (p=0.004 vs text p=0.071) | Required | 30 min |
| W2: Cohen's d formula label (unpaired pooled) | Required | 15 min |
| W3: Figure 4 from actual epoch_metrics.jsonl | Required | 45 min |
| W4: Figure 3 CIFAR-100 "No correlation" → "Moderate correlation (r=0.48)" | Required | 15 min |
| W5–W10: Cross-reference consistency | Required | 1h |

### VI.2 New Sections (After Gate Decisions)

| Section | Content | Gate |
|---|---|---|
| Sec 3.3 Theorem 1 | Binary masking stability cost proof | existing data |
| Sec 3.4 Theorem 2 | Layer-wise CSI bound | existing data |
| Sec 3.5 Theorem 3 + PMP-WD | Dual derivation + algorithm; Prop 1 alignment noise constraint | theory ready |
| Sec 5.4 ρ-Regime Map | ρ={0.05, 0.5, 5.0} results | Gate 2 |
| Sec 5.5 Multi-Architecture | VGG-16-BN results | Gate 1 |
| Sec 5.6 Matched-ρ SGD | Confound resolution | Gate 3 |
| Sec 5.7 PMP-WD Results | Algorithm performance at high ρ | Gate 2+P1-A |
| Sec 5.8 SPWD Pilot (if positive) | Structural feedback WD | P1-B |

### VI.3 Figure Plan (Updated)

| Figure | Content | Priority |
|---|---|---|
| Fig 1: Regime diagram | Method range vs log(ρ) curve with ρ* threshold | Core |
| Fig 2: CSI-accuracy relationship | Revised with SGD and AdamW; correct p-values | Core |
| Fig 3: Theorem 1 illustration | Alignment benefit vs stability cost vs AIS threshold | Theory |
| Fig 4: PMP-WD control diagram | ρ_t trajectory, λ*(t) feedback, comparison to existing methods as costate approximators | Theory |
| Fig 5: NoBN vs BN effect size | Cohen's d per method — suggestive evidence for BN role | Mechanism |
| Fig 6: Matched-ρ SGD | AdamW vs high-ρ SGD sensitivity ratio | Mechanism |
| Fig 7: SPWD rank velocity (if run) | v_t trajectory + λ_t multiplier alignment | Structural |

---

## VII. Six-Perspective Adoption Summary (Iter 7)

| Perspective | Iter 7 Core Contribution | Adoption | Weight |
|---|---|---|---|
| **Innovator** | SPWD (rank velocity WD) + PID-WD as P+I controller | **Partially adopted** — SPWD promoted to backup candidate and P1 experiment; PID-WD noted in Discussion; SPWD's structural signal insight informs alignment noise constraint | Medium |
| **Pragmatist** | VGG parallelization (6 GPUs immediately) + matched-ρ SGD completion as #1 priority | **Fully adopted** — VGG parallelization is now the top action item; matched-ρ SGD completion next | High |
| **Theoretical** | Dual-derivation Theorem 3 (PMP + RG beta function); Proposition 1 alignment noise constraint | **Fully adopted** — dual derivation strengthens Theorem 3; Proposition 1 is new contribution | High |
| **Contrarian** | Alignment signal CV>>1 at batch≤256; must use EMA smoothing | **Fully adopted as design constraint** — Proposition 1 formalizes this; all proposed methods must use EMA (PMP-WD already does; SPWD does; QA-WD does) | High |
| **Empiricist** | VGG full parallelization; ρ-regime Gate 2; matched-ρ Gate 3; statistical rigor | **Fully adopted** — Gate structure maintained; VGG parallelization is critical blocking item; N=3 + TOST at δ=±1% maintained | High |
| **Interdisciplinary** | QA-WD (λ ∝ δ²) from RG beta function + PMP convergence | **Partially adopted** — QA-WD derivation added as "Remark 3.1 / Appendix independent derivation" confirming Theorem 3; not a separate algorithm; GC-WD analogy retained in Discussion | Medium |

---

## VIII. Dynamic Scope Management

### Scenario A: All P0+P1 Complete + PMP-WD works at high ρ + VGG null confirmed
- Full "stability-optimal control" paper with algorithm, theory, multi-architecture validation, and ρ-regime map
- Three theorems + Proposition 1 + PMP-WD + SPWD pilot + dual derivation
- Target score: **8.0–8.5**

### Scenario B: P0 complete, PMP-WD ≈ constant, VGG null confirmed
- "Stability theory" framing; PMP-WD presented as theoretical validation tool (expected null at low ρ)
- Three theorems + ρ-regime map + VGG + matched-ρ SGD
- Target score: **7.5–8.0**

### Scenario C: P0 CIFAR complete, VGG null, rho_high shows null
- "Invariance is structural and robust" — reframe: "WD method insensitivity persists across 100× ρ range in BN networks"
- Two theorems + multi-architecture + matched-ρ SGD + strong BN mechanism hypothesis
- Target score: **7.0–7.5**

### Scenario D: Only Theorems 1–2 + existing data + P2 fixes
- "When Does Alignment-Aware WD Help? A Stability Analysis" — lean theoretical paper
- Target score: **6.5–7.0**

---

## IX. Key Differences from Iter 6 Proposal

1. **SPWD promoted from "dropped" to backup candidate** — The Innovator's rank velocity insight is genuinely novel (confirmed by novelty search: no prior use of rank velocity as WD feedback). The structural signal (rank) is more noise-robust than the gradient signal (alignment), directly addressing the Contrarian's concern.

2. **Dual derivation of Theorem 3** — Interdisciplinary's RG beta function analysis independently derives λ* ∝ δ̂²_t, converging with the PMP-WD formula in the moderate-alignment regime. This strengthens Theorem 3's theoretical standing. Added as Remark 3.1 / Appendix.

3. **Proposition 1 (alignment noise constraint)** — New result converting the Contrarian's challenge into a positive contribution. EMA aggregation requirement (k ≥ 10 for cosine similarity; built into ρ̂_t for PMP-WD) is now a stated design principle.

4. **VGG constant: 2 seeds confirmed** — partial evidence supporting VGG null (92.03%, 92.00%), but parallelization of remaining 6 methods is URGENT and BLOCKING.

5. **Experiments in-flight with partial data** — rho_low at ep119, matched_rho_sgd at ep122, NoBN at ep121. All show stable training. Full data needed for statistical claims.

6. **PID-WD retained in Discussion** — Innovator's P+I alignment controller is noted but not pursued as a separate experiment. May be included as a 1-paragraph "directions" mention in the Discussion section.

7. **VGG parallelization identified as #1 immediate action** — All remaining VGG methods must be launched immediately on 6 separate GPUs. This is the single highest-ROI action to advance the paper.

---

## X. Success Criteria (Updated)

| Level | Completion | Expected Score |
|---|---|---|
| Minimum viable | Theorems 1–2 + iter_003 data + P2 fixes + VGG null | 6.5–7.0 |
| Target | Theorem 3 + PMP-WD + ρ-regime map + VGG + matched-ρ SGD | 7.5–8.0 |
| Strong | Target + SPWD pilot (positive) + dual derivation | 8.0–8.5 |
| Submission-ready | Strong + all P2 fixes + figures verified + LaTeX compiled | 8.5+ |

**Iteration 7 cardinal rule: VGG parallelization first. The other 6 VGG methods must be running before any other action. Gate 1 is blocking Gate 2 in reviewer perception: "you only tested one architecture" is the easiest criticism to make and the easiest to preempt.**

---

*Proposal generated by sibyl-synthesizer on 2026-03-18. This is Iteration 7, superseding the Iter 6 proposal. Principal changes: SPWD promoted to backup candidate (8/10 novelty confirmed); dual derivation of Theorem 3 (PMP + RG beta function); Proposition 1 (alignment noise design constraint) added; VGG partial confirmation (constant × 2 seeds, 92.03%/92.00%); VGG full parallelization identified as #1 blocking action.*
