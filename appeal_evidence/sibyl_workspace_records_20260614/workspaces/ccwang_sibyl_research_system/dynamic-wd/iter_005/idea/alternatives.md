# Backup Research Directions — Iteration 7

**Date**: 2026-03-18
**Context**: Primary direction is "Stability-Optimal Control Theory of Dynamic WD" (Theorems 1-3 + PMP-WD + experiments). These are pivot options if the primary fails.

---

## Backup A: Pure Empirical Characterization (if Theorem proofs blocked)

**Title**: "Weight Decay Methods Are Equivalent Under AdamW: A Comprehensive Empirical Study"

**Pivot condition**: If the proofs for Theorem 1 (binary masking stability cost) encounter a gap in the independence assumption (A3: alignment events independent of stability), and we cannot recover with a weaker formulation in 2 iterations.

**Core claim**: We systematically show, across 2+ architectures, 2 datasets, 3+ scales, and 3+ seeds, that no dynamic WD method improves over constant WD under AdamW when ρ ≤ ρ* ≈ 1-5. The finding is presented as an empirical law analogous to the "batch size scaling law" — not a theorem, but a precise, reproducible characterization.

**Strengths**:
- Requires no theorem proving
- Fully executable with the Iter 6 experiment plan
- The ρ-regime map (positive finding at high ρ) prevents it from being purely negative

**Weaknesses**:
- Lower theoretical novelty without the theorems
- More vulnerable to "you can't prove a universal negative" objection
- The SWD/NeurIPS paper got top venue publication from clear negative findings — precedent exists

**Expected score without theorems**: 6.5-7.0

---

## Backup B: PMP-WD Algorithm Paper (if empirical validation is strong but theorems remain weak)

**Title**: "PMP-WD: A Pontryagin-Optimal Weight Decay Algorithm Derived from First Principles"

**Pivot condition**: If PMP-WD shows ≥ 0.5% improvement over constant at high-ρ or ImageNet scale, and the theoretical proofs remain incomplete.

**Core claim**: We derive a novel WD algorithm from Pontryagin's Maximum Principle and show empirically that it outperforms constant WD in regimes where dynamic WD is theoretically expected to help (high ρ, large batch, ImageNet scale). The existing methods (CWD, SWD, cosine) are presented as heuristic approximations to the PMP-optimal solution.

**Strengths**:
- A positive algorithm result is more impactful than a null result
- The PMP framework is genuinely novel (9/10 novelty per innovator)
- Algorithm papers with theoretical grounding have strong track records at NeurIPS/ICLR

**Weaknesses**:
- Requires PMP-WD experimental success (currently uncertain)
- If PMP-WD ≈ constant even at high ρ, this pivot fails
- The stochastic PMP extension (mini-batch case) requires careful handling

**Expected score if algorithm works**: 7.5-8.0

---

## Backup C: Regime-Dependent WD Decision Guide (if all theory fails but experiments are complete)

**Title**: "A Practitioner's Guide to Weight Decay Scheduling: When Constant WD Is Optimal and When It Isn't"

**Pivot condition**: If both the theorem proofs fail AND PMP-WD shows no empirical advantage, but the ρ-regime map experiments are complete (showing transition at ρ ≈ ρ*).

**Core claim**: Based on comprehensive experiments across ρ values, architectures, optimizers, and scales, we provide the first empirically-grounded decision diagram for WD method selection. The key variable is ρ = λ/η: below ρ* ≈ 1-5, constant WD is near-optimal; above ρ*, alignment-aware methods provide measurable gains. A secondary variable is batch size (large batch → alignment is more reliable → alignment-aware WD helps more).

**Strengths**:
- Pure empirical contribution, robust to theory failures
- High practical impact — practitioners can use the decision diagram
- ρ as the organizing variable is novel

**Weaknesses**:
- Without theory, it reads as a set of empirical observations without explanatory depth
- Vulnerable to "why does ρ matter?" without Theorem 1

**Expected score**: 6.0-6.5 (useful but not top-venue quality without theory)

---

---

## Iter 7 New Backup: SPWD (Spectral-Phase-Transition Weight Decay)

### When to use
If PMP-WD shows no advantage over constant WD even at high ρ, pivot to SPWD as the novel algorithm. SPWD is the Iter 7 Innovator's key contribution: a WD scheduler conditioned on per-layer stable rank velocity (structural signal, not gradient signal). Novelty confirmed at 8/10 — no prior work uses rank velocity as WD feedback.

### Algorithm
λ_t = λ₀·(1 + α·tanh(-v_t)) where v_t = EMA of per-layer stable rank change rate (β=0.9, α=0.5).

Key prediction: WD peaks during rapid rank collapse (when rank is decreasing fast), WD is minimal when rank stabilizes — opposite of "anneal WD near convergence" intuition.

### Why SPWD survives where PMP-WD might fail
PMP-WD depends on linear ρ-dynamics near steady state. SPWD uses a weight-space structural signal (rank), which is less susceptible to batch-noise (scalar norms vs cosine similarity inner products). Addresses the Contrarian's noise concern directly.

### Resource: 18 GPU-hours, ~4 wall-clock hours. Priority P1-B.

---

## Revised Pivot Decision Protocol

```
Gate 1 (VGG full complete):
  VGG range < 0.5% → Multi-architecture null → proceed primary direction
  VGG range > 1.0% → Architecture matters → investigate mechanism, update framing

Gate 2 (rho_high complete):
  rho_high range > 0.5% → Theorem 1 Corollary supported → Launch PMP-WD pilot (P1-A)
  rho_high range < 0.25% → Invariance robust to ρ → strengthen "BN is the mechanism" claim

Gate 3 (matched-ρ SGD complete):
  matched-ρ range < 0.25% → ρ fully explains SGD/AdamW gap
  matched-ρ range > 0.5% → AdamW ℓ∞ bias additionally important → update Theorem 1 conditions

P1 Pilots (PMP-WD + SPWD):
  PMP-WD > constant at high-ρ → Primary Scenario A
  PMP-WD ≈ constant, SPWD > constant on SGD → Pivot to SPWD algorithm
  Both ≈ constant → Strong null paper with Theorems 1-2 (Scenario C/D)
```
