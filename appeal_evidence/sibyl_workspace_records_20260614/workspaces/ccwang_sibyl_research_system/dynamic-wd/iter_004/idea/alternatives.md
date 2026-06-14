# Alternative Research Directions — Iteration 7

**Date**: 2026-03-18
**Purpose**: Pre-scripted contingency paths (not true pivots). Research direction is converged.

## Synthesis Verdict: NO PIVOT WARRANTED

The Iteration 7 synthesis confirms all foreseeable experimental outcomes lead to publishable papers within the current framework. The alternatives below are contingency narratives, not alternative research directions.

---

## Alternative A: "BN+AdamW as the Joint Mechanism for WD Invariance" (Pivot if Gate 1 Path B)

**Trigger**: Gate 1 (BN ablation) shows ResNet-20-NoBN breaks Phi Invariance under AdamW.

**Repositioning**: The finding becomes "BN and AdamW jointly create WD invariance; neither alone is sufficient." This is a genuinely novel mechanistic finding about the BN-AdamW interaction, directly building on D'Angelo (2024) with a precise experimental separation.

**Key contribution**: First experiment that cleanly separates the BN mechanism from the AdamW mechanism for WD irrelevance. D'Angelo (2024) shows WD irrelevance for BN+Adam but does not isolate the mechanisms.

**Paper framing**: "Two Co-Mechanisms of Weight Decay Inefficacy: Batch Normalization Scale-Invariance and AdamW's Implicit ℓ∞ Constraint"

**Experiment coverage**: The Gate 1 experiment (18 runs) directly provides the pivot data. No additional experiments needed beyond what is already planned.

**Score estimate**: 6.5–7.0 (reduced novelty vs. front-runner, but still publishable with the clean experimental separation).

---

## Alternative B: "ρ Invariance is Universal at All Tested Values Under AdamW" (Pivot if P1-1 Flat)

**Trigger**: P1-1 (λ sweep) shows spread < 0.5% even at ρ = 50 (λ = 5e-2).

**Repositioning**: The Trichotomy Conjecture is empirically falsified — invariance is stronger than theory predicts. This becomes: "AdamW's ℓ∞ constraint is so robust that even at ρ = 50 (100× standard WD), dynamic WD strategies remain equivalent." This is an unexpected, counterintuitive result.

**Key contribution**: Practical guidance for practitioners: WD strategy choice is irrelevant under AdamW at all tested settings. The only decision that matters is whether to use WD at all (and only under SGD).

**Paper framing**: "Dynamic Weight Decay Scheduling is Irrelevant Under AdamW at All Practical Settings: A Comprehensive Benchmark"

**Score estimate**: 6.5–7.0 (strong empirical statement, but negative result with limited algorithmic novelty).

---

## Alternative C: "ρ-Controller as Stand-Alone Algorithm Paper" (Pivot if Regime II Confirmed + Invariance Fails)

**Trigger**: P1-1 confirms spread > 1% at ρ = 5 AND P1-2 shows ρ-Controller improves accuracy by ≥ 0.5% at Regime II settings.

**Repositioning**: Instead of a "when does WD matter" paper, pivot to an algorithms paper introducing the ρ-Controller as a practical improvement over constant WD at high-ρ settings.

**Key contribution**: First computationally-free closed-loop WD algorithm with Lyapunov convergence backing. Demonstrated improvement in Regime II (high-λ) settings.

**Paper framing**: "ρ-Controller: Feedback-Stabilized Weight Decay via Gradient-to-Weight Ratio Targeting"

**Experiment coverage**: Requires P1-1 + P1-2 + VGG + ImageNet validation at Regime II settings.

**Score estimate**: 7.0–7.5 (if ρ-Controller improvement is robust and reproducible).

---

## Alternative D: "Architecture-Conditional WD Invariance: VGG Breaks the Pattern" (Pivot if VGG shows WD matters under AdamW)

**Trigger**: VGG-16-BN at n=3 confirms no_wd > constant under AdamW (as hinted by 1-seed pilot: +0.67%).

**Repositioning**: VGG-16-BN exhibits active WD harm under AdamW — inverse of the ResNet-20 pattern. This becomes a characterization of which architectural properties determine WD sensitivity.

**Key contribution**: Identifies VGG's global average pooling / fully-connected head as an architectural feature where WD hurts under AdamW. Provides actionable guidance: avoid WD on fully-connected layers when using AdamW.

**Paper framing**: "When Weight Decay Helps and When It Hurts: Architecture-Conditional Analysis Under AdamW"

**Score estimate**: 7.0 (strong empirical finding if confirmed at n=3; requires mechanistic explanation from BN structure analysis).

---

## Strategic Assessment

| Alternative | Trigger Event | Probability | Estimated Score |
|---|---|---|---|
| Front-runner (Trichotomy + ρ-Controller) | No trigger | 55% | 7.5–8.0 |
| Alt-A (BN+AdamW joint mechanism) | Gate 1 Path B | 25% | 6.5–7.0 |
| Alt-B (Universal invariance) | P1-1 flat | 20% | 6.5–7.0 |
| Alt-C (ρ-Controller stand-alone) | Regime II confirmed | 30%* | 7.0–7.5 |
| Alt-D (VGG breaks invariance) | VGG confirms pilot | 25% | 7.0 |

*Alt-C is not mutually exclusive with the front-runner — it can be a paper pivot IF the front-runner's invariance findings are less interesting than the Regime II algorithmic results.

All alternatives lead to publishable outcomes. The research direction has converged; remaining work is experimental validation determining which narrative is strongest.
