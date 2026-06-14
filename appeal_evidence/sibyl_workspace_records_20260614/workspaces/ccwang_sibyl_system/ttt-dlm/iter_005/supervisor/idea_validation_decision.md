# Idea Validation Decision: Denoising-as-Learning (DaL) -- Round 2

**Evaluator**: sibyl-idea-validation-decision
**Date**: 2026-03-11
**Iteration**: 5, Round 2

---

## Evidence Summary

### Pilot Results (Complete)

| Pilot | Status | Key Finding |
|-------|--------|-------------|
| Setup (env) | PASS | Both Dream-7B and LLaDA-8B verified; 98GB VRAM per GPU |
| impl_ttt_layer | PASS | 12/12 unit tests; Linear, MLP, Momentum all functional |
| P1: Feasibility | PASS | SSL loss -33% (raw), -53% (meta-trained), -72% (post-meta); grad norms <3.5 |
| P2: Signal Quality | PASS | Gradient SNR peaks at r=0.6, degrades at r>0.7; critical zone [0.4-0.6] confirmed |
| P3: Quick Eval (GSM8K-200) | **FAIL** | DaL-MLP 39.5% vs Vanilla 40.5% (-1.0pp); gate stuck at 0.007 |

### Root Cause Analysis of P3 Failure

The failure mode is highly specific and well-diagnosed:

1. **Gate initialization too conservative**: sigmoid(-5) = 0.007 means the TTT output contributes <1% to backbone activations. The TTT layer learns internally (SSL loss -52%) but this learning is silenced by the gate before reaching the backbone.

2. **Insufficient training compute**: Only 1K meta-training steps were used in P3. SSL loss was still declining at termination. The proposal's own failure matrix prescribes "increase to 5K-10K steps before pivoting."

3. **No phase-transition scheduling applied**: P2 clearly showed SNR degrades at mask ratio >0.7, yet P3 applied TTT at all 127 denoising steps, injecting noisy gradient updates at high mask ratios.

These are **engineering failures**, not fundamental theoretical incompatibilities. The mechanism (TTT fast weights learning from revealed tokens) works; the coupling (gate injection into backbone) and training budget were inadequate.

---

## Candidate Comparison

### cand_dal (DaL -- front runner)

**For advancing now:**
- P1, P2 demonstrate the core mechanism is sound
- Root cause of P3 failure is specific and addressable (gate init + training budget)
- The proposal already contains a concrete Phase 0 remediation plan with explicit GO/NO-GO gates
- 98GB VRAM is ample; overhead is only 1.2x

**Against advancing now:**
- P3 is a hard FAIL on the critical effectiveness gate
- Gate repair is untested -- it might not work
- SSL-task alignment (H_align) is unverified -- the 52% SSL improvement could be structurally decorrelated from task accuracy
- MetaState-GRU also underperforms vanilla (43.75% vs 50.0% at n=16), raising paradigm-level doubt

**Verdict**: Cannot advance. The P3 failure is disqualifying for full experiments. But the failure has a specific, testable root cause.

### cand_info_gain (Alternative A -- backup)

**Strengths:**
- Training-free (zero GPU-hours for training)
- Literature reports 3.6% avg improvement (Yang et al. 2026)
- A-CFG pilot showed +12.5pp on n=16 (directionally strong but statistically unreliable)
- 55% success probability estimate

**Weaknesses:**
- Low novelty (Info-Gain Sampler already published)
- n=16 A-CFG result has 4+ reversal precedents in project history
- Not yet piloted in this iteration

**Verdict**: Strong backup. Should run in parallel on day 1 regardless of DaL decision.

### cand_diagnostic (Alternative D -- fallback)

**Strengths:**
- 75% success probability (highest of all candidates)
- 22+ iterations of systematic negative results constitute genuine scientific value
- Requires only 10-15 GPU-hours of gap-filling analysis

**Weaknesses:**
- Negative-results paper requires exceptionally clear mechanistic explanations
- Lower venue ceiling (EMNLP/Findings vs NeurIPS)

**Verdict**: Insurance policy. Becomes primary if both DaL and Alternative A fail.

### cand_enhanced_optimizer (Alternative B -- dropped)

**Verdict**: Correctly dropped. MetaState-GRU itself underperforms vanilla; enhancing it is premature.

---

## Decision Reasoning

The critical question is whether to REFINE (try gate repair + extended training) or PIVOT (abandon DaL for Alternative A).

### Evidence favoring REFINE over PIVOT:

1. **Root cause is specific and testable**: The gate stuck at 0.007 is a ~10 line code change (init + lr). If this simple fix resolves P3, the entire DaL framework is vindicated. The proposal has already identified this fix.

2. **Failure matrix not exhausted**: The proposal's own decision tree states "P1 PASS + P2 PASS + P3 FAIL -> increase training compute (5K-10K steps) or try LLaDA-8B before pivoting." Neither has been done.

3. **D0c diagnostic provides a definitive test**: H_align (Pearson r between SSL loss and task accuracy) will answer the structural question of whether the SSL objective is fundamentally misaligned. This costs only 4-6 GPU-hours and provides a binary GO/NO-GO signal.

4. **Parallel Alternative A eliminates waste**: The plan already includes running Alternative A on a separate GPU from day 1. REFINE does not mean ignoring alternatives.

### Evidence favoring PIVOT over REFINE:

1. **History of failure**: 18+ rounds of iteration on DLM inference-time methods with no positive result at scale. This project has a strong base rate of failure.

2. **Paradigm-level doubt**: MetaState-GRU underperforming vanilla (even at n=16) suggests the entire "insert lightweight memory into frozen DLM" approach may be counterproductive.

3. **SSL-task decorrelation risk**: The 52% SSL improvement with -1.0pp task accuracy is concerning. If H_align shows r < 0.1, no gate fix will help.

### Weighing the evidence:

The REFINE path costs ~2 days and ~16 GPU-hours, with a clear kill switch (D0c). The PIVOT path abandons a well-diagnosed engineering problem before testing the fix. The optimal strategy is REFINE with strict time-boxing and parallel alternative execution.

**However**, the confidence in REFINE succeeding is low (30-35%). The plan must be structured so that REFINE failure automatically transitions to Alternative A without additional decision latency.

---

## Decision

**REFINE** -- The DaL framework's core mechanism is validated (P1, P2), but P3 failure blocks advancement to full experiments. The failure has a specific, testable root cause (gate initialization at sigmoid(-5) = 0.007) with a concrete remediation plan. Before pivoting, three targeted refinements must be tested:

1. **Gate repair** (highest priority): Initialize at sigmoid(-2) = 0.12, independent gate_lr = 10x meta_lr, optional warm-up loss
2. **D0c target alignment diagnostic**: Measure SSL-task correlation across 10+ configurations (binary GO/NO-GO for DaL)
3. **Extended training**: 5K-10K steps (vs 1K in P3) with phase-transition scheduling

These refinements are time-boxed to 2 days. Alternative A runs in parallel from day 1. If D0c correlation < 0.1 OR gate repair + 10K training still shows no improvement, PIVOT to Alternative A immediately.

### Strict GO/NO-GO Gates for Phase 0

| Gate | Criterion | Action if FAIL |
|------|-----------|----------------|
| D0c | Pearson r(SSL, accuracy) > 0.3 | PIVOT to Alternative A immediately |
| H_gate | Gate value >= 0.10 by 2K steps | Try sigmoid(0) init + warm-up loss; if still <0.05, PIVOT |
| Phase 0 combined | D0c PASS + H_gate PASS | Proceed to Phase 1 training (10K steps) |
| Phase 1 eval | DaL >= vanilla + 2% on 2/3 benchmarks | ADVANCE to full experiments |
| Phase 1 fail | DaL < vanilla + 2% after all remediation | PIVOT to Alternative A or D |

### Why Not ADVANCE

P3 is a hard NO-GO. No candidate should advance to full experiments (80+ GPU-hours) with a -1.0pp accuracy delta, regardless of how solid the theoretical foundations are.

### Why Not PIVOT

The failure analysis pathway prescribed by the proposal itself has not been exhausted. Gate repair is a ~10 line change that has not been tested. D0c will provide definitive evidence for or against the SSL-task alignment assumption. Pivoting before testing these would be premature.

---

SELECTED_CANDIDATE: none
CONFIDENCE: 0.35
DECISION: REFINE
