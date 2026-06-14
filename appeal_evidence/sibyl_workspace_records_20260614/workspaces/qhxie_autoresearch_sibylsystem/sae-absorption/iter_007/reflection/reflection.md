# Iteration 8 Reflection Report

**Project**: SAE Feature Absorption (sae-absorption)
**Iteration**: 8
**Date**: 2026-04-14
**Score**: 6.5 / 10 (Supervisor review.json)
**Dimension Scores**: Novelty 7, Soundness 5.5, Experiments 6, Reproducibility 5.5
**Verdict**: CONTINUE
**Quality Trajectory**: Stagnant (5.5 x3 -> 6.5 -> 6.0 -> 6.5 -> 6.5 -> **6.5**)

---

## 1. Iteration Summary

Iteration 8 produced **zero changes**. The paper.md is byte-identical to the iter_007 version. No experiments were executed, no writing revisions were made, no data pipeline fixes were implemented, and no zero-GPU analyses were computed. The score remains at 6.5 for the **third consecutive review** (iter 6, 7, 8).

This is the most severe stagnation in the project's history. Previous stagnation (iter 0-3 at 5.5) was broken by a strategic pivot and experimental execution in iter 4 (+1.0 to 6.5). The current stagnation (iter 6-8 at 6.5) requires the same intervention: executing the three critical experiments that have been identified as blocking since iter 6.

**What changed in iter 8**: Nothing. The review cycle ran on an unchanged paper.

**What did NOT change (now spanning 2+ iterations)**:
- Three critical experiments unexecuted (activation patching, tightened hedging, CMI at L0=22) -- 3rd consecutive review requesting
- validate_integration.py not implemented -- 8th consecutive iteration recommending
- Theoretical overclaiming not fixed -- 3rd consecutive review flagging "CMI predicts" language
- Zero-GPU analyses not computed (partial correlation, leave-one-out, threshold sensitivity reporting) -- 3rd consecutive review requesting
- Two-interpretation ambiguity not resolved
- Four of nine core words unnamed
- Confound decomposition file contradiction unaddressed
- Title, abstract, Section 5.3, notation all unchanged

---

## 2. Issue Analysis by Category

### EXPERIMENT (6 issues, all recurring for 3+ iterations)

The three "blocking experiments" remain the project's single most impactful investment:

| Experiment | GPU Cost | Info Gain | Reviews Requesting | Status |
|-----------|---------|-----------|-------------------|--------|
| Activation patching (9 core words) | 1h | Resolves central ambiguity | **3** | Never executed |
| Tightened hedging (parent-specific) | 1h | Validates headline 98.6% | **3** | Never executed |
| CMI at L0=22 (F1=1.0 probes) | 1h | Tests theoretical pillar | **3** | Never executed |

Additionally:
- **Threshold sensitivity results** (141KB computed data) unreported for 3 reviews -- zero cost
- **Control failure diagnosis** (analytical computation) never attempted -- zero GPU
- **CMI estimation diagnostics** (bootstrap CIs, convergence, k-sensitivity) absent -- 1 hour

The total cost to resolve ALL six experiment issues: 3 GPU-hours + 2 hours analysis. The expected score improvement: +1.0 to +1.5 points.

### SOUNDNESS (5 issues, all recurring)

1. **Hedging classification tautological** (3rd review): 98.6% is design-guaranteed upper bound. At L0=176, 99.2% tokens trivially resolve. No parent-specific verification. Supervisor: "near-tautological."

2. **Theoretical overclaiming** (4th iteration in pattern): Section 6 title still "CMI Predicts" from p=0.236 Bonferroni. Whack-a-mole pattern: each iteration fixes PREVIOUS overclaiming but current one persists. Root cause: incentive structure biases toward stronger claims.

3. **Two-interpretation ambiguity** (3rd review): "Metric does not transfer" conflates miscalibration with genuine absence. Paper favors one interpretation without discriminating evidence.

4. **Four unnamed core words** (2nd review, NEW in iter 7 review): hierarchy_details contains only 5 of 9 words. Data pipeline gap creates transparency vulnerability.

5. **Confound decomposition contradiction** (2nd review): 96.9% vs 1.4% hierarchy-driven at L0=22 from different classification criteria. Never reconciled in paper body.

### ANALYSIS (3 issues, all recurring for 3+ iterations)

1. **Probe quality confound uncontrolled** (3rd review): rho=-0.67 between absorption and probe F1. Partial correlation never computed. ZERO GPU, 30 minutes.

2. **Data pipeline residuals** (8th iteration): validate_integration.py never implemented. Companion documents still contain wrong numbers. The project's oldest systemic issue.

3. **Leave-one-out sensitivity** (3rd review): Letters S and K as outliers never analyzed. ZERO GPU, 30 minutes.

### WRITING (6 issues, all recurring)

All writing issues from iter 7 persist unchanged: misleading title, self-contradictory cross-domain novelty claim, confounded formal tests in Section 5.3, overlength abstract (~290 words), structural redundancy (~300 words wasted), notation inconsistencies in companion documents.

### PLANNING (1 issue, critical, recurring)

**Zero experiments for 2 consecutive iterations** is the project's most critical process failure. The system has been in a writing-review loop without experimental input. The binding constraint on score improvement is experiments (3 GPU-hours), not prose quality (already 7/10). Each empty review cycle consumes resources without progress.

### PIPELINE (1 issue, recurring for 8 iterations)

**validate_integration.py** has been recommended in every single reflection report since iteration 1. It has never been implemented. The cost is 1.5 hours and 0 GPU. The benefit is eliminating the data integrity failure pattern that has occurred in every iteration. This is the most severe priority inversion in the project.

---

## 3. Resource Efficiency Assessment

### GPU Utilization
- **Iter 8 GPU time**: 0 minutes (zero experiments executed -- 2nd consecutive iteration)
- **Iter 8 CPU time**: 0 minutes (zero analyses computed)
- **Available GPU budget**: 4 GPUs, unlimited polling
- **Utilization**: 0% -- tied with iter 7 for most underutilized iteration

### Bottleneck Analysis

The bottleneck is **execution priority**, not resources. Four GPUs are available. The three blocking experiments total 3 GPU-hours. The zero-GPU analyses total 2 hours. The system has spent 2 full review cycles (iter 7 + iter 8) without executing ANY of these tasks. The marginal value of additional reviews on an unchanged paper is zero.

### Cost-Benefit Analysis

| Investment | Cost | Expected Score Impact |
|-----------|------|----------------------|
| Zero-GPU analyses (Gate 0) | 3.5 hours | +0.25 |
| 3 critical experiments (Gate 1) | 3 GPU-hours | +0.75 to +1.0 |
| Writing revision (Gate 2) | 3 hours | +0.25 to +0.5 |
| **Total** | **3 GPU-hrs + 6.5 hours** | **+1.0 to +1.5** |

The path from 6.5 to 8.0 has been stable and explicitly described for 3 consecutive reviews. The total investment (~10 hours including 3 GPU-hours) is less than the time consumed by the last 2 empty review cycles.

### Scheduling Improvement

Iter 9 must enforce a gated execution model:
1. **Gate 0**: validate_integration.py + zero-GPU analyses (3.5 hours) -- BLOCKING
2. **Gate 1**: Three critical experiments (3 GPU-hours) -- BLOCKING for writing
3. **Gate 2**: Writing revision incorporating all new results (3 hours)
4. **Gate 3**: Review cycle

**Hard constraint**: No writing task may start before Gate 1 completes. The planner must encode this as an explicit dependency. This prevents the writing-loop-without-evidence pattern.

---

## 4. Quality Trend Assessment

```
Iter 0-3: 5.5, 5.5, 5.5, 5.5 -- Stagnant (PILOT mode, proxy models, pipeline errors)
Iter 4:   6.5                   -- Breakthrough (+1.0, strategic pivot + experiments)
Iter 5:   6.0                   -- Regression (-0.5, causal overclaiming)
Iter 6:   6.5                   -- Recovery (+0.5, metric audit framing)
Iter 7:   6.5                   -- Stagnant (writing-only, no experiments)
Iter 8:   6.5                   -- Stagnant (ZERO CHANGES, empty review cycle)
```

**Trajectory: STAGNANT** at 6.5 for 3 consecutive reviews.

**Key insight**: Every score improvement in this project's history was driven by experimental execution. The +1.0 jump in iter 4 came from strategic pivot + full-scale experiments. The recovery in iter 6 came from new experimental framing. Writing-only iterations (iter 7, 8) produce exactly +0.0. The paper is evidence-bound, not prose-bound.

**Prediction**: If iter 9 executes all three critical experiments, the score will reach 7.5-8.0 regardless of writing quality. If iter 9 is another writing-only pass, the score will remain 6.5 for a 4th consecutive review.

---

## 5. Root Cause Analysis

### Why has the system been stagnant for 2 iterations?

**Primary cause: Execution pipeline skips experiment phase.** The breadcrumb progression shows the system moves through writing -> review -> reflection stages without an experiment phase. The three critical experiments were identified as blocking in iter 6's reflection and lessons_learned. They appeared in iter 7's reflection as P1 priorities. They appeared again in iter 8's review. Yet no experiment code was written and no experiments were queued.

**Secondary cause: No enforcement mechanism for priority gates.** The lessons_learned correctly lists experiments as P1 blocking priorities. But nothing in the execution pipeline enforces this -- writing and review can proceed without experiments completing. The gate structure is advisory, not mandatory.

**Tertiary cause: Writing-review has lower activation energy.** Running experiments requires: writing code, SSH to GPU server, managing execution, collecting results. Writing revision requires: editing paper.md. The path of least resistance is always writing revision.

### Why has validate_integration.py not been implemented for 8 iterations?

**Priority inversion.** The script produces no visible output (prevents invisible errors). Writing and review produce visible output (paper revisions, review scores). The execution pipeline consistently prioritizes visible output over invisible prevention. The fix: make validate_integration.py a hard blocking dependency (Gate 0) that must complete before any other task.

### Why does theoretical overclaiming recur?

**Structural incentive bias.** Stronger claims make more publishable narratives. The writing agent optimizes for publication impact, which creates pressure toward overclaiming. Each iteration fixes PREVIOUS overclaiming when explicitly flagged but introduces new overclaiming or fails to fix current instances. A systemic fix: automated overclaiming check that flags words like "predicts," "criterion," "validates," "confirms" when they appear near non-significant p-values (p > 0.05 corrected).

---

## 6. Fix Tracking (Compared to Previous Action Plan)

### Issues from Iteration 7 Reflection

| Issue ID | Description | Status |
|----------|------------|--------|
| EXP001 | Activation patching on 9 core words | **RECURRING (3rd iteration)** |
| EXP002 | Tightened hedging classification | **RECURRING (3rd iteration)** |
| EXP003 | CMI replication at L0=22 | **RECURRING (3rd iteration)** |
| SND001 | Hedging classification near-tautological | **RECURRING** |
| SND002 | Theoretical overclaiming persistent | **RECURRING (4th iteration in pattern)** |
| SND003 | Two-interpretation ambiguity | **RECURRING** |
| EXP004 | Threshold sensitivity unreported | **RECURRING (3rd iteration)** |
| ANL001 | Probe quality confound in CMI | **RECURRING (3rd iteration)** |
| EXP005 | Control failure undiagnosed | **RECURRING (3rd iteration)** |
| DATA001 | validate_integration.py not implemented | **RECURRING (8th iteration)** |
| WRI001 | Title misleading | **RECURRING** |
| WRI002 | Cross-domain novelty self-contradiction | **RECURRING** |
| WRI003 | Section 5.3 confounded comparison | **RECURRING** |
| WRI004 | Four unnamed core words | **RECURRING** |
| WRI005 | Abstract overlength | **RECURRING** |
| ANL002 | Leave-one-out sensitivity missing | **RECURRING** |
| EXP006 | CMI estimation diagnostics missing | **RECURRING** |
| SND004 | Confound decomposition contradiction | **RECURRING** |
| PIPE001 | Zero experiments executed | **RECURRING (2nd iteration)** |
| PIPE002 | validate_integration.py (duplicate of DATA001) | **RECURRING** |

**Fixed in iter 8**: 0 issues.
**Recurring**: 20 of 20 issues (100%).
**New**: 0 issues.

This is a complete fix-rate failure. Not a single issue from the previous reflection was addressed.

---

## 7. System Self-Check Response

No `logs/self_check_diagnostics.json` found. No targeted self-check measures required.

---

## 8. Success Patterns (Preserve These)

1. **Honest negative results reporting** (8 consecutive iterations): H2/H4/H6/H7 reported with specific pre-registered targets vs actual results. Consistently the paper's strongest aspect. Non-negotiable quality standard.

2. **L0 phase transition** (most robust finding): 42.9% -> 0.8%, cross-layer CV < 10%, bootstrap CIs. Zero reviewer challenges. Directly actionable.

3. **Comprehensive control suite**: 4 controls x 5 domains. Universal control failure is devastating evidence.

4. **Infrastructure reliability**: Zero experiment failures for 4 consecutive iterations when experiments ARE executed.

5. **Strategic pivot capability**: System pivoted from epidemiological methods to metric audit based on evidence. The pivot was correct.

6. **Bootstrap CIs throughout**: All absorption rates with 95% CIs (10k resamples). Best practice.

7. **Per-letter granularity**: Enabled discovery of probe quality confound (rho=-0.67).

8. **Partial data integrity fix** (from iter 7, held): Paper body CMI numbers now match source JSON.

---

## 9. Critical Assessment: Is the Stagnation Recoverable?

**Yes, with 3 GPU-hours.** The stagnation is not caused by an intractable problem. The three blocking experiments are well-specified, computationally cheap, and guaranteed to produce publishable results regardless of outcome:

- **Activation patching**: Positive result = competitive exclusion confirmed at small scale. Negative result = all-hedging narrative validated. Both advance the paper.
- **Tightened hedging**: High strict rate = headline validated. Low strict rate = narrative changes but finding is equally publishable as "permissive classification inflates hedging estimates."
- **CMI at L0=22**: Significant result = theoretical pillar secured. Non-significant = cleanly downgrade to exploratory, paper still stands on two empirical pillars.

The risk is not scientific but procedural: will the system execute experiments in iter 9, or will it enter a 4th consecutive empty review cycle? The reflection's primary recommendation is: the planner MUST encode experiment execution as a hard blocking gate.
