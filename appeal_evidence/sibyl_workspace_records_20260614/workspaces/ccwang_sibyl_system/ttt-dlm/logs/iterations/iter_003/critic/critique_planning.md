# Planning Critique (Iteration 3 — Post-DTA Full-Scale Results)

**Reviewer**: Critic Agent
**Date**: 2026-03-10
**Scope**: `plan/methodology.md`, `plan/task_plan.json`, `plan/pilot_plan.json`

---

## Overall Assessment: 5.5/10

The methodology document is well-structured and thorough in its coverage of experimental design, evaluation metrics, and hyperparameter spaces. The 8-phase experimental plan was ambitious but largely feasible. However, the planning stage made two fatal errors: (1) it did not include an early falsification checkpoint for DTA's core hypothesis before committing to full-scale evaluation, and (2) it did not elevate DMI from "ablation baseline" to a first-class method deserving systematic investigation.

---

## CRITICAL Issues

### P1. No Early Falsification Protocol for DTA

The methodology prescribes 8 phases, with DTA core implementation in Phase 2 and full-scale validation in Phase 5. The plan includes a "Countdown 16-question pilot to verify numerical stability and basic effectiveness." But "basic effectiveness" at N=16 is meaningless---the paper itself documents that pilot effects diverge by up to 24pp from full-scale.

A proper falsification protocol would have been:
1. Phase 2 pilot (N=16): verify numerical stability only
2. **Phase 2.5 (NEW): intermediate validation (N=100, 1 seed) on Countdown**: if DTA < vanilla + 2pp, immediately pivot to DMI-focused investigation before investing in Phase 5-8

This checkpoint would have saved approximately 40+ GPU-hours spent on full-scale DTA evaluation that produced null results.

### P2. DMI Not Given Its Own Systematic Evaluation Plan

The methodology treats DMI as "Level 1" of the information augmentation spectrum, with a brief description and no dedicated ablation plan. DMI's hyperparameters (mixing weight alpha, temperature tau_soft, lookback depth) are mentioned in the method section but not in the experimental plan.

Given that DMI turned out to be the only effective method:
- There should be a full DMI ablation (alpha sweep, tau sweep, multi-step lookback)
- DMI should be evaluated on all benchmarks at full scale, not just Countdown
- DMI should be combined with remasking methods (DMI+ReMDM-conf) to test orthogonality

This planning oversight means the paper's strongest result has the least systematic experimental support.

---

## HIGH Issues

### P3. Phase 5-8 Were Designed Assuming DTA Success

The experimental plan allocates Phases 5-7 to full-scale DTA evaluation and ablation. These phases assume DTA works and focus on characterizing *how well* it works. There is no contingency plan for DTA failure at full scale.

The alternatives document (`alternatives.md`) does provide pivot options, but the methodology itself doesn't incorporate decision points. A better plan would have explicit decision gates:

```
Phase 2 complete → DTA effect > 2pp?
  YES → proceed to Phase 5 (full-scale DTA)
  NO → redirect to Phase 5-DMI (full-scale DMI evaluation + ablation)
```

### P4. MBPP/HumanEval/GSM8K Full-Scale Plan Never Executed

The methodology specifies:
- GSM8K: "full test set 1,319 problems"
- MBPP: "500 problems"
- HumanEval: "164 problems"

Actual execution:
- GSM8K: partial (1300/1319 for vanilla, 900/1319 for DTA, much less for others)
- MBPP: pilot only (N=16)
- HumanEval: not started

This leaves the paper with only one benchmark at full scale. For a methods paper claiming "task-dependent effectiveness," single-benchmark full-scale data is insufficient.

### P5. Statistical Testing Plan Not Connected to Decision Making

The methodology specifies McNemar + Bootstrap CI + Bonferroni correction. But these tests are planned only for the final reporting stage, not for intermediate decision-making. If interim statistical tests had been run after Phase 5 wave 1 (when DTA Countdown results became available), the team would have known immediately that DTA = vanilla and could have redirected effort.

---

## MEDIUM Issues

### P6. Compute Budget Was Reasonable But Allocation Was Suboptimal

The plan estimated 84 GPU-hours total. Actual allocation:
- DTA full-scale Countdown: ~30 GPU-hours (produced null result)
- DMI full-scale Countdown: ~10 GPU-hours (produced the paper's best result)
- SCP full-scale Countdown: ~60 GPU-hours (produced a result equivalent to DMI at 7x cost)
- DTA ablations at pilot scale: ~20 GPU-hours (ablating a method that doesn't work)

In hindsight, reallocating the SCP and DTA compute to DMI cross-benchmark evaluation would have been far more productive.

### P7. Missing Compute-Matched Baselines

The plan does not include compute-matched baselines. DTA costs ~4-6x vanilla per sample. A fair comparison would include "vanilla with 4x more denoising steps" (T=512 instead of T=128) or "Best-of-4 vanilla" (generate 4 samples, select best by confidence). If these compute-matched baselines outperform DTA, then DTA's overhead is not justified even if it showed improvement.

### P8. Pilot Design Lessons From Previous Iterations Not Applied

The previous 18 iterations established that N=16 pilots are unreliable. The methodology was written after this lesson was known, yet it still uses N=16 pilots for DTA validation. A minimum of N=100 for intermediate validation should have been specified.

---

## Positive Notes

1. **Comprehensive method coverage**: 7 methods across 3 groups (baselines, information augmentation, combined) provides good coverage of the design space.
2. **Clear hyperparameter specification**: Default values, sweep ranges, and rationales are well-documented.
3. **Reproducibility provisions**: Fixed seeds, saved generation text, version logging, and JSON result format are excellent.
4. **Phase structure**: The 8-phase plan provides logical progression from validation to full-scale evaluation.
5. **Resource awareness**: GPU allocation and wall-clock estimates are realistic and well-calibrated.
