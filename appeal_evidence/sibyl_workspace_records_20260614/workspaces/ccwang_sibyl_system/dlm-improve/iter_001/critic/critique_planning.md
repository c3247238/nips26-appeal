# Planning Critique

## What the Plan Gets Right

- It correctly pivots away from TIGER hero framing.
- It correctly assigns benchmark roles: GSM8K as headline, MATH500 as transfer, HumanEval as boundary.
- It correctly treats runtime metadata as part of the scientific claim.

## Where Execution Missed the Plan

### 1. The Batch-Size Rule Was Violated Without a Recorded Exception

- The methodology says formal experiments should not accept `batch_size=1` unless a blocker is explicitly recorded.
- `exp/results/baseline_core.json` still runs at `batch_size=1` and does not record such a blocker.

### 2. The Plan Asked for Signal-vs-Benefit Analysis, but the Delivered Audit Is Still Mostly Signal-vs-Error

- The methodology asks for benefit-oriented signal analysis.
- `exp/results/diag_signal_gap_audit.json` does not yet provide the paired bucket evidence needed for that promise.

### 3. The Plan Still Lacks a Concrete Uncertainty Work Package

- Single-seed risk was already known during ideation and review.
- Yet the task plan still does not contain a mandatory seed-robustness task for the headline claims.

### 4. Canonical Asset Discipline Is Too Weak

- `exp/results/math500_transfer.json` says MATH500 was skipped.
- `exp/results/diag_math500_shortlist.json` is the real transfer asset used by the manuscript.
- That ambiguity should not survive into a paper whose contribution is partly protocol discipline.

## Planning Corrections

- Add a P0 task for `benefit_bucket_audit.json`.
- Add a P0 task for `seed_sensitivity_spotcheck.json`.
- Add a canonical asset manifest declaring exactly which JSON supports each paper claim.
- Split the honest-compute story into two explicit questions:
  - realized pipeline cost
  - matched implementation fairness
- Freeze further method search until claim-evidence alignment is repaired.

## Bottom Line

- The planning logic is good.
- The planning closure is not yet good enough for a protocol paper, because the missing bucket audit, missing uncertainty check, and weak asset discipline are exactly the kinds of gaps this paper claims to correct.
