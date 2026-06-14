# Experiment Critique

## What Is Already Credible

- GSM8K does support one honest-compute reorder and a clear warning that nominal step counts hide real runtime differences.
- HumanEval does support a narrow structural-boundary claim: syntax can improve while execution still gets worse.

## Main Experimental Blockers

### 1. Runtime Fairness Is Still Implementation-Confounded

- `CORE-proxy-64` is a non-official proxy implementation with `batch_size=1` and `compile_enabled=false`.
- `Standard-64` and `DNB-84` run compiled and heavily batched.
- This is informative as realized-pipeline evidence, but it is not yet a clean algorithm-only comparison.

### 2. The Signal Audit Does Not Yet Answer the Proposal’s Mechanism Question

- The proposal asks about prediction of revision benefit, not only error correlation.
- The current audit reports error-facing diagnostics plus ad hoc control deltas, but not harmed/fixed/no-effect buckets.
- Calibration has no deployed controller and no benefit-correlation measurement in the current asset.

### 3. The Task-Dependence Axis Is Underpowered

- `exp/results/diag_math500_shortlist.json` is single-seed, `n=100`, and all methods are packed into a narrow `0.19-0.23` band.
- That is enough for a warning shot, not enough for a strong regime claim.

### 4. The Code Boundary Evidence Is Still Too Coarse

- `exp/results/diag_humaneval_guard_boundary.json` uses only `n=50`.
- It merges `Entropy/TIGER Ungated Revision` into one row, which blocks method-specific diagnosis.

### 5. There Is No Uncertainty Accounting

- No multi-seed robustness
- No paired significance tests
- No bootstrap confidence intervals on the key ranking differences

## Required Evidence Upgrades

- Produce `benefit_bucket_audit.json` with harmed, fixed, and no-effect counts for the main methods.
- Run a minimal `seed_sensitivity_spotcheck` on the headline comparisons, ideally three seeds plus paired tests.
- Decide whether the honest-compute claim is about realized pipelines or matched implementations, and then rerun or reframe accordingly.
- Document why `batch_size=1` and `compile_enabled=false` were unavoidable for `CORE-proxy`, or stop using it as a fairness-sensitive comparator.
- Split entropy and TIGER on HumanEval unless they are proven empirically interchangeable.

## Bottom Line

- The paper has enough evidence for a careful diagnostic correction paper.
- It does not yet have enough evidence for a strong mechanism paper or a strong task-regime paper.
