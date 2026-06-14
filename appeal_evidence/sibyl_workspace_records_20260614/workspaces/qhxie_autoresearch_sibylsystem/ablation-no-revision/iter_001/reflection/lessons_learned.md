# Lessons from This Iteration

## Must Improve

1. **Paper-code consistency verification**: Before finalizing any paper section that describes a formula or computation, the writer MUST inspect the actual experiment code to verify the formula matches the implementation. The Ea = -ln(c0) in the paper vs Ea = 1/c0 in the code was a catastrophic mismatch caused by insufficient cross-checking.
   - Solution: Add a mandatory "code verification" step in the writing pipeline where the writer reads the analysis script and confirms formulas match.

2. **Hypothesis evaluation logic must use ALL relevant metrics**: The H3 evaluation checked only threshold pass (75% accuracy) but ignored AUC=0.436 and Spearman=-0.063, leading to a "CONFIRMED" label that directly contradicts the paper's narrative.
   - Solution: Update evaluation templates to require ALL metrics to meet criteria for confirmation, or explicitly document which metric overrides others.

3. **Variable naming must be unambiguous**: The paper's c0 (consistency fraction at k=16) and code's c0 (fitted saturation parameter, bounds [0.1, 10.0]) are completely different quantities with the same name.
   - Solution: Use distinct names for distinct quantities (e.g., c_consistency for the fraction, tau_sat for the fitted parameter).

4. **Pilot gates before full experiments**: The g1_saturation task failed after 200 minutes due to answer extraction issues that a 5-minute pilot would have caught.
   - Solution: Every experiment task must include a pilot subtask (n=5-10) that runs first and validates extraction, formatting, and basic correctness before the full run.

5. **Bibliography must be created BEFORE citing**: The paper cites [1]-[11] but has no References section. Some citations ("Li 2026", "ACAR 2026") may not exist.
   - Solution: Add a "reference verification" step that requires every citation to have a verified bibliographic entry before it can be used in the paper.

## Watch Out

- **Post-hoc optimization disguised as discovery**: The 75% threshold pass and 25pp "error floor" are post-hoc artifacts. Always label post-hoc results as upper bounds, not measured quantities.
- **Small sample sizes labeled as "full"**: n=50 is pilot-scale. Be honest about sample size limitations and report confidence intervals.
- **Scope creep in paper structure**: The outline promised H4/H5 analyses that were not delivered. Match paper structure to actual completed work.
- **Aggregate metrics masking individual failure**: Aggregate R2=0.924 looks good but median per-problem R2=0.000. Always report both aggregate and per-problem statistics.
- **Level 5 Ea clustering suggests algorithmic saturation**: When fitted parameters hit their bounds (c0 lower bound = 0.1), the derived metric (Ea = 1/c0) will artificially cluster. Report raw fitted parameters alongside derived metrics.

## Keep Doing (success patterns)

- **Honest reporting of negative results**: The AUC=0.436 finding and per-problem fit failure were reported without spin. This is scientifically admirable and has been consistent across 6 consecutive iterations.
- **Structured JSON data outputs**: All experiment results saved in well-structured JSON enables cross-validation and recovery. This practice should continue.
- **Adaptive pivoting based on evidence**: The project pivoted from EDW-DPO to CCAR to Activation Energy Theory based on experimental results. This evidence-driven adaptation is a strength.
- **Clear hypothesis framework with explicit thresholds**: H1-H5 with specific metrics and pass criteria makes evaluation objective and reproducible.
- **Recovery system auto-fixing stale tasks**: The experiment state recovery correctly identified and fixed completed tasks from gpu_progress.json.
