# Planning Critique -- Iteration 9

## Executive Summary

The iteration 9 methodology document is the most focused and actionable plan in the project's history. It correctly identifies experiments as the binding constraint (not writing), organizes work into gated dependencies (Gate 0 -> Gate 1 -> Gate 2 -> Gate 3), and assigns realistic time estimates. The plan was executed: all Gate 0 and Gate 1 experiments are complete. However, the plan had blind spots that are now visible in hindsight.

## Strengths

### 1. Correct Diagnosis of Stagnation

The plan opens with "Score trajectory: 5.5 (x4) -> 6.5 -> 6.0 -> 6.5 -> 6.5 -> 6.5 (3 consecutive stagnation)" and correctly identifies that "Every score improvement was driven by experimental execution. Writing-only iterations produce +0.0." This evidence-based diagnosis prevented another writing-only cycle.

### 2. Gated Dependency Structure

The four-gate structure (zero-GPU analyses -> three GPU experiments -> writing revision -> review) correctly serializes dependencies. Gate 0 was correctly classified as BLOCKING with 0 GPU cost -- these analyses should have been done 6 iterations ago.

### 3. Pre-Registration of CMI Analysis

Pre-registering d'=10 as the primary subspace dimension for CMI at L0=22 before running the experiment is methodologically commendable. The result (rho=+0.044) would be less credible if d' were chosen post-hoc.

### 4. Realistic Resource Estimates

The plan estimated 3 GPU-hours and 6.5 CPU-hours. Actual execution appears to have completed within this budget. The plan correctly noted this is "Less than the time consumed by the last 2 empty review cycles."

## Weaknesses

### 1. PLAN DID NOT ANTICIPATE THE JUMPRELU THRESHOLD CONFOUND IN ACTIVATION PATCHING (Severity: Critical)

The methodology specifies decode-reencode patching (Section 1A) with a binary recovery criterion ("parent feature (probe-associated latent) recovers (activation > 0)"). But the plan does not discuss the JumpReLU hard threshold's impact on this criterion. The plan should have specified:
- Report pre-threshold parent activation before and after child zeroing
- Define a continuous recovery metric (change in pre-threshold activation / gap to theta_j)
- Include a positive control on L1-ReLU SAEs where competitive exclusion is known to exist

This omission means the 0/8 result has a confound that weakens its interpretive value.

### 2. CMI AT L0=22 SPECIFIED AS "~1 HOUR GPU" BUT MAY ONLY BE A PILOT (Severity: Critical)

The plan specifies "GPU cost: ~1 hour" for CMI at L0=22 and expects "CMI per letter, Spearman rho with absorption, bootstrap CIs, convergence curves." The source data shows a PILOT with 125 words. If the full run was never executed, the plan's resource allocation was correct but the execution may have stopped at the pilot.

### 3. PLAN DID NOT ANTICIPATE THE 93.8% NON-HEDGING PUZZLE (Severity: Major)

The tightened hedging experiment (Section 1B) correctly predicts two scenarios: "If strict rate << permissive rate, the 98.6% claim needs qualification." The result (6.2% strict vs 98.6% permissive) falls in this scenario. But the plan did not anticipate the follow-up question: what happens to the 93.8% of tokens that show zero parent-specific latent recovery but still resolve from FN status? The plan should have included a diagnostic for cross-L0 probe latent overlap.

### 4. MISSING POSITIVE CONTROL FOR ACTIVATION PATCHING (Severity: Major)

The plan includes a negative control ("For each word, we zero 10 randomly selected non-child features and check parent recovery -- expect no change") but no positive control. A proper experimental design would include at least one case where competitive exclusion is KNOWN to exist (e.g., GPT-2 Small with L1-ReLU SAEs where Chanin et al. report 15-35% absorption) to verify the patching pipeline can detect competitive exclusion when it is present.

### 5. NINE CORE WORDS BECAME EIGHT (Severity: Minor)

The plan specifies "9 persistent core words" and names 5 (eight/E, lower/L, liked/L, offer/O, often/O) with "4 unnamed." The activation patching found only 8 words: the 9th word ("wrong"/W) recovers at L0=82, making it hedging rather than persistent. The data_validation_report explains this reconciliation but the plan's expectation of 9 vs. the actual 8 created a minor discrepancy.

## Gate Execution Assessment

| Gate | Planned | Executed | Quality |
|------|---------|----------|---------|
| 0A: validate_integration | 1.5h CPU | Done | 84/85 claims match (excellent) |
| 0B: Partial correlation | 0.5h CPU | Done | rho=-0.328, p=0.118 (CMI weakened) |
| 0C: Leave-one-out | 0.5h CPU | Done | Max delta=0.088 (stable) |
| 0D: Threshold sensitivity reporting | 0.5h CPU | Done | CV=0.077, control failure structural |
| 0E: Control failure diagnosis | 0.5h CPU | Done | 23% candidates from random vector |
| 1A: Activation patching | 0.5-1h GPU | Done | 0/8 recovery (JumpReLU confound) |
| 1B: Tightened hedging | 0.5-1h GPU | Done | 6.2% strict (major narrative revision) |
| 1C: CMI at L0=22 | 1h GPU | Done (pilot?) | rho=+0.044 (null result, pilot concern) |
| Gate 2: Writing revision | 3h CPU | Done | Paper updated but Section 6 narrative incomplete |
| Gate 3: Review | -- | Done | Score 8/10 from writing review |

## Score Trajectory Assessment

The plan predicted:
- Gate 0 alone: 6.75
- Gate 0 + Gate 1: 7.5-8.0
- Gate 0 + Gate 1 + Gate 2: 8.0

The result debate scores 7.0. The gap between predicted 7.5-8.0 and actual 7.0 likely reflects: (a) CMI pillar collapse (plan assigned upside for significant CMI, which did not materialize), (b) activation patching being weaker evidence than expected due to JumpReLU confound, (c) the 93.8% non-hedging puzzle creating new questions rather than resolving existing ones.

The plan's prediction was reasonable given the information available. The 7.0 result is within the uncertainty range.

## Recommendations for Next Planning Cycle

1. **Always specify continuous metrics alongside binary criteria** for causal experiments. The 0/8 binary could mask analog signals.
2. **Include positive controls** for any causal intervention. If the patching pipeline cannot detect known competitive exclusion on L1-ReLU SAEs, it has construct validity failure.
3. **Anticipate follow-up questions** from tightened experiments. When an experiment is designed to distinguish two categories, plan the diagnostic for the residual.
4. **Verify pilot vs. full execution** before writing. The CMI L0=22 result is reported in the paper without specifying whether it comes from a 125-word pilot or a full run.
