# Planning Critique

## Iteration Pattern Analysis

The project has followed a pattern of ambitious planning followed by scope collapse:
- Plans consistently include ImageNet, non-BN ablation, PMP-WD, and increased seeds
- Execution consistently delivers only text edits and figure regeneration
- Quality scores oscillate: 5.0 -> 7.8 -> 8.2 -> 5.5 -> 5.5 -> 7.0 -> 5.0 -> 6.5 -> 6.5 -> 6.0
- Gains come from experiments; crashes come from rewrites

## Key Planning Failures

### 1. ImageNet -- 5 Iterations of Non-Execution
ImageNet has been in the plan for 5 consecutive iterations without a single run completing. The planning process has not adapted to this failure. Each iteration re-plans ImageNet identically without diagnosing why it fails. The correct plan is: (1) diagnose, (2) smoke test, (3) execute. Not: (1) plan ambitiously, (2) skip.

### 2. Scope Oscillation
The paper has oscillated between "theoretical + empirical" and "empirical only" across iterations. The scope reduction to empirical-only was correct but never fully committed: the proposal still references Lyapunov/PMP, and Figure 8 still uses PMP-WD data. Each iteration introduces scope remnants from previous ambitions.

### 3. Figure Regeneration Deprioritized
Figures 4 and 8 have been incorrect for 3-4 iterations. Text edits have been prioritized over figure fixes. This is backwards: figure-table consistency is a hard requirement; prose polish is an optimization.

## Recommendations for Next Iteration

1. **First action: Fix figures.** Zero compute cost, maximum impact. Fix Figures 4 and 8 before any text editing.
2. **Do NOT rewrite the paper.** Quality drops correlate with rewrites. Make targeted edits only.
3. **Plan only executable experiments.** If ImageNet cannot be diagnosed in 30 minutes, defer it and explicitly state why.
4. **Add 2 seeds for 4 key comparisons.** 8 runs, ~2 GPU-hours. Highest ROI experiment.
5. **Update proposal.md.** 5-minute task that prevents downstream penalization.
