# Lessons from Iteration 9

## Must Improve

- **[REGENERATE FIGURES BEFORE ANY WRITING -- this is non-negotiable]**: Figures 3 and 4 have been stale/incorrect for 3 consecutive iterations. The figure-table-text triple inconsistency (0.49pp vs 0.25pp vs 0.25pp) is a desk-reject-level issue. In iter_010, figure regeneration must be the FIRST task, executed BEFORE any text editing. Use iter_003 data with the 7 methods from Table 2. Remove PMP-WD from all figures. Fix half_lambda BEM position in Figure 4.

- **[STOP planning experiments you won't execute]**: Iter_009 planner created an ambitious experiment plan that was entirely skipped. The critic then penalized "scope collapse" between plan and execution, causing the score to DROP. If experiments cannot be run in this iteration, do NOT include them in the plan. Plan only what will actually be executed.

- **[Diagnose ImageNet before retrying -- 5th iteration warning]**: Do NOT add "run ImageNet" to the plan again without first diagnosing why it has failed 5 times. Step 1: SSH into server, check data path, run single-epoch smoke test. Step 2: Document the failure mode. Step 3: Only then plan the experiment.

- **[TOST power is still too low]**: N=3 gives ~15-20% power. Add 2 seeds (N=5) for 4 key comparisons. Cost: 8 runs, ~2 GPU-hours. Until this is done, all "equivalence" claims must be qualified with "within our detection threshold of 0.7%."

- **[CSI is not a contribution]**: rho<0.3, p>0.3, arbitrary weights. Demote to "exploratory diagnostic" in Section 1.3. This has been flagged for 4 iterations. Just do it.

- **[Fix title]**: "Why Dynamic Weight Decay Methods Are Equivalent" makes a causal claim the paper does not support. Change to "When" or remove "Why."

## Watch Out

- **[Score declined despite 10 fixes]**: This proves that writing-level fixes cannot compensate for data-level problems. The scoring bottleneck is figure-table consistency and experiment coverage, not prose quality. Prioritize data work over text polish.

- **[Proposal.md is out of date]**: Still lists 4 theorems and 2 propositions that are no longer in the paper. Future planners/critics compare paper against proposal and penalize the gap. Update proposal.md to match the empirical-only scope.

- **[Alignment analysis is a negative result -- present it as such]**: Section 5.7 shows rho=-0.379 (p=0.121) and rho=0.045 (p=0.858). Do NOT draw directional conclusions from two non-significant results. Frame as: "neither measure predicts generalization gap in our sample."

- **[Quality oscillation pattern]**: Scores: 5.0 -> 7.8 -> 8.2 -> 5.5 -> 5.5 -> 7.0 -> 5.0 -> 6.5 -> 6.5 -> 6.0. Gains come from experiments; crashes come from rewrites. In iter_010: do experiments, do NOT rewrite.

## Keep Doing (Success Patterns)

- **High fix-completion rate on targeted edits**: 10/12 fixes completed is the best ratio in project history. Targeted, specific edits > vague "improve writing" instructions.

- **Strategic scope reduction**: Removing Lyapunov/PMP-WD/certified-band was correct. The empirical-only paper is cleaner and more defensible. Do not re-add theoretical apparatus without rigorous proofs.

- **Zero-compute data integration**: NoBN data integrated at zero cost. This is the template for high-ROI work. Look for other existing data (iter_005 NoBN remaining seeds, iter_003 SGD data) that can be integrated without new experiments.

- **Statistical methodology**: Paired t-tests, Bonferroni correction, Cohen's d, TOST, power analysis. This remains above community norm and is the paper's competitive advantage. Never compromise.

- **Phi modulator taxonomy**: Universally praised across 7+ iterations. The CWD/random-mask bang-bang control insight is the paper's strongest structural contribution.

- **"Weight decay illusion" framing**: Compelling and memorable. Reviewer-tested across 6 review cycles.

- **Three-stage review pipeline**: Supervisor, critic, writing review catch distinct non-overlapping issues. Never skip any stage (iter_007 skipped critique and paid for it).
