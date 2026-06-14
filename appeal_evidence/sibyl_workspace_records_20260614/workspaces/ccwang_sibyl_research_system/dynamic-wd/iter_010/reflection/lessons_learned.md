# Lessons from Iteration 10

## Must Improve

- **[Figure 5 heatmap + Figure 8 ghost data -- fix BEFORE anything else]**: Figure 5 shows half_lambda BEM=0.000 (should be 0.500). Figure 8 contains PMP-WD data points for a method not in the paper (desk-reject level). Both are 10-30 minute fixes. Do them first. Visually verify every regenerated figure with the Read tool.

- **[Create a single paper_data.json as canonical data source]**: The recurring figure-table inconsistency bug (iter_007 through iter_010) stems from figures reading different data files. Create one `paper_data.json` containing all method accuracies, BEM/AIS/CSI values, and statistical test results. ALL figure scripts and table generators must read from this single file. This prevents data divergence permanently.

- **[ImageNet: DIAGNOSE FIRST, plan second -- mandatory Phase 0]**: 6 iterations of "add ImageNet to plan" with zero execution. Do NOT plan ImageNet experiments until Phase 0 is complete: (1) SSH into server, (2) verify ImageNet data path, (3) run 1-epoch smoke test, (4) document specific failure mode. Only then plan the experiment. If ImageNet is genuinely infeasible, document why and stop adding it to plans.

- **[Batch all pending text fixes at iteration START]**: 6 text-level fixes have persisted for 3+ iterations each because they are deprioritized below data work: CSI demotion (1hr), alignment Section 5.7 reframing (30min), title Why->When (5min), abstract factorial decomposition (5min), proposal.md update (30min), D'Angelo paragraph (20min). Total: ~2.5 hours. Do ALL of them as a single batch before experiment planning.

- **[Extra seeds for TOST power]**: N=3 is insufficient for null-result claims. Add seeds 789 and 1024 for 4 key AdamW CIFAR-10 comparisons. 8 runs, ~2 GPU-hours. Until done, qualify all equivalence claims with "within our detection threshold of 0.7%."

## Watch Out

- **[Do NOT rewrite -- targeted edits only]**: Score history proves it: rewrites cause crashes (iter_003: -2.7, iter_006: -2.0). Iter_010 gained +0.5 from targeted figure fixes, not prose rewrites. Continue this approach.

- **[Figure 8 PMP-WD is a new desk-reject issue]**: The critic caught PMP-WD data points in Figure 8 (alignment analysis). The supervisor's review.json did not flag this. Always cross-check critic findings against supervisor review -- they catch different issues.

- **[NoBN 2/7 results contradict BN hypothesis]**: Section 6.2 reports NoBN spread=0.11pp (narrower than with-BN 0.25pp), opposite to what the BN-mechanism hypothesis predicts. Either test all 7 methods without BN or remove the contradictory preliminary results. Do not present 2/7 as evidence.

- **[Quality ceiling at 6.5 without new experiments]**: Text-only iterations cannot push the score above 6.5. The supervisor estimates: ImageNet -> 7.5, + NoBN/seeds -> 8.0. Experiment execution is mandatory for progress.

- **[CSI demotion is a free win]**: Takes <1 hour, flagged for 5 iterations. Just do it. Change Section 1.3 item 2 from "contribution" to "exploratory diagnostic."

## Keep Doing (Success Patterns)

- **Figure regeneration as first task**: The iter_009 lesson was followed and 4/5 critical figure issues were fixed. Continue this data-first approach.

- **Targeted, specific edit instructions**: "Regenerate Figure X from data source Y, verify value Z" outperforms "improve writing quality." Every fix instruction must specify what + where + how to verify.

- **Statistical methodology rigor**: Paired t-tests with Bonferroni, Cohen's d, TOST, power analysis. This is the paper's competitive advantage against reviewers. Never compromise.

- **Three-stage review pipeline**: Supervisor, critic, and writing review catch distinct issues. Iter_010 proves this: critic caught Figure 8 PMP-WD ghost that supervisor missed. Never skip any stage.

- **Strategic scope reduction**: The empirical-only paper is cleaner than the theoretical-heavy proposal. Do not re-add Lyapunov/PMP/optimal control without rigorous proofs.

- **Phi modulator taxonomy + CWD/random-mask insight**: Universally praised across 8+ iterations. This is the paper's structural backbone.
