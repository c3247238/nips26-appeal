# Lessons from Iteration 11

## Must Improve

- **[ImageNet Phase 0 -- MANDATORY FIRST ACTION]**: 7 iterations of "add ImageNet" with zero progress. Do NOT plan ImageNet experiments until Phase 0 is complete: (1) SSH into server, (2) verify ImageNet data path, (3) run 1-epoch ResNet-50 smoke test, (4) document specific failure or confirm feasibility. If infeasible, document why and remove from plans permanently. This is the #1 priority for iter_012.

- **[BATCH ALL PENDING TEXT FIXES -- single 3-hour block]**: Six text fixes have persisted for 3-6 iterations each: CSI demotion from contribution (45min), appendix reference removal or creation (15min or 1.5hr), abstract 84+21 decomposition (5min), Section 5.7 condensation (30min), reproducibility details (1hr), orphan certified_band.png deletion (5min). Do ALL of them as a single batch DURING experiment setup, not after.

- **[Extra seeds for TOST power -- run with ImageNet]**: N=3 is insufficient for null-result claims. Add seeds 789 and 1024 for 4 key AdamW CIFAR-10 comparisons. 8 runs, ~2 GPU-hours. Schedule on GPUs not occupied by ImageNet. Until done, qualify claims with "within our detection threshold."

- **[NoBN: complete or remove -- no partial evidence]**: 2/7 NoBN methods is misleading. Either run remaining 5 methods (15 runs, ~4 GPU-hours) or remove partial results from Section 6.2. Do NOT present 2/7 as evidence for a 7-method hypothesis.

## Watch Out

- **[Do NOT rewrite -- targeted edits only]**: Score history proves it: rewrites cause crashes (iter_003: -2.7, iter_006: -2.0). Five text-only iterations (iter_007-011) show the ceiling is ~6.75 without new experiments. Only experiments can push past 7.0.

- **[Quality ceiling at 6.75 without experiments]**: Text-only iterations cannot exceed ~7.0. Supervisor estimates: ImageNet -> 7.5, + seeds + NoBN -> 8.0. Experiment execution is mandatory for progress.

- **[CSI demotion is overdue]**: 6 iterations unfixed. Takes <1 hour. Change Section 1.3 from "contribution" to "exploratory diagnostic." A reviewer will immediately attack a contribution claim for a metric with zero predictive value.

- **[No new issues = stable paper]**: Iter_011 found zero new issues. The paper is internally consistent for the first time in 5 iterations. Protect this stability -- do not introduce new sections or major restructuring.

## Keep Doing (Success Patterns)

- **Data-first ordering**: Fix figures and data before text. Confirmed effective across 3 iterations (009-011).

- **Targeted, verifiable fix instructions**: "Fix X in file Y, verify Z" outperforms "improve quality." Every instruction must answer: what + where + effort + verification.

- **Statistical methodology rigor**: Paired t-tests with Bonferroni, Cohen's d, TOST, power analysis. This is the paper's competitive advantage. Never compromise.

- **Three-stage review pipeline**: Supervisor, critic, and writing review catch distinct issues. Never skip any stage.

- **Phi modulator taxonomy + CWD/random-mask insight**: Structural backbone praised across 9+ iterations. Do not modify.

- **Strategic scope reduction**: Empirical-only paper is cleaner than theoretical-heavy proposal. Do not re-add Lyapunov/PMP/optimal control.
