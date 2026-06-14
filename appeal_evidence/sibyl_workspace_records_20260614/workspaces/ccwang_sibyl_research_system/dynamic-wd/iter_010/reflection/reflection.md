# Reflection Report -- Iteration 10

## Iteration Summary

**Score: 6.5/10** | **Verdict: continue** | **Trajectory: stagnant (6.0 -> 6.5)**

Iteration 10 delivered material progress on figure-table consistency -- the primary blocker from iter_009. Three critical issues were fixed: Figure 3 PMP-WD contamination removed, Figure 4 half_lambda BEM position corrected, and the triple spread contradiction (0.49pp/0.25pp/0.25pp) resolved. These were the RIGHT priorities. The score rose from 6.0 to 6.5.

However, the iteration ran NO new experiments. GPU utilization was 0%. The remaining issues are predominantly experiment-level (ImageNet, extra seeds, NoBN completion, VGG+AdamW) and cannot be resolved through text editing. The paper has reached a local ceiling at ~6.5 for text-only iterations.

## Issue Analysis by Category

### EXPERIMENT (7 issues, 4 high severity)
The dominant category. The paper's experiment coverage has three structural gaps:
1. **Scale**: No ImageNet (6 iterations unfixed). Claims about "modern deep learning" lack evidence beyond 15M params.
2. **Power**: N=3 is underpowered for null-result equivalence claims. TOST passes for only 50% of comparisons.
3. **Coverage**: VGG-16-BN only tested with SGD (confounds optimizer/architecture/BN effects). NoBN ablation incomplete (2/7 methods).
4. **Figure data integrity**: Figure 5 heatmap and Figure 8 still contain stale/ghost data.

### ANALYSIS (3 issues, 2 high severity)
- CSI has been flagged for 5+ iterations as non-predictive. Still listed as a contribution.
- Section 5.7 alignment analysis draws directional conclusions from non-significant results.
- D'Angelo et al. overlap not explicitly addressed.

### WRITING (3 issues, 1 high + 1 medium + 1 low)
- Abstract implies full factorial design (misleading).
- Title uses causal "Why" but paper shows correlation.
- Proposition 1 may still be labeled as such.

### PLANNING (1 issue, medium)
- Proposal.md is out of date (still lists 4 theorems).

### IDEATION (1 issue, medium)
- D'Angelo overlap needs explicit differentiation paragraph.

## Fix Tracking (vs. iter_009 Action Plan)

### FIXED (5 issues resolved)
1. Figure 3 PMP-WD contamination (iter_009 S1/S5) -- FIXED
2. Figure 4 half_lambda BEM position (iter_009 S2) -- FIXED
3. Data provenance mismatch Figure 3 vs Table 2 (iter_009 S1) -- FIXED
4. Triple spread contradiction 0.49pp/0.25pp (iter_009 S3) -- FIXED
5. Certified Band / Lyapunov V_t section (iter_008 S2) -- FIXED (earlier)

### RECURRING (8 issues, 3+ iterations old)
1. **ImageNet** -- 6th iteration. No diagnosis attempted. HIGHEST PRIORITY.
2. **Statistical power (TOST)** -- 4th iteration. No extra seeds run.
3. **CSI demotion** -- 5th iteration. Text fix taking <1 hour.
4. **Alignment analysis reframing** -- 4th iteration. Text fix taking 30 minutes.
5. **Abstract factorial decomposition** -- 3rd iteration. Text fix taking 5 minutes.
6. **Title "Why" fix** -- 3rd iteration. Text fix taking 5 minutes.
7. **Proposal.md update** -- 3rd iteration.
8. **Heatmap BEM bug** -- same class as iter_007-009 Figure 4 bug, now in Figure 5.

### NEW (4 issues discovered)
1. **Figure 8 PMP-WD ghost data** -- desk-reject level. Critic caught this; supervisor missed it.
2. **NoBN incomplete + contradictory** -- 2/7 methods, and the 2 results contradict the BN hypothesis.
3. **D'Angelo overlap** -- needs explicit differentiation paragraph.
4. **Reproducibility gaps** -- GPU type, wall-clock, pseudocode missing.

## Resource Efficiency Assessment

**GPU utilization: 0%**. No experiments were run in iter_010. All progress came from figure regeneration and text editing. This is a severe inefficiency given that 8x RTX PRO 6000 Blackwell GPUs are available.

**Bottleneck**: The experiment pipeline has been blocked for 2 iterations. The system prioritizes text fixes (which are necessary but insufficient) over experiment execution (which drives score improvements).

**Scheduling opportunity**: In iter_011, four independent experiment blocks can run in parallel:
- ImageNet-100: 9 runs on 2 GPUs (~4 hours wall-clock)
- Extra CIFAR-10 seeds: 8 runs on 2 GPUs (~1 hour wall-clock)
- NoBN completion: 15 runs on 2 GPUs (~2 hours wall-clock)
- VGG+AdamW: 21 runs on 2 GPUs (~3 hours wall-clock)
Total: 53 runs, ~4 hours wall-clock with 8 GPUs. This is achievable within 1 iteration.

## Quality Trend Assessment

Score history: 5.0 -> 7.8 -> 8.2 -> 5.5 -> 5.5 -> 7.0 -> 5.0 -> 6.5 -> 6.5 -> 6.0 -> 6.5

**Trajectory: stagnant**. The score has oscillated between 5.0 and 8.2 for 10 iterations. The pattern is clear:
- **Gains** come from experiments (iter_001: +2.8, iter_005: +1.5) and figure/data fixes (iter_010: +0.5)
- **Crashes** come from major rewrites (iter_003: -2.7, iter_006: -2.0)

The current 6.5 is a text-edit ceiling. Breaking through to 7.0+ requires new experiments (ImageNet being the single highest-ROI investment, worth ~1.0 points per supervisor estimate).

## Root Cause Analysis

### Why does ImageNet keep failing?
No one has ever diagnosed the root cause. Every iteration adds "run ImageNet" to the plan. Every iteration skips it. The system needs a mandatory PHASE 0: SSH diagnosis before planning any ImageNet experiments.

### Why do figure-table bugs recur?
Figure generation scripts read from different data sources. There is no single authoritative data file that all figures and tables reference. Each figure script independently loads data, creating divergence opportunities. Solution: create a single `paper_data.json` that is the canonical data source for ALL figures and tables.

### Why do 5-minute text fixes persist for 3+ iterations?
They are deprioritized below higher-severity figure/data work, which consumes the full iteration budget. Solution: batch all pending text fixes (<2 hours total) at the START of the iteration, before any experiment planning.

## Systemic Weaknesses

1. **No single source of truth for paper data**: Figures, tables, and text independently reference different data files/iterations. This is the root cause of the recurring figure-table consistency bugs.

2. **Experiment planning without execution capability validation**: The system plans ImageNet experiments without verifying data paths, checking for OOM, or running smoke tests. Planning without validation wastes iteration cycles.

3. **Text fix deprioritization**: Simple, high-impact text fixes (CSI demotion, title, abstract) accumulate because each iteration focuses exclusively on the highest-severity issue. A 2-hour batch at iteration start would clear the backlog.

4. **Rewrite risk**: The quality oscillation pattern shows that major rewrites cause score crashes. The system must avoid rewrites in favor of targeted edits.
