# Reflection Report -- Iteration 9
## "The Phi Invariance Conjecture: Dynamic Weight Decay Methods Under AdamW"

**Date:** 2026-03-19
**Iteration:** 9
**Quality Score:** 6.0/10 (Supervisor: 6.0, Critic: ~6.0, Writing Review: 6.0)
**Trajectory:** Declining (iter_008: 6.5 -> iter_009: 6.0)
**Verdict:** ITERATE -- figure-table-text data integrity is now the single blocking issue

---

## 1. Iteration Summary

Iteration 9 focused exclusively on writing-level fixes, successfully executing 10 of 12 planned structural edits:

**Completed (10 fixes):**
- Removed Section 5.7 (Certified Band / Lyapunov) -- eliminates 5-iteration-old V_t contradiction
- Removed build manifest from paper end
- Added evaluation metric specification ("best test accuracy")
- Defined alignment deviation symbols (delta_t, bar_delta_T, delta_T^sup)
- Added experiment count decomposition (84 + 21 = 105)
- Specified SWD h() function form
- Downgraded Proposition 1 to inline remark
- Added conjecture qualifiers to abstract
- Integrated NoBN data into Section 6.2
- Renumbered figures consistently (1-8)

**Not completed (2 critical items):**
- Figure regeneration (Figures 3, 4) -- the single most impactful fix, not executed
- No new experiments (zero GPU compute used)

**Net result:** Score DECLINED from 6.5 to 6.0. The writing fixes were necessary but insufficient. The score decline is attributed to the planner creating an ambitious experiment plan that was skipped, causing the critic to penalize "scope collapse" between plan and execution.

---

## 2. Issue Analysis by Category

### EXPERIMENT (4 issues, all recurring)
The experiment deficit is the paper's central weakness. No new experiments have been run since iter_006 (3 iterations ago). The paper operates on a fragmented data estate: iter_003 for tables, iter_005 for VGG/NoBN, iter_006 for figures. This multi-provenance problem is the root cause of nearly all critical issues.

- **S1 (data provenance)**: 3rd iteration flagged, unfixed
- **S4 (ImageNet)**: 5th iteration flagged, never diagnosed
- **F1 (TOST power)**: Recurring since iter_005
- **W1 (VGG AdamW)**: Recurring since iter_005

### ANALYSIS (5 issues, mix of recurring and new)
The analysis issues cluster around two themes: (1) metrics with no predictive value consuming paper space, and (2) non-significant statistical results presented misleadingly.

- **S2 (BEM scatter bug)**: 3rd iteration flagged, never debugged
- **S3 (triple spread contradiction)**: 2nd iteration flagged
- **F2 (CSI zero predictive value)**: Recurring since iter_006
- **F3 (alignment analysis non-significant)**: Recurring since iter_007
- **F5 (scope collapse)**: Recurring since iter_008

### WRITING (4 issues)
Writing issues are comparatively minor after iter_009's 10 fixes. Remaining items are title, reproducibility details, and figure filename consistency.

- **F4 (abstract decomposition)**: Partially fixed in iter_009
- **F6 (title causal claim)**: New
- **W2 (reproducibility gaps)**: Recurring
- **W3 (figure filenames)**: Low severity

### PIPELINE (1 issue, critical)
- **P1 (writing-experiment decoupling)**: 9th consecutive iteration. This is the systemic root cause: writing runs before data is clean, introducing inconsistencies that compound across iterations.

---

## 3. Resource Efficiency Assessment

### GPU Utilization: 0%
Zero GPU compute was used in iter_009. This is the third consecutive zero-compute iteration (iter_007, iter_008, iter_009). The 8x RTX PRO 6000 Blackwell GPUs have been completely idle for 3 iterations while the paper's most critical issues require compute work (figure regeneration from data, additional seeds, ImageNet).

### Bottleneck Analysis
The bottleneck is NOT compute availability -- GPUs are idle. The bottleneck is the pipeline ordering: the system enters the writing stage before completing experiments and figure generation. Every iteration since iter_006 has followed the pattern:
1. Plan ambitious experiments
2. Skip experiments, go directly to writing
3. Edit text around stale/inconsistent figures
4. Review catches figure-table-text mismatches
5. Score stagnates or declines
6. Repeat

### Scheduling Improvement
The single most impactful scheduling change: **block writing_sections until ALL figures have been regenerated from authoritative data**. This would break the 3-iteration figure stagnation cycle.

---

## 4. Quality Trend Assessment

```
Iter  Score  Trend
 0    5.0    --
 1    7.8    +2.8 (large gain)
 2    8.2    +0.4 (peak)
 3    5.5    -2.7 (scope pivot crash)
 4    5.5     0.0 (stagnant)
 5    7.0    +1.5 (experiment recovery)
 6    5.0    -2.0 (rewrite crash)
 7    6.5    +1.5 (partial recovery)
 8    6.5     0.0 (stagnant)
 9    6.0    -0.5 (declining)
```

**Pattern:** Score oscillates with amplitude ~2 points. Gains come from experiments (iter_001, iter_005) and recoveries (iter_007). Crashes come from scope pivots (iter_003) and full rewrites (iter_006). The NET trajectory is declining from the iter_002 peak of 8.2.

**Root cause of decline:** The paper's scope expanded (from narrow null-result to unified framework) while the evidence base did not keep pace. The iter_002 paper with 42 experiments on a focused claim scored 8.2. The current paper with 105 experiments on a broader claim scores 6.0 because the broader claim demands more evidence (ImageNet, higher power, more architectures) that has not been produced.

**Prognosis:** If figure regeneration + consistency rerun are executed in iter_010, score should recover to 7.0-7.5. Adding ImageNet would push to 8.0. Continued text-only iterations will stagnate at 5.5-6.5.

---

## 5. Fix Tracking (iter_008 -> iter_009)

### FIXED (10 items)
| iter_008 ID | Description | Status |
|-------------|-------------|--------|
| S2 | Lyapunov V_t contradiction / certified band | REMOVED (correct decision) |
| W5 | SWD h() undefined | FIXED |
| W6 | Conjecture scope qualifiers | FIXED |
| W7 | Best vs final accuracy unspecified | FIXED |
| W10 | Proposition 1 trivial | FIXED |
| W11 | Build manifest in paper | FIXED |
| D1 | NoBN data unused | INTEGRATED |
| W4 | Figure numbering | RENUMBERED |
| W1 | Abstract scope decomposition | PARTIALLY FIXED |
| W3 | Alignment symbols undefined | PARTIALLY FIXED |

### RECURRING (5 items -- these are the blocking issues)
| iter_008 ID | Description | Iterations Flagged |
|-------------|-------------|-------------------|
| S1 | Data provenance mismatch | 3 |
| F1/S5 | PMP-WD ghost method | 2 |
| F3 | BEM scatter bug | 3 |
| S4 | No ImageNet | 5 |
| D2 | TOST power insufficient | 4 |

### NEW (2 items)
- **F6**: Title makes unsupported causal claim ("Why")
- Score decline attributed to plan-execution gap (ambitious planner, no experiments executed)

---

## 6. Root Cause Analysis

### Why does the figure-table-text inconsistency persist for 3 iterations?

**Proximate cause:** The figure generation pipeline is never executed during the review-fix loop. Writing fixes edit text around existing figures rather than regenerating them.

**Root cause:** The orchestrator treats "writing_sections" and "figure_generation" as the same stage, when they should be separate stages with a dependency: figures must be generated BEFORE writing references them.

**Evidence:** In iter_007, iter_008, and iter_009, the planner included figure regeneration as an action item. In all three iterations, the writing stage started without figures being regenerated. The writing agent edited text to add captions and cross-references to stale figures, making the inconsistency worse.

### Why has ImageNet never been diagnosed?

**Proximate cause:** Each iteration adds "run ImageNet" to the plan without investigating why previous attempts failed.

**Root cause:** The system treats ImageNet as a "run experiment" task when the first required step is a "debug infrastructure" task (verify data path, check OOM, test CUDA compatibility). The orchestrator does not distinguish between "experiment that needs designing" and "experiment that needs debugging."

---

## 7. Success Pattern Extraction

1. **High fix-completion rate on writing tasks**: Iter_009 completed 10/12 planned fixes, the highest ratio in project history. Writing-level edits are well-suited to the current pipeline.

2. **Strategic scope reduction works**: Removing Lyapunov/PMP-WD/certified-band was the right call. The paper is more honest and defensible as an empirical contribution.

3. **Zero-compute data integration**: NoBN data integration from iter_005 was the highest ROI action in iter_009 -- zero GPU cost, directly addresses a reviewer concern.

4. **Statistical methodology is the moat**: Paired t-tests, Bonferroni correction, Cohen's d, TOST, power analysis -- this is above community norm and is the paper's strongest selling point.

---

## 8. Recommended Strategy for Iteration 10

**CRITICAL CHANGE: No writing until figures are clean.**

Iteration 10 must break the 3-iteration stagnation by executing the data/figure work that has been deferred:

1. **Figure regeneration (3 hours, zero compute)**: Regenerate Figures 3, 4, and any figure containing PMP-WD from iter_003 data using the 7 methods in Table 2. Verify each regenerated figure visually.

2. **Additional seeds (2 GPU-hours)**: Add seeds 789 and 1024 for constant, CWD, cosine, no_wd on CIFAR-10 AdamW ResNet-20. Raises TOST power from ~20% to ~55%.

3. **ImageNet diagnosis (2 hours, zero compute)**: SSH into server, check data path, run smoke test. Document findings regardless of outcome.

4. **Only then: incremental writing edits** to update tables/text to match new figures and data.

Expected score impact: Figure fixes alone -> 6.5-7.0. Plus additional seeds -> 7.0-7.5. Plus ImageNet-100 -> 8.0.
