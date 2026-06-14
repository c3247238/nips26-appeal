# Iteration 6 Reflection Report

**Project**: SAE Feature Absorption (sae-absorption)
**Iteration**: 6
**Date**: 2026-04-15
**Score**: 6.5 / 10 (Supervisor review.json)
**Verdict**: CONTINUE
**Quality Trajectory**: Improving (5.5 x3 -> 6.5 -> 6.0 -> 6.5)

---

## 1. Iteration Summary

Iteration 6 executed a major strategic pivot: from the epidemiological-methods-on-SAEBench approach (iter 4-5, focused on 48-SAE absorption-quality correlations with Baron-Kenny mediation) to a JumpReLU metric audit on Gemma 2 2B. This pivot was driven by the recognition that (1) the GPT-2 Small cross-domain experiments were categorically incapable of testing absorption due to 98% dead SAE features, (2) the causal overclaiming from observational cross-sectional data (within-width Gamma=1.0) was unfixable within the Baron-Kenny framework, and (3) the strongest findings from iter 4-5 (control failure, L0 phase transition) pointed toward a metric audit contribution rather than a causal mediation paper.

**Key Results**:
- **23/23 experiments completed** (zero failures, third consecutive iteration of perfect reliability)
- **Universal control failure**: Shuffled labels produce higher "absorption" than true labels in ALL five hierarchy domains (ratios 2.7x to infinity), the paper's most devastating finding
- **Hedging dominance**: 98.6% of false negatives at L0=22 classified as hedging, 1.4% as hierarchy-driven (9 persistent core words identified)
- **L0 phase transition**: 42.9% -> 0.8% monotonic decline, cross-layer CV < 10%, the most robust empirical finding
- **CMI-absorption association**: rho=-0.383 at d'=10 (p=0.059 uncorrected, p=0.236 Bonferroni-corrected), sign reversal at d'>=20

**Score recovered from 6.0 to 6.5**: The strategic pivot was correct -- the metric audit framing is stronger than the causal mediation framing. The score improvement reflects genuine progress: the universal control failure, confound decomposition, and L0 phase transition are defensible contributions that do not depend on the fragile CMI pillar. The 0.5-point gap to 7.0 (Weak Accept) is entirely attributable to three fixable issues: data integrity errors, theoretical overclaiming, and unexecuted blocking experiments.

---

## 2. Issue Analysis by Category

### ANALYSIS (5 issues: 2 high, 3 medium)

**[HIGH] DATA INTEGRITY (DATA001)**: The most urgent problem. Paper reports absorbed mean CMI=0.687 in three locations; source data has 0.6492 (5.9% error). Mann-Whitney U=41 vs source U=28; p=0.042 vs source p=0.045. Three different vocabulary sizes (1,092 / 1,196 / 1,204) across pipelines for the same task. Two different L0=82 absorption rates (14.39% vs 15.96%) within the same paper. This is the 6th consecutive iteration with a data pipeline integrity failure: iter 1-3 had PILOT mode propagation, iter 5 had Sobel z from wrong JSON path, iter 6 has CMI numbers from a different partition than the analysis. Root cause has been identified since iter 1 but never addressed: no automated source-to-summary cross-check.

**[HIGH] CMI DIMENSION INSTABILITY (EXP003)**: The sign reversal between d'=10 (rho=-0.383) and d'=20 (rho=+0.048) is a qualitative failure. A correlation that changes direction with a modest parameter change is not evidence for the underlying theory. After Bonferroni correction for 4 dimension tests, p=0.236. The d'=10 selection was not pre-registered. This is compounded by the probe quality confound (EXP004): absorption correlates with probe F1 at rho=-0.67, and the paper never controls for this.

**[MEDIUM] LETTER S OUTLIER (ANL001)**: CMI=0.961 (high) but absorption=31.43% (high) -- a clear exception to the claimed pattern. With n=25 and marginal significance, a single outlier can drive the result. No leave-one-out sensitivity analysis.

**[MEDIUM] HEDGING CLASSIFICATION BIAS (EXP001)**: The 98.6% hedging figure is computed as total_FN_at_L0_22 (657) minus persistent_core (9). It does NOT check whether the specific parent latent fires at higher L0 -- only whether the token stops being a false negative. At L0=176, 99.2% of all tokens resolve trivially. The figure is an upper bound that cannot distinguish genuine hedging from compensatory resolution.

### EXPERIMENT (5 issues: 3 high, 2 medium)

**[HIGH] ACTIVATION PATCHING UNEXECUTED (EXP002)**: The single most impactful missing experiment. The 9 persistent core words are the paper's strongest causal claim, but without activation patching (zero child -> check parent recovery), this rests on observational classification with known bias. Estimated 0.5-1 GPU-hour. Positive results confirm competitive exclusion at small scale; negative results strengthen the "all hedging" narrative. Either outcome strengthens the paper.

**[HIGH] CONTROL FAILURE UNDIAGNOSED (EXP005)**: The paper identifies THAT controls fail but not WHY. Without mechanistic diagnosis: cannot recommend recalibration, cannot distinguish miscalibration from structural difference, cannot predict transferability. The threshold sensitivity results (5x4 grid, 141KB) were already computed but not reported -- this is a zero-cost omission.

**[HIGH] CMI REPLICATION AT L0=22 NOT DONE (part of EXP003)**: At L0=22, all probes have F1=1.0 (eliminating probe quality confound) and absorption variance is maximal (0% to 66%). This is the maximally favorable condition for the CMI analysis. If the correlation replicates here with pre-registered d'=10, the theoretical pillar is secured.

**[MEDIUM] CMI ESTIMATION DIAGNOSTICS ABSENT (EXP006)**: No bootstrap variance, convergence curves, k-sensitivity, or alternative estimator comparison for the k-NN MI estimator.

### WRITING (6 issues: 1 high, 4 medium, 1 low)

**[HIGH] THEORETICAL OVERCLAIMING (WRI001)**: Section 6 title "CMI Predicts Absorption Susceptibility" from p=0.059/0.236 evidence. Abstract says "predicts." Introduction says "first information-theoretic criterion." Phase transition scale match is partly circular. This is the iter_006 instance of the persistent overclaiming pattern (iter 4: "validated quality indicator"; iter 5: "causal mediation"; iter 6: "CMI Predicts").

**[MEDIUM]**: Structural redundancy (WRI002), misleading title (WRI003), cross-domain novelty self-contradiction (WRI004), abstract too long/dense (WRI005).

**[LOW]**: Notation inconsistencies in companion documents (WRI006).

### PLANNING (2 issues: both medium)

**[MEDIUM]**: Breadth-first resource allocation (PLAN001) left 3 highest-value experiments unexecuted. Controls were not blocking in execution order (PLAN002).

### PIPELINE (1 issue: high)

**[HIGH]**: No automated integration cross-check (PIPE001), the 6-iteration persistent root cause of all data integrity failures.

---

## 3. Fix Tracking

### Fixed from Previous Iteration (FIXED)

| Previous Issue | Status | Notes |
|---|---|---|
| ITER5-EXP002: Data pipeline Sobel z / taxonomy rate | **FIXED** | Iter_006 pivoted away from mediation analysis; paper body numbers match source JSONs for L0 sweep, confound decomposition |
| ITER5-SND001: Causal overclaiming ('causally mediates') | **PARTIALLY FIXED** | Baron-Kenny mediation removed. New overclaiming in CMI theoretical sections |
| ITER5-SND002: Table 3 coefficient confusion | **FIXED** | Eliminated with pivot |
| ITER5-EXP001: H2 PARTIALLY_SUPPORTED reclassification | **FIXED** | Eliminated with pivot |
| ITER5-EXP003: GPT-2 Small capacity insufficiency | **FIXED** | All experiments on Gemma 2 2B |
| ITER5-WRI001: Figure numbering gaps | **FIXED** | 4 figures, sequential, all referenced |
| ITER5-PLAN002: Model fallback without pre-validation | **FIXED** | No fallback needed |
| ITER5-EXP004: GAM without architecture covariate | **FIXED** | GAM analysis not in iter_006 scope |

### Recurring Issues (RECURRING)

| Issue | History | Current Status |
|---|---|---|
| Data pipeline integrity | Iter 1-3: PILOT propagation; Iter 5: Sobel z; Iter 6: CMI partition | Same root cause, new manifestation |
| Framing overclaiming | Iter 4: quality indicator; Iter 5: causal mediation; Iter 6: CMI predicts | Each iteration fixes previous but introduces new |
| Notation inconsistencies | Iter 5-6: inverted definitions, wrong author name | Not yet fixed |
| Abstract density | Iter 5-6: too many numbers, too long | Improving but still exceeds target |

### New Issues (NEW)

11 new issues identified in iter_006 (see action_plan.json). Most critical new issues:
- Hedging classification bias (EXP001) -- new to this experimental design
- Activation patching gap (EXP002) -- recognized but unexecuted
- CMI dimension instability (EXP003) -- new challenge from rate-distortion theory
- Control failure diagnosis gap (EXP005) -- identifies central finding but does not explain WHY

---

## 4. Resource Efficiency Assessment

### GPU Utilization

| Metric | Value | Assessment |
|---|---|---|
| Total planned time | 305 min | |
| Total actual time | ~30 min | Experiments far faster than planned |
| GPU tasks | ~25 min | 23 experiments, all on RTX PRO 6000 Blackwell |
| CPU analysis | ~5 min | Geometric constant, phase transition prediction |
| GPU idle time | ~20 min | Between experiment batches |
| GPU utilization | ~80% | Good |
| Task failures | 0/23 | Excellent -- 3rd consecutive zero-failure iteration |

### Efficiency Assessment

**Positive**: Zero experiment failures for 50 consecutive tasks (iter 4-6). Environment setup reliable. Single GPU (RTX PRO 6000 Blackwell, 95GB VRAM) sufficient for all experiments. Planned/actual time ratios are conservative (10x overestimate typical), indicating room for additional experiments within time budget.

**Negative**: The 3 highest-value experiments (activation patching, control diagnosis, CMI replication at L0=22) collectively need only 2.5-4 GPU-hours but remain unexecuted. Meanwhile, experiments that produced low-value results (cross-domain measurements invalidated by control failure, unsupervised detection producing zero matches) consumed resources. This is not a GPU utilization problem -- it is a prioritization problem.

### Bottleneck Stages

1. **Unexecuted experiments**: The paper's path from 6.5 to 8.0 requires 3 experiments totaling 2.5-4 GPU-hours. These are not bottlenecked by compute but by task prioritization.
2. **Writing revision**: Multiple review stages independently discovering the same data integrity and overclaiming issues. An automated validation script would prevent the most costly revision cycles.
3. **Review feedback cycle**: Supervisor and Critic independently flagging identical issues (high validation confidence) but consuming substantial tokens on duplicate reports.

---

## 5. Quality Trajectory Assessment

### Score Trend

| Iteration | Score | Verdict | Key Event |
|---|---|---|---|
| 0 | 5.5 | CONTINUE | EDA + RAVEL + taxonomy (proxy model problems) |
| 1 | 5.5 | CONTINUE | H3 falsified + scope restructuring |
| 2 | 5.5 | CONTINUE | Stagnant (PILOT mode + framework overclaiming) |
| 3 | 5.5 | CONTINUE | Stagnant (PILOT mode + EncNorm narrative contradiction) |
| 4 | 6.5 | CONTINUE | Breakthrough (LV framework + full-scale execution) |
| 5 | 6.0 | REVISE | Deepened analysis (stricter review exposed causal/data issues) |
| 6 | 6.5 | CONTINUE | Strategic pivot to JumpReLU metric audit |

**Trajectory: IMPROVING**

The iter 4 -> 5 dip (6.5 -> 6.0) reflected stricter review standards, not quality regression. The iter 5 -> 6 recovery (6.0 -> 6.5) reflects a genuine strategic improvement: pivoting from the unfixable causal overclaiming problem to a metric audit framing where the evidence is stronger. The overall trajectory is:
- Iter 0-3: Plateau at 5.5 (wrong methods, wrong models, PILOT mode)
- Iter 4: Breakthrough to 6.5 (full-scale, methodological innovation)
- Iter 5-6: Stabilizing at 6.0-6.5 (refining framing, addressing deep methodological issues)

The paper is now positioned for a meaningful push to 7.5-8.0 if the three blocking experiments are executed and the data integrity issues resolved.

---

## 6. Root Cause Analysis

### Why is the score 6.5 and not 8.0?

Three root causes account for the entire gap:

1. **Claims exceed evidence (1.0 point)**: The CMI theoretical pillar is framed as "primary" but the evidence is marginal (p=0.236 corrected, sign reversal at d'>=20). The hedging classification is permissive enough to guarantee its result. The confound decomposition's "genuine competitive exclusion" claim for 9 words lacks causal validation. Fixing these involves either executing experiments (activation patching, CMI replication) or honest downgrading of language.

2. **Data integrity (0.5 points)**: Multiple numerical discrepancies between source data and paper text erode reviewer trust in ALL reported numbers. This is the same structural problem as every prior iteration, with a known fix (automated validation script) that has never been implemented.

3. **Missing diagnostic depth (0.5 points)**: The paper identifies WHAT (controls fail, hedging dominates) but not WHY (what about JumpReLU feature geometry causes the control failure? Does the specific parent latent fire at higher L0?). The threshold sensitivity data already exists but is unreported.

### Why do the same systemic patterns recur?

The data pipeline integrity problem has persisted for 6 iterations because the fix (automated validation script) requires proactive engineering work that is consistently deprioritized in favor of more scientifically interesting tasks. The overclaiming problem recurs because each new contribution has a structural incentive to present the strongest possible interpretation, and the anti-overclaiming check is applied retroactively in review rather than proactively in writing.

---

## 7. Success Pattern Extraction

### Patterns to Preserve

1. **Strategic pivot capability**: The system successfully pivoted from a failing research direction (causal mediation on observational data) to a stronger one (metric audit). The pivot was driven by evidence (within-width Gamma=1.0, control failure) rather than panic. This demonstrates healthy research judgment.

2. **Comprehensive control design**: The four-control suite (random probe, shuffled labels, dense probe ceiling, untrained SAE) deployed across five domains is the most thorough in any absorption study. This should be the standard for all future metric validation work.

3. **Honest negative results (6 consecutive iterations)**: The paper consistently earns its highest praise for transparent reporting of failed hypotheses. This pattern is now deeply embedded and must never be compromised.

4. **Zero experiment failures (3 consecutive iterations, 50/50 tasks)**: Infrastructure reliability is now a solved problem.

5. **Per-letter granularity**: Tracking probe F1, absorption rate, and CMI at the per-letter level enables confound identification that aggregate analysis would miss.

### Patterns to Address

1. **Engineering-debt avoidance**: The validation script that would prevent pipeline errors has been recommended for 3+ iterations but never implemented. Next iteration must treat it as a blocking task.

2. **Post-hoc dimension selection**: The CMI analysis tested 4 dimensions and selected the one with the best result. Pre-registration of primary analysis parameters must be enforced before execution.

---

## 8. System Self-Check Response

No `logs/self_check_diagnostics.json` detected. No system self-check response required.

---

## 9. Next Iteration Action Priority

### Critical Path to 8.0 (estimated 8-10 hours total)

| Priority | Task | Type | GPU | Hours | Impact |
|---|---|---|---|---|---|
| P0 | Fix data integrity + implement validate_integration.py | DATA/PIPE | Zero | 2.5 | Removes reviewer trust barrier |
| P0 | Downgrade theoretical overclaiming + add Bonferroni p | WRITING | Zero | 0.75 | Aligns claims with evidence |
| P1 | Activation patching on 9 core words | EXPERIMENT | 1h GPU | 1.0 | Only causal evidence for competitive exclusion |
| P1 | Tighten hedging classification (parent-specific) | EXPERIMENT | 1h GPU | 1.0 | Validates/invalidates 98.6% claim |
| P1 | CMI partial correlation + leave-one-out + probe F1 control | ANALYSIS | Zero | 0.5 | Determines CMI confound status |
| P2 | Include threshold sensitivity results + control diagnosis | ANALYSIS | Zero | 2.0 | Explains WHY controls fail |
| P2 | CMI replication at L0=22 with perfect probes | EXPERIMENT | 1h GPU | 1.0 | Determines theoretical pillar viability |
| P2 | Writing fixes (redundancy, title, abstract, notation) | WRITING | Zero | 1.5 | Polish |
| P3 | CMI estimation diagnostics (bootstrap, convergence, k) | ANALYSIS | Zero | 1.0 | Reproducibility |

**Expected score after P0+P1**: 7.0-7.5 (Weak Accept to Accept range)
**Expected score after P0+P1+P2**: 7.5-8.0 (Accept range)
**Expected score after all**: 8.0 (Strong Accept, depending on activation patching results)
