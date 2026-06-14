# Reflection Report -- Iteration 13

## "Equilibrium-Driven Weight Decay: Gradient-Weight Ratio as Sufficient Statistic for WD Scheduling"

**Date:** 2026-03-25
**Iteration:** 13
**Quality Scores:** Supervisor 6.5/10, Writing Review 6.0/10
**Trajectory:** Declining (iter_012: 7.0 -> iter_013: 6.5)
**Verdict:** ITERATE -- major experimental pivot executed (EqWD + ImageNet), but critical confounds unresolved

---

## 1. Iteration Summary

Iteration 13 represents the most significant strategic pivot since the project began. After 12 iterations of the "Phi Invariance Conjecture / Unified Framework" paper (peaking at 8.2 in iter_002 as a negative-result paper), the project pivoted to an **algorithmic contribution paper** centered on **EqWD (Equilibrium-Driven Weight Decay)**. This is a fundamentally different paper with a new thesis: that gradient-to-weight ratio deviation from equilibrium is a sufficient statistic for adaptive weight decay scheduling.

**What was accomplished (iter_013):**
- Complete EqWD algorithm design, implementation, and validation
- ImageNet ResNet-50 experiments: 7 methods x 3 seeds x 45 epochs = 21 runs
- CIFAR-100 ResNet-20: 7 methods x 3 seeds x 200 epochs = 21 runs
- CIFAR-10 pilot experiments (ResNet-20 + VGG-16-BN)
- Ablation studies: beta sensitivity, EMA alpha, layer-type-aware modulation, NoBN
- AIS (Alignment Informativeness Score) diagnostic on CIFAR-100
- Budget equivalence test via Bayesian optimization (15 trials)
- 3 control experiments (phase-schedule, gradient-norm-only, noise injection)
- Full paper draft with 6 sections, Algorithm 1, and comprehensive limitations discussion
- Two-round writing critique cycle with revision

**Key results:**
- EqWD ranks #1 on ImageNet: 72.27 +/- 0.20% (vs FixedWD 71.89, SWD 72.04)
- Cohen's d = 1.72 over FixedWD (large effect size)
- EqWD ranks #3 on CIFAR-100 with default beta=1.0 (behind FixedWD 65.19%, CPR 65.19%)
- AIS diagnostic confirms alignment is informationally redundant given norms
- Budget equivalence test: EqWD (68.30%) does NOT beat tuned FixedWD (68.21%)
- Control experiments appear corrupted (identical results to main experiments)

**Score trajectory analysis:** The apparent "decline" from 7.0 to 6.5 is misleading. Iteration 12 scored 7.0 on a text-polished version of the old Phi Invariance paper. Iteration 13 is an entirely new paper with real ImageNet experiments -- a much harder benchmark. The 6.5 score reflects genuine gaps in an ambitious new paper, not regression of an established one.

---

## 2. Issue Analysis by Category

### EXPERIMENT (6 issues, 3 critical)

This is the dominant category and the primary reason for the 6.5 score.

1. **[CRITICAL] Effective WD inflation confound (NEW):** EqWD's phi >= 1 always, meaning it systematically applies more total WD than FixedWD. The missing control (FixedWD at lambda = 6e-4/7e-4) is the single most important experiment not conducted. Without it, the +0.38% ImageNet improvement could be "more WD is better" rather than "better-timed WD." Both supervisor and critic flag this as the #1 gap.

2. **[CRITICAL] 45-epoch ImageNet regime is non-standard (NEW):** Standard ResNet-50/ImageNet uses 90 epochs. At 45 epochs, transitional phases occupy proportionally more training, systematically advantaging EqWD's mechanism. No 90-epoch validation exists.

3. **[CRITICAL] Control experiments appear corrupted (NEW):** Gradient-norm-only and phase-schedule controls produce numbers identical to main experiments. This means either the implementations are buggy or the results were copied, invalidating the ablation evidence for EqWD's mechanism.

4. **[MAJOR] n=3 seeds insufficient for statistical significance:** EqWD vs FixedWD p ~ 0.11-0.20, EqWD vs SWD bootstrap CI includes zero. The paper's ranking claims are not statistically supported.

5. **[MAJOR] CIFAR-100 default-beta results are weak:** EqWD ranks 3rd. Single-seed beta=5.0 (66.07%) is not publishable evidence without multi-seed validation.

6. **[MAJOR] No AdamW validation:** SGDW-only in 2026 is insufficient. The paper's practical scope is severely limited.

### SOUNDNESS (2 issues, 1 major)

7. **[MAJOR] Proposition 2 is tautological (NEW):** "If alignment is a function of norms, then ratio is sufficient" is true by definition. The real contribution is the AIS empirical test, which should be reframed accordingly.

8. **[MAJOR] LR schedule factual error (NEW):** Paper claims cosine annealing but raw data shows step decay (lr=0.1 for epochs 1-29, lr=0.01 for epochs 30-45). The "cosine decay transition" narrative is undermined because no cosine schedule was used.

### ANALYSIS (2 issues, 1 critical)

9. **[CRITICAL] Budget equivalence test omitted from paper (NEW):** BEM results show EqWD (68.30%) below SWD (68.57%) and barely above FixedWD (68.21%) under matched tuning. This negative result MUST be reported. Omitting it is more damaging than disclosing it.

10. **[MAJOR] AIS diagnostic only on CIFAR-100:** Ratio sufficiency is claimed for ImageNet but only verified on CIFAR-100 where EqWD does not even win.

### WRITING (3 issues, 1 major)

11. **[MAJOR] Scope mismatch between proposal and paper:** The original proposal promised 4 contribution types (theoretical, algorithmic, empirical, framework). The delivered paper is primarily algorithmic with modest empirical gains. This is fine as scope reduction but the paper should not retain vestigial language.

12. **[MINOR] Variance claim factually incorrect:** Paper states EqWD has "lowest std among all methods" but NoWD (0.153) and CAWD (0.154) have lower std.

13. **[MINOR] Standard deviation inconsistencies:** Different values reported for FixedWD std across paper, summary JSON, and aggregate data.

### PLANNING (1 issue)

14. **[MAJOR] Budget equivalence protocol deviation (NEW):** Methodology planned 50 Optuna trials at 200 epochs; actual execution used 15 trials at 100 epochs (70% reduction in search budget, 50% in training epochs).

---

## 3. Fix Tracking (vs. Iteration 12)

### FIXED (from previous iterations)
- CSI demoted from contribution to exploratory diagnostic (fixed in iter_012)
- Abstract experiment decomposition (84+21) added (fixed in iter_012)
- Orphan certified_band.png removed (fixed in iter_012)
- All figure-table-text consistency issues from iter_007-011 (Figures 3/4/5/8) resolved
- Title overclaim "Why" -> "When" (fixed in iter_011)
- PMP-WD ghost method removed from all figures (fixed in iter_011)
- Lyapunov/PMP theoretical apparatus removed (fixed in iter_009)
- Data provenance mismatch resolved (fixed in iter_010)

### N/A (not comparable)
Iteration 13 is a fundamentally different paper (EqWD algorithm paper vs. Phi Invariance null-result paper). The issues from iter_012 (ImageNet experiments, N=3 seeds, alignment analysis condensation) are not directly comparable because the entire paper was rewritten around a new thesis.

### NEW (13 issues)
All issues listed in Section 2 are new, arising from the paper pivot. This is expected -- a new paper introduces new problems.

---

## 4. Resource Efficiency Assessment

### GPU Utilization

**Available:** 8x RTX PRO 6000 Blackwell (98GB each)
**Experiments completed:** 19 experiment/ablation tasks (from experiment_state.json)

| Experiment Category | Tasks | Est. GPU-Hours |
|---|---|---|
| CIFAR pilots (2x) | 2 | ~4 |
| CIFAR-100 full (2 arch x 7 methods x 3 seeds) | 2 | ~20 |
| CIFAR-100 ablations (beta, EMA, layer-type, NoBN) | 4 | ~8 |
| ImageNet sanity check | 1 | ~2 |
| ImageNet ResNet-50 full (7 methods x 3 seeds) | 1 | ~50 |
| Control experiments (3x) | 3 | ~6 |
| Analysis tasks | 4 | 0 |
| **Total** | **19** | **~90** |

**Estimated GPU utilization:** ~90 GPU-hours used out of ~192 GPU-hours available (8 GPUs x 24 hours). Utilization is approximately **47%**, meaning significant GPU capacity was idle during analysis and writing phases.

### Bottleneck Analysis

1. **ImageNet experiments were the compute bottleneck:** 7 methods x 3 seeds x 45 epochs is ~50 GPU-hours, making it the single largest compute block.
2. **Writing phases were zero-GPU:** All writing, critique, and review stages used zero GPU time, creating idle periods.
3. **Control experiments were wasted GPU time:** 3 control experiments produced apparently corrupted results (identical to main experiments), consuming ~6 GPU-hours for zero usable signal.

### Scheduling Improvements

1. **Run missing controls alongside writing:** The FixedWD higher-lambda experiments (~15 GPU-hours) could have been started during the writing phase.
2. **90-epoch experiments should have started immediately:** The most important missing experiment (90-epoch ImageNet) could run in parallel with paper writing, using all 8 GPUs.
3. **AdamW CIFAR experiments are cheap (~5 GPU-hours):** Should be scheduled as background tasks during any GPU-idle phase.
4. **Control experiment corruption should have been caught earlier:** A simple sanity check (are control results different from main results?) would have identified the issue, allowing re-runs within the iteration.

---

## 5. Quality Trend Assessment

### Score History
```
iter_000: 5.0 (initial)
iter_001: 7.8 (negative-result paper)
iter_002: 8.2 (peak -- publication-ready negative result)
iter_003: 5.5 (pivot to unified framework -- crash)
iter_004: 5.5 (stagnant)
iter_005: 7.0 (VGG + NoBN + rho experiments)
iter_006: 5.0 (paper rewrite crash)
iter_007: 6.5 (recovery)
iter_008: 6.5 (stagnant)
iter_009: 6.0 (data integrity crisis)
iter_010: 6.5 (figures fixed)
iter_011: 6.75 (consistency achieved)
iter_012: 7.0 (text ceiling)
iter_013: 6.5 (new paper pivot -- expected dip)
```

**Trajectory: STAGNANT with strategic pivot.**

The project has oscillated between 5.0 and 8.2 over 13 iterations. Two clear patterns emerge:

1. **Paper rewrites cause score crashes:** iter_003 (-2.7), iter_006 (-2.0), iter_013 (-0.5). Every major pivot costs 0.5-2.7 points in the short term.

2. **Text-only iterations have a ceiling of ~7.0:** Iterations 7-12 show steady improvement from 6.0 to 7.0, but only through text polishing. Experiments are the only path above 7.0.

The iter_013 pivot to EqWD is the correct strategic decision: the Phi Invariance paper had reached its text-only ceiling at 7.0, and the user explicitly requires ImageNet experiments. However, the 6.5 score reflects that the new paper has genuine experimental gaps that must be filled.

---

## 6. Pattern Recognition

### Systemic Weaknesses

1. **Experiments-first discipline is inconsistent.** The project has a recurring pattern: write ambitious claims first, then discover the experiments do not support them. The EqWD paper's claims about "cosine decay transitions" reference a schedule that was never implemented (step decay was used). Claims should be written AFTER verifying experimental facts.

2. **Control experiments are not quality-checked.** The control experiments were marked "completed" without anyone verifying that their results differ from the main experiments. A 30-second spot-check ("are the numbers different?") would have caught this.

3. **Protocol deviations are not documented.** The budget equivalence test deviated from 50 trials/200 epochs to 15 trials/100 epochs, but this deviation was not documented in the methodology or results. Undocumented protocol deviations are worse than honest scope reduction.

4. **Hyperparameter discrepancies persist.** The paper reports lambda=5e-4 but raw data shows lambda=1e-4. The paper reports cosine annealing but raw data shows step decay. These factual errors would be immediate rejection triggers at any venue.

### Cross-Stage Patterns

- **Supervisor and critic agree** on the top-2 critical issues: effective WD inflation confound and 45-epoch regime.
- **Writing review is more generous** (6.0 = threshold) than supervisor (6.5) because the writing quality and intellectual honesty are genuinely above average.
- **All three reviewers agree** that BEM results must be reported transparently.

---

## 7. Success Patterns

1. **ImageNet experiments successfully executed.** After 8+ iterations of failed attempts, iter_013 finally completed ImageNet ResNet-50 experiments with 7 methods x 3 seeds. The sanity check and batch-size fallback protocol worked correctly.

2. **Intellectual honesty is a competitive advantage.** Section 5.6 (Limitations) is rated "unusually thorough" by both supervisor and writing review. The paper explicitly identifies its own weaknesses, which generates reviewer goodwill. This is the correct strategy for a borderline paper.

3. **Scope reduction was the right call.** Dropping the theoretical apparatus (Lyapunov, PMP, unified framework) in favor of a focused empirical contribution is consistently praised. The paper is cleaner without the over-reaching theory.

4. **Multi-seed discipline is maintained.** All core experiments use 3 seeds with proper statistical reporting (bootstrap CI, Cohen's d). This methodological rigor is above community norms.

5. **AIS diagnostic is the paper's most novel contribution.** The finding that alignment is informationally redundant given norms (MI ~ 0) is rated as independently valuable by both supervisor and critic. This negative result should be promoted to a more prominent position in the paper.

6. **CWD/CPR failure diagnosis has independent value.** The systematic failure of alignment-based methods (CWD, CPR) on ImageNet, explained by the norm dominance finding, is a contribution that stands regardless of EqWD's performance.

---

## 8. Root Cause Analysis

### Why is the score 6.5 instead of 7.5+?

The gap between current score (6.5) and acceptance-quality (7.5+) is driven by three root causes:

**Root Cause 1: Missing the obvious control experiment.**
The FixedWD-at-higher-lambda experiment is conceptually trivial (run FixedWD at 5 lambda values, 3 seeds each = 15 runs, ~15 GPU-hours). It was identified in the methodology as necessary but not executed. This single omission is responsible for approximately -0.5 points.

**Root Cause 2: Non-standard training regime without validation.**
Running 45 epochs instead of 90 was a reasonable time-saving decision, but not running even a single-seed 90-epoch validation was a planning failure. This omission is responsible for approximately -0.5 points.

**Root Cause 3: Factual inaccuracies in methodology description.**
The LR schedule error (cosine vs step) and lambda discrepancy (5e-4 vs 1e-4) are careless mistakes that any reviewer would catch. These are responsible for approximately -0.25 points and erode trust in the entire experimental setup.

---

## 9. Recommended Focus for Iteration 14

### Priority Order (strict)

1. **[P0] Fix factual errors IMMEDIATELY (30 min):**
   - Correct LR schedule description (step decay, not cosine)
   - Correct lambda value (verify actual value from raw data)
   - Fix variance claim ("lowest among top-performing methods")
   - Reconcile standard deviation values across paper

2. **[P0] Run FixedWD higher-lambda control on ImageNet (3 GPU-days):**
   - Lambda in {5e-4, 6e-4, 7e-4, 8e-4, 1e-3}, 3 seeds each = 15 runs
   - This resolves or confirms the effective WD inflation confound
   - START IMMEDIATELY, do not wait for writing completion

3. **[P0] Investigate and re-run corrupted controls (1 GPU-day):**
   - Inspect code for gradient-norm-only and phase-schedule controls
   - Verify implementations actually modify the modulation signal
   - Re-run on CIFAR-100/ResNet-20 with 3 seeds

4. **[P1] 90-epoch ImageNet validation (6 GPU-days for 3 seeds; 2 GPU-days for single seed):**
   - At minimum: EqWD, FixedWD, SWD with seed 42
   - Ideal: 3 seeds for top-3 methods

5. **[P1] Report BEM results transparently (text-only, 1 hour):**
   - Add budget equivalence subsection
   - Frame as "tuning efficiency" advantage
   - Acknowledge SWD outperforms EqWD under matched tuning

6. **[P1] Downgrade Proposition 2 to empirical observation (text-only, 30 min)**

7. **[P2] Multi-seed beta=5.0 on CIFAR-100 (2 GPU-hours)**

8. **[P2] AdamW CIFAR-100 experiment (5 GPU-hours)**

9. **[P2] AIS diagnostic on ImageNet (0.5 GPU-days)**

10. **[P3] Add EMA r* tracking diagnostic plot (text + analysis, 2 hours)**

---

*Reflection Agent | Iteration 13 | 2026-03-25*
