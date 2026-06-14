# Reflection Report: Iteration 1 — Activation Energy Theory

**Date**: 2026-04-29
**Paper**: The Limits of Consistency-Based Activation Energy for Problem-Level Routing in Mathematical Reasoning
**Supervisor Score**: 5.5/10 (REVISE)
**Critic Score**: 5/10 (experiment), 6/10 (planning), 6/10 (ideation), 7/10 (writing)

---

## Iteration Summary

This iteration produced a paper testing whether consistency-derived "activation energy" (Ea) can predict single-pass solveability for LLM mathematical reasoning. The paper presents a defensible negative result---Ea has zero predictive power (AUC=0.436, Spearman=-0.063)---but is critically undermined by a formula-data inconsistency that makes the entire Ea computation unreproducible. Additional issues include contradictory source data labeling, a tiny sample size (n=50), missing bibliography, and a narrow novelty claim resting entirely on a single negative result.

The iteration involved multiple pivots: Round 1 (EDW-DPO, falsified), Round 2 (CCAR, blocked by GPU incompatibility then resolved with PyTorch 2.11.0), Round 3 (API-based inference, blocked by missing API key), and Round 4 (Activation Energy Theory, current paper). The adaptive pivoting demonstrates evidence-driven decision-making, but the current direction has fundamental technical issues that must be resolved before proceeding.

---

## Issue Analysis by Category

### SYSTEM (2 issues)

1. **Stale/failed tasks accumulate in experiment state** (low): experiment_state.json contains many stale tasks from prior rounds (train_g1_stepdpo, eval_g3_adaptive_routing, setup_api_environment) causing confusion during recovery. The recovery system auto-fixed some tasks but the clutter remains.

2. **GPU idle time from failed dependency chains** (medium): Tasks like setup_api_environment and g0_baseline_eval failed immediately due to missing API keys but had already consumed scheduling overhead. The g1_saturation task ran for 200 minutes before failing due to answer extraction issues, then had to be re-run.

### EXPERIMENT (6 issues)

1. **CRITICAL: Ea formula-data inconsistency**: The paper defines Ea = -ln(c0) where c0 is consistency fraction at k=16. The actual code (analysis_h3.py, analysis_h2.py) computes Ea = 1/c0 where c0 is a fitted saturation parameter from curve_fit with bounds [0.1, 10.0]. For the fitted c0=0.10565, 1/c0=9.465 matches the data; -ln(0.10565)=2.247 does not. The notation.md Usage Note 3 incorrectly claims "Ea ~9.47 corresponds to c0 ~0.106" under Ea = -ln(c0), which is mathematically impossible. This is a naming collision: the paper's c0 (consistency fraction, ~0.1-1.0) and code's c0 (fitted parameter, ~0.1-10.0) are different variables with the same name.

2. **CRITICAL: H3 source data mislabeled as CONFIRMED**: analysis_h3.json reports h3_evaluation.status="CONFIRMED" with note "Ea is a useful routing signal," but predictor_quality.auc=0.436 and spearman_ea_accuracy=-0.063 show zero predictive power. The evaluation logic only checks threshold pass (low_ea_accuracy >= 0.75) and ignores AUC/Spearman evidence.

3. **Sample size too small** (major): n=50 problems with only 2 at Level 1 and 13 at Level 5. No confidence intervals reported. Per-level statistics are unstable.

4. **Per-problem fit failure undermines H1** (major): Median R2=0.000, 80% fit failure, only 8/50 prefer exponential by AICc. Aggregate R2=0.924 may be an averaging artifact.

5. **Table 8 lacks quantitative error classification** (major): Claims "execution errors dominate" but provides no counts or proportions.

6. **Level 5 Ea shows algorithmic saturation** (minor): All 13 Level 5 problems have Ea ~10.0 with sigma ~1.9e-6, suggesting the fitted c0 hits its lower bound of 0.1, causing Ea = 1/0.1 = 10.0 to cluster artificially.

### WRITING (4 issues)

1. **CRITICAL: Missing bibliography**: Paper cites [1] through [11] but has no References section. "Li 2026" and "ACAR 2026" may not exist as citable works.

2. **Inconsistent terminology** (minor): "Arrhenius kinetics" vs "Arrhenius-like kinetics." Table 7 status capitalization inconsistent.

3. **Section 3.2 product interpretation misleading** (minor): P_infty * k0 conflates ceiling and rate parameters.

4. **"Prove" is too strong** (minor): Section 5.3 uses "prove" for statistical evidence.

### ANALYSIS (3 issues)

1. **Post-hoc threshold = data leakage** (major): Optimal threshold 9.9999999999 is optimized on evaluation data. The 25pp "error floor" is an optimistic upper bound.

2. **H5 falsification under-reported** (major): Spearman=-0.219 (p=0.54) challenges internal consistency of the theoretical framework but is only briefly mentioned.

3. **No confidence intervals or bootstrap estimates** (major): All key statistics reported as point estimates only.

### PLANNING (2 issues)

1. **No cross-validation protocol for threshold selection** (critical in planning critique): Methodology does not specify how threshold is selected or whether cross-validation is used.

2. **No protocol for error classification (H4)** (major): Methodology does not describe how errors would be classified.

### PIPELINE (1 issue)

1. **H4/H5 promised but not delivered** (minor): Paper structure promises diagnostic analysis and alternative signal comparison but does not deliver.

### IDEATION (2 issues)

1. **Novelty claim is narrow and fragile** (major): Rests entirely on H3 negative result. H1/H2 collide with Yang et al. (2025). ACAR has related finding.

2. **"Activation energy" framing may be misleading** (minor from ideation critique): Physical analogy is weak; two Ea measures are uncorrelated.

### EFFICIENCY (1 issue)

1. **GPU scheduling inefficiency**: Failed dependency chains and lack of pilot gates waste GPU time. g1_saturation failed after 200 minutes due to answer extraction issues that a 5-minute pilot would have caught.

---

## Resource Efficiency Assessment

### GPU Utilization

- **Total completed GPU tasks**: 14 (gen_calibration_dataset, eval_g0_baseline, train_g1_stepdpo, train_g2_calibration_round2, g0_qwen_baseline, g1_saturation_v2, g2_consistency, g3_routing, setup_model, full_g1_saturation, g2_ranked_voting, analysis_h1, analysis_h2, analysis_h3)
- **Failed GPU tasks**: 5 (train_g2_calibration, eval_g3_adaptive_routing, g1_saturation, g0_baseline, plus API/setup failures)
- **Longest successful task**: full_g1_saturation at 329 minutes (planned 240 min, overshot by 37%)
- **Most wasteful failure**: g1_saturation at 200 minutes before failing due to answer extraction

### Bottleneck Analysis

1. **Experiment cycle** was the primary bottleneck: full_g1_saturation took 5.5 hours for 50 problems x 16 samples = 800 total inferences. This is reasonable for single-GPU inference but could be parallelized.
2. **Analysis tasks** (analysis_h1, h2, h3) are CPU-only and completed in minutes, but were scheduled sequentially. They could run in parallel.
3. **Dependency chains** blocked parallel execution: g0_baseline -> g1_saturation -> g2_consistency -> g3_routing is a linear chain, but g0_baseline and g1_saturation could potentially run in parallel.

### Scheduling Improvements

- Add a **pilot gate** (n=5-10 problems) before full experiments to catch extraction/formatting issues early. Would have saved ~200 minutes on g1_saturation.
- **Parallelize independent tasks**: g0_baseline and g1_saturation use different datasets/models and could run on separate GPUs.
- **Check prerequisites before scheduling**: API key check should happen before task registration, not after.
- Analysis tasks are CPU-only and can all run in parallel after data is collected.

---

## Quality Trend Assessment

**Trajectory: STAGNANT**

The paper's core score (5.5/10) is borderline for publication. The honest negative result is the paper's strongest aspect and has been consistent across iterations. However, critical technical issues (formula inconsistency, missing bibliography) prevent any forward progress. The quality is not declining---the experimental execution is competent and data is well-structured---but it is not improving because fundamental problems remain unaddressed.

Key quality metrics:
- Novelty: 5/10 (narrow, single negative result)
- Soundness: 4/10 (formula inconsistency is showstopper)
- Experiments: 6/10 (honest but small sample, no CIs)
- Reproducibility: 4/10 (missing bibliography, undocumented computation)

---

## Root Cause Analysis

### Primary Root Cause: Paper-Code Divergence

The most critical issue is that the paper describes a different computation than the code implements. This is not a simple typo---it is a fundamental mismatch between the theoretical framework (Ea = -ln(c0), c0 = consistency fraction) and the implementation (Ea = 1/c0, c0 = fitted saturation parameter). The root causes are:

1. **Inadequate code review before writing**: The experimenter agent wrote analysis code that used a sensible approach (fitting a saturation curve to consistency trajectory), but the paper writer interpreted the results using a different formula without verifying the code.
2. **Variable naming collision**: Both the paper and code use "c0" for different quantities, making it easy to confuse them.
3. **No validation step**: There was no explicit check that "does Ea = -ln(c0) produce the reported values?" before finalizing the paper.

### Secondary Root Cause: Evaluation Logic Oversimplification

The H3 evaluation logic checks only one metric (threshold pass) while ignoring AUC and Spearman. This is a pattern: the evaluation framework was designed with a single pass criterion in mind, but the paper's narrative correctly uses multiple metrics. The evaluation code and paper narrative diverged.

### Tertiary Root Cause: Scope Creep in Paper Structure

The paper promises H4 (error classification) and H5 (entropy routing) analyses that were not completed. The outline and paper structure were written with an optimistic view of what could be delivered, but resource constraints (time, API keys, GPU scheduling) prevented completion.

---

## System Self-Check Response

No `logs/self_check_diagnostics.json` file was present in this iteration. The self-heal system was not triggered. However, the experiment state recovery system did function: it auto-fixed several tasks (g0_qwen_baseline, g1_saturation_v2, g2_consistency, g3_routing, full_g1_saturation) by checking gpu_progress.json and marking them as completed. This is a positive system behavior.

---

## Recommended Next Steps

1. **Block submission** until the Ea formula inconsistency is resolved. This is a showstopper.
2. Fix H3 evaluation logic and update source data files.
3. Add complete bibliography with verified references.
4. Add bootstrap confidence intervals for key statistics.
5. Temper generalizability claims to model-specific language.
6. Consider whether to expand the paper (add second model, n=100-200) or reframe as a focused falsification study.

The paper has a defensible negative result at its core, but substantial additional work is needed before it reaches publication quality.
