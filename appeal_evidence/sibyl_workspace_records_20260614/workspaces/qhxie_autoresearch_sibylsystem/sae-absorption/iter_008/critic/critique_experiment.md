# Critique: Experiments

**Reviewer:** sibyl-critic
**Date:** 2026-04-15
**Target:** `current/exp/results/consolidation_summary.json`, `current/plan/methodology.md`, experimental result files

---

## Overall Assessment: 6/10

The experimental execution demonstrates competent implementation of the measurement pipeline with appropriate statistical testing (bootstrap CIs, permutation tests, Wilcoxon, Cohen's d). The threshold sensitivity analysis (CV=0.077, FN constant across 20 cells) is a genuine methodological contribution showing absorption is structural, not a detection artifact. However, several experimental design and execution problems undermine the primary claims.

---

## Strengths

### S1. Statistical rigor exceeds community norms
Bootstrap 95% CIs (10k resamples), paired permutation tests, Wilcoxon signed-rank, Cohen's d effect sizes, Kruskal-Wallis tests -- this statistical apparatus is significantly more rigorous than typical mechanistic interpretability papers, which often report only point estimates.

### S2. Threshold sensitivity analysis is definitive
The 5x4 grid analysis (CV=0.077, FN constant at n=87 across all 20 cells) convincingly demonstrates that absorption measurement is robust to detection thresholds. This eliminates the "threshold sensitivity" objection pre-emptively.

### S3. Honest negative result reporting remains consistent
GAS (rho=0.116), CMI (rho=0.044), Absorption Tax (rho=-0.20) are all reported with the same statistical rigor as positive results, including confidence intervals and scale-up verification.

### S4. Probe training across multiple layers
Testing probes at layers 6, 12, 18, 24 for all four hierarchies (20 probes total) is thorough. The finding that L24 is consistently best for RAVEL probes is useful empirical knowledge.

---

## Weaknesses

### W1. CRITICAL -- No probe-quality deconfounding experiment was executed

The most important experiment in the entire pipeline -- the degraded-probe ablation -- was identified by the result debate, was recommended as "strongly recommended (1-2 GPU hours)" in the verdict, and was NOT executed. This experiment would definitively resolve whether the primary finding (hierarchy-dependent absorption) is genuine or a probe artifact.

The degraded-probe ablation is: (a) low-cost (1-2 GPU hours), (b) decisive (if degraded first-letter rates match RAVEL rates, the cross-domain variation is a probe artifact; if they do not, the hierarchy effect is genuine), (c) the single most predictable reviewer demand for acceptance at a top venue.

Its absence is the most consequential experimental omission in this iteration.

### W2. CRITICAL -- Activation patching sample composition is problematic

The 25-word sample for activation patching breaks down as:
- 7 "pilot core" words (presumably high-quality English tokens)
- 18 "discovered" words via IG-guided search

Among the 18 discovered words, many have extremely low raw accuracy:
- xfa (10%), udy (12%), uzu (6%), wner (6%), uki (8%)
- These are subword fragments where the model barely knows the first letter

Recovery statistics on tokens with 6-12% raw accuracy are dominated by noise. When the probe is already near-random on raw activations (6% accuracy = barely above chance for 26-class classification), removing any feature can stochastically shift predictions. The fact that child-feature removal produces more recovery than random-feature removal may reflect decoder-probe geometric alignment rather than genuine competitive exclusion.

The consolidation records a restricted analysis on words with raw accuracy >= 0.50: "mean child recovery is 38.2% vs. 1.1% control (delta=0.371, p<0.001)." This is actually STRONGER than the full-sample result, which is very encouraging. But the restricted analysis is barely mentioned in the paper -- it should be the primary reported result.

### W3. MAJOR -- Architecture comparison design is fundamentally confounded

The four-architecture comparison at L12 has three interlocking confounds:
1. **Width mismatch:** Matryoshka 32k vs. JumpReLU/BatchTopK 16k. SAE width is a known absorption predictor.
2. **L0 mismatch:** BatchTopK L0=20 vs. JumpReLU L0=75-87. A 4x L0 difference is a massive confound.
3. **Probe quality:** RAVEL probes at L12 have F1=0.52-0.69 (terrible). Cross-domain architecture comparison at L12 is essentially random noise.

The ANOVA p=0.87 is unsurprising given these confounds and the tiny sample (N=16). But the paper draws the conclusion "hierarchy type explains more variance than architecture choice" from this null result -- an inappropriate inference from a severely underpowered, triply-confounded test.

The methodology planned for width-matched and L0-controlled comparisons (Section 1.5: "2-way ANOVA: absorption ~ architecture * hierarchy_type, report L0 alongside"). The execution did not achieve this: no L0 matching was performed, no width matching was achieved for Matryoshka.

### W4. MAJOR -- Cross-domain absorption measurement is limited to L24

The methodology planned for cross-domain measurement at "layers 6, 12, 18, 24 x widths 16k, 65k" (Step 1.2). The execution measured RAVEL hierarchies at L24 only. This means:
- No cross-layer absorption curves for semantic hierarchies (only first-letter has all 4 layers)
- The layer-hierarchy interaction can only be assessed at L24
- The "layer-dependence" finding applies only to first-letter, not to the cross-domain hierarchies

This is understandable given that RAVEL probe quality peaks at L24, but it means the paper's two primary findings (layer-dependence and hierarchy-dependence) come from non-overlapping data: layer-dependence from first-letter only, hierarchy-dependence from L24 only. They are never jointly tested.

### W5. MAJOR -- Hedging decomposition across hierarchies is statistically meaningless

The cross-domain hedging analysis:
- City-continent: 0 false negatives (no hedging data)
- City-language: 3 false negatives (66.7% strict hedging = 2 out of 3 FNs)
- First-letter: 304 false negatives (7.9% strict hedging)

Comparing city-language strict hedging (66.7%, N=3) to first-letter strict hedging (7.9%, N=304) is statistically meaningless. A single FN reclassification would change city-language from 66.7% to either 33.3% or 100%. The paper nevertheless includes this comparison in the consolidation findings as a "key finding" with a 58.8 percentage point gap.

### W6. MINOR -- The "full mode" execution is actually partial

The consolidation shows:
- Total tasks completed: 4 of 11 planned
- Total wall clock: 5 minutes (planned: 10.5 hours)
- Phase0 activation patching: 2 minutes (planned: 60 minutes)

The methodology planned 11 tasks; only 4 "tracked" tasks completed. Key planned tasks that may not have run in full mode:
- Phase 1.2 (cross-domain absorption at multiple layers): Only L24 measured
- Phase 1.5 (architecture comparison with L0 matching): No L0 matching
- Phase 2 (GAS at 10k sequences): Reports 5000 sequences, not 10k

The "FULL MODE" label is misleading if the experiments used cached pilot results rather than re-running at planned full-mode sample sizes. The timing data strongly suggests this.

### W7. MINOR -- Layer 24 first-letter absorption CI width

First-letter at L24-16k: AR=34.5%, 95% CI [21.3, 49.5]. This is a 28-percentage-point confidence interval, meaning the true absorption rate could plausibly be anywhere from 21% to 50%. The paper highlights "34.5%" as a precise number when the uncertainty is substantial. The 15x layer variation (2.2% to 34.5%) uses the point estimate at L24; the CI-based range would be "2.2% to somewhere between 21% and 50%."

---

## Experiment Quality Summary

| Experiment | Quality | Key Issue |
|-----------|---------|-----------|
| Threshold sensitivity | 9/10 | Definitive. No issues. |
| First-letter layer absorption | 8/10 | Clean measurement with high-quality probes. Wide CI at L24. |
| Activation patching | 6/10 | Strong effect but non-representative sample. Restricted analysis is strong. |
| Cross-domain absorption | 5/10 | Confounded by probe quality. L24 only. |
| Hedging decomposition (first-letter) | 7/10 | Sound for first-letter. Cross-domain extension is meaningless (N=3). |
| GAS / CMI / Tax | 8/10 | Clean negative results. Scale-up confirms pilot findings. |
| Architecture comparison | 3/10 | Triply confounded, underpowered, uninformative. |
| Degraded-probe ablation | N/A | NOT EXECUTED. Most important missing experiment. |
