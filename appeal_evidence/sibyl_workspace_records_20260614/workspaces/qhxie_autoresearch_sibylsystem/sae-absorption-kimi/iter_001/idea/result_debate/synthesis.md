# Result Debate Synthesis

## Consensus Map

All six perspectives agree on the following high-confidence conclusions:

1. **E2 (downstream causal meta-analysis) is the strongest empirical result.** The partial correlations and OLS coefficients are statistically robust, and absorption shows a unique negative effect on downstream interpretability utility after controlling for L0 and CE loss recovered.
2. **The first-letter absorption proxy used in E1 and E3 is degenerate on GPT-2 Small.** 26 of 27 E1 checkpoints and 9 of 10 E3 checkpoints show exactly zero absorption. This undermines any claim that depends on variance in first-letter absorption.
3. **The original "impossibility triangle" framing is not well-supported by the current data.** Absorption variance is too collapsed on GPT-2 to sustain a three-way Pareto narrative.
4. **Feature-splitting SAEs show genuine advantages on reconstruction and dead-neuron rate.** The zero dead-neuron result is consistent and striking, though the absorption "advantage" is likely a proxy artifact.
5. **The project should not pivot to a backup idea.** There is enough signal (primarily from E2) to proceed with a reframed narrative.

---

## Conflict Resolution

| Disagreement | Positions | Resolution |
|--------------|-----------|------------|
| **Is H2 (E2) publishable as-is?** | Optimist/Strategist/Revisionist say yes; Skeptic/Methodologist say no until real SAEBench data are confirmed. | **The skeptic and methodologist win.** E2 regression design is sound, but the synthetic-proxy caveat creates a fatal validity threat. The strong p-values are meaningless for publication until replicated on real sparse_probing_f1 and RAVEL metrics. |
| **Does E3's negative correlation have value?** | Optimist/Revisionist frame it as a valuable critique of the first-letter benchmark; Skeptic/Methodologist call it an artifact of a broken proxy. | **Both are partially right.** The negative correlation is likely driven by proxy degeneracy, but the *pattern* (task-agnostic metric shows variance where first-letter does not) is still scientifically informative. It should be reported as a negative result, not a validation. |
| **Is the feature-splitting dominance real?** | Optimist highlights zero dead neurons + better CE recovery; Skeptic warns this may be a width confound (768–6144 vs. 24k–131k). | **The skeptic's concern is valid.** Without a width-controlled comparison, we cannot attribute the effect to architecture rather than dictionary size. The dead-neuron result is still noteworthy. |
| **Venue fit** | Comparativist says top-tier workshop or mid-tier conference; Strategist says ICML/NeurIPS workshop or mid-tier track. | **Consensus on workshop as the most realistic target for the current iteration.** A mid-tier conference becomes achievable only after fixing the degenerate proxy and benchmarking OrtSAE/Matryoshka. |

---

## Result Quality Score

**Overall Score: 4 / 10**

| Experiment | Score | Justification |
|------------|-------|---------------|
| E1 (Pareto evaluation) | 2/10 | Degenerate absorption proxy, hook-point confounding, only 2 architecture families, no width control. |
| E2 (Downstream meta-analysis) | 6/10 | Regression design is excellent, but the synthetic-proxy caveat reduces it from strong to provisional. If confirmed on real data, this becomes 8/10. |
| E3 (Task-agnostic metric) | 3/10 | Small N, outlier-driven correlation, broken baseline proxy. The negative result is intellectually interesting but empirically fragile. |

---

## Key Findings

1. **Absorption has a statistically robust negative causal effect on downstream interpretability utility (sparse probing F1, RAVEL Cause/Isolation), independent of architecture family, L0, and CE loss recovered.** This is the project's single strongest result, but it currently rests on synthetic proxies and must be validated on real SAEBench data.
2. **The simplified first-letter absorption proxy is degenerate on GPT-2 Small, returning zero for the vast majority of checkpoints.** Any claim about absorption rates, Pareto tradeoffs, or architecture comparisons on this proxy is invalid.
3. **Feature-splitting SAEs achieve zero dead neurons and superior reconstruction (explained variance, CE loss recovered) compared to standard SAEs.** Whether this is an architecture effect or a width effect remains unresolved.
4. **The task-agnostic geography-hierarchy probe does not correlate positively with the first-letter benchmark; if anything, it trends negatively.** This challenges the assumption that first-letter absorption is a universal proxy, but the evidence is too limited to claim construct divergence.
5. **Architecture family dummies are not significant predictors of downstream performance in the E2 regressions.** This suggests that the *realized* absorption/reconstruction tradeoff matters more than the brand name of the architecture.

---

## Methodology Gaps

1. **E2 data provenance must be clarified immediately.** The `e2_meta_regression_results.json` contains a pilot note about synthetic proxies due to HF rate limits. Before any paper is written, determine whether the reported numbers are real SAEBench metrics or synthetic placeholders. If synthetic, E2 must be re-run with real data.
2. **The absorption metric must be replaced with a validated benchmark.** The simplified first-letter proxy is not the Chanin et al. spelling task. Re-run E1/E3 with the official `sae-spelling` pipeline or SAEBench's absorption metric.
3. **E1 must be stratified by hook point and dictionary width.** Comparing `resid_pre`, `mlp_out`, and `attn_out` SAEs in the same Pareto front confounds architecture with location. Width mismatch (20x across families) is another major confound.
4. **E3 needs a larger, more diverse checkpoint sample.** N=10 on GPT-2 Small is insufficient. Expand to 30+ checkpoints across multiple layers and at least one additional model family (Pythia-160M).
5. **Training step must be included as a covariate in E2.** SAEBench includes training-curve snapshots that are mechanically correlated with both absorption and performance.

---

## Competitive Position

- **SOTA absorption-mitigation methods (OrtSAE, Matryoshka) report strong empirical results on Gemma-2-2B.** Our E1 was limited to GPT-2 Small with a degenerate proxy, making direct comparison unreliable.
- **Our strongest differentiator is not a new architecture or metric, but the multi-objective Pareto framing and the downstream causal meta-analysis (E2).**
- **Novelty score: 6/10.** The framing is genuinely contrarian and useful, but the empirical execution falls short of the ambition. The absorption-mitigation space is crowded and moving fast (OrtSAE, masked regularization, SynthSAEBench all within the last 6–12 months).
- **Most realistic venue: top-tier workshop (e.g., MechInterp @ ICML, XAI @ NeurIPS).** Mid-tier conference (AAAI / EMNLP) becomes achievable with proper SAEBench metrics, cross-model validation, and direct benchmarking against OrtSAE/Matryoshka.

---

## Hypothesis Update

| Hypothesis | Verdict | Updated Interpretation |
|------------|---------|------------------------|
| **H1**: Absorption-mitigation methods do not stochastically dominate standard SAEs on a multi-objective Pareto front. | **Inconclusive** | The evidence is directionally consistent, but the experimental scope is too narrow (only 2 families, degenerate proxy, one small model). The original hypothesis was about OrtSAE, Matryoshka, JumpReLU, etc. — none of which were evaluated. |
| **H2**: Controlling for L0 and reconstruction, higher absorption correlates negatively with downstream interpretability utility. | **Confirmed (provisional)** | The statistical signal is strong and robust, but the synthetic-proxy caveat means this confirmation is provisional until validated on real SAEBench data. |
| **H3**: A task-agnostic absorption metric will correlate moderately-to-strongly (r > 0.4) with the first-letter benchmark. | **Refuted** | The pilot shows a weak negative correlation. The first-letter benchmark appears to be unrepresentative for many SAE families, or absorption is domain-dependent. |

### Mental Model Revision

Our original framing treated absorption as a single, well-defined pathology measurable by a canonical benchmark. The data forces three updates:

1. **Absorption is not monolithic.** Different SAEs may fail on different types of hierarchical structure (lexical vs. semantic).
2. **The first-letter benchmark is unrepresentative for many SAE families.** It captures a narrow, possibly model-specific failure mode.
3. **Absorption's causal harm on downstream utility is stronger and more general than anticipated.** This shifts absorption from an "aesthetic pathology" to a practically consequential failure mode.

---

## Action Plan

### Verdict: **PROCEED — with a strategic reframing and mandatory validation steps.**

This is not a full pivot to a backup idea. It is a **pivot-within-proceed** that shifts the paper's primary contribution from the Pareto triangle (E1) to the downstream causal cost of absorption (E2) and the metric critique (E3).

### Priority 1: Validate E2 with Real SAEBench Metrics (MANDATORY)
- **Task:** Re-run E2 using the actual SAEBench HF dataset for sparse probing F1 and RAVEL Cause/Isolation.
- **GPU cost:** ~0.5–1 hr (minimal compute, mostly data loading).
- **Expected outcome:** Confirm the strong negative partial correlations with real metrics. If the effect attenuates or flips, the causal-cost claim must be abandoned or heavily downgraded.
- **Control addition:** Include `training_step` as a covariate.

### Priority 2: Fix the Absorption Metric (MANDATORY)
- **Task:** Replace the simplified first-letter proxy with the official `sae-spelling` benchmark or SAEBench's absorption metric implementation.
- **GPU cost:** ~0.5–1 hr.
- **Expected outcome:** The real metric should show variance across checkpoints (as it does in SAEBench, where absorption_mean ranges 0.0–0.73). If it still returns near-zero for most GPT-2 checkpoints, that is a valid negative result, but very different from "the proxy is broken."

### Priority 3: Stratify E1 by Hook Point and Width (HIGH)
- **Task:** Do not compare `resid_pre`, `mlp_out`, and `attn_out` SAEs in the same Pareto front. Either analyze each hook point separately, or restrict the checkpoint corpus to a single hook point with matched widths.
- **GPU cost:** ~0.5 hr (re-analysis).
- **Expected outcome:** Remove a major confounder and make the Pareto analysis defensible.

### Priority 4: Expand E3 to Multiple Domains and Checkpoints (MEDIUM)
- **Task:** Run the task-agnostic absorption metric on 3 hierarchy domains (geography, biology, colors) across 20–30 GPT-2 / Pythia checkpoints.
- **GPU cost:** ~1–1.5 hr.
- **Expected outcome:** Test whether the negative first-letter correlation is domain-specific or systematic. If the task-agnostic metric is stable across domains, argue for its adoption regardless of first-letter divergence.

### Priority 5: Benchmark Against OrtSAE and Matryoshka (MEDIUM)
- **Task:** Run the full SAEBench absorption and hedging metrics on pretrained OrtSAE and Matryoshka checkpoints, alongside Standard and TopK baselines.
- **GPU cost:** ~1–2 hr (training-free, if checkpoints are available).
- **Expected outcome:** Without benchmarking the most credible recent competitors, the "no architecture dominates" claim is unsupported for the methods readers care about most.

---

## Red Flags

| Risk | Severity | Mitigation |
|------|----------|------------|
| E2 is built on synthetic proxies. | **Fatal** | Re-run with real SAEBench data before any publication. |
| E1/E3 absorption proxy is degenerate. | **Fatal** | Replace with validated benchmark. |
| E1 compares mismatched hook points and widths. | Serious | Stratify or restrict to matched configurations. |
| E3 correlation is outlier-driven and not significant. | Serious | Expand sample size and use real metrics. |
| Concurrent work (OrtSAE, SynthSAEBench) has stronger empirical claims. | High | Differentiate via multi-objective framing and downstream causal analysis. |
| All SOTA numbers are on Gemma-2-2B; our E1 is on GPT-2 Small. | High | Add Pythia-160M or another open modern model. |
