# Section Critique: 4. Experiments

**Reviewer**: Section Critic Agent
**Section**: Experiments (experiments.md)
**Date**: 2026-03-10
**Overall Score**: 7/10

---

## Summary

The Experiments section is comprehensive and well-structured, covering setup, main results, ablations, generalization, compute fairness, information-theoretic validation, and hypothesis verification. The honest reporting of negative results (4 falsified hypotheses) and the systematic hypothesis tracking are commendable. However, several issues weaken the section's persuasiveness: inconsistent comparison scales, missing full-scale validation for the proposed methods, and some presentation choices that obscure the actual contribution magnitude.

---

## Critical Issues

### C1. Comparing pilot-scale and full-scale results in the same table (Table 1)
- **Location**: Section 4.2, Table 1 (lines 33-42)
- **Severity**: Critical
- **Detail**: Table 1 mixes Countdown-500 (3-seed, n=500) results for some methods (Vanilla, ReMDM, RCR, DMI) with Countdown-16 (single-seed, n=16) results for others (BSD, A-CFG, DTA). This is fundamentally misleading. A reader scanning the table sees DMI at 9.3% and A-CFG at 12.5% and may conclude A-CFG is better, but these numbers are on entirely different evaluation scales with incomparable statistical power. The "---" entries make it worse: the table implies BSD/A-CFG were not tested at full scale, but this is buried rather than highlighted.
- **Suggestion**: Either (a) split into two separate tables (Table 1a: Full-scale Countdown-500; Table 1b: Pilot Countdown-16 + GSM8K-16), or (b) add a prominent "Scale" column and a footnote warning that pilot and full-scale results are not directly comparable. The current layout risks reviewers accusing the paper of cherry-picking evaluation scales.

### C2. No full-scale validation of the proposed methods (BSD, A-CFG)
- **Location**: Section 4.2 (lines 44-48), Section 4.7 (line 187, H1 "Pending")
- **Severity**: Critical
- **Detail**: The paper's two proposed methods (BSD and A-CFG) have only been evaluated at pilot scale (n=16). The section acknowledges this repeatedly but does not adequately convey why publication-ready conclusions are being drawn from n=16 experiments. The statistical analysis paragraph (lines 54) correctly notes that no pairwise comparison reaches significance, yet the narrative throughout Sections 4.3-4.4 speaks as if 12.5% vs 0% differences are reliable. At n=16, 12.5% accuracy means exactly 2 correct out of 16 --- the difference between 0/16 and 2/16 is within random fluctuation.
- **Suggestion**: Add a "Limitations of Pilot Evaluation" paragraph early in 4.2 (not just in the statistical analysis) that explicitly states: (1) the minimum detectable effect at n=16, (2) that all proposed-method results should be interpreted as directional evidence, not confirmed improvements, and (3) a concrete plan/timeline for full-scale validation. Consider moving full-scale validation to a "required before camera-ready" status.

---

## Major Issues

### M1. Vanilla baseline inconsistency across pilot and full-scale
- **Location**: Table 1 (lines 35-36) vs ablation tables
- **Severity**: Major
- **Detail**: Vanilla achieves 4.7% on Countdown-500 but 0.0% on Countdown-16 in some tables and 0.0% in the BSD k-parameter ablation (line 67). If vanilla gets 0/16 on the pilot but 4.7% on 500 samples, this suggests the 16-sample pilot is not representative. This undermines all pilot-based comparisons. The text does not discuss why the pilot baseline differs so dramatically from the full-scale baseline.
- **Suggestion**: Add a paragraph discussing pilot-to-full-scale baseline calibration. Explain whether the 16 samples were randomly drawn or selected. If vanilla gets 0/16 on the pilot but ~23/500 on the full evaluation, the pilot sample may be biased toward harder problems. This is essential context for interpreting all pilot results.

### M2. Compute fairness analysis undermines the paper's contribution
- **Location**: Section 4.5 (lines 149-164)
- **Severity**: Major
- **Detail**: The compute-fair analysis shows that vanilla with matched FLOPs is competitive with or equal to all proposed methods. BSD is *worse* than matched-FLOPs vanilla (-6.2pp). A-CFG ties vanilla at 2x FLOPs. This is a devastating result that the text acknowledges but then hand-waves with "caveats" about sample size and full-scale DMI results. A hostile reviewer will read Table 2 as: "simply running vanilla for more steps achieves the same or better results --- why bother with BSD/A-CFG?"
- **Suggestion**: The narrative needs to be restructured. Lead with the full-scale DMI result (where compute fairness genuinely favors the method), then frame BSD/A-CFG compute fairness as an open question pending full-scale validation. Consider adding a Pareto frontier figure that shows DMI's position at 1.05x FLOPs, which is the strongest argument for the approach.

### M3. BSD+A-CFG combination failure is under-analyzed in this section
- **Location**: Line 52, Section 4.2
- **Severity**: Major
- **Detail**: The combination achieving 6.2% (below both individual methods) is described in one sentence and deferred entirely to Section 5. However, this is a central experimental finding that directly tests one of the paper's hypotheses (H7). The experiments section should present the empirical evidence more thoroughly --- at minimum, show the quality metrics, entropy trajectories, or any diagnostic data for the combination.
- **Suggestion**: Add a subsection 4.3.6 "Combination Analysis" with: (a) quality metrics for BSD+A-CFG, (b) belief entropy trajectory comparison (BSD alone vs BSD+A-CFG), (c) per-sample flip analysis (which problems does the combination solve/break compared to individual methods). Reserve the mechanistic explanation for Section 5, but present the evidence here.

### M4. Missing error bars and confidence intervals in ablation tables
- **Location**: Sections 4.3.1-4.3.5 (lines 64-124)
- **Severity**: Major
- **Detail**: All ablation results are reported as point estimates without any uncertainty quantification. At n=16, the difference between 0% (0/16) and 6.2% (1/16) and 12.5% (2/16) is 1-2 samples. Without CIs or at minimum bootstrap intervals, readers cannot assess whether observed differences are meaningful.
- **Suggestion**: Add bootstrap 95% CIs to all ablation accuracy values. Even if CIs are wide (which they will be), showing them is more honest than omitting them. Alternatively, report raw counts (e.g., "1/16" instead of "6.2%") to make the sample size viscerally clear.

---

## Minor Issues

### m1. Citation placeholders in baselines
- **Location**: Line 13, `\citep[our prior iteration]{}`
- **Severity**: Minor
- **Detail**: DMI and DTA have empty citation keys `\citep[our prior iteration]{}`. If these are from prior iterations of this project (not published), this should be made explicit (e.g., "from our prior iteration, detailed in Appendix C") rather than using an empty citation.
- **Suggestion**: Replace with a self-citation to the appendix or a footnote explaining these are prior unpublished iterations.

### m2. Inconsistent notation for k parameter
- **Location**: Lines 19, 60-71
- **Severity**: Minor
- **Detail**: The setup (line 19) says "k=0.75" but the ablation (line 69) uses "T/4 x 3" and "3T/4". Line 62 explains the mapping but the two notations are confusing. Is k a fraction of T or a step number?
- **Suggestion**: Standardize to one notation. Use k as a fraction (k=0.25, 0.5, 0.75) throughout, with the "Belief Phase %" column providing the intuitive interpretation.

### m3. Figure 4 is only described, not rendered
- **Location**: Line 126-128
- **Severity**: Minor
- **Detail**: Figure 4 is described via a blockquote placeholder. This is acceptable for a draft but should be flagged --- the ablation grid is critical for visual comprehension of the results.
- **Suggestion**: Ensure the generation script (`gen_fig4_ablation_grid.py`) is prioritized. The ablation results are the section's strongest content and deserve proper visualization.

### m4. Table 1 missing data for remasking baselines on pilot
- **Location**: Lines 36-37
- **Severity**: Minor
- **Detail**: ReMDM-conf and RCR have "---" for Countdown-16 and GSM8K-16. Were they not evaluated on the pilot? If so, why not? This makes the grouping comparison ("cross-step > remasking") asymmetric.
- **Suggestion**: Either run the baselines on the pilot or explain why they were omitted (e.g., "evaluated only at full scale in prior iterations").

### m5. Section 4.6 information-theoretic validation could be an ablation subsection
- **Location**: Lines 166-177
- **Severity**: Minor
- **Detail**: Section 4.6 is really a validation of BSD's mechanism, not a standalone experiment. It fits more naturally as Section 4.3.6 "BSD Information-Theoretic Validation" or as part of the Method section (where the outline places Figure 3).
- **Suggestion**: Consider moving to a BSD-specific analysis subsection or at minimum adding a transition sentence explaining why this appears in Experiments rather than Method.

### m6. Hypothesis table placement
- **Location**: Lines 179-199
- **Severity**: Minor
- **Detail**: The hypothesis verification summary (Table 3) is in Section 4.7 but the outline suggests it could go in Discussion or Appendix. Its current placement adds bulk to an already long section (207 lines). The detailed verdict explanations are somewhat redundant with the ablation subsections.
- **Suggestion**: Keep the table but shorten the follow-up paragraph (lines 199). The reader has already seen all the evidence; a brief sentence summarizing the 4/4/1/2 split suffices.

### m7. "Flip analysis" terminology introduced without definition
- **Location**: Line 143
- **Severity**: Minor
- **Detail**: "The flip analysis supports A-CFG's generalization: of the 4 disagreements between A-CFG and vanilla..." introduces the concept of flip analysis without defining it earlier. This is a useful analysis but needs a brief setup.
- **Suggestion**: Add one sentence defining flip analysis when first used: "We perform a flip analysis, examining the subset of problems where two methods disagree (one correct, one incorrect)."

---

## Visual Communication Assessment

### Strengths
- The Figure & Table plan is thorough (3 tables + 2 figures in this section).
- Table 1 provides a comprehensive single-point comparison across all methods.
- Figure 4 (ablation grid) is well-designed for rapid comprehension.
- Figure 5 (generalization) directly addresses a key question.

### Weaknesses
- **No Pareto frontier figure**: Section 4.5 discusses Pareto optimality verbally but does not include a figure. A FLOPs-vs-accuracy scatter with Pareto frontier would be far more effective than Table 2 alone.
- **No per-sample analysis visualization**: At n=16, showing which individual problems each method solves (e.g., a binary heatmap: methods x problems) would be more informative than aggregate accuracy percentages.
- **Missing confidence interval visualization**: Bootstrap CIs are described in the statistical analysis but never plotted. A forest plot of effect sizes with CIs would powerfully convey the uncertainty.

---

## Score Justification: 7/10

**Strengths** (+): Comprehensive coverage of all methods and ablations; honest reporting of negative results and falsified hypotheses; systematic hypothesis tracking; appropriate statistical protocol described; compute fairness analysis included; information-theoretic validation provides mechanistic insight.

**Weaknesses** (-): Critical issue of mixing pilot and full-scale results in one table; no full-scale validation of proposed methods; compute fairness results undermine contribution without adequate framing; ablation tables lack error bars; baseline inconsistency (0% vs 4.7% vanilla) unexplained; combination failure under-analyzed in this section.

The section is a solid draft that would benefit substantially from (1) separating pilot and full-scale results, (2) adding uncertainty quantification to all ablations, and (3) restructuring the compute fairness narrative. With these revisions, it could reach 8-9/10.
