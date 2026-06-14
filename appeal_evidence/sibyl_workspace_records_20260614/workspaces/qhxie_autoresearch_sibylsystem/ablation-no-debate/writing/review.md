# Writing Quality Review

## Summary

This paper introduces multi-child proportional ablation as a new measurement methodology for feature absorption in Sparse Autoencoders (SAEs). On synthetic feature hierarchies with ground truth, the authors show that trained SAEs exhibit absorption rates of 0.50 versus 0.059 for random decoders (Cohen's d = 8.94), decompose absorption into geometric (0.299) and learned (0.185) contributions via a 2x2 factorial design, validate causal consequences through steering (1.62x sensitivity ratio for absorbed features), and find no elevated absorption risk for safety-critical features (p = 0.665). The paper also reports negative results: absorption does not follow competitive exclusion dynamics (rho = +0.17, opposite the predicted direction).

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a clear problem-approach-evidence-conclusion arc. The introduction frames four open questions, the methodology section addresses each with a specific experimental design, and the experiments section reports results in a consistent format (hypothesis, falsification criterion, procedure, results, verdict). Transitions between sections are functional if not elegant. The abstract accurately represents the paper's content and results, including both supported and falsified hypotheses.

One structural issue: the paper contains no Related Work section as a standalone section. Background material is distributed across subsections of Section 2, which works but makes the paper feel front-loaded with citations before the reader understands the methodology. This is a minor issue.

A more notable issue: the Conclusion (Section 5.8) restates findings from the Abstract almost verbatim in places. The final paragraph ("Ultimately, this work demonstrates...") adds little beyond what Section 5.1-5.4 already covered. Some trimming would improve the ending.

### Notation & Terminology Consistency: 7/10

The paper maintains reasonable notation consistency, but several issues exist:

1. **Mixed notation for cosine similarity**: Section 3.1 uses `cos(p, c_1)` in the table but `cos(p, {c_1, c_2}) = 0.67` in Section 4.1. The latter is ambiguous -- does it mean cosine with each child individually, or with the set?

2. **Inconsistent subscripting**: `d_{sae}` vs `d_{model}` -- the latter should probably be `d_{model}` consistently, but the paper uses `d=512` in Section 5.6 without the subscript.

3. **"abs_k" vs "absorption rate"**: The paper uses `abs_k` in equations but "absorption rate" in tables and text. This is acceptable but could be tighter.

4. **"H_Mech" vs "H_mech"**: Section 3.5 labels the hypothesis "H_Mech" but the figure filename uses "fig5_h_mech_factorial.pdf" (lowercase 'm'). The paper is internally consistent on "H_Mech" in text, but the filename discrepancy suggests a notation drift that could confuse figure cross-referencing in LaTeX compilation.

5. **"act(p)" undefined**: Equation for `abs_k` uses `act(p)` without defining it. From context it appears to mean activation magnitude, but this should be explicit.

6. **"overlap method" undefined**: Section 3.7 mentions measuring absorption "via the overlap method" without defining it. This term does not appear elsewhere in the paper.

### Claim-Evidence Integrity: 7/10

Most claims are well-supported with specific numbers and citations to tables/figures. However, several integrity issues exist:

1. **Table 6 steering verification table contains degenerate values**: The "Mean Delta Percent" column shows values like "134,717,856%" and "1.52 billion%". These are clearly computation artifacts (division by near-zero baseline), yet they are presented without comment. A reviewer would immediately flag this as either a bug or a failure to normalize correctly. The paper should either fix the computation, explain why these percentages are not meaningful, or remove the column.

2. **H3 sample size inconsistency**: Section 3.6 says "We test n=20 absorbed and n=20 non-absorbed features." Section 4.5.1 says the same. But the visual audit mentions "corrected H3 implementation uses n=20 absorbed and n=20 non-absorbed features, eliminating the severe power limitation of the earlier n=7 analysis." The paper does not mention the earlier n=7 analysis anywhere in the main text. This is fine (the final paper reports the correct numbers), but the reader may wonder why the sample size is exactly 20 -- was this pre-registered or determined by power analysis? The pre-registration in Section 3.8 does not specify sample sizes for H3.

3. **Table 1 shows Std = 0.0000 for Trained SAE**: This suggests the value 0.5000 is exact (possibly averaged across seeds with no variance, or a single measurement). Given that other conditions have non-zero standard deviations, this zero is suspicious. If it's because the value is exact across all 5 seeds, the paper should state that explicitly. If it's a single measurement, standard deviation should not be reported.

4. **Table 5 condition C has Std = 0.010**: This is identical to condition A. Given that condition C uses a random encoder with trained decoder, some variance across seeds is expected. The identical standard deviations to 3 decimal places is suspicious and should be verified.

5. **"p < 10^-133" in Abstract vs "3.16e-133" in Table 2**: These are consistent (3.16e-133 < 10^-133), but the Abstract rounds to a cleaner bound while the table gives the exact value. This is acceptable but slightly inconsistent style.

6. **Missing source data reference**: The paper references `exp/results/summary.md` in the task template, but this file does not exist in the workspace. The results appear to come from JSON files in `iter_001/data/` and `iter_001/exp/results/`, but the paper does not provide a data availability statement or reference these sources.

### Visual Communication: 8/10

The paper has 7 figures and 8 tables, meeting the minimum requirements (method diagram, results table, analysis figure). All figures are referenced before they appear. Captions are self-explanatory and include key takeaways. The visual audit confirms completeness and proper ordering.

Issues:

1. **Figure 1 is referenced but not described in text**: Section 3.4 says "Figure 1: Multi-Child Proportional Ablation Procedure" and embeds the PDF, but there is no textual description of what the figure shows. The caption is present, but the text should walk the reader through the figure.

2. **Table 6's degenerate percentage column** (noted above) is a visual communication issue -- it undermines confidence in the steering results.

3. **No figure for the synthetic hierarchy geometry**: Section 3.1 describes a 3-level hierarchy with specific cosine constraints. A simple diagram showing parent, children, and grandchildren with angle annotations would help readers visualize the setup. This is not critical but would improve clarity.

### Writing Quality: 7/10

The writing is generally clear and direct, with good use of specific numbers. However, several issues exist:

**Banned patterns found:**

1. "Despite growing interest in absorption, the field faces a **measurement crisis**." (Introduction, paragraph 3) -- "growing interest" is a mild generic opening. Not the worst offender but could be tighter.

2. "This finding is consistent with the interpretation that..." (Section 5.3) -- filler hedging. Just state the interpretation.

3. "Several interpretations are possible." (Section 5.2) -- weak transition. The paragraph that follows presents hypotheses, not interpretations.

**Unclear sentences:**

1. "The ecological analogy between feature absorption and biological competitive exclusion has been proposed as a theoretical framework for predicting when absorption occurs (Korznikov et al., 2025; Kalmykov & Kalmykov, 2012)." (Section 2.5) -- This sentence attributes the ecological analogy to two papers, but Kalmykov & Kalmykov (2012) is a physics paper on competitive exclusion principle, not an SAE paper. The citation seems mismatched or the analogy attribution is unclear.

2. "The parent-children cosine similarity of 0.67 means the parent is a weighted sum of its children, consistent with the absorption hypothesis." (Section 3.1) -- A cosine similarity of 0.67 does not mean the parent is a weighted sum. It means the parent has a 0.67 cosine with each child, which is consistent with the parent being in the span of the children, but "weighted sum" is an overclaim.

3. "Steering verification confirms that steering changes activations measurably:" (Section 4.5.2) -- This is tautological. The table that follows should speak for itself.

**Passive voice overuse:**

Several sentences use passive constructions where active would be clearer:
- "absorption was first systematically documented by Chanin et al." (Section 2.2) -- acceptable
- "The absorption phenomenon was first systematically documented" -- could be "Chanin et al. first documented absorption systematically"

**Jargon density:**

Section 2.5 on competitive exclusion introduces Lotka-Volterra dynamics without explaining them. Readers from ML backgrounds may not know this refers to predator-prey models. A brief parenthetical would help.

## Issues for the Editor

1. **[Critical] Table 6 degenerate percentage values**: Section 4.5.2, steering verification table. The "Mean Delta Percent" column shows impossible values (134 million %, 1.52 billion %). These are clearly computational artifacts from dividing by near-zero baseline. **Fix**: Remove the percentage column entirely, or recalculate using a meaningful baseline (e.g., percent change relative to non-zero alpha=0.5 baseline, or simply report absolute delta norms which are already present and meaningful).

2. **[Major] "overlap method" undefined**: Section 3.7, safety-critical feature analysis. The paper states absorption is measured "via the overlap method" but never defines this method. Given that the rest of the paper uses multi-child proportional ablation, the sudden switch to "overlap method" for H_Safe is confusing. **Fix**: Define the overlap method in Section 3.7, or if it is equivalent to the multi-child method, state that explicitly. If it is a different method, explain why a different method was used for real-model features.

3. **[Major] Table 1 zero standard deviation**: Section 4.2.2, Table 1. The Trained SAE row shows Std = 0.0000, which is either exact (unlikely across 5 seeds) or indicates a single measurement was reported with a misleading std column. **Fix**: Verify the source data. If the value is truly constant across seeds, add a footnote stating "Constant across all seeds." If it's a single measurement, remove the Std column for that row or report N/A.

4. **[Major] Figure 1 lacks textual walkthrough**: Section 3.4. The figure is embedded with only a caption, no textual description. **Fix**: Add 2-3 sentences describing what Figure 1 illustrates -- e.g., "Figure 1 illustrates the multi-child proportional ablation pipeline. Panel (a) shows the synthetic hierarchy construction... Panel (b) shows the ablation procedure..."

5. **[Minor] Abstract line 5 is overloaded**: The abstract packs five hypotheses with results into a single long sentence. **Fix**: Split into two sentences for readability -- one for supported hypotheses, one for falsified ones.

## What Works Well

1. **Honest reporting of negative results**: Sections 4.3 and 4.6 report falsified hypotheses (H2, H_Safe) with the same rigor as supported hypotheses. The abstract and conclusion prominently feature these negative results, which is excellent scientific practice and builds reader trust.

2. **Consistent experimental reporting format**: Each hypothesis section follows the same structure (hypothesis, falsification criterion, procedure, results, verdict). This makes the paper easy to scan and compare across experiments.

3. **Visual audit quality**: The figure and table plan is well-executed. All 7 figures and 8 tables are present, properly numbered, referenced before appearance, and have self-explanatory captions. The visual narrative effectively supports the text.

SCORE: 7
