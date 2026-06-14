# Critique: Results (Experiments)

## Summary Assessment

The Results section presents a compelling narrative with strong quantitative claims, but it suffers from a critical data integrity problem: the section reports full 5-replicate results for all 7 conditions (including MultiScale, Gating, and Full Matryoshka), yet the underlying data files confirm only 3 variants (Baseline, TopK, Orthogonality) in the canonical `full_summary.json`. The remaining 4 variants' data exist in individual seed files but were apparently aggregated manually for the paper. This creates a discrepancy between what the Method section claims ("3 of 6 variants have full 5-replicate data") and what the Results section presents (all 7 conditions with means and standard deviations). Additionally, there are numerical inconsistencies in Cohen's d values between sections, correlation point counts that don't match, and a scope note in the Method that directly contradicts the Results' confident claims.

## Score: 5/10

**Justification**: The section has strong structural organization, clear hypothesis testing, and good visual communication. However, the data integrity issues---reporting full statistics for variants whose completion status is disputed between Method and Results---are serious enough that a reviewer would demand clarification. The numerical inconsistencies (Cohen's d values, correlation statistics) across sections further erode confidence. To reach 7/10, the section needs to reconcile its claims with the actual data availability, fix all numerical inconsistencies, and either remove provisional data or clearly mark it as such.

---

## Critical Issues

### Issue 1: Data Integrity -- Reporting Full Statistics for Variants Method Claims Are Incomplete

- **Location**: Table 3 (lines 24-32), Sections 4.2-4.7
- **Quote**: "Table 3 reports the full experiment results for all 7 conditions (5 replicates each, seeds 42, 123, 456, 789, 1011)." (line 22) and "+MultiScale | **0.055 +/- 0.024** | 0.220 +/- 0.001 | 7.10 +/- 0.30 | **50.0** | 0.235 +/- 0.019 | **78.3%** | **4.81**" (line 28)
- **Problem**: The Method section (Section 3.2, line 26) explicitly states: "At the time of writing, full 5-replicate experiments are complete for Baseline, +TopK, and +Orthogonality. +MultiScale has pilot data (single replicate, 1,024 features) but the full experiment is pending. +Gating and +Full Matryoshka are listed in the design but have not yet been trained." Yet the Results section reports full 5-replicate statistics for MultiScale (0.055 +/- 0.024), Gating (0.261 +/- 0.050), Full Matryoshka (0.066 +/- 0.029), and Random (0.534 +/- 0.050) as if they were complete. I verified the raw data: individual seed files exist for all 4 remaining variants across all 5 seeds, and my own aggregation confirms the reported means are approximately correct (MultiScale: 0.0548 +/- 0.0268; Gating: 0.2613 +/- 0.0555; Matryoshka: 0.0662 +/- 0.0326; Random: 0.5343 +/- 0.0561). However, the Method's scope note creates a direct contradiction that undermines reader trust.
- **Fix**: Either (a) update the Method section's scope note to reflect that all variants now have full data, or (b) if the data was aggregated post-hoc without going through the same analysis pipeline as the canonical summary, add a footnote to Table 3 explaining the data source for each variant group. The cleanest fix is to regenerate `full_summary.json` to include all 7 variants and update the Method scope note accordingly.

### Issue 2: Inconsistent Cohen's d Values Across Sections

- **Location**: Table 3, Section 4.3, Abstract, Intro, Discussion
- **Quote**: Table 3 shows TopK d = 4.93, but the Abstract states "Cohen's d = 4.93 and 4.81" (for TopK and MultiScale), while the Intro (Section 1.6, line 68) states "Cohen's d = 5.51", and the Discussion (Section 5.1, line 6) states "Cohen's d = 5.51".
- **Problem**: The TopK Cohen's d appears as three different values: 4.93 (Results/Abstract), 4.81 (MultiScale in Results), and 5.51 (Intro/Discussion). Checking the raw data: `full_summary.json` lists TopK cohens_d = 5.5086567338514865. The value 4.93 does not appear in the canonical data file. This is a serious numerical inconsistency that suggests either (a) the Results section used a different calculation method, (b) the numbers were manually edited and diverged from the source, or (c) there was a copy-paste error from an earlier draft.
- **Fix**: Use the value from `full_summary.json` (d = 5.51) consistently across all sections. Regenerate all Cohen's d values from the canonical statistical analysis output and audit every numerical claim in the paper against the source data files.

### Issue 3: Correlation Statistics Inconsistent Between Results and Discussion

- **Location**: Section 4.8 (line 86) vs. Discussion Section 5.1 (line 9)
- **Quote**: Results: "r = 0.865, p = 0.012 across all 7 variants" (line 86); Discussion: "r approx +0.93 across n = 4 variants, p = 0.067" (line 9)
- **Problem**: The Results section reports a Pearson correlation of r = 0.865 across n = 7 variants (p = 0.012), while the Discussion section reports r approx +0.93 across n = 4 variants (p = 0.067). These are completely different analyses with different sample sizes, different correlation values, and different significance levels. The n = 4 version appears to be from an earlier draft when only Baseline, TopK, Orthogonality, and Random had data. The n = 7 version uses all variants. Both cannot be correct descriptions of "the" correlation.
- **Fix**: Standardize on the n = 7 analysis (r = 0.865, p = 0.012) since all variants now have data. Remove the n = 4 version from the Discussion. If the n = 4 analysis is retained for historical comparison, it must be explicitly labeled as "from preliminary analysis with 4 variants."

---

## Major Issues

### Issue 4: Method Scope Note Directly Contradicts Results Claims

- **Location**: Method Section 3.2 (line 26) vs. entire Results section
- **Quote**: "Results reported in this paper are therefore provisional and the component ranking may change when the full variant set is completed." (Method)
- **Problem**: The Method section tells readers the results are provisional because only 3 of 6 variants have full data. The Results section presents all 7 conditions with confidence, hypothesis tests, and definitive language ("H1 is SUPPORTED", "H2 is SUPPORTED"). This contradiction is confusing: should readers trust the hypothesis tests or treat them as provisional?
- **Fix**: If all variants truly have full 5-replicate data now, remove the provisional language from the Method. If some variants' data is considered less reliable, add a footnote to Table 3 and the hypothesis test sections indicating which variants have canonical pipeline output vs. manually aggregated data.

### Issue 5: Missing Statistical Test Details for MultiScale, Gating, Matryoshka, Random

- **Location**: Sections 4.4-4.7, Table 4
- **Problem**: The Results section reports p-values for Orthogonality (p = 0.845) and Gating (p = 0.797) vs. Baseline, but the canonical `statistical_analysis.json` only contains data for Baseline, TopK, and Orthogonality. The p-values for MultiScale, Gating, Matryoshka, and Random are not traceable to the canonical analysis output. A reviewer would ask: how were these p-values computed? Were they from the same ANOVA/Tukey HSD pipeline or from separate t-tests?
- **Fix**: Either (a) regenerate the full statistical analysis with all 7 variants in the canonical pipeline and cite the output file, or (b) for each p-value reported, explicitly state the test used (e.g., "independent two-sample t-test, Welch's correction") and degrees of freedom.

### Issue 6: Additive Expectation Computation Is Mathematically Dubious

- **Location**: Section 4.10 (lines 110-111)
- **Quote**: "The additive expectation for Full Matryoshka, computed from the individual reductions of TopK and MultiScale, is negative (-0.142), which is physically impossible for an absorption rate bounded at zero."
- **Problem**: The additive expectation of -0.142 is computed as Baseline - (Baseline - TopK) - (Baseline - MultiScale) = 0.252 - 0.196 - 0.197 = -0.141. This is indeed below zero. However, the paper uses this impossible value to argue for "antagonistic interaction" without acknowledging that additive expectations on bounded metrics are inherently problematic. A more appropriate analysis would use a multiplicative model (relative risk) or log-odds, which respect the [0,1] bound.
- **Fix**: Add a sentence acknowledging that additive models on bounded metrics have limitations, and consider supplementing with a relative risk framing: "The observed relative risk is 0.066/0.055 = 1.20, indicating Matryoshka is 20% worse than MultiScale alone."

### Issue 7: Figure References Use PDF Placeholders, Not Actual Generated Figures

- **Location**: Lines 40, 84, 96, 106, 120
- **Quote**: "![Figure 2: Absorption rate by SAE variant with error bars...](figures/fig2_absorption_bars.pdf)"
- **Problem**: The figure references use PDF filenames as placeholders. While this is acceptable for a draft, the captions are embedded in the markdown image syntax rather than as proper LaTeX figure environments. More critically, there is no verification that these figures actually exist or match the data in Table 3.
- **Fix**: Verify that each referenced figure file exists and its content matches the data in Table 3. For the final LaTeX version, convert these to proper `\begin{figure}` environments.

---

## Minor Issues

- **Line 1**: Section heading "# 4. Results" is fine, but the file is named `experiments.md` while the content is labeled "Results". Standardize file names with section titles.
- **Line 27**: "MSE ($\times 10^{-3}$)" column header -- the values in the column (10.44, 7.68, etc.) are already scaled, but the caption says "MSE values are reported as $\times 10^{-3}$ (e.g., 10.44 corresponds to MSE = 0.01044)". This is clear but slightly awkward. Consider using scientific notation in the table.
- **Line 34**: "MSE $\approx 3 \times 10^{-5}$" -- the table shows 0.03 +/- 0.00 in the $\times 10^{-3}$ column, which corresponds to 3e-5. This is correct but readers may miss the conversion.
- **Line 36**: "The Random control (absorption = 0.534)" -- the table shows 0.534 +/- 0.050. Using the mean without uncertainty in the text is acceptable for brevity but inconsistent with how other variants are discussed.
- **Line 46**: "$F(6, 28) = 73.36$" -- the degrees of freedom assume 7 groups x 5 replicates = 35 total, with df_between = 6, df_within = 28. This is correct only if all 7 variants have 5 replicates each. Verify this matches the actual data.
- **Line 54**: "Orthogonality and Gating are not significantly different from Baseline ($p > 0.79$)" -- the specific p-values (0.845, 0.797) are reported in the hypothesis sections, but "p > 0.79" is an unusual way to summarize. Consider "$p > 0.79$ for both" or list them separately.
- **Line 86**: "$r = 0.865$, $p = 0.012$ across $n = 7$ variants" -- with n = 7, df = 5. The critical value for r at alpha = 0.05 (two-tailed) with df = 5 is approximately 0.754. The observed r = 0.865 exceeds this, so p = 0.012 is plausible. However, n = 7 is extremely small for correlation claims. The section appropriately notes this is "suggestive" rather than definitive.
- **Line 101**: "TopK ($d = 4.93$) and MultiScale ($d = 4.81$) are in a class of their own, roughly 35x larger than Orthogonality ($d = 0.13$)" -- the ratio 4.93/0.13 = 37.9, so "roughly 35x" is approximately correct. But if using d = 5.51, the ratio is 5.51/0.13 = 42.4.
- **Line 108**: "The additive expectation... is negative (-0.142)" -- rechecking: 0.252 - (0.252-0.056) - (0.252-0.055) = 0.252 - 0.196 - 0.197 = -0.141. The paper says -0.142, which is close enough (rounding differences). But the text says "worse than either TopK (0.056) or MultiScale (0.055)" -- Matryoshka at 0.066 is indeed worse.
- **Line 116**: "MSE ~0.007--0.010" -- the table shows TopK 7.68e-3, MultiScale 7.10e-3, Gating 8.21e-3, Matryoshka 7.10e-3. Baseline is 10.44e-3. So the range is approximately 0.007--0.010. Correct.
- **Line 116**: "Random control: MSE = 0.649" -- table shows 649.1 x 10^-3 = 0.6491. The text says 0.649. The table header says the column is x 10^-3, so 649.1 means 0.6491. But wait -- the Random MSE in the raw data is 0.649121 (from my aggregation), which in x 10^-3 would be 649.1. The text says "MSE = 0.649" without the x 10^-3 qualifier, which is inconsistent with how other variants are discussed.
- **Lines 147-156**: The figure generation comments at the end are useful for the writing pipeline but should be removed in the final version.

---

## Visual Element Assessment

- [x] Figures/tables match outline plan -- Table 3 matches the outline's Table 1 specification. Figure 2-6 match the outline's Figure 2-6 plan. Table 4 (hypothesis summary) is a useful addition not in the outline.
- [x] All visuals referenced before appearance -- Each figure is referenced in text before the image markdown. Good.
- [~] Captions are self-explanatory -- Captions are descriptive but some rely on context from the main text (e.g., "Color coding: green = large effect" in Figure 2 caption assumes readers know the threshold values).
- [~] No text-heavy sections that need visual support -- Section 4.12 (MCC) and 4.13 (Hedging) are text-only summaries of invariant metrics. A small inline table or even a single sentence would suffice; the current paragraph-length treatment is disproportionate given the null findings.

---

## Cross-Section Consistency Issues

1. **Cohen's d values**: Results uses 4.93/4.81; Intro/Discussion uses 5.51. Must standardize on the canonical value from `full_summary.json` (5.51 for TopK).

2. **Correlation statistics**: Results uses r = 0.865, n = 7, p = 0.012; Discussion uses r approx 0.93, n = 4, p = 0.067. Must standardize.

3. **Scope/Provisional language**: Method says results are provisional (3 of 6 variants); Results presents all 7 with confidence. Must reconcile.

4. **MultiScale effect size**: Results Table 3 shows d = 4.81 for MultiScale; Discussion Section 5.4 says "effect size roughly five times larger than multi-scale decomposition" (referring to TopK's 5.51 vs MultiScale's pilot d approx 1.1). But the Results now reports MultiScale d = 4.81, which is nearly equal to TopK. The Discussion's claim that TopK is "roughly five times larger" is now incorrect.

5. **Hedging trade-off**: Results Section 4.13 says "hedging is invariant to architecture"; Discussion Section 5.3 discusses "The Missing Trade-off" citing Chanin et al. (2025). This is consistent.

6. **Terminology**: "absorption rate" is used consistently (not "absorption score"). "L0 sparsity" vs "L0" -- both used, should standardize. The glossary says "L0 sparsity" is preferred.

---

## What Works Well

1. **Clear hypothesis-test structure (Sections 4.4-4.7)**: Each hypothesis gets its own subsection with prediction, evidence, and decision. This is reviewer-friendly and makes the paper's logic transparent.

2. **Strong opening with pilot validation (Section 4.1)**: Starting with pilot results that informed the full design demonstrates scientific rigor and shows the authors thought about experimental design rather than just reporting outcomes.

3. **The sparsity-absorption correlation narrative (Section 4.8)**: This is the intellectual core of the paper. The scatter plot concept (Figure 3) effectively communicates that absorption may be a sparsity phenomenon, which reframes the field's research direction. The correlation is visually compelling even if statistically underpowered.
