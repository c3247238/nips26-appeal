# Critique: Experiments and Results (Section 5)

## Summary Assessment

The experiments section suffers from a fundamental structural mismatch with the rest of the paper. While the Introduction, Method, Discussion, and Conclusion have all been restructured around the H6-H10 "Local Inhibition Graph" framework, the Experiments section still presents only H1-H5 results from prior work. The H6-H10 validation experiments are described in the Method (Section 4) but have no corresponding results in Section 5. This renders the paper's central claims unverified and creates a severe coherence problem that would likely result in rejection at any top-tier venue.

## Score: 3/10
**Justification**: The section is well-written in isolation---the H1-H5 results are presented clearly with proper statistical reporting. However, the section fails to deliver on the paper's stated contribution. A score of 3 reflects that the existing content is competent but the section is fundamentally incomplete for the paper it purports to be part of. To reach the next level, the section must include actual H6-H10 results with data-driven figures and tables.

## Critical Issues

### Issue 1: Missing H6-H10 Results (Central Claim Unverified)
- **Location**: Entire Section 5.2 and beyond
- **Quote**: "The H6--H10 experiments address this limitation by changing the prediction target..." (Section 5.4) --- but no H6-H10 results are actually presented.
- **Problem**: The paper's title, abstract, introduction, and method all promise validation of the Local Inhibition Graph framework through H6-H10. The Method section (4.2-4.8) describes six experimental phases with specific predictions. Yet Section 5 contains only H1-H5 results from prior experiments. The central claim---that "edges in the local inhibition graph predict known absorption pairs"---has zero empirical support in the results section.
- **Cross-section impact**: The Introduction (1.5) promises "precision@20 = X.XX vs. 0.004 chance" and the Conclusion (8.1) states "We constructed a local inhibition graph from decoder correlations and validated its predictive power against known absorption pairs." Neither claim is supported by data in the Experiments section.
- **Fix**: Either (a) run the H6-H10 experiments and populate Section 5 with actual results, or (b) restructure the entire paper to be a purely theoretical/methodological contribution with no empirical validation claims. Option (a) is strongly preferred.

### Issue 2: Section 5.2 Describes Experiments Instead of Reporting Results
- **Location**: Section 5.2, "Validation Protocol: Testing the Inhibition Framework"
- **Quote**: "**Prediction.** Precision@20 >= 0.10 (25x enrichment over chance). If precision@20 <= 0.05, the structural correspondence fails..." (H6 subsection)
- **Problem**: This subsection describes experimental design and predictions but contains NO actual results. It reads like a methodology section that was accidentally placed in the results. Every subsection in 5.2 (H6-H10) follows the same pattern: ground truth, metrics, prediction---but no numbers, no figures, no statistical tests.
- **Fix**: Replace the prediction text with actual results. For H6: report precision@k values, AUPR, Fisher exact test p-values, enrichment factors. For H7: report Pearson r and p-values for inhibition vs. recall/precision correlations. For H8-H10: report the actual statistics computed from the data.

### Issue 3: Placeholder Values in Key Results Preview
- **Location**: Intro Section 1.5, cross-referenced throughout
- **Quote**: "precision@20 = X.XX vs. 0.004 chance (XX-fold enrichment)" --- the "X.XX" and "XX" are literal placeholders.
- **Problem**: These placeholders propagate through the entire paper. The Introduction promises specific results that the Experiments section never delivers. This is not a draft issue---it is a structural gap between what the paper claims and what it contains.
- **Fix**: Run the experiments, fill in the actual numbers, and update all placeholder references throughout the paper.

### Issue 4: Figure/Table Inventory Mismatch
- **Location**: Section 5 figure/table comments and outline.md
- **Quote**: "<!-- FIGURES - None (data-driven figures for H6-H10 pending experiment execution) -->" (end of Section 5)
- **Problem**: The outline.md plans Figures 2-5 and Tables 1-3 for H6-H10 results. The actual paper has Figure 1 (LCA correspondence) and Figure 2 (suppression mechanism) from the theoretical framework, but no data-driven figures for H6-H10. Table 1 in the experiments section is actually the H1-H3 hypothesis summary, not the H6-H10 summary planned in the outline.
- **Fix**: Execute the experiments and generate the planned figures and tables. The figure filename mismatches noted in the cross-cutting critique (fig3/fig4/fig5 numbering) may become moot once the correct figures are produced.

## Major Issues

### Issue 5: H1-H5 Results Are Framed as "Prior Work" But Occupy the Entire Results Section
- **Location**: Section 5.1
- **Problem**: The H1-H5 results (absorption detection, random baseline validation, steering/probing correlations, precision-recall asymmetry) are presented as empirical context for the inhibition framework. However, they consume the entire results section, leaving no space for the actual validation experiments. The framing in 5.1 ("Our experiments build on a systematic measurement...then present the validation experiments") promises a transition to new results that never comes.
- **Fix**: Condense H1-H5 into a 1-paragraph summary (these are genuinely prior results that established the phenomenon). Dedicate the bulk of Section 5 to H6-H10 results.

### Issue 6: Table Numbering Divergence
- **Location**: Throughout Section 5
- **Problem**: The experiments section contains Tables 1-5 (H1-H5 results), but the outline plans Tables 1-3 for H6-H10 results. The actual Table 1 (H1-H3 summary) conflicts with the outline's planned Table 1 (H6-H10 summary). This creates confusion about which tables belong to which framework.
- **Fix**: Renumber tables to clearly distinguish "prior results" tables from "current study" tables. Alternatively, move H1-H5 tables to an appendix and use Table 1-3 for H6-H10 as planned.

### Issue 7: Missing Statistical Details for H6-H10
- **Location**: Section 5.2 (H6-H10 subsections)
- **Problem**: The H6-H10 subsections lack the statistical rigor present in H1-H5. For example, H6 mentions a Fisher exact test but gives no alpha level, no power analysis, and no correction for multiple comparisons across k values. H7 gives correlation thresholds but no justification for the r > 0.3 / r < -0.3 thresholds.
- **Fix**: Add statistical detail matching the rigor of H1-H5: sample sizes, test statistics, effect sizes, power analyses, and multiple comparison corrections where applicable.

### Issue 8: H1b Labeled "Supported" Without Multiple Comparison Correction (from prior critique, still unresolved)
- **Location**: Table 1, H1b row
- **Quote**: "H1b (Delta steering) | 8 | -0.431 | 0.028 | 0.186 | **Supported**"
- **Problem**: The H1b result (p = 0.028) is labeled "Supported" but the paper elsewhere states that no hypothesis survives multiple comparison correction. With 12 tests, Bonferroni and BH-FDR yield no significant results. Labeling H1b as "Supported" without noting the correction failure is misleading.
- **Fix**: Add a footnote: "H1b at layer 8 passes the uncorrected threshold (p < 0.05) but does not survive Bonferroni or BH-FDR correction for 12 tests." Or add a "Corrected p-value" column.

## Minor Issues

- **Section 5.1, paragraph 3**: "Raw steering success rates at s = 50 ranged from 0.40 to 1.00" --- no citation or table reference for this claim. Add reference to the source data.
- **Section 5.4**: "The H6--H10 experiments address this limitation..." --- this paragraph discusses H6-H10 as if they have been run, but they have not. This is misleading.
- **Table 2**: The "Supporting Evidence" column cites H1-H5 results, which is appropriate for an integration table, but the table appears in Section 5.3 which is supposed to present new results. Consider moving Table 2 to the Discussion section.
- **Section 5.1, Table 3**: The maximum absorption rate (0.242 for feature U at layer 8) is highlighted in the caption but not in the table. Consider bolding the cell.
- **Section 5.1, paragraph 5**: "This limited variance constrains correlation analyses but is itself informative" --- Good insight, but add a quantitative framing: "Only 6/26 features (23%) at layer 4 exceed the 10% threshold."

## Visual Element Assessment
- [ ] Figures/tables match outline plan: **NO** --- outline plans H6-H10 figures/tables; paper delivers H1-H5 tables only
- [ ] All visuals referenced before appearing: N/A (no H6-H10 visuals exist)
- [x] Captions are self-explanatory: Existing captions are adequate for H1-H5 content
- [ ] No text-heavy sections that need visual support: Section 5.2 is entirely text descriptions of experiments with no results visuals --- this is the most critical gap

## What Works Well

1. **H1-H5 statistical reporting is rigorous**: Table 1 presents correlation coefficients, p-values, R-squared values, and clear result labels. The random baseline validation (Table 4) includes t-statistics and Cohen's d. This level of rigor should be maintained for H6-H10.

2. **Precision-recall decomposition (Table 5) is well-presented**: The asymmetry is clearly demonstrated with specific numbers (precision std 0.028-0.054 vs. recall std 0.192-0.199), and the explanation connects directly to the competitive suppression theory.

3. **Integration table (Table 2) effectively bridges old and new**: Mapping prior findings to inhibition explanations demonstrates how the new framework unifies existing observations. This is a strong structural element that should be retained.
