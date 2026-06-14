# Critique: Experiments and Results

## Summary Assessment

The experiments section presents a well-structured summary of prior empirical findings (H1--H5) and a clear validation protocol for the new inhibition framework (H6--H10). The section successfully integrates old and new results through Table 2, which maps each prior finding to its inhibition explanation. However, there are critical issues regarding missing experimental results for H6--H10, inconsistent hypothesis numbering, and a structural problem where the section promises validation experiments that do not appear to have been executed.

## Score: 5/10
**Justification**: The section earns points for clear organization, accurate reporting of H1--H5 results, and the excellent integration table (Table 2). However, it loses significant points because (1) H6--H10 are described as validation protocols but no actual results are reported, (2) hypothesis numbering is inconsistent with the hypotheses section (which only defines H1--H3), and (3) the section contains placeholder values (X.XX) in the intro's key results preview that propagate into the text. To reach a 7+, the section needs actual H6--H10 results or a clear reframing that acknowledges these as proposed future experiments.

## Critical Issues

### Issue 1: H6--H10 Described as Validation Protocols But No Results Reported
- **Location**: Sections 5.2--5.4 and throughout
- **Quote**: "We now describe the validation experiments (H6--H10) that test whether decoder correlations predict absorption pairs, explain the precision--recall asymmetry, identify at-risk features, vary across layers, and enable post-hoc repair."
- **Problem**: The section describes experimental protocols for H6--H10 in detail (ground truth, metrics, predictions, falsification criteria) but never reports any actual results. Section 5.2 ends after describing the protocols without presenting precision@k values, Fisher test results, or AUPR. The reader is left expecting results that never arrive. The comment block at the end ("None (data-driven figures for H6-H10 pending experiment execution)") confirms these experiments were not run.
- **Fix**: Either (a) execute H6--H10 experiments and report results, or (b) reframe this section to clearly state that H6--H10 are *proposed* validation experiments that follow from the theory but were not executed in this study. If (b), move the detailed protocols to the Methodology section or an appendix, and use the Results section to discuss what the theory predicts and how future work could test it.

### Issue 2: Hypothesis Numbering Inconsistency
- **Location**: Throughout the paper
- **Quote**: Hypotheses section defines H1, H1b, H2, H3. Experiments section refers to H1--H5 and H6--H10.
- **Problem**: The hypotheses section (Section 3.2 / `hypotheses.md`) only defines H1, H1b, H2, and H3. There is no definition of H4, H5, H6, H7, H8, H9, or H10 anywhere in the paper's hypothesis section. Yet the experiments section freely references H1--H5 (from prior work) and H6--H10 (new validation experiments). H4 and H5 appear in `final_synthesis.json` (EC50 and precision-recall), but these are never formally defined in the hypotheses section. H6--H10 are defined only in the experiments section's protocol descriptions, not in a dedicated hypotheses section.
- **Fix**: Add a dedicated hypotheses section that defines all hypotheses H1--H10 with their formal statements, directional predictions, and falsification criteria. Alternatively, renumber the new hypotheses (e.g., H1'--H5') to avoid confusion with the prior hypothesis set, and clearly distinguish "prior hypotheses" from "validation hypotheses."

### Issue 3: Placeholder Values in Key Results Preview
- **Location**: Intro Section 1.5 and implied in experiments section structure
- **Quote**: "precision@20 = X.XX (vs. ~0.004 chance), a XX-fold enrichment" (from intro)
- **Problem**: The introduction's key results preview contains placeholder "X.XX" values for H6 results. These placeholders suggest the experiments were planned but not executed. The experiments section structure (with detailed protocols but no results) reinforces this impression.
- **Fix**: If H6--H10 experiments are not executed, remove all placeholder values from the intro and replace with language like "we propose five validation experiments" rather than "our validation experiments test."

## Major Issues

### Issue 4: Table 1 Reports H1b as "Supported" Without Multiple Comparison Caveat
- **Location**: Table 1, row H1b (Delta steering) at layer 8
- **Quote**: "H1b (Delta steering) | 8 | -0.431 | 0.028 | 0.186 | **Supported**"
- **Problem**: The table labels H1b at layer 8 as "Supported" based on uncorrected p = 0.028. However, the paper elsewhere acknowledges that this result does not survive multiple comparison correction (Section 5.4: "The H6--H10 experiments address this limitation by changing the prediction target" and the abstract: "this does not survive multiple comparison correction"). The "Supported" label in Table 1 without an immediate caveat is misleading. A reviewer would flag this as overclaiming.
- **Fix**: Change the Table 1 label to "Supported (uncorrected)" or "Partially supported" and add a footnote: "Significant at uncorrected threshold; does not survive Bonferroni or BH-FDR correction for 12 tests." This aligns with the more careful treatment in the abstract and Section 5.4.

### Issue 5: Section 5.1 Title Mismatch with Outline
- **Location**: Section heading "5.1 Empirical Context: Absorption Detection and Downstream Tasks"
- **Problem**: The outline (Section 5.1) calls for "Graph Construction Statistics" with a table of graph statistics by layer (mean edge weight, density, clustering). The actual section 5.1 is a summary of prior H1--H5 results. The graph construction statistics (H9) are never presented. This is a significant deviation from the outline's promise.
- **Fix**: Either (a) add graph construction statistics to Section 5.1 if the graphs were constructed, or (b) update the outline to match the actual section structure, or (c) reframe Section 5.1 as "Prior Empirical Findings" and add a new Section 5.2 for graph construction if that data exists.

### Issue 6: Missing Statistical Power Discussion for H6--H10
- **Location**: Section 5.4
- **Quote**: "The H6--H10 experiments address this limitation by changing the prediction target... a structural prediction with a clear chance baseline (precision@20 ≈ 0.004)."
- **Problem**: While Section 5.4 correctly discusses power limitations for H1--H3 (n=26, limited power), it does not discuss the power or sample size considerations for H6--H10. For H6, the number of absorption pairs is small (the table shows 4--6 HIGH features per layer, each with potentially multiple children). The Fisher exact test power depends on the number of true absorption pairs, which is not stated. For H7--H8, the same n=26 limitation applies.
- **Fix**: Add a paragraph discussing the sample size and power for H6--H10. For H6, report the total number of absorption pairs used as ground truth. For H7--H8, note that n=26 features limits correlation power just as it did for H1--H3.

### Issue 7: Table 4 Cohen's d Mismatch with Source Data
- **Location**: Table 4
- **Quote**: Table 4 reports Cohen's d = 1.26 (L4) and 1.18 (L8). The `paper_summary_stats.json` source reports d = 1.79 (L4) and 1.61 (L8).
- **Problem**: The Cohen's d values in Table 4 do not match the source data. Table 4 shows 1.26 and 1.18, but `paper_summary_stats.json` shows 1.79 and 1.61. This is a data inconsistency that undermines credibility.
- **Fix**: Verify the correct Cohen's d values and update Table 4 to match the source data. If 1.26/1.18 are correct, update the JSON. If 1.79/1.61 are correct, update the table.

## Minor Issues

- **Section 5.1, paragraph 3**: "Raw steering success rates at $s = 50$ ranged from 0.40 to 1.00 across features." → The source data does not explicitly confirm this range. Add a citation to the specific results file or verify the min/max values.

- **Section 5.1, Table 3 caption**: "The maximum absorption rate observed was 0.242 for feature U at layer 8." → The `paper_summary_stats.json` reports max = 0.2416 at L8. The rounding to 0.242 is fine, but the feature "U" is not verifiable from the JSON. Ensure this is correct.

- **Section 5.2, H6 protocol**: "expected precision@20 ≈ 0.004" → This is 20/24576 ≈ 0.00081, not 0.004. The value 0.004 would be correct for k=100 (100/24576 ≈ 0.004). Verify: 20/24576 = 0.000814. The text says precision@20 ≈ 0.004, which is incorrect by a factor of 5.

- **Section 5.2, H6 prediction**: "Precision@20 ≥ 0.10 (25× enrichment over chance)." → If chance is 0.000814, then 0.10 is 123× enrichment, not 25×. If chance is 0.004, then 0.10 is 25× enrichment. The chance baseline and enrichment factor are inconsistent.

- **Section 5.4**: "The 95% confidence interval for r = −0.301 (layer 8 H1) is approximately [−0.62, +0.10]" → Verify this CI calculation. For n=26, Fisher's z-transform gives CI(r) that should be checked.

- **Section 5.1, Table 1**: The H1b row uses bold for r = -0.431 and p = 0.028 but the H1 row at L8 uses non-bold for r = -0.301 and p = 0.136. The bolding convention is clear (significant results), but given the multiple comparison issue, consider adding a dagger or footnote symbol instead of bold to flag the uncorrected nature.

- **Missing figure references**: Section 5.1 references Table 3, Table 4, Table 5, and Table 1, but no figures. The outline promised Figure 2 (H6 results), Figure 3 (H7 scatter plots), Figure 4 (H9 layer bars), and Figure 5 (H10 rebalancing). None of these are referenced because the experiments were not executed.

- **Section 5.3, Table 2, row 7**: "No correlation with probing F1" → The inhibition explanation says "Probing measures decoder direction quality, not encoder activation; inhibition affects the latter." This is a strong claim. Probing at high k uses many latents, but at k=1 or k=5, the parent's contribution matters more. The explanation should acknowledge that the null result at high k may not generalize to very sparse probes.

## Visual Element Assessment

- [x] Figures/tables match outline plan (for H1--H5 content)
- [ ] Figures/tables for H6--H10 are missing (experiments not executed)
- [x] All visuals referenced before appearance (for existing tables)
- [x] Captions are self-explanatory (Table captions are clear)
- [ ] Text-heavy sections need visual support: Section 5.2 (H6--H10 protocols) is entirely text-based and would benefit from a flow diagram showing the validation pipeline

## What Works Well

1. **Table 2 (Integration table)** is excellent. It provides a clear, row-by-row mapping of each prior finding to its inhibition explanation with specific supporting evidence. This is the strongest element of the section and effectively sells the theoretical framework.

2. **The contrast between H1 and H1b** (paragraph beginning "The contrast between H1 and H1b is critical") is well-explained and provides genuine methodological insight. The explanation of why delta correction reveals what raw metrics mask is clear and actionable.

3. **Section 5.4 (Power Analysis)** is appropriately humble about limitations. The discussion of limited power, the 95% CI for r = -0.301, and the explicit framing of the study's contribution as methodological rather than conclusive is exactly the right tone for a top venue.

## Cross-Section Consistency Notes

- **Hypothesis numbering**: The hypotheses section (H1, H1b, H2, H3) does not align with the experiments section (H1--H5, H6--H10). This needs resolution.
- **Method section consistency**: The method section (4.4--4.8) describes H6--H10 protocols that match the experiments section descriptions. This is consistent, but both sections describe experiments that were not executed.
- **Intro consistency**: The intro's key results preview (Section 1.5) promises H6--H10 results with specific predicted values. These promises are not fulfilled in the experiments section.
- **Abstract consistency**: The abstract correctly focuses on H1--H3 results and does not overclaim H6--H10. This is good.
- **Table 4 vs. paper_summary_stats.json**: Cohen's d values differ (1.26/1.18 in table vs. 1.79/1.61 in JSON). This is a data inconsistency.
