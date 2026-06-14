# Critique: Conclusion

## Summary Assessment
The conclusion effectively summarizes the main contributions and provides a practical takeaway list for interpretability practitioners. However, it has one critical gap (missing Figure 5 architecture diagram referenced but not delivered) and one major issue (imprecise language about non-absorbed baseline comparison that could mislead readers about the actual relationship).

## Score: 7/10
**Justification**: The conclusion is well-structured and logically follows from the experimental results. The contributions are clearly articulated and the limitations are appropriately scoped. However, the missing visual element (Figure 5) and imprecise "approximately equal" claim about absorbed vs non-absorbed features weaken the section. Reaching 8/10 would require resolving both issues.

## Critical Issues

### Issue 1: Missing Figure 5 Architecture Diagram
- **Location**: End of section, figure reference comment (lines 53-55)
- **Quote**: "<!-- FIGURES\n- Figure 5: fig5_mechanism_desc.md — Architecture diagram describing context-sensitive vs bypass routing mechanism\n- None\n-->"
- **Problem**: The conclusion discusses the mechanistic hypothesis (context-sensitive vs bypass routing) in Section 6.1 but references Figure 5 as a planned visual element that has not been created. The comment `fig5_mechanism_desc.md` suggests this should be a manually created diagram, but it is absent from the `writing/sections/figures/` directory. A reader who reaches the conclusion and wants to understand the mechanism will find no visual support.
- **Fix**: Either (a) create the architecture diagram as described, or (b) remove the figure reference and replace the text explanation with a more detailed written description of the routing mechanism.

## Major Issues

### Issue 2: Imprecise "Approximately Equal" Claim
- **Location**: Line 3 (opening sentence)
- **Quote**: "...and approximately equal to non-absorbed features (0.102)"
- **Problem**: High-CV mean effect is 0.5251 and non-absorbed mean effect is 0.102 (from experiments.md Section 4.5). These values differ by a factor of 5, not "approximately equal." The word "approximately" suggests near-equivalence, which misleads readers about the actual relationship.
- **Fix**: Replace with specific comparison: "and comparable to non-absorbed features at approximately 20% of their effect (0.525 vs 0.102)" or clarify the ratio explicitly.

### Issue 3: Cross-Architecture Validation Status Unclear
- **Location**: Section 6.2 (Limitations), first bullet
- **Quote**: "While we completed cross-architecture validation on Gemma-2-2B, detailed analysis of those results is pending."
- **Problem**: The conclusion mentions cross-architecture validation as completed but pending detailed analysis. However, experiments.md states "The experiment was completed and marked by `full_cross_architecture_DONE`." The conclusion should clarify what "completed" means: were results obtained but not analyzed, or is the experiment itself pending?
- **Fix**: Distinguish between (a) experiment completed with results available but pending analysis, and (b) experiment not yet executed. The current text conflates these.

### Issue 4: 733x CV Ratio Claim Unverified
- **Location**: Section 6.1, third contribution
- **Quote**: "The variance paradox (CV_absorbed = 7.33 vs CV_non-absorbed = 0.01, 733x ratio)"
- **Problem**: This specific ratio appears in the proposal (idea/proposal.md) but the experiments section does not report the actual CV measurements for non-absorbed features. The ratio is referenced as if confirmed, but no experimental data in exp/results/ verifies CV_non-absorbed = 0.01.
- **Fix**: Either (a) cite a specific experimental result file that contains this measurement, or (b) soften the claim: "The variance paradox (absorbed features showing dramatically higher CV than non-absorbed features, as observed in pilot analysis)"

## Minor Issues

- **Line 3**: "1.47x larger on average" is accurate per the data (effect_ratio: 1.47). This is correct.
- **Line 11**: "predominantly low-CV" is reasonable inference but could note that this is hypothesized, not confirmed.
- **Section 6.4 bullet 3**: "Features with CV > 1.0 are significantly more likely to produce measurable effects (1.47x larger on average)" — "significantly" could be misread as statistically significant. The 1.47x is the aggregate effect ratio; "measurably larger" would be less ambiguous.

## Visual Element Assessment
- [ ] Figure 5 (mechanism diagram) referenced but NOT created — CRITICAL gap
- [x] All other visuals appear to be in figures/ directory
- [x] No text-heavy sections that would benefit from additional visuals
- [ ] Figure 5 should illustrate the context-sensitive vs bypass routing mechanism described in the mechanistic hypothesis

## What Works Well

1. **Section 6.1 bullet 1** (lines 7-8): The t-statistic and p-value are correctly cited: "At strength +5, high-CV features achieve mean effect 0.522 versus 0.355 for low-CV (t = 9.73, p < 0.01)". This matches experimental data exactly.

2. **Section 6.3 Future Work** is well-scoped: The five directions (larger models, prospective threshold validation, causal mechanism via timecourse, cross-SAE architecture testing, behavioral downstream validation) are all genuine next steps that follow logically from the paper's limitations.

3. **Section 6.4 Takeaways** is a strong practical contribution: The numbered list provides actionable guidance for practitioners. Bullet 4 ("absorption metrics identify what is absorbed; CV predicts what is steerable") is particularly well-articulated as it connects the existing A_j metric to the paper's new contribution.

4. **Section 6.2 Limitations** appropriately scopes the work: The three limitations (GPT-2 Small only, logit change only, empirically derived threshold) are all valid and properly caveatted.