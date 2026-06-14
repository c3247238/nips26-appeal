# Critique: Experiments

## Summary Assessment
The Experiments section is well-structured and clearly written, presenting strong evidence for the primary hypothesis (H1: CV predicts steering) and properly reporting negative results for H6. The section flows logically from pilot validation through full experiments to cross-architecture validation. However, the H4 "variance paradox" claim in Table 4 lacks supporting evidence in this section (no CV values reported for absorbed vs non-absorbed), and the cross-architecture results (Section 4.6) are incomplete/pending, which weakens the contribution narrative.

## Score: 7/10
**Justification**: Strong methodological rigor and clear reporting of primary results. Deducted points for: (1) H4 evidence missing from this section (only stated as confirmed in Table 4 without data), (2) Cross-architecture validation status is ambiguous ("requires detailed integration"), (3) Section 4.5 (Non-Absorbed Baseline) reports a surprisingly small difference (0.102 vs 0.097) that contradicts the "robust absorbed" narrative without adequate discussion.

## Critical Issues

### Issue 1: H4 "Variance Paradox" Evidence Missing from This Section
- **Location**: Section 4.7 (Hypothesis Status Summary), Table 4 row for H4
- **Quote**: "H4 | Variance paradox | CV_absorbed = 7.33 vs CV_non-absorbed = 0.01 (733x ratio) | **CONFIRMED**"
- **Problem**: The key evidence for H4 (the 733x CV ratio) is not presented anywhere in the Experiments section. This is the second confirmed hypothesis and is central to the "variance paradox" framing, yet no experimental results show CV measurements comparing absorbed vs non-absorbed features. The reader must infer this from the theoretical framework in the Method section or look elsewhere.
- **Fix**: Add a dedicated subsection or incorporate into Section 4.1/4.2 showing the CV distribution comparison between absorbed and non-absorbed features, with specific numbers (7.33 vs 0.01). This could be a brief paragraph with a reference to Figure 3 (which shows this distribution).

## Major Issues

### Issue 2: Non-Absorbed Baseline Contradicts "Robust Absorbed" Narrative
- **Location**: Section 4.5, lines 65-67
- **Quote**: "Non-absorbed features showed mean absolute steering effect of 0.102 (SD: 0.078), compared to 0.097 for absorbed high-CV features. The difference is 0.0045, which is not practically significant."
- **Problem**: The "robust absorbed" framing in the theory section suggests high-CV absorbed features should be MORE steerable than non-absorbed features (since they route through context-sensitive channels). However, the actual result shows virtually identical steering effects (0.097 vs 0.102). The text acknowledges this is "not practically significant" but does not reconcile it with the theoretical narrative that high-CV absorbed features are "better" or "more steerable" than non-absorbed features.
- **Fix**: Either (a) reframe the "robust absorbed" contribution as "absorbed features are NOT degraded relative to non-absorbed" rather than "absorbed features are MORE steerable," or (b) provide additional analysis showing where the advantage of robust absorbed features actually lies (e.g., in specificity, or in contexts where non-absorbed features fail).

### Issue 3: Cross-Architecture Validation Status is Ambiguous
- **Location**: Section 4.6, lines 71-75
- **Quote**: "The experiment was completed and marked by `full_cross_architecture_DONE`. Status: Cross-architecture results require detailed integration. Preliminary analysis suggests the CV-steering relationship may generalize but the threshold may need adjustment for different architectures."
- **Problem**: The section states the experiment is complete but results are "pending detailed integration." This leaves the reader uncertain whether the cross-architecture validation succeeded or failed. The hedge language ("may generalize," "may need adjustment") suggests the results are inconclusive or possibly negative.
- **Fix**: Either (a) report specific results with numbers if available, or (b) explicitly state "Cross-architecture validation is pending further analysis; results will be integrated in the final version." The current status is worst-case: it sounds like the experiment ran but the researchers don't know how to interpret or present the findings.

### Issue 4: SD Values Inconsistent Between Text and Table 1
- **Location**: Section 4.3, line 35 vs Table 1 (outline)
- **Quote (text)**: "At strength +3, high-CV features showed mean effect 0.3079 (SD: 0.15) versus low-CV 0.2103 (SD: 0.12)"
- **Quote (outline Table 1)**: "+3 | 0.3079 | 0.15 | 0.2103 | 0.12 | 9.96 | < 0.01 | Yes"
- **Problem**: The SD values in the text for +5 and +7 appear potentially incorrect. The text reports SD: 0.25 for high-CV at +5 and SD: 0.35 at +7, but these may be misreported. Additionally, the non-absorbed baseline SD (0.078) is reported but no comparison SD is given.
- **Fix**: Verify the correct SD values for all strength levels and ensure consistency. The outline Table 1 shows: +5: High-CV SD=0.25, Low-CV SD=0.20; +7: High-CV SD=0.35, Low-CV SD=0.28. Cross-check against actual experimental data.

## Minor Issues
- **Line 31**: "accounting for feature reuse across different CV groups" - This parenthetical statement is confusing. Clarify what "feature reuse" means in this context.
- **Line 41**: "The dose-response relationship shows consistent scaling" - The term "dose-response" is jargon from pharmacology; consider "steering strength-response" or simply "response scaling."
- **Table 1 caption**: Table 1 appears in the text before Table 2 and Table 3 are explicitly introduced. Verify table numbering is correct.
- **Line 75**: "The experiment was completed and marked by `full_cross_architecture_DONE`" - The code-style marker is inappropriate for a paper. Remove or rephrase.

## Visual Element Assessment
- [x] Figures/tables match outline plan (Figure 1, 2, 4 all present)
- [x] All visuals referenced before appearance (Figure 2 at line 13, Figure 1 at line 33, Figure 4 at line 55)
- [x] Figure 2 shows error bars (SD across measurement contexts, line 15)
- [ ] Figure 3 (CV Distribution) is referenced in outline but NOT in experiments section - the H4 variance paradox evidence would benefit from this figure being referenced here
- [ ] No Table 1 caption visible in text - Table 1 is embedded in the text but not clearly labeled as "Table 1"

## What Works Well

1. **Proper negative result reporting (Section 4.4)**: H6 is correctly labeled "NOT_SUPPORTED" with full statistical evidence (Welch's t = 1.77, p = 0.079; Pearson r = -0.136, Spearman rho = -0.204). This is honest scientific reporting.

2. **Clear t-statistics and BH correction (Section 4.3)**: "t-statistics: 9.96, 9.73, 9.49" with p < 0.01 and Benjamini-Hochberg correction explicitly mentioned. This level of statistical detail is exemplary.

3. **Activation patching methodology (Section 4.1)**: The recovery formula $R_{parent} = (x_{patched} - x_{zero}) / (x_{clean} - x_{zero})$ is correctly specified, and Figure 2 clearly shows all 9 words passing the 10% threshold. This establishes genuine causal structure convincingly.
