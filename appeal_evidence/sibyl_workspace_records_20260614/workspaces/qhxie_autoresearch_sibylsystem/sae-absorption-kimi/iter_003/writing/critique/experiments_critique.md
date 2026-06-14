# Critique: Results (Experiments)

## Summary Assessment

The Results section presents the empirical findings clearly and structures them around the three hypotheses declared in the Method section. The numbers are accurate against the source data, figures are referenced before appearance, and the statistical reporting is generally sound. However, the section suffers from a missing Figure 1 (architecture ranking comparison) that was planned in the outline, an underdeveloped GPT-2 replication subsection, and several instances where the prose could be more direct. The section earns a solid but not exceptional score.

## Score: 6/10

**Justification**: The section is competent---numbers are correct, structure follows the hypotheses, and claims are appropriately hedged. To reach 7-8, it needs: (1) the missing Figure 1 that was planned in the outline, (2) more substantive treatment of the GPT-2 replication, and (3) sharper prose that leads with findings rather than setup. To reach 9-10, it would need explicit effect-size reporting beyond p-values and a more nuanced discussion of what "inconclusive" means for practice.

---

## Critical Issues

*None identified. The central claims are supported by the data, numbers are accurate, and the section does not overclaim beyond what the evidence supports.*

---

## Major Issues

### Issue 1: Missing Figure 1 (Architecture Ranking Comparison)

- **Location**: Section 3.1, first paragraph
- **Quote**: "Table 1 reports per-architecture absorption scores across all three conditions..."
- **Problem**: The outline's Figure & Table Plan (line 201-208) explicitly calls for "Figure 1: Architecture Ranking Comparison"---a grouped bar chart showing side-by-side first-letter vs. semantic-hierarchy absorption scores across all architectures. This figure is referenced in the outline as "already generated as `fig1_architecture_ranking.png`" but does not appear in the section. The section jumps directly to Table 1 without the visual overview that the outline promised.
- **Fix**: Insert Figure 1 before Table 1, or immediately after the first paragraph of 3.1. The figure should show grouped bars (first-letter blue, semantic-hierarchy orange) per architecture. Caption: "Comparison of first-letter and semantic-hierarchy absorption scores across 8 SAE architectures on Pythia-160M layer 8."

### Issue 2: GPT-2 Replication Is Underdeveloped

- **Location**: Section 3.6
- **Quote**: "Absolute scores are near-zero compared to Pythia-160M, and the pattern differs---on GPT-2, hierarchy absorption is lower than non-hierarchy absorption, which is directionally consistent with hierarchy specificity, but the magnitudes are too small to support confident interpretation."
- **Problem**: This subsection is only 85 words and treats the replication as an afterthought. The outline (lines 115-118) dedicates a full subsection to it, and the finding is genuinely important: it shows model-dependent behavior that undermines generalizability claims. Yet the section offers no figure reference (Figure 5 was generated per `statistical_analysis_summary.json`), no statistical comparison, and no integration with the main Pythia-160M findings. The reader is left wondering whether the GPT-2 results contradict or complement the main story.
- **Fix**: Expand to at least 150 words. Reference Figure 5 explicitly. Add a sentence comparing the magnitude difference: on Pythia-160M, the hierarchy vs. non-hierarchy gap is -0.096 (non-hierarchy higher), while on GPT-2 it is +0.060 (Standard) and +0.094 (TopK)---hierarchy lower, as theory predicts. Discuss whether this model-dependent reversal strengthens or weakens the central claim about metric degeneracy.

### Issue 3: Inconsistent Handling of Random SAE in H2

- **Location**: Section 3.3, second paragraph
- **Quote**: "Every architecture except TopK shows this reversal."
- **Problem**: The Method section (2.6) states: "The Random-SAE control is excluded from H1 and H3 because its first-letter score is an outlier by design, but it is included in H2 because the hierarchy-specificity test applies to all configurations." Yet the Results section says "Every architecture except TopK" without clarifying whether this includes Random SAE. Checking the data: Random SAE has semantic=0.352 and non-hierarchy=0.416, so it also shows the reversal (non-hierarchy > hierarchy). The claim "except TopK" is technically correct if Random is included (TopK is the only one where semantic > non-hierarchy: 0.250 > 0.311 is false---actually 0.250 < 0.311, so TopK also shows the reversal). Wait: TopK semantic=0.250, non-hierarchy=0.311, so TopK also shows non-hierarchy > hierarchy. The claim "Every architecture except TopK" is therefore incorrect.
- **Fix**: Correct to "All eight architectures show this reversal" or explain why TopK is singled out. Verify: TopK semantic=0.250, non-hierarchy=0.311. 0.250 < 0.311, so TopK also shows higher non-hierarchy. The text is factually wrong.

### Issue 4: Missing Effect Sizes for H2

- **Location**: Section 3.3
- **Quote**: "A paired t-test yields t = -4.748, p = 0.0032."
- **Problem**: The section reports t-statistic and p-value but omits Cohen's d or any standardized effect size. For a paired t-test with n=8, t=-4.748 corresponds to Cohen's d = -1.68 (large effect). Reporting only p-values is outdated practice; effect sizes are essential for interpreting practical significance, especially in a field moving toward better statistical hygiene.
- **Fix**: Add Cohen's d = -1.68 (95% CI: [-2.87, -0.49]) or simply report as "large effect" alongside the t-test results.

---

## Minor Issues

- **Section 3.1, line 5**: "The Random-SAE control scores 0.030 on first-letter absorption---near zero, as expected---but 0.352 on semantic-hierarchy absorption, identical to the Standard SAE." The phrase "as expected" is slightly presumptuous without prior justification. The Method section does not explain why Random SAE should score near-zero on first-letter. Fix: remove "as expected" or add a forward reference to the Method's description of the Random-SAE construction.

- **Section 3.2, line 24**: "The point estimate suggests a moderate positive relationship, but the interval spans from moderately negative to near-perfect correlation." The phrase "moderately negative" is imprecise---the lower bound is -0.389, which is weak-to-moderate negative. Fix: "weakly-to-moderately negative" for precision.

- **Section 3.4, line 44**: "The correlation is numerically stable (r = 0.468, 0.463, 0.471)"---the values are rounded inconsistently with Table 2, which shows 0.468, 0.463, 0.471. Actually this is consistent. But the text says "0.468--0.471" in the caption while the body says "0.468, 0.463, 0.471". The dash range in the caption is fine but the body should use the same format or explain the range.

- **Section 3.5, line 62**: "This is the most striking finding in our study"---this evaluative claim appears in both Results (3.5) and Discussion (4.3). While not a contradiction, it is repetitive. The Results section should present the finding neutrally and let the Discussion interpret its significance.

- **Table 1 caption**: "Lowest score per column is bold"---but the Random row shows 0.030 for first-letter, which is lower than GatedSAE's 0.008. Wait: 0.008 < 0.030, so GatedSAE is correctly bold. But Random's 0.030 is not bold, which is correct since 0.008 is lower. However, the semantic-hierarchy column has PAnneal at 0.064 (bold) and Random at 0.352 (not bold)---correct. Non-hierarchy has PAnneal at 0.131 (bold) and Random at 0.416---correct. The bolding is accurate.

- **Section 3.1, paragraph 2**: "PAnneal achieves the lowest scores in both semantic conditions (0.064 and 0.131)"---this is accurate but the phrasing "semantic conditions" is slightly ambiguous (could mean "both semantic-related conditions" vs. "the semantic condition and the non-hierarchy condition"). Fix: "in both the semantic-hierarchy and non-hierarchy conditions" for clarity.

- **Missing transition at section end**: Section 3.6 ends abruptly without a transition sentence to the Discussion. The outline specifies: "These results demand careful interpretation of what they mean for the field." Add this transition or similar.

---

## Visual Element Assessment

- [ ] Figures/tables match outline plan
  - **FAIL**: Figure 1 (architecture ranking comparison) is missing from the section despite being planned in the outline and listed as already generated.
  - **FAIL**: Figure 5 (GPT-2 replication) is not referenced in the section despite being generated.
- [x] All visuals referenced before appearance
  - Table 1 and Figure 2, 3, 4 are all referenced in text before they appear.
- [x] Captions are self-explanatory
  - All captions include the key statistical result and interpretation.
- [ ] No text-heavy sections that need visual support
  - Section 3.6 (GPT-2) is text-only and would benefit from Figure 5.

---

## What Works Well

1. **Hypothesis-driven structure**: The section organizes results around H1, H2, H3 rather than presenting a data dump. This makes the narrative easy to follow and directly connects to the Method section's statistical analysis plan (Section 2.6).

2. **Appropriate hedging**: The section correctly labels H1 as "inconclusive" rather than claiming non-significance or significance. The bootstrap CI is reported with precision, and the limitations (small n, diffuse distribution) are acknowledged inline rather than deferred to Discussion only.

3. **Random-SAE integration**: The control is woven throughout the section (3.1, 3.2, 3.3, 3.5) rather than treated as an appendix. This strengthens the central argument about metric degeneracy by showing the same control result from multiple angles.
