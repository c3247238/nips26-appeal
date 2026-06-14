# Critique: Experiments (Section 4)

## Summary Assessment

The experiments section is the strongest part of the paper: it presents four phases of analysis with pre-registered hypotheses, clear quantitative results, and appropriate statistical methods. The Phase 1 confound resolution (Section 4.1) is thorough and well-structured, with the suppression effect providing a genuinely surprising and publication-worthy finding. Phase 3 (scaling surface) is clean and the interaction GAM result is strong. However, the section suffers from a critical cross-section data inconsistency in Phase 4 (taxonomy correction), where the numbers reported here contradict those in the conclusion and outline. Phase 2 (cross-domain) is well-handled as a negative/diagnostic result but would benefit from a clearer summary of what was and was not learned. The section also lacks Figure 4 (referenced in the outline as "layer-wise breakdown") and Figure 6 (gradient surface), and some mediation terminology deviates from the underlying data.

## Score: 7/10
**Justification**: Strong quantitative rigor in Phases 1 and 3; appropriate framing of the Phase 2 negative result; good use of controls. To reach 8: fix the critical Phase 4 data inconsistency, add the missing figures, tighten the mediation terminology to exactly match the data, and add a brief cross-phase synthesis paragraph. To reach 9: resolve the `n_full_mediations: 0` vs. "full mediation" discrepancy in the data pipeline and add within-width effect size estimates (even if null) to strengthen the discussion of that limitation.

## Critical Issues

### Issue 1: Phase 4 taxonomy correction numbers contradict conclusion and outline
- **Location**: Section 4.4, paragraphs 2-3; versus conclusion Section 6 paragraph 4; versus outline Section 4.4
- **Quote (experiments)**: "19 letters change classification: the corrected comprehensive rate drops to 19.2% (95% CI [3.8%, 34.6%])."
- **Quote (conclusion)**: "0 of 26 letters changed classification... confirming the original 92.3% combined absorption rate"
- **Quote (outline)**: "Zero letters changed classification. Corrected combined rate: 92.3% (identical to original)."
- **Problem**: The experiments section reports the numbers from the detailed `P4_taxonomy_correction.json` (corrected rate = 19.2%, 19 letters changed, corrected Type II count = 4). The conclusion and outline report numbers from `final_results.json` (corrected rate = 92.3%, delta = 0.0). The detailed JSON is clearly the authoritative source -- it has per-letter results, comparison strategies, and diagnostic data. The integration file appears to have incorrectly propagated the original rate as the corrected rate. This means the experiments section is correct and the conclusion/outline are wrong, but the contradiction across sections will be immediately flagged by any reviewer.
- **Fix**: (1) Confirm that the detailed P4 JSON (corrected rate 19.2%, 19 letters changed) is the ground truth. (2) Update the conclusion and outline to match. (3) Update `final_results.json` H5 entry to reflect the actual corrected values. The narrative pivot is significant: the taxonomy correction is no longer "validates original rate" but "reveals the magnitude-ratio metric was inflated by feature specificity, while the Chanin false-negative metric validates 73.1% absorption."

### Issue 2: Sparse probing mediation labeled "Indirect only" but data shows Baron-Kenny type "none"
- **Location**: Table 3, row 1 (SP-F1)
- **Quote**: "SP-F1 | Indirect only | -0.469 | -0.727 | -0.270 (0.001) | 0.015 | [0.007, 0.028] | 4.08 (4.4e-5)"
- **Problem**: The underlying `P1_mediation.json` records `mediation_type: "none"` for sparse probing because Baron-Kenny step 3 fails (total effect c is non-significant, p=0.45). The section labels this as "Indirect only" mediation, which is a valid alternative classification per Zhao et al. (2010), but this terminology is never defined in the paper. A reader familiar with Baron-Kenny will see "indirect only" as a non-standard label and wonder whether the authors are aware that the total effect is non-significant. The paragraph below Table 3 does explain this ("the total effect is non-significant, p = 0.45"), but the table label is misleading without the Zhao et al. citation.
- **Fix**: Either (a) relabel the table cell as "None (Baron-Kenny)" and explain the indirect-effect significance in the text, or (b) cite Zhao et al. (2010) to justify the "Indirect only" terminology and add a footnote defining it. Option (b) is preferable as it better communicates the genuine finding.

## Major Issues

### Issue 3: Missing Figure 4 (cross-domain by layer) and Figure 6 (gradient surface)
- **Location**: Entire Section 4.2 and Section 4.3
- **Problem**: The outline plan specifies Figure 4 ("Cross-Domain Absorption Rates by Layer") and Figure 6 ("Gradient Magnitude Surface with Phase Boundary"). Figure 4 is listed in the outline as generated (`fig7_crossdomain_by_layer.png`). Figure 6 is listed as generated (`P3_gradient_surface.png`). Neither appears in the experiments section. Figure 4 would strengthen Section 4.2 by visually demonstrating the layer 11 decrease pattern described in the text. Figure 6 would strengthen Section 4.3 by visualizing the phase boundary ridge that is currently described only in text.
- **Fix**: Insert Figure 4 in Section 4.2 after Table 5 (layer-wise pattern) and Figure 6 in Section 4.3 after the phase boundary paragraph.

### Issue 4: Section 4.3 reports "phase_boundary_detected: false" in data but claims phase boundary in text
- **Location**: Section 4.3, "Phase Boundary Detection" paragraph
- **Quote**: "Gradient analysis of the GAM surface identifies a transition zone at log2(L0) in [2.7, 3.8]"
- **Problem**: The `final_results.json` records `phase_boundary_detected: false` for H3. Yet the text confidently describes a transition zone with specific log2(L0) bounds and 443 ridge points. The section should acknowledge what the automated detection found (no formal phase boundary by whatever criterion was pre-registered) and frame the gradient analysis as characterizing a transition zone rather than detecting a sharp phase boundary. As written, the text implies the transition zone was formally detected, while the data suggests otherwise.
- **Fix**: Add a sentence noting that the gradient ridge does not constitute a sharp phase boundary in the mathematical sense (the gradient is continuous, not discontinuous) but marks a region of maximum sensitivity. Alternatively, clarify what `phase_boundary_detected` means in the pipeline and whether the log2(L0) range was identified despite the flag being false.

### Issue 5: No cross-phase synthesis before moving to Discussion
- **Location**: End of Section 4 (after Section 4.4)
- **Problem**: The section ends abruptly after the taxonomy correction results. A top-venue experiments section typically concludes with 2-3 sentences linking the four phases: Phase 1 establishes the causal link, Phase 2 reveals metric limitations, Phase 3 maps the parameter space, Phase 4 validates/corrects the baseline measurement. Without this synthesis, the reader must wait until Section 5 to understand how the pieces fit together.
- **Fix**: Add a 3-sentence closing paragraph after Section 4.4 that connects the phases and previews the discussion. For example: "Phase 1 establishes that absorption independently predicts quality degradation, Phase 2 reveals that the standard metric does not generalize to knowledge hierarchies, Phase 3 maps the width-L0 regime where absorption concentrates, and Phase 4 corrects the original taxonomy baseline. Section 5 interprets these findings."

### Issue 6: Table 3 direct effect c' for SP-F1 does not match text explanation
- **Location**: Table 3 and paragraph below it
- **Quote (table)**: "Direct c' (p): -0.270 (0.001)"
- **Quote (text)**: "the total effect is non-significant (p = 0.45), yielding an unstable proportion-mediated ratio"
- **Problem**: The table reports c' = -0.270 (p = 0.001), showing the direct effect IS significant. The text correctly notes the total effect c is non-significant (p = 0.45). However, the interplay between a significant direct effect and a non-significant total effect is confusing. When c is non-significant but c' is significant and ab is significant, this indicates suppression rather than mediation. The text partially addresses this but does not make it crisp. A reader may wonder how the direct effect can be significant at p=0.001 but the total effect at p=0.45 -- this requires explicit explanation of sign reversal (the direct and indirect effects operate in opposite directions).
- **Fix**: Add a sentence explaining: "The direct effect c' = -0.270 (p = 0.001) is negative and significant, while the indirect effect ab = 0.015 is positive, reflecting opposite-sign pathways: L0 directly degrades quality through a non-absorption mechanism, but simultaneously reduces absorption, which improves quality. These opposing paths cancel in the total effect."

## Minor Issues

- **Section 4.1, paragraph 1**: "Six canonical SAEs lacking reported L0 are excluded" -- the word "canonical" is ambiguous. Does this mean they are canonical in some other sense (official Gemma Scope releases)? Clarify: "Six SAEs from the original 54 that lack reported L0 values are excluded."
- **Section 4.1, Table 1**: The column "Bivariate r" is reported for all metrics but never defined. Add "(Pearson r, no covariates)" to the header or a table note.
- **Section 4.1, suppression paragraph**: "In classical statistical terms, L0 shares variance with absorption that is irrelevant to sparse probing quality." This is a good explanation but the word "irrelevant" is imprecise. L0 shares variance with absorption that does not predict quality, but it may be relevant in other senses. Use "that does not predict" instead.
- **Section 4.1, width-stratified**: The pooled analysis results (SP-F1 rho = -0.455, SCR rho = -0.376, TPP rho = -0.524) are reported without specifying which method was used. Are these pooled Spearman across all strata, or ignoring strata? Clarify.
- **Section 4.2, probe quality paragraph**: "84.2--85.0%" for Language binary -- the range across layers is very narrow. Report it as a single number (mean +/- SD) for cleaner presentation.
- **Section 4.2, Table 5**: The column header "$R_{\text{abs}}$ (dom >= 1.0)" is identical to the "FN Rate" column for all rows. If these values are definitionally identical under the chosen threshold, explain this or merge the columns.
- **Section 4.3, paragraph 2**: "the standard threshold of 5" for VIF -- this threshold appears in Section 4.1, not 4.3. It seems misplaced if it refers to the Phase 1 collinearity check.
- **Section 4.3, Table 6**: AIC for the additive GAM (-844) is higher (worse) than AIC for the interaction GAM (-917), but the additive model has lower R^2. The text focuses on R^2 and interaction p-value. Add a sentence noting that AIC also favors the interaction model.
- **Section 4.4**: "Letters F, G, I, and V retain their original Type II classification because they had pre-existing comparison tokens (1 each)." This contradicts the n_comparison_tokens_original = 0 premise. If these letters had comparison tokens, they were not affected by the fallback baseline. The sentence needs rephrasing to clarify that these 4 letters were already correctly classified and thus their Type II status is genuine.
- **Figure reference**: Section 4.3 references "Figure 5" but Section 4.1 references "Figure 1" and "Figure 2" -- no reference to Figures 3 or 4 appears in Sections 4.2 or 4.3 inline text, only in the header reference for Figure 3. Ensure all figures are referenced in running text before their appearance.
- **Bradford Hill paragraph** at end of Section 4.1: This assessment seems to belong in the Discussion (Section 5.1 also presents it). Having it in both places is redundant. Consider keeping only a brief mention in Section 4.1 with a forward reference to the full treatment in Section 5.1.

## Visual Element Assessment
- [x] Figures/tables partially match outline plan (Figures 1, 2, 3, 5 present; Figures 4, 6, 8 missing)
- [ ] All visuals referenced before appearance -- Figure 3 is referenced in the section header but not in running text before its appearance
- [x] Captions are present (as image alt text)
- [ ] No text-heavy sections that need visual support -- the Phase Boundary Detection paragraph in Section 4.3 describes gradient ridges and transition zones purely in text; Figure 6 (gradient surface) would significantly help

## What Works Well

- **The suppression effect presentation (Section 4.1)** is the highlight. Table 1 clearly shows the strengthening from r = -0.664 to r = -0.746, and the explanation in the text is precise and accessible. This will catch reviewers' attention immediately as a non-obvious, counterintuitive result.
- **The shuffled control design (Section 4.2)** turns what could have been a disappointing null result into the section's most compelling methodological contribution. The 100% shuffled rate is presented matter-of-factly, and the interpretation -- that the metric measures feature concentration, not probe-direction absorption -- is sharp and actionable.
- **Table 4 (Rosenbaum sensitivity)** presents five matching strategies side by side, allowing readers to see exactly which strategies detect effects and which do not. The Gamma = 2.65 for TPP under Mahalanobis matching is the strongest robustness evidence and is appropriately highlighted.
