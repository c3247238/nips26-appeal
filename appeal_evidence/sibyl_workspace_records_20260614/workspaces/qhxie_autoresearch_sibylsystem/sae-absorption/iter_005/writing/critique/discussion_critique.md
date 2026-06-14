# Critique: Discussion

## Summary Assessment
The Discussion section does serious interpretive work, connecting the statistical results from Section 4 to broader implications for the SAE field. The causal inference discussion (5.1) is the strongest subsection, carefully walking through suppression effects, mediation, and Rosenbaum bounds with appropriate citations. The metric failure analysis (5.2) and scaling surface implications (5.3) provide genuinely useful guidance for practitioners. The Limitations (5.4) section is commendably honest, though several issues identified below weaken the overall execution.

## Score: 7/10
**Justification**: Strong conceptual structure and honest limitation reporting earn credit, but inconsistencies with the Experiments section (particularly the taxonomy correction narrative), missing cross-references to specific figures/tables in the Results, a few unsupported claims, and some redundancy with Section 4 prevent it from reaching 8. Reaching 8 requires fixing the taxonomy contradiction, tightening the evidence linkage, and eliminating the redundant re-reporting of statistical results already in Section 4.

## Critical Issues

### Issue 1: Taxonomy correction narrative contradicts the Experiments section
- **Location**: Section 5.4, paragraph beginning "Taxonomy correction reveals diagnostic complexity"
- **Quote**: "The corrected comprehensive rate dropped from 92.3% to 19.2%, while the Chanin false-negative-based absorption rate independently validates absorption in 73.1% of letters."
- **Problem**: This directly contradicts the Experiments section (4.4) and the Conclusion (Section 6). Section 4.4 states: "Zero letters changed classification. Corrected combined rate: 92.3% (identical to original)." The Conclusion states: "The taxonomy correction confirms the original 92.3% combined absorption rate on GPT-2 Small (0 of 26 letters changed classification under frequency-matched comparison tokens)." However, the *actual* Experiments text (Section 4.4 body) says: "After frequency-matched correction using Strategy A (non-letter context comparison), 19 letters change classification: the corrected comprehensive rate drops to 19.2%." The outline summary at lines 109-113 says "Zero letters changed" while the full experiments text says "19 letters changed." Either the Discussion is correct and the outline summary is wrong, or vice versa. This is the most critical inconsistency in the paper because the Conclusion makes the opposite claim from the Discussion about the same result.
- **Fix**: Reconcile across all three sections (4.4, 5.4, 6). Read the actual experiment output data to determine the ground truth. If the corrected rate is 19.2% (as the Discussion and the body of Section 4.4 both state), update the Conclusion and the outline summary. If the rate remains 92.3%, update the Discussion paragraph. This must be a single, consistent narrative.

### Issue 2: Inconsistent mediation description for SCR between Discussion and Experiments
- **Location**: Section 5.1, paragraph 2
- **Quote**: "the total effect of L0 on SCR is significant ($c = 0.022$, $p = 0.001$)"
- **Problem**: The Experiments section Table 3 reports the direct effect $c' = -0.029$ ($p = 0.71$) for SCR. The total effect ($c$) is not explicitly reported in Table 3 but is implied by the mediation structure. The Discussion introduces the value $c = 0.022$ ($p = 0.001$) which does not appear anywhere in the Experiments section. The reader cannot verify this number from the Results. Additionally, the sign difference ($c = 0.022$ positive in Discussion vs. the negative correlations reported throughout) is confusing without explanation -- it likely reflects that the total effect is in unstandardized units with a different sign convention, but this is never clarified.
- **Fix**: Either (a) add the total effect $c$ to Table 3 in the Experiments section so the Discussion can reference it, or (b) cite the table explicitly and use consistent sign conventions. Clarify whether $c = 0.022$ is standardized or unstandardized and why its sign differs from the negative correlations reported elsewhere.

## Major Issues

### Issue 3: Excessive re-reporting of statistical results already in Section 4
- **Location**: Throughout Sections 5.1, 5.2, and 5.3
- **Problem**: The Discussion re-reports nearly every statistical result from the Experiments section with full numeric precision. Section 5.1 alone contains 15+ specific p-values, correlation coefficients, and confidence intervals that are already in Section 4.1. Section 5.2 re-states the "51--85%" and "100%" and "0%" rates verbatim. A Discussion should *interpret* results, not re-present them. This redundancy consumes space (the Discussion is ~74 lines, of which roughly 30 are numeric re-reporting) and blurs the line between Results and Discussion.
- **Fix**: Replace most numeric re-reporting with references to specific tables and figures: "The suppression effect (Table 1; Figure 1) is the central finding..." rather than re-quoting $r = -0.664$ to $r = -0.746$ ($p = 1.2 \times 10^{-9}$). Reserve full numeric citation for the 2-3 most critical results that anchor interpretive arguments.

### Issue 4: The Bradford Hill table (Table 6) needs clearer connection to Section 4 evidence
- **Location**: Section 5.1, Table 6
- **Quote**: "Table 6 presents a systematic assessment of the absorption-quality causal claim against the nine Bradford Hill criteria"
- **Problem**: The table numbering is inconsistent with the Experiments section. The Experiments section uses Tables 1-6, with Table 6 being the "Scaling Surface Model Comparison." The Discussion's Table 6 (Bradford Hill criteria) would then be a duplicate number. More substantively, several cells cite evidence that is not in Section 4. The "Temporality" row states "L0 is a training hyperparameter set before training; absorption emerges during training; quality is measured post-hoc" -- this temporal ordering argument is asserted but not empirically demonstrated (all measurements are cross-sectional, as the table itself notes). The "Analogy" row cites "immunodominance" without any prior mention in the paper.
- **Fix**: (1) Renumber the table to avoid collision with Experiments Table 6 (use Table 7 or move Experiments tables to a different numbering scheme). (2) For the Temporality row, strengthen the caveat or downgrade the assessment from "Plausible" to explicitly state that no temporal data exists. (3) For the Analogy row, either add a brief parenthetical explaining immunodominance or remove it if not developed elsewhere in the paper.

### Issue 5: The three-regime picture in 5.3 introduces the "hedging regime" without evidence
- **Location**: Section 5.3, the three numbered regimes
- **Quote**: "Hedging regime (low width, any L0): The dictionary lacks capacity to represent fine-grained features; the SAE merges correlated concepts into polysemantic 'hedge' latents."
- **Problem**: This paper does not measure hedging. The hedging regime is attributed to Chanin & Garriga-Alonso (2025) but no data from the current study supports or characterizes it. The Discussion presents the three-regime picture as if it follows from the paper's results, but only the absorption regime is empirically grounded here. The hedging boundary is explicitly acknowledged as "not characterized in this study," but this caveat comes 5 lines after the three-regime list, and the list reads as a finding rather than speculation.
- **Fix**: Reframe the three-regime picture explicitly as a synthesis of this paper's absorption results with prior work on hedging. Use language like: "Combining the absorption scaling surface (this work) with the hedging results of Chanin & Garriga-Alonso (2025) *suggests* a three-regime picture..." and move the hedging caveat into the list item itself.

### Issue 6: Missing Figure/Table references for cross-domain discussion
- **Location**: Section 5.2
- **Problem**: Section 5.2 discusses the cross-domain results extensively but never references Figure 3, Figure 4, or Table 5 from the Experiments section. The reader must mentally map the Discussion's claims back to the Results without guidance. Section 5.1 similarly fails to reference Figure 8 (Rosenbaum sensitivity, per the outline).
- **Fix**: Add explicit cross-references: "The shuffled control (Figure 3)" in the first paragraph of 5.2; "The layer-wise pattern (Figure 4; Table 5)" in the fourth paragraph; "The Rosenbaum bounds (Figure 8; Table 4)" in 5.1.

## Minor Issues

- **Section 5.1, paragraph 3**: "By Rosenbaum's classification, $\Gamma > 2.0$ indicates strong robustness to hidden bias (Rosenbaum, 2002)." This threshold characterization is not standard in the Rosenbaum (2002) text -- Rosenbaum does not define categorical thresholds. The glossary correctly defines Gamma without categorical labels. Either cite a secondary source that proposes these thresholds or soften to "is considered relatively robust."
- **Section 5.2, paragraph 1**: "51--85% absorption" -- the Experiments section reports "11.3--96.2%." The range cited in the Discussion excludes the extremes without explanation.
- **Section 5.2, paragraph 3**: "This is not simply a measurement failure; it reflects a genuine property of the model" -- this reads as an unsupported assertion. The evidence shows the *metric* fails, not that the model genuinely lacks knowledge-hierarchy absorption. The next sentence about probe accuracy (86--93%) actually undermines the claim by showing the model *does* encode this information.
- **Section 5.3, paragraph 2**: "each doubling of width adds approximately 5.4 percentage points of absorption at fixed L0" -- the coefficient reported is +0.054 in log space. Doubling width = $\Delta \log(\text{width}) = \log(2) \approx 0.693$, giving $0.054 \times 0.693 \approx 0.037$ (3.7 pp), not 5.4 pp. Check whether the coefficient is on $\log_2$ or $\ln$ scale.
- **Section 5.4, paragraph 5**: "proportion mediated exceeds 1.0 for SCR" is mentioned in 5.1 but the sentence explaining inconsistent mediation is dense and could benefit from a clearer lead-in.
- **Section 5.1, paragraph 1**: Uses $\rho = -0.47$ (Spearman) for the L0-absorption correlation but Section 4 does not report this exact number. Verify or add to Results.

## Visual Element Assessment
- [ ] Figures/tables match outline plan -- **Partial**: Table 6 (Bradford Hill) is present as planned. But the Discussion does not reference Figures 3, 4, or 8 from the Results, which outline says belong to Sections 4.1 and 4.2 and are relevant to the Discussion's arguments.
- [x] All visuals referenced before appearance -- Table 6 is referenced before it appears.
- [x] Captions are self-explanatory -- Table 6's column headers are clear.
- [ ] No text-heavy sections that need visual support -- Section 5.3's three-regime picture would benefit from a conceptual diagram (this is not planned in the outline but would strengthen the argument considerably).

## What Works Well

1. **Section 5.1's suppression effect interpretation** (paragraph 1) is genuinely excellent. The explanation of how L0 acts as a suppression variable, with the Conger (1974) citation and the step-by-step logic of how controlling for irrelevant variance reveals the true association, is the kind of precise causal reasoning that distinguishes this paper from correlational SAE studies.

2. **Section 5.2's root cause analysis** of the metric failure is incisive. Identifying Feature 8213 as a polysemantic "super-absorber," explaining why it dominates at both real and shuffled false-negative positions, and connecting this to the 98% dead feature rate provides a mechanistic explanation rather than simply reporting the failure. The recommendation for cosine-calibrated metrics is actionable and well-motivated.

3. **The Limitations section (5.4)** is unusually thorough for a Discussion. The acknowledgment that within-width evidence is insufficient, that GPT-2 Small may not be representative, that observational design cannot rule out reverse causation, and that the suppression effect interpretation depends on the assumed causal structure demonstrates the kind of intellectual honesty that reviewers respect. The specific suggestion of OrtSAE and Matryoshka as interventional tools is concrete.
