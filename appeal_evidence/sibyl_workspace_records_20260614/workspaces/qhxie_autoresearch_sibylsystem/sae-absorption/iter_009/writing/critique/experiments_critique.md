# Critique: Experiments (Section 4)

## Summary Assessment
Section 4 presents a well-structured cross-domain absorption characterization with strong statistical backing (Kruskal-Wallis $p$ = 7.4e-66) and transparent reporting of probe quality limitations. The per-class analysis (Section 4.3) is a highlight, revealing that within-hierarchy variance (23x) exceeds between-hierarchy variance (4x). However, several numerical inconsistencies between this section and other parts of the paper, a missing figure from the outline plan, and a few instances of unsupported interpretive claims weaken an otherwise solid section.

## Score: 7/10
**Justification**: Strong data presentation and honest caveats earn credit, but numerical inconsistencies across sections (the first-letter L24 absorption rate is 27.1% here but 42.9% in the discussion and conclusion), a missing cross-section figure panel from the outline (Figure 2 was planned as a multi-hierarchy line plot but RAVEL hierarchies only appear as data points at L24), and several paragraphs that interpret without grounding in evidence pull the score down. Fixing the number inconsistencies and tightening the interpretive prose would bring this to an 8.

## Critical Issues

### Issue 1: First-letter absorption rate at L24 is inconsistent across sections
- **Location**: Section 4.1 (line 7) and Section 4.2 (line 19)
- **Quote**: "first-letter (27.1%, CI [24.5, 29.5])" (Section 4.1); "then jumps to 27.1% at layer 24" (Section 4.2)
- **Problem**: The discussion section (7.5) reports "absorption rises from 2.4% at L6 to 42.9% at L24 -- an 18x increase." The conclusion (contribution 1) also states "rising 18x from layer 6 (2.4%) to layer 24 (42.9%) for first-letter." The introduction reports first-letter at 27.1% consistent with Section 4. The consolidation_summary.json shows iter_008 at 34.5% and iter_009 at 27.1%. The 42.9% figure appearing in Discussion/Conclusion contradicts Section 4's 27.1%. Similarly, the L6 rate is reported as "1.0%" in Section 4.2 but as "2.4%" in Discussion and Conclusion. One of these must be wrong, and the discrepancy is critical because it affects the quoted multiplication factor (27x in Section 4 vs. 18x in Discussion/Conclusion).
- **Fix**: Verify against the source data (`phase1/absorption_firstletter.json`). Reconcile all instances across Sections 4, 7, and 8 to a single consistent set of numbers. If 27.1% is correct (per consolidation_summary), update Discussion/Conclusion. If 42.9% reflects a different measurement (e.g., iter_008 with different probes), state this explicitly.

### Issue 2: First-letter probe F1 reported as 1.0 in experiments but 0.97 in consolidation_summary and outline
- **Location**: Section 4.1 (line 15), Section 4.2 (line 23)
- **Quote**: "First-letter probes achieve $F_1$ = 1.0 at all layers" (4.1); "with $F_1 = 1.0$" (4.2)
- **Problem**: The consolidation_summary.json reports first-letter F1 = 0.97 at L24 (strict gate pass) and the outline states F1 = 0.97. The method section (3.2) claims "binary probes...achieve $F_1 = 1.0$ (weighted) at all four layers." This inconsistency undermines the "gold-standard anchor" claim. An F1 of 0.97 still passes the strict gate but is not 1.0, and the distinction matters for the claim that first-letter absorption rates are "uncontaminated by probe error."
- **Fix**: Choose one number and use it consistently. If the probes genuinely achieve 1.0 for the binary (letter-specific) probes but 0.97 for the weighted multi-class metric, state this distinction explicitly. The consolidation_summary at 0.97 suggests the 1.0 claim needs qualification.

## Major Issues

### Issue 3: Outline promised Figure 2 as a multi-hierarchy line plot across layers, but RAVEL hierarchies are only shown at L24
- **Location**: Section 4.2, Figure 2 description (line 21)
- **Quote**: "RAVEL hierarchies are measured only at L24 because RAVEL probes achieve their best F1 at this layer."
- **Problem**: The outline (Section 4, Figure 2) specifies "4 lines (one per hierarchy) across layers 6, 12, 18, 24 for the 16k SAE." The actual Figure 2 only has first-letter as a line plot with RAVEL hierarchies as L24-only data points. The rationale (poor probe quality at earlier layers) is reasonable but the outline was never updated to reflect this change. A reader who sees a line plot labeled "Layer-dependent absorption profile" expects all hierarchies to appear at all layers. The figure as described is a hybrid (lines for first-letter, points for others) that may confuse readers expecting the advertised format.
- **Fix**: Either (a) update the outline to match the actual figure, or (b) present RAVEL absorption at all layers with appropriate caveats about probe quality at earlier layers (perhaps using dashed lines or shaded uncertainty bands). Option (b) would strengthen the claim about layer dependence being universal, not just a first-letter phenomenon. At minimum, the figure caption should explicitly explain the hybrid format.

### Issue 4: "27x increase" claim in Section 4.2 is arithmetically wrong
- **Location**: Section 4.2 (line 19)
- **Quote**: "a 27x increase from layer 6 to layer 24"
- **Problem**: The text states absorption goes from 1.0% (L6) to 27.1% (L24). 27.1 / 1.0 = 27.1x, which is close to 27x but misleading because it implies the ratio is exactly 27. More importantly, the discussion (Section 7.5) says "15x--18x" between L6 and L24, which contradicts the 27x claim here. If the L6 value is actually 2.4% (as in Discussion), then 27.1/2.4 = 11.3x -- neither 27x nor 18x.
- **Fix**: Use the exact numbers and compute the ratio precisely. Reconcile with Discussion/Conclusion. If different measurement runs give different L6 values, clarify which measurement is canonical.

### Issue 5: Bonferroni p-values in pairwise tests do not match consolidation_summary
- **Location**: Section 4.1 (line 11)
- **Quote**: "city-language absorbs significantly less than first-letter at both 16k ($p_{\text{Bonf}}$ = 0.003, Cohen's $h$ = $-$0.73)"
- **Problem**: The consolidation_summary.json reports the raw permutation $p$ for city-language vs. first-letter as 0.0005, not 0.003. With 6 pairwise comparisons and Bonferroni correction, 0.0005 x 6 = 0.003, which matches -- but this should be stated more clearly. More concerning: the consolidation_summary reports city-continent vs. first-letter as $p$ = 0.34 (raw), which after Bonferroni would be $p$ = 2.04 (capped at 1.0). The section reports $p_{\text{Bonf}}$ = 1.0, which is correct only if there are 3+ comparisons where the corrected value exceeds 1. The text says "Bonferroni correction for 6 pairwise comparisons" in the method but reports "Bonferroni correction" without specifying the number of tests in Section 4. This omission weakens reproducibility.
- **Fix**: State the number of pairwise comparisons (6) and the correction factor explicitly in Section 4, not just in Method. The reader should not need to count.

### Issue 6: The non-monotonic dip interpretation lacks mechanistic support
- **Location**: Section 4.2, paragraph 2 (line 23)
- **Quote**: "The non-monotonic pattern -- a dip at layer 18 relative to layer 12 for first-letter -- suggests that absorption is not a simple function of layer depth. Instead, it reflects the model's computational demands at each layer..."
- **Problem**: This is an interpretive claim without supporting evidence. The section provides no mechanistic analysis of what happens at layer 18 vs. layer 12 to explain the dip. The claim that "layer 24 is the final residual stream position before the unembedding matrix, where the model concentrates its predictions" is accurate but does not explain why layer 18 dips below layer 12. An alternative explanation -- that the dip is a measurement artifact from different SAE training quality at different layers -- is not discussed.
- **Fix**: Either (a) remove the mechanistic speculation and simply report the non-monotonic pattern as an observation, or (b) provide evidence (e.g., SAE reconstruction error by layer, or probe quality by layer) that distinguishes mechanistic from artifactual explanations. At minimum, acknowledge the alternative explanation.

### Issue 7: Width effect discussion in Section 4.4 is entirely speculative
- **Location**: Section 4.4, paragraph 2 (lines 41-42)
- **Quote**: "The asymmetric response to width suggests that absorption in city-continent is driven by a structural property of the hierarchy (perhaps the small number of classes, $K$ = 6, combined with the extreme Europe concentration) that adding more features does not resolve."
- **Problem**: This paragraph is pure speculation with no supporting data. The word "perhaps" signals the lack of evidence. No analysis tests whether $K$, class balance, or Europe's dominance actually drive the width-invariance. The final sentence ("more dictionary entries provide more slots for parent features to survive alongside their children") is also ungrounded -- it could equally be that wider SAEs change the competitive dynamics in hierarchy-specific ways.
- **Fix**: Either (a) test the hypothesized relationship between $K$ / class balance and width effect (a simple correlation across the four hierarchies could be mentioned), or (b) present this as a hypothesis for future work rather than an explanation. Replace "suggests" and "consistent with the intuition" with "one possible explanation, untested in this work, is..."

## Minor Issues
- **Section 4.1, line 7**: "first-letter (27.1%, CI [24.5, 29.5])" -- The outline states "first-letter 42.9%" for L24_16k. Verify which number is canonical and fix all references.
- **Section 4.1, line 13**: "pattern is hierarchy-specific: absorption severity depends on the particular hierarchy's class count, balance, and the model's internal representation of that attribute" -- This lists three factors but none are tested in this section. Either cite a later section that tests them or soften to "may depend on."
- **Section 4.2, line 19**: The L18 values (2.0% at 16k, 1.0% at 65k) are reported inline but not discussed. A brief note on whether the 65k SAE's sharper L18 dip is significant would be informative.
- **Section 4.3, line 35**: "entity count in the training data affects whether the SAE allocates dedicated parent features" -- This is an untested causal claim. The correlation between entity count and absorption is observed, but "affects" implies causation. Use "correlates with" instead.
- **Section 4.5, line 47**: "The false negative count remains constant at 87/576 across all 20 cells" -- This is an extremely strong claim (exactly constant across 20 cells) that deserves more emphasis. The CV of 0.077 contradicts "constant" -- if FN count were truly constant, CV would be 0.0. Clarify: is the count exactly 87 in every cell, or approximately 87 with CV = 0.077?
- **Section 4.5, line 49**: The shuffled hierarchy control is described in one sentence without specific numbers. Report the actual shuffled-control absorption rate (presumably near 0%) with a CI.
- **Section 4.3, line 29**: "Europe absorbs at 90.2% (16k) and 92.0% (65k)" -- the 65k rate being higher than 16k contradicts the general trend reported in Section 4.4 that wider SAEs reduce absorption. This exception should be flagged explicitly.
- **Section title**: "Section 4: Cross-Domain Absorption Results" -- The outline labels this "Section 4" but the section title in the .md file includes the number. Ensure consistency with LaTeX sectioning (most templates auto-number).

## Visual Element Assessment
- [x] Figures/tables match outline plan (with the caveat in Issue 3 about Figure 2's format)
- [x] All visuals referenced before they appear (Table 2 in line 1, Figure 2 in line 1, Figure 3 in line 29)
- [x] Captions are self-explanatory (all three captions contain enough detail to be parsed independently)
- [ ] No text-heavy sections that need visual support -- Section 4.4 (Width Effect) and Section 4.5 (Statistical Robustness) are purely text-based. A small summary table or bar chart of the width effect by hierarchy would replace 4.4's prose. Section 4.5's threshold sensitivity claim would benefit from referencing the Appendix A table inline with a miniature version or a sentence like "see Table A1."

## What Works Well
- **Section 4.1's probe quality caveat paragraph** (line 15) is exemplary honest reporting. It explicitly names the gold-standard anchor (first-letter F1 = 1.0), explains why RAVEL rates are upper bounds, and identifies the specific hierarchy most susceptible to inflation (city-country at 80 classes). This is exactly the kind of pre-emptive reviewer concern management that strengthens a paper.
- **Section 4.3's per-class analysis** reveals a finding (23x within-hierarchy variance > 4x between-hierarchy variance) that is more striking than the headline result. The specific examples (USA: 0% absorption, Albania/Algeria: 100%; Europe: 90%, Africa: 4%) are vivid and immediately convey the magnitude of the effect. The cross-hierarchy pattern (city-country shows the same phenomenon) strengthens the generality.
- **The statistical reporting throughout Section 4** is strong: every claim carries a test statistic, $p$-value, effect size, or CI. The use of Kruskal-Wallis (non-parametric, appropriate for rate data) with Bonferroni-corrected pairwise tests and Cohen's $h$ for effect sizes reflects careful statistical practice.
