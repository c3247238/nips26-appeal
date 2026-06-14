# Critique: Introduction

## Summary Assessment
The introduction is well-structured and technically dense, clearly framing three weaknesses in the absorption literature and previewing three corresponding contributions. The writing is direct, quantitative, and avoids most filler patterns. However, the contribution previews front-load excessive statistical detail that belongs in Results/Method, the cross-domain contribution (Contribution 2) undersells its finding by burying the punchline, and several claims have inconsistencies with the Method and Experiments sections that must be resolved before submission.

## Score: 7/10
**Justification**: Strong problem framing, quantitative claims throughout, clear logical structure. Loses points for (a) Contribution 2's result numbers conflicting with Experiments Table 5, (b) taxonomy correction numbers in Experiments contradicting the intro's claim of "zero letters changed classification" and "92.3% identical to original," (c) excessive detail in contribution previews making the intro read like a compressed Results section, and (d) missing forward references to Figure 8 (Rosenbaum) which is in the outline. Reaching 8/10 requires resolving the factual inconsistencies, tightening contribution previews, and improving the cross-domain contribution framing.

## Critical Issues

### Issue 1: Taxonomy correction claim contradicts Experiments section
- **Location**: Paragraph 9 (Contribution 2) and final summary paragraph
- **Quote (intro, line 37-40)**: "The metric does not discriminate real from randomized hierarchies..." followed later by the taxonomy section summary
- **Quote (intro, line 50)**: "Together, these contributions transform feature absorption from a narrowly validated observation..."
- **Problem**: The intro does not explicitly preview Contribution 4 (taxonomy correction) as a numbered contribution, but the outline and Experiments section include Section 4.4. More critically, the experiments text (Section 4.4) reports that after frequency-matched correction, the comprehensive rate **drops from 92.3% to 19.2%** and "19 letters change classification." This directly contradicts the outline's Section 4.4 summary which says "Zero letters changed classification. Corrected combined rate: 92.3% (identical to original)." The intro must be reconciled with whichever version is correct. If the Experiments section is authoritative, the intro's framing of the taxonomy as a minor validation exercise is wrong -- it is a major finding that the original 92.3% rate was an artifact.
- **Fix**: Determine the ground-truth taxonomy result (19.2% corrected or 92.3% unchanged). If the Experiments section (19.2%) is correct, add a fourth contribution bullet previewing this finding and update the outline. If the outline (92.3%) is correct, update the Experiments section.

### Issue 2: Cross-domain absorption rate range inconsistent with Experiments data
- **Location**: Contribution 2 paragraph (lines 36-40)
- **Quote**: "the standard dominance-based absorption metric (Chanin et al., 2024) produces 51--85% absorption rates across knowledge domains"
- **Problem**: Experiments Table 5 reports absorption rates ranging from 11.3% (Country binary US, layer 11) to 96.2% (Language binary English, layer 8). The intro's "51--85%" range does not match. The low end excludes the 11.3% result at layer 11, and the high end excludes the 96.2% result. This appears to be a different aggregation (perhaps domain-level means), but it is stated as if it is the raw range. The reader who checks Table 5 will see a discrepancy.
- **Fix**: Either report the actual range (11.3--96.2%) or explicitly state "domain-averaged absorption rates of 51--85%." If a specific aggregation is used, state the aggregation method.

### Issue 3: Number of SAEs inconsistent between intro and method
- **Location**: Contribution 1 paragraph (line 28)
- **Quote**: "We apply partial correlation analysis, Baron-Kenny mediation, and Rosenbaum sensitivity analysis to 48 Gemma Scope SAEs"
- **Problem**: The proposal (Section "Motivation") references "54 Gemma Scope SAEs" from iter_4, and the Method section explains that 6 canonical SAEs were excluded due to missing L0 values, yielding 48. The intro correctly says 48, but does not explain the discrepancy with the 54 mentioned in the confound problem paragraph (line 11: "correlation of r = -0.595 across 54 Gemma Scope SAEs"). A reader may wonder why the analysis uses 48 but the prior work used 54.
- **Fix**: Add a parenthetical: "to 48 Gemma Scope SAEs (the 54 in the prior study minus 6 lacking reported L0)" or similar. The Method section already explains this, but the intro should not leave the reader confused.

## Major Issues

### Issue 4: Contribution previews are too detailed for an introduction
- **Location**: Lines 27-48 (Contributions 1-3)
- **Quote (example from Contribution 1)**: "After controlling for log(L0), the partial correlation between absorption and sparse probing F1 *strengthens* from r = -0.664 to r = -0.746 (p = 1.2 x 10^{-9}), a classical suppression effect..."
- **Problem**: Each contribution paragraph reads like a compressed results section. Contribution 1 alone reports 7 specific numbers (two partial correlations, a p-value, two more partial correlations with p-values, a direct effect, and a Sobel z). Contribution 3 reports 5 numbers (R^2 values, a p-value, log2(L0) range, and L0 thresholds). While being specific is good, this level of detail obscures the high-level narrative. An introduction should tell the reader *what* was found and *why it matters*, with 1-2 headline numbers per contribution. The full statistical detail belongs in Sections 4.1-4.3.
- **Fix**: Reduce each contribution to its single most important quantitative finding plus a one-sentence "so what." For example, Contribution 1: "After controlling for L0, the absorption-quality association strengthens rather than weakens (a suppression effect), and mediation analysis establishes absorption as the primary pathway through which L0 affects quality." Move the specific r values, p-values, and sensitivity bounds to Results.

### Issue 5: No explicit statement of what absorption costs practitioners
- **Location**: Opening paragraphs (lines 1-8)
- **Problem**: The intro explains what absorption is (line 6) and that it "directly undermines this promise" (line 5), but never states what the *practical consequence* is. Does absorption cause incorrect model edits? Does it make circuit analysis unreliable? Does it reduce steering performance? The reader knows absorption makes latents fail to fire, but needs a concrete example of what goes wrong downstream. The strongest motivation would connect to a real use case.
- **Fix**: Add 1-2 sentences after line 7 illustrating a practical failure. For example: "When a researcher steers model behavior by activating the 'starts with S' latent, absorption means September-related inputs are silently excluded from the intervention. More broadly, any downstream analysis that relies on SAE latent activations -- circuit discovery, feature attribution, model editing -- inherits absorption's silent failures."

### Issue 6: Missing forward reference to Figure 8 (Rosenbaum sensitivity)
- **Location**: Contribution 1 paragraph (line 33)
- **Quote**: "see Figure 1 for partial correlation comparisons and Figure 2 for the mediation path diagram"
- **Problem**: The outline includes Figure 8 (Rosenbaum Sensitivity Analysis), which is a key visual for Contribution 1's robustness claim. The intro references Figures 1 and 2 but omits Figure 8. The Rosenbaum Gamma = 2.65 is mentioned in the text (line 33) but without a figure reference.
- **Fix**: Add "and Figure 8 for the Rosenbaum sensitivity bounds" to the forward reference in Contribution 1.

### Issue 7: Roadmap paragraph is mechanical and wastes space
- **Location**: Lines 53-58
- **Quote**: "The remainder of this paper is organized as follows. Section 2 reviews SAE architectures..."
- **Problem**: The "organized as follows" roadmap is a standard template that adds no information for a reader who can see the section headings. At top venues, this space is better used for a brief paragraph on the paper's broader significance or a crisp summary of the actionable takeaways. The current roadmap is 6 lines that merely restate the table of contents.
- **Fix**: Either remove entirely (most top-venue papers omit this) or compress to 2 sentences: "Section 2 contextualizes the confound problem. Sections 3--4 present the methods and results for each contribution in sequence, followed by discussion (Section 5) and practical recommendations (Section 6)."

## Minor Issues

- **Line 7**: "On the first-letter spelling task---where a general 'starts with S' latent is absorbed by a specific 'September' latent" -- The absorbing direction is reversed from the glossary definition. The glossary says "a more specific co-occurring latent subsumes its role," meaning the "starts with S" latent fails because "September" subsumes it. The intro's phrasing "'starts with S' latent is absorbed by a specific 'September' latent" reads as if "starts with S" is the object being absorbed, which is the correct interpretation but the verb phrase is slightly ambiguous. Clarify: "the general 'starts with S' latent is silently suppressed by a specific 'September' latent." -> This is a style preference; the current version is technically correct but could be misread.
- **Line 12**: "But this correlation was computed without controlling for L0 (the expected number of active latents per input)." -- L0 is defined here, which is correct for first use. However, the glossary specifies "Always styled as 'L0' (capital L, zero), not 'l0' or '$\ell_0$' in running text." The intro uses $L_0$ in equations, which is fine per the notation table ($L_0$ in equations). Consistent.
- **Line 18**: "At least five papers explicitly call out this limitation" -- Consider citing them inline rather than in a parenthetical list. The current format (Chanin et al., 2024; Karvonen et al., 2025; Korznikov et al., 2025; Bussmann et al., 2025; Li et al., 2025) is correct but dense. This is fine for an ML venue.
- **Line 28**: "methods standard in epidemiology and social science but never previously applied to SAE evaluation" -- This is a strong novelty claim. Verify no one has used mediation analysis in SAE or mechanistic interpretability work since the proposal was written.
- **Line 36**: "five probe types over 3,552 cities from the RAVEL dataset on GPT-2 Small" -- Method section confirms GPT-2 Small was used due to Gemma 2B access restrictions. The intro should note this is a fallback rather than a deliberate choice, or at minimum not imply it was the planned model.
- **Line 43**: "spanning dictionary widths from 2,304 to 1,048,576" -- This is consistent with Method Section 3.3.1. Good.
- **Line 44**: "A generalized additive model (GAM)" -- First use with expansion. Good.
- **Figure comment block (lines 60-65)**: The HTML comment block referencing figure generation scripts should be removed before submission.

## Visual Element Assessment
- [x] Figures/tables match outline plan (Figures 1, 2, 5, 6 are referenced)
- [x] All visuals referenced before they appear (Figures 1, 2 referenced in Contribution 1; Figures 5, 6 in Contribution 3)
- [ ] Figure 3 (cross-domain with shuffled controls) is NOT referenced in the intro despite being the key visual for Contribution 2
- [ ] Figure 8 (Rosenbaum sensitivity) is NOT referenced despite being relevant to Contribution 1
- [x] No text-heavy sections that need additional visual support

## What Works Well

1. **The three-weakness framing (lines 9-23)** is exceptionally clear. Each weakness is named ("confound problem," "single-task problem," "scaling problem"), described in one paragraph with specific numbers, and directly mapped to a contribution. This structure makes the paper's value proposition immediately obvious to a reviewer.

2. **The confound problem paragraph (lines 11-15)** is the strongest paragraph in the section. It identifies the exact statistical confound (all high-absorption SAEs are 1M/low-L0, all low-absorption are 16k-65k/high-L0), names the alternative explanation ("merely a proxy for L0"), and states the consequence ("the entire absorption-quality narrative collapses"). This is the kind of precise problem statement that gets a reviewer's attention.

3. **Contribution 2's honesty about a "critical methodological limitation"** (line 37). Rather than spinning the cross-domain result as a success, the intro directly states that the metric fails, the shuffled controls show 100%, and the cosine-calibrated variant detects 0%. This candor about negative/mixed results is rare and strengthens credibility.
