# Critique: Experiments (Sections 4, 5, 6)

## Summary Assessment

The experiments section delivers three well-structured empirical pillars: a metric audit with universal control failure, an L0-absorption phase transition, and a rate-distortion diagnostic. The data are reported with commendable precision and appropriate statistical qualifications. The section's strengths are its rigorous control analysis and honest reporting of negative results. However, several internal inconsistencies between reported numbers, a missing Table 3, inconsistent confidence interval reporting, and the dimension sensitivity problem in Section 6 weaken the overall presentation. The section reads as roughly 70% of the way to a strong NeurIPS submission.

## Score: 6/10
**Justification**: The empirical content is substantial and the negative finding (control failure) is genuinely important. To reach 7-8, the section needs: (1) resolution of the numerical inconsistencies between Sections 4 and 5, (2) the missing Table 3, (3) tighter CI reporting, (4) a more honest reckoning with the CMI dimension instability rather than presenting it as a "limitation" while still treating the d'=10 result as the headline finding, and (5) clearer visual-textual integration. The raw material is strong; the execution needs one more editing pass.

## Critical Issues

### Issue 1: Numerical inconsistency between Section 4.2 and Section 5.1 on L0=82 absorption rate
- **Location**: Section 4.2 (paragraph 3 final sentence) vs. Section 5.1 (paragraph 2)
- **Quote (4.2)**: "The 96.9% hierarchy-driven figure from the pilot was a methodological artifact of single-L0 classification."
- **Quote (5.1)**: "The L0=82 rates from the multi-L0 confound decomposition (14.39%, measured on 1,195 words with F1=1.0 probes) are slightly lower than the improved first-letter rates (15.96%, measured on 1,204 words with mean F1=0.817 probes)."
- **Problem**: Section 4.3 reports the improved first-letter aggregate absorption rate as 15.96% on L12-16k at L0=82, and Section 5.1 reports 14.39% from the confound decomposition at the same L0. These two numbers are measuring absorption at L0=82 on essentially the same SAE, but the text passes quickly over this discrepancy with a one-sentence explanation about "probe quality gate." The reader is left uncertain which number to trust for the L0=82 absorption rate on L12-16k. This is a central figure in the paper and the ambiguity is damaging.
- **Fix**: Add a dedicated paragraph or footnote explicitly reconciling these two numbers. State the exact vocabulary overlap (1,195 vs. 1,204 words), the exact probe difference (F1=1.0 at L0=22 projected to L0=82 vs. probes trained at L0=82 with mean F1=0.817), and whether the 14.39% or 15.96% is the canonical L0=82 number for this paper. Use one consistently in all subsequent references.

### Issue 2: Confidence interval for first-letter absorption contains its own point estimate outside the interval
- **Location**: Section 4.3, first paragraph
- **Quote**: "the aggregate absorption rate is 15.96% (95% CI: [8.4%, 17.5%])"
- **Problem**: A 95% CI of [8.4%, 17.5%] with a point estimate of 15.96% is technically possible but highly asymmetric (the point estimate is only 1.54 percentage points from the upper bound but 7.56 from the lower bound). This extreme left skew of the bootstrap distribution is unusual for a rate and should be explained. It suggests the distribution has a heavy left tail, possibly driven by a few letters with very high absorption dragging the aggregate. Without explanation, a reviewer will flag this as a potential implementation error.
- **Fix**: Either (a) verify and briefly explain the extreme asymmetry (e.g., "the wide CI reflects high per-letter variance, with 6 of 25 letters showing 0% absorption"), or (b) if this is a reporting error, correct it.

### Issue 3: Section 4.4 reports CI for city-language as [0%, 4.3%] but the point estimate is 6.56%
- **Location**: Section 4.4, first sentence
- **Quote**: "city-language 6.56% (CI: [0%, 4.3%])"
- **Problem**: A point estimate of 6.56% cannot lie outside its own 95% confidence interval of [0%, 4.3%]. This is a clear error---either the CI or the point estimate is wrong.
- **Fix**: Verify and correct. If the CI is from a different computation (e.g., per-class bootstrap), state this explicitly. Otherwise replace with the correct interval.

## Major Issues

### Issue 4: Table 3 is referenced in the outline but missing from the section
- **Location**: Section 4.3 (entire subsection)
- **Problem**: The outline specifies "Table 3: Improved First-Letter Results" with columns for Letter, N_words, Probe F1, Absorption Rate, Shuffled Rate, and Net Signal. This table does not appear in the experiments section. The per-letter data are partially described in text (Section 4.3 paragraph 3: "Per-letter absorption rates range from 0.0% (J, K, Q, V, Y, Z) to 36.76% (P)") but the systematic tabulation is absent. This is a significant gap for a paper that positions improved first-letter results as a key methodological contribution.
- **Fix**: Add Table 3 as specified in the outline, or explicitly redirect to Appendix A with a forward reference. The table is essential for reviewers to verify the per-letter claims.

### Issue 5: Confound decomposition classification is internally inconsistent with the glossary definition
- **Location**: Section 4.2, classification definitions
- **Quote (Section 4.2)**: "Hedging: 648 of 657 (98.6%) --- tokens whose parent information is spread across many latents, none clearing the JumpReLU threshold at this sparsity level, but which resolve at higher L0."
- **Quote (glossary)**: "Hedging: ...Distinguished from absorption by persistence across all L0 values."
- **Quote (method 3.4)**: "Hedging. The token is a false negative at the current L0 but recovers at a higher L0"
- **Problem**: The glossary says hedging tokens persist "across all L0 values." The method section (3.4) and experiments section (4.2) say hedging tokens recover at higher L0. Meanwhile, the glossary defines hierarchy-driven absorption as "appearing only at low L0 values and recovering at higher L0"---which sounds like the definition the experiments section uses for hedging. The classification criteria are inverted between the glossary and the actual analysis. A reviewer checking the glossary against the method will be confused.
- **Fix**: This is likely a glossary error (the glossary seems to have the definitions swapped). The method section and experiments section are internally consistent with each other. Update the glossary to match: hedging = resolves at higher L0; hierarchy-driven = persists across all L0 values.

### Issue 6: Section 6.5 dimension sensitivity undermines Section 6.2 headline result without adequate statistical framing
- **Location**: Section 6.5 and Section 6.2
- **Problem**: The paper presents the d'=10 CMI result as the headline finding of Section 6 (Spearman rho = -0.383, Cohen's d = -0.924, Mann-Whitney p = 0.045). Section 6.5 then reveals that (a) the correlation reverses sign at d' >= 20, (b) Bonferroni-corrected p = 0.236, (c) this is a 4-test sweep. The uncorrected p = 0.059 already misses the 0.05 threshold for the rank correlation. After correction, neither the rank correlation nor the group comparison is significant. The section concludes by calling d'=10 "the strongest information-theoretic signal in our data" with a "large effect size," but this is a post-hoc claim from a 4-dimensional search where only 1 of 4 dimensions shows the expected direction. A critical reviewer will read this as p-hacking or at minimum as an inadequately corrected multiple comparison.
- **Fix**: Either (a) present d'=10 as pre-registered (if it was---check the proposal, which does say "pre-registered analysis dimension d'=10") and argue that the Bonferroni correction over the sensitivity analysis dimensions is overly conservative since d'=10 was the primary analysis, or (b) lead with the Bonferroni-corrected result and frame the d'=10 finding as "suggestive but not significant after correction." The current structure---headline first, caveat later---will draw reviewer fire. The proposal does mention d'=10 as the primary dimension, which is a legitimate defense but needs to be stated explicitly in the experiments section.

### Issue 7: Section 5.2 Table 4 has missing cells and unclear presentation
- **Location**: Section 5.2, Table 4
- **Problem**: Table 4 is described as a "34-configuration scaling grid" but the presented table is a 3x5 matrix with many empty cells ("--") and ranges instead of single values. The "Low/Mid/High L0" binning loses the individual configuration data. The outline specifies "Rows: L0 values {22, 41, 82, 176}; Columns: widths {16k, 65k}" but the actual table uses 5 width columns (16k through 262k) with 3 L0 bins. This is more detailed than the outline's 2-width plan but less interpretable due to the sparseness and binning.
- **Fix**: Either (a) present the full 34-configuration data in the appendix and show a clean summary table in the main text with individual L0 values (not bins) for the most important widths, or (b) present a heatmap figure that shows the full surface. The current table is neither compact enough for a summary nor complete enough for verification.

### Issue 8: Section 5.3 cross-architecture comparison lacks matched controls
- **Location**: Section 5.3, final paragraph
- **Quote**: "The comparison is cross-model (Gemma 2 2B, d_model = 2304 vs. GPT-2 Small, d_model = 768), so model capacity differences confound the architecture comparison."
- **Problem**: The section presents JumpReLU vs. L1-ReLU absorption rates (42.9%-0.8% vs. 61-67%) as a key finding about architecture-dependent phase transitions, then acknowledges the comparison is confounded. The caveat comes only at the end after the reader has already absorbed the comparison as meaningful. The method section (3.1) and discussion (7.5) repeat this limitation, but the experiments section should flag it at the point of comparison, not afterward.
- **Fix**: Move the cross-model caveat to the first sentence of Section 5.3 or integrate it into the comparison: "Although this comparison is confounded by model capacity differences (Gemma 2 2B, d_model=2304 vs. GPT-2 Small, d_model=768), the directional contrast is informative: GPT-2 Small with L1-ReLU SAEs shows..."

## Minor Issues

- **Section 4.1, paragraph 1**: "The random probe control, expected to produce < 2% absorption, yields 11.8% on first-letter and 9.2--34.3% across other domains." The range 9.2--34.3% does not match Table 2, which shows random probe values of 11.8%, 12.9%, 20.8%, 34.3%, 19.0%. The range should be 12.9%--34.3% for the non-first-letter domains. Fix: correct to "12.9--34.3% across other domains" or "9.2--34.3% across all domains."

- **Section 4.1, Table 2**: The "Ratio" column header is defined in the caption as "shuffled/measured" but the table header just says "Ratio." Add the definition to the header or a table footnote for self-explanatory reading.

- **Section 4.3**: "with 10 of 25 letters passing the F1 > 0.85 quality gate" contradicts Section 3.2 which says "At L0=82 on L12-16k, 10 of 25 letters pass this gate." This is consistent, but the outline says "Table 3: 18/25 letters pass F1 > 0.85 gate." The outline number (18/25) differs from the text (10/25). Verify which is correct.

- **Section 5.1**: "Spearman rho_s = -1.0, p < 0.001" for a 4-point monotonic sequence. With n=4, Spearman rho = -1.0 has an exact p-value of 0.042 (or 0.083 depending on the test variant), not < 0.001. This appears to be an incorrect p-value. Fix: report the exact p-value for n=4.

- **Section 6.1**: "3,691 total samples" --- the sample count jumps from the text without prior context. The method section (3.5) says "word vocabulary plus 10,000 corpus tokens." The experiments section says "1,092 word activations plus 2,599 corpus token activations." The 2,599 corpus tokens is much smaller than the 10,000 stated in the method. Fix: reconcile or explain the difference (e.g., filtering, per-letter subsetting).

- **Section 6.2**: "Bootstrap 95% CI on mean difference: [-0.408, -0.013]" --- this CI excludes zero, which would normally imply significance at the 0.05 level, but the Spearman correlation p-value is 0.059 (not significant). The apparent contradiction arises because the bootstrap CI is on the group-mean difference (a different test) while the p-value is from the rank correlation. This is not wrong but should be flagged to prevent reviewer confusion. Fix: add a parenthetical: "the bootstrap CI tests the group-mean difference while the Spearman p tests the rank-order relationship."

- **Section 6.3**: "lambda is the effective sparsity pressure inferred from the empirical half-maximum L0 (22.4)" --- lambda is not defined numerically. The reader cannot verify the calculation. Fix: state lambda's value.

- **Figure/Table numbering**: The discussion (Section 7.3) references "Figure 5" for the dimension sensitivity plot, but the experiments section does not include Figure 5 or reference it. The experiments section covers Figures 1-4. Section 6.5 presents the dimension sensitivity data in a table but no figure. Fix: either add Figure 5 to Section 6.5 or move it to the discussion as currently planned, but add a cross-reference in Section 6.5 ("see Figure 5 in Section 7.3" or similar).

- **Terminology**: Section 4.2 uses "false negatives" without defining the term relative to the absorption criterion. The method (3.2) defines the absorption criterion but does not explicitly state that tokens meeting conditions (1) and (2) are "false negatives." Fix: add a brief definition in Section 4.2: "false negatives (tokens where probe-associated latents fail to fire despite correct probe classification)."

## Visual Element Assessment

- [x] Figures/tables match outline plan (Figures 1-4 and Tables 2, 4 present; Table 3 is missing)
- [x] All visuals referenced before appearance (Figure 1 referenced in 4.1, Figure 2 in 4.2, Figure 3 in 5.1, Figure 4 in 6.2)
- [ ] Captions are self-explanatory --- Figure captions are written as alt-text descriptions inside markdown image syntax rather than formal captions. They are reasonably descriptive but use informal phrasing and would benefit from more structured caption formatting (e.g., "(a) ... (b) ..." for multi-panel figures, explicit axis definitions).
- [ ] No text-heavy sections that need visual support --- Section 5.2 (Width-L0 Interaction) reports dense statistical results (OLS, GAM, regression coefficients) without a visual. A heatmap of the 34-configuration surface would substantially improve readability. Section 4.4 (Cross-Domain Rates) is entirely textual with no dedicated figure; the data appears in Table 2 but a per-class breakdown (especially the Asia outlier) would help.
- [ ] Table 3 is missing (outline-specified, not present)

## What Works Well

1. **Section 4.1's opening paragraph** delivers the core finding with maximal clarity: five domains, five ratios, one conclusion. The structure (domain: measured vs. shuffled, ratio) is immediately scannable. The "infinity" ratio for city-country (0.0% measured vs. 10.3% shuffled) is a particularly effective data point.

2. **Section 4.2's decomposition narrative** is well-structured: it presents the breakdown at L0=22, shows the monotonic trend across all four L0 values, identifies the 9 persistent words by name, and connects back to the hypothesis (H2 falsification) with specific numbers. The identification of specific words ("eight," "lower," "liked," "offer," "often") grounds the analysis concretely.

3. **Section 6.4's geometric constant degeneration** is an efficient theoretical simplification that the section explains clearly in two paragraphs: the constant is nearly invariant (mean 0.960, CV 2.16%), adding it to CMI does not improve the correlation, and the practical implication is that for normalized SAEs the threshold simplifies to lambda/CMI. This is clean technical writing.
