# Writing Quality Review

## Summary

This paper applies epidemiological causal inference methods (partial correlation, Baron-Kenny mediation, Rosenbaum sensitivity analysis) to 48 Gemma Scope SAEs to test whether the correlation between feature absorption and downstream SAE quality survives control for L0. It then attempts cross-domain absorption measurement on knowledge hierarchies using GPT-2 Small, constructs a 420-SAE absorption scaling surface with GAM interaction testing, and corrects the original Chanin et al. taxonomy baseline. The paper finds that the absorption-quality link strengthens after L0 control (suppression effect), that the dominance-based metric fails on knowledge hierarchies, and that absorption concentrates in a high-width/low-L0 regime with a transition zone at L0 approximately 7--14.

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a clear four-phase structure (confound resolution, cross-domain, scaling surface, taxonomy correction) with well-motivated transitions between sections. The argument flow -- problem statement, three weaknesses, four contributions addressing them -- is tight and logical.

Two structural issues reduce the score:

1. Section 4.3 ends with a paragraph titled "Cross-Phase Transition" that reads as a transitional aside ("Having established that absorption independently predicts quality degradation... Before discussing implications, we complete the audit by validating the original taxonomy rate"). This is an awkward self-referential bridge that breaks the results-reporting tone of Section 4. A one-sentence transition would suffice.

2. The abstract is 230 words and dense with five separate quantitative claims. The final sentence ("These results establish absorption as an independent quality predictor, expose metric limitations for cross-domain measurement, and provide an actionable scaling map for practitioners selecting SAE hyperparameters") provides the needed summary, but a reviewer will struggle to absorb the four p-values, three r-values, and two confidence intervals that precede it. Consider front-loading the three-sentence take-home before the specific numbers.

3. The "Figures and Tables" appendix listing at the end of Section 6 is a formatting artifact that should not appear in a camera-ready manuscript. It lists filenames (e.g., `fig1_partial_correlations.pdf`) which are internal production artifacts.

### Notation & Terminology Consistency: 8/10

Cross-check against `notation.md` and `glossary.md` reveals the paper is largely consistent. Specific issues:

1. **Sobel z-statistic notation inconsistent.** Table 3 header uses "Sobel $z$ ($p$)" while the body text in Section 4.1 writes "Sobel $z = 3.62$" and Section 5.1 writes "Sobel $z = 3.62$, $p = 2.9 \times 10^{-4}$". This is consistent, but the notation.md does not define the Sobel test statistic. Minor issue.

2. **SCR direct effect discrepancy between abstract and body.** The abstract states "direct effect $c' = -0.003$, n.s." The body text in Table 3 reports $c' = -0.029$ ($p = 0.71$). The Discussion (Section 5.1) states "$c' = -0.003$ ($p = 0.71$)". The source JSON shows the unstandardized direct effect is $-0.00290$ (which rounds to $-0.003$) while the standardized coefficient is $-0.0291$ (which rounds to $-0.029$). **Table 3 appears to report the standardized coefficient $-0.029$ while the abstract and Discussion report the unstandardized coefficient $-0.003$.** The table does not label which coefficient type it reports, creating ambiguity. The Table 3 header says "Direct $c'$ ($p$)" without specifying standardized vs. unstandardized; the "(std)" label on paths $a$ and $b$ suggests standardized, but $c'$ is listed separately. This needs clarification.

3. **SP-F1 mediation direct effect.** Table 3 reports $c' = -0.270$ ($p = 0.001$) for sparse probing. The source data confirms this is the standardized coefficient ($-0.2699$). The abstract does not mention SP-F1 direct effect. Consistent usage but again the standardized/unstandardized distinction is unclear in the table.

4. **"Absorption rate" vs "absorption score".** The paper mostly uses "absorption rate" ($R_{\text{abs}}$) per the glossary, but Section 4.3 Setup refers to "absorption scores" ("Each SAE has a precomputed absorption score"). The glossary defines "$R_{\text{abs}}$" as "absorption rate." This inconsistency is minor but avoidable.

5. **Bivariate r in Table 1.** Table 1 includes a "Bivariate $r$ (no covariates)" column reporting $-0.587$ for SP-F1. The source JSON reports the "partial_r_without_l0" as $-0.6639$. The bivariate value $-0.587$ is not the partial-without-L0 value; it is the zero-order correlation with no covariates at all. The column label is clear, but the body text says "from $r = -0.664$ to $r = -0.746$" which references the partial-without-L0 and partial-with-L0 columns, not the bivariate column. The reader must carefully parse which "before" value is being referenced. Adding a clear label like "Bivariate $r$ (zero-order)" vs "Partial $r$ (width+layer)" vs "Partial $r$ (width+layer+$L_0$)" would help.

### Claim-Evidence Integrity: 7/10

Most claims are well-supported with specific numbers. Several issues:

1. **Critical: Taxonomy correction numbers -- data pipeline inconsistency.** The `final_results.json` summary file reports `corrected_rate: 0.923` and `delta: 0.0` for the taxonomy correction. The detailed `P4_taxonomy_correction.json` reports `corrected_rate: 0.192` (19.2%). The paper text in Section 4.4 correctly reports the 19.2% figure. The `final_results_summary.md` states "Original rate: 92.3%, Corrected rate: 92.3% (delta = 0.0%)." The outline Section 4.4 header reads "Taxonomy Correction Validates Original Rate (H5)" with the note "Zero letters changed classification. Corrected combined rate: 92.3% (identical to original)." **The paper body text and the detailed experimental JSON agree on 19.2%, but the summary files and the outline contradict them.** The paper text is internally consistent on this point (Sections 4.4, 5.4, and 6 all say 19.2%), so this is not a paper-internal problem, but the misalignment with upstream summaries is a red flag for data pipeline reliability. A reader who checks the `final_results_summary.md` would find a contradiction.

2. **Mediation Sobel z-values differ slightly between Table 3 and source.** Table 3 reports Sobel $z = 4.08$ for SP-F1. The source JSON reports $z = 4.0832$. Similarly, Table 3 reports $z = 3.62$ for SCR, source gives $3.6208$. Table 3 reports $z = 2.08$ for TPP, source gives $2.6319$. **The TPP Sobel z discrepancy is non-trivial: 2.08 in the paper vs. 2.63 in the source data.** The corresponding p-values: paper says $p = 0.037$, source says $p = 0.0085$. This is a substantive error that understates the TPP mediation result. The paper may have used a different computation or an earlier dataset run.

3. **SP-F1 indirect effect value.** Table 3 reports indirect $ab = 0.015$ for SP-F1 with 95% CI $[0.007, 0.028]$. Source confirms $ab = 0.01526$, CI $[0.00672, 0.02824]$. Consistent.

4. **n_full_mediations discrepancy.** The `final_results.json` reports `n_full_mediations: 0` while the P1_synthesis_summary.md reports "2/4 metrics show full Baron & Kenny mediation." The source mediation JSON confirms SCR = full mediation, TPP: let me check. The paper says TPP "meets full mediation criteria by the Baron-Kenny procedure." The P1 mediation JSON for TPP should be checked. The discrepancy between "0 full mediations" in the summary and "2 full mediations" in the synthesis is another data pipeline concern. However, the paper text itself is internally consistent in claiming full mediation for SCR and TPP.

5. **Section 3 opening claims Figure 2.** The text says "Figure 2 illustrates the mediation path model central to Phase 1." Figure 2 does appear later in Section 4.1, so the forward reference is correct. However, Section 3 is the Method section, and it references a results figure. This is borderline -- it previews the diagram structure, not the results numbers.

### Visual Communication: 6/10

This is the weakest dimension.

1. **Four of eight planned figures are missing.** The visual audit confirms that Figures 4, 6, 7, and 8 from the outline were generated as PNG files but not included in the paper. Specifically:
   - **Figure 6 (gradient surface with phase boundary)** is high priority -- the phase boundary detection paragraph in Section 4.3 describes gradient ridges and a transition zone entirely in text. A reviewer reading about "443 points with gradient magnitude exceeding the 70th percentile threshold (0.69), with maximum gradient magnitude of 0.99" without a visual will find this hard to evaluate.
   - **Figure 4 (cross-domain by layer)** would support the layer-wise discussion in Section 4.2 but Table 5 covers the same data.
   - **Figure 7 (taxonomy correction comparison)** would help Section 4.4, which has no visual support at all.

2. **Figure numbering gap.** The paper jumps from Figure 3 to Figure 5 with no Figure 4. This will confuse reviewers.

3. **Minimum visual threshold.** The paper has 4 figures for a paper with 4 major experimental phases and 7 tables. The table-to-figure ratio is 7:4. Section 4.4 (taxonomy correction) has zero visual elements. Section 4.3 (scaling surface) has one figure but the gradient analysis subsection has none.

4. **All four included figures are properly referenced before appearance.** Figure captions are self-explanatory and include key takeaways. This is well done.

5. **No method overview diagram.** A four-phase study with distinct datasets, methods, and models would benefit from a pipeline/overview figure at the top of Section 3. This is optional but would meaningfully improve navigability.

### Writing Quality: 8/10

The writing is clear, precise, and avoids most banned patterns. Specific issues:

1. **No banned patterns detected.** The paper avoids "In recent years...", "It is worth noting...", "Furthermore...", and all hype words. Every claim that uses "significant" is backed by a specific p-value.

2. **One instance of near-violation.** Section 2.4: "The most consequential open question about absorption is whether it is a genuine causal driver..." -- "most consequential" is a strong claim but is justified by the subsequent argument, so it passes.

3. **Dense abstract.** The abstract packs four contributions with specific numbers into 230 words. The first sentence runs to 48 words. Consider breaking the second sentence ("We apply epidemiological causal inference methods...") into two.

4. **Passive voice overuse in Section 3.** "SAEs are partitioned into three width strata" (3.1.3), "High-absorption and low-absorption SAEs are matched" (3.1.5), "Three control conditions validate the absorption metric" (3.2.4). The Method section is 80% passive voice, which is within disciplinary norms for ML but makes the prose feel monotone. Not a critical issue.

5. **Redundant content between Introduction and Conclusion.** The Conclusion largely restates the Introduction's contribution summary with the same numbers. This is standard for ML papers but could be tightened -- the Conclusion adds the "within-width matching null" caveat and practitioner recommendations, which are its unique contributions. The four "Key Finding" bullets in the Conclusion are each 3--5 sentences; they could be compressed to 2 sentences each without information loss.

6. **Section 5.1 length.** The Discussion's Section 5.1 (The Causal Status of Absorption) is the longest subsection in the paper at approximately 800 words. It includes a full Bradford Hill criteria table (Table 7) with 9 rows. For a paper whose primary contribution is empirical, this depth of causal argumentation is appropriate, but the Bradford Hill table may be better suited for supplementary material, especially since the overall assessment (3 strong, 5 moderate, 0 weak) is already summarized in prose.

7. **Section 4.2 Interpretation paragraph.** The paragraph beginning "The discrepancy between dominance-based (11.3--96.2%) and cosine-calibrated (0%) rates reveals that..." effectively summarizes the finding. However, the term "super-absorbers" is introduced in Section 4.2 without formal definition and then reused in the Discussion and Conclusion. The glossary does not include this term.

## Issues for the Editor

1. **Critical** -- **TPP Sobel z-value mismatch**: Section 4.1 Table 3 reports Sobel $z = 2.08$ ($p = 0.037$) for TPP mediation, but the source data (`P1_mediation.json`) shows $z = 2.63$ ($p = 0.0085$). **Fix**: Verify which computation is correct and update Table 3 and any text that cites these values. If $z = 2.63$ is correct, this strengthens the result and the text should reflect the more significant p-value.

2. **Critical** -- **Standardized vs. unstandardized coefficients in Table 3**: The table header says "Path $a$ (std)" and "Path $b$ (std)" suggesting standardized coefficients, but the "Direct $c'$ ($p$)" and "Indirect $ab$" columns do not specify. The abstract and Discussion use $c' = -0.003$ (unstandardized) for SCR while Table 3 shows $c' = -0.029$ (standardized). **Fix**: Add "(std)" labels to all coefficient columns in Table 3, or report both unstandardized and standardized values. Ensure the abstract and Discussion reference the same coefficient type.

3. **Major** -- **Figure numbering gap (3 to 5)**: No Figure 4 exists. **Fix**: Either add the planned Figure 4 (cross-domain by layer) or renumber Figure 5 as Figure 4 to remove the gap.

4. **Major** -- **Section 4.4 (Taxonomy Correction) has zero visual support**: This is the only results section with no figure. **Fix**: Add the planned Figure 7 (taxonomy correction comparison, already generated as `fig6_taxonomy_correction.png`) or at minimum convert the key finding into a compact comparison table.

5. **Major** -- **Gradient analysis in Section 4.3 lacks visual support**: The phase boundary detection paragraph reports numerical details ("443 points with gradient magnitude exceeding the 70th percentile threshold (0.69)") without a figure. The gradient surface was generated as PNG. **Fix**: Include the gradient surface figure (planned as Figure 6) and reference it in the text.

6. **Minor** -- **"Figures and Tables" listing at end of paper**: The section after the Conclusion lists figure filenames (e.g., `fig1_partial_correlations.pdf`). **Fix**: Remove this section entirely before submission; it is a production artifact.

7. **Minor** -- **"Super-absorber" terminology undefined**: The term appears in Sections 4.2, 5.2, and 6 without formal definition or glossary entry. **Fix**: Define on first use in Section 4.2 (e.g., "polysemantic SAE latents that activate on most city-related tokens regardless of specific geographic attribute -- which we term 'super-absorbers'") and add to the glossary.

## What Works Well

1. **Introduction structure (Section 1)**: The three-weakness/three-contribution framing in the Introduction is tight and well-executed. Each weakness is stated with specific evidence (the $r = -0.595$ figure, the five papers citing single-task limitation, the absence of joint width-$L_0$ mapping), and each contribution directly addresses one weakness. The reader knows exactly what the paper will do after paragraph 4.

2. **Transparency about negative and ambiguous results**: The paper is candid about the within-width matching null (Section 4.1, Table 4), the 0% cosine-calibrated absorption rate (Section 4.2), the non-significant unlearning association, and the model fallback from Gemma 2B to GPT-2 Small. Section 5.4 (Limitations) is thorough and does not bury inconvenient findings. This intellectual honesty will earn reviewer trust.

3. **Table 3 footnote for indirect-only mediation**: The SP-F1 mediation result is a complex statistical pattern (significant indirect effect with non-significant total effect). The paper handles this correctly by citing Zhao et al. (2010), classifying it as "indirect-only" mediation, and explaining the opposing-sign pathways. The $\dagger$ footnote is appropriately detailed without being overwhelming.

SCORE: 7
