# Writing Quality Review

## Summary

The paper presents a three-part empirical study of feature absorption in sparse autoencoders: (1) testing whether a Lotka-Volterra competitive exclusion coefficient can serve as an unsupervised absorption detector (it cannot; F1 = 0.128), (2) testing whether corpus PMI predicts absorption (it does not; partial R^2 = 0.0006), and (3) testing whether absorption scores correlate with downstream SAE quality across 54 Gemma Scope SAEs (they do; r = -0.595 for sparse probing, surviving Bonferroni correction). A three-tier taxonomy classifies 92.3% of letter features as showing some absorption, though the dominant Type II component is acknowledged as likely inflated by heuristic parent identification. The paper honestly reports two negative results and one strong positive result, with appropriate caveats throughout.

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a clear problem-approach-evidence-conclusion arc. The three tensions introduced in the Introduction (measurement vs. phenomenon, mechanism vs. intervention, metric vs. impact) organize the entire paper and reappear in the Discussion and Conclusion, providing strong thematic unity. Section transitions are generally well-motivated: Section 2 identifies gaps, Section 3 formalizes the framework, Section 4 specifies the protocol, Section 5 reports results, and Section 7 interprets them.

Two structural issues:

1. Section 5.6 (Cross-Architecture Generalization) feels orphaned. It is not tied to any pre-registered hypothesis (H1-H4) and lacks a dedicated discussion subsection. It appears as an appendix-level supplement attached to the main results. Either promote it to a formal hypothesis or move it to an appendix/ablation section.

2. The Conclusion (Section 8) effectively restates the Discussion without adding new synthesis. The "Limitations" paragraph at the end of Section 8 repeats content from Section 7.6 nearly verbatim. Consider compressing the Conclusion and eliminating the duplicated limitations paragraph, or merging the two into a single "Discussion and Conclusion" section to save space.

### Notation & Terminology Consistency: 9/10

Cross-checking against `notation.md` and `glossary.md` reveals strong consistency. All major symbols ($\alpha_{ij}$, $\sigma_{ij}$, $f_i$, DAS, $L_0$, $D$, $d$) match their definitions. Terminology follows the glossary: "parent feature" / "child feature," "competition coefficient $\alpha_{ij}$" (not "LV coefficient"), "Chanin metric," "SAEBench" (capital B), "first-letter task" (hyphenated).

Minor deviations:

1. The abstract uses "latent directions" in the first sentence, then switches to "latents" throughout the rest of the paper. The glossary lists "latent" as the preferred formal term and "feature" as acceptable when unambiguous. The opening sentence should use "latents" for consistency with the rest of the paper.

2. Section 2.1 paragraph 3 uses "feature sensitivity" (Tian et al.) without defining it in the glossary. Since this is an external term being cited rather than adopted, this is acceptable, but a parenthetical clarification would help.

3. The notation table defines $\mathbf{d}_i$ as "Decoder column vector for latent $i$" but the paper sometimes writes "decoder columns" (plural, referring to the weight matrix) and sometimes "decoder column" (singular, referring to $\mathbf{d}_i$). Both usages are correct but the glossary could be more explicit about this distinction.

### Claim-Evidence Integrity: 8/10

All major numerical claims are verified against the source data files:

- H3 correlations (Table 5): Pearson r values match `C3A_saebench_corr.json` to the reported precision (e.g., -0.5948 reported as -0.595).
- PMI regression (Table 3): Coefficients match `C2C_regression_results.json` exactly (e.g., $\beta_4 = -0.006303$ reported as $-0.006$).
- Safety probe (Table 6): AUC values and probe gaps match `C3C_safety_probe.json` (e.g., highest-absorption dense AUC = 0.998, correctly reported as $\approx 1.0$; sparse AUC = 0.947, gap = 0.051).
- Taxonomy (Table 4): Letter classifications match `C2D_taxonomy.json`.

Issues requiring attention:

1. **Table 6 safety probe standard deviations**: The paper reports $0.882 \pm 0.067$ for the lowest-absorption sparse probe. The source data gives std = 0.0668. The rounded value 0.067 is acceptable but the notation $\pm$ SD is not explicitly declared (it could be SE). The table caption should clarify whether $\pm$ denotes standard deviation or standard error across CV folds.

2. **Taxonomy classification logic and Type III undercount**: Letter B in `C2D_taxonomy.json` has DAS(k=3) = 0.695 (exceeding the 0.6 threshold), yet is classified as Type II because the sequential evaluation checks Type II first. The paper acknowledges this ordering in Table 1 and Section 3.3, but the consequence is that the Type III count of 0 is partly an artifact of the sequential evaluation order. If Type III were evaluated before Type II, some letters currently classified as Type II would become Type III. This is a methodological point worth noting explicitly---the zero Type III rate at 24k width in Table 4 could mislead readers into thinking distributed absorption does not exist at that width, when in fact it is masked by the evaluation order.

3. **The "92.3% comprehensive rate" claim**: The paper appropriately flags this as an upper bound in multiple locations (abstract, Section 5.3, Section 7.4, Section 8). The caveat infrastructure is thorough. However, the abstract's first mention reads "classifies 92.3% of letter features as showing some absorption (vs. the canonical 15--35%)" with the Type II caveat deferred to a parenthetical 20 words later. A reader scanning only the abstract could take 92.3% at face value. Consider adding "(upper bound)" immediately after "92.3%" in the abstract, which is done later in Table 4 but not in the abstract's first mention.

### Visual Communication: 7/10

The paper has 4 generated PDF figures plus a conceptual diagram described in markdown but not yet rendered, and 8 inline tables. The visual audit confirms style consistency across generated figures.

Issues:

1. **Figure 1 does not exist as a PDF.** The `fig_lv_framework.pdf` file is referenced in Section 3.1 (`![LV Competition Framework](figures/fig_lv_framework.pdf)`) but the visual audit confirms it has not been generated. Only a markdown description (`fig_lv_framework_desc.md`) exists. This is the paper's conceptual anchor figure---its absence is a critical gap for compilation readiness. The paper references "Figure 1" in the Introduction (line 29: "Figure 1 illustrates the LV competitive exclusion framework") and again in Section 3 (line 104). Without this figure, the paper cannot be reviewed externally.

2. **Missing partial regression plot for H2.** The outline planned a partial regression scatter plot (residualized absorption vs. residualized log PMI) for Section 5.2. The experiments generated `C2C_partial_regression_plot.png` but it was not converted to a publication-quality figure. The H2 null result is currently supported only by Table 3 and text. A flat scatter cloud would provide immediate visual evidence. This is a minor gap since the null result is clearly communicated, but it would strengthen the presentation.

3. **Minimum visual elements check**: The paper has 1 method diagram (planned but absent), 1 results table (Table 5), and 3 analysis figures (Figures 2, 4, 5). The minimum threshold (1 method diagram, 1 results table, 1 analysis figure) is met in count but the method diagram is not yet rendered. Figure 3 (taxonomy bar chart) is an additional effective visualization.

4. **Table density in Section 5.4**: Tables 5 and 6 appear in quick succession with limited text between them. The safety probe results (Table 6) could benefit from visual separation or a brief introductory sentence.

### Writing Quality: 8/10

The writing is clear, direct, and appropriately technical. Sentences are generally well-constructed. Numbers are specific. Claims are qualified when evidence is limited. The paper earns credit for honest reporting of negative results.

Specific issues:

1. **One surviving banned pattern**: Section 2.4, last paragraph: "provides *complementary* evidence by testing whether the competition-coefficient formulation captures the dynamics that produce these partial minima." The word "complementary" is vague filler here; the sentence should specify *how* the evidence relates (e.g., "tests on real models whether the competition-coefficient formulation captures the dynamics that this framework identifies in synthetic settings").

2. **Sentence length outlier**: Abstract, sentence 3 (starting "First, we formalize...") runs 52 words before the period. While grammatically correct, it packs too many technical details into a single sentence. Consider splitting after the definition of $\alpha_{ij}$.

3. **Passive voice overuse in Section 4.2**: "Ground-truth absorption labels come from..." / "Letters A--M serve as..." / "The threshold $\tau$ ... is selected to maximize..." Three consecutive passive constructions. Rewriting the first as "We obtain ground-truth absorption labels from `sae-spelling`..." would improve readability.

4. **Redundant content between Discussion and Conclusion**: Section 7.3 and Section 8 both list the same four correlation values ($r = -0.595$, $-0.454$, $-0.431$, $-0.175$) with identical qualifying text. The Conclusion's "Limitations" paragraph duplicates Section 7.6 material. This repetition adds no information and costs roughly 150 words of space.

5. **Ambiguous referent in Section 5.1**: "The first $\alpha_{ij}$ bin ($[0, 0.1]$, $n = 369$) shows an anomalously high absorption rate of 0.848. This artifact arises because very low $\alpha_{ij}$ values correspond to pairs where both latents have similar low frequencies and high co-activation." The phrase "This artifact" is not clearly linked to the mechanism---it explains what the artifact *is* but not *why* high co-activation + similar low frequencies produces high absorption. An additional sentence clarifying the causal mechanism would help.

## Issues for the Editor

1. **Critical** -- **Figure 1 missing**: Section 3.1. The LV framework conceptual diagram is described in `fig_lv_framework_desc.md` but no PDF has been generated. The paper references "Figure 1" in both Section 1 and Section 3. **Fix**: Generate `fig_lv_framework.pdf` from the TikZ description before compilation. This blocks external review.

2. **Major** -- **Type III masking by evaluation order not discussed**: Section 5.3 / Table 4. The sequential evaluation order (Type II before Type III) means that letters meeting both Type II and Type III criteria are counted as Type II only. Letter B has DAS(k=3) = 0.695 > 0.6 but is classified Type II. The zero Type III rate at 24k width is partly an artifact of this ordering. **Fix**: Add one sentence in Section 5.3 acknowledging that some Type II features also meet the Type III criterion, and that the zero Type III count should not be interpreted as evidence that distributed absorption is absent.

3. **Major** -- **Redundancy between Discussion and Conclusion**: Sections 7.3 / 7.6 and Section 8. The same correlation values and limitations are stated twice. **Fix**: Compress Section 8 to reference the Discussion rather than re-listing numbers. Remove the duplicated Limitations paragraph from Section 8 or replace with "See Section 7.6 for a full discussion of limitations."

4. **Minor** -- **Abstract 92.3% without "upper bound" qualifier**: Abstract, line 5. The taxonomy rate is introduced without the upper-bound flag that appears in Table 4 and all subsequent mentions. **Fix**: Change "classifies 92.3% of letter features as showing some absorption" to "classifies 92.3% of letter features as showing some absorption (upper bound; see Section 5.3 caveat)."

5. **Minor** -- **Safety probe Table 6 lacks $\pm$ definition**: Section 5.4 / Table 6. The $\pm$ values after AUC scores are not labeled as SD or SE. **Fix**: Add a note to the table caption: "Values after $\pm$ denote standard deviation across 5 CV folds."

## What Works Well

1. **Transparent reporting of negative results**: The paper honestly reports that both H1 (LV detector, F1 = 0.128) and H2 (PMI, partial R^2 = 0.0006) failed against their pre-registered success criteria. Section 5.1's sharpness test goes further, explaining *why* the LV analogy fails structurally (no phase transition, frequency ratio confounds rarity with suppression). This intellectual honesty strengthens the paper's credibility for the positive H3 finding.

2. **Caveat infrastructure for the Type II inflation**: The paper flags the Type II limitation in Table 1 (footnote), Section 5.3 (dedicated "Caveat" paragraph), Section 7.4 (full subsection), and Section 8 (limitations paragraph). This repeated, graduated disclosure ensures readers at any level of engagement encounter the caveat. The suggestion that "a conservative interpretation would count only Type I (3.8%) and Type III (0%) as validated absorption" in Section 7.4 is particularly responsible.

3. **Partial correlation strengthening in H3**: The observation that partial correlations controlling for width, layer, and architecture *strengthen* the absorption-quality relationship (e.g., SCR from r = -0.431 to partial r = -0.677) is a strong analytical move that distinguishes genuine absorption signal from confounding. The accompanying caution about suppressor effects and wide confidence intervals at n = 49 shows statistical maturity.

SCORE: 8
