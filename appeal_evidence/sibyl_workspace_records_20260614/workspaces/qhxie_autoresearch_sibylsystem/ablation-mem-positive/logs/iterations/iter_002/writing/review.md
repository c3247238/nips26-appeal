# Writing Quality Review

## Summary

This paper investigates feature absorption in sparse autoencoders (SAEs) across two architectural families (GemmaScope JumpReLU and GPT-2 ReLU), using 60 semantic probes to validate scalable probe construction, test a training-free detector ($A_j$), and establish cross-architecture stability of projection-based absorption metrics. The paper discovers an unexpected layer-dependent $A_j$ correlation pattern that peaks at mid-layers and demonstrates that decoder norm constraints persist across architectures. The prior review (SCORE: 6) identified six critical/major issues; an editor round has addressed all of them. The paper now presents a coherent, internally consistent narrative with honest hypothesis framing and specific evidence leading every claim.

---

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a standard IMRaD structure with a clear argument chain: problem (absorption undermines interpretability) $\rightarrow$ three gaps $\rightarrow$ five contributions $\rightarrow$ three experiments validating three hypotheses $\rightarrow$ implications for benchmarks. The abstract accurately represents content and key numbers (91.2% vs 98.2%, $\rho = 0.705$, $p = 0.023$).

The H2 narrative arc has been fixed. Contribution 3 now honestly frames the layer-dependent pattern as emerging from a failed hypothesis: "While our original hypothesis (H2) that $A_j$ would correlate positively across all GPT-2 layers failed (mean $\rho = -0.194$), we discovered a non-monotonic correlation..." The Conclusion reinforces this: "H2 failed as originally stated... However, this failure revealed an exploratory finding." This transparent framing strengthens rather than weakens credibility.

Minor deduction: Section 1.4's paper organization paragraph remains generic ("Section 2 reviews... Section 3 describes..."). It could be tightened by adding interpretive guidance about what the reader should take away from each section.

### Notation & Terminology Consistency: 9/10

Notation is consistent and well-defined. Key symbols ($A_{\text{abl}}$, $A_{\text{proj}}$, $A_j$, $j^*$, $\rho$, $d_j$, $e_j$) are defined before first use and used consistently. The $A_j$ formula was corrected from the incorrect "j-th column" to the dimensionally correct "j-th encoder row" formulation.

Hyphenation is consistent throughout: "ablation-based," "projection-based," "training-free," "cross-architecture," "layer-dependent," "mid-layer."

The "training-invariant" overclaim has been weakened to "Evidence that decoder norm constraints persist across the two architectures tested" with the explicit caveat: "Architectural effects cannot be ruled out with only two SAE families." This is appropriately scoped.

One minor issue: The term "functionally insensitive" is used consistently for the ablation metric, but the paper also uses "near-universally zero" (Section 3.3.1) and "functional insensitivity" (Section 5.4). These are conceptually consistent but the phrasing variety is slightly distracting.

### Claim-Evidence Integrity: 8/10

Most claims cite specific numbers, figures, tables, or references. The paper excels at leading with evidence. Tables 1-4 are all referenced before they appear. The pilot study claim ("3 failed probes out of 30") now includes context: "our pilot with 5 hyponyms (Iteration 2)"---though it still lacks a direct citation to a specific file.

The ablation rate inconsistency has been fixed. Section 5.4 now correctly states: "While the ablation metric flags 0-33% of probes as absorbed (0.0% on GemmaScope E3v2, 33.3% on GPT-2 E7), the projection metric flags 91-98%. This 60-70 point gap---not the low ablation rate itself---is the evidence for functional insensitivity." This is precise and internally consistent.

Cohen's $d = 1.82$ is now correctly characterized as "a large effect size by conventional thresholds" (Section 3.3.2, Conclusion 6.1), while noting "the absolute difference is modest." This distinction between statistical and practical significance is clear.

Minor issue: Table 3's footnote about the 91.9% ablation score difference still appears below the table rather than inline, which could mislead a scanning reader.

### Visual Communication: 5/10

The paper has 6 figures and 4 tables, meeting minimum requirements. All figures/tables are referenced before they appear. Captions are self-explanatory and include key statistics.

**Critical issue: No compiled figures exist.** The visual audit confirms that all 6 referenced figures (Figures 1, 3, 4, 5, 6, 7) have TikZ source files but no compiled PDFs in the workspace. The paper references `figures/fig1_pipeline.pdf`, `figures/fig3_auroc.pdf`, etc., but the `writing/figures/` directory is empty. This is a blocking issue for compilation and external review---a paper with 6 referenced figures and zero actual figures cannot be submitted.

Figure numbering has a gap (1, 3, 4, 5, 6, 7) since Figure 2 was intentionally omitted. This is acceptable but slightly odd.

Text-heavy sections that would benefit from visuals: Section 3.3 (three absorption metrics) would benefit from a small diagram visually distinguishing ablation, projection, and $A_j$. Section 5.3 (three hypotheses for the layer-dependent pattern) would benefit from a conceptual flowchart.

### Writing Quality: 8/10

The writing is clear, with short direct sentences and specific numbers leading most claims. Banned patterns have been eliminated---no "Moreover," "Furthermore," or "It is worth noting that" survive in the text.

The Discussion sections (5.1-5.3) have been revised to lead with interpretive questions rather than number restatement. Section 5.1 opens with "Why do JumpReLU and ReLU SAEs... produce projection absorption rates that differ by only 7.7 percentage points?" Section 5.2 leads with the implication for $A_j$ interpretation. Section 5.3 leads with practical guidance for practitioners. This is a significant improvement over the prior draft.

Passive voice is present but not excessive. The writing maintains an appropriate academic register without becoming ponderous.

Minor issues:
- Section 3.4 still contains the unsupported claim that "GemmaScope JumpReLU also maintains norms near 1.0, though whether this is enforced architecturally or emerges from training is not verified in our analysis." The caveat is present but the sentence structure still implies a contrast with GPT-2 that may not exist.
- The phrase "Distinguishing these explanations requires feature-level analysis beyond the scope of this work" appears verbatim in both Section 4.4 and Section 5.3. This self-repetition should be removed from one location.

---

## Issues for the Editor

1. **Critical** **Missing compiled figures**: All 6 referenced figures have TikZ source files in `iter_003/exp/results/pilots/f1_tikz_figures/` but no compiled PDFs exist in `writing/figures/`. The paper cannot be compiled or reviewed without figures. **Fix**: Compile all TikZ sources to PDF and place them in `writing/figures/`. At minimum: `fig1_pipeline.pdf`, `fig3_auroc.pdf`, `fig4_aj_scatter.pdf`, `fig5_cross_arch.pdf`, `fig6_dec_norm.pdf`, `fig7_layer_corr.pdf`.

2. **Major** **Duplicate sentence across sections**: "Distinguishing these explanations requires feature-level analysis beyond the scope of this work" appears verbatim in Section 4.4 (line ~292) and Section 5.3 (line ~333). **Fix**: Remove from Section 4.4 (Experiments should not speculate about scope) or rephrase Section 5.3's version.

3. **Major** **Section 5.6 is underdeveloped relative to its importance**: This is the most actionable subsection but remains only 2 paragraphs. The recommendations for benchmarks ("layer-aware thresholds should replace global thresholds") lack concrete guidance. **Fix**: Expand with specific recommendations, e.g., stratification thresholds for relative layer depth and explicit sub-metrics SAEBench should report.

4. **Minor** **Table 3 footnote placement**: The footnote explaining the misleading 91.9% difference appears after the table. A scanning reader might misinterpret the percentage. **Fix**: Move the footnote marker closer to the cell or add an inline parenthetical.

5. **Minor** **Missing direct citation for pilot study**: The "3 failed probes out of 30" claim (Section 1.2, Gap 1) references "Iteration 2" but does not cite a specific file. A reviewer cannot verify this without searching. **Fix**: Add a parenthetical citation: `(see iter_002/exp/results/pilot_summary.md)`.

---

## What Works Well

1. **Honest hypothesis framing**: The paper transparently reports H2 as failed and reframes the layer-dependent pattern as an exploratory finding requiring validation. This is exactly how failed hypotheses should be handled---it builds reviewer trust rather than undermining it.

2. **Specific numbers lead every claim**: The Abstract opens with "91.2% vs 98.2%"; Contribution 5 opens with "Projection absorption differs by 7.7%"; the Conclusion summarizes four concrete results with exact statistics. This evidence-first writing is reviewer-resonant.

3. **Section 5.3's three-hypothesis structure**: The discussion of why $A_j$ correlation peaks at mid-layers is organized into three clear, competing explanations (feature hierarchies, distributed representations, semantic structure). This structure helps readers understand interpretive uncertainty and effectively sets up future work directions.

4. **The visual audit and revision tracking**: The `visual_audit.md` document comprehensively tracks what was fixed and what remains, demonstrating editorial discipline.

---

SCORE: 7
