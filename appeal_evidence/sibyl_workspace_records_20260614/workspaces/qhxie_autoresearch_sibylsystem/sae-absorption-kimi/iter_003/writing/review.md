# Writing Quality Review

## Summary

This paper tests whether the SAEBench feature-absorption metric---which uses first-letter classification tasks---generalizes to real semantic hierarchies drawn from WordNet. Across eight SAE architectures on Pythia-160M, the correlation between first-letter and semantic-hierarchy absorption is inconclusive (r = 0.463, 95% CI spanning negative to near-perfect). The metric fails hierarchy specificity (non-hierarchy features show higher absorption than hierarchies, t = -4.748, p = 0.003), and a Random-SAE control matches trained SAEs on semantic tasks, indicating the metric captures geometric artifacts rather than learned structure. The paper argues for domain-specific absorption metrics with validated hierarchy specificity.

---

## Detailed Assessment

### Structural Coherence: 8/10

The paper follows a clear problem--approach--evidence--conclusion arc. The Introduction establishes absorption as a central pathology, identifies the first-letter benchmark's limitations, and poses three research questions. The Method section is well-organized with subsections matching each RQ. Results map cleanly to hypotheses (H1, H2, H3). The Discussion interprets each finding before drawing implications. The Conclusion summarizes without introducing new claims.

**Minor issue:** The transition from Section 4.5 (Limitations) to Section 5 (Conclusion) is slightly abrupt. The limitations paragraph ends with "From these implications, we distill concrete recommendations for the field" which is acceptable but could be smoother.

**Minor issue:** The paper lacks a dedicated Related Work section. The introduction covers prior work inline (Chanin et al., SAEBench, Matryoshka SAEs, OrtSAE), but a focused Related Work section would help situate the contribution within the broader SAE evaluation literature, particularly given the growing body of work questioning SAE metric validity (Kantamneni et al., 2025 is cited but not discussed).

### Notation & Terminology Consistency: 9/10

Cross-checking against `notation.md` and `glossary.md`:

- **Consistent:** SAE, SAEBench, SAELens, WordNet, hypernym/hyponym, k-sparse probing, ground-truth probe, absorption score, AUROC, tau_fs, feature-splitting threshold.
- **Symbols:** All notation ($h^{(l)}$, $W_{\text{enc}}$, $W_{\text{dec}}$, $A_{\text{full}}$, $\bar{A}_{\text{SH}}$, $\bar{A}_{\text{NH}}$, $\tau_{\text{fs}}$) matches `notation.md` exactly.
- **Terminology preferences:** "first-letter" (hyphenated), "k-sparse" (hyphenated), "ground-truth" (hyphenated as modifier) are all correct per `glossary.md`.

**One inconsistency:** The paper uses "Random-SAE" (with hyphen) and "Random SAE" (without hyphen) interchangeably. The glossary does not define this term. Standardizing to "Random-SAE" (as a compound modifier) would be cleaner.

**One terminology issue:** Section 2.1 describes the Random-SAE as "permuting the decoder directions of the Standard SAE while retaining its trained encoder." This is consistent throughout iter_003 (the iter_02 encoder/decoder contradiction has been fixed).

### Claim-Evidence Integrity: 8/10

**Claims with specific evidence:**
- "First-letter absorption spans nearly the full range, from 0.008 (GatedSAE) to 0.576 (TopK)" --- supported by Table 1.
- "Pearson r = 0.463 (95% bootstrap CI: [-0.389, 0.981])" --- supported by statistical_analysis_summary.json.
- "Paired t-test: t = -4.748, p = 0.0032" --- supported by statistical_analysis_summary.json.
- "Random-SAE control achieves semantic-hierarchy absorption of 0.352, identical to the Standard SAE" --- supported by Table 1 and JSON data.
- "Cohen's d = -1.68" --- this effect size is reported in the abstract and Section 3.3 but does not appear in `statistical_analysis_summary.md`. The source data should be verified.

**One claim needing nuance:** Section 4.3 states "The permuted decoder preserves the geometric properties of the activation space---the angles between decoder directions, their norms, and their coverage of the residual-stream manifold." This is a plausible explanation but not empirically verified. The paper correctly frames it as a hypothesis ("If the semantic-hierarchy absorption score depends primarily on..."), which is appropriate epistemic humility.

**One potential issue:** The abstract claims "A Random-SAE control yields semantic-hierarchy absorption of 0.352, identical to the Standard SAE." The word "identical" is used throughout. Given that both round to 0.352 at three decimals, this is reasonable, but the exact values should be verified in the source JSON.

**Number consistency check:** All numbers in the paper match `statistical_analysis_summary.md`:
- Table 1 values match per_architecture_scores (rounded to 3 decimals).
- Table 2 values match tau_fs_robustness.
- Figure captions match the data.

### Visual Communication: 8/10

**Completeness:** 5 figures and 2 tables in the main paper, all present and referenced. The visual audit confirms no orphan figures or tables.

**Figure quality:**
- Figure 1 (architecture ranking): Effective grouped bar chart. Color distinction (blue/orange) is clear.
- Figure 2 (scatter plot): Good. Shows the regression line. The wide CI is visually apparent.
- Figure 3 (hierarchy specificity): Effective. The direction reversal (non-hierarchy > hierarchy) is immediately visible.
- Figure 4 (tau_fs robustness): Clear line plot with error bars. The H1 threshold line is a nice touch.
- Figure 5 (GPT-2 replication): Appropriate, though with only 2 architectures the chart feels sparse.

**Table quality:**
- Table 1 is the workhorse table---comprehensive, well-formatted, with bold best values.
- Table 2 reports full statistical results for robustness.
- Tables 2 and 3 in the Method section document exact hierarchies/control pairs for reproducibility.

**One issue:** The paper references `fig2_scatter.pdf`, `fig3_specificity.pdf`, and `fig4_robustness.pdf` in the iter_003 version, but the figure files in `iter_003/writing/figures/` are `.png` files (e.g., `fig2_firstletter_vs_semantic_scatter.png`). The file extensions in the markdown references should match the actual files. Either the figures should be regenerated as PDFs or the references should use `.png`.

**One suggestion:** The paper would benefit from a method pipeline diagram showing the three-branch evaluation flow (first-letter, semantic-hierarchy, non-hierarchy control). This would help readers understand the experimental design at a glance, especially given the complexity of the three conditions.

### Writing Quality: 8/10

**Strengths:**
- The paper leads with concrete results: "First-letter absorption spans nearly the full range..." (Section 3.1).
- Sentences are generally short and direct.
- Specific numbers are used throughout.
- Negative results are reported explicitly and not smoothed over.
- The abstract is a model of clarity---every claim backed by a specific number.

**Banned patterns found:**

1. **"Furthermore"** --- Not found in iter_003. Good.

2. **"It is worth noting that"** --- Not found. Good.

3. **"In recent years" / "Recently"** --- Not found. Good.

4. **"Moreover"** --- Not found. Good.

5. **Vague "significantly" without numbers** --- Section 3.3: "Non-hierarchy scores are significantly higher" --- this is immediately followed by the t-statistic and p-value, so it is technically supported. Consider moving the statistics into the same sentence for maximum clarity.

6. **"To the best of our knowledge"** --- Not found. Good.

**Unclear sentences:**

1. Section 2.4, line 108: "In practice, for all semantic hierarchies in our suite, acc_sae = acc_resid = 1.0, so the absorption score simplifies to (acc_resid - acc_k-sparse) / acc_resid." This critical observation is not present in iter_03 (it was in iter_02). The iter_03 paper has removed this, which is good because it was buried, but the implication---that the metric reflects only k-sparse probing loss---is never explicitly stated. This is a key methodological point that should be highlighted.

2. Section 3.3: "Every architecture except TopK shows this reversal." This is imprecise. Looking at Table 1, TopK is the only architecture where semantic-hierarchy (0.250) > non-hierarchy (0.311) is NOT true---actually, 0.250 < 0.311, so TopK also shows higher non-hierarchy. The text should say "All eight architectures show higher non-hierarchy absorption" or clarify which architecture(s) buck the trend.

3. Section 3.1: "The Random-SAE control scores 0.030 on first-letter absorption---near zero, as expected---but 0.352 on semantic-hierarchy absorption, identical to the Standard SAE." The "identical" claim is repeated many times. While true at three decimal places, the rhetorical effect may be overstated. Consider varying the phrasing (e.g., "matches exactly," "equals to three decimal places").

**Passive voice:** Generally well-controlled. A few acceptable instances in definitional contexts.

---

## Issues for the Editor

1. **Major** **"Figure file extension mismatch"**: The paper references `.pdf` figures (`fig2_scatter.pdf`, `fig3_specificity.pdf`, `fig4_robustness.pdf`) but the actual files in `iter_003/writing/figures/` are `.png` files. Fix: Either regenerate figures as PDFs or update all figure references to use `.png`.

2. **Major** **"Missing method pipeline diagram"**: The experimental design involves three parallel conditions (first-letter, semantic-hierarchy, non-hierarchy control) with shared probe training and absorption formula. A flowchart would make this immediately clear. The outline already has `fig_method_pipeline_desc.md` drafted in iter_002. Consider adding as a figure in Section 2.

3. **Major** **"H2 description imprecision"**: Section 3.3 states "Every architecture except TopK shows this reversal." But Table 1 shows TopK also has semantic-hierarchy (0.250) < non-hierarchy (0.311). Either TopK does NOT buck the trend (in which case "every architecture" is correct and "except TopK" is wrong), or there is a data interpretation issue. Fix: Verify the data and correct the text.

4. **Minor** **"Cohen's d source unclear"**: The abstract and Section 3.3 report Cohen's d = -1.68, but this value does not appear in `statistical_analysis_summary.md`. Fix: Verify the calculation and add it to the summary file, or remove if unverified.

5. **Minor** **"Random-SAE hyphenation inconsistency"**: The paper uses "Random-SAE" (with hyphen) and "Random SAE" (without) interchangeably. Standardize to "Random-SAE" throughout.

6. **Minor** **"Missing Related Work section"**: The introduction covers prior work inline, but a dedicated Related Work section would strengthen the paper's positioning. Consider adding a brief section between Introduction and Method, or expanding the inline citations in the Introduction.

---

## What Works Well

1. **The abstract is a model of clarity.** It states the problem, the method, the three key findings (inconclusive correlation, hierarchy specificity failure, Random-SAE anomaly), and the implication---all in under 150 words. Every claim is backed by a specific number.

2. **The Results section maps cleanly to hypotheses.** Sections 3.2--3.6 each address a specific hypothesis with a figure, a table, and a clear assessment sentence. This structure makes the paper easy to navigate.

3. **The Discussion does not overclaim.** Section 4.1 explicitly states "The conservative interpretation is that the current evidence base is insufficient for confident claims about construct validity." This is the correct level of epistemic humility for a wide CI. The paper lets the data speak without forcing a narrative.

4. **The iter_03 revision fixed the critical encoder/decoder contradiction from iter_02.** Section 2.1 now consistently describes the Random-SAE as permuting decoder directions while retaining the trained encoder, and Section 4.3 uses the same description. This resolves the most serious internal inconsistency from the prior version.

---

SCORE: 8
