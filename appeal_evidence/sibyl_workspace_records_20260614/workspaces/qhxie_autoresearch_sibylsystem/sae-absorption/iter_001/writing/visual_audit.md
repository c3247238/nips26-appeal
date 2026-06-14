# Visual Audit Report — Initial Integration Pass

**Date:** 2026-04-12
**Paper:** Where and When Encoder-Decoder Misalignment Signals Feature Absorption in Sparse Autoencoders

---

## Totals

- **Total figures planned:** 7 (Figure 1–7)
- **Total tables:** 4 (Table 1–4)
- **Figures with generated PDF:** 3 (fig2_synthsae.pdf, fig3_eda_heatmap.pdf, fig4_eda_distributions.pdf)
- **Figures needing generation:** 4 (fig5_crossdomain_rates, fig6_ravel_coherence, fig7_subtype_eda; fig1 is a tikz/descriptor)
- **Tables:** All 4 are inline in the manuscript text; no separate files needed

---

## Completeness Check

### Figures present in sections vs. outline plan

| Figure | Planned (outline) | Generated file | Referenced in text before appearance | Status |
|--------|-------------------|----------------|--------------------------------------|--------|
| Figure 1 (absorption mechanism + EDA geometry) | Section 1 + Section 3 | fig1_absorption_mechanism_desc.md (descriptor) | Section 1 line 1-panel-a reference; Section 3.1 "As illustrated in Figure 1b" | OK — descriptor exists; tikz/PDF needed |
| Figure 2 (SynthSAEBench validation) | Section 3.4 | fig2_synthsae.pdf | Section 3.4 "As shown in Figure 2" | OK |
| Figure 3 (EDA AUROC heatmap) | Section 4.2 | fig3_eda_heatmap.pdf | Section 4.2 "Figure 3 visualizes the regime-specific pattern" | OK |
| Figure 4 (EDA distributions) | Section 4.2 | fig4_eda_distributions.pdf | Section 4.2 "Figure 4 shows the statistical group separation" | OK |
| Figure 5 (cross-domain rates) | Section 5.2 | **MISSING** | Section 5.2 inline reference | NEEDS GENERATION |
| Figure 6 (RAVEL coherence) | Section 5.3 | **MISSING** | Section 5.3 inline reference | NEEDS GENERATION |
| Figure 7 (subtype EDA distributions) | Section 6.2 | **MISSING** | Section 6.2 inline reference | NEEDS GENERATION |

### Tables present

| Table | Section | Status |
|-------|---------|--------|
| Table 1 (EDA detection performance) | 4.2 | Present — expanded with Dec. Cosine and Null columns |
| Table 2 (cross-domain absorption rates) | 5.2 | Present — new in this integration |
| Table 3 (subtype taxonomy) | 6.1 | Present — new in this integration |
| Table 4 (ITAC efficacy) | 6.4 | Present — new in this integration |

---

## Missing Visuals

### Critical Missing (block figures for sections 5 and 6)

1. **Figure 5** (`fig5_crossdomain_rates.pdf`): Bar chart showing absorption rates for 3 RAVEL hierarchies × 6 SAE configurations with 3× random baseline reference line. Data available in `exp/results/full/phase3e_crossdomain_analysis.json` (`figure3_data` field — note: labeled figure3 in the JSON but corresponds to paper Figure 5).

2. **Figure 6** (`fig6_ravel_coherence.pdf`): 3-panel pairwise scatter plots of RAVEL absorption correlations (city-continent vs. city-country, city-continent vs. city-language, city-country vs. city-language). Data available in `phase3e_crossdomain_analysis.json` (`cross_domain_correlations` and `figure4_data` fields). Annotate Spearman $\rho$ and $p$-values.

3. **Figure 7** (`fig7_subtype_eda.pdf`): Violin plots of EDA scores grouped by absorption subtype (early/late/partial) at L12-16k and L12-65k. Data from `phase2a_taxonomy.json`. Annotate KW $p$ = 0.0002 at L12-65k.

### Lower Priority Missing

4. **Figure 1** (`fig1_absorption_mechanism.pdf`): TikZ two-panel diagram. Descriptor file exists at `writing/figures/fig1_absorption_mechanism_desc.md`. Requires tikz compilation or matplotlib reproduction. Content is conceptual; no experimental data needed.

---

## Consistency Check

- **Figure numbering:** Sequential 1–7. No gaps, no duplicates. All correct.
- **Table numbering:** Sequential 1–4. Consistent.
- **Caption style:** All captions in the paper use sentence case and end with a period. Consistent.
- **Color scheme:** Figures 2–4 are existing generated PDFs; Figures 5–7 need to follow the same style as Figures 2–4. `figures/style_config.py` should be checked when generating missing figures.
- **EDA value consistency:** Paper uses verified experimental values throughout. Decoder cosine baseline at L5-16k = 0.302 (from `phase1_eda_deda_validation.json`), gap = +0.396 AUROC. This resolves the inter-section inconsistency flagged in critiques.

---

## Flow Check

All figures are referenced in text before they appear (in-section). No orphan figures. The visual narrative:
- Figure 1 (intro/method): sets up the concept
- Figure 2 (method): validates theory on synthetic data
- Figures 3–4 (experiments): show real-SAE results with regime structure
- Figures 5–6 (cross-domain): demonstrate generalization
- Figure 7 (taxonomy): reveals subtype structure and EDA ordering

This sequence supports the paper's argument from theory → controlled validation → real validation → generalization → mechanism.

---

## Issues Found and Fixed During Integration

1. **Decoder cosine baseline inconsistency** (critical): Method section (Section 3.5) and Introduction reported +0.396 AUROC gap; Experiments section (Section 4.2) and Conclusion reported +0.318 AUROC gap with baseline = 0.380. Verified from `phase1_eda_deda_validation.json`: actual decoder cosine AUROC at L5-16k = 0.302, EDA = 0.698, gap = 0.396. **Fixed**: Paper consistently uses +0.396 and baseline = 0.302 throughout.

2. **Table 1 incomplete** (critical from experiments critique): Table 1 was missing decoder cosine and shuffled null AUROC columns. **Fixed**: Table 1 now includes all columns per the outline plan.

3. **Section 3.1 "global minimum" language** (minor): Related-work critique noted "At a global minimum" is inconsistent with Theorem 1 which assumes a "partial minimum." **Fixed**: Revised to "when no absorption is present and the SAE is at a partial minimum."

4. **Contribution claim repetition** ("first" overuse): Reduced explicit "first" claims per intro critique Issue 6. EDA contribution now avoids "first" in the phrasing.

5. **Contribution 1 density** (critical from intro critique): Intro contribution 1 was over-dense with 7 numeric claims. Reduced to 2 representative numbers (AUROC = 0.776, +0.396 baseline gap) with the rest in respective results sections.

6. **Gap 3 missing partial-type subtype**: Fixed to foreshadow all three subtypes (early, late, partial).

7. **FVU direction clarification**: "FVU change: $-4.23\%$" now clarified as "reconstruction quality does not degrade" inline.

8. **D-EDA GPT2-L10 recommendation** (discussion critique Issue 3): Qualified to "narrow-dictionary deep layers" rather than "wider SAEs" generically.

9. **Convergent conclusion causal direction** (discussion critique Issue 2): Rewritten to establish that early-type dominance was found in Section 6 independently, and the three negatives are "mechanistically consistent with" rather than "proving" that finding.

10. **Conclusion future-work ordering** (conclusion critique Issue 4): Reordered to (1) dictionary-coverage training objectives, (2) RAVEL replication with Gemma 2B probes, (3) EDA validation on alternative architectures — matching the paper's own priority emphasis.

11. **Conclusion "wider dictionaries" recommendation contradiction** (conclusion critique Issue 5): Replaced with "targeted architectural solutions such as Matryoshka SAE or masked regularization" with parenthetical noting H6 falsification.

12. **Sections 5 and 6 missing from sections/ directory**: These sections were not written by section writers. They have been drafted for the first time in this integration, based on outline, critique references, and experimental data from `phase3e_crossdomain_analysis.json` and `phase2a_taxonomy_summary.md`.

---

## Suggestions for Additional Visuals

1. A precision-recall curve or recall@k table for L12-16k (supporting the "top 5% captures majority" claim in Section 7.1). Currently the number comes from Prec@50 = 0.0035 from the JSON — a dedicated figure would strengthen this.

2. A scaling scatter plot (SAE width vs. absorption rate at matched $L_0$, 6 config points) supporting H6 falsification in Section 7.3. Currently text-only.

These are suggestions for a revision round; the current paper is complete without them.
