# Writing Quality Review

## Summary

This paper presents an empirical study on Sparse Autoencoder (SAE) feature absorption, testing whether absorbed features (subsumed by child features) exhibit lower steering sensitivity than non-absorbed features. Contrary to the prevailing hypothesis that absorption degrades causal reliability, the authors find that high-absorption features are actually **more** steerable than low-absorption features (Spearman r=+0.35, p<0.001, 18% higher mean effect). The paper proposes the "entanglement hypothesis" to explain this finding and validates an Unsupervised Absorption Score (UAS) as a practical training-time metric. The writing is generally clear and the narrative flows logically from problem to evidence to interpretation, though several sections suffer from formulaic transitions and some tables would benefit from expanded context.

## Detailed Assessment

### Structural Coherence: 7/10

The paper follows a logical progression: Introduction (problem context) → Background (SAE basics and absorption phenomenon) → UAS (proposed metric) → Experiments (H3 central result first, then H2/H4/H5) → Discussion (entanglement hypothesis and implications) → Conclusion. This structure is sound and mirrors the outline closely.

**Issues**:
- The abstract mentions "absorption peaks in middle layers" (H1), but H1 is marked UNRESOLVED in Section 6.3. The abstract should either remove this claim or explicitly caveat it.
- Section 4.4 (H2 Mitigation Methods Pilot) and Section 4.5 (H5 Downstream Task Dependence) are labeled as pilots/partial results but appear alongside full-scale H3 results. The structural balance is uneven.
- Section 5 (Entanglement Hypothesis) lacks clear separation from Section 6 (Discussion). The hypothesis could be presented more crisply before the discussion of implications.

**Positive**: The abstract accurately previews the central finding, and each section flows into the next. Transitions between Background → UAS → Experiments are well-motivated.

### Notation & Terminology Consistency: 6/10

The paper generally adheres to notation.md definitions, but several deviations and ambiguities exist:

**Symbol violations**:
- In Section 3.1, the paper writes `UAS = cos_variance × 1.0 + act_freq × 0.5`, but notation.md defines `UAS(f) = α * cos_sim_variance(f) + β * freq_skewness(f)`. The paper uses `act_freq` instead of `freq_skewness`, and uses fixed coefficients (1.0, 0.5) instead of tunable α, β. This is a meaningful deviation that should be reconciled.
- The paper uses `α` for both the steering coefficient and the UAS coefficient weight (Section 3.1). These are different quantities with the same symbol.

**Terminology inconsistencies**:
- "Steering sensitivity" appears in the paper without being defined in the text. Glossary defines it, but the paper body should include the definition on first use.
- "Effect ratio (high/low) = 1.18" in Section 4.2 — but Table 4.2 header says "Effect ratio (high/low) | **1.18**". The relationship should be clarified: is it high/low = 1.18 or low/high = 1.18? The prose says "High-absorption mean effect / Low-absorption mean effect = 0.1035 / 0.0874 = 1.18", implying high/low = 1.18.

**Positive**: Most symbols (d_model, d_sae, SAE, UAS) are consistent with notation.md. The paper correctly distinguishes activation sensitivity from steering sensitivity.

### Claim-Evidence Integrity: 7/10

The paper backs most claims with specific numbers, figures, or references. The reported numbers generally match the source data in exp/results/full_h3_summary.md:

**Verified claims**:
- Spearman r=+0.35, p<0.001 ✓ (source: +0.3548, p=2.92e-04)
- High-absorption mean effect: 0.1035 ✓
- Low-absorption mean effect: 0.0874 ✓
- Effect ratio: 1.18 ✓
- UAS validation correlation: r=0.65-0.79 ✓ (Table 3 shows r=0.8147, 0.7603, 0.7875)

**Issues**:
- Section 3.1 states "UAS achieves strong correlation (r=0.65-0.79) with supervised absorption" but the table shows r=0.8147 and 0.7603. The lower bound (0.65) appears nowhere in the table. This should either be corrected to r=0.76-0.81 or the source should be clarified.
- Section 6.3 (Limitations) mentions "18%" but the abstract says "18%" too. The exact ratio is 1.18x = 18% higher, which is correctly calculated but could be more precisely stated as "18.4%" to match Section 4.2.
- The H5 AUC values in the paper (Section 4.5, Table Simple: 0.710/0.636, Causal: 0.547/0.522) are consistent with proposal.md.

**Positive**: Null control results (Section 4.3) are well-evidenced with p-values. The table format for results is consistent and scannable.

### Visual Communication: 6/10

The outline specifies 6 figures and 3 tables, but the paper contains **no embedded figures**. All visual elements are described as tables only. The figures referenced in the outline (scatter plots, bar charts, atlas) are not present in the paper.

**Issues**:
- No Figure 1 (UAS vs Steering Sensitivity scatter plot) — the central finding lacks visual support
- No Figure 4 (bar chart of High/Low absorption steering effects) — the 18% difference would benefit from visual presentation
- All tables are present (H3 results, UAS validation, downstream discriminability), but without figure captions
- Section 4.2 describes findings that would be much clearer with Figure 1 referenced
- "Figure" and "Table" are never explicitly referenced in the text before appearing

**Positive**: Tables are well-formatted and self-explanatory. Captions are present for all tables. The limitation that figures are missing is critical but tables alone are acceptable for a text-based review.

### Writing Quality: 6/10

The writing is generally clear and direct, avoiding excessive jargon. However, several issues remain:

**Banned patterns found**:
- Section 2.1: "Recent work by Anthropic (Towards Monosemanticity, 2023) demonstrated that approximately 70% of SAE features are genuinely interpretable by human evaluators." — This is close to "In recent years..." framing but more of a citation hook. Acceptable.
- Section 3.1: "We need a **practical** metric" — "practical" is slightly hollow without quantification. Acceptable with context.
- "statistically significant but categorical difference is modest (18%)" — "categorical" appears to be a typo for "categorical" (meaning discrete/group-level) or should be "marginal." This is unclear.

**Unclear sentences**:
- Section 4.3: "Feature-based directions show effects above the random baseline (high: p<0.001, low: p<0.001, two-sample t-test)." — The p-values are already stated; clarify what is being tested.
- Section 6.3: "Null controls use α=5 only while main H3 aggregates [1, 3, 5, 10, 20]. The absorption-sensitivity relationship is driven primarily by high-alpha (10, 20) conditions." — This is important methodological context that should appear in Section 4.1 (Setup), not buried in Limitations.
- Section 6.3: "H5 results are preliminary; a >8% causal delta was the intended pass criterion but only 2.51% was achieved." — "causal delta" is unclear; does this refer to the task-dependence delta (4.95%) or the causal task AUC degradation (2.51%)?

**Redundancy**:
- Section 6.2 and Section 5.3 both discuss UAS as a practical tool. This content is somewhat redundant.
- The abstract and Section 4.2 both present the same H3 results verbatim.

## Issues for the Editor

1. **[Critical] Missing Figures**: The outline specifies 6 figures, but none are present in the paper. The central finding (Figure 1: UAS vs Steering Sensitivity) is a critical gap. Add all figures from the outline or explicitly state they are deferred to supplementary materials.

2. **[Major] Abstract Mismatch with H1**: The abstract claims "absorption peaks in middle layers" (H1), but H1 is UNRESOLVED in Section 6.3. Remove this claim from the abstract or add a caveat that H1 is not confirmed.

3. **[Major] UAS Definition Inconsistency**: Section 3.1 defines UAS as `cos_variance × 1.0 + act_freq × 0.5`, but notation.md uses `α * cos_sim_variance + β * freq_skewness`. Reconcile the formula and use consistent terminology.

4. **[Major] Methodology Context Missing from Setup**: Section 4.3 mentions that null controls use α=5 only while main H3 uses [1, 3, 5, 10, 20], and that the effect is driven by high-alpha conditions. This is critical context for interpreting the 18% finding. Move this to Section 4.1 (Setup).

5. **[Minor] UAS Correlation Range Error**: Section 3.1 claims r=0.65-0.79 but Table 3 shows r=0.76-0.81. Correct the range to match the data.

6. **[Minor] "Categorical" Typo**: Section 6.3 says "statistically significant but categorical difference is modest." Replace "categorical" with "categorical" (if intentional) or "marginal" for clarity.

7. **[Minor] Redundant UAS Discussion**: Section 5.3 and Section 6.2 both emphasize UAS as a practical tool. Consolidate or differentiate the discussions.

8. **[Minor] Clarify "Causal Delta" in Limitations**: Section 6.3 mentions ">8% causal delta" and "2.51% achieved" without clarifying what delta refers to. Make this precise.

## What Works Well

1. **Clear central finding**: The paper's main result (high-absorption features are more steerable) is clearly stated in the abstract, Introduction, and Section 4.2. The narrative arc from hypothesis to reversal is compelling.

2. **Well-structured tables**: Tables 3.1 (UAS validation), 4.2 (H3 results), and 4.5 (H5 AUC) are clearly formatted, self-explanatory, and contain all necessary data.

3. **Honest limitations section**: Section 6.3 transparently acknowledges pilot-scale experiments, metric variation, null control protocol caveats, effect size, and downstream task scope. This strengthens rather than weakens the paper.

4. **Effective hypothesis framing**: The paper explicitly states H3's prediction, shows the reversal, and proposes the entanglement hypothesis as an explanation. This structure is clear and scientifically honest.

SCORE: 6
