# Critique of Writing: Quantifying Feature Absorption in Sparse Autoencoders

## Summary

The paper is generally well-written with clear methodology and strong negative result reporting. The writing avoids banned filler patterns, leads with concrete numbers, and does not soften falsified hypotheses. However, three issues undermine its credibility: (1) **five figure files are missing** — all figure references will render as broken image links; (2) **Section 5.5 contains inappropriate self-critique** — the passage "the decision to not run H2 was not data-driven" reads as peer review of the authors' own work rather than scientific reporting; (3) **Section 5.6/5.7 redundancy** adds length without new information.

---

## What Works Well

### Clear negative result reporting
The paper excels at reporting falsified hypotheses without softening. Example from Section 5.1:
> "Only 46 of 24,576 latents (0.19%) have $A_f > 0.5$... falsifying the hypothesis at this layer."

This directness is exemplary for negative results papers. Numbers are specific and not rounded away.

### Strong metric definition (Section 3)
The absorption score formula is precisely stated with clear operationalization. The five-step definition (activating tokens, top-5 co-firers, partial reconstruction, RVE, threshold) is fully reproducible.

### Appropriate random control baselines
The paper consistently reports random dictionary control results alongside real SAE results, correctly noting that "random controls register 0.00% at all sizes, correctly scaled." This is good scientific practice.

### Notation and terminology consistency
Cross-checking against `notation.md` and `glossary.md` confirms strong consistency. The absorption score formula is correctly stated. Symbols are used correctly throughout.

---

## Critical Issues

### 1. Missing Figures (Critical)

The paper references 5 figures in text and appendix but **no figure files exist** in the writing directory:

| Figure | Referenced In | File Path | Status |
|--------|--------------|-----------|--------|
| gen_pipeline.pdf | Section 4.1 | writing/figures/gen_pipeline.pdf | MISSING |
| fig1_layer_absorption.pdf | Section 5.2 | writing/figures/fig1_layer_absorption.pdf | MISSING |
| fig2_dict_size.pdf | Section 5.4 | writing/figures/fig2_dict_size.pdf | MISSING |
| fig4_layer4_histogram.pdf | Section 5.2 | writing/figures/fig4_layer4_histogram.pdf | MISSING |
| fig3_faithfulness.pdf | Section 5.3 | writing/figures/fig3_faithfulness.pdf | MISSING |

All `![Figure N...](figures/...)` LaTeX syntax will render as broken image links. This is a **critical production issue** that must be resolved before submission.

### 2. Section 5.5 Self-Undermining (Major)

The passage in Section 5.5 (lines 227-228):
> "the decision to not run H2 was not data-driven. A principled decision framework requires either running H2 at layer 4... or formally retiring H2 with explicit justification."

This reads as peer review criticism of the authors' own prior decisions, which is inappropriate in an academic manuscript. The passage characterizes the prior decision as wrong rather than reporting the limitation straightforwardly.

**Fix**: Remove the editorializing. Replace with:
> "H2 was not tested due to insufficient absorption variance at layer 8 (46 of 24,576 latents with $A_f > 0.5$). However, layer 4 has 49.3% absorption (~12,000 latents) — 260x more data than layer 8. A redesigned H2 experiment should use layer 4 as the primary test layer, where absorption variance is sufficient for correlation analysis."

### 3. Sections 5.6/5.7 Redundancy (Minor)

Table 1 in Section 5.6 already summarizes all 5 hypotheses with falsification status. Section 5.7 then repeats the layer-by-layer breakdown in Table 2. This adds length without new information.

**Fix**: Either integrate Table 2 into Section 5.6 as a second table, or move Section 5.7 to the Appendix.

---

## Other Issues

### H1 Layer Discrepancy (Major)

The paper states H1 is "falsified at layer 8 but not at layer 4" but the pre-registered falsification criterion was "<10% prevalence across layers 4-10 collectively." Section 5.1 should explicitly state that layer 8 was the primary test layer, with layer 4 treated as exploratory. The discrepancy between layers (0.19% vs 49.3%) is itself a finding consistent with the H3 inverted-U pattern, but this should be stated explicitly, not left for the reader to infer.

### H4 Design Inconsistency (Minor)

Section 5.3 explains the H4 failure as a design flaw (task-agnostic latent selection) but Section 5.5 uses "insufficient absorption variance at layer 8" as the reason H2 was skipped — yet layer 4 has 260x more absorbed latents. The justifications should be aligned.

### Figure Caption Completeness (Minor)

Figure captions describe content but do not specify file format, generation method, or data source. For reproducibility, add "Generated via matplotlib from iter_002/exp/results/[source_file]." to each caption.

---

## Abstract Assessment

The abstract accurately represents the paper's content and results. It correctly notes the layer-dependence of H1 (0.19% at layer 8 vs 49.3% at layer 4) and previews why the discrepancy is a finding rather than a contradiction. The abstract's phrasing "falsifying the hypothesis at that layer" is accurate given the layer-8-specific claim.

---

## Recommendation

The writing quality is high and the negative result reporting is exemplary. However, the **missing figures are a critical production blocking issue** and **Section 5.5's self-critique undermines scientific authority**. Address these before submission.