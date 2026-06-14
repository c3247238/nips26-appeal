# Writing Quality Review

## Summary

This paper investigates whether SAE feature absorption affects steering effectiveness in language models. The central empirical question: do absorbed features (those subsumed by child features in the hierarchy) produce different steering sensitivity than non-absorbed features? Using a matched-group design on GPT-2 Small (layer 8), the paper reports no statistically significant difference in steering sensitivity between high-absorption and low-absorption groups (aggregated p=0.299). Null controls confirm that feature-based steering significantly outperforms random directions (p<10^-12). The paper also validates the Unsupervised Absorption Score (UAS) as a practical proxy for supervised absorption metrics (r=0.65-0.79). The writing is clear and the paper is internally consistent; however, two critical issues from the supervisor review remain unaddressed: the methodology evolution behind the r=+0.35 to p=0.299 shift is unexplained, and the beta=20 finding (low-absorption > high-absorption, p=0.015) contradicts the null-result framing and does not survive Bonferroni correction.

## Detailed Assessment

### Structural Coherence: 6/10

The paper follows a logical arc: Introduction (problem) → Background (SAEs, absorption, steering) → UAS methodology → Experiments (H3 primary, H2/H5 pilots) → Discussion → Conclusion. The abstract accurately previews the content. Section headings are descriptive.

**Issues**:
- Section 5 (Discussion) is titled only "Discussion" with three subsections. A more descriptive title would aid navigation (e.g., "5. Discussion and Implications").
- Sections 4.4 (H2 Mitigation) and 4.5 (H5 Downstream) are pilot-scale evaluations but appear as peer-level sections alongside H3. This misrepresents their evidentiary status. They should be in a "Preliminary Results" subsection or moved to supplements.
- The beta=20 anomaly (Section 4.2, Table 1: low-absorption > high-absorption, p=0.015) is mentioned once but never discussed or contextualized in Discussion. This dangling finding weakens the paper's coherence.
- The paper does not have a dedicated "Related Work" subsection; Section 2 merges background and related work without distinguishing contributions.

### Notation & Terminology Consistency: 8/10

The LaTeX version uses `\beta` for steering coefficient and `\alpha`/`\beta` for UAS weights (Section 3.1) — consistent and unambiguous. Layer references (4, 8) are consistent. Spearman r notation is used correctly. "Steering sensitivity" is the dominant term throughout.

**Minor violations**:
- Section 3.1 defines `\cosvar(f)` and `\freqskew(f)` but the markdown paper (paper/paper.md) defines UAS as `UAS = cos_variance × 1.0 + act_freq × 0.5` — these are different formula instantiations. Since the canonical paper is LaTeX, the LaTeX definition governs.
- "Absorption" is used as both noun ("the absorption phenomenon") and adjective ("absorption level") consistently.
- No notation glossary file exists in iter_004/writing/; if one were present, symbol consistency could be cross-checked more rigorously.

**Correct items**:
- `d_sae = 24576` is consistent
- Spearman correlation reported as r with confidence intervals where applicable
- Statistical notation (p-values, t-statistics, AUC) is consistent

### Claim-Evidence Integrity: 7/10

**Verified claims from LaTeX** (cross-checked against tables):

| Claim | Stated Value | Table Evidence |
|-------|-------------|----------------|
| Aggregated p-value (H3) | 0.299 | Table 1 (tab:h3-results) |
| Beta=20 p-value | 0.015 | Table 1 |
| High-absorption mean at beta=20 | 2.3323 | Table 1 |
| Low-absorption mean at beta=20 | 2.4628 | Table 1 |
| High vs Random t-statistic | 7.02 | Table 2 (tab:null-controls) |
| Low vs Random t-statistic | 7.18 | Table 2 |
| UAS Layer 4 Spearman r | 0.8147 | Table (tab:uas-validation) |
| UAS Layer 8 Spearman r | 0.7603 | Table |
| Combined UAS Spearman r | 0.7875 | Table |
| Simple classification AUC (low) | 0.710 | Table 4 (tab:h5-downstream) |
| Simple classification AUC (high) | 0.636 | Table 4 |

**Issues**:
- No `exp/results/summary.md` or source data files exist to verify numbers against experimental output. Numbers are self-reported from the paper.
- The paper reports the original r=+0.35 (p<0.001) analysis as background context but does not explain WHAT methodology changed. This is a claim without full evidence explanation.
- The paper does not cite external sources for the claim "70% of SAE features are genuinely interpretable" (Section 2.1).
- References contain unprofessional placeholders: "et al.", "and others" — inappropriate for conference submission.

### Visual Communication: 4/10

**What exists**:
- 4 well-formatted LaTeX tables (UAS validation, H3 results, null controls, H5 downstream)
- SAE architecture described textually but no diagram
- LaTeX `\beta` notation and mathematical expressions are clean

**Critical gaps**:
- **Zero figures**. The finding r=+0.35 (even as abandoned analysis) would benefit from a scatter plot. The comparison across beta values (Table 1) would benefit from a bar chart. NeurIPS/ICLR submissions typically expect at least 1-2 figures.
- Both `writing/figures/` and `writing/latex/figures/` directories are empty.
- The paper has no visualization of the UAS score distribution, the feature selection methodology, or the steering effect magnitudes.
- Table captions are present and descriptive, but without figures, the paper is visually sparse.

### Writing Quality: 7/10

**Banned patterns (none found)**:
- "In recent years" — not present (good)
- "Furthermore" — not present (good)
- "It is worth noting that" — not present (good)
- "significantly" without numbers — not present (good; all uses of "significant" are qualified)
- "To the best of our knowledge" — not present (good)

**Writing strengths**:
- Sentences are direct and specific. Example: "We need a practical metric for predicting absorption without expensive supervised probes" — leads with the practical need.
- Statistical hedging is appropriate: "marginal degradation is not statistically significant (p=0.42)", "the 7.45% and 2.51% differences are not statistically significant."
- The pilot-scale caveats in Sections 4.4 and 4.5 are appropriately hedged.

**Issues**:
- Section 3.1: "In the notation table we use \alpha for the weight... in the steering equation we use \beta... The context disambiguates" — this is weak. Better to state explicitly: "We denote the UAS weight for cos_variance as \alpha and for act_freq as \beta in Equation 1."
- Section 4.2 Note: "Our initial analysis found a Spearman correlation of r=+0.35 (p<0.001) between UAS and steering sensitivity. This analysis was conducted with a different feature selection methodology." — this is vague and leaves readers confused about whether the original finding was a confound artifact.
- Section 5 (Discussion) could use more depth on the beta=20 reversal — saturation confound, implication for steering research, etc.
- The paper does not explicitly contrast its predictions with prior work's predictions.

## Issues for the Editor

1. [Critical] **Methodology Evolution Unresolved**: Section 4.2 Note mentions "different feature selection methodology" produced r=+0.35 vs p=0.299, but does not specify WHAT changed. Without this, readers cannot assess whether the original analysis was confounded. **Fix**: Add one paragraph explaining the confound: e.g., "Original analysis selected features by UAS alone; matched design additionally controlled for activation frequency and decoder L2 norm. After this control, the positive correlation disappears."

2. [Critical] **Beta=20 Framing Contradicts Null Result**: Table 1 shows p=0.015 (low-absorption > high-absorption at beta=20) — opposite direction of the hypothesis. After Bonferroni correction for 5 comparisons, p>0.01 threshold is not met. The title/abstract claim "no significant difference" is technically accurate but incomplete. **Fix**: Apply Bonferroni correction, report corrected p-values, and add a paragraph in Discussion contextualizing the saturation confound for beta=20.

3. [Major] **No Figures**: Zero visual elements beyond tables. NeurIPS/ICLR expects at least a method diagram and a results figure. **Fix**: Generate: (1) SAE architecture diagram (even a simple block diagram), (2) scatter plot of UAS vs steering sensitivity, (3) bar chart comparing high vs low absorption across beta values.

4. [Major] **Entanglement Hypothesis Has No Grounding**: The discussion mentions an "entanglement hypothesis" (absorbed features are deeply integrated) but no evidence supports it. Since the positive finding (r=+0.35) was abandoned, the hypothesis is speculation. **Fix**: Either add mechanistic experiments (activation patching) or demote to a brief 2-3 sentence post-hoc speculation paragraph about the beta=20 reversal.

5. [Major] **Pilot Sections Misrepresented as Main Results**: Sections 4.4 and 4.5 are pilot-scale but appear as peer sections alongside H3. **Fix**: Create a "Preliminary Results" subsection for H2/H5, or move to supplementary material.

6. [Minor] **References Contain Placeholders**: "et al." and "and others" are unprofessional in conference submissions. **Fix**: Replace with full author lists from original publications.

7. [Minor] **10 Test Prompts Not Listed**: Reproducibility concern — readers cannot replicate the exact prompt set. **Fix**: List prompts in appendix or supplementary.

## What Works Well

1. **Internally Consistent Finding**: Both markdown and LaTeX versions now consistently report "no significant difference" (p=0.299). The earlier critical issue of contradictory versions is resolved.

2. **Well-Formatted Tables**: Tables are professionally formatted with clear captions, proper statistical notation, and consistent column headers. The UAS validation table and null controls table are particularly effective.

3. **Honest Negative Results**: The paper appropriately reports the beta=20 finding as nominally significant (even though it contradicts the null framing) and acknowledges pilot-scale status for H2/H5. This honest reporting builds reviewer trust.

4. **Clear Empirical Question**: The central question ("Does absorption reduce steering effectiveness?") is clearly stated and the methodology (matched high/low absorption groups, null controls, full beta range) is well-described in Section 4.1.

5. **Appropriate Hedging**: Statistical language is consistently precise — "no statistically significant difference", "p > 0.05", "does not survive correction" — rather than vague claims like "slightly higher."

---

SCORE: 5/10

The paper has clear writing, internally consistent numbers, and honest reporting of null results. However, three critical issues prevent it from passing the quality gate: (1) the methodology evolution behind the r=+0.35 to p=0.299 shift is unexplained, (2) the beta=20 finding contradicts the null framing and does not survive Bonferroni correction, and (3) zero figures exist despite NeurIPS/ICLR expectations. A score of 5 reflects a paper that is readable and methodologically sound in its final form but still requires revision before external review. The score would rise to 7+ with the addition of 2 key figures, the methodology evolution explanation, and proper Bonferroni-corrected reporting.
