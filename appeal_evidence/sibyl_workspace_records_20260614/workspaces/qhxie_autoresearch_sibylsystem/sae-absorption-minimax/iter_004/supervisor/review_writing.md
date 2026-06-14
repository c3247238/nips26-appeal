# Supervisor Review

**Iteration**: Current (post-iteration 7+)
**Paper**: Absorption and Steering Sensitivity in Sparse Autoencoders: No Significant Difference Between High and Low Absorption Features
**Reviewer**: Supervisor Agent (sibyl-supervisor)
**Date**: 2026-04-29

---

## Overall Assessment

**Score: 6.5 -- Borderline Accept (Top 30%)**

**Verdict: Revise**

The paper has made a major empirical pivot from claiming "high-absorption features are more steerable" (r=+0.35) to claiming "no significant difference" (p=0.299). This is a defensible and honest shift. Two of three prior critical issues are resolved: causal language has been removed and null controls now use the full beta range. However, a critical new issue emerges: the paper reports two contradictory empirical results (r=+0.35 and p=0.299) without reconciling them. The beta=20 anomaly (low-absorption significantly MORE steerable, p=0.015) contradicts the overall null result framing. The entanglement hypothesis is now doubly orphaned — it was designed to explain a positive finding that has been abandoned. Score 6.5 reflects genuine improvement from honest empirical reporting, but the paper requires revision to address the new critical issues before external review.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Novelty & Significance** | 7 | The finding (absorption does not predict steering effectiveness) is genuinely useful and counter-intuitive. The beta-conditional reversal at high steering magnitude is the paper's most novel result. UAS as a practical unsupervised metric adds value. |
| **Technical Soundness** | 6 | Matched feature selection design is methodologically sound. However: the methodology evolution is unexplained, Bonferroni correction not applied to beta-stratified comparisons, entanglement hypothesis now has no supporting evidence, and saturation confound for beta=20 not discussed. |
| **Experimental Rigor** | 6 | Null controls now use full beta range — good. Honest negative results reporting remains exemplary. But: beta=20 finding loses significance after Bonferroni correction, no figures exist, H2/H5 pilots mislocated, scope remains GPT-2 Small only. |
| **Reproducibility** | 6 | Model and SAE publicly available. But: feature selection not fully documented, 10 test prompts not listed, UAS hyperparameters not validated, no code repository reference, no figures. |

---

## Critical Issues (Must Fix)

### 1. Two Contradictory Empirical Results Without Reconciliation

**Severity: Critical**

The paper acknowledges two analyses with contradictory results: (1) original analysis: r=+0.35, p<0.001, and (2) controlled matched design: p=0.299. The note in Section 4.2 says "different methodology" without explaining WHAT changed and WHY the result changed. The critical question is unanswered: did the original methodology introduce confounds (e.g., high-absorption features also having higher activation frequency or decoder L2 norm) that produced a spurious positive correlation? If so, the original r=+0.35 is a confound artifact. The paper leaves readers to guess.

**Action required**: Add a dedicated paragraph explaining the methodology evolution: what confound was controlled in the matched design? E.g., "Original analysis selected by UAS alone; matched design additionally controlled for activation frequency and decoder L2 norm. After this control, the positive correlation disappears, suggesting the original r=+0.35 reflected confounds rather than a genuine absorption-steering relationship."

### 2. Beta=20 Finding Contradicts Null Result Framing

**Severity: Critical**

Table 1 shows p=0.015 at beta=20 (low-absorption features MORE steerable). This is a significant finding in the OPPOSITE direction of the hypothesis. The title and abstract claim "no significant difference" but the data shows a conditional effect. Additionally, with 5 beta comparisons, Bonferroni-corrected threshold is p<0.01; the beta=20 finding at p=0.015 is NOT significant after correction.

**Action required**: Report Bonferroni-corrected p-values. Update title/abstract to acknowledge the beta-conditional nature. E.g., "Absorption Does Not Predict Steering Sensitivity at Most Steering Magnitudes."

### 3. Entanglement Hypothesis Now Doubly Orphaned

**Severity: Critical**

The entanglement hypothesis was designed to explain why high-absorption features would be MORE steerable (r=+0.35 finding). Since that finding has been abandoned, the hypothesis now has zero empirical grounding. A hypothesis without supporting evidence for the main effect it was meant to explain is pure speculation.

**Action required**: Either add mechanistic experiments (activation patching, path tracing) or demote to a brief 2-3 sentence paragraph repositioned as post-hoc speculation about the beta=20 reversal (saturation confound).

---

## Major Issues

### 4. Saturation Confound Not Discussed for Beta=20

The beta=20 reversal may reflect saturation: high-absorption features have higher decoder L2 norms by construction, so they saturate faster at high steering magnitudes. The paper does not discuss this natural explanation.

### 5. Duplicate Paper Versions Persist

Two contradictory versions: paper/paper.md and writing/latex/main.tex. LaTeX version should be designated canonical. Markdown version should be deleted.

### 6. No Figures

Zero figures beyond LaTeX tables. Figure directories are empty. Essential for NeurIPS/ICLR submission.

### 7. H2/H5 Pilot Mislocation

Sections 4.4 and 4.5 appear as peer sections without structural distinction from H3. Add a "Preliminary Results" subsection.

### 8. References Contain Placeholders

"et al.", "others" unprofessional for conference submission.

---

## Minor Issues

- 10 test prompts not listed (reproducibility concern)
- UAS hyperparameters not documented
- Prior work predictions not explicitly contrasted
- UAS not compared against trivial baselines

---

## What Works Well

1. **Major empirical pivot**: Changing the central claim from r=+0.35 to "no significant difference" is honest and defensible.
2. **Null controls fixed**: Full beta range [1,3,5,10,20] now used — critical methodological flaw from prior reviews is resolved.
3. **Causal language removed**: "More Causal" and "are More Causal" have been replaced with neutral framing.
4. **Honest negative results**: H2, H4, H6, H7 falsification reporting remains exemplary across all iterations.
5. **UAS validation**: N=314 meta-analysis with r=0.65-0.79 remains the paper's strongest empirical contribution.
6. **Beta=conditional effect is interesting**: The reversal at high steering magnitude is the paper's most novel finding, worth highlighting properly.

---

## Path to Raising Score to 7.5+

The five changes that would most improve this paper:

1. **Explain the methodology evolution**: Add one paragraph explaining what confound the matched design controlled that eliminated the original r=+0.35. This is the single most important clarification.
2. **Fix the beta=20 framing**: Apply Bonferroni correction, acknowledge saturation confound, update title/abstract to reflect the conditional nature.
3. **Remove or add mechanism for entanglement hypothesis**: Either demote to brief speculation or add activation patching experiments.
4. **Generate 2 key figures**: Scatter plot of UAS vs steering sensitivity, bar chart by beta value.
5. **Delete duplicate paper/paper.md**: Designate LaTeX as canonical version.

These changes address all critical issues and would likely raise the score to 7.5 (Borderline Accept / Weak Accept).
