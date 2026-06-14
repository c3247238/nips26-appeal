# Writing Critique

## Executive Summary

The paper is well-written overall -- clear structure, honest negative results, appropriate statistical reporting, and good use of tables and figures. However, there are systematic issues with the abstract and conclusion that overclaim relative to the evidence, and the strongest finding (H3) is presented without adequate attention to the width confound.

---

## Major Issues

### 1. Abstract overclaims on the 92.3% taxonomy figure

The abstract states: "A three-tier taxonomy applied to GPT-2 Small classifies 92.3% of letter features as showing some absorption (vs. the canonical 15-35%), though the Type II (partial) rate is likely inflated by heuristic parent identification."

The caveat is present but grammatically subordinate ("though..."). The 92.3% figure is what readers will remember. The paper's own evidence quality flags state this rate is "CRITICAL[ly]... likely inflated." The abstract should lead with the validated rates (3.8% Type I strict, 80.8% Chanin any-absorption) and present 92.3% as an upper bound.

### 2. "Validated quality indicator" language

The abstract uses "absorption as a validated quality indicator" and the conclusion repeats "establish absorption as a validated quality indicator (H3)." Correlational evidence cannot "validate" a metric -- it can show association. "Validated" implies interventional evidence (reduce absorption and observe quality improvement). This should be "quality-associated metric" or "quality-correlated metric."

### 3. Title misaligns with findings

"When Features Compete" implies the competitive exclusion framework is informative. The paper's main finding is that competition-based detection fails. A title like "Feature Absorption Correlates with SAE Quality But Resists Unsupervised Detection" would better reflect the actual contributions.

### 4. Downstream correlation discussion omits width stratification

Section 7.3 discusses H3 at length (3+ paragraphs) without addressing within-width stratification. The matched RAVEL comparison explicitly selects low-absorption (16k/65k) vs. high-absorption (1M) SAEs. The paper notes in passing that partial correlations control for width but never asks: does the correlation hold WITHIN the 16k tier? Within 65k? Within 1M? This is the most important robustness check and it is absent.

### 5. The introduction claims three contributions but two are negative

The three contributions listed in Section 1 are:
1. A formal test of LV competitive exclusion -- this failed
2. A three-tier taxonomy -- Type II is inflated
3. First systematic correlational evidence for absorption-downstream link -- this is real

Only #3 is a genuine positive contribution. The paper frames failed predictions as "contributions" (testing a hypothesis), which is technically correct for science but creates a mismatch between the claimed contributions and what a reader expects from a NeurIPS submission.

---

## Minor Issues

### 6. Section 2.2 includes unsupported speculation

"JumpReLU achieves state-of-the-art reconstruction fidelity on Gemma 2 9B but exhibits the highest absorption rates in SAEBench. We hypothesize this is consistent with longer training amplifying the sparsity-gradient dynamics that produce absorption."

This hypothesis is never tested and has no supporting evidence. Remove or qualify.

### 7. Table numbering and references

The inline tables (Tables 1-8) are well-formatted but the figure references in the markdown comments don't align with actual figure generation. The paper references fig_sharpness.pdf, fig_taxonomy_bar.pdf, etc. but these are in a figures/ subdirectory -- verify they exist and are correct.

### 8. Statistical reporting inconsistencies

- Section 5.2 reports the regression as "8.7% of absorption variance (R^2 = 0.087)" -- the 8.7% is R^2, not "absorption variance explained." With skewness = 5.186 and non-normal residuals, R^2 interpretation is questionable.
- Section 5.4 reports "Pearson r = -0.595 (p < 0.001)" but the JSON data shows p = 0.0 (likely a rounding artifact in scipy). Reporting p < 0.001 is correct; confirming this is not a numerical zero would be good practice.

### 9. Discussion length vs. contribution weight

Section 7.1 (Why LV Fails) gets 2 paragraphs; Section 7.3 (Absorption Predicts Quality) gets 4 paragraphs. The page allocation should reflect contribution importance: H3 deserves this space, but the LV failure analysis could be richer. Why does alpha_ij anti-predict absorption (ROC-AUC < 0.5)? This is an interesting mechanistic finding that deserves more discussion than it receives.

---

## Positive Aspects

1. **Honest negative-result reporting.** H1 (F1 = 0.128, below cosine baseline), H2 (partial R^2 = 0.0006), and the partially-supported H4 are all reported candidly. The paper does not hide failures.

2. **Pre-registered predictions with explicit falsification.** All hypotheses have success criteria, and deviations are acknowledged. H3 falsification is explicitly called out as the pre-registered prediction failing.

3. **Good use of caveats in results sections.** Section 5.3 flags Type II inflation in the results section itself, not just in limitations. Section 5.4 notes the safety probe confounds immediately.

4. **Clear table formatting.** Tables 2-8 are well-structured with appropriate precision and clear column headers.

5. **Appropriate related work framing.** Section 2 identifies specific gaps and explains how each component fills a gap. The related work is organized by axis (detection, architecture, downstream) rather than chronologically.

---

## Verdict

The paper is well-crafted for a study with mostly negative results. The writing quality is above average. The main concern is that the abstract and conclusion overstate the strength of the positive findings (H3 correlation, 92.3% taxonomy) without adequate qualification of their limitations (width confound, invalid Type II measurement). Fixing the abstract and adding within-width stratification would substantially improve the paper's credibility.
