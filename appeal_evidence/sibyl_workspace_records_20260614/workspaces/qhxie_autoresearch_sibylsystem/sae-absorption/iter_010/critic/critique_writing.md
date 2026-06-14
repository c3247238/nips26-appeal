# Writing Critique -- Iteration 10

## Overall Assessment

The paper is well-structured with a clear problem-approach-evidence-conclusion arc. The methodology section (Section 3) is comprehensive and preemptive. The probe degradation ablation (Section 4.6) is the standout section -- creative, rigorous, and directly responsive to the most obvious reviewer objection. Negative results are reported with unusual transparency. The writing review scores the paper at 7/10, which is a fair assessment.

However, the paper suffers from: (1) internal data contradictions that undermine credibility, (2) significant redundancy across sections, (3) a contribution ordering that leads with the weakest contribution, and (4) figure-text inconsistencies that would trigger immediate reviewer suspicion.

---

## Critical Writing Issue: Table 3 CI Inversion

This is the single most damaging writing problem. In 5 of 7 rows of Table 3, the 95% CI lower bound exceeds the point estimate:

| Target F1 | Absorption (%) | CI Lower | CI Upper | Problem |
|-----------|---------------|----------|----------|---------|
| 0.70 | 36.1 | 37.9 | 42.1 | CI_lower > point |
| 0.75 | 35.3 | 39.2 | 43.4 | CI_lower > point |
| 0.80 | 34.4 | 37.8 | 41.8 | CI_lower > point |
| 0.85 | 33.6 | 37.0 | 40.9 | CI_lower > point |
| 0.90 | 32.4 | 34.6 | 38.3 | CI_lower > point |
| 0.95 | 28.9 | 30.7 | 34.2 | CI_lower > point |
| 1.00 | 21.6 | 21.6 | 24.7 | OK (barely) |

A reviewer who notices this (and they will) will conclude either: (a) the authors do not understand confidence intervals, or (b) the data pipeline has undiscovered bugs. Either conclusion is devastating. This MUST be fixed before submission.

---

## Major Writing Issues

### 1. Contribution Ordering Misrepresents Strength

The abstract and introduction list contributions as:
1. Cross-domain absorption characterization (weakest -- 2/3 hierarchies explained by probe quality)
2. Universal causal mechanism (strong)
3. Decoder information entanglement (weakest with circularity)

The probe degradation ablation (R^2=0.777, methodologically novel, directly addresses the reviewer's #1 objection) is buried as a sub-finding within Contribution 1. The result debate verdict (score 6.5) recommended restructuring to lead with probe degradation. This recommendation was not implemented.

**Recommended ordering:**
1. Probe degradation as a calibration methodology for cross-domain absorption measurement (the methodological advance)
2. Universal causal mechanism via activation patching (the empirical advance)
3. Cross-domain absorption characterization with quantitative confound decomposition (the application)
4. Quadruple failure of correlational predictors (the negative boundary)

### 2. Section 7.2 Is Near-Verbatim Repetition

Section 7.2 ("Universal Competitive Exclusion with Hierarchy-Dependent Recovery") restates the activation patching results from Sections 5.1-5.2 almost word-for-word:

- Recovery rates: "32.5% for first-letter, 61.9% for city-continent, 34.2% for city-language"
- Effect sizes: "d=1.33, d=1.50, d=0.75"
- Control rates: "1.5%, 5.2%, 6.8%"
- Interpretation: "likely reflects hierarchy-dependent information distribution..."

The Discussion should add NEW analysis: what does universality imply for SAE deployment? How does it constrain theories of absorption? What experiments does it motivate? Instead, it merely restates numbers already presented in full.

### 3. Results Appear in Four Locations

The activation patching recovery rates, effect sizes, and p-values appear with full numerical detail in:
- Section 5.1 (first-letter)
- Section 5.2 (cross-domain)
- Section 7.2 (discussion)
- Section 8 (conclusion)

Three of these four are redundant. The conclusion should summarize at the sentence level, not repeat all nine statistical values.

### 4. Abstract Information Density

The abstract packs at least 15 distinct quantitative results into ~290 words:
- 4 absorption rates
- 1 Kruskal-Wallis p-value
- 2 R^2 values
- 1 p-value for regression
- 1 Spearman rho
- 3 recovery rates with effect sizes
- 4 correlational failure metrics

This is too many numbers. An abstract should convey the story with 5-7 key numbers. The rest belong in the introduction or results.

### 5. Layer Multiplier Inconsistency

- Section 4.2: "26x increase from layer 6 (1.0%) to layer 24 (27.1%)"
- Discussion 7.4: "15x higher than at earlier layers"
- Proposal: "15x variation (0.7% to 42.9%)"
- Consolidation summary L6 rates: "2-2.4%"

If L6 = 1.0%, the multiplier is 27x. If L6 = 2.4%, it is 11x. The paper contains at least three different multipliers for the same phenomenon. This is sloppy and immediately noticeable.

---

## Figure-Text Inconsistencies

1. **Figure 4**: Panel 1 says "L12, 25 words" while Panels 2-3 say "L24, 128/201 entities." Layer mismatch across panels in a comparison figure is unexplained.

2. **Figure 5**: Caption promises overlaid distributions for first-letter (N=158) and city-continent (N=1,464). The figure appears to show only city-continent (one histogram). Legend says N=1471 but text says N=1,464.

3. **Figure 6**: Shows "Hier: p=0.063" at L24 but Section 6 text says "hierarchy p=0.041." Figure and text disagree on the p-value.

4. **Table 2**: First-letter CI is [26.3, 34.7] -- consistent across the paper. But Table 3 shows F1=1.0 absorption as 21.6%, which falls outside this CI. Same hierarchy, same layer, same SAE, different number.

---

## Notation and Terminology

The writing review identified 5 notation/terminology deviations, all minor:
1. K=25 in paper vs K=26 in notation.md (paper is correct, notation table stale)
2. Activation function sigma not defined in notation.md
3. "Hedging Decomposition" heading vs glossary's "strict/loose hedging classification" distinction
4. Figure 4 layer mismatch (substantive issue, not just notation)
5. Figure 5 N mismatch (substantive issue)

---

## What Works Well

1. **Section 4.6 (Probe Degradation Ablation)** is exceptional. The three-way decomposition (city-continent explained, city-country mostly explained, city-language genuine outlier) is presented with precision and appropriate hedging. This section alone would justify the paper.

2. **Section 3 (Methodology)** is comprehensive and well-organized. Quality gates, aggregation methods, cosine similarity threshold validation, circularity caveats, and activation patching controls are all documented BEFORE results. This prevents many reviewer complaints.

3. **Table 5 (Hypothesis Verdicts)** with MIXED/REFUTED/NOT_SUPPORTED verdicts is outstanding. Rarely does a paper present an 11-hypothesis framework with 4 negative results, 1 refuted, and 1 mixed alongside 4 supported. This gives the paper unusual credibility.

4. **Hedging of overclaims** has improved substantially from earlier iterations. "Effect not detected with limited statistical power" (Section 6), "descriptive range" qualification (Section 4.1), circularity caveat (Section 5.3) -- these show scientific maturity.

5. **Negative results integration.** The "quadruple negative" framing (Section 7.1) converts four disappointments into a coherent methodological contribution. Table 5 with honest verdicts, the "causal methods succeed where correlational methods fail" narrative, and the explicit recommendation for interventional methods are effective.

---

## Specific Text Fixes

1. Section 2.1: "dominant tool" -> "widely adopted tool" (unsupported superlative)
2. Section 7.4: "Three aspects of our findings amplify this concern" -> "Three aspects are relevant" (hype word)
3. Abstract: Split the 4-rate + 1-KW-p + 2-R^2 + 1-rho sentence into 2-3 shorter sentences
4. Section 7.2: Delete numerical repetition; discuss implications only
5. Section 8: Remove per-hierarchy numerical detail; summarize at the level of "d=0.75-1.50, all p<10^-3"
6. Table 3: Fix CI inversion (critical)
7. All instances of "4.1x": Replace with "3.9x" or "2.7x (quality-gated)" depending on context
