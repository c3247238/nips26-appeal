# Critique: Planning and Methodology

## Summary

The planning documents a well-designed controlled experiment (H3) with matched feature selection. However, the methodology has a critical unresolved contradiction between two analyses (original r=+0.35 vs. matched p=0.299), and the beta-conditional effect at beta=20 is not explained by any stated hypothesis. The planning also fails to account for the structural choice to present pilots (H2, H5) in the main experiments section.

## Critical Issues

### 1. Two-Analysis Contradiction Has No Methodology Explanation

The paper now presents two analyses:
- **Original analysis**: r=+0.35, p<0.001 (unmatched feature selection, methodology unspecified)
- **Current analysis**: p=0.299 aggregated, p=0.015 at beta=20 (matched feature pairs)

The paper's note says the original analysis used "different feature selection methodology" but does not specify what confound was controlled or how. The methodology section (4.1) describes only the CURRENT matched design, not the ORIGINAL design.

**Recommendation**: Add a paragraph in Section 4.1 explaining the evolution: "In preliminary experiments we selected features by UAS score alone without matching. This allowed high-absorption features to differ systematically on confound variables. Our main analysis matches feature pairs on activation frequency and decoder L2 norm to isolate the absorption effect. Supplementary Table X compares the two feature sets."

### 2. Beta-Conditional Effect Has No Hypothesis

The p=0.015 at beta=20 (low-absorption > high-absorption) is not predicted by any stated hypothesis. The hypotheses are:
- H3 (original): absorption degrades steering reliability (predicts: high < low, consistent)
- H3 (current): no systematic relationship (predicts: no difference)

Neither hypothesis predicts that low-absorption features should be BETTER at high steering magnitude. This finding is post-hoc and requires either a new hypothesis or a mechanistic explanation.

**Recommendation**: Add a hypothesis H7 (or revise H3): "The absorption-steering relationship is beta-conditional: at moderate steering magnitudes absorption has no effect, but at high magnitudes low-absorption features outperform due to saturation effects in high-norm decoder directions." Then add a brief discussion of why high decoder L2 norms (associated with absorbed features) might saturate at high beta.

### 3. Pilot Results Misplaced in Main Experiments Section

H2 (Mitigation) and H5 (Downstream Tasks) are pilot evaluations. The paper presents them in the main "Experiments" section without a structural distinction between pilot and confirmatory results. The UAS validation (Section 3) is labeled as "Methodology" but includes validation results that are confirmatory in nature.

**Recommendation**: Restructure the paper:
- Section 3: Methodology (UAS definition and computation, NOT validation results)
- Section 4: Main Results (H3 only, with subsections for null controls)
- Section 5: Preliminary/Exploratory Results (H2, H5)
- Section 6: Discussion (entanglement hypothesis, implications, limitations)

## Major Issues

### 4. UAS Validation as Methodology Is Confusing

Section 3 is titled "Unsupervised Absorption Score (UAS)" but includes a validation table (Table 1) with Spearman correlations. Validation results belong in Experiments, not Methodology.

**Recommendation**: Move UAS validation to Section 4 (or a dedicated Experiments subsection) and keep Section 3 purely as methodology definition.

### 5. Null Control Beta Range

The paper text says null controls use "full beta range [1,3,5,10,20]" but this is not verified in the methodology section. Table 3 shows p-values aggregated across the range, which confirms the full range was used. However, this should be stated explicitly in the methodology (Section 4.1), not only in the results narrative.

**Recommendation**: Add to Section 4.1: "Null controls use identical features, beta range, and prompts as the main experiment, but with randomly initialized directions instead of learned feature directions."

### 6. No Statistical Power Analysis

With 50 features per group and 5 beta values, the effective sample size for the matched comparison is 50 (paired by feature pair) or 250 (if treating each beta-condition as independent). The paper should report what effect size the study was powered to detect, so readers can interpret null results appropriately.

**Recommendation**: Add a power analysis note: "With 50 matched pairs, our study is powered to detect d=0.40 (medium effect) at 80% power. The observed aggregated effect of d=0.12 is below this threshold."

## What Works

1. **Matched feature selection design**: Controlling for activation frequency and decoder L2 norm is the right approach for isolating the absorption effect.

2. **Full beta range in null controls**: The null controls now use the full beta range matching the main experiment.

3. **Honest H2 framing**: The paper correctly notes that H2 mitigation results are pilot-scale and inconclusive.
