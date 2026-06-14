# Planning Critique: Feature Absorption as Optimal Compression

## Overview

The methodology document describes a well-structured experimental plan with clear phases, baselines, and evaluation metrics. However, the planning reveals several issues: the plan was revised post-hoc after H6 falsification, remaining experiments (H9, H10) have fundamental methodological flaws, and the experimental design does not include a causal intervention that would validate the central mechanistic claim.

## Critical Planning Issues

### 1. No Causal Intervention to Validate Competitive Suppression

The central claim of the paper is that competitive suppression (via decoder correlations) causes absorption. But the methodology includes no experiment that manipulates the putative causal variable (decoder correlations) and measures the effect on absorption.

**What is missing:** A causal intervention such as:
- Orthogonalizing a subset of decoder directions and measuring whether absorption rates change
- Modifying the inhibition graph (e.g., zeroing out specific G_ij entries) and measuring the effect on parent firing
- Training an SAE with orthogonality constraints and comparing absorption rates to the baseline

**Why this matters:** Without a causal intervention, the competitive suppression mechanism is a theoretical hypothesis, not an empirically validated mechanism. The precision-recall asymmetry is consistent with competitive suppression but also consistent with multiple alternative explanations.

**Recommendation:** Add a causal intervention experiment to the plan. If this is not feasible, reframe the paper to acknowledge that the mechanism is theoretical.

### 2. H9 Operationalization is Flawed by Design

The H9 plan measures p_11 as "fraction of child prompts where parent also fires" and tests correlation with absorption_rate. But by construction:
- If parent fires on a child prompt: NOT counted as absorption
- If parent does not fire on a child prompt: counted as absorption
- Therefore: p_11 + absorption_rate = 1.0 (perfect negative correlation)

This tautology should have been caught in the planning phase. The plan should have specified an independent co-occurrence measure (e.g., from a held-out corpus, not the same prompts used for absorption detection).

**Recommendation:** Revise H9 to use an independent co-occurrence measure. If this is not feasible, drop H9.

### 3. H10 Random SAE Design Does Not Test the Right Hypothesis

The H10 plan hypothesizes that "random SAE baselines exhibit absorption-like patterns, confirming absorption is partially structural." But the actual result (random > trained) is the opposite of the prediction. The plan should have been prepared for either outcome:
- If random < trained: absorption is a learned pathology
- If random > trained: the Chanin metric is not specific to learned structure
- If random ≈ trained: the metric is insensitive to training

The plan only considered the first outcome and had no protocol for interpreting the others.

**Recommendation:** The plan should specify interpretation protocols for all possible outcomes, not just the predicted one.

### 4. Sample Size Not Justified by Power Analysis

The plan specifies n=26 features and 100 samples per feature but does not include a power analysis. With n=26, the study has limited power to detect small-to-medium effects.

**What is missing:**
- Minimum detectable effect size given n=26 and alpha=0.00417 (Bonferroni)
- Power to detect r=0.3, r=0.5, r=0.7
- Justification for n=26 (why 26 first-letter features rather than more?)

**Recommendation:** Add a power analysis to the methodology. If power is low, acknowledge this as a limitation and avoid strong negative conclusions.

### 5. Multiple Comparison Correction Applied Post-Hoc

The plan mentions multiple comparison correction (Bonferroni, BH-FDR) but the correction was applied after seeing the results. The plan does not pre-register the correction method or the primary vs. secondary hypotheses.

**What is missing:**
- Pre-registered primary hypotheses (H1, H2) vs. secondary (H4, H5) vs. exploratory (H9, H10)
- Pre-specified correction method for each hypothesis family
- Pre-specified analysis plan (e.g., "We will test H1 at L4 and L8; if both are non-significant, we will not test H1b")

**Recommendation:** For future work, pre-register the analysis plan. For the current paper, report the post-hoc nature of the correction transparently.

### 6. No Protocol for Handling Inconsistent Cross-Layer Results

The plan tests hypotheses at both L4 and L8 but does not specify how to interpret inconsistent results. H1 shows opposite signs (+0.008 at L4, -0.301 at L8). H1b shows opposite signs (+0.245 at L4, -0.431 at L8).

**What is missing:**
- A protocol for combining results across layers (meta-analysis, random effects model)
- Criteria for determining whether cross-layer inconsistency falsifies a hypothesis

**Recommendation:** Add a cross-layer combination protocol. The current approach (reporting both layers separately) is acceptable but could be strengthened with a formal meta-analysis.

### 7. Missing Baseline: Random Feature Steering Without SAE

The plan includes a random steering baseline (steering random latents) but does not include a baseline that steers without any SAE (e.g., adding random vectors to the residual stream). This would help distinguish SAE-specific effects from generic steering effects.

**Recommendation:** Add a non-SAE steering baseline for future work.

### 8. The "Optimal Compression" Framing Lacks an Experimental Test

The reframed H7 (rate-distortion optimal) and H8 (information redistribution) are presented as hypotheses but the methodology does not specify how to test them:
- H7: How to measure "rate" (sparsity loss) and "distortion" (reconstruction error) for absorbed vs. non-absorbed features?
- H8: How to quantify "information redistribution"? Mutual information is mentioned but not computed.

**Recommendation:** Either add experimental tests for H7 and H8 or reframe them as theoretical conjectures.

### 9. Pilot Experiments Were Not Properly Powered

The pilot experiments (H9, H10) used the same n=26 sample as the main experiments. Pilots should typically use a smaller sample to save resources, with the understanding that results are exploratory. Using the full sample for pilots blurs the line between pilot and main experiment.

**Recommendation:** For future work, use smaller pilot samples (e.g., n=10 features) to test operationalizations before running full experiments.

### 10. No Protocol for Outlier Handling

The plan does not specify how to handle outliers. Feature U has 24.2% absorption (an outlier), and Feature X has 100% parent firing (p_11 = 1.0, also an outlier). These outliers could disproportionately influence correlations.

**Recommendation:** Add an outlier analysis (e.g., correlation with and without outliers) and report both.

## Positive Aspects

1. **Clear phase structure** (detection, downstream evaluation, graph validation, precision-recall)
2. **Multiple baselines** (random steering, multiple comparison correction, cross-layer validation)
3. **Explicit falsification criteria** for each hypothesis
4. **Resource estimates** provided for remaining experiments
5. **Shared resources** documented (datasets, checkpoints, code paths)

## Summary

The planning is methodologically sound in structure but has critical gaps: no causal intervention to validate the central mechanism, flawed H9 operationalization, underpowered design, and post-hoc correction application. The "optimal compression" reframing lacks experimental tests. The plan would benefit from pre-registration, power analysis, and protocols for handling all possible outcomes (not just predicted ones).
