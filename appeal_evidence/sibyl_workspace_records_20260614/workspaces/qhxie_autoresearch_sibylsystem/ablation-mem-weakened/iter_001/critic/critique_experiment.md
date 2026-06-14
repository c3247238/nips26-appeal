# Experiment Critique

## Overall Assessment

The experimental design is conceptually sound but severely underpowered and limited in scope. The four-phase pipeline (absorption detection -> steering -> probing -> correlation analysis) is well-conceived, but the execution has critical flaws that prevent the null results from supporting the paper's conclusions.

## Critical Issues

### 1. Severe Underpowering (Critical)

- n=26 features (A--Z) with strongly right-skewed absorption distribution
- 18-26 of 26 features per layer show absorption below 10%
- Observed range spans only 0.242 (24.2 percentage points)
- ~65% power to detect |r| >= 0.50 at alpha=0.05; observed correlations (-0.30 to +0.01) fall well below detection threshold
- Layer 8 H1: r=-0.301, p=0.136 -- a negative trend that would likely achieve significance with n~85

The paper acknowledges this in Section 6.1 ("Low absorption variance") and Section 6.3 ("underpowered for small-to-medium effects"), but the abstract and conclusion still frame the null result as informative evidence. This is a fundamental mismatch between what the data can support and what the paper claims.

**Fix:** Add a formal power analysis table. Reinterpret all null results as "no detectable relationship given power constraints" rather than "no relationship." Treat the layer 8 trend as suggestive, not conclusive.

### 2. Single Model Family with Shallow Features (Critical)

- Only GPT-2 Small (124M parameters) tested
- First-letter features (A--Z) have a shallow, uniform hierarchy
- The paper's original target was Gemma-2-2B but gated access prevented loading
- GPT-2 Small may not exhibit absorption strongly enough to produce measurable task degradation

The paper acknowledges this as Limitation 1, 3, and 4, but the conclusion still generalizes beyond what the data support.

**Fix:** Temper all conclusions with explicit scope: "in GPT-2 Small with first-letter features at layers 4 and 8." Add a sensitivity analysis showing what absorption rate would be needed to produce detectable correlation given observed task variance.

### 3. Steering Robustness Confound (Major)

Steering adds the decoder direction W_dec[phi(f)] directly to the residual stream, bypassing the encoder entirely. Even if the parent latent fails to fire naturally, the injected direction still influences the output. This mechanism may be inherently robust to the type of absorption measured by differential correlation, which captures activation redistribution among latents rather than direction degradation in the decoder.

Feature U (A=0.242) achieving 100% steering success at s=50 is presented as surprising evidence, but it is exactly what we would expect if steering bypasses the encoder. The paper does not distinguish between:
- **Encoder absorption**: Parent latent fails to fire (what Chanin measures)
- **Decoder degradation**: Decoder direction is corrupted (would affect steering)

**Fix:** Add explicit discussion of why steering may be robust to encoder absorption. Distinguish between encoder and decoder degradation. Consider adding an experiment that tests whether the decoder direction itself is degraded for absorbed features (e.g., by measuring reconstruction quality when only the parent latent is active).

### 4. Missing Random Feature Baseline for Steering (Major)

The paper acknowledges this as Limitation 7 but does not quantify its impact. Without a random baseline, we cannot determine whether steering effects are specific to the feature direction or would occur with any decoder direction. If random directions also produce steering success rates of 0.70+, the observed steering effects are uninformative about absorption.

**Fix:** Add a random feature steering baseline (26 random latents from the same layer). This is a quick experiment that would substantially strengthen or weaken the paper's claims.

### 5. Only Two Downstream Tasks (Major)

Only steering and sparse probing were tested, and only at layers 4 and 8. Circuit finding with activation patching and model editing---tasks that require precise feature isolation---may be more sensitive to absorption. The sparse probing task uses L1-regularized logistic regression which can leverage correlated latents, making it resilient to single-feature absorption.

**Fix:** Narrow the conclusion scope to "for steering and sparse probing in GPT-2 Small." Add discussion of why other tasks may show different sensitivity. Consider adding a lightweight circuit-finding proxy if time permits.

### 6. Single Absorption Metric (Major)

Only the Chanin differential correlation metric was used. SAEBench includes an ablation-based absorption metric that may capture different failure modes. The null result may be specific to the differential correlation definition rather than absorption generally.

**Fix:** Add validation using SAEBench's ablation-based metric on a subset of features. If the alternative metric also shows no correlation, the null result is more robust.

## Minor Issues

### 7. H3 Tested with Only Two Layers (Minor)

H3 (cross-layer consistency) is tested with only layers 4 and 8. With n=2 slope pairs, the CV=0.932 has no statistical meaning. The opposite-sign finding for H1 is sufficient to reject consistency, but the magnitude comparison is not meaningful.

**Fix:** Acknowledge that H3 testing with two layers is exploratory. Remove the CV value or note it is computed from n=2.

### 8. Missing Full-Activation Probing Data (Minor)

F1_full is defined in Section 4.5 but never reported. Section 5.3 claims full-activation probing outperforms k-sparse but provides no numbers.

**Fix:** Add a table or remove the claim.

### 9. Exact Chanin Threshold Not Reported (Minor)

The metric requires a threshold to flag absorbing children, but the threshold value is not reported.

**Fix:** Report the exact threshold in Section 4.3.

## Score: 5/10

The experimental pipeline is well-designed, but severe underpowering, limited scope, and missing controls prevent the null results from supporting the paper's conclusions.
