# Critique of Planning and Methodology

## Research Design

### Issue 1: Pre-Registration Missing

The research hypotheses evolved across iterations (H1 → H1b, addition of H5-H10, H9 declared tautological). This evolutionary process is normal in exploratory research but means:
- Some hypotheses were modified after seeing data
- The "multiple comparison correction" was applied post-hoc to tests that were not pre-registered at the same significance threshold

The paper should clearly state which hypotheses were pre-registered and which were added post-hoc.

### Issue 2: Post-Hoc Power Analysis

Section 3.6 reports "approximately 20% power to detect a medium effect size." Post-hoc power analysis is methodologically problematic because:
- It uses the observed effect size to calculate power
- This is circular (if you found no effect, power will appear low)
- A-priori power analysis should guide sample size decisions

### Issue 3: Multiple Comparisons Applied Inconsistently

The paper applies Bonferroni and BH-FDR correction to the 12 main hypothesis tests, but:
- H5 (precision-recall asymmetry) is presented as "SUPPORTED" without correction
- H7 (trained < random absorption) is presented as "SUPPORTED" with p<0.001 — but this is a different type of test (comparison of means vs. correlation)

The correction framework should be consistently applied.

## Theoretical Framework

### Issue 4: LCA Connection is Theoretically Interesting but Empirically Unsupported

The paper's theoretical contribution is connecting the Locally Competitive Algorithm (LCA) to SAE absorption. However:
- The primary prediction of this framework (decoder graph predicts absorption pairs, H6) is falsified
- The secondary prediction (graph statistics predict at-risk features, H8) is not supported
- The "homeostatic rebalancing" intervention (H10) was not tested

The LCA connection is presented as established theory but the empirical validation is lacking.

### Issue 5: "Optimal Compression" Framing is Post-Hoc

The reframing of absorption as "rate-distortion optimal compression behavior" emerged from the H7/H10 results showing trained SAEs have lower absorption than random SAEs. This framing is:
- Not pre-registered
- Not directly tested
- Used to explain why null results should be interpreted positively

If the framing is rejected by reviewers, the paper loses its primary theoretical contribution.

## Methodological Choices

### Issue 6: First-Letter Features are a Shallow Hierarchy

The 26 first-letter features (A-Z) represent a very shallow hierarchy. Semantic hierarchies (animal → dog → poodle) would likely show different absorption patterns. The narrow feature set limits generalizability.

### Issue 7: Steering vs. Probing as Downstream Tasks

The paper uses steering and probing as downstream tasks, but:
- Steering bypasses the encoder (adds decoder direction directly)
- Steering robustness is expected regardless of absorption
- Probing is a better test of encoder functionality, but probing results are also null

The choice of downstream tasks is appropriate, but their interpretation should be clearer.

## Risk Assessment

### Risk 1: Reviewers Will Focus on Power Issue

Any competent reviewer will note that n=26 provides inadequate power. The paper's framing of null results as positive findings ("honest null-result reporting") will likely be challenged.

**Mitigation**: Strengthen language around limitations. Use effect sizes and confidence intervals rather than p-values.

### Risk 2: The Metric Validity Challenge

The paper acknowledges that random SAEs show high absorption (0.278), suggesting the Chanin metric measures "dictionary structure" not "learned pathology." But the paper then uses this flawed metric to support its conclusions.

**Mitigation**: Either develop a structure-insensitive metric, or clearly acknowledge that the metric validity question undermines the conclusions.

### Risk 3: The H6 Falsification Undermines the Theoretical Story

H6 (decoder graph predicts absorption) is falsified with precision@20=0.0. This directly undermines the LCA/competitive suppression theoretical framework that the paper relies on.

**Mitigation**: Do not present the LCA framework as an established explanation. Frame it as a hypothesis for future work.

## Recommendations

1. **Pre-register hypotheses**: Clearly separate pre-registered from post-hoc hypotheses.

2. **Remove post-hoc power analysis**: Discuss a-priori power analysis instead.

3. **Strengthen limitations section**: Acknowledge that null results at low power are uninterpretable.

4. **Do not present falsified hypotheses as theory**: H6 should be prominently reported as falsified, not used as theoretical motivation.

5. **Clarify what would falsify the theory**: State clearly what evidence would contradict the "absorption is benign" claim.
