# Experiment Critique: Local Inhibition Graph Framework

## Summary

The experimental design for the LIG framework is well-conceptualized: training-free approach, clear falsification thresholds, random baseline controls, and a logical six-phase pipeline. However, the critical issue is that H6-H10 have NOT been executed. The paper presents a detailed methodology for experiments that were never run. Beyond this fundamental gap, several methodological issues weaken the proposed experiments: (1) H8 circularity; (2) H10 sign ambiguity; (3) graph specificity untested; (4) n=26 underpowered; (5) random baseline miscalculation; (6) cross-model validation buried.

## Critical Issues

### 1. H6 (Gatekeeper Experiment) Has Not Been Executed (CRITICAL)

The entire empirical contribution of the LIG framework hinges on H6: whether decoder correlation edges predict known absorption pairs. This is described as a ~15-minute computation, yet it has not been run. The paper presents precision@20 predictions (>= 0.10) as if they were results. Without H6, the paper has zero validated empirical claims.

**Impact**: The paper is entirely theoretical until H6 is executed. All claims about graph predictive power, at-risk feature identification, and repair efficacy are speculative.

**Fix**: Execute H6 immediately. This is the single highest-priority action item. If precision@20 >= 0.10, proceed with H7-H10. If precision@20 <= 0.05, pivot to Alternative C (trade-off analysis).

### 2. H8 Circularity: Same Data for Construction and Validation (CRITICAL)

H8 tests whether graph statistics (total_inhibition) predict absorption rate. But both quantities are derived from the same SAE decoder correlations and the same 26 first-letter features. Total_inhibition is computed from decoder correlations (G = W_dec^T W_dec). Absorption rate is computed from activations using the Chanin metric on the same SAE. Correlating two quantities from the same source on the same small sample is methodologically invalid.

**Impact**: Even if H8 is executed, the results will not be credible due to circularity.

**Fix**: Use LOOCV (leave-one-feature-out) or cross-layer prediction (graph on L4, predict on L8). If n=26 is too small for LOOCV, treat H8 as purely exploratory with explicit caveat.

## Major Issues

### 3. H10 Rebalancing Sign Ambiguity

The rebalancing formula is z'_i = z_i + alpha * inh_i where inh_i = sum G_ij * z_j. If G_ij > 0 (positive decoder correlation) and z_j > 0 (child fires), then inh_i > 0, so z'_i = z_i + alpha * positive_value. This INCREASES parent activation. But the mechanism is unclear: why does adding inhibition restore parent firing? The Skeptic correctly identified this as "arbitrary activation boosting" rather than homeostatic rebalancing.

**Fix**: Test BOTH additive (z_i + alpha*inh_i) and subtractive (z_i - alpha*inh_i) rules empirically. The subtractive rule might make more sense: if inhibition suppresses the parent, subtracting it should restore the parent. Report which works. If neither works, drop H10 entirely.

### 4. Graph Specificity Untested

The inhibition graph may predict ANY correlated decoder pair, not specifically absorption pairs. High decoder correlation could indicate any relationship (synonymy, antonymy, co-occurrence), not just parent-child absorption. Without a non-absorbed correlated pair control, enrichment could reflect decoder correlation structure rather than absorption-specific structure.

**Fix**: Add non-absorbed correlated pair control to H6: for each parent latent, identify top-k most correlated neighbors that are NOT absorption pairs. Compare edge weights and enrichment of true absorption pairs vs. non-absorbed correlated pairs.

### 5. Sample Size Is Severely Underpowered

With n=26 features and 4-6 high-absorption features per layer:
- Power to detect |r| >= 0.3: ~25%
- Power to detect |r| >= 0.5: ~65%
- H7 predicts r(recall, inhibition) < -0.3; H8 predicts r > 0.3

Both H7 and H8 are likely underpowered. The paper acknowledges this for H1-H5 but not for the new hypotheses.

**Fix**: Add power analysis for H7 and H8. If power < 50%, treat as exploratory. Expand feature set using WordNet hierarchies if feasible.

### 6. Random Baseline Miscalculation

Expected precision@20 is stated as ~0.004 (20/24000) but the correct value for GPT-2 Small res-jb (24,576 latents) is 20/24576 = 0.000814. This 5x error appears in proposal.md and affects the enrichment claim.

**Fix**: Correct to 0.000814. Recalculate enrichment: precision@20 = 0.10 is 123x enrichment, not 25x.

### 7. Cross-Model Pythia Validation Is Buried

Pythia-70M cross-validation (r=-0.041, p=0.841) shows no signal and receives only one sentence. This is important negative evidence. If the layer 8 GPT-2 trend were real, it should replicate directionally. The Pythia result supports the null-result interpretation but is hidden.

**Fix**: Dedicate a paragraph to Pythia results. Discuss why the trend did not replicate. Note that Pythia used a different metric (embedding similarity vs. binary success rate), which limits comparability.

## Minor Issues

8. **EC50 Feature U contradiction not highlighted**: Feature U (highest absorption, 24.2%) has the LOWEST EC50 (9.17), contradicting H4. This strong null result is consistent with the competitive suppression framework and should be discussed.

9. **No SAELens version pinned**: Reproducibility requires exact version specification.

10. **Exact Chanin threshold not reported**: The differential correlation threshold used for absorption detection is not stated in the paper.

## What Works Well

1. **Training-free methodology** lowers barrier to replication
2. **Clear falsification thresholds** for all hypotheses
3. **Random baseline control** is specified
4. **Multiple comparison correction** is planned for H6-H10
5. **Six-phase pipeline** is logical and well-structured
6. **Power analysis** is honest for H1-H5
7. **Risk assessment table** includes genuine failure modes
