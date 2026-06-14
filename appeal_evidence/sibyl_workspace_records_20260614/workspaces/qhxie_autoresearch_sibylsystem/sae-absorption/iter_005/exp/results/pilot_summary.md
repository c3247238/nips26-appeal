# Pilot Summary: P1_confound_go_nogo

**Status**: GO | **Runtime**: ~1min | **Task**: L0 Covariate Go/No-Go Test

## Pass Criteria Assessment

**Criterion**: "At least one quality metric retains |partial_r| > 0.2 after L0 control. Script runs without error on full 54-SAE dataset."

**Result**: PASSED (3/4 metrics pass). Strong evidence that absorption is NOT merely a proxy for L0.

## Key Findings

### 1. Absorption retains substantial independent effect after L0 control

| Metric | Bivariate r | Partial r (no L0) | Partial r (with L0) | Delta | Pass |
|--------|------------|-------------------|---------------------|-------|------|
| Sparse Probing F1 | -0.587 | -0.664 | **-0.746** | -0.082 | YES |
| SCR | -0.449 | -0.692 | **-0.570** | +0.122 | YES |
| RAVEL TPP | -0.471 | -0.488 | **-0.331** | +0.157 | YES |
| Unlearning | -0.182 | -0.082 | -0.123 | -0.042 | NO |

### 2. Sparse Probing F1 shows suppression effect (SURPRISE)
- Adding L0 as covariate STRENGTHENED the absorption-quality correlation (-0.664 -> -0.746)
- This is a "suppression variable" effect: L0 was partially masking absorption's true effect
- Interpretation: L0 and absorption have opposite partial effects on sparse probing F1

### 3. Low multicollinearity confirms clean identification
- VIF(log_L0) = 1.09, VIF(log_width) = 1.08 -- all well below 10
- log(L0) vs log(width) correlation: r = -0.278 (p = 0.055) -- barely correlated
- The partial correlations are well-identified despite controlling for multiple covariates

### 4. arch_class is constant after L0 filtering
- All 48 SAEs with known L0 are gemma_scope_2b; 6 canonical SAEs have null L0
- arch_class dropped from covariates (constant column)
- Reference partial correlations on the full 54-SAE dataset (with arch_class) match iter_004 closely

## GO/NO-GO Decision: **GO**

3/4 quality metrics retain |partial_r| > 0.2 AND statistical significance (p < 0.05) after adding log(L0). The absorption-quality relationship is robust to L0 confound control. Proceed with full Phase 1 analysis (width-stratified, mediation, Rosenbaum).

---

# Pilot Summary: P1_clustered_regression

**Status**: GO | **Runtime**: 0.7s | **Task**: Clustered SE Regression for C2C PMI

## Pass Criteria Assessment

**Criterion**: "Clustered SE regression completes. Report whether HC3 and clustered SE produce materially different conclusions."

**Result**: PASSED. The analysis completed successfully on all 806 observations (26 letter clusters x 31 SAE configs).

## Key Findings

### 1. PMI remains non-significant under clustered SE
- HC3: beta_PMI = -0.0063, p = 0.593
- Clustered SE: beta_PMI = -0.0063, p = 0.667
- Both methods agree: PMI does not predict absorption in OLS

### 2. log(L0) significance CHANGES (important for Phase 1)
- HC3: beta_L0 = +0.013, p = 0.012 (significant)
- Clustered SE: beta_L0 = +0.013, p = 0.206 (NOT significant)
- SE ratio = 1.99x -- clustering nearly doubles the L0 standard error
- This matters for the Phase 1 confound resolution: L0's effect on absorption may be weaker than iter_004 suggested

### 3. Distributional misspecification is severe
- 58.6% of observations are exactly zero (zero-inflated)
- Skewness = 4.812, kurtosis = 30.781
- OLS is fundamentally the wrong model for this data

### 4. Hurdle model reveals PMI IS significant
- Logistic part (P(absorption > 0)): PMI beta = -1.37, clustered p = 0.006
- Interpretation: Higher PMI letters are LESS LIKELY to show any absorption
- This is the opposite direction from the OLS null but is masked by the zero inflation

### 5. Beta regression also finds PMI significant
- Beta reg PMI: beta = -0.24, p = 0.005
- When properly modeling the bounded [0,1] response, PMI matters

## Implications for Full Experiment

1. The Phase 1 analyses should use clustered SE throughout (not HC3)
2. The PMI-absorption relationship exists but is hidden by distributional misspecification
3. A hurdle model should be the primary specification in the paper
4. The log(L0) -> absorption pathway is weaker than iter_004 reported (p changes from 0.012 to 0.206 with clustering)

---

# Pilot Summary: P1_rosenbaum

**Status**: PASS | **Runtime**: 0.4s | **Task**: Rosenbaum Sensitivity Analysis with Propensity Matching

## Pass Criteria Assessment

**Criterion**: "At least 5 matched pairs formed. Gamma > 1.0."

**Result**: PASSED. Mahalanobis matching yields 17 pairs with Gamma = 2.65 (strong robustness).

## Key Findings

### 1. Five matching strategies tested

| Strategy | N Pairs | Sparse Probing p | TPP p | Max Gamma |
|----------|---------|------------------|-------|-----------|
| Exact width + NN L0 | 4 | N/A (too few) | N/A | N/A |
| Median split within width | 23 | 0.126 | 0.580 | 1.00 |
| Propensity score | 6 | 0.031 | 0.031 | 1.80 |
| Mahalanobis distance | 17 | 0.007 | 0.001 | 2.65 |
| Tertile within width | 16 | 0.231 | 0.252 | 1.00 |

### 2. Within-width matching shows NO significant quality differences

Both within-width strategies (median split: 23 pairs, tertile: 16 pairs) fail to produce significant Wilcoxon tests for any quality metric. This is the most conservative test: it eliminates width as a confound entirely.

**Interpretation**: Absorption's association with quality may be partly or fully driven by cross-width variation, not within-width variation. This is a critical caveat for H1.

### 3. Cross-width matching shows STRONG effects with good robustness

Mahalanobis matching (17 pairs) finds significant effects for:
- **Sparse probing F1**: p = 0.007, Gamma = 1.85 (moderate robustness)
- **RAVEL TPP**: p = 0.001, Gamma = 2.65 (strong robustness)

The TPP result can withstand a hidden confounder with a 2.65:1 odds ratio -- this is strong evidence that the quality difference is not easily explained away by unobserved confounds.

### 4. SCR is non-significant across ALL matching strategies

Unlike the partial correlation analysis (where SCR showed strong effects), SCR does not survive matched-pair analysis. This suggests SCR's correlation with absorption may be more sensitive to functional form assumptions than the rank-based matched-pair test.

### 5. Rosenbaum sensitivity interpretation

- **Gamma = 1.00**: Result not significant even without hidden bias (within-width strategies)
- **Gamma = 1.80**: Moderate robustness (propensity score matching)
- **Gamma = 2.65**: Strong robustness (Mahalanobis matching on TPP)

## Critical Caveats

1. **Width confound not fully resolved**: Mahalanobis matching does not enforce exact width balance. High-absorption SAEs (concentrated in 1M) may be matched with low-absorption SAEs of different widths.
2. **Within-width null is informative**: The null result from within-width matching is an important negative finding that must be honestly reported.
3. **Sample size limitation**: Even the best strategy only yields 17 pairs from 48 SAEs.
4. **Only 2 of 4 quality metrics show significance**: Sparse probing F1 and TPP survive, but SCR and unlearning do not.

## Implications for Paper Narrative

The Rosenbaum analysis provides a nuanced picture:
- Absorption IS associated with quality differences even after matching on confounders (cross-width)
- But the association is NOT detectable within width strata (within-width null)
- The paper should present both results transparently and discuss whether the effect is a genuine causal chain or an emergent property of cross-width architecture differences
