# Testable Hypotheses with Expected Outcomes

## H1: Absorption-Quality Causal Chain

### Statement
The negative correlation between SAE feature absorption rate and downstream task performance (sparse probing, RAVEL, SCR, unlearning) observed across 54 Gemma Scope SAEs reflects a genuine causal mechanism, not a confound with SAE width or L0 sparsity.

### Operationalization
- **H1a (L0 Covariate)**: partial_corr(absorption, sparse_probing | log_width, layer, arch_class, log_L0) has |r| > 0.3 and p < 0.05
- **H1b (Width Stratification)**: Within at least 2 of 3 width groups (16k, 65k, 1M), Spearman correlation between absorption and at least one quality metric has bootstrap 95% CI excluding 0
- **H1c (Mediation)**: Absorption mediates > 30% of L0's total effect on downstream quality (proportion mediated > 0.30, bootstrap CI excluding 0)
- **H1d (Sensitivity)**: Rosenbaum Gamma > 1.5 for the matched-pair comparison

### Expected Outcome
Mixed. The L0-controlled partial correlation (H1a) likely drops from |0.595| to the range |0.3-0.45| but remains significant. Width stratification (H1b) likely succeeds for the 1M stratum (which has sufficient L0 variation) but may fail for 16k and 65k (where L0 variation within-width is limited). Mediation (H1c) is the most uncertain -- it depends on the functional form of the L0-absorption-quality chain.

### Falsification Criterion
H1 falsified if: |r| < 0.2 after L0 control AND no stratum has significant correlation AND mediation CI includes 0 AND Gamma < 1.2. All four conditions must hold simultaneously.

---

## H2: Cross-Domain Absorption Existence

### Statement
Feature absorption occurs in knowledge-domain hierarchies (city -> country, city -> continent, city -> language) at rates comparable to or exceeding first-letter absorption rates (15-35%), and exceeds a shuffled-hierarchy control by at least 3x.

### Operationalization
- **H2a (Existence)**: Absorption rate for city-country features on Gemma Scope 16k at layer 12 exceeds 10%
- **H2b (Specificity)**: Absorption rate on city-country exceeds 3x the rate on shuffled city-random_country (expect shuffled < 5%)
- **H2c (Cross-Domain Comparison)**: Correlation between absorption rates across domains (first-letter vs. country vs. continent) is positive (Spearman rho > 0.3 across SAE configurations)
- **H2d (Hierarchy Sharpness)**: Absorption severity correlates with hierarchy sharpness (MI between parent and child features), with first-letter (deterministic) showing highest absorption and sentiment-topic (probabilistic) showing lowest

### Expected Outcome
H2a likely succeeds: knowledge hierarchies have deterministic co-occurrence (India is always in Asia) similar to first-letter, so the sparsity incentive exists. H2b likely succeeds given H2a. H2c is uncertain -- different attributes may have different absorption profiles. H2d is the most informative: if the contrarian is right that absorption is task-dependent, the hierarchy sharpness correlation will be strong and first-letter rates will be outliers.

### Falsification Criterion
H2 falsified if: absorption rate < 5% on all three knowledge attributes after probe quality gate (accuracy >= 85%) OR shuffled-hierarchy baseline > 15% (metric is broken).

---

## H3: Absorption Scaling Surface Structure

### Statement
In a GAM fit of absorption rate on (log(width), log(L0)) across 200+ SAEs, the interaction term ti(log(width), log(L0)) is significant, indicating that absorption depends on the joint structure of width and L0 rather than either parameter independently.

### Operationalization
- **H3a (Interaction)**: The tensor interaction term in the GAM has p < 0.05 and explains > 5% of deviance beyond the additive model
- **H3b (Monotonicity)**: Absorption rate increases monotonically with width at fixed L0 (partial derivative d(absorption)/d(width) > 0 for most of the parameter space)
- **H3c (L0 Effect)**: At fixed width, absorption rate decreases with L0 in the mid-L0 range (L0 = 50-200) but potentially increases at very low L0 where hedging dominates
- **H3d (Phase Regions)**: The gradient magnitude surface has at least one ridge (region of maximal gradient) separating a low-absorption region from a high-absorption region

### Expected Outcome
H3a is moderately likely: the absorption-width and absorption-L0 relationships are both established independently but their interaction has never been tested. H3b is well-supported by prior work (wider SAEs show more absorption). H3c is predicted by the theoretical framework. H3d is uncertain -- the transition may be smooth rather than sharp.

### Falsification Criterion
H3 falsified if: interaction term p > 0.10 and the additive model captures > 95% of explained deviance. This would mean absorption is additively separable in width and L0.

---

## H4: Cross-Domain Early Absorption Dominance

### Statement
Early-type absorption (decoder-absent: no decoder direction within cosine > tau of parent probe direction, dead features excluded) accounts for > 50% of absorbed instances in knowledge-domain hierarchies, consistent with iter_001 findings on first-letter features.

### Operationalization
- **H4a (Early Dominance)**: At the data-driven tau threshold (elbow of early-fraction vs. tau curve), early-type fraction > 50% across at least 2 SAE configurations on knowledge attributes
- **H4b (Width Effect)**: Early-type fraction decreases with SAE width (Spearman rho < -0.3 between width and early fraction)
- **H4c (Robustness)**: Early-type dominance is confirmed by k-nearest-decoder alternative (k=5: none of top-5 decoders close to parent probe)

### Expected Outcome
H4a is supported by iter_001 but at n=2, requiring replication. H4b tests whether wider dictionaries reduce coverage gaps -- likely yes based on the coverage hypothesis. H4c provides threshold-free robustness.

### Falsification Criterion
H4 falsified if: early-type fraction < 30% in >= 2/3 of configurations at tau in {0.2, 0.3, 0.4} AND k-nearest-decoder also disagrees.

---

## H5: L0 Confound in Taxonomy

### Statement
Recomputing the absorption taxonomy with proper comparison tokens (frequency-matched rather than global fallback) reduces the Type II rate from 88.5% to < 60%, and the combined absorption rate from 92.3% to < 80%.

### Operationalization
- **H5a**: Type II rate with proper comparison tokens < 60%
- **H5b**: Combined absorption rate (Type I + corrected Type II + Chanin) < 80%
- **H5c**: The corrected combined rate has a 95% bootstrap CI width < 15 percentage points

### Expected Outcome
Likely to succeed: the 92.3% rate is inflated by construction (n_comparison_tokens=0 fallback). The corrected rate will be lower but the extent of reduction is uncertain.

### Falsification Criterion
H5 is not falsifiable in the traditional sense -- it is a measurement correction. However, if the corrected rate is within 5% of the original (> 87.3%), then the fallback was not producing substantial inflation and the original rate is approximately correct.
