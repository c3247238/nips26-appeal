# Experiment Critique -- Iteration 9

## Executive Summary

Iteration 9 represents a genuine breakthrough: the three experiments that were blocking for three consecutive reviews (activation patching, tightened hedging, CMI replication at L0=22) are all executed. The validate_integration script confirms 84/85 numerical claims match source data. The experimental foundation is now substantially stronger than any prior iteration. However, three critical issues remain that could derail reviewer confidence.

## Critical Issues

### 1. ACTIVATION PATCHING CONFOUNDED BY JUMPRELU HARD THRESHOLD (Severity: Critical)

The 0/8 parent recovery result is the paper's only metric-independent causal evidence. But the experimental design has a confound unique to JumpReLU SAEs: the decode-reencode pipeline passes through hard thresholds theta_j. Consider the scenario where competitive exclusion IS the true mechanism -- child latent z_c pushes the parent latent's pre-threshold activation 5% below theta_j. Zeroing z_c raises the pre-threshold value by 5%, but it still does not cross theta_j. Result: 0/8 recovery, even though competitive exclusion is occurring.

This confound does not exist for L1-ReLU SAEs (graded suppression means any change in competition produces a proportional change in activation). The paper's own motivation notes that JumpReLU's "binary activation regime -- fire or zero -- differs qualitatively from L1-ReLU's graded suppression." The activation patching experiment does not account for this qualitative difference.

**What the paper should report but does not**: pre-threshold parent activation magnitudes before and after child zeroing. If the pre-threshold value increases substantially (e.g., >30% of the gap to theta_j) without crossing, this is evidence FOR competitive exclusion masked by the nonlinearity. If the pre-threshold value does not change at all, competitive exclusion is genuinely absent.

**Recommended fix**: Either (a) report pre-threshold activation changes, or (b) replicate on GPT-2/L1-ReLU as a positive control where competitive exclusion is known to exist, or (c) add a more extended discussion acknowledging this confound with a back-of-envelope estimate of its severity.

### 2. CMI REPLICATION MAY BE PILOT ONLY (Severity: Critical)

The file `cmi_replication_l0_22.json` in the pilots directory shows:
- `"mode": "PILOT"`
- `"total_words": 125` (5 per letter)
- `"n_corpus_samples": 7594`

The full CMI analysis at L0=82 used 1,092 words. The k-NN MI estimator (k=5) with only 5 positive examples per letter in a d'=10 subspace is at the estimator's absolute reliability floor. With n=5 positive examples per letter, the class-conditional density estimation in 10 dimensions is degenerate -- the estimator has 5 points in a 10-dimensional space, which is below the theoretical minimum for reliable density estimation.

The paper reports rho=+0.044 (p=0.835) at L0=22 as the definitive replication, but if this comes from the pilot, the null result may reflect estimator failure rather than true absence of signal. A reviewer who checks the source data will see "PILOT" and question the entire Section 6.3.

**Recommended fix**: Either (a) run the full CMI estimation at L0=22 with the complete vocabulary, or (b) clearly label the result as a pilot finding and discuss the reduced power.

### 3. 93.8% NON-HEDGING CATEGORY UNEXPLAINED (Severity: Critical)

The tightened hedging analysis reveals an unexpected finding that the paper barely discusses. Of 656 FN tokens at L0=22:
- 41 (6.2%) show strict hedging (parent-specific latents fire at L0=176)
- 615 (93.8%) show NO parent-specific latents firing at L0=176
- Yet nearly all 615 are NOT FN at L0=176

This means these tokens' FN status resolves at higher L0, but through entirely different features -- not through any of the 5 parent-associated latents identified at L0=22. The most parsimonious explanation: the L0=176 SAE uses different latents for first-letter classification than the L0=22 SAE. The "parent features" are probe-specific and L0-specific, not universal features of the SAE.

This undermines the confound decomposition framework's fundamental assumption: that cross-L0 persistence of FN status indicates a property of the token-feature pair. If the probe latents change across L0 values, the persistence criterion is comparing different measurement instruments.

**Recommended fix**: Compute the Jaccard similarity between the top-5 probe latents at L0=22 and L0=176 for each letter. If overlap is low (Jaccard < 0.3), acknowledge that the confound decomposition compares different feature sets across L0 values and discuss implications.

## Major Issues

### 4. SMALL ACTIVATION PATCHING SAMPLE (8 WORDS)

The 0/8 result has a 95% CI of [0%, 36.9%] (Clopper-Pearson). The data are consistent with competitive exclusion rates up to 37%. Five of the 8 words (other, often, offer, under, until) are high-frequency function-adjacent words in the top 1000 most frequent English words. These words may have fundamentally different SAE encoding dynamics than lower-frequency content words.

**Recommended fix**: Soften language from "rules out" to "provides evidence against." If GPU time is available, extend patching to 50+ words from the 186 FNs at L0=82.

### 5. THREE VOCABULARY SIZES (1204/1196/125)

The paper uses three different vocabulary sizes for different analyses, creating traceability problems:
- 1,204 words for first-letter replication (Section 4.7)
- 1,196 / 1,195 tested for confound decomposition (Section 4.4)
- 125 words for CMI pilot (Section 6.3?)

The FN count is 657 in confound decomposition but 656 in tightened hedging. While each vocabulary choice is individually justified, the cumulative effect is that the reader must track which denominator is in use at each point.

### 6. LETTER G DOMINATES STRICT HEDGING

Letter G accounts for 19/41 (46%) of all strict-hedging cases with a rate of 90.5%. Excluding G, strict hedging drops from 6.2% to 3.5% -- barely above the shuffled control of 3.4%. The z-test without G is not reported.

## Positive Assessment

### Data Integrity: Excellent

The validate_integration script confirmed 84/85 claims match source JSONs (integrity 98.8%). The one missing-data item (leave-one-out max-influence letter V) is supported by the jackknife analysis. This is a significant quality improvement over prior iterations.

### Statistical Methodology: Strong

- Bootstrap CIs with 10k resamples throughout
- Bonferroni correction applied to all multiple comparisons
- Effect sizes (Cohen's d) reported for group comparisons
- Leave-one-out sensitivity for CMI correlation
- Shuffled controls with multiple replicates (n=5 or n=10)
- The progressive weakening table (Table 6) for CMI is exemplary

### Negative Result Reporting: The Paper's Strongest Aspect

Consistently across all reviews, the honest reporting of falsified hypotheses (H2, H4, H5, H6, H7, H8, H9, H10 -- 8 of 10 falsified) with pre-registered targets vs. observed values is rated as the strongest methodological feature. This is commendable.

## Experiment Completeness Scorecard

| Experiment | Status | Quality | Comments |
|-----------|--------|---------|----------|
| First-letter replication | DONE | High | 15.96% matches published range |
| Four-control suite | DONE | High | Universal failure, well-documented |
| Confound decomposition | DONE | High | Multi-L0 persistence, clean |
| L0 phase transition | DONE | High | Cross-layer validation, CV<10% |
| Cross-domain (5 domains) | DONE | Medium | All below controls, uninterpretable |
| CMI estimation (L0=82) | DONE | High | Full vocabulary, proper k-NN |
| Partial correlation | DONE | High | Eliminates confound signal |
| Leave-one-out | DONE | High | Stable (max delta=0.088) |
| Threshold sensitivity | DONE | High | 5x4 grid, CV=0.077 |
| Control failure diagnosis | DONE | High | 1000 random vectors |
| **Activation patching** | **DONE** | **Medium** | JumpReLU threshold confound |
| **Tightened hedging** | **DONE** | **High** | Reveals 6.2% vs 98.6% gap |
| **CMI at L0=22** | **DONE (PILOT?)** | **Uncertain** | May be 125-word pilot only |
| Validate integration | DONE | High | 84/85 claims verified |
