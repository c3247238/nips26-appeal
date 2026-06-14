# Skeptic Analysis: Iteration 6 Experiment Results (Updated)

## Executive Summary

The results suffer from **two fatal flaws** and **four serious concerns** that collectively undermine most of the paper's claimed contributions. The cross-domain absorption claims are invalidated by universal control failure (shuffled-label rates 2x-47x above measured rates). The CMI rate-distortion diagnostic is built on a dimension-dependent artifact that reverses at d'>10 and fails statistical significance even at the cherry-picked d'=10. The confound decomposition directly contradicts the proposal's H2 hypothesis (96.9% hierarchy-driven in the pilot; 1.4% in the full experiment -- a 69x discrepancy). The first-letter improved experiment raises its own problems: the shuffled control worsened to 74.6%, and the random probe control (11.8%) exceeds the 2% target by 6x. Only the first-letter replication rate (15.96%) and the geometric constant negative result survive scrutiny as credible findings.

---

## 1. Statistical Risk Inventory (Top 3)

### Risk 1: Universal Control Failure Invalidates All Absorption Claims

The cross-domain comparative and the improved first-letter experiment show that **every domain's measured absorption rate falls below its shuffled-label control**. The improved experiment makes this worse, not better:

**Cross-Domain Comparative (original dataset, L12-16k):**

| Domain | Measured Rate | Shuffled Control | Random Control | Net Signal vs Shuffled |
|--------|-------------|-----------------|---------------|----------------------|
| First Letter | 13.4% | **59.6%** | 9.2% | **-46.2%** |
| City->Country | 0.0% | 10.3% | 19.0% | -10.3% |
| City->Continent | 6.5% | **45.2%** | 12.9% | **-38.7%** |
| City->Language | 6.6% | 18.0% | 20.8% | -11.5% |
| Animal->Class | 1.4% | **39.3%** | **34.3%** | **-37.9%** |

**Improved First-Letter (1,204 words, L12-16k-L0=82):**

| Control | Target | Actual | Status |
|---------|--------|--------|--------|
| C1 Random Probe | < 2% | **11.8%** | FAIL (6x above target) |
| C2 Shuffled Labels | < 20% | **74.6%** (mean of 5 shuffles, range 71.4-77.5%) | FAIL (3.7x above target) |

The shuffled-label control WORSENED from 59.6% (original) to 74.6% (improved experiment with more words). This is the opposite of what refinement should achieve. A shuffled-label control randomly reassigns which parent feature "belongs" to which letters. If the metric is measuring genuine hierarchical absorption, shuffled labels should produce near-zero signal because the probe direction no longer aligns with the actual parent-child hierarchy. When shuffled labels produce 4.7x the signal of the real experiment (74.6% vs 15.96%), the metric is fundamentally broken for this model.

Additionally, the random probe control at 11.8% is dangerously close to the measured rate of 15.96%. The "signal above random" is only 4.16 percentage points, which for n=1,203 words gives a razor-thin margin.

The `pass_criteria.overall` field in the JSON is set to `true` with the note "Even if not all criteria pass, meaningful absorption > 0% is informative." This is goalpost-moving: the pass criteria were pre-registered, and 3 of 4 criteria fail. Declaring "overall pass" by fiat undermines the entire pre-registration framework.

**Severity: FATAL FLAW.** The paper cannot claim any absorption rate as meaningful when the shuffled control produces 4.7x higher rates. H1 (cross-domain) is invalidated. Even the first-letter rate, while replicating Chanin et al. in magnitude, cannot be attributed to genuine absorption when the control structure fails.

### Risk 2: CMI-Absorption Correlation Is Dimension-Dependent and Non-Robust

The CMI estimation results:

| d' (subspace dim) | Spearman rho | p-value | Cohen's d |
|---|---|---|---|
| 10 | **-0.383** | **0.059** | **-0.924** |
| 20 | +0.048 | 0.818 | +0.226 |
| 30 | +0.299 | 0.147 | +0.616 |
| 50 | +0.197 | 0.345 | +0.499 |

Critical problems:

1. **Sign reversal across dimensions.** The negative correlation exists ONLY at d'=10. At d'=20, 30, and 50 the correlation is POSITIVE (higher CMI correlates with MORE absorption), which is the exact opposite of the theoretical prediction. A robust information-theoretic relationship should be monotonic with respect to the subspace dimensionality -- adding more relevant dimensions should either maintain or strengthen the signal, not reverse it.

2. **Not statistically significant.** Even at d'=10, p=0.059 does not reach conventional significance (p<0.05). The pilot summary labels this "PASS" with the note that Mann-Whitney p=0.045 is significant -- but Mann-Whitney tests a different hypothesis (group mean difference, not rank correlation). The Spearman rho test (the pre-registered criterion for H3) does NOT pass.

3. **Multiple comparisons not corrected.** With 4 dimension candidates tested, Bonferroni-corrected p = 0.059 * 4 = 0.236. The pilot summary does not apply this correction.

4. **Contradicts earlier result.** The original `successive_refinement` task (using the first validation dataset with 576 samples) found rho = +0.14 (wrong direction, p > 0.2). The sign flip from +0.14 to -0.383 between datasets is concerning. The improvement is attributed to "better absorption rate estimates from 50+ words/letter" and "larger sample for k-NN MI estimation" -- but if the result is this sensitive to dataset construction, it is not robust.

5. **Confounding with probe quality acknowledged but not controlled.** The phase_transition_prediction pilot explicitly states: "Confounding with probe quality: High-absorption letters tend to have lower probe F1, which could confound the CMI-absorption relationship." If low-CMI letters happen to have poor probes, and poor probes produce more false negatives (measured as "absorption"), the CMI-absorption correlation could be entirely driven by probe quality variation.

**Severity: FATAL FLAW.** The paper's primary theoretical contribution (H3: rate-distortion diagnostic via CMI) is supported by exactly one cherry-picked dimensionality setting at a non-significant p-value. This does not constitute evidence for the successive refinement theorem's applicability to SAE absorption.

### Risk 3: Confound Decomposition Contradicts H2 by 69x

The proposal's H2: "hierarchy-driven absorption accounts for > 80% of total false negatives at L0=22." The pilot claimed 96.9%. The full-scale result:

| L0 | Absorption Rate | n_FN | Hierarchy-Driven | Hedging | Reconstruction Error |
|----|----------------|------|-----------------|---------|---------------------|
| 22 | 42.9% | 657 | **9 (1.4%)** | 648 (98.6%) | 0 (0%) |
| 41 | 37.5% | 489 | 9 (1.8%) | 480 (98.2%) | 0 (0%) |
| 82 | 14.4% | 185 | 9 (4.9%) | 176 (95.1%) | 0 (0%) |
| 176 | 0.8% | 10 | 9 (90.0%) | 1 (10.0%) | 0 (0%) |

The 96.9% vs 1.4% discrepancy (69x ratio) is not a statistical fluctuation. It indicates a fundamental methodological error in the pilot. The fact that exactly 9 words appear as "hierarchy-driven" at EVERY L0 value -- including L0=176 where only 10 total false negatives exist -- strongly suggests these are fixed dataset artifacts (words whose first letter simply is not well-represented in the SAE) rather than genuine hierarchy-driven absorption. The number 9 being constant across all 4 L0 values is a red flag for a data-processing bug or a misspecified classification criterion.

**Severity: SERIOUS CONCERN.** H2 is falsified by a factor of 57x. The pilot-to-full discrepancy (96.9% vs 1.4%) also undermines confidence in the experimental pipeline's reproducibility.

---

## 2. Alternative Explanations

### For the 15.96% first-letter absorption rate:

**Alternative: The metric measures L0-dependent false negative rate, not hierarchical absorption.** The confound decomposition shows that at L0=22 (lowest L0 tested), absorption is 42.9%, while at L0=176 it drops to 0.8%. The "improved" experiment uses L0=82 and finds 15.96%. This is consistent with a monotonic relationship between L0 and false negative rate that has nothing specifically to do with hierarchical feature absorption -- it merely reflects that sparser SAEs miss more features. The 98.6% classification of false negatives as "hedging" at L0=22 supports this: nearly all false negatives are L0-induced sparsity artifacts, not hierarchy-specific absorption.

### For the CMI-absorption "correlation" (H3):

**Alternative: Low-d' subspace captures probe quality, not absorption-relevant information.** At d'=10, the top 10 singular vectors of the decoder matrix capture high-variance directions that correlate with probe quality. Letters with poor probes (low F1) have both higher measured "absorption" (because poor probes produce more false negatives) and different CMI characteristics (because their parent features are poorly reconstructed). At d'=20+, additional dimensions dilute this confound, and the correlation disappears or reverses. The correlation is driven by the confound, not by the information-theoretic quantity the theory predicts.

### For the L0_crit scale match (predicted 24.7 vs empirical 22.4):

**Alternative: Tautological fitting.** The phase_transition_prediction pilot explicitly states: "Lambda estimation is circular: Lambda is fit FROM the empirical data (half-max L0), so the scale match is partially by construction." The 10.2% relative error between a fitted parameter and its source data is mathematically expected. The independent test -- binary classification of which letters are absorbed -- gives 36% accuracy, WORSE than the 64% baseline of always predicting "not absorbed." The rank-order correlation rho=0.333 (p=0.103) does not reach significance. The "excellent scale match" is a tautology, not a prediction.

### For the cross-domain absorption rates:

**Alternative: Probe quality floor noise.** City-continent shows 6.5% absorption with mean probe F1=0.795. If 20% of parent feature classifications are wrong (1 - F1), a 6.5% apparent absorption rate is well within the noise floor of a ~20% error rate metric. Moreover, zero domains have `control_credible: true` in the comparative analysis. The measured rates cannot be attributed to genuine absorption rather than metric noise.

### For the L12-65k degenerate result (63.5% absorption, probe F1=0.276):

**Alternative: Dead feature pathology, not absorption.** The setup_verification shows 88.2% dead features in the 16k SAE. The 65k SAE likely has even higher dead feature rates. With probe F1=0.276, the probe is essentially random (F1=0.25 for random 4-class). The 63.5% "absorption" rate is meaningless when the measurement instrument (the probe) has no discriminative power.

---

## 3. Proxy Metric Audit

### Gap 1: Threshold sensitivity grid shows the metric is floor-dominated, not robust

The improved first-letter threshold sensitivity grid (5x4 = 20 cells) shows absorption rates ranging from 0.1480 to 0.1613. The CV of 0.0277 appears to indicate stability, but this is misleading. The range (14.8% to 16.1%) spans only 1.3 percentage points. This "stability" actually means the metric is saturated at a floor: regardless of thresholds, approximately 16% of words are false negatives. When varying cosine threshold 5x (from 0.01 to 0.05) and magnitude gap 4x (from 0.5 to 2.0) produces negligible rate variation, the thresholds are not the binding constraint. The false negatives are driven by something else entirely (likely L0-induced sparsity).

This is not "robust to threshold choice" -- it is "insensitive to the diagnostic criterion," which means the metric is not measuring what the thresholds are intended to test (whether a specific child feature absorbed the parent's signal).

### Gap 2: "Absorption" conflates multiple failure modes

The Chanin metric classifies a false negative as "absorbed" if a child feature fires with cosine > threshold and magnitude gap > threshold. But several other failure modes produce the same signature:
- **Polysemantic features:** A high-activation feature may have high cosine to the probe direction simply because it encodes multiple concepts, one of which happens to include the parent class.
- **Reconstruction error:** If the SAE's reconstruction is poor for certain tokens (especially rare words), the residual error could coincidentally align with the probe direction.
- **Tokenization artifacts:** Words tokenized into multiple tokens may have different activation patterns than single-token words.

The random probe control (11.8% for improved first-letter) shows that even a random direction in d_model space will produce substantial "absorption" rates. This means the metric's specificity is poor -- it flags non-hierarchical false negatives as absorption.

### Gap 3: The confound decomposition classification is circular

A false negative is classified as "hierarchy-driven" (not hedging) based on whether it persists across L0 values. But the 9 persistent false negatives across all 4 L0 values are more likely words that are intrinsically hard for the SAE to represent (rare words, unusual spellings) rather than genuine victims of hierarchical absorption. The classification conflates "hard-to-represent word" with "absorbed by child feature." Without an independent measure of absorption (e.g., intervening on the child feature and checking if the parent recovers), the decomposition is not informative.

### Gap 4: CMI estimation uses inadequate sample size for k-NN MI estimation

The CMI is computed via k-NN estimation (k=5) with corpus tokens in a 10-dimensional subspace. At k=5, the k-NN MI estimator has known high bias (Gao et al., 2015; Kraskov et al., 2004). The standard error scales as O(1/sqrt(n)), and with ~400 tokens per letter (10,000 total / 25 letters), the per-letter CMI estimates have ~5% relative error at best -- comparable to the between-group difference being tested. The fact that the result reverses at d'=20 is consistent with noise-dominated estimates where the signal-to-noise ratio changes with dimensionality.

---

## 4. Severity Classification

### Fatal Flaws (invalidate main claims -- must fix before proceeding)

1. **F1: Universal control failure -- WORSENED in improved experiment.** The shuffled control went from 59.6% to 74.6%. The random probe control is 11.8% (target: <2%). The metric produces higher "absorption" on shuffled labels than on real labels. This invalidates ALL absorption rate claims. H1 (cross-domain) is INVALIDATED.

2. **F2: CMI dimension dependence.** The H3 result exists only at d'=10, does not reach significance (p=0.059 uncorrected, 0.236 Bonferroni-corrected), and reverses sign at all other dimensions. The phase transition "prediction" is tautological (lambda fitted from data). H3 (rate-distortion diagnostic) is NOT SUPPORTED.

### Serious Concerns (weaken claims -- should address in next iteration)

3. **S1: H2 falsified by 69x.** Full-scale decomposition shows 1.4% hierarchy-driven at L0=22, not >80%. The pilot's 96.9% was wrong. The 9 persistent false negatives across all L0 values are likely dataset artifacts, not hierarchy-driven absorption.

4. **S2: Absorption rate varies 3x with vocabulary.** L0=22 shows 42.9% absorption in the multi-L0 decomposition but 13.4% in the original first-letter validation -- same SAE, different word lists. The "improved" experiment at L0=82 shows 15.96%. Absorption rate is not a stable property of the SAE; it is heavily dependent on the evaluation vocabulary and L0 setting. This means absorption rates are not comparable across studies or configurations without exact vocabulary matching.

5. **S3: Bifurcation hypothesis failed.** The original prediction ("JumpReLU = bimodal, L1 = continuous") is directly falsified: both architectures show bimodal distributions. The reframing ("the phase transition IS the bifurcation across L0 values") is post-hoc rationalization. The KS test (D=0.607, p~0) shows distributions differ, but in level rather than shape, which does not support the lateral inhibition mechanism as proposed.

6. **S4: Cross-model confound in bifurcation analysis.** L1 results are from GPT-2 Small (768-dim, 24k SAE) while JumpReLU is from Gemma 2 2B (2304-dim, 16k/65k). Comparing absorption rates across different models (61-67% for L1 GPT-2 vs 1-64% for JumpReLU Gemma) conflates architecture differences with model capacity, training data, and SAE quality differences. This is not a controlled comparison.

### Minor Caveats

7. **M1: N=5 domains for cross-domain correlations.** Hierarchy predictor correlations (H6) computed over 5 data points have confidence intervals spanning [-1, 1]. This is underpowered by design and the results are uninformative regardless of the point estimates.

8. **M2: Unsupervised detection remains negative.** ITAC shows no clear separation. Zero matching pairs for most letters. H4 was expected to fail and is correctly reported as negative.

9. **M3: Geometric constant is mathematically trivial.** For unit-normalized decoders, c = sin^2(angle). With mean cosine 0.17, sin^2 ~ 0.97 everywhere. The "finding" that c provides no additional information beyond CMI is a consequence of the normalization constraint, not an empirical discovery. The CV of 2.16% guarantees near-constant values.

10. **M4: L12-65k results are meaningless.** With probe F1=0.276 and 63.5% "absorption," this configuration is degenerate. Including it in any analysis would be misleading. The L12-65k result should be reported as "probe quality below threshold -- excluded from analysis," not as evidence of high absorption.

---

## 5. Concrete Remediation

### For F1 (Control failure -- HIGHEST PRIORITY):

**Experiment 1: Investigate shuffled control mechanics.**
- For each of the 5 shuffled runs (rates: 73.3%, 71.4%, 75.4%, 75.4%, 77.5%), log per-letter shuffled absorption rates.
- The tight range (71.4-77.5%) across 5 independent shuffles suggests a systematic, not stochastic, phenomenon.
- Hypothesis: With shuffled labels, every word's "parent probe" now points to a random letter direction. But k-sparse probes with k=5 features always produce some high-cosine direction in the 16k-dimensional SAE space. The absorption criterion (cosine > 0.025, magnitude gap > 1.0) is met by coincidence.
- **Test**: Compute the expected shuffled absorption rate analytically from the cosine distribution of SAE features to the random probe directions. If the expected rate matches 74.6%, the metric has poor specificity by construction.

**Experiment 2: Chanin et al. reference implementation comparison.**
- Download sae-spelling from the Chanin et al. repository. Run their exact code on GPT-2 Small with the same control structure. If their shuffled control is <5%, the discrepancy is model-specific (Gemma 2 2B has different SAE feature statistics). If their shuffled control is also high, the issue is metric-inherent.
- Expected outcome: Chanin et al.'s implementation likely operates on a different SAE with different L0 and feature statistics. The metric may be well-calibrated for GPT-2 SAEs but poorly calibrated for Gemma Scope SAEs with JumpReLU activation.

**Experiment 3: Null domain benchmark.**
- Measure "absorption" on a non-hierarchical task: classify words by length (short: 3-4 letters, medium: 5-6, long: 7+) or by frequency bin. If "absorption" rates are >5% on these non-hierarchical tasks, the metric lacks specificity for hierarchical absorption.

### For F2 (CMI dimension dependence):

**Experiment 4: Cross-validated dimension selection.**
- Split 25 letters into discovery set (12 letters, randomly selected) and validation set (13 letters).
- On discovery set, test d' in {5, 10, 15, 20, 25, 30, 40, 50}. Select best d'.
- On validation set, test ONLY at the discovery-optimal d'.
- Report pre-registered: if validation rho < -0.3 at the discovery-optimal d', H3 is supported. Otherwise, it is not.
- Repeat 100 times with random splits and report the distribution of validation rho values.

**Experiment 5: Control for probe quality confound.**
- Compute partial correlation between CMI and absorption rate, controlling for probe F1.
- If the partial correlation is non-significant, the CMI-absorption relationship is mediated by probe quality.
- Also test: correlate probe F1 with CMI at d'=10. If |rho| > 0.3, the confound is confirmed.

**Experiment 6: Increase corpus size.**
- Re-run CMI estimation with 100k tokens (10x current) at all four dimensions.
- k-NN MI estimation quality improves with sample size. If the d'=10 result persists with 100k tokens and the reversal at d'>10 also persists, the dimension dependence is genuine (the theory applies only in a narrow subspace, which limits its usefulness).

### For S1 (H2 falsification):

**Action: Report H2 as falsified and investigate the pilot error.**
- Document exactly what methodological difference between the pilot and full experiment caused the 96.9% vs 1.4% discrepancy. Possible causes:
  - Different definition of "hierarchy-driven" (pilot may have used a simpler criterion)
  - Different L0 value in the pilot (pilot was at L0=22 but with a different vocabulary)
  - Bug in the pilot decomposition code
- The 9 persistent hierarchy-driven words across all L0 values should be examined individually. Are they genuinely absorbed (child feature fires for them) or simply not well-represented in the vocabulary?

### For S2 (Absorption rate instability):

**Action: Report absorption rate as vocabulary-dependent and L0-dependent.**
- The absorption rate is NOT a stable property of the SAE. It varies from 0.8% (L0=176) to 42.9% (L0=22) within the same SAE at different L0 settings, and from 13.4% to 15.96% at L0=82 with different vocabularies.
- The paper must report absorption rates with explicit L0 and vocabulary specification.
- Cross-study comparisons (e.g., "our 15.96% vs Chanin et al.'s 15-35%") are only meaningful if the L0 and vocabulary are comparable.

### For S3 (Bifurcation failure):

**Action: Report the original prediction as falsified; reframe cautiously.**
- The reframing ("phase transition across L0 values is the bifurcation") is acceptable as an observation but should NOT be presented as confirming the lateral inhibition theory. The original, pre-registered prediction (JumpReLU = bimodal, L1 = continuous) failed. The cross-model confound (GPT-2 vs Gemma 2 2B) makes the comparison uninformative for architecture-specific claims.

---

## Summary Verdict

| Hypothesis | Status | Key Evidence |
|-----------|--------|-------------|
| H1 (Cross-domain >= 5%) | **INVALIDATED** | All controls fail; shuffled >> measured in every domain |
| H2 (>80% hierarchy-driven) | **FALSIFIED** | 1.4% at L0=22 (57x below threshold); 69x pilot-full discrepancy |
| H3 (CMI rho < -0.3) | **NOT SUPPORTED** | Only at d'=10, p=0.059 uncorrected (0.236 corrected), reverses at d'>10 |
| H4 (Unsupervised rho > 0.3) | **NEGATIVE** | Zero matching pairs; ITAC no separation |
| H5 (Width-L0 interaction) | **INCOMPLETE** | Scaling surface completed but significance not reported in current summaries |
| H6 (Hierarchy predictors) | **UNDERPOWERED** | N=5 domains; all CIs span [-1, 1] |
| H7 (Bifurcation) | **FALSIFIED then reframed** | Both architectures bimodal; original prediction wrong |

### Salvageable Findings

1. **First-letter absorption on Gemma 2 2B is detectable (15.96% at L0=82).** This replicates Chanin et al. in magnitude, though the control failure means we cannot definitively attribute it to hierarchical absorption vs. metric noise. The fact that three different layer/width configurations (L10-16k: 13.9%, L12-16k: 16.0%, L20-16k: 13.6%) produce consistent rates is mildly reassuring.

2. **The geometric constant degenerates for normalized SAEs.** This is clean and constrains the rate-distortion theory. However, it is mathematically trivial rather than empirically surprising.

3. **The control failure is itself a publishable finding.** "Probe-based absorption metrics fail to transfer from GPT-2 to Gemma 2 2B without recalibration" is a useful methodological contribution. The systematic investigation of WHY the shuffled control exceeds the measured rate could be a genuine contribution to the absorption measurement literature.

4. **The L0-absorption monotonic relationship is robust.** Absorption decreases monotonically with L0 (42.9% -> 37.5% -> 14.4% -> 0.8%). This is informative about SAE sparsity dynamics even if the decomposition into absorption types is unreliable.

5. **The L0-dependent phase transition in JumpReLU SAEs.** While the bifurcation prediction failed, the observation that JumpReLU absorption drops from 36.2% (L0=22) to 1.1% (L0=176) with a dramatic phase transition around L0=82 is a clean empirical finding.

### Recommendation

**Before proceeding to writing, the control failure (F1) must be resolved.** This is a blocking issue. If the shuffled control anomaly persists after thorough investigation (Experiments 1-3 above), the paper's framing must pivot from:
- "Cross-domain absorption characterization with rate-distortion diagnostics"

to:
- "Methodological audit of probe-based absorption metrics: control failure, dimension dependence, and vocabulary sensitivity on Gemma 2 2B"

The second framing is a perfectly publishable paper (methodological critique papers are valued), but it requires honesty about what the experiments actually show rather than what they were designed to show.
