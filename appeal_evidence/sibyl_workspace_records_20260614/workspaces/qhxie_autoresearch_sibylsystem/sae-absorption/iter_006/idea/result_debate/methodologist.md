# Methodologist Analysis: Cross-Domain Absorption Characterization with Rate-Distortion Diagnostics

## Executive Summary

This methodology audit identifies **three critical validity threats** and **two moderate concerns** that collectively undermine the primary claims of the study. The most serious issue is that **no domain passes control checks** -- shuffled and random controls exceed measured absorption rates across all five domains -- making the central empirical claims about absorption rates unreliable as currently constituted. The rate-distortion theoretical contribution (CMI-absorption correlation) shows a signal at one subspace dimension (rho = -0.383, p = 0.059 at d'=10) but the correlation **reverses sign** at all other tested dimensions (d'=20, 30, 50), raising serious concerns about estimator fragility versus genuine signal. The confound decomposition is methodologically sound at L0=22 but its interpretation shifts dramatically depending on definition, with the pilot's "96.9% hierarchy-driven" headline contradicted by the full experiment under a different but equally defensible classification scheme.

**Bottom line**: The study has an excellent structural design (5 domains, multi-L0 sweeps, pre-registered gates, honest negative result reporting), but the execution reveals three problems that must be resolved before the main claims can be credibly advanced: (1) control failure across all domains, (2) CMI instability across subspace dimensions, (3) inconsistency between pilot and full decomposition results.

---

## 1. Baseline Fairness Audit

### 1.1 First-Letter Baseline vs. Published Numbers

The first-letter absorption rate of 13.4% (CI: 7.2-18.1%) is compared against the published range of 15-35% from Chanin et al. (2024). This comparison has **several asymmetries**:

- **Model difference**: Chanin et al. used GPT-2 Small; this study uses Gemma 2 2B. Different model architectures, training data, and scales make direct rate comparisons imprecise. The slightly-below-range rate (13.4% vs. 15%) may reflect a genuine model difference rather than a methodological deficiency.
- **SAE architecture difference**: Chanin et al. used L1-trained SAEs; this study uses Gemma Scope JumpReLU SAEs. The bifurcation analysis (H7) explicitly predicts these should behave differently, yet the baseline comparison implicitly assumes comparability.
- **SAE operating point inconsistency**: The primary first-letter experiment (`first_letter_improved`) uses `layer_12/width_16k/average_l0_82`, while the confound decomposition sweeps L0={22, 41, 82, 176}. At L0=22, probes achieve F1=1.0 across all 25 letters and the absorption rate jumps to **42.9%** (CI: 40.1-45.6%). At L0=82, mean probe F1 drops to 0.565 and only 4/25 letters pass the F1 > 0.85 gate. These are effectively measuring different phenomena on different SAE configurations, complicating direct comparison and raising the question of which operating point represents the "true" absorption rate.

**Verdict**: The first-letter baseline is reasonable as an approximate consistency check. However, the SAE operating point used for the primary claim (L0=82) produces poor probe quality, while the operating point with good probes (L0=22) produces a very different absorption rate. This internal inconsistency needs explicit discussion.

### 1.2 Cross-Domain Baselines

The five domains have dramatically asymmetric characteristics:

| Domain | N Children | N Parents Above F1>0.85 Gate | Mean Probe F1 | Per-Parent Samples (mean) |
|--------|-----------|------------------------------|---------------|--------------------------|
| First-letter | 576 | 4 of 25 (16%) | 0.565 | 23 |
| City-country | 184 | 9 of 28 (32%) | 0.602 | 6.5 |
| City-continent | 185 | 4 of 6 (67%) | 0.795 | 30.8 |
| City-language | 183 | 11 of 18 (61%) | 0.745 | 10.2 |
| Animal-class | 140 | 4 of 6 (67%) | 0.696 | 23.3 |

**Critical asymmetry**: Only 4/25 first-letter parents pass the F1 > 0.85 quality gate, yet the aggregate absorption rate of 13.4% is reported across *all* parents. The gated-only absorption rate (`pct_parents_with_absorption_gated`) is **0.0%** for first-letter. This means the entire 13.4% headline number derives from parents whose probes are too poor to reliably distinguish absorption from noise. City-continent (50% gated parents show absorption) and city-language (9.1% gated) fare slightly better, but the point stands: the quality gate, which was designed to ensure reliable measurement, effectively nullifies the primary empirical claim when applied as intended.

### 1.3 Control Quality -- THE CRITICAL THREAT

**This is the most serious methodological concern in the study.**

All five domains have `control_credible: false`. Specific failures:

| Domain | Measured Absorption | Shuffled Control | Random Control | Net Signal |
|--------|-------------------|-----------------|----------------|------------|
| First-letter | 13.4% | 59.6% | 9.2% | **-46.2%** |
| City-country | 0.0% | 10.3% | 19.0% | **-10.3%** |
| City-continent | 6.5% | 45.2% | 12.9% | **-38.7%** |
| City-language | 6.6% | 18.0% | 20.8% | **-11.5%** |
| Animal-class | 1.4% | 39.3% | 34.3% | **-37.9%** |

The net signal (measured minus shuffled) is **negative for all domains**. A well-calibrated absorption metric should produce near-zero absorption under shuffled labels, because destroying the true parent-child relationship should eliminate absorption. The shuffled control producing *higher* absorption rates across the board indicates one or more of:

1. **Probe quality confound**: When labels are shuffled, the k-sparse probe still identifies some features, but these features' firing patterns produce spurious matches with the cosine/magnitude criteria at a higher rate than the true letter features. This would mean low-F1 probes contaminate the metric.
2. **Split-feature selection artifact**: The k-sparse probe may identify features that happen to have high decoder cosine similarity with many other features, independent of the actual letter relationship. Shuffled labels would then produce even more "absorption" because the random probe directions have even less specificity.
3. **Implementation error**: The shuffled control may not retrain probes from scratch, or may reuse split features identified from the true-label experiment.

The methodology document prescribes "investigate control implementation before proceeding" as a decision gate after first-letter, but the experiments appear to have proceeded through all stages without resolving this. **No absorption rate claim in this study can be considered calibrated until the control failure is diagnosed and resolved.**

---

## 2. Metric-Claim Alignment

### 2.1 Claims Mapped to Metrics

| Claim | Metric | Alignment Assessment |
|-------|--------|---------------------|
| Cross-domain absorption exists (H1) | Absorption rate >= 5% in 2+ domains | **MISALIGNED**: Metric not calibrated (controls fail across all domains); quality-gated rate is 0% for first-letter |
| Absorption is mostly hierarchy-driven (H2) | % hierarchy-driven false negatives at L0=22 | **PARTIALLY ALIGNED**: But definition is L0-sweep-dependent; pilot/full inconsistency |
| CMI predicts absorption (H3) | Spearman rho(CMI, absorption) < -0.3 | **ALIGNED but fragile**: Met at d'=10 only; sign reverses at d'=20, 30, 50 |
| Unsupervised detection works (H4) | Spearman rho > 0.3 | **ALIGNED**: Correctly reported as negative result |
| Width-L0 interaction (H5) | GAM p-value | **ALIGNED**: Appropriate statistical framework |
| Hierarchy predictors (H6) | Spearman rho > 0.3, Bonferroni-corrected | **UNDERPOWERED**: n=5 domains; bootstrap CIs span [-1, 1]; partial correlations degenerate |

### 2.2 Confound Decomposition: Definitional Dependence

The pilot reported 96.9% hierarchy-driven absorption at L0=22. The full multi-L0 decomposition tells a different story:

| L0 | Total FN | N Absorbed | Absorption Rate | Hierarchy-Driven Fraction |
|----|----------|-----------|-----------------|--------------------------|
| 22 | 657 | 512 | 42.9% | Low (most tokens recoverable at higher L0) |
| 41 | 489 | 448 | 37.5% | Moderate |
| 82 | ~similar | ~similar | ~similar | Higher |
| 176 | ~10 | ~9 | very low | ~90% (only persistent FNs remain) |

The classification depends on which other L0 values are included in the sweep:

- At L0=22 (most sparse), nearly all false negatives are classified as "hedging" because they recover at higher L0 (41, 82, or 176). This is reasonable but changes the interpretation: absorption at L0=22 is severe but mostly *reversible* by relaxing the sparsity constraint.
- At L0=176 (least sparse), only ~10 false negatives remain, and ~9 are "hierarchy-driven" (persistent across all tested L0 values). This is also reasonable but has very low statistical power (n=10).

**The key methodological issue**: If L0=300 were added and those 9 persistent tokens recovered, they would be reclassified as "hedging." The decomposition categories are not intrinsic properties of tokens but are contingent on the L0 sweep range. The pilot's "96.9% hierarchy-driven" used a different classification approach (likely comparing only to the L0=22 operating point without multi-L0 cross-referencing), producing an inconsistent headline number.

This is not necessarily fatal -- the multi-L0 profile itself is informative and novel. But the framing must be corrected: the decomposition shows how false-negative *composition changes with L0*, not a stable partition of tokens into fixed categories.

### 2.3 CMI-Absorption Correlation: Dimension Instability

The headline CMI result is the strongest individual piece of evidence for the rate-distortion theory:

| Subspace d' | Spearman rho | p-value | Cohen's d | Mann-Whitney p |
|-------------|-------------|---------|-----------|----------------|
| 10 | **-0.383** | 0.059 | **-0.924** | **0.045** |
| 20 | +0.048 | 0.818 | +0.226 | 0.548 |
| 30 | +0.299 | 0.147 | +0.616 | 0.182 |
| 50 | +0.197 | 0.345 | +0.499 | 0.285 |

**The sign flip is the central methodological concern for the theoretical contribution.** At d'=10, absorbed letters have significantly lower CMI than non-absorbed letters (mean 0.649 vs 0.861, Cohen's d = -0.924, p = 0.045). This is a large effect. But at d'=20 and above, the difference vanishes or reverses.

If CMI truly captured the information-theoretic quantity I(X; w_parent | f_child), the correlation should be directionally stable across reasonable subspace projections, with possible attenuation at very high dimensions (curse of dimensionality for k-NN estimation). Instead, the reversal at d'=20 suggests either:

1. The k-NN estimator (Kraskov et al., with k=5) is unreliable at this sample size (n=3000 per letter) for d' > 10
2. The subspace projection at d'=10 happens to align with the discriminative direction by coincidence
3. The true effect is concentrated in a low-dimensional subspace (consistent with theory but needs principled argument)

Selecting d'=10 as "best" and reporting rho=-0.383 as the primary result constitutes **post-hoc optimization over estimator hyperparameters**. The study tests 4 dimension values and reports the best one, which is analogous to running 4 analyses and reporting the most favorable. Without a pre-registered choice of d' or a principled cross-validation procedure, this is a form of researcher degrees of freedom that inflates the apparent evidence.

---

## 3. Validity Threats Checklist

### Data Leakage
- [x] **Low risk**: Pre-trained model and SAEs with no fine-tuning. First-letter words are common English words naturally in pretraining data, which is by design. No test-set leakage.

### Contamination
- [x] **Low risk**: The study does not benchmark model capabilities. Absorption is a property of SAE representations. Benchmark contamination is irrelevant.

### Selection Bias / Threshold Sensitivity
- [ ] **HIGH RISK -- Threshold sensitivity**: The absorption metric uses hard thresholds (cosine > 0.025, magnitude gap >= 1.0) from Chanin et al., calibrated for GPT-2 Small. The threshold sensitivity ablation was performed but with only 577 words (25 per letter) on the same SAE (L0=82), where most probes have F1 < 0.85. It does not test sensitivity for the cross-domain experiments, which is where threshold effects could be most consequential.
- [ ] **HIGH RISK -- Absorbed/non-absorbed partition**: The CMI analysis uses absorption_rate > 0.10 to define the "absorbed" group and < 0.05 for "non-absorbed." Letters between 0.05 and 0.10 are excluded as "ambiguous." Changing these thresholds (e.g., to 0.05/0.02) would alter which letters fall in which group, potentially changing the Mann-Whitney test result (p=0.045, currently barely significant). The sensitivity to this binary partition threshold is not tested.
- [ ] **MODERATE RISK -- SAE operating point selection**: First-letter results are reported at L0=82, but the confound decomposition uses L0=22 where probes are perfect. The choice of operating point substantially changes absorption rates (13.4% at L0=82 vs 42.9% at L0=22 vs ~0% at L0=176 with vanishingly few FNs).

### Overfitting to Single Model / SAE Family
- [ ] **MODERATE RISK**: All absorption measurements use Gemma 2 2B + Gemma Scope SAEs. Claims about "domain-dependent absorption rates" generalize from a single model-SAE combination. The bifurcation analysis attempted cross-architecture comparison but used GPT-2 Small L1 SAEs, introducing model size (124M vs 2B) and architecture (GPT-2 vs Gemma) as confounds inseparable from L1 vs JumpReLU.

### Multiple Comparisons
- [x] **Partially addressed**: FDR correction (Benjamini-Hochberg) and Bonferroni correction are applied for the hierarchy predictor analysis. Bootstrap CIs use 10,000 resamples with seed=42. However, the implicit search over d'={10, 20, 30, 50} for CMI estimation constitutes an unacknowledged multiple comparison.

---

## 4. Ablation Gap Analysis

| Component | Ablation Present? | Quality Assessment |
|-----------|------------------|-------------------|
| Cosine threshold | Yes (5 values: 0.01-0.05) | Adequate |
| Magnitude gap threshold | Yes (4 values: 0.5-2.0) | Adequate |
| Probe sparsity k | Yes (k=1, 3, 5, 10) | Adequate |
| SAE L0 operating point | Yes (L0=22, 41, 82, 176 in decomposition) | Adequate |
| SAE width | Yes (16k, 32k, 65k, 131k in scaling surface) | Adequate |
| SAE layer | Yes (L10, L12, L20 in scaling surface) | Adequate |
| CMI subspace dimension | Tested but not properly ablated | **PROBLEMATIC**: 4 values tested, best selected post-hoc |
| **K-NN k for CMI** | **No** -- fixed at k=5 | **MISSING**: k-NN MI estimation is known to be sensitive to k |
| **Word vocabulary** | **No** -- same word list throughout | **MISSING**: No bootstrap or alternative-vocabulary test |
| **Prompt format** | **No** -- single ICL prompt format | **MISSING**: No test of prompt sensitivity |
| **Corpus size for CMI** | **No** -- fixed at n=3000 per letter | **MISSING**: Convergence not assessed |
| **Absorbed/non-absorbed partition threshold** | **No** -- fixed at 0.10/0.05 | **MISSING**: Sensitivity to this classification threshold not tested |

### Missing Ablations That Could Change Conclusions

1. **CMI k-NN hyperparameter (k)**: The Kraskov estimator with k=5 at d'=10 with n=3000 samples is operating in a regime where the estimator may be unreliable. Testing k={3, 5, 7, 10, 20} would reveal whether rho=-0.383 is robust to estimator settings or an artifact.

2. **CMI sample size convergence**: With only 3000 samples per letter, the k-NN MI estimator may not have converged. A convergence check (computing CMI at n={1000, 2000, 3000, 5000, 10000}) would assess stability.

3. **Vocabulary sensitivity**: The word list contains highly variable counts per letter (X: 1 word, S: 70 words). Letters with very few words (X=1, Z=7-9, Y=8-12) may have unreliable absorption rates. A bootstrap vocabulary sensitivity test would quantify this effect.

4. **Cross-domain threshold sensitivity**: The 5x4 threshold grid was only run on first-letter. Running it on at least city-continent and city-language would show whether the ~6.5% cross-domain rates are threshold-robust or threshold-artifacts.

---

## 5. Reproducibility Score: 3/5

| Criterion | Score | Notes |
|-----------|-------|-------|
| Random seeds fixed | 5/5 | Seed=42 used consistently across all experiments |
| Hyperparameters specified | 4/5 | Most specified; CMI k-NN k=5 stated but not justified; absorbed/non-absorbed thresholds stated |
| Code/data available | 3/5 | Code in exp/code/; data from HuggingFace (Gemma Scope, RAVEL). Word lists embedded in code but not separately versioned |
| Hardware documented | 2/5 | Methodology states "single GPU >= 24GB" but actual hardware, VRAM per experiment, and wall-clock times not systematically reported |
| Reproducible within 10% | 2/5 | **Uncertain**: CMI correlation (rho=-0.383, p=0.059) is near the significance threshold -- small changes in vocabulary, seed, or corpus could cross it. The control failure would reproduce but its diagnosis is unresolved. The pilot-to-full decomposition inconsistency (96.9% vs dependent-on-definition) suggests implementation details are load-bearing. |

**Specific reproducibility concerns**:
- The CMI result at d'=10 sits at p=0.059. A reproducer changing the vocabulary or corpus sample would likely obtain a different p-value, potentially crossing or falling well short of 0.05.
- The confound decomposition rates at L0=22 (42.9% absorption rate, with probe F1=1.0) would reproduce on the same SAE, but the "hierarchy-driven" classification depends on which other L0 values are in the sweep.
- The control failure (shuffled > measured) should reproduce but its root cause is unknown, making the interpretation of any successful reproduction ambiguous.

---

## 6. Top-3 Methodology Improvements (Ordered by Effort-to-Credibility Ratio)

### Recommendation 1: Diagnose and Resolve the Control Failure (CRITICAL -- Prerequisite)

**Effort**: Medium (2-3 days of systematic debugging)
**Credibility gain**: Maximal -- currently no absorption claim is calibrated

The shuffled control producing higher absorption than true labels across all five domains is an existential threat to every empirical claim. The study cannot credibly claim "absorption exists" when a random baseline produces more "absorption." Concrete steps:

1. **Verify shuffled control implementation**: Confirm labels are truly randomly permuted, probes are retrained from scratch on shuffled labels, and split features are re-identified (not reused from the true-label experiment). The most likely bug is reusing split features identified from true labels.
2. **Run a "null SAE" control**: Use a randomly initialized (untrained) SAE. If it also shows high "absorption," the metric itself is flawed for this SAE architecture.
3. **Test the no-probe control**: Measure absorption using random probe directions (not trained). If random probes also produce high absorption, the cosine/magnitude criteria are too permissive for Gemma Scope decoder geometry.
4. **Consider a calibrated net metric**: Report absorption_rate - E[absorption_rate | shuffled labels] as the net signal. If this is consistently negative, the study should pivot to reporting that "the Chanin metric does not transfer to Gemma 2 2B / JumpReLU SAEs" -- itself a valuable methodological finding.
5. **Investigate whether the issue is operating-point-specific**: The confound decomposition at L0=22 with F1=1.0 probes might have calibrated controls. Run shuffled controls at L0=22 to check.

### Recommendation 2: Stabilize the CMI Correlation or Bound Its Dimensionality (HIGH PRIORITY)

**Effort**: Low-medium (1-2 days)
**Credibility gain**: High -- transforms the rate-distortion contribution from cherry-picked to robust

The sign flip from rho=-0.383 at d'=10 to rho=+0.048 at d'=20 is the most vulnerable point for the theoretical contribution. A reviewer will immediately ask "why does the correlation reverse?" Concrete options:

1. **Provide a principled argument for d'=10**: Compute the effective dimensionality of the decoder-direction subspace (e.g., via the eigenspectrum of the covariance matrix projected onto decoder directions). If the effective dimensionality is approximately 10, then d'=10 is the correct choice and d'>10 adds noise that degrades the estimate. This would transform the dimension selection from post-hoc to theory-motivated.
2. **Use a dimension-free MI estimator**: MINE, KNIFE, or sliced MI estimators avoid the subspace projection entirely. If these produce a consistent negative correlation, the d'=10 result is validated.
3. **Cross-validate the dimension**: Split the 25 letters into training (15) and test (10) sets. Select d' on training set, evaluate on test set. Repeat across splits. Report the mean rho and CI across splits.
4. **At minimum, report transparently**: If none of the above fully resolves the instability, clearly state: "The CMI-absorption correlation is negative and significant only at d'=10 (rho=-0.383, Mann-Whitney p=0.045, Cohen's d=-0.924). At higher subspace dimensions, the correlation attenuates and reverses, possibly due to k-NN estimator degradation in higher dimensions." Do not claim "CMI predicts absorption" without this qualification.

### Recommendation 3: Reframe the Confound Decomposition Around L0 Profiles (MODERATE PRIORITY)

**Effort**: Low (0.5-1 day of reframing)
**Credibility gain**: Moderate -- resolves pilot/full inconsistency and makes the contribution clearer

The confound decomposition is actually a strong contribution, but the framing is confused by the pilot-to-full inconsistency. Fix by:

1. **Anchor the headline on L0=22 with F1=1.0 probes**: At this operating point, absorption rate is 42.9% and probes are perfectly calibrated. This is the most reliable measurement point. Frame the result as: "At the SAE operating point with maximal probe quality (L0=22, F1=1.0), 42.9% of tokens exhibit absorption."
2. **Frame the multi-L0 profile as the contribution**: The novel finding is not "X% is hierarchy-driven" but rather "the fraction of false negatives attributable to hedging vs. persistent absorption changes systematically across L0, with hedging dominant at low L0 and persistent absorption dominant at high L0." This L0-profile is intrinsically interesting and does not depend on arbitrary category definitions.
3. **Retire the "96.9% hierarchy-driven" headline**: Acknowledge that the pilot used a different methodology and that the full experiment reveals a more nuanced picture. The honest framing strengthens credibility.

---

## Supplementary Observations

### On the Bifurcation Analysis (H7)

The prediction was JumpReLU SAEs show bimodal absorption distributions while L1 SAEs show continuous distributions. Result: both show bimodal distributions (`bimodal_by_bc` for all tested SAEs). Status: "MIXED." The critical confound is that L1 comparison used GPT-2 Small SAEs (124M parameter model, different architecture, different training data), not Gemma 2 2B L1 SAEs. Model difference and SAE architecture difference are completely confounded. This comparison is **not interpretable** for the intended JumpReLU-vs-L1 hypothesis.

The study could salvage this by: (a) training a small custom L1 SAE on Gemma 2 2B activations (methodology document suggests this as a possibility), or (b) clearly framing the GPT-2 comparison as a pilot with major caveats. Do not claim H7 is tested.

### On the Phase Transition Prediction

Binary classification accuracy is 36% vs. 64% chance level (worse than random). The rank-order correlation (rho=0.333, p=0.103) is theory-consistent in direction but not significant. The absorbed-vs-non-absorbed Mann-Whitney test (p=0.042) is the strongest evidence, but this is the same test as the CMI analysis -- it provides no independent validation of the phase transition prediction. The "predicted L0_crit" (mean 24.7, range 13.7-42.1) is in the plausible range but the wide spread and low classification accuracy suggest the theory's quantitative predictions are too imprecise to be validated at this sample size.

### On the Hierarchy Predictor Analysis (H6)

With n=5 domains, bootstrap CIs for all Spearman correlations span [-1.0, 1.0]. Partial correlations show degenerate behavior (r=1.0 for some predictors), likely because n=5 is too small for partial correlation with 2+ controls. This analysis provides no meaningful statistical inference and should be reframed as purely descriptive. The point estimates (co-occurrence ratio rho=0.4, depth rho=-0.577) are directionally interesting but any quantitative interpretation at n=5 is inappropriate.

### On the Unsupervised Pipeline (H4) -- A Methodological Strength

The honest reporting of ITAC's failure (no significant separation between candidate and random pairs, MannWhitney not significant, per-letter matching scores all zero) is a genuine methodological strength. The pre-registered decision gate (rho < 0.3 = negative result) was applied correctly. This transparency should be highlighted in the paper.

### On the Scaling Surface (H5)

The scaling surface analysis measured absorption across 34 first-letter SAE configurations and 5 cross-domain conditions. The cross-domain SAEs at different operating points show that many SAE configs produce zero measurable absorption (all continents below probe gate at L0=22). This is informative but limits the statistical power for modeling the width-L0 interaction surface. The GAM analysis was planned but it is unclear from the results whether it was successfully fit with the architecture covariate.

### Probe Quality as a Systematic Confound

A pattern emerges across the study: probe quality (F1) systematically correlates with measured absorption rates. Letters/domains with low F1 show high measured absorption; those with high F1 show low or zero absorption. This is consistent with absorption being a real phenomenon (it suppresses the features probes rely on), but also consistent with low probe quality inflating the false-negative rate and hence the "absorption" metric. Disentangling these requires the L0=22 operating point (where F1=1.0 for all letters), which is why Recommendation 3 suggests anchoring the headline results there.

---

## Summary Assessment

| Aspect | Rating | Key Evidence |
|--------|--------|-------------|
| Study design | **Good** | 5 domains, multi-L0 sweeps, pre-registered decision gates, multiple SAE configs |
| Control implementation | **Poor** | All domains: shuffled > measured; no control is credible; investigation prescribed but not completed |
| Statistical methodology | **Mixed** | FDR/Bonferroni corrections proper; bootstrap CIs adequate; but n=5 for cross-domain correlations is severely underpowered; d' selection for CMI is unacknowledged multiple comparison |
| Theoretical validation (CMI) | **Fragile** | Strong effect at d'=10 (Cohen's d=-0.924, p=0.045) but sign-reverses at d'=20+; not robust across estimator settings |
| Confound decomposition | **Informative but mislabeled** | L0 profiles are genuinely novel; but pilot/full inconsistency and sweep-dependent definitions weaken the headline claims |
| Reproducibility | **Moderate** (3/5) | Seeds fixed; key results near significance thresholds; control cause undiagnosed |
| Negative result reporting | **Strong** | Honest about ITAC failure, within-width null, H4 falsification |
| Overall | The study has excellent structural design with rigorous pre-registration, but three unresolved issues (control failure, CMI instability, decomposition framing) must be addressed before the primary claims can withstand peer review. The L0=22 operating point with perfect probes is the study's strongest empirical ground. |
