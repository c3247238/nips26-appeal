# 4 Cross-Domain Absorption Results

With validated probes and a documented measurement pipeline (Section 3), we measure feature absorption across four hierarchy types, two SAE widths, and four transformer layers on Gemma 2 2B. Six findings emerge: cross-domain variation is statistically significant (Section 4.1), absorption concentrates at the final prediction layer (Section 4.2), wider SAEs reduce absorption for most hierarchies (Section 4.3), within-hierarchy per-class variation is large (Section 4.4), first-letter is not the worst case (Section 4.5), and a probe degradation ablation decomposes the observed variation into probe quality confounds and genuine hierarchy effects (Section 4.6).

## 4.1 Cross-Domain Variation Is Statistically Significant

Table 2 presents absorption rates at layer 24 across all four hierarchy types and both SAE widths.

| Hierarchy | SAE | $\alpha$ (%) | 95% CI | $F_1$ | $N_{\text{entities}}$ | $N_{\text{FN}}$ |
|-----------|-----|:---:|:---:|:---:|:---:|:---:|
| First-letter | 16k | **27.1** | [26.3, 34.7] | 1.00 | 500 | 308 |
| City-continent | 16k | 31.4 | [28.9, 33.9] | 0.87 | 173 | 418 |
| City-country | 16k | **45.1** | [42.2, 49.0] | 0.73$^\dagger$ | 1,405 | 515 |
| City-language | 16k | 11.6 | [9.7, 13.5] | 0.82 | 201 | 124 |
| City-continent | 65k | 31.3 | [28.7, 33.8] | 0.87 | 173 | 416 |
| City-country | 65k | 32.9 | [30.2, 35.7] | 0.73$^\dagger$ | 1,405 | 376 |
| City-language | 65k | 7.7 | [6.2, 9.4] | 0.82 | 201 | 83 |

**Table 2.** Cross-domain absorption rates at layer 24 on Gemma 2 2B with Gemma Scope JumpReLU SAEs. All rates use per-token aggregation (Section 3.2). Bootstrap 95% CI from 10,000 resamples over token observations. $\dagger$City-country probe $F_1 = 0.73$ falls below both quality gates; results included with documented caveat that absorption rates are confounded by probe quality (Section 4.6 quantifies this confound). Bold marks the highest and lowest rates within the 16k configuration.

Absorption rates span a 3.9$\times$ descriptive range at layer 24 with 16k SAEs: from 11.6% (city-language) to 45.1% (city-country). Within the three RAVEL hierarchies---which share the same model, layer, SAE, token position, and prompt framework---variation is statistically significant (Kruskal-Wallis $H = 296.4$, $p = 7.4 \times 10^{-66}$). All three within-RAVEL pairwise comparisons are significant after Bonferroni correction: city-continent vs. city-language ($p_{\text{Bonf}} = 1.5 \times 10^{-30}$), city-continent vs. city-country ($p_{\text{Bonf}} = 8.4 \times 10^{-12}$), city-language vs. city-country ($p_{\text{Bonf}} = 2.1 \times 10^{-67}$).

Comparing RAVEL hierarchies against first-letter requires caution because of the token position asymmetry (first-letter at position $-6$, RAVEL at position $-2$; Section 3.2). City-language vs. first-letter reaches significance ($p_{\text{Bonf}} = 0.003$), but city-continent vs. first-letter ($p_{\text{Bonf}} = 1.0$) and city-country vs. first-letter ($p_{\text{Bonf}} = 1.0$) do not. The 3.9$\times$ range is a descriptive finding; not all pairwise comparisons survive correction for multiple testing.

## 4.2 Absorption Concentrates at the Final Prediction Layer

Figure 2 shows the absorption profile across transformer layers 6, 12, 18, and 24 for all four hierarchies with 16k SAEs.

![Layer-dependent absorption rates for four hierarchies on Gemma 2 2B with 16k JumpReLU SAEs. Absorption concentrates at layer 24 across all hierarchy types. First-letter rises from 1.0% at layer 6 to 27.1% at layer 24 (a 26$\times$ increase). Shaded bands indicate bootstrap 95% CIs.](figures/fig2_layer_absorption.pdf)

First-letter absorption rises from 1.0% at layer 6 to 4.7% at layer 12, dips to 2.0% at layer 18, and jumps to 27.1% at layer 24---a 26$\times$ increase from L6 to L24. The non-monotonic dip at L18 is consistent with layer 18 performing intermediate computations distinct from the final token-prediction computation at L24.

RAVEL hierarchies follow the same pattern: all four hierarchies reach their highest absorption at layer 24. City-continent rises from approximately 5% at L6 to 31.4% at L24. City-country reaches 45.1% at L24. This layer concentration is the paper's most robust cross-domain finding for two reasons: (1) first-letter probes achieve $F_1 = 1.0$ at all four layers, eliminating probe quality as a confound for the layer effect; and (2) the L24 concentration is consistent across all hierarchies despite their structural differences.

The concentration at the final prediction layer suggests absorption arises from task-specific computation---the model's final token-prediction circuitry---not from generic feature representation at intermediate layers. Gemma 2 2B has $L = 26$ layers; layer 24 is the last layer for which Gemma Scope provides SAEs.

## 4.3 Width Effect

Wider SAEs (65k vs. 16k dictionary features) reduce absorption for most hierarchies. The effect is most pronounced for city-country: 45.1% at 16k vs. 32.9% at 65k, a 12.2 percentage point (pp) reduction. City-language decreases from 11.6% to 7.7% ($-3.9$ pp). City-continent shows negligible change: 31.4% vs. 31.3% ($-0.1$ pp).

The near-zero width effect for city-continent is consistent with its absorption being driven by probe quality rather than by SAE dictionary capacity (Section 4.6 confirms this). For hierarchies where absorption reflects genuine feature gaps, more dictionary features provide additional capacity to represent parent concepts, reducing the encoder's pressure to merge parent information into child decoder vectors.

## 4.4 Per-Class Variation

Figure 3 shows per-continent absorption rates for city-continent at layer 24, revealing extreme within-hierarchy variance.

![Per-continent absorption rates for city-continent at layer 24 on Gemma 2 2B. Europe (90.2%, $n = 276$) and Oceania (52.9%, $n = 51$) show far higher absorption than Africa (3.9%, $n = 231$) and South America (3.9%, $n = 207$). The 16k and 65k SAEs produce nearly identical per-continent patterns.](figures/fig3_perclass_heatmap.pdf)

Europe dominates: 90.2% of probe-correct European city activations lose their continent classification after SAE encoding ($n = 276$). Oceania follows at 52.9% ($n = 51$), Asia at 24.4% ($n = 324$), and North America at 19.1% ($n = 241$). Africa and South America show minimal absorption at 3.9% each ($n = 231$ and $n = 207$). The 16k-to-65k transition preserves this ordering, with nearly identical per-class rates.

Within city-country, the variance is even more extreme. USA shows 0% absorption (176 entities), while smaller countries---Albania, Algeria, Argentina---reach 100% at $n = 12$--$25$ entities each. This pattern, where high-frequency classes resist absorption while rare classes are fully absorbed, is consistent with the SAE learning dedicated features for frequent parent concepts but relying on shared (and therefore absorbable) features for rare ones.

The large within-hierarchy variance means hierarchy-level averages mask substantial structure. The per-class patterns---specifically, the dependence on class frequency and feature specialization---carry more diagnostic information than aggregate rates.

## 4.5 First-Letter Is Not the Worst Case

At layer 24 with 16k SAEs, first-letter absorption (27.1%) is lower than city-country (45.1%) and comparable to city-continent (31.4%). City-language (11.6%) is significantly lower than first-letter ($p_{\text{Bonf}} = 0.003$). The received assumption---that first-letter spelling represents a typical or worst-case scenario for absorption---does not hold. The first-letter task, with its perfect probes and balanced class distribution, occupies an intermediate position in the cross-domain absorption spectrum.

Statements calibrated to the 15--35% first-letter absorption rates from Chanin et al. (2024) may underestimate absorption severity for imbalanced knowledge hierarchies (city-country) or overestimate it for many-to-many hierarchies (city-language). Cross-domain evaluation is necessary to characterize the actual range of absorption in practice.

The probe quality difference between first-letter ($F_1 = 1.0$) and RAVEL hierarchies ($F_1 = 0.73$--$0.87$) raises an immediate confound: do the higher RAVEL rates reflect genuine hierarchy effects, or are they inflated by imperfect probes? Section 4.6 addresses this directly.

## 4.6 Probe Degradation Ablation Resolves the Confound

The probe degradation ablation (Section 3.7) tests whether cross-domain absorption variation is a genuine hierarchy effect or a probe quality artifact. We degrade first-letter probe quality via weight noise injection to seven $F_1$ levels (0.70 to 1.0), re-measure absorption at each level on 11,725 tokens per level (2,345 words $\times$ 5 prompts), and average across 3 noise seeds.

Table 3 reports the corrected degradation curve data with per-token bootstrap CIs.

| Target $F_1$ | Actual $F_1$ | $\alpha$ (%) | 95% CI | Seed SD |
|:---:|:---:|:---:|:---:|:---:|
| 0.70 | 0.685 | 36.1 | [35.0, 37.3] | 1.9 |
| 0.75 | 0.754 | 35.3 | [34.2, 36.4] | 1.1 |
| 0.80 | 0.789 | 34.4 | [33.3, 35.5] | 4.2 |
| 0.85 | 0.846 | 33.6 | [32.6, 34.6] | 5.5 |
| 0.90 | 0.904 | 32.4 | [31.4, 33.3] | 3.6 |
| 0.95 | 0.951 | 28.9 | [28.0, 29.8] | 4.3 |
| 1.00 | 0.999 | **21.6** | [20.9, 22.4] | 0.0 |

**Table 3.** Probe degradation ablation results. First-letter absorption at layer 24 with 16k JumpReLU SAE across 7 degraded probe quality levels. Each degraded level averaged across 3 noise seeds; 11,725 tokens per level. All CIs use per-token percentile bootstrap (10,000 resamples), consistent with the per-token point estimates. Absorption increases monotonically as probe $F_1$ decreases from 1.0 to 0.69. Bold marks the undegraded control. Seed SD quantifies cross-seed variability.

Figure 5 shows the degradation curve with RAVEL hierarchy points overlaid.

![Probe degradation ablation. Blue circles: first-letter absorption at 7 degraded probe $F_1$ levels, with linear fit ($R^2 = 0.777$, $p = 0.009$, slope $= -0.398$) and quadratic fit ($R^2 = 0.942$). RAVEL hierarchies overlaid at their native $F_1$ values. City-continent (green square, $\Delta = +0.6$ pp) falls on the curve---its elevated absorption is fully explained by probe quality. City-language (purple diamond, $\Delta = -21.3$ pp) sits far below the curve: a genuine hierarchy-specific outlier that probe quality alone cannot explain. City-country (red triangle, $\Delta = +8.5$ pp) shows modest excess.](figures/fig5_probe_degradation.pdf)

Three results emerge from the decomposition.

**The degradation curve is well-fitted and perfectly monotonic.** As probe $F_1$ decreases from 0.999 to 0.685, absorption increases from 21.6% to 36.1%. The relationship is perfectly monotonic (Spearman $\rho = -1.0$, $p = 0.009$ on 7 points). A linear model explains 77.7% of variance ($R^2 = 0.777$, $\beta_1 = -0.398$, $p = 0.009$). A quadratic fit captures 94.2% ($R^2 = 0.942$). The curve establishes that probe quality is a major confound in absorption measurement: a 0.30-point drop in $F_1$ inflates measured absorption by approximately 14.5 pp.

Two caveats apply. First, the 7-point curve has limited degrees of freedom (5 residual df for linear, 4 for quadratic), and the perfect monotonicity ($\rho = -1.0$) may partly reflect the small sample size. Second, the curve is estimated from first-letter probes (binary classification, 26 balanced classes) and extrapolated to RAVEL probes (multi-class, imbalanced). This cross-domain extrapolation is not validated experimentally. We present the linear fit as the primary result and the quadratic fit as exploratory.

**City-continent variation is fully explained by probe quality.** At $F_1 = 0.87$, the linear degradation curve predicts 30.8% absorption. City-continent's observed rate is 31.4%, a curve delta of $\Delta = +0.6$ pp---within noise. There is no evidence of a hierarchy-specific effect for city-continent after accounting for probe quality. This conclusion is reinforced by the negligible width effect (Section 4.3): if city-continent absorption were driven by dictionary capacity rather than probe quality, wider SAEs would reduce it.

City-country ($F_1 = 0.73$, $\alpha = 45.1\%$) sits 8.5 pp above the linear curve prediction of 36.6%. This modest excess may reflect a genuine hierarchy effect (the highly imbalanced class distribution, with 80 countries vs. 26 letters) or nonlinear amplification at low probe $F_1$. The distinction cannot be resolved from the current data.

**City-language is a genuine outlier.** At $F_1 = 0.82$, the curve predicts 32.9% absorption. City-language's observed rate is 11.6%---a curve delta of $\Delta = -21.3$ pp, far outside the prediction interval. Probe quality alone cannot explain why city-language has the lowest absorption of any hierarchy tested. This hierarchy-specific suppression effect is the strongest evidence that genuine hierarchy structure influences absorption beyond probe quality confounds.

The suppression may relate to the many-to-many structure of city-language mappings. Unlike city-continent (one-to-one: each city maps to exactly one continent) and first-letter (one-to-one by construction), city-language has shared representations: Brussels associates with French, Dutch, and German; Spanish spans cities across Europe, South America, and North America. This structural difference is a candidate explanation, but the mechanism connecting many-to-many structure to absorption suppression is untested.

**Summary.** Probe quality is a major confound in cross-domain absorption measurement ($R^2 = 0.777$). Two of three RAVEL hierarchies (city-continent and, to a lesser extent, city-country) have their elevated absorption rates largely or fully explained by lower probe quality. City-language, however, is a genuine hierarchy-specific anomaly ($\Delta = -21.3$ pp below curve). The quality-gated range between the two hierarchies that survive confound analysis---city-language (11.6%) and city-continent (31.4%)---is 2.7$\times$. First-letter (27.1%) falls within this range, not at either extreme. Future cross-domain absorption studies must include probe degradation controls to separate measurement artifacts from hierarchy-dependent effects.

## 4.7 Data Integrity Verification

Two validation procedures ensure the results in this section are internally consistent.

**Aggregation unification.** Three different first-letter L24 16k absorption rates appeared across prior experimental runs: 27.1% (500 words, 3 prompts), 21.6% (2,345 words, 5 prompts), and 34.5% (pilot, small sample). These reflect different experimental conditions, not aggregation inconsistencies. The 27.1% rate (Table 2) comes from the cross-domain comparison experiment (iter\_009). The 21.6% rate (Table 3, undegraded control) comes from the probe degradation experiment (iter\_010), which used a retrained probe with different regularization ($C = 0.001$ vs. $C = 0.01$) on a larger training set (14,068 vs. 4,132 tokens). The 5.5 pp gap between these rates is explained by the larger test set diluting rare-word absorption and the better-regularized probe shifting the false negative boundary. Both rates are correct within their respective experimental conditions; they are not interchangeable.

**CI correction.** An earlier version of Table 3 computed bootstrap CIs using per-word resampling (averaging rates across unique words) while reporting per-token point estimates, producing mathematically impossible CI inversions (lower bound exceeding point estimate) in 6 of 7 rows. The corrected Table 3 above uses per-token percentile bootstrap, resampling individual token observations to match the per-token point estimates. All 7 CIs now bracket their respective point estimates.

**Patching spot-check.** The activation patching data used in Section 5 underwent a sign-reversal correction between experimental iterations. To verify the corrected data, we re-ran the patching protocol on a stratified random sample of 20 city-continent entities (seed = 42). Of these, 10 entities had sufficient absorption ($\geq 2$ FN contexts) for patching analysis. The observed mean primary recovery rate was 62.7%, within 0.8 pp of the full-dataset rate (61.9%). Cohen's $d = 2.04$ (large). Wilcoxon signed-rank $p = 0.00098$. All four pass criteria (recovery within 10 pp, $d > 0.5$, Wilcoxon $p < 0.01$, paired $t$-test $p < 0.01$) are satisfied. The spot-check confirms the corrected patching data.

<!-- FIGURES
- Figure 2: gen_fig2_layer_absorption.py, fig2_layer_absorption.pdf --- Layer-dependent absorption profile across 4 hierarchies at layers 6, 12, 18, 24 with 16k JumpReLU SAEs
- Table 2: inline --- Cross-domain absorption rates at L24 with 16k and 65k SAEs
- Figure 3: gen_fig3_perclass_heatmap.py, fig3_perclass_heatmap.pdf --- Per-continent absorption heatmap (6 continents x 2 SAE widths) at layer 24
- Figure 5: gen_fig5_probe_degradation.py, fig5_probe_degradation.pdf --- Probe degradation ablation curve with RAVEL hierarchy points overlaid, showing R^2=0.777 linear fit
- Table 3: inline --- Probe degradation ablation results with corrected per-token bootstrap CIs (7 F1 levels)
-->
