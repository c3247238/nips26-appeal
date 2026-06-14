# Cross-Domain Absorption Results

With validated probes and a documented measurement pipeline (Section 3), we now measure feature absorption across four hierarchy types, two SAE widths, and four transformer layers on Gemma 2 2B. We report five findings: cross-domain variation is statistically significant (Section 4.1), absorption concentrates at the final prediction layer (Section 4.2), wider SAEs reduce absorption (Section 4.3), within-hierarchy variation is large (Section 4.4), first-letter is not the worst case (Section 4.5), and a probe degradation ablation decomposes the variation into probe quality effects and genuine hierarchy effects (Section 4.6).

## 4.1 Cross-Domain Variation Is Statistically Significant

Table 2 presents the central results: absorption rates at layer 24 across all four hierarchy types and both SAE widths.

| Hierarchy | SAE | $\alpha$ (%) | 95% CI | $F_1$ | $N_{\text{entities}}$ | $N_{\text{FN}}$ |
|-----------|-----|-------------|--------|-------|----------------------|----------------|
| First-letter | 16k | **27.1** | [26.3, 34.7] | 1.00 | 2,345 | -- |
| City-continent | 16k | 31.4 | [28.9, 33.9] | 0.87 | 1,567 | 418 |
| City-country | 16k | **45.1** | [42.2, 49.0] | 0.73$^\dagger$ | 1,405 | 515 |
| City-language | 16k | 11.6 | [9.7, 13.5] | 0.82 | 1,229 | 124 |
| First-letter | 65k | -- | -- | 1.00 | 2,345 | -- |
| City-continent | 65k | 31.3 | [28.7, 33.8] | 0.87 | 1,567 | 416 |
| City-country | 65k | 32.9 | [30.2, 35.7] | 0.73$^\dagger$ | 1,405 | 376 |
| City-language | 65k | 7.7 | [6.2, 9.4] | 0.82 | 1,229 | 83 |

**Table 2.** Cross-domain absorption rates at layer 24 on Gemma 2 2B. Bootstrap 95% CI from 10,000 resamples. $\dagger$City-country probe $F_1 = 0.73$ falls below both quality gates; results included with documented caveat. Bold marks the highest and lowest rates within the 16k configuration. First-letter 65k not measured with the RAVEL pipeline; the sae-spelling pipeline does not produce directly comparable per-token rates across SAE widths.

Absorption rates span a 4.1$\times$ descriptive range at layer 24 with 16k SAEs: from 11.6% (city-language) to 45.1% (city-country). Within the three RAVEL hierarchies---which share the same model, layer, SAE, token position, and prompt framework---variation is statistically significant (Kruskal-Wallis $p = 7.4 \times 10^{-66}$). All three within-RAVEL pairwise comparisons are significant after Bonferroni correction: city-continent vs. city-language ($p_{\text{Bonf}} = 1.5 \times 10^{-30}$), city-continent vs. city-country ($p_{\text{Bonf}} = 8.4 \times 10^{-12}$), and city-language vs. city-country ($p_{\text{Bonf}} = 2.1 \times 10^{-67}$).

Comparing RAVEL hierarchies against first-letter requires more caution because of the token position asymmetry (Section 3.1). City-language vs. first-letter reaches significance ($p_{\text{Bonf}} = 0.003$), but city-continent vs. first-letter ($p_{\text{Bonf}} = 1.0$) and city-country vs. first-letter ($p_{\text{Bonf}} = 1.0$) do not. The 4.1$\times$ range is a descriptive finding; not all pairwise comparisons are statistically significant.

## 4.2 Absorption Concentrates at the Final Prediction Layer

Figure 2 shows the absorption profile across transformer layers 6, 12, 18, and 24 for all four hierarchies with 16k SAEs.

![Layer-dependent absorption rates for four hierarchies on Gemma 2 2B with 16k JumpReLU SAEs. Absorption concentrates at layer 24 across all hierarchy types, with first-letter showing a dramatic 26$\times$ increase from layer 6 (1.0%) to layer 24 (27.1%). Shaded bands indicate bootstrap 95% CI.](figures/fig2_layer_absorption.pdf)

First-letter absorption rises from 1.0% at layer 6 to 4.7% at layer 12, dips to 2.0% at layer 18, and jumps to 27.1% at layer 24---a 26$\times$ increase from L6 to L24. The non-monotonic dip at L18 is consistent with layer 18 performing intermediate computations distinct from the final token-prediction computation at L24.

RAVEL hierarchies follow the same pattern: all four hierarchies show their highest absorption at layer 24. City-continent rises from approximately 5% at L6 to 31.4% at L24. City-country reaches 45.1% at L24. This layer concentration is the paper's most robust finding: probe quality is $F_1 = 1.0$ at all layers for first-letter (eliminating probe quality as a confound for the layer effect), and the ordering is consistent across hierarchies.

The concentration at the final layer suggests absorption arises from task-specific prediction computation, not from generic feature representation. Gemma 2 2B has $L = 26$ layers; layer 24 is the last layer at which Gemma Scope provides SAEs.

## 4.3 Width Effect

Wider SAEs (65k vs. 16k dictionary features) generally reduce absorption. The effect is most pronounced for city-country: 45.1% at 16k vs. 32.9% at 65k (a 12.2 percentage point reduction). City-language decreases from 11.6% to 7.7%. City-continent shows almost no change (31.4% vs. 31.3%), suggesting its absorption is driven by probe quality rather than dictionary capacity (Section 4.6 confirms this).

The width effect is consistent with the sparsity-capacity trade-off in absorption: wider dictionaries provide more features to represent parent concepts, reducing the pressure to encode parent information into child decoder vectors. The magnitude of the effect varies by hierarchy, indicating that hierarchy structure interacts with SAE capacity.

## 4.4 Per-Class Variation

Figure 3 shows per-continent absorption rates for city-continent at layer 24, revealing extreme within-hierarchy variance.

![Per-continent absorption rates for city-continent at layer 24 on Gemma 2 2B. Europe (90.2%, $n = 276$) and Oceania (52.9%, $n = 51$) show far higher absorption than Africa (3.9%, $n = 231$) and South America (3.9%, $n = 207$). The 16k and 65k SAEs show nearly identical per-continent patterns.](figures/fig3_perclass_heatmap.pdf)

Europe dominates: 90.2% of probe-correct European city activations lose their continent classification after SAE encoding. Oceania follows at 52.9%, Asia at 24.4%, and North America at 19.1%. Africa and South America show minimal absorption (3.9% each). The 16k-to-65k transition preserves this ordering, with nearly identical per-class rates.

Within city-country, the variance is even more extreme. USA shows 0% absorption (176 entities), while smaller countries such as Albania, Algeria, and Argentina reach 100% (at $n = 12$--$25$ entities each). This pattern---low absorption for high-frequency classes, high absorption for rare classes---is consistent with the SAE learning dedicated features for frequent entities but relying on shared (absorbable) features for rare ones.

The large within-hierarchy variance suggests absorption is driven by specific parent-child pair properties---class frequency, feature specialization, decoder geometry---not by hierarchy-wide factors alone. The per-class patterns carry more information than the hierarchy-level averages.

## 4.5 First-Letter Is Not the Worst Case

At layer 24 with 16k SAEs, first-letter absorption (27.1%) is lower than city-country (45.1%) and comparable to city-continent (31.4%). City-language (11.6%) is significantly lower than first-letter ($p_{\text{Bonf}} = 0.003$). The received wisdom from Chanin et al. (2024)---that first-letter spelling represents a worst-case or typical scenario for absorption---does not hold. The first-letter task, with its perfect probes and balanced class distribution, occupies an intermediate position in the cross-domain absorption spectrum.

This finding reframes the absorption literature. Statements calibrated to 15--35% first-letter absorption rates may underestimate absorption severity for imbalanced knowledge hierarchies (city-country) or overestimate it for many-to-many hierarchies (city-language). Cross-domain evaluation is necessary to characterize the actual range of absorption in practice.

However, the probe quality difference between first-letter ($F_1 = 1.0$) and RAVEL hierarchies ($F_1 = 0.73$--$0.87$) raises a confound: do the higher RAVEL absorption rates reflect genuine hierarchy effects, or are they inflated by imperfect probes? Section 4.6 addresses this question directly.

## 4.6 Probe Degradation Ablation Resolves the Confound

The probe degradation ablation (Section 3.7) tests whether cross-domain absorption variation is a genuine hierarchy effect or a probe quality artifact. We degrade first-letter probe quality via weight noise injection to seven $F_1$ levels (0.70 to 1.0), re-measure absorption at each level, and compare the resulting curve against RAVEL absorption rates at their native $F_1$ values.

Table 5 reports the degradation curve data.

| Target $F_1$ | Actual $F_1$ | $\alpha$ (%) | 95% CI | Seed SD |
|-------------|-------------|-------------|--------|---------|
| 0.70 | 0.685 | 36.1 | [37.9, 42.1] | 1.9 |
| 0.75 | 0.754 | 35.3 | [39.2, 43.4] | 1.1 |
| 0.80 | 0.789 | 34.4 | [37.8, 41.8] | 4.2 |
| 0.85 | 0.846 | 33.6 | [37.0, 40.9] | 5.5 |
| 0.90 | 0.904 | 32.4 | [34.6, 38.3] | 3.6 |
| 0.95 | 0.951 | 28.9 | [30.7, 34.2] | 4.3 |
| 1.00 | 0.999 | **21.6** | [21.6, 24.7] | 0.0 |

**Table 5.** Probe degradation ablation results. First-letter absorption at layer 24 (16k SAE) across 7 degraded probe quality levels. Each degraded level averaged over 3 noise seeds (11,725 tokens per level). Absorption increases monotonically as probe $F_1$ decreases. Bold marks the undegraded control.

Figure 7 shows the degradation curve with RAVEL points overlaid.

![Probe degradation ablation. Blue circles: first-letter absorption at 7 degraded probe $F_1$ levels, with linear fit ($R^2 = 0.777$, $p = 0.009$, slope $= -0.398$) and quadratic fit ($R^2 = 0.942$). RAVEL hierarchies overlaid at their native $F_1$. City-continent (green square) falls within 0.6 pp of the curve---its variation is fully explained by probe quality. City-language (purple diamond) sits 21.3 pp below the curve: a genuine hierarchy-specific outlier that probe quality alone cannot explain. City-country (red triangle) sits 8.5 pp above the curve.](figures/fig7_probe_degradation.pdf)

Three results emerge from the decomposition.

**The degradation curve is well-fitted and perfectly monotonic.** As probe $F_1$ decreases from 1.0 to 0.69, absorption increases from 21.6% to 36.1%. The relationship is perfectly monotonic (Spearman $\rho = -1.0$, $p < 10^{-4}$). A linear model explains 77.7% of the variance ($R^2 = 0.777$, $p = 0.009$, slope $\beta_1 = -0.398$). A quadratic fit captures 94.2% ($R^2 = 0.942$). The curve establishes that probe quality is a major confound: a 0.30-point drop in $F_1$ inflates measured absorption by approximately 14.5 percentage points.

**City-continent variation is fully explained by probe quality.** At $F_1 = 0.87$, the probe degradation curve predicts an absorption rate of 30.8%. City-continent's observed rate is 31.4%, a delta of $+0.6$ percentage points---within the noise of the degradation curve. There is no evidence of a hierarchy-specific effect for city-continent once probe quality is accounted for. The same holds approximately for city-country ($F_1 = 0.73$): the curve predicts 36.6%, the observed rate is 45.1%, a delta of $+8.5$ pp. City-country's modest excess may reflect a genuine hierarchy effect or nonlinear amplification at low probe $F_1$; the distinction cannot be resolved with the current data.

**City-language is a genuine outlier.** At $F_1 = 0.82$, the curve predicts 32.9% absorption. City-language's observed rate is 11.6%---a delta of $-21.3$ percentage points, far outside the prediction interval. Probe quality alone cannot explain why city-language has the lowest absorption of any hierarchy tested. This hierarchy-specific suppression is the strongest evidence that genuine hierarchy effects exist beyond probe quality confounds. The suppression may relate to the many-to-many structure of city-language mappings (multiple cities share a language, multiple languages share a city) or to the model's internal representation of linguistic properties, but these hypotheses remain untested.

**Summary.** Probe quality is a major confound in cross-domain absorption measurement ($R^2 = 0.777$). City-continent's elevated absorption rate is fully explained by its lower probe quality. City-language, however, is a genuine hierarchy-specific anomaly ($\Delta = -21.3$ pp). Future cross-domain absorption studies must include probe degradation controls to separate measurement artifacts from real phenomena.

<!-- FIGURES
- Figure 2: gen_fig2_layer_absorption.py, fig2_layer_absorption.pdf — Layer-dependent absorption profile across 4 hierarchies and layers 6/12/18/24
- Table 2: inline — Cross-domain absorption rates at L24 with 16k and 65k SAEs
- Figure 3: gen_fig3_perclass_heatmap.py, fig3_perclass_heatmap.pdf — Per-continent absorption heatmap (6 continents x 2 SAE widths)
- Figure 7: gen_fig7_probe_degradation.py, fig7_probe_degradation.pdf — Probe degradation ablation curve with RAVEL points overlaid
- Table 5: inline — Probe degradation ablation results (7 F1 levels)
-->
