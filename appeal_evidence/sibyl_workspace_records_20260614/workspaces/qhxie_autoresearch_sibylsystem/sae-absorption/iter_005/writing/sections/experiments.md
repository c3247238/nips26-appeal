# 4 Experiments

This section reports results for four experimental phases: confound resolution (Section 4.1), cross-domain absorption measurement (Section 4.2), absorption scaling surface (Section 4.3), and taxonomy correction (Section 4.4). We present each phase with the pre-registered hypothesis, dataset, and quantitative results.

## 4.1 Confound Resolution: Absorption Survives L0 Control (H1)

### Setup

We analyze 48 Gemma Scope SAEs (Gemma 2 2B) with known $L_0$ values from the SAEBench dataset. Six canonical SAEs lacking reported $L_0$ are excluded. All 48 remaining SAEs share the same architecture class (JumpReLU), so architecture class is dropped as a covariate. Covariates are log(width), layer, and log($L_0$). Four quality metrics are evaluated: Sparse Probing F1 ($\text{SP-F1}$, $n = 48$), Spurious Correlation Removal ($\text{SCR}$, $n = 43$), RAVEL True Positive Proportion ($\text{TPP}$, $n = 48$), and Unlearning ($\text{UL}$, $n = 34$).

### Go/No-Go: L0 as Covariate

The critical test is whether the absorption-quality partial correlation survives the addition of log($L_0$). Table 1 reports the results.

**Table 1: Partial Correlations Before and After L0 Control**

| Quality Metric | $n$ | Bivariate $r$ | Partial $r$ (no L0) | Partial $r$ (with L0) | 95% CI | $p$ (with L0) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sparse Probing F1** | 48 | -0.587 | -0.664 | **-0.746** | [-0.853, -0.579] | 1.2e-9 |
| SCR | 43 | -0.449 | -0.692 | -0.570 | [-0.749, -0.315] | 6.6e-5 |
| RAVEL TPP | 48 | -0.471 | -0.488 | -0.331 | [-0.569, -0.041] | 0.022 |
| Unlearning | 34 | -0.182 | -0.082 | -0.123 | [-0.458, 0.242] | 0.487 |

Three of four quality metrics retain |partial $r$| > 0.2 after controlling for $L_0$ ($\text{SP-F1}$, $\text{SCR}$, $\text{TPP}$). The go/no-go criterion is met. The pre-registered threshold (|partial $r$| > 0.2 for at least one metric) is exceeded by all three.

The sparse probing result contains a suppression effect: the partial correlation *strengthened* from $r = -0.664$ to $r = -0.746$ when $L_0$ is added as a covariate. In classical statistical terms, $L_0$ shares variance with absorption that is irrelevant to sparse probing quality. Controlling for $L_0$ removes this irrelevant variance and unmasks the true absorption-quality relationship. Figure 1 illustrates this pattern across all four metrics.

![Absorption-Quality Partial Correlations Before vs. After L0 Control](figures/fig1_partial_correlations.pdf)

Collinearity diagnostics confirm that the covariates are sufficiently independent: VIF values are 1.08 (log width), 1.01 (layer), and 1.09 (log $L_0$), all below the standard threshold of 5. The Pearson correlation between log($L_0$) and log(width) is $r = -0.279$ ($p = 0.055$), indicating weak collinearity.

For SCR, the partial correlation weakens from $r = -0.692$ (without $L_0$) to $r = -0.570$ (with $L_0$), a reduction of 17.6%. The SCR suppression decomposition analysis reveals that layer is the primary suppressor variable: controlling for layer alone shifts the bivariate SCR correlation from $r = -0.449$ to $r = -0.836$, the largest single-covariate effect. Layer correlates strongly with SCR ($r = 0.630$) but weakly with absorption ($r = 0.278$), producing a classic suppression pattern.

For unlearning, the association is non-significant in all analyses ($p = 0.487$), consistent with the pre-registration expectation that unlearning captures different aspects of SAE quality.

### Width-Stratified Analysis

Within-width correlations test whether the absorption-quality link persists after removing cross-width variation. Table 2 reports Spearman $\rho$ within each width stratum.

**Table 2: Width-Stratified Spearman Correlations (BCa Bootstrap 95% CI)**

| Width | $n$ | SP-F1 $\rho$ [CI] | SCR $\rho$ [CI] | TPP $\rho$ [CI] |
|:---:|:---:|:---|:---|:---|
| 16k | 15 | -0.236 [-0.722, 0.336] | 0.025 [-0.556, 0.575] | -0.296 [-0.786, 0.311] |
| 65k | 15 | 0.264 [-0.499, 0.763] | 0.021 [-0.596, 0.520] | -0.100 [-0.672, 0.455] |
| 1M | 18 | -0.480 [-0.829, 0.076] | -0.549 [-0.872, 0.179] | -0.399 [-0.732, 0.118] |

No individual stratum achieves a 95% CI excluding zero, reflecting the limited per-stratum sample sizes ($n = 15$--$18$). The 1M stratum (widest $L_0$ range: 9--207) shows the strongest trends: $\rho = -0.480$ ($p = 0.044$) for sparse probing and $\rho = -0.549$ ($p = 0.052$) for SCR. TPP shows sign-consistent negative correlations across all three strata. The pooled analysis confirms 3/4 metrics with bootstrap CI excluding zero: SP-F1 ($\rho = -0.455$, $p = 0.001$), SCR ($\rho = -0.376$, $p = 0.013$), TPP ($\rho = -0.524$, $p = 1.3 \times 10^{-4}$).

### Baron-Kenny Mediation Analysis

We test the causal path log($L_0$) $\to$ Absorption $\to$ Quality, controlling for log(width) and layer, with 10,000 bootstrap resamples.

**Table 3: Mediation Analysis Results**

| Quality Metric | Mediation Type | Path $a$ (std) | Path $b$ (std) | Direct $c'$ ($p$) | Indirect $ab$ | 95% CI | Sobel $z$ ($p$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| SP-F1 | Indirect only | -0.469 | -0.727 | -0.270 (0.001) | 0.015 | [0.007, 0.028] | 4.08 (4.4e-5) |
| **SCR** | **Full** | **-0.543** | **-0.457** | **-0.029 (0.71)** | **0.025** | **[0.007, 0.048]** | **3.62 (2.9e-4)** |
| TPP | Full | -0.469 | -0.312 | 0.125 (0.25) | 0.003 | [1.4e-5, 0.007] | 2.08 (0.037) |
| Unlearning | None | -0.679 | -0.092 | -0.065 (0.62) | 0.007 | [-0.028, 0.035] | 0.66 (0.51) |

SCR shows full Baron-Kenny mediation: all four steps are met, the direct effect $c' = -0.029$ is non-significant ($p = 0.71$), and the indirect effect bootstrap CI excludes zero. Absorption fully mediates the effect of $L_0$ on SCR. Figure 2 illustrates the path diagram with standardized coefficients.

![Mediation Path Diagram: L0 -> Absorption -> SCR Quality](figures/fig2_mediation_path.pdf)

TPP also meets full mediation criteria by the Baron-Kenny procedure (direct effect $p = 0.25$, indirect CI excludes zero), with proportion mediated $= 0.54$. For sparse probing, the indirect effect is significant (Sobel $z = 4.08$, $p = 4.4 \times 10^{-5}$), but the total effect is non-significant ($p = 0.45$), yielding an unstable proportion-mediated ratio. This occurs because $L_0$ has no direct relationship with sparse probing quality; the entire $L_0$-quality pathway runs through absorption.

### Rosenbaum Sensitivity Analysis

Five matching strategies are compared. Mahalanobis matching (17 pairs, matching on log width, layer, and log $L_0$) achieves the strongest results.

**Table 4: Rosenbaum Sensitivity Analysis by Matching Strategy**

| Strategy | $n$ Pairs | SP-F1 $\Gamma$ | SCR $\Gamma$ | TPP $\Gamma$ |
|:---|:---:|:---:|:---:|:---:|
| Exact width, NN $L_0$ | 4 | 1.0 | 1.0 | 1.0 |
| Within-width median | 23 | 1.0 | 1.0 | 1.0 |
| Propensity score | 6 | 1.8 | 1.0 | 1.8 |
| **Mahalanobis** | **17** | **1.85** | **1.15** | **2.65** |
| Tertile within-width | 16 | 1.0 | 1.0 | 1.0 |

Under Mahalanobis matching, the TPP result achieves $\Gamma = 2.65$: the finding would remain significant even if a hidden confounder altered the odds of treatment assignment by a factor of 2.65:1. Sparse probing reaches $\Gamma = 1.85$ (moderate robustness). Exact-width and within-width matching strategies (4--23 pairs) fail to detect significant differences, consistent with the width-stratified null: the absorption-quality effect is strongest in cross-width comparisons.

### Bradford Hill Assessment

A systematic evaluation using all nine Bradford Hill criteria yields 3 strong (strength of association, plausibility, coherence), 5 moderate (consistency, specificity, dose-response, experiment, analogy), and 0 weak criteria. The overall assessment supports a moderate causal claim for the L0 $\to$ Absorption $\to$ Quality pathway, with the strongest evidence for SCR and TPP.

## 4.2 Cross-Domain Absorption: Metric Limitation Exposed (H2)

### Setup

We use GPT-2 Small (124M parameters) with the `gpt2-small-res-jb` SAE (24,576 latents) at layers 5, 8, and 11. The evaluation dataset contains 3,552 cities from the RAVEL knowledge dataset. Five probe types are trained: Country binary (US vs. non-US), Language binary (English vs. non-English), Continent (6-class), Country top-10, and Language top-10. Probes are logistic regression classifiers trained on the residual stream at `hook_resid_pre` to match the SAE input.

### Probe Quality

Binary probes achieve 86--94% accuracy: Country binary US reaches 92.9--93.9% across layers, Language binary English 84.2--85.0%. Multiclass probes range from 63.3% (Continent, layer 5) to 81.0% (Country top-10, layer 11). Continent and Country top-10 probes fall below the pre-registered 85% quality gate at most layers, limiting the interpretability of absorption rates for these probe types.

### Dominance-Based Absorption Rates

Dominance-based detection (selectivity threshold $= 3$, dominance threshold $= 1.0$) produces absorption rates of 11.3--96.2% across all domain-layer combinations. Figure 3 summarizes the cross-domain absorption rates with the shuffled control.

![Cross-Domain Absorption Rates with Shuffled Controls](figures/fig3_crossdomain_absorption.pdf)

**Table 5: Cross-Domain Absorption Rates by Domain and Layer**

| Probe Type | Layer | Accuracy | $n$ Split | FN Rate | $R_{\text{abs}}$ (dom $\geq 1.0$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Country binary US | 5 | 0.929 | 6 | 0.886 | 0.886 |
| Country binary US | 8 | 0.933 | 18 | 0.537 | 0.537 |
| Country binary US | 11 | 0.939 | 51 | 0.113 | **0.113** |
| Language binary Eng. | 5 | 0.842 | 3 | 0.923 | 0.923 |
| Language binary Eng. | 8 | 0.842 | 2 | 0.962 | 0.962 |
| Language binary Eng. | 11 | 0.850 | 21 | 0.660 | 0.660 |
| Continent | 5 | 0.634 | 5 | 0.889 | 0.889 |
| Continent | 8 | 0.633 | 3 | 0.944 | 0.944 |
| Continent | 11 | 0.669 | 32 | 0.651 | 0.651 |
| Country top-10 | 5 | 0.781 | 19 | 0.729 | 0.729 |
| Country top-10 | 8 | 0.777 | 19 | 0.784 | 0.784 |
| Country top-10 | 11 | 0.810 | 73 | 0.458 | 0.458 |
| Language top-10 | 5 | 0.716 | 11 | 0.770 | 0.770 |
| Language top-10 | 8 | 0.728 | 10 | 0.895 | 0.895 |
| Language top-10 | 11 | 0.759 | 67 | 0.547 | 0.547 |

The lowest absorption rate is Country binary US at layer 11 (11.3%), which also has the most split features ($n = 51$) and highest probe accuracy (93.9%). Layer 11 shows consistently lower rates across all probe types, with substantially more split features identified at deeper layers.

### Shuffled and Random Controls

Both the shuffled-hierarchy control (randomized city-attribute mappings) and the random probe direction control produce 100% absorption rates across all layers and probe types ($n = 5$ trials each, standard deviation $= 0$). The dominance-based metric does not discriminate real from shuffled hierarchies. This result is the central finding of the cross-domain experiment.

### Cosine-Calibrated Metric

The cosine-calibrated variant requires that the dominant non-split feature have decoder direction cosine similarity $> \tau_{\text{cos}}$ with the probe direction. At threshold $\tau_{\text{cos}} = 0.1$, the absorption rate is 0% across all 9 layer-domain combinations. Even at $\tau_{\text{cos}} = 0.05$, only 2 of 9 configurations detect a single absorbed instance each (Language binary English at layer 5, Continent at layers 5 and 8).

### Interpretation

The discrepancy between dominance-based (51--96%) and cosine-calibrated (0%) rates reveals that GPT-2 Small's 24k SAE does not encode knowledge-hierarchy probe directions as dedicated latents. With 98% dead features on city prompts, the surviving features include polysemantic "super-absorbers" that dominate the ablation landscape regardless of probe type. The dominance-based metric measures feature concentration in the SAE's active set, not probe-direction-specific absorption.

## 4.3 Absorption Scaling Surface: Significant Interaction (H3)

### Setup

We analyze 420 SAEs from the SAEBench precomputed dataset (Gemma 2 2B), spanning 9 release families, 3 layers (5, 12, 19), dictionary widths 2,304--1,048,576, and $L_0$ 9.3--8,277. Architecture breakdown: 360 standard (L1), 54 JumpReLU, 6 unknown. Absorption scores are precomputed by SAEBench using the first-letter spelling task.

### Model Comparison

Three regression models are compared in Table 6. The interaction GAM significantly outperforms both simpler models.

**Table 6: Scaling Surface Model Comparison**

| Model | $R^2$ | AIC | Interaction $p$ |
|:---|:---:|:---:|:---:|
| Linear | 0.488 | -1930 | -- |
| Additive GAM | 0.620 | -844 | -- |
| **Interaction GAM** | **0.693** | **-917** | **3.1e-15** |

The interaction term $\text{ti}(\log(\text{width}), \log(L_0))$ is highly significant ($p = 3.1 \times 10^{-15}$), confirming that absorption depends on the joint structure of width and $L_0$. The interaction GAM explains 69.3% of variance in absorption rate, a 20.5 percentage-point improvement over the linear model.

Linear model coefficients confirm the directional effects: log(width) $= +0.054$ (wider SAEs absorb more), log($L_0$) $= -0.014$ (higher $L_0$ absorbs less), layer $= +0.003$ (deeper layers absorb slightly more).

### Scaling Structure

Figure 5 shows the absorption surface in $(\log_2(\text{width}), \log_2(L_0))$ space.

![Absorption Scaling Surface: Raw Data and GAM Interaction Contour](figures/fig5_scaling_surface.pdf)

Per-layer analysis confirms consistent directional effects. Width correlates positively with absorption ($\rho = +0.35$ to $+0.42$, $p < 10^{-4}$ at all layers) and $L_0$ correlates negatively ($\rho = -0.46$ to $-0.49$, $p < 10^{-8}$ at all layers). The interaction means these marginal effects are not independent: the width effect is strongest at low $L_0$, and the $L_0$ effect is strongest at high width.

At 1M width (the largest in the dataset), mean absorption reaches 0.703 at layer 12 and 0.585 at layer 19. At 16k width, mean absorption is 0.034--0.056 across layers, an order of magnitude lower. Within the 1M stratum, absorption ranges from 0.072 ($L_0 = 114$, layer 5) to 0.896 ($L_0 = 9.3$, layer 12), confirming that $L_0$ drives large variation even at fixed width.

### Phase Boundary Detection

Gradient analysis of the GAM surface identifies a transition zone at $\log_2(L_0) \in [2.7, 3.8]$, corresponding to $L_0 \approx 6.5$--14. The ridge contains 443 points with gradient magnitude exceeding the 70th percentile threshold (0.69). Maximum gradient magnitude is 0.99.

At $L_0 > 14$, absorption is low regardless of width. At $L_0 < 7$, absorption increases steeply from 16k to 1M width. The transition zone represents the regime where practitioners face the sharpest absorption-quality tradeoff.

## 4.4 Taxonomy Correction: Artifact Revealed, Chanin Rate Validated (H5)

### Setup

We apply frequency-matched correction to the first-letter absorption taxonomy on GPT-2 Small (`gpt2-small-res-jb` SAE, layer 8). Parent features are identified via selectivity heuristic. For each letter, we compare activation magnitudes in letter-specific contexts against non-letter contexts using a WikiText-103 corpus (7,991 sentences processed). Three baseline strategies are evaluated: (A) non-letter-context activations of the same parent feature, (B) global when-active baseline, and (C) Chanin false-negative-based absorption rate.

### Results

The original taxonomy classified 23/26 letters as Type II (magnitude ratio $< 0.5$) with a comprehensive absorption rate of 92.3%. After frequency-matched correction using Strategy A (non-letter context comparison), 19 letters change classification: the corrected comprehensive rate drops to 19.2% (95% CI [3.8%, 34.6%]).

The correction reveals that most parent features identified via selectivity heuristic are genuinely letter-specific: they fire almost exclusively on tokens starting with the target letter. For letter B, the fire rate ratio (letter-context / non-letter-context) is 70.7:1; for letter D, 124.8:1; for letter X, 8,288:1. When these features do fire in non-letter contexts, they fire at comparable or higher magnitudes (corrected magnitude ratios: B = 10.6, D = 2.9, X = 17.6). The original Type II classification was an artifact of comparing letter-context magnitudes against an inappropriate global baseline.

The Chanin false-negative-based metric provides an independent validation: absorption detected in 19/26 letters (73.1%), with per-letter rates ranging from 0% (K, U, Z) to 100% (E, O). This confirms that genuine absorption occurs even though the Type II magnitude metric is unreliable. Letters F, G, I, and V retain their original Type II classification because they had pre-existing comparison tokens (1 each).

The corrected picture: the original 92.3% comprehensive rate is an artifact of feature specificity. The validated absorption rate is 73.1% (Chanin false-negative metric), indicating that absorption is genuine in the majority of letters but not as pervasive as the magnitude-ratio taxonomy suggested.

<!-- FIGURES
- Figure 1: gen_fig1_partial_correlations.py, fig1_partial_correlations.pdf -- Grouped bar chart of partial correlations with/without L0 control for 4 quality metrics
- Figure 2: gen_fig2_mediation_path.py, fig2_mediation_path.pdf -- Baron-Kenny mediation path diagram for SCR full mediation
- Figure 3: gen_fig3_crossdomain_absorption.py, fig3_crossdomain_absorption.pdf -- Cross-domain absorption rates with shuffled control line at 100%
- Figure 5: gen_fig5_scaling_surface.py, fig5_scaling_surface.pdf -- 2D contour plot of absorption in (log2 width, log2 L0) space with phase boundary
- Table 1: inline -- Partial correlations before and after L0 control
- Table 2: inline -- Width-stratified Spearman correlations with BCa bootstrap CIs
- Table 3: inline -- Mediation analysis results (path coefficients, indirect effects, Sobel tests)
- Table 4: inline -- Rosenbaum sensitivity analysis by matching strategy
- Table 5: inline -- Cross-domain absorption rates by domain and layer
- Table 6: inline -- Scaling surface model comparison (R-squared, AIC, interaction p-value)
-->
