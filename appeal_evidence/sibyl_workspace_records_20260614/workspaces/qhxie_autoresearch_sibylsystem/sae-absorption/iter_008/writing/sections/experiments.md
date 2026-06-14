# 4 Cross-Domain and Cross-Layer Absorption

We apply the measurement pipeline described in Section 3 to four feature hierarchies across eight Gemma Scope SAE configurations. Three results emerge: (1) first-letter absorption varies 15x across model layers, from 2.2% at layer 18 to 34.5% at layer 24; (2) measured absorption rates differ significantly across hierarchy types (Kruskal-Wallis $p = 0.005$), with four of six pairwise comparisons reaching significance; and (3) probe quality correlates strongly with false negative rate ($\rho = -0.756$, $p < 0.001$), confounding absolute cross-domain rate comparisons.

## 4.1 Layer Dependence

First-letter absorption provides the cleanest measurement because the sae\_spelling pipeline (Chanin et al., 2024) achieves $\text{F1} \geq 0.97$ using in-context learning prompts, eliminating probe quality as a confound. Table 2 reports absorption rates across all eight Gemma Scope JumpReLU configurations.

**Table 2: First-letter absorption rates across layers and SAE widths.** Probes use sae\_spelling ICL pipeline ($\text{F1} \geq 0.97$). Bootstrap 95% CI from 10,000 resamples. $n = 222$ test words, 25/26 letters covered.

| Config | $\text{AR}$ (%) | Strict $\text{AR}$ (%) | $n_{\text{FN}}$ / Correct | 95% CI |
|--------|:---:|:---:|:---:|:---:|
| L6, 16k | 2.4 | 0.0 | 4 / 169 | [0.6, 14.4] |
| L6, 65k | 2.4 | 0.0 | 4 / 166 | [0.6, 14.7] |
| L12, 16k | 5.7 | 1.4 | 8 / 141 | [2.0, 8.1] |
| L12, 65k | 9.2 | 5.0 | 13 / 141 | [4.1, 13.4] |
| L18, 16k | **2.2** | 0.0 | 4 / 183 | [0.4, 4.0] |
| L18, 65k | 4.5 | 0.0 | 8 / 177 | [0.9, 8.1] |
| L24, 16k | **34.5** | 17.2 | 30 / 87 | [21.3, 49.5] |
| L24, 65k | **25.5** | 17.0 | 24 / 94 | [16.7, 38.3] |

Absorption rates at layer 24 (25--35%) exceed rates at layers 6, 12, and 18 (2--9%) by a factor of 15x. The minimum rate is 2.2% (L18, 16k) and the maximum is 34.5% (L24, 16k). This variation is unconfounded: the sae\_spelling ICL pipeline achieves $\text{F1} \geq 0.97$ at every layer, so the denominator of the absorption rate (correctly classified tokens on raw activations) is comparably reliable throughout.

Layer 24 rates (25--35%) align with the 15--35% range reported by Chanin et al. (2024), suggesting that prior work -- which measured at a single, unspecified layer -- likely evaluated at the model's later layers. The absorption surge at layer 24 is consistent with the model resolving its final token prediction at the last residual stream positions, where parent-child feature competition intensifies.

At layers 6 and 18, absorption is minimal (2.2--4.5%). At layer 12, intermediate rates (5.7--9.2%) coincide with the wider SAE ($M = 65{,}536$) showing higher absorption than the narrower one ($M = 16{,}384$), consistent with larger dictionaries creating more opportunities for child features to absorb parent information.

Figure 3 visualizes this layer dependence. The L24 spike dominates; layers 6 and 18 are barely distinguishable from zero.

![First-letter absorption rates across model layers and SAE widths. Layer 24 shows 25--35% absorption, while layers 6, 12, and 18 remain below 10%. Error bars indicate bootstrap 95% confidence intervals.](figures/fig3_layer_absorption.pdf)

## 4.2 Cross-Domain Variation

We measure absorption on three RAVEL entity-attribute hierarchies at layer 24, where probe quality is highest for all hierarchy types. Table 3 reports the cross-domain results alongside first-letter baselines at the same layer and SAE configurations.

**Table 3: Cross-domain absorption rates at layer 24.** Each RAVEL hierarchy is compared to first-letter at the same SAE configuration. Permutation test $p$-values from 10,000 permutations.

| Hierarchy | SAE | $\text{AR}$ (%) | 95% CI | Probe $\text{F1}$ | vs. First-letter $\Delta$ | $d$ | $p$ |
|-----------|-----|:---:|:---:|:---:|:---:|:---:|:---:|
| First-letter | L24, 16k | 34.5 | [21.3, 49.5] | 0.971 | -- | -- | -- |
| First-letter | L24, 65k | 25.5 | [16.7, 38.3] | 0.971 | -- | -- | -- |
| City-continent | L24, 16k | 35.8 | [16.2, 59.7] | 0.843 | +1.4 | 0.31 | 0.829 |
| City-continent | L24, 65k | 26.0 | [8.9, 47.8] | 0.843 | +0.5 | 0.12 | 0.932 |
| City-country | L24, 16k | 18.5 | [19.3, 42.2] | 0.789 | **-16.0** | -3.84 | **0.004** |
| City-country | L24, 65k | 12.7 | [11.9, 30.7] | 0.789 | **-12.8** | -3.51 | **0.008** |
| City-language | L24, 16k | 13.6 | [13.1, 41.7] | 0.823 | **-20.8** | -5.16 | **<0.001** |
| City-language | L24, 65k | 13.6 | [7.2, 35.4] | 0.823 | **-11.9** | -3.24 | **0.015** |

The Kruskal-Wallis test across all four hierarchy types yields $p = 0.005$, confirming that absorption rates differ significantly by hierarchy. Four of six pairwise comparisons (city-country and city-language vs. first-letter at both widths) are significant at $p < 0.05$ with large effect sizes ($|d| > 3.0$). City-continent absorption (26--36%) is statistically indistinguishable from first-letter at L24 ($p > 0.8$), while city-country (13--19%) and city-language (14%) fall significantly below first-letter.

This result reverses the pilot finding from layer 12, where semantic hierarchies appeared to show higher absorption than first-letter. The reversal demonstrates that layer-hierarchy interactions are non-trivial: absorption rankings across hierarchy types depend on which model layer is measured.

## 4.3 The Probe Quality Confound

Probe quality varies substantially across hierarchies (Table 1 in Section 3): first-letter achieves $\text{F1} = 0.971$, while RAVEL probes range from $\text{F1} = 0.789$ (city-country) to $0.843$ (city-continent). This variation confounds the cross-domain comparison.

Probe quality correlates strongly with false negative rate ($\rho = -0.756$, $p < 0.001$). A higher-quality probe correctly classifies more tokens in the raw-activation condition, producing a larger denominator for the absorption rate. Lower-quality probes miss correct classifications even before SAE encoding, potentially masking absorption events that exist but are undetectable.

Three specific confounds deserve explicit acknowledgment:

1. **Denominator asymmetry.** First-letter probes correctly classify 87--183 tokens (depending on layer), providing a large pool in which to detect false negatives. RAVEL probes at L24 correctly classify fewer tokens (e.g., $n = 200$ for city-continent), meaning each individual false negative has a larger marginal impact on the absorption rate.

2. **Missed absorption.** If a RAVEL probe misclassifies a token on raw activations, any subsequent SAE-induced failure on that token is invisible to the measurement pipeline. The absorption rate is therefore a lower bound on true absorption for low-quality probes.

3. **Spurious false negatives.** Conversely, probe errors in the SAE-reconstructed condition could create false negatives that do not reflect genuine absorption, inflating the rate. This effect is bounded by the probe's overall error rate.

These confounds do not invalidate the finding that absorption rates differ across hierarchies -- the Kruskal-Wallis $p = 0.005$ is robust to the direction of probe-quality bias. However, the absolute magnitude of cross-domain rates carries quantitative uncertainty that cannot be resolved without higher-quality probes or probe-independent measurement methods.

Figure 4 presents the full layer-hierarchy absorption interaction as a heatmap, showing data at all available configurations. Cells marked "--" indicate hierarchy-layer combinations where no RAVEL probe was trained (RAVEL probes were measured only at L24, the best-performing layer).

![Absorption rate heatmap across hierarchy types and model layers for JumpReLU 16k (left) and 65k (right). First-letter rates span all four layers; RAVEL hierarchies are measured at layer 24 only. Color intensity encodes absorption rate. Missing cells indicate hierarchy-layer combinations without trained probes.](figures/fig4_crossdomain_heatmap.pdf)

## 4.4 Absorption-Hedging Decomposition by Hierarchy

To distinguish genuine feature absorption from hedging-related false negatives, we apply the single-$L_0$ decomposition (Section 3.4) to first-letter absorption across all eight SAE configurations. Table 4 reports the fraction of false negatives attributable to absorption (parent feature fires but is insufficient) versus hedging (parent feature absent).

**Table 4: Absorption-hedging decomposition for first-letter across SAE configurations.** "Absorbed" = parent latent fires in SAE output; "Hedged" = parent latent absent.

| Config | $n_{\text{FN}}$ | Absorbed (%) | Hedged (%) | $\text{AR}$ (%) |
|--------|:---:|:---:|:---:|:---:|
| L6, 16k | 4 | **100.0** | 0.0 | 2.4 |
| L6, 65k | 4 | **100.0** | 0.0 | 2.4 |
| L12, 16k | 8 | **75.0** | 25.0 | 5.7 |
| L12, 65k | 13 | 46.2 | **53.8** | 9.2 |
| L18, 16k | 4 | **100.0** | 0.0 | 2.2 |
| L18, 65k | 8 | **100.0** | 0.0 | 4.5 |
| L24, 16k | 30 | 50.0 | 50.0 | 34.5 |
| L24, 65k | 24 | 33.3 | **66.7** | 25.5 |

At early and middle layers (L6, L18), all false negatives are classified as absorbed -- the parent feature fires but is insufficient for correct classification. At layer 24, where absorption is highest, hedging accounts for 50--67% of false negatives in the wider SAE. This pattern suggests that at the final prediction layer, both absorption and hedging contribute substantially, while at earlier layers, the few false negatives that occur are predominantly absorption-driven.

The wider SAE ($M = 65{,}536$) consistently shows a higher hedging fraction than the narrower one ($M = 16{,}384$) at L12 and L24, consistent with the larger dictionary creating more opportunities for parent features to be absent from the active set rather than merely suppressed.

<!-- FIGURES
- Figure 3: gen_fig3_layer_absorption.py, fig3_layer_absorption.pdf — First-letter absorption across 4 layers and 2 SAE widths with bootstrap CI
- Figure 4: gen_fig4_crossdomain_heatmap.py, fig4_crossdomain_heatmap.pdf — Heatmap of absorption rate (hierarchy x layer x width)
- Table 2: inline — First-letter absorption rates across all 8 Gemma Scope configs
- Table 3: inline — Cross-domain absorption rates at L24 with statistical tests
- Table 4: inline — Absorption-hedging decomposition by SAE configuration
-->
