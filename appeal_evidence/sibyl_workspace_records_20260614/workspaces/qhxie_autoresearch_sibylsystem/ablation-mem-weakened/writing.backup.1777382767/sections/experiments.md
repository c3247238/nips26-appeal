# 5. Experiments

## 5.1 Absorption Detection Results

We measured feature absorption rates for 26 first-letter features (A--Z) across layers 0, 4, 8, and 10 of GPT-2 Small using the Chanin et al. differential correlation metric. Table 3 summarizes the layer-level statistics.

| Layer | Mean Absorption | Max Absorption | HIGH ($\geq$10%) | MEDIUM (5--10%) | LOW ($<$5%) |
|:-----:|:---------------:|:--------------:|:----------------:|:---------------:|:-----------:|
| 0     | 0.021           | 0.094          | 0                | 0               | 26          |
| 4     | 0.039           | 0.160          | 6                | 2               | 18          |
| 8     | 0.034           | 0.242          | 4                | 0               | 22          |
| 10    | 0.029           | 0.209          | 4                | 1               | 21          |

**Table 3:** Absorption detection summary per layer. The majority of features fall into the LOW category across all layers. No features at layers 0 or 8 fall into the MEDIUM category.

Figure 2 shows the absorption rate for each feature and layer.

![Absorption rates across layers for all 26 first-letter features. Most features show near-zero absorption; only 4--6 features per layer exceed the 10% threshold.](figures/fig2_absorption_rates.pdf)

**Figure 2:** Absorption rates for 26 first-letter features across layers 0, 4, 8, and 10. Layer 4 shows the highest mean absorption (0.039) and the most features exceeding the 10% threshold (6/26). Layer 0 has the lowest variance, with no features above 10% absorption. The maximum absorption rate observed was 0.242 for feature U at layer 8.

The overall distribution is strongly right-skewed: 18--26 of 26 features per layer show absorption rates below 10%. This limited variance constrains the statistical power of our subsequent correlation analyses.

## 5.2 Random Baseline Validation

Before testing hypotheses, we validate that feature-specific steering captures meaningful directions. Table 4 compares feature-specific steering success at $s = 50$ against random feature steering at the same strength.

| Layer | Feature-Specific Mean | Random Mean | Delta | $t$-statistic | $p$-value | Cohen's $d$ |
|:-----:|:---------------------:|:-----------:|:-----:|:-------------:|:---------:|:-----------:|
| 4     | 0.796                 | 0.344       | +0.452| 6.41          | $<$0.0001 | 1.26        |
| 8     | 0.854                 | 0.379       | +0.475| 6.02          | $<$0.0001 | 1.18        |

**Table 4:** Random baseline validation. Feature-specific steering significantly exceeds random baseline at both layers ($p < 0.0001$, large effect size). This validates that the decoder directions we steer are not arbitrary but capture task-relevant structure.

Feature-specific steering outperforms random directions by 132% (layer 4) and 126% (layer 8), with large effect sizes ($d > 1.1$). This confirms that the feature-specific decoder directions capture meaningful structure and that the random baseline is an appropriate control.

## 5.3 Feature Steering Results

We tested feature steering effectiveness at six strengths ($s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$) for layers 4 and 8. Steering and probing were conducted on layers 4 and 8 as representative mid-network layers where feature abstraction is substantial but not yet dominated by output-specific representations. Layer 0 (near-input) lacks sufficient feature abstraction for meaningful first-letter features, and layer 10 approaches the output layer where steering effects may be confounded by the unembedding.

Raw steering success rates at $s = 50$ ranged from 0.40 to 1.00 across features, with most features achieving $S(f, 50) \geq 0.70$. The null steering condition ($s = 0$) produced stable baselines with mean target-token probability $P_0(t) = 0.042$ across all features, confirming that observed effects are due to steering rather than prompt variability.

Figure 6 shows dose-response curves grouped by absorption level. Steering success increases monotonically with strength for all absorption categories. At $s = 50$, HIGH-absorption features achieve mean success rates of 0.767 (layer 4) and 0.725 (layer 8), while LOW-absorption features achieve 0.789 (layer 4) and 0.882 (layer 8). The differences between categories are small and inconsistent in direction.

![Steering dose-response curves grouped by absorption level (HIGH: $\geq$10%, MEDIUM: 5--10%, LOW: $<$5%). Success increases with strength for all categories, with no consistent ordering by absorption level.](figures/fig6_dose_response.pdf)

**Figure 6:** Steering dose-response curves by absorption category. Success increases monotonically with steering strength for all categories, with no consistent ordering by absorption level.

Notably, even the most absorbed feature in our sample (U at layer 8, $A(U) = 0.242$) achieves 100% raw steering success at $s = 50$. This single observation already challenges the intuition that high absorption necessarily implies steering failure.

## 5.4 Sparse Probing Results

We trained k-sparse linear probes for first-letter classification at $k \in \{1, 5, 10, 20\}$. At $k = 5$, F1 scores ranged from 0.182 (feature C at layer 4) to 1.00 (feature X at both layers), with substantial variance that does not align with absorption rates. Feature X achieves F1 = 1.00 at layer 4 with zero absorption; feature Z achieves F1 = 0.889 at layer 4, also with zero absorption. Conversely, feature G at layer 4 ($A(G) = 0.146$) achieves F1 = 0.69, while feature Q at layer 4 ($A(Q) = 0.160$) achieves F1 = 0.58.

Full-activation probing (using all 24,576 latents) consistently outperforms k-sparse probing, indicating that task-relevant information is distributed across many latents and is recoverable even when individual features are absorbed.

## 5.5 Hypothesis Testing

Table 1 presents the complete hypothesis test results.

| Hypothesis | Layer | Pearson $r$ | $p$-value | $R^2$ | Result |
|:-----------|:-----:|:-----------:|:---------:|:-----:|:-------|
| H1 (Raw steering) | 4 | +0.008 | 0.970 | 0.000 | Not supported |
| H1 (Raw steering) | 8 | $-$0.301 | 0.136 | 0.090 | Not supported |
| H1b (Delta steering) | 4 | +0.245 | 0.227 | 0.060 | Not supported |
| H1b (Delta steering) | 8 | $\mathbf{-0.431}$ | $\mathbf{0.028}$ | 0.186 | **Supported** |
| H2 (Probing) | 4 | $-$0.003 | 0.987 | 0.000 | Not supported |
| H2 (Probing) | 8 | $-$0.107 | 0.604 | 0.011 | Not supported |

**Table 1:** Summary of hypothesis tests. Only H1b at layer 8 passes the significance threshold ($p < 0.05$). Bold indicates significant results.

**H1: Absorption vs. Raw Steering Effectiveness.** Figure 3 plots absorption rate against raw steering success rate at $s = 50$ for layers 4 and 8. At layer 4, the Pearson correlation is $r = +0.008$ ($p = 0.970$, $R^2 = 0.000$), indicating no linear relationship. At layer 8, $r = -0.301$ ($p = 0.136$, $R^2 = 0.090$) shows a negative trend but fails to reach significance. The Spearman rank correlations are similarly weak: $\rho = +0.029$ ($p = 0.889$) at layer 4 and $\rho = -0.222$ ($p = 0.275$) at layer 8.

![Absorption rate versus raw steering success rate at strength $s = 50$ for layers 4 and 8. Regression lines are shown in gray. Neither layer shows a significant negative correlation.](figures/fig3_absorption_vs_steering.pdf)

**Figure 3:** Absorption rate versus raw steering success rate at $s = 50$ for layers 4 and 8. Regression lines in gray. Neither layer shows a significant negative correlation.

**H1b: Absorption vs. Delta Steering Effectiveness.** Figure 4 plots absorption rate against delta steering success ($\Delta S(f, 50) = S(f, 50) - S_{\text{rand}}(50)$) for layers 4 and 8. At layer 4, $r = +0.245$ ($p = 0.227$, $R^2 = 0.060$), showing no relationship. At layer 8, $r = -0.431$ ($p = 0.028$, $R^2 = 0.186$) achieves significance. The Spearman rank correlation at layer 8 is $\rho = -0.502$ ($p = 0.009$), confirming robustness to non-linear relationships and outliers.

![Absorption rate versus delta steering success (feature-specific minus random baseline) at strength $s = 50$ for layers 4 and 8. Regression lines in gray. Layer 8 shows a significant negative correlation ($r = -0.431$, $p = 0.028$).](figures/fig4_absorption_vs_delta_steering.pdf)

**Figure 4:** Absorption rate versus delta steering success at $s = 50$ for layers 4 and 8. Regression lines in gray. Layer 8 shows a significant negative correlation ($r = -0.431$, $p = 0.028$; Spearman $\rho = -0.502$, $p = 0.009$). The asterisk (*) marks significance at $p < 0.05$.

The contrast between H1 and H1b is stark: the same absorption rates and the same feature-specific steering data produce no correlation in raw form but a significant negative correlation after baseline subtraction. Random baseline steering at layer 8 achieves 37.9% success, and this generic directional effect masks the feature-specific degradation that H1b reveals.

**H2: Absorption vs. Sparse Probing F1.** Figure 5 plots absorption rate against probing F1 at $k = 5$. At layer 4, $r = -0.003$ ($p = 0.987$, $R^2 = 0.000$). At layer 8, $r = -0.107$ ($p = 0.604$, $R^2 = 0.011$). Both correlations are statistically indistinguishable from zero.

![Absorption rate versus sparse probing F1 at $k = 5$ for layers 4 and 8. No significant correlation is observed in either layer.](figures/fig5_absorption_vs_probing.pdf)

**Figure 5:** Absorption rate versus sparse probing F1 at $k = 5$ for layers 4 and 8. No significant correlation is observed in either layer.

**H3: Cross-Layer Consistency.** The linear regression slopes for H1 have opposite signs across layers ($\beta_4 = +0.024$, $\beta_8 = -0.630$), directly inconsistent with the consistency hypothesis regardless of magnitude. For H1b, slopes also have opposite signs ($\beta_4 = +1.441$, $\beta_8 = -2.491$). For H2, the slopes share the same sign ($\beta_4 = -0.010$, $\beta_8 = -0.286$) but differ substantially in magnitude; the coefficient of variation $\text{CV} = \sigma / |\mu| = 1.317$ exceeds the 0.5 threshold. The relationship between absorption and task performance is therefore not consistent across layers.

Table 2 lists the top-absorbed features at layers 4 and 8 and their task performance, illustrating that high absorption does not preclude high raw steering success but does associate with lower delta steering success.

| Feature | Layer | $A(f)$ | $S(f, 50)$ | $\Delta S(f, 50)$ | F1$(f, 5)$ |
|:-------:|:-----:|:------:|:----------:|:-----------------:|:-----------:|
| U       | 8     | 0.242  | 1.00       | 0.62              | 0.46        |
| H       | 8     | 0.190  | 0.55       | $-$0.10           | 0.40        |
| Q       | 4     | 0.160  | 0.80       | 0.37              | 0.58        |
| S       | 8     | 0.160  | 0.65       | 0.12              | 0.18        |
| P       | 4     | 0.148  | 0.70       | 0.24              | 0.44        |
| V       | 8     | 0.147  | 0.70       | 0.12              | 0.67        |
| G       | 4     | 0.146  | 0.80       | 0.38              | 0.69        |
| R       | 4     | 0.140  | 0.40       | $-$0.05           | 0.44        |

**Table 2:** Top 8 most absorbed features at layers 4 and 8 and their steering success ($s = 50$), delta steering success, and probing F1 ($k = 5$). Feature U achieves high raw steering success (1.00) but its delta success (0.62) is below the layer 8 mean (0.475). Feature H shows negative delta success ($-$0.10), indicating its feature-specific effect is weaker than random baseline.

With $n = 26$ features and observed correlations in the $-0.30$ to $+0.01$ range for H1 and H2, our study has limited power to detect small-to-medium effects. The 95% confidence interval for $r = -0.301$ (layer 8 H1) is approximately $[-0.62, +0.10]$, which includes moderate negative correlations that would support H1. For H1b at layer 8, the significant $r = -0.431$ provides stronger evidence, though the $R^2 = 0.186$ indicates that absorption explains only 18.6% of the variance in delta steering success.

<!-- FIGURES
- Figure 2: gen_fig2_absorption_rates.py, fig2_absorption_rates.pdf — Grouped bar chart showing absorption rates for 26 first-letter features across layers 0, 4, 8, and 10
- Figure 3: gen_fig3_absorption_vs_steering.py, fig3_absorption_vs_steering.pdf — Scatter plots of absorption rate vs. raw steering success for layers 4 and 8 with regression lines
- Figure 4: gen_fig4_absorption_vs_delta_steering.py, fig4_absorption_vs_delta_steering.pdf — Scatter plots of absorption rate vs. delta steering success for layers 4 and 8 with regression lines and significance annotation
- Figure 5: gen_fig5_absorption_vs_probing.py, fig5_absorption_vs_probing.pdf — Scatter plots of absorption rate vs. probing F1 for layers 4 and 8 with regression lines
- Figure 6: gen_fig6_dose_response.py, fig6_dose_response.pdf — Dose-response curves showing steering success vs. strength by absorption category
- Table 1: inline — Hypothesis test summary with Pearson r, p-value, and R^2 (includes H1, H1b, H2)
- Table 2: inline — Top absorbed features with their task performance (includes delta steering)
- Table 3: inline — Layer-level absorption detection summary
- Table 4: inline — Random baseline validation with t-statistic and Cohen's d
-->
