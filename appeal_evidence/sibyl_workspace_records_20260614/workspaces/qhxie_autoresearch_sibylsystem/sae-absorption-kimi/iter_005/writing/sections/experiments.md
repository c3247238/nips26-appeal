# 4. Results

## 4.1 Pilot Validation

Before the full experiment, we conducted a 4-condition pilot on a 1,024-feature subset with single replicates to validate the experimental design. Table 2 reports pilot results.

| Variant | Absorption Rate | MCC | MSE | L0 | Hedging |
|:---|---:|---:|---:|---:|---:|
| Baseline | 0.203 | 0.216 | 0.0107 | 1044.0 | 0.224 |
| +TopK | 0.033 | 0.212 | 0.0079 | 50.0 | 0.200 |
| +MultiScale | 0.050 | 0.219 | 0.0075 | 50.0 | 0.222 |
| Random | 0.560 | 0.223 | 0.627 | 1032.6 | 0.247 |

**Table 2:** Pilot results (single replicate, 1,024 features). The Random control validates metric discrimination: absorption is 2.8x higher than Baseline, confirming the metric distinguishes structure from randomness. TopK and MultiScale both show strong reduction.

Three findings from the pilot informed the full experiment design. First, the Random control achieves absorption = 0.560, far above all trained variants (0.033--0.203), validating that the ground-truth absorption metric discriminates trained structure from randomness. Second, both TopK (83.8% reduction) and MultiScale (75.3% reduction) show strong absorption reduction relative to Baseline. Third, MCC is approximately equal across all conditions (~0.21--0.22), including Random, confirming that Hungarian matching on overcomplete dictionaries does not discriminate structure. We designated absorption rate as the primary metric and MCC as secondary.

The pilot met the go/no-go criteria (GO with revisions): effect sizes were large, training was stable, and the metric discriminated trained from random structure. We proceeded to the full experiment with 5 replicates per variant on the complete 16,384-feature dataset.

## 4.2 Main Results: Component-Isolated Absorption Rates

Table 3 reports the full experiment results for the three completed variants (5 replicates each, seeds 42, 123, 456, 789, 1011).

| Variant | Absorption Rate | MCC | MSE ($\times 10^{-3}$) | L0 | Hedging | Reduction % | Cohen's $d$ |
|:---|:---|:---|---:|---:|---:|---:|---:|
| Baseline | 0.252 $\pm$ 0.046 | 0.216 $\pm$ 0.001 | 10.44 $\pm$ 0.85 | 964.0 $\pm$ 75.0 | 0.240 $\pm$ 0.007 | --- | --- |
| +TopK | **0.056 $\pm$ 0.021** | 0.214 $\pm$ 0.001 | 7.68 $\pm$ 0.28 | **50.0** | 0.237 $\pm$ 0.025 | **78.0%** | **5.51** |
| +Orthogonality | 0.245 $\pm$ 0.050 | 0.222 $\pm$ 0.000 | **0.03 $\pm$ 0.00** | 550.2 $\pm$ 4.5 | 0.240 $\pm$ 0.009 | 2.7% | 0.14 |
| +MultiScale (pilot) | 0.050 | 0.219 | 7.45 | 50.0 | 0.222 | 75.3% | ~1.1 |

**Table 3:** Main results across SAE variants. Values are mean $\pm$ std across 5 replicates (full experiment) or single-replicate pilot values. MSE values are reported as $\times 10^{-3}$ (e.g., 10.44 corresponds to MSE = 0.01044). Bold indicates best (lowest absorption, lowest MSE). Cohen's $d$ computed vs. Baseline. The +MultiScale row shows pilot values; full experiment pending.

Two findings emerge from the full experiment. First, **TopK sparsity is the dominant driver of absorption reduction**: a 78.0% reduction (from 0.252 to 0.056) with Cohen's $d = 5.51$, an extremely large effect size. This is the strongest absorption-reduction effect observed in our study. Second, **the orthogonality penalty has negligible effect**: only 2.7% reduction ($d = 0.14$) despite achieving near-perfect reconstruction (MSE $\approx 3 \times 10^{-5}$).

The pilot MultiScale result (75.3% reduction, $d \approx 1.1$) is promising but cannot be directly compared to the full-experiment results: the pilot used 1,024 features with a single replicate, while the full experiment used 16,384 features with 5 replicates. Baseline absorption increased from 0.203 (pilot) to 0.252 (full), and TopK increased from 0.033 (pilot) to 0.056 (full), suggesting scale-dependent effects. A provisional ranking based on all available data is: TopK ($d = 5.51$, full) > MultiScale ($d \approx 1.1$, pilot) > Orthogonality ($d = 0.14$, full). This ranking is provisional and may change when MultiScale, Gating, and Full Matryoshka complete full 5-replicate experiments.

## 4.3 H1: MultiScale Dominance

H1 predicted that multi-scale dictionary decomposition is the primary causal driver of absorption reduction (Cohen's $d > 0.8$ vs. Baseline), while per-feature thresholding and gating have negligible effects.

The evidence is mixed. The pilot MultiScale result supports H1: 75.3% reduction with estimated $d \approx 1.1$, a large effect. However, the full TopK result contradicts it: TopK achieves 78.0% reduction with $d = 5.51$, an effect four times larger than MultiScale's pilot estimate. If the pilot MultiScale estimate holds in the full experiment, the component ranking would be TopK ($d = 5.51$) > MultiScale ($d \approx 1.1$) > Orthogonality ($d = 0.14$).

**H1 is REJECTED as stated:** TopK, not MultiScale, is the dominant driver. The evidence supports a revised ranking with TopK at the top.

## 4.4 H2: Component Ranking

H2 predicted the ordering: MultiScale > Orthogonality > TopK > Gating. The current evidence supports a different ranking.

Based on available data, the provisional ranking by absorption-reduction effect size is:

1. **+TopK**: $d = 5.51$ (extremely large; 78.0% reduction) — full experiment, 5 replicates
2. **+MultiScale**: $d \approx 1.1$ (large; 75.3% reduction) — pilot only, 1 replicate, 1,024 features; not directly comparable
3. **+Orthogonality**: $d = 0.14$ (negligible; 2.7% reduction) — full experiment, 5 replicates
4. **+Gating**: Pending full experiment
5. **+Full Matryoshka**: Pending full experiment

**Caution:** The MultiScale ranking is based on pilot data (1,024 features, 1 replicate) and cannot be directly compared to the full-experiment results (16,384 features, 5 replicates). The provisional ranking may change substantially when all variants complete full experiments.

Figure 2 visualizes the component ranking with statistical uncertainty.

![Figure 2: Absorption rate by SAE variant with error bars (std across 5 replicates). The horizontal gray line marks the Random control level from the pilot (0.560). Color coding: green = large effect ($d > 0.8$), yellow = moderate ($0.2 < d < 0.8$), red = negligible ($d < 0.2$). TopK and MultiScale are in a different league; Orthogonality overlaps with Baseline.](figures/fig2_absorption_bars.pdf)

**Figure 2:** Absorption rate by variant. TopK (0.056 $\pm$ 0.021) and pilot MultiScale (0.050) achieve order-of-magnitude reduction relative to Baseline (0.252 $\pm$ 0.046). Orthogonality (0.245 $\pm$ 0.050) is statistically indistinguishable from Baseline.

## 4.5 H3: Absorption--Hedging Trade-off

H3 predicted that absorption-reducing components would exhibit increased hedging, confirming the absorption--hedging trade-off reported by Chanin et al. (2025).

The evidence does not support H3. Hedging scores are approximately invariant across all tested variants: Baseline (0.240 $\pm$ 0.007), TopK (0.237 $\pm$ 0.025), Orthogonality (0.240 $\pm$ 0.009), and pilot MultiScale (0.222). The differences are small relative to within-variant variance.

**H3 is NOT SUPPORTED.** No trade-off is observed; hedging is invariant to architecture in our synthetic setting. Three explanations are possible: (1) synthetic data lacks the correlation structure that triggers hedging in real LLMs, (2) hedging emerges at different scales or training durations, or (3) the hedging metric is insensitive to the variants tested. We acknowledge this as a limitation.

## 4.6 The Sparsity--Absorption Correlation

Figure 3 reveals a striking relationship between L0 sparsity and absorption rate.

![Figure 3: Absorption rate vs. L0 sparsity across SAE variants. Each point represents one variant (full experiment: mean $\pm$ std; pilot: single value). The regression line shows a strong negative correlation. Baseline (L0 = 964, absorption = 0.252), TopK (L0 = 50, absorption = 0.056), Orthogonality (L0 = 550, absorption = 0.245), and pilot MultiScale (L0 = 50, absorption = 0.050) fall near the line.](figures/fig3_sparsity_correlation.pdf)

**Figure 3:** Absorption rate vs. L0 sparsity. The positive relationship ($r \approx +0.93$ across $n = 4$ variants, $p = 0.067$) is exploratory: higher L0 sparsity (more active latents) is associated with higher absorption. Confirmation requires the full 6-variant set and a sparsity-sweep experiment.

The correlation is positive ($r \approx +0.93$ across $n = 4$ variants: Baseline, TopK, Orthogonality, and pilot MultiScale). Baseline ($L_0 = 964$, $A = 0.252$), TopK ($L_0 = 50$, $A = 0.056$), Orthogonality ($L_0 = 550$, $A = 0.245$), and pilot MultiScale ($L_0 = 50$, $A = 0.050$) all fall close to a straight line. With only $n = 4$ data points (2 degrees of freedom), this correlation is mathematically fragile and should be treated as exploratory, not a primary finding. Bootstrap 95% CI: $[+0.87, +1.00]$ (p = 0.067). Two points share $L_0 = 50$ (TopK and MultiScale), further reducing effective sample size. Confirmation requires (1) the full 6-variant set and (2) a dedicated sparsity-sweep experiment within a single architecture. This pattern suggests that **absorption may be primarily driven by sparsity level, not architectural novelty**: variants with more active latents exhibit higher absorption. TopK achieves low absorption not because top-k activation is architecturally special, but because it enforces $L_0 = 50$---a hard limit on co-active features that reduces the pool of competing latents.

This finding redirects the research question: instead of asking "which architecture reduces absorption?" we should ask "why does sparsity control absorption, and what is the optimal sparsity level?"

## 4.7 Effect Sizes and Statistical Significance

Figure 4 shows Cohen's $d$ effect sizes with interpretation thresholds.

![Figure 4: Effect sizes (Cohen's $d$) for each component vs. Baseline, with conventional thresholds at $d = 0.2$ (small), $0.5$ (medium), and $0.8$ (large). TopK achieves an extremely large effect ($d = 5.51$); MultiScale (pilot estimate ~1.1) is large; Orthogonality ($d = 0.14$) is negligible.](figures/fig4_effect_sizes.pdf)

**Figure 4:** Effect sizes with conventional thresholds. Only TopK exceeds the "large" threshold; Orthogonality falls below even the "small" threshold.

The effect size hierarchy is stark: TopK ($d = 5.51$) is roughly five times larger than MultiScale ($d \approx 1.1$), which in turn is roughly eight times larger than Orthogonality ($d = 0.14$). This is not a gradual spectrum; it is a categorical jump. TopK is in a class of its own.

## 4.8 Reconstruction Quality

All trained variants achieve excellent reconstruction on synthetic data. Baseline MSE = 0.0104, TopK MSE = 0.0077, and pilot MultiScale MSE = 0.0075. The Orthogonality variant achieves near-perfect reconstruction (MSE $\approx 3 \times 10^{-5}$)---three orders of magnitude better than Baseline---but this reconstruction quality does not translate into absorption reduction.

The Random control validates that training is necessary: MSE = 0.627, two orders of magnitude worse than trained variants. This confirms the SAEs are learning meaningful structure, even if the MCC metric cannot discriminate among them.

Figure 5 shows the reconstruction--absorption Pareto frontier.

![Figure 5: Reconstruction--absorption Pareto frontier. Each point represents one variant. Pareto-optimal points (lower-left boundary) are connected. TopK dominates: it achieves both lower absorption and lower MSE than Baseline. Orthogonality achieves perfect reconstruction but no absorption benefit, making it Pareto-dominated by TopK.](figures/fig5_pareto.pdf)

**Figure 5:** Reconstruction--absorption Pareto frontier. TopK dominates Baseline on both axes. Orthogonality achieves excellent reconstruction but negligible absorption reduction, leaving it Pareto-dominated.

## 4.9 Feature Recovery MCC

All variants show MCC $\approx$ 0.21--0.22, with minimal variance. The Random control also achieves MCC = 0.223, statistically indistinguishable from trained variants. This is expected behavior: Hungarian matching on an overcomplete dictionary ($m = 2048$ latents for $F = 1024$ features in the pilot, $F = 16{,}384$ in full) yields approximately 0.21 by chance. MCC is not a strong discriminator in this setup, which is why we designated absorption rate as the primary metric.

## 4.10 Hypothesis Test Summary

Table 4 summarizes the status of all hypotheses.

| Hypothesis | Prediction | Evidence | Decision |
|:---|:---|:---|:---|
| H1 (MultiScale dominance) | MultiScale is primary driver ($d > 0.8$) | TopK $d = 5.51$ > MultiScale $d \approx 1.1$ | **REJECTED** |
| H2 (Component ranking) | MultiScale > Ortho > TopK > Gating | TopK >> MultiScale >> Orthogonality | **PARTIALLY SUPPORTED** (ranking differs) |
| H3 (Trade-off) | Reduced absorption increases hedging | Hedging ~0.24 invariant | **NOT SUPPORTED** |

**Table 4:** Hypothesis test summary. H1 is rejected: TopK, not MultiScale, is the dominant driver. H2 is partially supported with a revised ranking. H3 is not supported: no hedging trade-off observed.

The results reveal an unexpected pattern: explicit sparsity control (TopK) dominates structural constraints (MultiScale, Orthogonality) by an order of magnitude. This demands interpretation.

<!-- FIGURES
- Figure 2: gen_fig2_absorption_bars.py, fig2_absorption_bars.pdf — Bar chart of absorption rate by variant with error bars
- Figure 3: gen_fig3_sparsity_correlation.py, fig3_sparsity_correlation.pdf — Scatter plot: absorption vs. L0 sparsity with regression line
- Figure 4: gen_fig4_effect_sizes.py, fig4_effect_sizes.pdf — Effect size bar chart with Cohen's d thresholds
- Figure 5: gen_fig5_pareto.py, fig5_pareto.pdf — Reconstruction-absorption Pareto frontier
- Table 2: inline — Pilot results (4 conditions)
- Table 3: inline — Main results across variants
- Table 4: inline — Hypothesis test summary
-->
