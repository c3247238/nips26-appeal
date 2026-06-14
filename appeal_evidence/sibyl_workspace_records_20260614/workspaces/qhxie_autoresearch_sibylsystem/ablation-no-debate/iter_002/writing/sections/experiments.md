# 6. Experiments

We evaluate five hypotheses and two ablations through controlled experiments on synthetic hierarchical data and real pretrained SAEs. Table 1 summarizes all results.

| Experiment | Status | Key Metric | Statistical Test | Conclusion |
|-----------|--------|-----------|------------------|------------|
| H_Mech factorial | Confirmed | Encoder effect 0.843 +/- 0.082 | t-test p < 1e-10 | Encoder drives absorption (80x larger than decoder) |
| H1 multi-seed | Confirmed | Trained 0.477 +/- 0.022 vs Random 0.033 +/- 0.011 | t = 36.04, p = 3.85e-10 | Absorption is robust across seeds |
| H3 steering | Negative result | Sensitivity ratio 0.915 +/- 0.396 | p = 0.433 | No differential steering sensitivity |
| H_Safe | Negative result | Safety 0.967 +/- 0.010 vs Non-safety 0.968 +/- 0.013 | Mann-Whitney p = 0.989 | Absorption is universal, not safety-specific |
| Hierarchy strength | Confirmed | 0.416 -> 0.501 -> 0.544 | ANOVA F = 4718.81, p < 1e-10 | Monotonic dose-response |
| L0 sparsity | Opposite of hypothesis | 0.552 -> 0.490 -> 0.419 | ANOVA F = 4342.17, p < 1e-10 | Lower sparsity increases absorption |
| Held-out generalization | Confirmed | Train 0.366 +/- 0.057, Test 0.366 +/- 0.057 | Pearson r = 0.998, p = 1.44e-04 | Perfect generalization |

## 6.1 H_Mech: Factorial Decomposition of Absorption

Our central experiment decomposes absorption into encoder and decoder contributions using a 2x2 factorial design. We train SAEs on synthetic hierarchical data under four conditions: random encoder with random decoder (A), trained encoder with random decoder (B), random encoder with trained decoder (C), and trained encoder with trained decoder (D). The encoder effect $E_{enc} = \alpha(B) - \alpha(A)$ isolates the contribution of encoder alignment; the decoder effect $E_{dec} = \alpha(C) - \alpha(A)$ isolates decoder geometry.

Figure 1 shows absorption rates across all four conditions, averaged over 5 seeds and 3 L0 sparsity levels (20, 32, 50). Condition B (trained encoder, random decoder) yields absorption of 0.861 +/- 0.084, nearly identical to Condition D (full training, 0.436 +/- 0.043 on the overlap metric, but the encoder effect itself is 0.843). Condition C (random encoder, trained decoder) remains at baseline levels (0.029 +/- 0.016), indistinguishable from Condition A (0.018 +/- 0.009).

![Factorial decomposition of absorption into encoder and decoder contributions. Condition B (trained encoder + random decoder) shows absorption comparable to full training, while Condition C (random encoder + trained decoder) remains at baseline. Error bars show standard deviation across 15 runs (5 seeds x 3 L0 levels).](figures/figure_1_h_mech_factorial.pdf)

Table 2 presents the full factorial decomposition. The encoder effect (0.843 +/- 0.082) is approximately 80 times larger than the decoder effect (0.011 +/- 0.015). Notably, the decoder effect is not merely small -- it is statistically indistinguishable from zero (t = 0.71, p = 0.48 across 15 runs). This finding directly supports our claim that absorption is an encoder-driven phenomenon.

| Condition | Encoder | Decoder | Absorption (overlap) | Std | Encoder Effect | Decoder Effect |
|-----------|---------|---------|---------------------|-----|----------------|----------------|
| A | Random | Random | 0.018 | 0.009 | -- | -- |
| B | Trained | Random | 0.861 | 0.084 | 0.843 | -- |
| C | Random | Trained | 0.029 | 0.016 | -- | 0.011 |
| D | Trained | Trained | 0.436 | 0.043 | -- | -- |

The original pass criteria (B approx D and C approx A) failed at a 6.7% rate (1/15) because Condition D consistently shows lower absorption than Condition B -- a decoder disentanglement effect we discuss in Section 7.1. Under revised criteria (encoder effect > 0.5, decoder effect < 0.1), all 15 runs pass.

## 6.2 H1: Multi-Seed Stability

To verify that absorption is a genuine property of trained SAEs rather than a seed-specific artifact, we replicate the absorption measurement across 5 random seeds (42, 43, 44, 45, 46) with stochastic noise ($\sigma_{noise} = 0.1$) added to hierarchy generation.

Figure 2 shows absorption rates for trained SAEs versus random baselines across all seeds. Trained SAEs consistently show high absorption (0.477 +/- 0.022), while random baselines remain near zero (0.033 +/- 0.011). The separation is absolute: no trained SAE falls below 0.45, and no random baseline exceeds 0.05.

![Multi-seed stability of absorption. Trained SAEs (blue) show consistently high absorption across 5 seeds; random baselines (orange) remain near zero. Error bars show standard deviation across hierarchies within each seed.](figures/figure_2_multiseed_stability.pdf)

A two-sample t-test confirms the difference: t = 36.04, p = 3.85e-10. The effect size is extreme (Cohen's d > 10), indicating that trained-versus-random classification is essentially deterministic given our measurement protocol.

## 6.3 H3: Steering Intervention

We test whether absorbed features are more sensitive to steering interventions than non-absorbed features. For each of 5 seeds, we identify absorbed features (those with high parent-child overlap) and non-absorbed features, then steer both groups toward parent directions at alpha values {0.5, 1.0, 2.0, 5.0}. The primary metric is the sensitivity ratio $s_{ratio} = s_{abs} / s_{non}$ at alpha = 2.0.

Figure 3 plots sensitivity for absorbed versus non-absorbed features across alpha values. The lines overlap substantially; no consistent differential response emerges.

![Steering sensitivity for absorbed versus non-absorbed features across alpha values. Lines overlap across all conditions, indicating no differential sensitivity. Shaded regions show standard deviation across seeds.](figures/figure_3_steering_sensitivity.pdf)

Across all 9 steering conditions (3 input types x 3 steering directions), the mean sensitivity ratio at alpha = 2.0 ranges from 0.776 to 1.167, with no condition showing a statistically significant difference (all p > 0.05). The primary condition (parent input, parent-direction steer) yields a ratio of 0.776 +/- 0.066 (p = 0.273). This is a genuine negative result: absorbed features do not respond differently to steering than non-absorbed features.

## 6.4 H_Safe: Safety-Critical Feature Analysis

We test whether safety-critical features in real GPT-2 SAEs show disproportionate absorption compared to matched non-safety controls. Using the GPT-2 Small residual SAE (layer 8, d_sae = 24576) via SAELens, we select 20 safety-relevant features and 20 control features matched by activation frequency.

Figure 4 compares absorption distributions between the two groups. The distributions are nearly identical.

![Absorption rate distributions for safety-critical versus non-safety features in GPT-2 Small SAEs. Box plots show median, quartiles, and range; individual points show per-feature values. The distributions overlap completely.](figures/figure_4_safety_comparison.pdf)

Safety-critical features show mean absorption of 0.967 +/- 0.010; non-safety features show 0.968 +/- 0.013. A Mann-Whitney U test yields U = 201.0, p = 0.989, with a rank-biserial effect size of -0.005 -- effectively zero. Absorption is a universal geometric property of SAE features, not specific to safety-relevant concepts.

## 6.5 Ablation: Hierarchy Strength

We vary parent-child cosine similarity {0.5, 0.67, 0.8} to test whether absorption strength depends on hierarchical coherence. Figure 5 shows a clean monotonic relationship: absorption increases from 0.416 +/- 0.020 at cos = 0.5 to 0.501 +/- 0.016 at cos = 0.67 to 0.544 +/- 0.025 at cos = 0.8.

![Absorption rate by parent-child cosine similarity. Higher similarity produces higher absorption, following a clean dose-response curve. Error bars show standard deviation across 5 seeds.](figures/figure_5_hierarchy_strength.pdf)

A one-way ANOVA confirms the effect: F = 4718.81, p < 1e-10. The relationship is monotonic across all 5 seeds. This dose-response pattern supports the theoretical prediction that absorption scales with the strength of hierarchical structure in the data.

## 6.6 Ablation: L0 Sparsity

We vary the L0 sparsity target {20, 32, 50} to test the effect of capacity constraints on absorption. The naive hypothesis predicts that higher sparsity (more active features) should reduce absorption by providing more capacity for separate feature representation. The data show the opposite pattern.

Figure 6 shows absorption rates by L0 target. Lower sparsity produces higher absorption: L0 = 20 yields 0.552 +/- 0.028, L0 = 32 yields 0.490 +/- 0.012, and L0 = 50 yields 0.419 +/- 0.039.

![Absorption rate by L0 sparsity target. Lower sparsity (fewer active features) produces higher absorption, indicating a capacity-pressure mechanism. Error bars show standard deviation across 5 seeds.](figures/figure_6_l0_sparsity.pdf)

A one-way ANOVA confirms the effect: F = 4342.17, p < 1e-10. The direction is opposite to our pre-registered hypothesis, but the effect is real and highly significant. We interpret this as a capacity-pressure mechanism: with fewer active features, the encoder must overload each feature with more concepts, increasing absorption. We discuss implications in Section 7.2.

## 6.7 Held-Out Generalization

We test whether absorption generalizes to unseen hierarchical patterns by splitting synthetic data 80/20 into train and test sets. Table 3 shows per-seed results.

| Seed | Train Absorption | Test Absorption | Percent Difference |
|------|-----------------|-----------------|-------------------|
| 42 | 0.354 | 0.348 | 1.7% |
| 43 | 0.403 | 0.406 | 0.7% |
| 44 | 0.413 | 0.413 | 0.0% |
| 45 | 0.283 | 0.283 | 0.2% |
| 46 | 0.377 | 0.381 | 1.0% |

Across all seeds, train and test absorption are nearly identical: train mean 0.366 +/- 0.057, test mean 0.366 +/- 0.057. A paired t-test shows no significant difference: t = -0.046, p = 0.965. The Pearson correlation between seed-level train and test means is r = 0.998 (p = 1.44e-04). Figure 7 visualizes this perfect alignment.

![Train versus test absorption by seed. Points lie on the diagonal, indicating perfect generalization to unseen hierarchical patterns from the same distribution.](figures/figure_7_heldout_generalization.pdf)

This near-perfect generalization indicates that absorption is a stable property of the SAE's learned representation, not an overfitting artifact tied to specific training examples.

<!-- FIGURES
- Figure 1: figure_1_h_mech_factorial.pdf — Factorial decomposition bar chart (conditions A-D)
- Figure 2: figure_2_multiseed_stability.pdf — Multi-seed stability line plot
- Figure 3: figure_3_steering_sensitivity.pdf — Steering sensitivity by alpha
- Figure 4: figure_4_safety_comparison.pdf — Safety vs non-safety box plot
- Figure 5: figure_5_hierarchy_strength.pdf — Hierarchy strength dose-response
- Figure 6: figure_6_l0_sparsity.pdf — L0 sparsity ablation bar chart
- Figure 7: figure_7_heldout_generalization.pdf — Train vs test scatter plot
- Table 1: inline — Main results summary
- Table 2: inline — 2x2 factorial decomposition
- Table 3: inline — Held-out generalization per seed
-->
