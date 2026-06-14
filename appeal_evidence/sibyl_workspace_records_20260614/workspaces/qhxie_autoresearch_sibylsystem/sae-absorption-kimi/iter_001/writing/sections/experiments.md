# 4 Experiments

We test the three hypotheses through a training-free evaluation pipeline using publicly released pretrained SAE checkpoints. All code and checkpoint identifiers are available in the project repository.

## 4.1 Experiment 1: Multi-Objective Pareto Evaluation

**Scope.** We evaluate 27 GPT-2 Small checkpoints spanning the Standard and feature-splitting families. Metrics include absorption rate ($\alpha$), hedging rate ($h$), explained variance (EV), CE loss recovered, $L_0$ sparsity, and dead-neuron rate ($\delta_{\text{dead}}$). Checkpoints are drawn from SAELens releases (`gpt2-small-res-jb`, `gpt2-small-resid-post-v5`, `gpt2-small-mlp-out-v5`, `gpt2-small-attn-out-v5`, `gpt2-small-hook-z-kk`, `gpt2-small-mlp-tm`, and `gpt2-small-res-jb-feature-splitting`).

**Pareto results.** 17 of 27 checkpoints lie on the empirical Pareto front. Feature-splitting checkpoints achieve 0% dead neurons and strong explained variance (0.976–0.986), but do not dominate on hedging or absorption. Table 1 reports per-family averages.

| Family | $L_0$ | EV | $\delta_{\text{dead}}$ | CE Rec. | $\alpha$ | $h$ |
|---|---|---|---|---|---|---|
| Standard (n=23) | 33.9 | 0.830 | 0.197 | 1.054 | 0.015 | 0.833 |
| Feature Splitting (n=4) | 39.5 | 0.982 | 0.000 | 1.172 | 0.000 | 0.888 |

**Dominance tests.** Mann-Whitney U tests show no significant difference in absorption ($U = 48.0$, $p = 0.754$) or hedging ($U = 50.0$, $p = 0.810$) between standard and feature-splitting families. Feature splitting significantly outperforms on CE loss recovered ($U = 4.0$, $p = 0.001$), but this is a single-metric advantage, not multi-objective dominance.

Figure 1 visualizes the tradeoff space. Point size scales with explained variance, and stars mark Pareto-optimal points. Standard checkpoints occupy a broad region: attention-output SAEs show low hedging (0.284–0.323) and low CE loss recovered (0.595–0.985), while residual-stream SAEs cluster at high hedging ($\geq 0.745$) and high explained variance ($\geq 0.928$). The hook-z SAE is the only standard checkpoint with non-zero absorption ($\alpha = 0.345$), and it sits on the Pareto front despite low explained variance (0.632).

![Figure 1: Pareto front scatter plot showing absorption vs. hedging across 27 GPT-2 Small checkpoints. Point size scales with explained variance; stars mark Pareto-optimal points.](figures/fig1_pareto.pdf)

These results support H1: no architecture family stochastically dominates across the full multi-objective front.

## 4.2 Experiment 2: Downstream Causal Cost Meta-Analysis

**Scope.** We analyze 314 SAEBench checkpoints with precomputed absorption, $L_0$, CE loss recovered, sparse probing F1, and RAVEL Cause/Isolation scores. The corpus spans Standard, TopK, JumpReLU, GatedSAE, MatryoshkaBatchTopK, and PAnneal families on Gemma-2-2B and Pythia-160M.

**Raw correlations.** Absorption correlates negatively with sparse probing F1 ($r = -0.348$, $p < 0.001$), RAVEL Cause ($r = -0.264$, $p < 0.001$), and RAVEL Isolation ($r = -0.263$, $p < 0.001$).

**Partial correlations.** After controlling for $L_0$ and CE loss recovered, the negative relationships strengthen for sparse probing F1 ($r_{\text{partial}} = -0.385$, $p < 0.001$) and remain significant for RAVEL Cause ($r_{\text{partial}} = -0.237$, $p < 0.001$) and RAVEL Isolation ($r_{\text{partial}} = -0.266$, $p < 0.001$).

**Regression results.** OLS with cluster-robust standard errors confirms absorption as a significant negative predictor of all three outcomes after controlling for $L_0$, CE loss recovered, width, and architecture family dummies (Table 2).

| Predictor | Sparse Probing F1 | RAVEL Cause | RAVEL Isolation |
|---|---|---|---|
| Absorption ($\alpha$) | $-0.037$*** (0.006) | $-0.022$*** (0.006) | $-0.023$*** (0.006) |
| $L_0$ | $0.020$* (0.010) | $0.032$*** (0.010) | $0.010$ (0.010) |
| CE loss recovered | $0.057$*** (0.009) | $0.043$*** (0.010) | $0.034$*** (0.010) |
| Width | $0.006$ (0.005) | $0.002$ (0.005) | $-0.000$ (0.005) |
| $R^2$ | 0.321 | 0.166 | 0.150 |
| $N$ | 312 | 312 | 312 |

*Note. Cluster-robust standard errors in parentheses. * $p < 0.05$, *** $p < 0.001$.*

Figure 2 plots the partial-regression relationship between absorption and sparse probing F1. The downward trend is visible across architecture families, with the fit line showing a clear negative slope even after residualizing out $L_0$ and reconstruction quality.

![Figure 2: Partial correlation scatter of absorption vs. sparse probing F1 across 314 SAEBench checkpoints. The dashed line shows the partial-regression fit with 95% confidence band.](figures/fig2_partial_corr.pdf)

These results support H2: absorption carries a unique negative causal cost for downstream interpretability utility, independent of confounders.

## 4.3 Experiment 3: Task-Agnostic Absorption Metric Pilot

**Design.** We construct a task-agnostic absorption metric using automated hierarchy discovery (geography: continent $\rightarrow$ country) on GPT-2 Small, then compare it to the simplified first-letter benchmark across 10 checkpoints.

**Results.** Pearson $r = -0.592$, Spearman $\rho = -0.529$, $p = 0.116$. The correlation is negative and not statistically significant.

Table 3 reports the per-checkpoint values. The first-letter metric is degenerate on 9 of 10 checkpoints ($\alpha_{\text{FL}} = 0.0$), while the task-agnostic metric shows meaningful variance ($\alpha_{\text{TA}} = 0.0$–$0.24$). The single outlier—TopK attention-output SAE—drives much of the negative correlation: it has high first-letter absorption ($\alpha_{\text{FL}} = 0.654$) but zero task-agnostic absorption.

| Release | SAE ID | Family | $\alpha_{\text{TA}}$ | $\alpha_{\text{FL}}$ |
|---|---|---|---|---|
| gpt2-small-res-jb | blocks.0.hook_resid_pre | Standard | 0.073 | 0.000 |
| gpt2-small-res-jb | blocks.4.hook_resid_pre | Standard | 0.160 | 0.000 |
| gpt2-small-res-jb | blocks.8.hook_resid_pre | Standard | 0.240 | 0.000 |
| gpt2-small-res-jb | blocks.11.hook_resid_pre | Standard | 0.140 | 0.000 |
| gpt2-small-resid-post-v5-32k | blocks.4.hook_resid_post | TopK | 0.140 | 0.000 |
| gpt2-small-resid-post-v5-32k | blocks.8.hook_resid_post | TopK | 0.167 | 0.000 |
| gpt2-small-resid-post-v5-128k | blocks.4.hook_resid_post | TopK | 0.167 | 0.000 |
| gpt2-small-resid-post-v5-128k | blocks.8.hook_resid_post | TopK | 0.167 | 0.000 |
| gpt2-small-mlp-out-v5-32k | blocks.8.hook_mlp_out | TopK_MLP | 0.013 | 0.000 |
| gpt2-small-attn-out-v5-32k | blocks.8.hook_attn_out | TopK_Attn | 0.000 | 0.654 |

Figure 3 visualizes the divergence. Most points cluster along the y-axis ($\alpha_{\text{FL}} = 0$) with varying task-agnostic scores, while the TopK_Attn checkpoint sits far to the right at $\alpha_{\text{FL}} = 0.654$.

![Figure 3: Task-agnostic vs. first-letter absorption across 10 GPT-2 Small checkpoints. The two metrics are weakly negatively correlated ($r = -0.59$, $p = 0.12$).](figures/fig3_task_agnostic.pdf)

H3 is not supported. This negative result suggests the first-letter benchmark may be unrepresentative of general absorption behavior, or that different SAE architectures absorb different kinds of hierarchical structure in qualitatively different ways.

<!-- FIGURES
- Figure 1: gen_fig1_pareto.py, fig1_pareto.pdf — Pareto front scatter plot (absorption vs. hedging, point size = explained variance, stars = Pareto-optimal)
- Figure 2: gen_fig2_partial_corr.py, fig2_partial_corr.pdf — Partial correlation scatter of absorption vs. sparse probing F1 with regression fit and 95% CI
- Figure 3: gen_fig3_task_agnostic.py, fig3_task_agnostic.pdf — Task-agnostic vs. first-letter absorption correlation scatter
- Table 1: inline — Architecture family comparison across L0, EV, dead-neuron rate, CE recovered, absorption, and hedging
- Table 2: inline — OLS regression coefficients for downstream causal cost of absorption
- Table 3: inline — Per-checkpoint task-agnostic vs. first-letter absorption values
-->
