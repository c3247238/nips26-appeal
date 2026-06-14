# 7. Discussion

## 7.1 Synthesis of Findings

**No free lunch in SAE design.** Experiment 1 evaluated 27 GPT-2 Small checkpoints across Standard and feature-splitting families on a six-objective Pareto front. Feature splitting eliminates dead neurons ($\delta_{\text{dead}} = 0.000$ vs. $0.197$ for Standard) and improves CE loss recovered ($1.172$ vs. $1.054$), yet Mann-Whitney U tests show no significant advantage on absorption ($U = 48.0$, $p = 0.754$) or hedging ($U = 50.0$, $p = 0.810$). These results support the "impossibility triangle" framing: absorption, hedging, and reconstruction fidelity are in tension, and no architecture family dominates across all three.

**Absorption is causally harmful.** Experiment 2 provides the largest-scale evidence to date that absorption has a unique negative effect on downstream interpretability utility. After controlling for $L_0$ and CE loss recovered, absorption shows a significant negative partial correlation with sparse probing F1 ($r_{\text{partial}} = -0.385$, $p < 0.001$), RAVEL Cause ($r_{\text{partial}} = -0.237$, $p < 0.001$), and RAVEL Isolation ($r_{\text{partial}} = -0.266$, $p < 0.001$). OLS regression with cluster-robust standard errors confirms absorption as a significant negative predictor of all three outcomes (Figure 4). This justifies continued attention to absorption---but not at the expense of other metrics.

![Figure 4: Left: standardized regression coefficients for absorption as a predictor of downstream outcomes. Right: partial-correlation scatter of absorption vs. residualized sparse probing F1. Both panels support H2: absorption carries a unique negative causal cost for interpretability utility.](figures/fig4_absorption_downstream.pdf)

**We need better benchmarks.** Experiment 3 reveals a weak negative correlation between the task-agnostic geography-hierarchy metric and the canonical first-letter benchmark ($r = -0.592$, $p = 0.12$). The first-letter metric is degenerate ($\alpha_{\text{FL}} = 0.0$) on 9 of 10 checkpoints, while the task-agnostic metric shows meaningful variance ($\alpha_{\text{TA}} = 0.0$--$0.24$). H3 is not supported. This negative result suggests the first-letter benchmark may be too narrow, or that different architectures absorb different kinds of hierarchical structure in qualitatively different ways.

## 7.2 Reframing the Research Agenda

Rather than treating absorption as a bug to be fixed by the next architecture, the field should shift toward three priorities:

1. **Multi-objective evaluation as standard practice.** SAE research should report Pareto-front coverage and tradeoff curves, not single-metric leaderboards. A method that improves absorption while degrading hedging or reconstruction is not a universal improvement---it is a different point in tradeoff space.

2. **Task-adaptive SAE selection.** Practitioners should select checkpoints based on the desired tradeoff region. If the goal is causal interpretability (high RAVEL scores), low-absorption checkpoints are preferable. If the goal is reconstruction fidelity with zero dead neurons, feature splitting may be preferable despite no absorption advantage.

3. **Benchmark development beyond single-task proxies.** The first-letter benchmark has served as a useful comparability anchor, but our pilot suggests it does not generalize cleanly to arbitrary semantic hierarchies. Future work should validate absorption metrics across multiple domains (geography, biology, color, temporal) and model scales.

## 7.3 Limitations

**Model scope.** Experiments 1 and 3 are limited to GPT-2 Small due to gated-model access constraints (Gemma-2-2B requires HuggingFace authentication). The SAEBench meta-analysis in Experiment 2 includes Gemma-2-2B and Pythia-160M, but the controlled Pareto evaluation lacks modern model coverage.

**Metric proxies.** The first-letter absorption metric and hedging proxy used in Experiments 1 and 3 are simplified implementations. Full SAEBench integration would improve comparability with prior work. The task-agnostic metric was tested on only one hierarchy domain (geography) and 10 checkpoints.

**Causal claims.** While we control for major confounders ($L_0$, CE loss recovered, width, architecture family), unobserved architecture differences may still bias regression estimates. We report associations conditional on these controls, not definitive causal effects.

**Sample size for E3.** Only 10 checkpoints and one hierarchy domain were tested. A larger validation on 20--50 checkpoints across multiple domains would be needed to draw firm conclusions about benchmark generalizability.

## 7.4 Future Work

Three directions seem especially promising:

- **Scaling the task-agnostic metric.** Extending the automated hierarchy-discovery pipeline to multiple semantic domains and larger models would clarify whether the first-letter benchmark is unrepresentative in general, or whether the divergence we observe is specific to GPT-2 Small.

- **Controlled training experiments.** Our training-free design ensures ecological validity, but it limits architectural coverage. Future work could train matched OrtSAE and Matryoshka checkpoints on GPT-2 Small to test whether their reported absorption reductions hold under multi-objective evaluation.

- **Mechanistic explanations for tradeoffs.** Why does feature splitting improve reconstruction and dead-neuron rate without improving absorption? Why does the attention-output TopK SAE show high first-letter absorption but zero task-agnostic absorption? Answering these questions requires deeper mechanistic analysis of how different architectures allocate latents across the feature hierarchy.

<!-- FIGURES
- Figure 4: gen_fig4_absorption_downstream.py, fig4_absorption_downstream.pdf — Combined bar chart of standardized absorption regression coefficients and partial-correlation scatter
-->
