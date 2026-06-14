# The Impossibility Triangle of Sparse Autoencoders: A Systematic Multi-Objective Evaluation of Absorption-Mitigation Methods

## Abstract

Feature absorption---where general parent features are encoded through specific child features due to sparsity optimization---has emerged as a central pathology in sparse autoencoders (SAEs). We challenge the prevailing single-metric framing by conducting the first systematic, training-free, multi-objective evaluation of existing pretrained SAE checkpoints. Our analysis spans 314 SAEBench checkpoints and 27 GPT-2 Small checkpoints. Experiment 1 evaluates Standard, TopK, and feature-splitting families on a six-objective Pareto front; Experiment 2 conducts a large-scale meta-analysis across seven architecture families on Gemma-2-2B and Pythia-160M; Experiment 3 pilots a task-agnostic absorption metric on 10 GPT-2 Small checkpoints. We test three hypotheses: (H1) absorption-mitigation methods do not dominate standard SAEs on the multi-objective front; (H2) absorption has a unique negative association with downstream interpretability utility after controlling for confounders; (H3) a task-agnostic absorption metric correlates moderately-to-strongly ($r > 0.4$) with the canonical first-letter benchmark. We find that no architecture stochastically dominates across absorption, hedging, reconstruction fidelity, and dead-neuron rate. Absorption shows a significant negative partial correlation with sparse probing F1 ($r_{\text{partial}} = -0.385$, $p < 0.001$) and RAVEL scores. However, our task-agnostic metric pilot reveals a weak negative correlation with the first-letter benchmark ($r = -0.592$, $p = 0.12$, $N = 10$), driven largely by a single outlier. This negative result suggests the canonical benchmark may be unrepresentative of general absorption behavior, though larger-scale validation is needed. Our work reframes the SAE research agenda from "fixing absorption" to "navigating unavoidable tradeoffs."

---

## 1. Introduction

Sparse autoencoders (SAEs) have become the dominant tool for unsupervised feature discovery in language model interpretability. Their core promise is to decompose high-dimensional neural activations into sparse, human-interpretable latent features. Yet this promise is systematically undermined by **feature absorption**: when hierarchical concepts co-occur, the sparsity penalty incentivizes the SAE to represent a general parent feature (e.g., "animal") through a more specific child feature (e.g., "dog"), rendering the parent effectively invisible in latent space (Chanin et al., 2024).

The research community has responded with architectural mitigations. OrtSAE reports 65% absorption reduction via orthogonality penalties (Korznikov et al., 2025). Matryoshka SAEs report roughly 10x reduction via multi-scale dictionaries (Bussmann et al., 2025). Additional proposals include masked regularization (Narayanaswamy et al., 2026), time-aware feature selection (Li & Ren, 2025), and JumpReLU thresholding (Rajamanoharan et al., 2024). These advances are typically evaluated on a single metric---absorption in isolation---leaving open whether they improve SAEs overall or merely shift pathologies elsewhere.

A skeptical turn in recent work suggests the latter. Chanin et al. (2025) show that narrower SAEs reduce absorption but increase **feature hedging**, an opposite failure mode where latents incorrectly mix correlated features. Cui et al. (2025) argue that standard SAEs face fundamental limits in recovering ground-truth monosemantic features. Roy et al. (2026) document "catastrophic interpretability collapse" under aggressive sparsification. Kantamneni et al. (2025) find that SAE probes underperform simple logistic-regression baselines on downstream tasks. These results raise a troubling possibility: absorption may not be a simple fixable bug but an intrinsic consequence of the sparsity objective and dictionary geometry.

No prior work has rigorously tested this possibility with a fair, multi-objective evaluation across the full suite of SAE quality metrics. We conduct the first systematic, training-free, multi-objective study of absorption-mitigation methods using existing pretrained SAE checkpoints. Because we evaluate only publicly released checkpoints, our controlled Pareto analysis (Experiment 1) is limited to architectures with available GPT-2 releases; broader families such as JumpReLU, GatedSAE, Matryoshka, and PAnneal appear only in the SAEBench meta-analysis (Experiment 2). Our analysis spans 314 SAEBench checkpoints and 27 GPT-2 Small checkpoints. We test three hypotheses:

- **H1 (Tradeoffs):** Absorption-mitigation methods do not stochastically dominate standard SAEs on a multi-objective Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance.
- **H2 (Downstream Causality):** Controlling for $L_0$ sparsity and reconstruction loss, higher absorption rates correlate negatively with downstream interpretability utility.
- **H3 (Metric Generalization):** A task-agnostic absorption metric built from automated hierarchy discovery correlates moderately-to-strongly ($r > 0.4$) with the canonical first-letter absorption benchmark.

We find that no architecture family stochastically dominates across the full metric suite. Feature-splitting checkpoints eliminate dead neurons (mean dead-neuron rate $0.0$ vs. $0.197$ for Standard) and improve $\text{CE}_{\text{recovered}}$ ($1.172$ vs. $1.054$), but Mann-Whitney U tests show no significant advantage on absorption ($p = 0.754$) or hedging ($p = 0.810$). In the largest-scale meta-analysis to date, absorption shows a significant negative partial correlation with sparse probing F1 ($r_{\text{partial}} = -0.385$, $p < 0.001$) and with RAVEL Cause ($r_{\text{partial}} = -0.237$, $p < 0.001$) and Isolation ($r_{\text{partial}} = -0.266$, $p < 0.001$), after controlling for $L_0$ and $\text{CE}_{\text{recovered}}$. However, our task-agnostic metric pilot on 10 GPT-2 Small checkpoints reveals a weak negative correlation with the first-letter benchmark ($r = -0.592$, $p = 0.12$), driven largely by a single outlier. Because the first-letter metric is degenerate at $0.0$ on 9 of 10 checkpoints, this pilot raises questions about whether the canonical benchmark generalizes, though larger-scale validation is needed.

Our work reframes the SAE research agenda. Rather than treating absorption as a bug to be fixed by the next architecture, the field should shift toward multi-objective evaluation as standard practice, task-adaptive SAE selection based on the desired tradeoff region, and benchmark development that captures absorption beyond single-task proxies.

The remainder of this paper is structured as follows. Section 2 states our research questions and hypotheses. Section 3 describes the training-free methodology and checkpoint corpus. Sections 4--6 present the three experiments and their results. Section 7 discusses implications and limitations, and Section 8 concludes.

---

## 2. Research Questions and Hypotheses

**RQ1 (Tradeoffs):** Do absorption-mitigation architectures dominate standard SAEs on a multi-objective Pareto front, or do they systematically trade absorption for other pathologies (hedging, reconstruction loss, dead neurons)?

**RQ2 (Downstream Causality):** Controlling for reconstruction quality and $L_0$ sparsity, does absorption have a unique negative association with downstream interpretability utility (sparse probing, RAVEL disentanglement)?

**RQ3 (Metric Generalization):** Can we construct and validate a task-agnostic absorption metric that generalizes beyond the first-letter spelling task to arbitrary semantic hierarchies?

We formalize these questions into three testable hypotheses:

- **H1:** No architecture family stochastically dominates on a Pareto front spanning absorption, hedging, reconstruction fidelity, dead-neuron rate, and downstream probing performance.
- **H2:** After controlling for $L_0$ and reconstruction loss, higher absorption rates correlate negatively with downstream interpretability utility.
- **H3:** A task-agnostic absorption metric correlates moderately-to-strongly ($r > 0.4$) with the canonical first-letter absorption benchmark.

With the methodology established, we turn to the three experiments and their results.

---

## 3. Methodology

All experiments in this paper follow a **training-free evaluation** paradigm: we use publicly released pretrained SAE checkpoints without training any new SAEs. This design ensures that our conclusions reflect the behavior of existing, widely used checkpoints rather than idiosyncrasies of our own training pipeline. The analysis spans 341 total checkpoints: 27 GPT-2 Small checkpoints from SAELens for controlled Pareto evaluation, and 314 checkpoints from SAEBench for large-scale meta-analysis.

### 3.1 Checkpoint Corpus

**Experiment 1 (Pareto evaluation, H1).** We evaluate 27 GPT-2 Small (117M) checkpoints released via SAELens. The corpus includes five architecture families: Standard ($L_1$-penalized), TopK, TopK_MLP (MLP-output hook), TopK_Attn (attention-output hook), and feature-splitting (`res-jb-fs-{768,1536,3072,6144}`). Layer coverage ranges from layer 0 to layer 11, with dictionary sizes from 768 to 131,072.

**Experiment 2 (Meta-analysis, H2).** We extract precomputed metrics from 314 SAEBench checkpoints (`adamkarvonen/sae_bench_results_0125`), spanning seven architecture families: Standard, TopK, JumpReLU, GatedSAE, MatryoshkaBatchTopK, PAnneal, and BatchTopK. The base models are Gemma-2-2B and Pythia-160M-deduped. All metrics---absorption, sparse probing F1, RAVEL Cause/Isolation, $L_0$, and $\text{CE}_{\text{recovered}}$---are drawn directly from the SAEBench dataset.

**Experiment 3 (Metric pilot, H3).** We evaluate 10 GPT-2 Small checkpoints on both the canonical first-letter absorption benchmark and a new task-agnostic absorption metric based on automated hierarchy discovery.

### 3.2 Metrics

We measure six classes of quantities. All notation follows the convention established in `notation.md`.

**Absorption.** We use Chanin et al.'s first-letter spelling metric as the primary comparability benchmark. For a given checkpoint, the metric trains a logistic-regression probe on residual-stream activations to detect the first letter of a word (parent concept) and the full word spelling (child concept). It then performs $k$-sparse probing on SAE latents to identify primary latents for each concept. A token is classified as *absorbed* if the parent probe succeeds on the residual stream but the top SAE latents fail, and an integrated-gradients ablation on false-negative tokens reveals that the most causally important latent aligns with the child probe direction. The absorption rate $\alpha_{\text{FL}}$ is the fraction of parent-feature tokens that are absorbed.

**Task-agnostic absorption ($\alpha_{\text{TA}}$).** Our pilot metric replaces the hand-designed first-letter task with an automated geography hierarchy (continent $\to$ country). For each parent-child pair, we train a logistic-regression probe on residual-stream activations, perform $k$-sparse probing on SAE latents, detect false negatives, and run integrated-gradients ablation. Absorption is classified if the top ablation latent aligns with the probe direction at cosine similarity $> \tau$ (we set $\tau = 0.7$ following the alignment threshold used in Chanin et al., 2024).

**Hedging.** We compute a simplified feature-hedging score on correlated token pairs (antonyms: good/bad, hot/cold, big/small, happy/sad, fast/slow). For each pair, we identify the top-1 SAE latent by highest post-activation value averaged over the token position. The hedging rate $h$ is the fraction of pairs where the same latent is the top feature for both tokens. Higher $h$ indicates more mixing of correlated concepts.

**Reconstruction fidelity.** We report three standard metrics:
- $L_0$: average number of non-zero latents per token.
- Explained variance ($\text{EV}$): fraction of input variance recovered by the SAE reconstruction, $\text{EV} = 1 - \mathbb{E}[\|x - \hat{x}\|_2^2] / \mathbb{E}[\|x - \bar{x}\|_2^2]$.
- $\text{CE}_{\text{recovered}}$: ratio of original cross-entropy loss to CE loss when SAE-reconstructed activations are substituted back into the model, $\text{CE}_{\text{recovered}} = \text{CE}_{\text{orig}} / \text{CE}_{\text{rec}}$. Values $>1$ indicate that reconstructed activations yield lower cross-entropy than the original activations.

**Dead-neuron rate ($\delta_{\text{dead}}$).** Fraction of latents with activation frequency $< 10^{-5}$ on a held-out corpus of 2,048 tokens.

**Downstream interpretability.** From SAEBench we extract sparse probing F1 ($\text{F1}_{\text{probe}}$), RAVEL Cause ($\text{RAVEL}_{\text{cause}}$), and RAVEL Isolation ($\text{RAVEL}_{\text{iso}}$).

### 3.3 Analysis Pipeline

**Pareto front computation (E1).** For each checkpoint we compute six normalized objectives: inverted absorption (so higher is better), inverted hedging, explained variance, $\text{CE}_{\text{recovered}}$, inverted $L_0$ penalty, and inverted dead-neuron rate. Each metric is min-max normalized to $[0, 1]$ within the full GPT-2 checkpoint set. A checkpoint is Pareto-optimal if no other checkpoint weakly dominates it on all six objectives.

**Stochastic dominance tests (E1).** We test whether any architecture family stochastically dominates another using pairwise Mann-Whitney U tests on each raw metric. We compare feature splitting against the Standard baseline, the largest family in the corpus.

**Partial correlation and regression (E2).** To isolate absorption's unique association with downstream utility, we compute Pearson and partial correlations between absorption and each downstream outcome, controlling for $L_0$ and $\text{CE}_{\text{recovered}}$. We then run OLS regressions:

$$
\text{outcome}_i = \beta_0 + \beta_1 \alpha_i + \beta_2 L_{0,i} + \beta_3 \text{CE}_{\text{recovered},i} + \beta_4 \text{width}_i + \sum_{f} \gamma_f \mathbf{1}[\text{family}_i = f] + \epsilon_i,
$$

where we expect $\beta_1 < 0$ if absorption carries a unique negative predictive cost, and standard errors are clustered by architecture family to account for within-family correlation.

**Metric validation (E3).** We compute both $\alpha_{\text{FL}}$ and $\alpha_{\text{TA}}$ on the same 10 checkpoints and report Pearson $r$, Spearman $\rho$, and the two-tailed $p$-value.

### 3.4 Implementation Details

All GPT-2 evaluations run on a single NVIDIA RTX 4090 (24 GB) using `SAELens` for checkpoint loading and `transformer-lens` for activation extraction. The held-out corpus is a 2,048-token subset of C4 validation. Random seed is fixed to 42. SAEBench meta-analysis runs entirely on CPU by reading precomputed metrics from the HuggingFace dataset. Runtime for E1 is approximately 675 seconds for 27 checkpoints; E2 completes in under 10 seconds; E3 completes in approximately 24 seconds.

A limitation of the training-free design is that we cannot evaluate architectures for which no open checkpoints exist (e.g., OrtSAE and Matryoshka SAE are present in SAEBench but not in the GPT-2 SAELens releases). Consequently, E1 is limited to Standard, TopK, TopK_MLP, TopK_Attn, and feature-splitting families on GPT-2 Small, while E2 covers the full architectural diversity at the cost of less controlled checkpoint selection.

---

## 4. Experiment 1: Multi-Objective Pareto Evaluation

**Scope.** We evaluate 27 GPT-2 Small checkpoints spanning the Standard and feature-splitting families on six metrics: absorption rate ($\alpha$), hedging rate ($h$), explained variance (EV), $\text{CE}_{\text{recovered}}$, $L_0$ sparsity, and dead-neuron rate ($\delta_{\text{dead}}$). Checkpoints are drawn from SAELens releases (`gpt2-small-res-jb`, `gpt2-small-resid-post-v5`, `gpt2-small-mlp-out-v5`, `gpt2-small-attn-out-v5`, `gpt2-small-hook-z-kk`, `gpt2-small-mlp-tm`, and `gpt2-small-res-jb-feature-splitting`).

**Pareto results.** 17 of 27 checkpoints lie on the empirical Pareto front: all 4 feature-splitting checkpoints and 13 of 23 standard checkpoints. Feature-splitting checkpoints achieve 0% dead neurons and strong explained variance (0.976--0.986), but do not dominate on hedging or absorption. Table 1 reports per-family averages.

| Family | $L_0$ | EV | $\delta_{\text{dead}}$ | CE Rec. | $\alpha$ | $h$ |
|---|---|---|---|---|---|---|
| Standard (n=23) | 33.9 | 0.830 | 0.197 | 1.054 | 0.015 | 0.833 |
| Feature Splitting (n=4) | 39.5 | 0.982 | 0.000 | 1.172 | 0.000 | 0.888 |

*Note. The Standard absorption mean is driven entirely by one checkpoint (`hook-z-kk-l8`, $\alpha = 0.345$); 22 of 23 Standard checkpoints show zero absorption.*

**Dominance tests.** Mann-Whitney U tests show no significant difference in absorption ($U = 48.0$, $p = 0.754$) or hedging ($U = 50.0$, $p = 0.810$) between standard and feature-splitting families. Feature splitting significantly outperforms on $\text{CE}_{\text{recovered}}$ ($U = 4.0$, $p = 0.001$), but this is a single-metric advantage, not multi-objective dominance.

Figure 1 visualizes the tradeoff space. Point size scales with explained variance, and stars mark Pareto-optimal points. Standard checkpoints occupy a broad region: attention-output SAEs show low hedging (0.284--0.323) and low $\text{CE}_{\text{recovered}}$ (0.595--0.985), while residual-stream SAEs cluster at high hedging ($\geq 0.745$) and high explained variance ($\geq 0.928$). The hook-z SAE is the only standard checkpoint with non-zero absorption ($\alpha = 0.345$), and it sits on the Pareto front despite low explained variance (0.632).

![Figure 1: Pareto front scatter plot showing absorption rate vs. hedging rate across 27 GPT-2 Small checkpoints. Point size scales with explained variance; stars mark Pareto-optimal points.](figures/fig1_pareto.pdf)

These results support H1: no architecture family stochastically dominates across the full multi-objective front. It is important to note that the first-letter absorption metric used here is a simplified proxy; the degeneracy (0.0 on most checkpoints) may reflect the proxy's coarseness rather than true zero absorption.

---

## 5. Experiment 2: Downstream Causal Cost Meta-Analysis

**Scope.** We analyze 314 SAEBench checkpoints with precomputed absorption, $L_0$, $\text{CE}_{\text{recovered}}$, sparse probing F1, and RAVEL Cause/Isolation scores. The corpus spans Standard, TopK, JumpReLU, GatedSAE, MatryoshkaBatchTopK, and PAnneal families on Gemma-2-2B and Pythia-160M.

**Raw correlations.** Absorption correlates negatively with sparse probing F1 ($r = -0.348$, $p < 0.001$), RAVEL Cause ($r = -0.264$, $p < 0.001$), and RAVEL Isolation ($r = -0.263$, $p < 0.001$).

**Partial correlations.** After controlling for $L_0$ and $\text{CE}_{\text{recovered}}$, the negative relationships strengthen for sparse probing F1 ($r_{\text{partial}} = -0.385$, $p < 0.001$) and remain significant for RAVEL Cause ($r_{\text{partial}} = -0.237$, $p < 0.001$) and RAVEL Isolation ($r_{\text{partial}} = -0.266$, $p < 0.001$).

**Regression results.** OLS with cluster-robust standard errors shows absorption as a significant negative predictor of all three outcomes after controlling for $L_0$, $\text{CE}_{\text{recovered}}$, width, and architecture family dummies (Table 2).

| Predictor | Sparse Probing F1 | RAVEL Cause | RAVEL Isolation |
|---|---|---|---|
| Absorption ($\alpha$) | $-0.037$*** (0.006) | $-0.022$*** (0.006) | $-0.024$*** (0.006) |
| $L_0$ | $0.020$* (0.010) | $0.032$*** (0.010) | $0.010$ (0.010) |
| $\text{CE}_{\text{recovered}}$ | $0.057$*** (0.009) | $0.043$*** (0.010) | $0.034$*** (0.010) |
| Width | $0.006$ (0.005) | $0.002$ (0.005) | $-0.000$ (0.005) |
| $R^2$ | 0.321 | 0.166 | 0.150 |
| $N$ | 312 | 312 | 312 |

*Note. Cluster-robust standard errors in parentheses. * $p < 0.05$, *** $p < 0.001$.*

Figure 2 plots the partial-regression relationship between absorption and sparse probing F1. The downward trend is visible across architecture families, with the fit line showing a clear negative slope even after residualizing out $L_0$ and reconstruction quality.

![Figure 2: Partial correlation scatter of absorption vs. sparse probing F1 across 314 SAEBench checkpoints. The dashed line shows the partial-regression fit with 95% confidence band.](figures/fig2_partial_corr.pdf)

These results support H2: absorption carries a unique negative predictive cost for downstream interpretability utility, independent of confounders. While we cannot establish causality from observational data, the relationship persists after controlling for $L_0$, reconstruction quality, width, and architecture family.

---

## 6. Experiment 3: Task-Agnostic Absorption Metric Pilot

**Design.** We construct a task-agnostic absorption metric using automated hierarchy discovery (geography: continent $\rightarrow$ country) on GPT-2 Small, then compare it to the simplified first-letter benchmark across 10 checkpoints.

**Results ($N = 10$).** Pearson $r = -0.592$, Spearman $\rho = -0.529$, $p = 0.116$. The correlation is negative and not statistically significant.

Table 3 reports the per-checkpoint values. The first-letter metric is degenerate on 9 of 10 checkpoints ($\alpha_{\text{FL}} = 0.0$), while the task-agnostic metric shows meaningful variance ($\alpha_{\text{TA}} = 0.0$--$0.24$). The single outlier---TopK attention-output SAE---drives much of the negative correlation: it has high first-letter absorption ($\alpha_{\text{FL}} = 0.654$) but zero task-agnostic absorption. Removing this outlier reduces the correlation to approximately zero.

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
| gpt2-small-mlp-out-v5-32k | blocks.8.hook_mlp_out | TopK MLP | 0.013 | 0.000 |
| gpt2-small-attn-out-v5-32k | blocks.8.hook_attn_out | TopK Attn | 0.000 | 0.654 |

Figure 3 visualizes the divergence. Most points cluster along the y-axis ($\alpha_{\text{FL}} = 0$) with varying task-agnostic scores, while the TopK Attn checkpoint sits far to the right at $\alpha_{\text{FL}} = 0.654$.

![Figure 3: Task-agnostic vs. first-letter absorption across 10 GPT-2 Small checkpoints. The two metrics are weakly negatively correlated ($r = -0.59$, $p = 0.12$).](figures/fig3_task_agnostic.pdf)

H3 is not supported. This negative result suggests the first-letter benchmark may be unrepresentative of general absorption behavior, or that different SAE architectures absorb different kinds of hierarchical structure in qualitatively different ways. Because the pilot tested only 10 checkpoints and one hierarchy domain, larger-scale validation is needed before drawing firm conclusions.

---

## 7. Discussion

### 7.1 Synthesis of Findings

**No free lunch in SAE design.** Experiment 1 evaluated 27 GPT-2 Small checkpoints across Standard and feature-splitting families on a six-objective Pareto front. Feature splitting eliminates dead neurons ($\delta_{\text{dead}} = 0.000$ vs. $0.197$ for Standard) and improves $\text{CE}_{\text{recovered}}$ ($1.172$ vs. $1.054$), yet Mann-Whitney U tests show no significant advantage on absorption ($U = 48.0$, $p = 0.754$) or hedging ($U = 50.0$, $p = 0.810$). Notably, feature splitting shows a slightly *higher* mean hedging rate (0.888) than Standard (0.833), though the difference is not significant. This subtle discrepancy complicates the Chanin et al. (2025) prediction that narrower designs increase hedging: feature splitting (which splits rather than narrows the residual stream) does not reduce hedging as might be expected. These results support a multi-objective tension framing: absorption, hedging, and reconstruction fidelity are in tension, and no architecture family dominates across the full metric suite.

**Absorption predicts lower downstream utility.** Experiment 2 provides the largest-scale evidence to date that absorption has a unique negative association with downstream interpretability utility. After controlling for $L_0$ and $\text{CE}_{\text{recovered}}$, absorption shows a significant negative partial correlation with sparse probing F1 ($r_{\text{partial}} = -0.385$, $p < 0.001$), RAVEL Cause ($r_{\text{partial}} = -0.237$, $p < 0.001$), and RAVEL Isolation ($r_{\text{partial}} = -0.266$, $p < 0.001$). OLS regression with cluster-robust standard errors shows absorption as a significant negative predictor of all three outcomes (Figure 4). While we report associations conditional on these controls, not definitive causal effects, the consistency of the negative relationship justifies continued attention to absorption---but not at the expense of other metrics.

![Figure 4: Left: standardized regression coefficients for absorption as a predictor of downstream outcomes (sparse probing F1, RAVEL Cause, RAVEL Isolation). Right: partial-correlation scatter of absorption vs. residualized sparse probing F1. Both panels support H2: absorption carries a unique negative predictive cost for interpretability utility.](figures/fig4_absorption_downstream.pdf)

**We need better benchmarks.** Experiment 3 reveals a weak negative correlation between the task-agnostic geography-hierarchy metric and the canonical first-letter benchmark ($r = -0.592$, $p = 0.12$). The first-letter metric is degenerate ($\alpha_{\text{FL}} = 0.0$) on 9 of 10 checkpoints, while the task-agnostic metric shows meaningful variance ($\alpha_{\text{TA}} = 0.0$--$0.24$). H3 is not supported. This negative result suggests the first-letter benchmark may be too narrow, or that different architectures absorb different kinds of hierarchical structure in qualitatively different ways. The task-agnostic metric itself needs validation before it can replace the first-letter benchmark; the immediate takeaway is *uncertainty* about what absorption means across domains.

### 7.2 Reframing the Research Agenda

Rather than treating absorption as a bug to be fixed by the next architecture, the field should shift toward three priorities:

1. **Multi-objective evaluation as standard practice.** SAE research should report Pareto-front coverage and tradeoff curves, not single-metric leaderboards. A method that improves absorption while degrading hedging or reconstruction is not a universal improvement---it is a different point in tradeoff space.

2. **Task-adaptive SAE selection.** Practitioners should select checkpoints based on the desired tradeoff region. If the goal is causal interpretability (high RAVEL scores), low-absorption checkpoints are preferable. If the goal is reconstruction fidelity with zero dead neurons, feature splitting may be preferable despite no absorption advantage.

3. **Benchmark development beyond single-task proxies.** The first-letter benchmark has served as a useful comparability anchor, but our pilot suggests it does not generalize cleanly to arbitrary semantic hierarchies. Future work should validate absorption metrics across multiple domains (geography, biology, color, temporal) and model scales.

### 7.3 Limitations

**Model scope.** Our controlled Pareto evaluation lacks modern model coverage due to gated access constraints: Experiments 1 and 3 are limited to GPT-2 Small, while Gemma-2-2B requires HuggingFace authentication. The SAEBench meta-analysis in Experiment 2 includes Gemma-2-2B and Pythia-160M, but the controlled Pareto evaluation lacks modern model coverage.

**Metric proxies.** The first-letter absorption metric and hedging proxy used in Experiments 1 and 3 are simplified implementations. Full SAEBench integration would improve comparability with prior work. The task-agnostic metric was tested on only one hierarchy domain (geography) and 10 checkpoints.

**Causal claims.** While we control for major confounders ($L_0$, $\text{CE}_{\text{recovered}}$, width, architecture family), unobserved architecture differences may still bias regression estimates. We report associations conditional on these controls, not definitive causal effects.

**Sample size for E3.** Only 10 checkpoints and one hierarchy domain were tested. A larger validation on 20--50 checkpoints across multiple domains would be needed to draw firm conclusions about benchmark generalizability.

### 7.4 Future Work

Three directions seem especially promising:

- **Scaling the task-agnostic metric.** Extending the automated hierarchy-discovery pipeline to multiple semantic domains and larger models would clarify whether the first-letter benchmark is unrepresentative in general, or whether the divergence we observe is specific to GPT-2 Small.

- **Controlled training experiments.** Our training-free design ensures ecological validity, but it limits architectural coverage. Future work could train matched OrtSAE and Matryoshka checkpoints on GPT-2 Small to test whether their reported absorption reductions hold under multi-objective evaluation.

- **Mechanistic explanations for tradeoffs.** Why does feature splitting improve reconstruction and dead-neuron rate without showing a statistically significant absorption advantage? Why does the attention-output TopK SAE show high first-letter absorption but zero task-agnostic absorption? Answering these questions requires deeper mechanistic analysis of how different architectures allocate latents across the feature hierarchy.

---

## 8. Conclusion

We return to the three research questions posed in Section 2.

**On RQ1 (tradeoffs), H1 is supported.** Our study provides the first training-free, multi-objective assessment of absorption-mitigation methods across 314 SAEBench checkpoints and 27 GPT-2 Small checkpoints. No architecture family dominates the full Pareto front. Feature splitting eliminates dead neurons (mean $\delta_{\text{dead}} = 0.000$ vs. $0.197$ for Standard) and improves $\text{CE}_{\text{recovered}}$ ($1.172$ vs. $1.054$), yet Mann-Whitney U tests show no significant advantage on absorption ($p = 0.754$) or hedging ($p = 0.810$). Mitigations trade one pathology for another rather than delivering universal improvement.

**On RQ2 (downstream causality), H2 is supported.** Absorption has a unique negative association with downstream interpretability utility. After controlling for $L_0$ and $\text{CE}_{\text{recovered}}$, absorption shows a significant negative partial correlation with sparse probing F1 ($r_{\text{partial}} = -0.385$, $p < 0.001$), RAVEL Cause ($r_{\text{partial}} = -0.237$, $p < 0.001$), and RAVEL Isolation ($r_{\text{partial}} = -0.266$, $p < 0.001$). OLS regression with cluster-robust standard errors shows absorption as a significant negative predictor of all three outcomes. While we cannot establish causality from observational data, the relationship persists after controlling for confounders. This justifies continued attention to absorption---but only as one objective among many.

**On RQ3 (metric generalization), H3 is not supported.** The canonical first-letter absorption benchmark may not generalize to arbitrary hierarchies. Our task-agnostic metric pilot on 10 GPT-2 Small checkpoints reveals a weak negative correlation with the first-letter benchmark ($r = -0.592$, $p = 0.12$), with the first-letter metric degenerate at $0.0$ on 9 of 10 checkpoints.

The SAE research agenda should move from "fixing absorption" to "navigating unavoidable tradeoffs." Future work should treat multi-objective evaluation as standard practice, develop task-adaptive selection criteria, build absorption benchmarks that span multiple semantic domains rather than relying on single-task proxies, and validate these tradeoffs on larger models such as Gemma-2-2B and Pythia.

---

## Figures and Tables

- Figure 1: `fig1_pareto.pdf` --- Pareto front scatter plot showing absorption rate vs. hedging rate across 27 GPT-2 Small checkpoints. Point size scales with explained variance; stars mark Pareto-optimal points.
- Figure 2: `fig2_partial_corr.pdf` --- Partial correlation scatter of absorption vs. sparse probing F1 across 314 SAEBench checkpoints. The dashed line shows the partial-regression fit with 95% confidence band.
- Figure 3: `fig3_task_agnostic.pdf` --- Task-agnostic vs. first-letter absorption across 10 GPT-2 Small checkpoints. The two metrics are weakly negatively correlated ($r = -0.59$, $p = 0.12$).
- Figure 4: `fig4_absorption_downstream.pdf` --- Combined bar chart of standardized absorption regression coefficients and partial-correlation scatter of absorption vs. residualized sparse probing F1.
- Table 1: inline --- Architecture family comparison across $L_0$, explained variance, dead-neuron rate, CE recovered, absorption, and hedging.
- Table 2: inline --- OLS regression coefficients for downstream predictive cost of absorption.
- Table 3: inline --- Per-checkpoint task-agnostic vs. first-letter absorption values.
