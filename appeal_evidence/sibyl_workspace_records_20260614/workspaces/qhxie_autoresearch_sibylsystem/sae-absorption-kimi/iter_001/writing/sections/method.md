# 3. Methodology

All experiments in this paper follow a **training-free evaluation** paradigm: we use publicly released pretrained SAE checkpoints without training any new SAEs. This design ensures that our conclusions reflect the behavior of existing, widely used checkpoints rather than idiosyncrasies of our own training pipeline. The analysis spans 341 total checkpoints: 27 GPT-2 Small checkpoints from SAELens for controlled Pareto evaluation, and 314 checkpoints from SAEBench for large-scale meta-analysis.

## 3.1 Checkpoint Corpus

**Experiment 1 (Pareto evaluation, H1).** We evaluate 27 GPT-2 Small (117M) checkpoints released via SAELens. The corpus includes four architecture families: Standard ($L_1$-penalized), TopK, TopK_MLP (MLP-output hook), and TopK_Attn (attention-output hook). We also include four feature-splitting checkpoints (`res-jb-fs-{768,1536,3072,6144}`) that split the residual stream into subspaces before applying separate SAEs. Layer coverage ranges from layer 0 to layer 11, with dictionary sizes from 768 to 131,072. Table 1 summarizes the family-level composition.

**Experiment 2 (Meta-analysis, H2).** We extract precomputed metrics from 314 SAEBench checkpoints (`adamkarvonen/sae_bench_results_0125`), spanning seven architecture families: Standard, TopK, JumpReLU, GatedSAE, MatryoshkaBatchTopK, PAnneal, and BatchTopK. The base models are Gemma-2-2B and Pythia-160M-deduped. All metrics---absorption, sparse probing F1, RAVEL Cause/Isolation, L0, and CE loss recovered---are drawn directly from the SAEBench dataset.

**Experiment 3 (Metric pilot, H3).** We evaluate 10 GPT-2 Small checkpoints (the same set as Experiment 1) on both the canonical first-letter absorption benchmark and a new task-agnostic absorption metric based on automated hierarchy discovery.

## 3.2 Metrics

We measure six classes of quantities. All notation follows the convention established in `notation.md`.

**Absorption.** We use Chanin et al.'s first-letter spelling metric as the primary comparability benchmark. For a given checkpoint, the metric trains a logistic-regression probe on residual-stream activations to detect the first letter of a word (parent concept) and the full word spelling (child concept). It then performs $k$-sparse probing on SAE latents to identify primary latents for each concept. A token is classified as *absorbed* if the parent probe succeeds on the residual stream but the top SAE latents fail, and an integrated-gradients ablation on false-negative tokens reveals that the most causally important latent aligns with the child probe direction. The absorption rate $\alpha_{\text{FL}}$ is the fraction of parent-feature tokens that are absorbed.

**Task-agnostic absorption ($\alpha_{\text{TA}}$).** Our pilot metric replaces the hand-designed first-letter task with an automated geography hierarchy (continent $\to$ country). For each parent-child pair, we train a logistic-regression probe on residual-stream activations, perform $k$-sparse probing on SAE latents, detect false negatives, and run integrated-gradients ablation. Absorption is classified if the top ablation latent aligns with the probe direction at cosine similarity $> \tau$ (we set $\tau = 0.7$).

**Hedging.** We compute a simplified feature-hedging score on correlated token pairs (antonyms). For each pair, we identify the top-1 SAE latent for both tokens. The hedging rate $h$ is the fraction of pairs where the same latent is the top feature for both tokens. Higher $h$ indicates more mixing of correlated concepts.

**Reconstruction fidelity.** We report three standard metrics:
- $L_0$: average number of non-zero latents per token.
- Explained variance ($\text{EV}$): fraction of input variance recovered by the SAE reconstruction, $\text{EV} = 1 - \mathbb{E}[\|x - \hat{x}\|_2^2] / \mathbb{E}[\|x - \bar{x}\|_2^2]$.
- CE loss recovered ($\text{CE}_{\text{recovered}}$): ratio of original cross-entropy loss to CE loss when SAE-reconstructed activations are substituted back into the model, $\text{CE}_{\text{recovered}} = \text{CE}_{\text{orig}} / \text{CE}_{\text{rec}}$.

**Dead-neuron rate ($\delta_{\text{dead}}$).** Fraction of latents with activation frequency $< 10^{-5}$ on a held-out corpus of 2,048 tokens.

**Downstream interpretability.** From SAEBench we extract sparse probing F1 ($\text{F1}_{\text{probe}}$), RAVEL Cause ($\text{RAVEL}_{\text{cause}}$), and RAVEL Isolation ($\text{RAVEL}_{\text{iso}}$).

## 3.3 Analysis Pipeline

**Pareto front computation (E1).** For each checkpoint we compute six normalized objectives: inverted absorption (so higher is better), inverted hedging, explained variance, CE loss recovered, inverted $L_0$ penalty, and inverted dead-neuron rate. Each metric is min-max normalized to $[0, 1]$ within the full GPT-2 checkpoint set. A checkpoint is Pareto-optimal if no other checkpoint weakly dominates it on all six objectives.

**Stochastic dominance tests (E1).** We test whether any architecture family stochastically dominates another using pairwise Mann-Whitney U tests on each raw metric. We compare feature splitting against the Standard baseline, the largest family in the corpus.

**Partial correlation and regression (E2).** To isolate absorption's unique effect on downstream utility, we compute Pearson and partial correlations between absorption and each downstream outcome, controlling for $L_0$ and CE loss recovered. We then run OLS regressions:

$$
\text{outcome}_i = \beta_0 + \beta_1 \alpha_i + \beta_2 L_{0,i} + \beta_3 \text{CE}_{\text{recovered},i} + \beta_4 \text{width}_i + \sum_{f} \gamma_f \mathbf{1}[\text{family}_i = f] + \epsilon_i,
$$

where standard errors are clustered by architecture family to account for within-family correlation.

**Metric validation (E3).** We compute both $\alpha_{\text{FL}}$ and $\alpha_{\text{TA}}$ on the same 10 checkpoints and report Pearson $r$, Spearman $\rho$, and the two-tailed $p$-value.

## 3.4 Implementation Details

All GPT-2 evaluations run on a single NVIDIA RTX 4090 (24 GB) using `SAELens` for checkpoint loading and `transformer-lens` for activation extraction. The held-out corpus is a 2,048-token subset of C4 validation. Random seed is fixed to 42. SAEBench meta-analysis runs entirely on CPU by reading precomputed metrics from the HuggingFace dataset. Runtime for E1 is approximately 675 seconds for 27 checkpoints; E2 completes in under 10 seconds; E3 completes in approximately 24 seconds.

A limitation of the training-free design is that we cannot evaluate architectures for which no open checkpoints exist (e.g., OrtSAE and Matryoshka SAE are present in SAEBench but not in the GPT-2 SAELens releases). Consequently, E1 is limited to Standard, TopK, and feature-splitting families on GPT-2 Small, while E2 covers the full architectural diversity at the cost of less controlled checkpoint selection.

![Overview of the training-free multi-objective evaluation pipeline. Checkpoints are loaded from SAELens and SAEBench; metrics are computed without SAE training; outputs feed into Pareto analysis, dominance tests, partial-correlation meta-analysis, and task-agnostic metric validation.](figures/fig_method_pipeline.pdf)

<!-- FIGURES
- Figure 1: fig_method_pipeline_desc.md, fig_method_pipeline.pdf — Overview of the training-free evaluation pipeline combining SAELens/SAEBench checkpoints, metric computation, and downstream analysis stages.
-->
