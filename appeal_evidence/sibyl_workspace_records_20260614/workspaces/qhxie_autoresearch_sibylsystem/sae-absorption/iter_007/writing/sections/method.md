# 3 Methodology

## 3.1 Models and SAE Configurations

Table 1 summarizes the Sparse Autoencoder (SAE) configurations used in this study. Our primary analysis uses Gemma 2 2B ($d_{\text{model}} = 2304$) with JumpReLU SAEs from Gemma Scope (Lieberum et al., 2024). These SAEs use hard per-latent activation thresholds $\theta_j$: latent $j$ fires with activation $z_j$ if $W_{e,j} x + b_{e,j} > \theta_j$, and outputs exactly zero otherwise. This binary activation regime contrasts with L1-ReLU SAEs, where a continuous penalty produces graded suppression.

We systematically vary three axes:

1. **$L_0$ operating point.** We test four configurations at layer 12 with $d_{\text{SAE}} = 16{,}384$: $L_0 \in \{22, 41, 82, 176\}$. This range spans capacity-starved ($L_0 = 22$, only 22 active latents representing $\sim$2304-dimensional inputs) to capacity-sufficient ($L_0 = 176$) regimes.

2. **Layer.** At fixed $L_0 = 82$ and $d_{\text{SAE}} = 16{,}384$, we compare layers 10, 12, and 20 to test whether absorption is layer-specific or controlled by sparsity.

3. **Dictionary width.** At layer 12, we compare $d_{\text{SAE}} = 16{,}384$ with $d_{\text{SAE}} = 65{,}536$ at $L_0 = 82$.

As a secondary reference, we include GPT-2 Small ($d_{\text{model}} = 768$) with L1-ReLU SAEs ($d_{\text{SAE}} = 24{,}576$) at layers 8, 10, and 11 to contextualize our results against the original Chanin et al. (2024) setup. This cross-architecture comparison is confounded by model size, architecture, training data, and $L_0$; we report it as context only.

| Configuration | Model | Layer | $d_{\text{SAE}}$ | $L_0$ | Architecture |
|:---|:---|:---:|:---:|:---:|:---|
| L12-16k-L0\_22 | Gemma 2 2B | 12 | 16,384 | 22 | JumpReLU |
| L12-16k-L0\_41 | Gemma 2 2B | 12 | 16,384 | 41 | JumpReLU |
| L12-16k-L0\_82 | Gemma 2 2B | 12 | 16,384 | 82 | JumpReLU |
| L12-16k-L0\_176 | Gemma 2 2B | 12 | 16,384 | 176 | JumpReLU |
| L12-65k-L0\_82 | Gemma 2 2B | 12 | 65,536 | 82 | JumpReLU |
| L10-16k-L0\_82 | Gemma 2 2B | 10 | 16,384 | 82 | JumpReLU |
| L20-16k-L0\_82 | Gemma 2 2B | 20 | 16,384 | 82 | JumpReLU |
| GPT2-L8 | GPT-2 Small | 8 | 24,576 | -- | L1-ReLU |
| GPT2-L10 | GPT-2 Small | 10 | 24,576 | -- | L1-ReLU |
| GPT2-L11 | GPT-2 Small | 11 | 24,576 | -- | L1-ReLU |

**Table 1.** SAE configurations used in this study. Primary analyses use Gemma 2 2B JumpReLU SAEs at layer 12; cross-layer and cross-width comparisons probe generality. GPT-2 Small L1-ReLU SAEs serve as a confounded secondary reference.

## 3.2 Absorption Measurement Protocol

We follow the measurement protocol of Chanin et al. (2024) with explicit quality controls. The protocol operates on hierarchy domains where a parent concept (e.g., "starts with letter A") subsumes child concepts (individual word tokens).

**Vocabulary.** Our primary domain is first-letter spelling: 1,204 single-token English words across 25 tested letters (X is excluded due to having only 1 token). For confound decomposition analyses that require consistent vocabulary across $L_0$ values, we use a 1,196-word subset tokenized identically at all four operating points.

**Probe training.** For each parent class (letter), we train a $k$-sparse logistic regression probe ($k = 5$) on SAE latent activations. The probe selects the $k$ latents whose decoder columns $d_j$ have the highest cosine similarity $\cos(d_j, v_p)$ with the probe direction $v_p$ and fits a logistic classifier on their activations.

**Quality gate.** We require probe F1 $> 0.85$ per parent class before computing absorption. At $L_0 = 82$, 10 of 25 letters pass this gate; at $L_0 = 22$, all 25 letters achieve F1 $= 1.0$. Letters failing the gate are excluded from aggregate statistics but reported individually for transparency.

**Candidate identification.** A latent $j$ is a candidate absorbing feature if $\cos(d_j, v_p) \geq \tau_{\cos}$ where $\tau_{\cos} = 0.025$ (default). Section 3.6 evaluates sensitivity to this threshold.

**Absorption criterion.** A token is classified as absorbed (false negative, FN) when: (1) the probe correctly classifies it as belonging to the parent class, and (2) all $k$ probe-associated latents have zero activation. The absorption rate is the fraction of parent-positive tokens meeting both conditions.

**Statistical inference.** All confidence intervals are 95\% bootstrap CIs with 10,000 resamples (seed $= 42$). Effect sizes use Cohen's $d$ for group comparisons and Spearman $\rho_s$ for rank correlations. Multiple comparisons are corrected via Bonferroni.

## 3.3 Four-Control Suite

A valid absorption metric should exceed its controls: measured absorption should be higher than shuffled-label absorption, since randomly assigned parent labels should not exhibit systematic false negatives. We implement four controls (C1--C4) to test this.

**C1: Random probe.** A probe trained on a random direction in $\mathbb{R}^{d_{\text{SAE}}}$ rather than a meaningful class boundary. This baseline measures chance-level false-negative rates. Expected: near-zero absorption.

**C2: Shuffled labels.** The same probe pipeline applied to randomly permuted class assignments (labels shuffled across tokens, destroying the true letter--word relationship). Expected: absorption rate $\leq$ measured rate. If shuffled controls exceed measured absorption, the metric cannot distinguish genuine absorption from measurement artifact.

**C3: Dense probe.** An all-feature logistic regression (no sparsity constraint) on the same task. This tests whether the SAE activations contain sufficient information for classification independent of the $k$-sparse selection step. If the dense probe achieves high F1, false negatives in the $k$-sparse probe reflect the sparsity constraint, not missing information.

**C4: Untrained SAE.** Absorption computed using pre-training (randomly initialized) decoder columns. This null control should produce 0\% absorption since random decoder directions have no semantic structure.

## 3.4 Confound Decomposition

Feature absorption and feature hedging produce identical observational signatures: a parent latent fails to fire on a parent-positive input. We introduce confound decomposition to separate these two sources.

**Multi-$L_0$ persistence analysis.** We measure false negatives at all four $L_0$ values $\{22, 41, 82, 176\}$ using the 1,196-word consistent vocabulary. Hedging-driven false negatives should resolve at higher $L_0$ (where the SAE has more capacity), while hierarchy-driven false negatives should persist.

**Permissive hedging classification.** A token is classified as hedging if it ceases to be a false negative at *any* higher $L_0$ value. This is an upper bound: it counts every FN that eventually resolves, regardless of mechanism.

**Strict hedging classification.** We check whether the *specific* $k = 5$ parent-associated latents (those selected by the probe at $L_0 = 22$) fire at $L_0 = 176$. A token is classified as strict hedging only if at least 1 of these 5 latents activates at the highest $L_0$. This is a conservative lower bound: it requires the parent concept to be recoverable through the same latent pathway.

**Persistent core words.** Tokens classified as FN at all four $L_0$ values constitute the persistent core -- candidates for genuine competitive exclusion that cannot be explained by hedging at any tested sparsity level. We identify 8 such words: *eight* (E), *liked* (L), *lower* (L), *offer* (O), *often* (O), *other* (O), *under* (U), *until* (U).

**Shuffled control.** We apply the strict classification to 10 shuffled-label replicates to establish a baseline rate, testing whether the true strict hedging rate exceeds chance via a $z$-test.

## 3.5 Activation Patching

Confound decomposition identifies persistent false negatives but cannot establish causality. The 8 persistent core words might be FN because a child latent actively suppresses the parent (competitive exclusion) or because the SAE simply does not encode the first-letter concept for those tokens (reconstruction failure). Activation patching resolves this ambiguity.

For each persistent core word, we perform three causal interventions at $L_0 = 82$:

1. **Primary patching (decode-reencode).** Identify the strongest child candidate feature (highest activation among features with $\cos(d_j, v_p) \geq 0.025$). Zero the child feature's activation in the SAE encoding $z$, decode through $W_d$ to obtain a modified reconstruction, then re-encode through $W_e$ to check whether any of the 5 parent latents recover (activation goes from 0 to $> 0$).

2. **All-children patching.** Zero *all* absorbing candidate features simultaneously, not just the strongest. This tests whether the parent is suppressed by a combination of child features rather than a single dominant one.

3. **Residual patching.** Subtract the child feature's decoder contribution ($z_c \cdot d_c$) directly from the raw residual stream activation $x$, then re-encode. This tests whether competitive exclusion operates through the decoder geometry rather than the SAE's encoding dynamics.

**Control.** For each word, we zero 10 randomly selected non-child features (features with zero cosine to the probe direction) and verify that parent latents do not spuriously activate.

If $\geq 5/8$ words show parent recovery, competitive exclusion is confirmed at small scale. If $< 3/8$ recover, the all-hedging interpretation is strengthened.

## 3.6 Threshold Sensitivity Analysis

The absorption metric depends on two tunable thresholds: the cosine similarity cutoff $\tau_{\cos}$ for candidate identification and the magnitude gap $\tau_{\text{mag}}$ for absorption confirmation. We evaluate robustness across a $5 \times 4$ parameter grid:

- $\tau_{\cos} \in \{0.01, 0.02, 0.025, 0.03, 0.05\}$
- $\tau_{\text{mag}} \in \{0.5, 1.0, 1.5, 2.0\}$

For each of the 20 cells, we compute the aggregate absorption rate, per-letter absorption rankings (Kendall $\tau$ with default parameters), and whether the shuffled-label control exceeds the measured rate. We report the coefficient of variation (CV) across all cells as a measure of threshold sensitivity.

## 3.7 Conditional Mutual Information Estimation

We estimate Conditional Mutual Information (CMI), $I(X; f_p \mid f_c)$, to quantify how much unique information a parent latent encodes about the input $X$ beyond what the child latent encodes. The theoretical motivation comes from successive refinement (Equitz & Cover, 1991): if $X \to f_c \to f_p$ forms a Markov chain, then CMI $= 0$ and absorption is information-theoretically lossless. When CMI $> 0$, absorption destroys unique parent information.

**Estimation procedure.** We use a $k$-nearest neighbor mutual information estimator ($k_{\text{nn}} = 5$) operating in a $d'$-dimensional subspace formed by the top principal components of the parent and child decoder columns. The primary pre-registered subspace dimension is $d' = 10$. For each letter, we collect SAE activations on 5,000 corpus token positions plus all word tokens for that letter, project into the decoder subspace, and estimate $I(X; f_p \mid f_c) = I(X; f_p, f_c) - I(X; f_c)$.

**Cross-$L_0$ replication.** We estimate CMI at both $L_0 = 82$ (where probe quality varies: 10/25 letters pass F1 $> 0.85$) and $L_0 = 22$ (where all 25 probes achieve F1 $= 1.0$). This design eliminates the probe quality confound and tests whether the CMI--absorption association replicates under controlled conditions.

**Partial correlation.** At $L_0 = 82$, probe F1 correlates with both CMI and absorption rate ($\rho_s = -0.69$ between absorption and F1). We compute partial Spearman $\rho_s$(CMI, absorption $\mid$ probe F1) to control for this confound, with permutation-based $p$-values (10,000 permutations).

**Dimension sensitivity.** We repeat CMI estimation at $d' \in \{10, 20, 30, 50\}$ at both $L_0$ values to test whether the correlation direction is stable across subspace choices.

**Leave-one-out sensitivity.** We perform jackknife analysis, removing each letter in turn and recomputing Spearman $\rho_s$, to test whether any single letter drives the overall correlation.

<!-- FIGURES
- Table 1: inline -- SAE configurations used in the study (model, layer, d_SAE, L0, architecture)
- None (visual elements for this section appear in Sections 4-6; the method section uses inline tables only)
-->
