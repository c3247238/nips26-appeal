# 3 Methodology

## 3.1 Models and SAE Configurations

Table 1 summarizes the SAE configurations. The primary model is Gemma 2 2B ($d_{\text{model}} = 2304$) with Gemma Scope JumpReLU SAEs (Lieberum et al., 2024). Layer 12 is the primary analysis layer, tested across four $L_0$ operating points (22, 41, 82, 176) at 16k width ($d_{\text{SAE}} = 16{,}384$) and one configuration at 65k width ($d_{\text{SAE}} = 65{,}536$). Layers 10 and 20 at 16k width provide cross-layer validation. For cross-architecture comparison, we include GPT-2 Small ($d_{\text{model}} = 768$) with SAELens L1-ReLU SAEs ($d_{\text{SAE}} = 24{,}576$) at layers 8, 10, and 11.

JumpReLU SAEs differ from L1-ReLU SAEs in a way that is directly relevant to absorption measurement. L1-ReLU SAEs apply a continuous penalty, producing graded suppression of latent activations. JumpReLU SAEs impose a hard per-latent threshold $\theta_j$: activations either exceed the threshold and fire or equal zero. This binary activation regime means that a parent latent is either fully active or fully silent---there is no partial suppression. The Chanin metric's cosine and magnitude thresholds were developed on L1-ReLU SAEs, where this binary regime does not hold.

| Config | Model | Layer | $d_{\text{SAE}}$ | $L_0$ | Architecture |
|--------|-------|-------|-------------------|-------|-------------|
| L12-16k-L0=22 | Gemma 2 2B | 12 | 16,384 | 22 | JumpReLU |
| L12-16k-L0=41 | Gemma 2 2B | 12 | 16,384 | 41 | JumpReLU |
| L12-16k-L0=82 | Gemma 2 2B | 12 | 16,384 | 82 | JumpReLU |
| L12-16k-L0=176 | Gemma 2 2B | 12 | 16,384 | 176 | JumpReLU |
| L12-65k | Gemma 2 2B | 12 | 65,536 | varies | JumpReLU |
| L10-16k | Gemma 2 2B | 10 | 16,384 | 82 | JumpReLU |
| L20-16k | Gemma 2 2B | 20 | 16,384 | 82 | JumpReLU |
| GPT2-L8/L10/L11 | GPT-2 Small | 8/10/11 | 24,576 | -- | L1-ReLU |

**Table 1:** SAE configurations. Gemma Scope SAEs provide configurable $L_0$ via per-latent JumpReLU thresholds. GPT-2 SAEs use L1-ReLU with no explicit $L_0$ control.

## 3.2 Absorption Measurement Protocol

We adapt the Chanin et al. (2024) protocol for Gemma 2 2B with the following refinements.

**Vocabulary.** For the first-letter task, we construct a vocabulary of 1,204 single-token alphabetic words across 26 letters (15 letters with 50+ words, 22 letters with $\geq 20$ words; letter X excluded from analysis due to $n = 1$, yielding 25 tested letters and 1,203 tested words at $L_0$=82). The confound decomposition (Section 3.4) uses a separately tokenized vocabulary of 1,196 words (1,195 tested after excluding X). For cross-domain experiments, we use 189 single-token cities across 29 countries from RAVEL (hij/ravel on HuggingFace) and 140 animals across 6 taxonomic classes from WordNet.

**Probe training.** $k$-sparse logistic regression probes ($k = 5$) are trained per parent class on SAE latent activations, following the Chanin et al. protocol. The probe direction $v_p$ is the learned weight vector.

**Quality gate.** Probe F1 $> 0.85$ per parent class is required for inclusion in primary absorption claims. At $L_0$=82 on L12-16k, 10 of 25 letters pass this gate (mean probe F1 = 0.817); at $L_0$=22, all 25 letters achieve F1 = 1.0. Parents failing the gate are reported separately with stratification by quality tier (F1 0.70--0.85 vs. $< 0.70$).

**Main feature identification.** SAE latents with decoder cosine similarity $\cos(d_j, v_p) \geq \tau_{\cos} = 0.025$ to the probe direction are identified as candidate features for the parent.

**Absorption criterion.** A token is classified as absorbed when (1) all $k$ probe-associated latents fail to activate ($z_j = 0$) and (2) the probe correctly classifies the token. Among these false negatives, absorption is confirmed when the highest-activation latent has magnitude ratio $\geq \tau_{\text{mag}} = 1.0$ relative to the second-highest.

**Confidence intervals.** All absorption rates are reported with 95% bootstrap CIs (10,000 resamples, seed = 42).

## 3.3 Control Suite

Four controls assess metric validity:

- **C1 (Random probe):** A random unit-sphere direction replaces the trained probe. Expected: $< 2\%$. Observed: 11.8% on first-letter (L12-16k, $L_0$=82), indicating substantial baseline metric noise.
- **C2 (Shuffled labels):** Parent-class labels are randomly permuted before probe training. Expected: absorption rate lower than measured. Observed: 74.6% on first-letter (mean over 5 shuffles, $\sigma = 2.1\%$), exceeding the true-label rate of 15.96% by 4.7$\times$. This control fails across all 5 tested domains: city-continent (45.2% vs. 6.49%), animal-class (39.3% vs. 1.43%), city-language (18.0% vs. 6.56%), and city-country (10.3% vs. 0.0%).
- **C3 (Dense probe):** Logistic regression on raw model activations provides a probe quality ceiling. Mean dense probe F1 = 0.929 across 25 first-letter classes, with 23 of 25 letters exceeding F1 = 0.85.
- **C4 (Untrained SAE):** Random encoder/decoder of the same dimensions yields 0.0% absorption, confirming the measured signal depends on the trained SAE structure.

## 3.4 Confound Decomposition

We decompose false negatives across four $L_0$ operating points (22, 41, 82, 176) on L12-16k, with 1,195 words tested at each setting and all 25 probes achieving F1 = 1.0 at $L_0$=22. For each false-negative token at each $L_0$, we classify the cause:

- **Hedging.** The token is a false negative at the current $L_0$ but recovers at a higher $L_0$---its parent information is distributed across multiple latents, none clearing the JumpReLU threshold at the current sparsity level.
- **Hierarchy-driven absorption.** The token is a false negative at all four tested $L_0$ values with reconstruction error below $2\sigma$---genuine competitive exclusion that persists regardless of sparsity pressure.
- **Reconstruction error.** The SAE fails to reconstruct the parent direction (residual norm $> 2\sigma$ above the per-$L_0$ mean).

The classification is operational: a false negative that resolves at any higher $L_0$ is hedging; one that persists across all four $L_0$ values is hierarchy-driven. At $L_0$=22, 657 false negatives decompose into 648 hedging (98.6%), 9 hierarchy-driven (1.4%), and 0 reconstruction error. The 9 hierarchy-driven words---including "eight," "lower," "liked," "offer," and "often"---persist at all four $L_0$ values and constitute the persistent core candidates for genuine competitive exclusion.

## 3.5 CMI Estimation

Conditional mutual information $I(X; f_{\text{parent}} \mid f_{\text{child}})$ is estimated via the $k$-nearest neighbor method (Kraskov et al., 2004; $k_{\text{NN}} = 5$). For each of 25 first-letter features, we collect activations from the word vocabulary plus 10,000 corpus tokens and project them onto a $d'$-dimensional subspace spanned by the top decoder directions associated with each letter's parent-child pair. The primary analysis uses $d' = 10$; sensitivity is reported at $d' \in \{20, 30, 50\}$. The absorbed/non-absorbed partition uses an absorption rate threshold from the L12-16k $L_0$=82 improved first-letter results: absorbed letters ($\alpha \geq 0.10$; $n = 13$) versus non-absorbed ($\alpha < 0.05$; $n = 9$), with 3 letters in between excluded from the group comparison.

## 3.6 Cross-Domain Hierarchy Suite

Five hierarchy domains are tested, spanning syntactic, geographic, linguistic, and taxonomic relationships:

| Domain | Parent Feature | $N_{\text{parents}}$ | $N_{\text{children}}$ | Source |
|--------|---------------|---------------------|----------------------|--------|
| First-letter | "starts with X" | 25 | 1,204 | Chanin et al. (2024) |
| City $\to$ Country | "in France" | 28 | 184 | RAVEL |
| City $\to$ Continent | "in Europe" | 6 | 185 | RAVEL |
| City $\to$ Language | "French-speaking" | 18 | 183 | RAVEL |
| Animal $\to$ Class | "mammal" | 6 | 140 | WordNet |

Probes, quality gates (F1 $> 0.85$), and all four controls are applied per domain. Countries with fewer than 5 cities or probe F1 $< 0.50$ are excluded from absorption claims. For hierarchy predictor analysis, per-domain co-occurrence frequency ratio, fan-out, and parent frequency are computed and correlated with absorption rate using Spearman $\rho$ with Bonferroni correction.

Section 4 presents the metric audit results, which constrain the interpretation of all subsequent findings.

<!-- FIGURES
- Table 1: inline --- SAE configurations used in this study
-->
