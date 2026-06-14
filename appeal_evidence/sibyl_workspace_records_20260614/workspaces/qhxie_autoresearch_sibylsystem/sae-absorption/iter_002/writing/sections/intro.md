# Introduction

Sparse autoencoders (SAEs) have become the primary tool for decomposing neural network
activations into interpretable features, with applications ranging from circuit discovery
to safety auditing of large language models
\cite{bricken2023monosemanticity, conmy2023towards, anthropic2024scaling}.
Their utility rests on the assumption that each SAE latent reliably activates whenever its
corresponding concept is present in the input. Feature absorption breaks this assumption:
a child feature in a parent-child hierarchy fails to fire because the parent feature has
absorbed the activation signal \cite{chanin2024absorption}. The child latent appears to
encode a concept, but that encoding is systematically suppressed on 92–97\% of inputs where
the concept is present (Section~\ref{sec:experiments}). An SAE audit that trusts activation
presence alone will conclude that the child concept is absent when it is not.

The existing detection method for feature absorption, the FeatureAbsorptionCalculator of
Chanin et al. \citeyear{chanin2024absorption}, requires pre-specified probe directions trained
on labeled data: the researcher must know in advance which concept to look for and must hold
out activation data to train a probe. This dependence on external supervision has two
consequences. First, it precludes systematic absorption auditing of all $d_\text{sae}$ latents
in a deployed SAE: for a 24,576-width SAE, evaluating every potential parent-child pair would
require $O(d_\text{sae}^2)$ probe-training and activation-collection steps. Second, it cannot
detect absorption of concepts the researcher did not anticipate auditing---including
safety-relevant features that an adversarial input might exploit.

We ask whether the SAE weight matrices alone---encoder $E \in \mathbb{R}^{d_\text{sae} \times d}$
and decoder $D \in \mathbb{R}^{d \times d_\text{sae}}$---can identify absorbed features,
without probes, without activation data, and without knowing which concept to look for.

**Our answer is yes, via encoder-decoder dissociation (EDA).** In a healthy, non-absorbed
feature, the encoder direction $\hat{e}_j$ and decoder direction $d_j$ point in the same
direction: the feature detects and reconstructs the same concept. In an absorbed feature,
the child encoder is pulled toward the parent decoder direction by gradient pressure from
parent-only contexts, while the child decoder remains anchored to the child concept by
reconstruction loss. The resulting intra-feature misalignment, $\text{EDA}(j) = 1 -
\cos(\hat{e}_j, d_j)$, is the geometric fingerprint of absorption. Figure~\ref{fig:method}
illustrates this geometry.

This paper makes three contributions.

**Contribution 1: Theory.** We prove that the rate-distortion training objective produces an
absorption preference when the sparsity penalty $\lambda$ exceeds $\sin^2(\theta_{p,c})$,
the squared sine of the decoder angle between parent $p$ and child $c$ (Proposition~1).
A critical and counterintuitive corollary: the co-occurrence frequency $p_\text{co}$ cancels
from the threshold, so absorption risk is determined entirely by decoder geometry and sparsity
penalty, not by how often parent and child concepts co-occur. We also derive a mechanistic
conjecture (Proposition~2) explaining why absorbed child features develop EDA: the child
encoder is pulled toward the parent decoder direction during training.

**Contribution 2: Empirical validation.** On GPT-2 Small layer 6, 24,576-width SAE,
EDA achieves AUROC = 0.650 against exact FeatureAbsorptionCalculator labels ($n_+ = 18$,
$z = 2.49$ above permutation null). A cross-directional variant, $\cos(\hat{e}_p, d_c)$
(parent encoder aligned with child decoder), achieves AUROC = 0.730 (Cohen's $d = 0.552$,
$p = 2.8 \times 10^{-9}$). We also report a negative result that strengthens the signal:
the decoder-decoder cosine predictor fails entirely (AUROC = 0.206), demonstrating that
absorption geometry lives in the encoder, not between decoders. EDA inverts at layer 10
(AUROC = 0.256, Cohen's $d = -0.890$), consistent with post-absorption encoder re-alignment
at late layers.

**Contribution 3: Phase characterization.** Absorption rates across all 11 tested SAE
configurations (layers 2--10, widths 12k--98k) lie in the narrow range 0.919--0.968.
A likelihood-ratio test finds no evidence for a sparsity-dependent phase transition
(LRT $p = 0.456$; BIC difference = $-3.22$ against a sigmoid model), and Spearman
$\rho = 0.191$ ($p = 0.574$) confirms the absence of sparsity dependence. AJT-trained SAEs
show reversed EDA polarity (negative EDA delta), demonstrating that the training regime
is a critical variable in absorption geometry. Semantic hierarchy absorption is absent in
GPT-2 Small (animate-inanimate, noun-proper: ratio-to-null = 1.0), scoping absorption to
orthographic hierarchies at this model scale.

Together, these results provide the first weight-only, probe-free diagnostic for SAE feature
absorption, characterize its geometric mechanism, and establish that absorption is phase-stable
across the standard SAE hyperparameter space. We first characterize the theoretical conditions
under which absorption is preferred before comparing the results with our geometric predictions
(Section~\ref{sec:related}), formalizing the theoretical framework and experimental setup
(Section~\ref{sec:method}), reporting detection and characterization experiments
(Section~\ref{sec:experiments}), and synthesizing findings and limitations
(Sections~\ref{sec:discussion}--\ref{sec:conclusion}).

![Method diagram: non-absorbed vs. absorbed feature geometry](figures/fig1_eda_method.pdf)

<!-- FIGURES
- Figure 1: fig1_eda_method.pdf — Method diagram illustrating non-absorbed (EDA≈0) vs. absorbed (high EDA) feature geometry with formula inset; sourced from exp/results/full/fig1_eda_method.pdf
-->
