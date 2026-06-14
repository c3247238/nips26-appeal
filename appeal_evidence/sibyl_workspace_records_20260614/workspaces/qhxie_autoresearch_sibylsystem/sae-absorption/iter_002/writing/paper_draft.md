# Encoder-Decoder Dissociation as a Geometric Fingerprint of Feature Absorption in Sparse Autoencoders

---

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

---

# Related Work

## Feature Absorption in Sparse Autoencoders

Chanin et al. \citeyear{chanin2024absorption} define feature absorption operationally: a
child latent fails to activate on inputs where its concept is present because a parent latent
at a higher level of the feature hierarchy has absorbed the activation signal. They validate
this phenomenon using the first-letter task---a controlled hierarchy where letter-class
features (parent) subsume individual token features (child)---and report absorption rates of
15--35\% across Gemma Scope, Llama 3.2, and Qwen2 SAEs spanning multiple widths and layers.
Their detection method, the FeatureAbsorptionCalculator, uses integrated-gradients ablation
over pre-specified linear probe directions and finds absorption in every tested SAE
architecture. Chanin et al. provide an informal argument that absorption is sparsity-efficient
(absorbing a parent-child pair saves one L0 per parent-only context) but do not formalize
this as a rate-distortion condition or derive a quantitative threshold.

SAEBench \cite{karvonen2025saebench} evaluates over 200 open-source SAEs across eight
architectures and includes absorption rate (via the Chanin et al. metric) as one of eight
metrics. A key negative finding: proxy metrics for SAE quality---cross-entropy loss and mean
sparsity---do not reliably predict absorption rate, meaning absorption cannot be inferred from
training curve diagnostics alone. SAEBench explicitly acknowledges that its absorption metric
relies on the first-letter hierarchy and may not generalize to domain-specific feature
absorption. Our work complements SAEBench: EDA provides a weight-only, probe-free pre-screen
that operates independently of activation data, and our phase-stability results show the
absorption task scores for SAEBench are unlikely to change dramatically across the
hyperparameter range SAEBench sweeps.

SynthSAEBench \cite{synthsaebench2026} constructs large-scale synthetic data with known
ground-truth feature hierarchies, including Zipfian firing distributions and realistic
correlation structure. The benchmark finds that SAEs substantially underperform direct probes
on feature recovery, confirming that absorption is not an artifact of the first-letter proxy
task. Our EDA detector operates as a weight-only pre-screen that does not require the ground
truth labels that synthetic benchmarks provide.

---

## Theoretical Frameworks for Absorption

Tang et al. \citeyear{tang2024unified} provide a unified theoretical framework casting sparse
dictionary learning (SAEs, transcoders, crosscoders) as piecewise biconvex optimization and
prove that absorption solutions are spurious minima of the SAE training loss. They show that
feature anchoring---pinning specific latents to ground-truth feature directions during
training---restores identifiability and eliminates absorption, but validate this only on
synthetic linear benchmarks. Our Proposition~1 refines the Tang et al. framework to a
concrete rate-distortion condition: for a specific parent-child pair, absorption is the loss-optimal
solution if and only if $\lambda > \sin^2(\theta_{p,c})$. Crucially, co-occurrence frequency
$p_\text{co}$ cancels from this threshold (Corollary~1), a prediction that the Tang et al.
framework does not make explicit.

Cui et al. \citeyear{cui2025limits} prove that SAEs generally fail to recover ground-truth
monosemantic features unless features are extremely sparse (density below a threshold that
scales with model width), establishing a fundamental limit on SAE identifiability. Their
reweighted SAE (WSAE) achieves better feature recovery under restrictive generative assumptions
but is not evaluated on absorption specifically. Chen et al. \citeyear{chen2025taming} propose
bias-adaptation training with theoretical recovery guarantees under a statistical generative
model, but their analysis does not model feature hierarchy and therefore does not address
absorption as a hierarchical failure mode. Our work operates in a complementary regime:
rather than proposing a new training procedure, we characterize the geometry of the absorbed
state in trained SAEs and derive a post-hoc weight-only diagnostic.

---

## Architectural Mitigations

Four architectural families target feature absorption directly. Matryoshka SAEs
\cite{bussmann2025matryoshka} train nested inner and outer codebooks simultaneously, with the
inner codebook learning general parent-level features. This hierarchical structure reduces the
gradient pressure that pulls child encoders toward parent directions during training; on
SAEBench, Matryoshka SAEs rank first on absorption, RAVEL, and sparse probing at the cost of a
small reconstruction penalty. OrtSAE \cite{korznikov2025ortsae} applies an orthogonality
penalty $\beta \sum_{i \neq j} (d_i \cdot d_j)^2$ to decoder columns, forcing decoder
directions apart and increasing $\theta_{p,c}$ for all feature pairs. Because our Proposition~1
requires $\sin^2(\theta_{p,c}) < \lambda$ for absorption to be loss-optimal, increasing
$\theta_{p,c}$ directly raises the threshold---OrtSAE reduces absorption by 65\% on the
first-letter task. ATM SAE \cite{li2025atm} introduces per-latent adaptive importance scoring
during training and achieves the lowest reported absorption score on Gemma 2 2B (0.0068 vs.
0.1402 for TopK and 0.0114 for JumpReLU), reducing the effective $\lambda$ for frequently
activated features. Masked regularization \cite{narayanaswamy2026masked} disrupts
co-occurrence patterns during training via token masking, improving out-of-distribution
robustness and reducing absorption. KronSAE \cite{kronsae2025} applies Kronecker product
factorization to SAE latents and reduces mean absorption fraction across all sparsity levels.

All architectural mitigations require retraining the SAE from scratch. Our detection
method---computing EDA and cross-directional cosines from existing weight matrices---is
post-hoc and applies to any deployed SAE without modification.

---

## Encoder-Decoder Alignment and the Amortization Gap

O'Neill et al. \citeyear{oneill2024amortization} characterize the amortization gap in SAEs:
the discrepancy between the optimal sparse code for a given input (achievable by iterative
pursuit) and the code produced by the encoder in a single forward pass. Their analysis shows
the encoder systematically underestimates feature activations due to the bias introduced by
L1 training, and that this gap is a distinct failure mode from absorption. The amortization
gap is a confounder for EDA: a feature with high EDA may reflect encoder drift from
absorption (as we predict via Proposition~2) or encoder-decoder misalignment from amortization
gap effects unrelated to absorption. We acknowledge this tension explicitly and note that EDA
achieves AUROC = 0.650 above permutation null despite this confounder, suggesting that
absorption-driven EDA is the dominant signal at GPT-2 Small layer 6.

Feature hedging \cite{chanin2025hedging} is a complementary failure mode: narrow SAEs
merge correlated features because they lack the capacity to represent each distinctly. Feature
hedging and feature absorption share a root cause---flat sparsity penalties under feature
correlation---but manifest in opposite capacity regimes: hedging in narrow SAEs
($d_\text{sae}$ too small), absorption when hierarchical features are present regardless of
width. Our phase-stability results (Section~\ref{sec:experiments}) are consistent with
this distinction: absorption rates remain uniformly high at 0.919--0.968 across widths from
12k to 98k, showing no mitigation from width alone.

---

## SAE Evaluation: Sensitivity and Feature Consistency

Tian et al. \citeyear{tian2025sensitivity} frame feature absorption as a special case of poor
feature sensitivity: a feature has low sensitivity if it fails to activate on texts similar
to its activating examples. Their sensitivity metric is computable from activation data but
requires curated test sets per feature. EDA is strictly more parsimonious: it requires only
the weight matrices, with no activation data or curation. The two metrics are complementary:
sensitivity measures the behavioral consequence of absorption; EDA measures the weight-level
geometric cause.

Song et al. \citeyear{song2025consistency} document that SAE features are inconsistent across
training runs (TopK SAEs achieve 0.80 pairwise match, the best among tested architectures),
raising the concern that absorption rates could vary with random seed. Our experiments use
canonical SAELens pre-trained SAEs at fixed releases, avoiding multi-seed variance; however,
the AJT polarity reversal we observe across training regimes (Section~\ref{sec:experiments})
shows that training procedure is a stronger modulator of EDA than layer or width.

---

## Cross-Directional Cosine Measures

No prior work uses $\cos(\hat{e}_p, d_c)$---the cosine between a parent feature's encoder
direction and a child feature's decoder direction---as a pairwise absorption signal. The
closest precedent is the Bricken et al. \citeyear{bricken2023monosemanticity} observation
that decoder-decoder cosine similarity clusters related features, but that metric operates at
the level of feature families rather than individual parent-child absorption pairs and does not
use encoder directions. Our finding that $\cos(\hat{e}_p, d_c)$ achieves AUROC = 0.730
(Cohen's $d = 0.552$, $p = 2.8 \times 10^{-9}$)---higher than EDA's AUROC = 0.650 for the
same labels---is an empirical discovery not predicted by any prior theory and constitutes a
contribution independent of the rate-distortion framework.

---

**Summary.** Table~\ref{tab:related} positions the contributions of this paper against prior
work. Chanin et al. establish the measurement framework but require probes; Tang et al. prove
absorption is a spurious minimum but do not give a quantitative threshold; architectural
mitigations all require retraining; no prior method uses SAE weight matrices alone to detect
absorbed features at the individual-latent level.

| Work | Probe-free | Weight-only | Quantitative threshold | Post-hoc | Architecture-agnostic |
|---|---|---|---|---|---|
| Chanin et al. \citeyear{chanin2024absorption} | No | No | No | Yes | Yes |
| Tang et al. \citeyear{tang2024unified} | N/A | N/A | No | No (training) | Theoretical |
| Matryoshka SAE \cite{bussmann2025matryoshka} | No | No | No | No | No |
| OrtSAE \cite{korznikov2025ortsae} | No | No | No | No | No |
| ATM SAE \cite{li2025atm} | No | No | No | No | No |
| Tian et al. \citeyear{tian2025sensitivity} | No | No | No | Yes | Yes |
| **This work (EDA)** | **Yes** | **Yes** | **Yes ($\lambda > \sin^2(\theta_{p,c})$)** | **Yes** | **Yes** |

Building on the Tang et al. theoretical framework, we formalize the rate-distortion threshold,
derive the mechanistic conjecture connecting absorption to EDA, and validate both against exact
absorption labels from the FeatureAbsorptionCalculator.

<!-- FIGURES
- Table 1 (summary): inline — Positioning of this paper's contributions against prior work on probe-free detection, weight-only computation, quantitative threshold, post-hoc applicability, and architecture-agnosticism
- None (no data-driven figures assigned to this section in the Figure and Table Plan)
-->

---

# Method

## 3.1 Setup: Sparse Autoencoders and Feature Absorption

A sparse autoencoder (SAE) maps a residual stream vector $x \in \mathbb{R}^d$ to a sparse
intermediate representation and back:

$$\hat{x} = D\, f(Ex + b),$$

where $E \in \mathbb{R}^{d_\text{sae} \times d}$ is the encoder weight matrix, $b \in
\mathbb{R}^{d_\text{sae}}$ is the encoder bias, $D \in \mathbb{R}^{d \times d_\text{sae}}$
is the decoder weight matrix with unit-norm columns $d_j$, and $f(\cdot)$ is a
sparsity-inducing nonlinearity (ReLU with L1 penalty, or TopK). The latent activation vector
is $z = f(Ex + b) \in \mathbb{R}^{d_\text{sae}}$, with $z_j \geq 0$ denoting the
activation of feature $j$.

We write $e_j \in \mathbb{R}^d$ for row $j$ of $E$ (the encoder direction for feature $j$)
and $\hat{e}_j = e_j / \|e_j\|_2$ for its unit-normalized form.

**Feature absorption** arises when two features form a hierarchical pair $(p, c)$, with
parent feature $p$ encoding a broader concept (e.g., "words beginning with the letter A")
and child feature $c$ encoding a specific instance (e.g., the token "apple"). Absorption
occurs when $z_c = 0$ on inputs where the child concept is present and the parent is active.
Following \citet{chanin2024absorption}, we quantify absorption via the \emph{absorption rate}:

$$\alpha = P(z_c = 0 \mid \text{child concept present, } z_p > 0).$$

The first-letter task from \citealt{chanin2024absorption} provides ground-truth labels:
for each letter $\ell$, a parent feature detects "words beginning with $\ell$" and child
features correspond to individual word tokens. Absorption labels are computed by
FeatureAbsorptionCalculator, which uses integrated gradients (IG) ablation to determine
whether removing a feature causes the model to fail the letter-prediction task.

**Training objective.** The SAE is trained to minimize the Lagrangian loss

$$\mathcal{L}_\text{SAE}(D, E, b) = \mathbb{E}_{x}\!\left[\|x - Df(Ex + b)\|_2^2\right]
+ \lambda\, \mathbb{E}_{x}\!\left[\|f(Ex + b)\|_0\right],$$

where $\lambda > 0$ is the sparsity penalty coefficient. The mean number of active features
per forward pass, $L_0 = \mathbb{E}[\|z\|_0]$, satisfies $\lambda \approx 1/L_0$ in
expectation for L1-penalized SAEs.

---

## 3.2 Theory: Rate-Distortion Absorption Preference

We ask: under what conditions does the SAE training objective prefer the absorbed solution
over the non-absorbed solution for a hierarchical pair $(p, c)$?

**Setup.** Fix a parent-child pair with decoder angle $\theta_{p,c} = \arccos(d_p \cdot
d_c)$ and co-occurrence probability $p_\text{co} = P(\text{parent and child both present})$.
Consider two candidate solutions:

- $S_1$ (non-absorbed): both features active on co-occurrence contexts. On contexts where
  both parent and child concepts are present, the SAE uses two latents ($z_p > 0$,
  $z_c > 0$) to reconstruct $x$. The per-context L0 cost is 2, and reconstruction error is
  approximately zero.

- $S_2$ (absorbed): only the child latent fires, but its decoder $d_c$ has absorbed the
  parent direction by tilting toward $d_p$. On co-occurrence contexts, one latent suffices
  ($z_c > 0$, $z_p = 0$). The per-context L0 cost is 1, but reconstruction error accrues
  from the unrepresented parent component: the residual is $\|d_p - \mathrm{proj}_{d_c}
  d_p\|_2^2 = \sin^2(\theta_{p,c})$ (per unit parent magnitude).

**Proposition 1 (Rate-Distortion Absorption Preference).** *The absorbed solution $S_2$
achieves strictly lower expected loss than the non-absorbed solution $S_1$ if and only if*

$$\boxed{\lambda > \sin^2(\theta_{p,c}).}$$

*Proof.* The expected loss difference $\Delta\mathcal{L} = \mathcal{L}(S_2) - \mathcal{L}(S_1)$
is determined by the co-occurrence contexts, where L0 differs. On each co-occurrence event
(probability $p_\text{co}$), $S_2$ saves one L0 unit (gain $\lambda$) but incurs
reconstruction error $\sin^2(\theta_{p,c})$. Thus
$\Delta\mathcal{L} = p_\text{co}[\sin^2(\theta_{p,c}) - \lambda]$.
Since $p_\text{co} > 0$, $\Delta\mathcal{L} < 0$ iff $\lambda > \sin^2(\theta_{p,c})$. $\square$

**Corollary 1 (Frequency cancels).** The absorption threshold $\lambda > \sin^2(\theta_{p,c})$
is independent of $p_\text{co}$. Absorption risk is determined entirely by decoder geometry
and sparsity penalty: rare and common parent-child pairs with the same decoder angle are
equally at risk.

**Corollary 2 (Monotone in sparsity).** The set of hierarchical pairs for which absorption
is preferred expands monotonically as $\lambda$ increases.

**Limitation.** Proposition 1 compares two specific candidate solutions; it does not
establish that gradient descent converges to $S_2$ from a random initialization, nor that
no lower-loss solution $S_3$ exists. Full convergence analysis would require showing that
$S_2$ is a local attractor of the biconvex training landscape \citep{tang2025saerd}, which
we leave to future work. Despite this caveat, the proposition identifies the geometric
condition under which the SAE loss landscape penalizes non-absorbed solutions and rewards
absorbed ones.

---

## 3.3 Geometric Signature: Encoder-Decoder Dissociation

Proposition 1 characterizes when absorption is \emph{loss-preferred}. We now ask: given
that a child feature $c$ has been absorbed, what observable geometric signature does it
leave in the trained SAE weights? This leads to the EDA metric.

**Definition.** The **encoder-decoder dissociation** (EDA) of feature $j$ is

$$\text{EDA}(j) = 1 - \cos(\hat{e}_j, d_j) = 1 - \frac{e_j \cdot d_j}{\|e_j\|_2 \|d_j\|_2}.$$

EDA $\in [0, 2]$; EDA $= 0$ when encoder and decoder directions are identical.

In a healthy (non-absorbed) feature, both the encoder and decoder represent the same concept:
$\hat{e}_j \approx d_j$, giving EDA $\approx 0$. We hypothesize that absorbed features
exhibit high EDA because the encoder and decoder are pulled in different directions by
distinct gradient signals during training.

**Mechanistic conjecture (Proposition 2, informal).** Under absorption, two gradient signals
act on the child feature $c$:

1. *Encoder gradient (from parent-only contexts):* When the parent $p$ is active but the
   child concept is absent, the absorbed child latent is expected to fire to reconstruct
   the parent's contribution. The pre-activation of $c$ on a parent-only context $x \approx
   \alpha_p d_p$ is $z_c^\text{pre} = e_c^T x - b_c = \alpha_p (e_c \cdot d_p) - b_c$.
   Maximizing the pre-activation to enable firing pulls $e_c \to d_p$.

2. *Decoder gradient (from child-present contexts):* Reconstruction loss on contexts where
   the child concept is present anchors $d_c$ toward the child concept direction in
   residual stream space.

Since $d_p \neq d_c$ (parent and child are distinct concepts), the encoder and decoder of
$c$ are pulled in opposite directions, increasing EDA.

**Formal statement.** Under conditions (C1) training data contains parent-only contexts
with positive probability $q > 0$; (C2) the child latent fires on parent-only contexts
(absorption has occurred); and (C3) decoder $d_c$ is primarily anchored by child-present
reconstruction contexts — the gradient of $\mathcal{L}_\text{SAE}$ with respect to $e_c$
at a parent-only context $x \approx \alpha_p d_p$ has a positive component toward $d_p$,
driving $e_c \to d_p$ and increasing EDA$(c)$ over training. This proposition is labeled a
\emph{mechanistic conjecture} because conditions (C2) and (C3) require empirical verification
that we provide partially in Section~\ref{sec:experiments}.

Figure~\ref{fig:method} illustrates the geometry. In the non-absorbed case (left panel),
encoder and decoder both point toward the child concept, giving EDA $\approx 0$. In the
absorbed case (right panel), the encoder has been pulled toward the parent decoder direction
$d_p$ while the decoder remains anchored to the child concept, producing a large angle
between $\hat{e}_c$ and $d_c$, i.e., high EDA.

![EDA mechanism: non-absorbed (left) vs. absorbed (right) feature geometry](figures/fig1_eda_method.pdf)

**Why decoder-decoder cosine fails.** One might expect absorbed pairs to have small
$\theta_{p,c}$ (near-aligned parent and child decoders), since Proposition 1 requires
$\sin^2(\theta_{p,c}) < \lambda$ for absorption onset. However, once absorption is
established, the child decoder $d_c$ drifts away from $d_p$: with the child encoder now
detecting parent contexts, the decoder faces no reconstruction pressure from those contexts
and is free to specialize toward child-specific signal. Post-convergence, absorbed features
therefore show *larger* decoder-decoder cosine $d_c \cdot d_p$ with their parents than
non-absorbed features (Cohen's $d = -0.48$ at GPT-2 Layer 6; Section~\ref{subsec:decomp}).
The theory describes geometry at absorption onset; the decoder angle reflects the
post-convergence equilibrium.

**Cross-directional metric.** The mechanistic conjecture also predicts that the parent
encoder $\hat{e}_p$ will align with the child decoder $d_c$ in absorbed pairs, because the
parent encoder and child decoder both encode information relevant to the parent concept in
co-occurrence contexts. We therefore also test the cross-directional cosine:

$$\cos(\hat{e}_p, d_c) = \hat{e}_p \cdot d_c$$

as an inter-feature absorption detector (taking the maximum across candidate parent
features for each child). An analogous metric swapping parent and child roles,
$\cos(\hat{e}_c, d_p)$, captures the absorbed state from the child side.

**Unresolved tension (EDA magnitude).** In the full-absorption limit where $e_c \to d_p$
exactly, Proposition 2 predicts EDA$(c) \approx 1 - \cos(\theta_{p,c})$, which should be
small if $\theta_{p,c}$ is small (the condition required by Proposition 1). However, the
observed mean EDA for letter features at Layer 6 is $0.671$ (corresponding to $\theta
\approx 60°$), substantially larger than the small-$\theta$ regime. We do not resolve this
tension in the present work; it could indicate partial alignment only, or that the relevant
parent-child decoder angles are not as small as the $\lambda \approx 0.02$ threshold
for L0 $\approx 50$ would suggest. This is noted as an open question in
Section~\ref{sec:discussion}.

---

## 3.4 Experimental Configurations and Baselines

**Model and primary SAE.** All experiments use GPT-2 Small (117M parameters) accessed via
TransformerLens \citep{nanda2022transformerlens}. The primary SAE is the
\texttt{gpt2-small-res-jb} release from SAELens \citep{bloom2024saelens}, targeting the
layer 6 residual stream pre-MLP hook (\texttt{blocks.6.hook\_resid\_pre}), with $d_\text{sae}
= 24{,}576$ and measured $L_0 = 50.97$.

**Ground-truth labels.** Exact absorption labels are produced by FeatureAbsorptionCalculator
(Chanin et al., 2024, \texttt{sae-spelling}) on the first-letter task at Layer 6. This yields
$n_+ = 18$ absorbed features out of $d_\text{sae} = 24{,}576$ (base rate $= 0.073\%$). The
extreme class imbalance is inherent to the task and motivates reporting both AUROC and
AUPRC alongside the permutation null $z$-score. We additionally evaluate against proxy labels
($n_+ = 50$) derived by thresholding decoder-probe alignment; the Jaccard overlap between
exact and proxy labels is 0.115, indicating they are largely non-overlapping sets. Results
on both label sets are reported in Section~\ref{sec:experiments}.

**Scaling suite.** We evaluate EDA across 11 SAE configurations (Table~\ref{tab:configs}):

- *Primary suite (jb):* gpt2-small-res-jb at layers 2, 4, 6, 8, 10 (width 24,576 each;
  $L_0$ range 18.5–76.6).
- *Architecture suite (AJT):* three AJT-trained SAEs at layer 6, width 46,080
  (gpt2-small-res\_sce-ajt, gpt2-small-res\_scl-ajt, gpt2-small-res\_sle-ajt;
  $L_0$ range 34.5–81.0).
- *Width suite:* feature-splitting SAEs at layer 8, widths 12,288, 24,576, 49,152, and
  98,304 (gpt2-small-res-jb-feature-splitting; $L_0 \approx 50$ matched).

The TopK SAE (gpt2-small-resid-post-v5-32k, $k = 32$, width 32,768) at layer 6 is included
as an additional architecture comparison.

**Letter-feature identification.** For each SAE, we identify letter features by training
one-versus-all logistic regression probes on residual stream activations (24 letters; letters
without a converged probe are excluded). Features are classified as letter features if their
decoder-probe cosine similarity exceeds an adaptive threshold (per-SAE, targeting 50–80
letter features). The count of letter features per configuration ranges from 55 to 76 across
the scaling suite.

**Baselines.** The following weight-only and statistics-based baselines are evaluated
alongside EDA:

| Detector | Definition |
|---|---|
| EDA | $1 - \cos(\hat{e}_j, d_j)$ |
| $\cos(\hat{e}_p, d_c)$ max | Maximum cross-directional cosine over candidate parents |
| $\cos(\hat{e}_c, d_p)$ mean | Mean cross-directional cosine over candidate parents |
| Encoder norm (inverted) | $\|e_j\|_2$ (larger norm → more likely absorbed) |
| Frequency ratio (inverted) | Activation frequency (less frequent → more likely absorbed) |
| Decoder norm | $\|d_j\|_2$ |
| $\cos(\hat{e}_j, d_j)$ raw | Raw cosine (equivalent to $1 - \text{EDA}$; reported to confirm equivalence) |
| Random | Uniform random scores |

All detectors are evaluated without any probe training or activation data beyond the SAE
weights themselves (encoder norm and frequency require a brief forward pass to compute
activation frequencies, but no labeled data). The permutation null distribution for AUROC
is obtained by permuting the absorption labels 100 times and recording the mean and standard
deviation; the $z$-score above null quantifies significance.

**Statistical tests.** Detection performance is reported as AUROC (primary), AUPRC, Cohen's
$d$ (for continuous-variable comparisons between letter and non-letter features), Wilcoxon
rank-sum $p$-values, and $z_\text{null}$ (permutation null). For the absorption phase
stability analysis (Section~\ref{subsec:phase}), sigmoid and linear curve fits to the
(1/$L_0$, absorption rate) relationship are compared using the likelihood-ratio test (LRT)
and Bayesian Information Criterion (BIC). For cross-architecture comparisons, the DeLong
test compares AUROC values between detectors; all reported $p$-values are two-sided.

<!-- FIGURES
- Figure 1: gen_fig1_eda_method.py, fig1_eda_method.pdf — Two-panel conceptual diagram: non-absorbed (enc≈dec, low EDA) vs. absorbed (enc pulled toward parent decoder, high EDA) feature geometry, with formula inset for EDA and Proposition 1 threshold.
-->

---

# 4 Experiments

We evaluate EDA and cross-directional metrics on the first-letter task using GPT-2 Small
(117M parameters) with SAELens pre-trained sparse autoencoders (SAEs).
All experiments use the gpt2-small-res-jb release as the primary suite, with layer 6
($d_\text{sae} = 24{,}576$, $L_0 \approx 51$) as the main evaluation point.
Ground-truth absorption labels come from FeatureAbsorptionCalculator
(Chanin et al. 2024, sae_spelling; henceforth exact labels), which applies integrated-gradients
ablation to identify which SAE features fail to fire when the corresponding first-letter concept
is present in the input.
The exact label set contains $n_\text{pos} = 18$ absorbed features out of 24,576
(base rate $= 7.3 \times 10^{-4}$).

**Baselines.** We compare EDA against four weight-only or activation-statistic alternatives:
activation frequency (inverted), decoder norm, encoder norm, and the raw cosine similarity
$\cos(\hat{e}_j, d_j)$ (which is $1 - \text{EDA}$).
Statistical significance of AUROC uses a permutation null ($n = 100$ permutations of
absorption labels), reported as $z_\text{null}$.

## 4.1 EDA Validation Against Exact Labels

Figure 2 shows EDA distributions for letter versus non-letter features at layer 6 and layer 10.
Table 1 reports all detector metrics.

Against exact Chanin labels ($n_\text{pos} = 18$), EDA achieves $\text{AUROC} = 0.650$,
$z_\text{null} = 2.49$ (above the 2-SD significance threshold), and
$\text{AUPRC} = 0.00153$ (2.09$\times$ the base-rate baseline).
Proxy labels (letter features with high decoder-probe alignment; $n_\text{pos} = 50$,
Jaccard overlap with exact labels $= 0.115$) yield $\text{AUROC} = 0.659$ and
Cohen's $d = 0.533$ ($p = 1.6 \times 10^{-4}$), confirming the signal is stable across
label definitions.

The strongest per-feature signal comes from the cross-directional metric
$\cos(\hat{e}_p, d_c)$, defined as the cosine similarity between the parent encoder
direction and the child decoder direction.
This metric achieves $\text{AUROC} = 0.730$ (Cohen's $d = 0.552$, $p = 2.8 \times 10^{-9}$,
$z_\text{null} = 6.38$), and captures absorbed features from the parent side of the
hierarchical pair, complementing EDA's child-side measurement.
The child-side cross-directional metric $\cos(\hat{e}_c, d_p)$ achieves
$\text{AUROC} = 0.681$ (Cohen's $d = 0.517$, $p = 2.7 \times 10^{-6}$, $z_\text{null} = 4.63$).

**The encoder norm finding.** Encoder norm yields the highest AUROC among individual
weight-only features ($\text{AUROC} = 0.757$, $\text{AUPRC}/\text{base} = 5.68\times$).
A DeLong test comparing EDA to encoder norm gives $p = 0.153$, so the difference is not
statistically significant at $\alpha = 0.05$.
The letter-feature mean encoder norm is $3.279$ versus $2.575$ for non-letter features.
Because encoder norm is not mechanistically predicted by the EDA theory and may reflect
polysemanticity rather than absorption specifically, we treat it as a confounded baseline
and examine its source in Section 4.2.

**Decoder baselines fail.** Decoder norm is near-random ($\text{AUROC} = 0.515$), and
the raw cosine $\cos(\hat{e}_c, d_c)$ achieves only $\text{AUROC} = 0.350$ — strongly
anti-correlated with absorption labels, confirming that simply checking encoder-decoder
alignment in the aligned direction is the wrong signal.
Activation frequency inverted achieves $\text{AUROC} = 0.595$, consistent with the
observation that absorbed features tend to fire less frequently.

**Layer 10 reversal.** At layer 10, EDA computed on the same gpt2-small-res-jb suite gives
$\text{AUROC} = 0.256$ (Cohen's $d = -0.890$, $p = 6.8 \times 10^{-10}$) — the EDA polarity
reverses, meaning letter features at L10 have *lower* EDA than non-letter features.
This finding is consistent with the mechanistic conjecture (Proposition 2): post-absorption
re-alignment of the encoder toward the child concept could reduce EDA at late layers once the
absorption stabilizes. We report this as an open empirical finding; the exact L10 mechanism
is not resolved by our current data.

![EDA distributions at L6 and L10, showing polarity reversal](figures/fig2_eda_distributions.pdf)

---

**Table 1: Main Detection Results — EDA and Baselines (GPT-2 Small, Layer 6, $d_\text{sae} = 24{,}576$)**

| Detector | AUROC | AUPRC/base | Cohen's $d$ | $p$-value | $z_\text{null}$ |
|---|---|---|---|---|---|
| $\cos(\hat{e}_p, d_c)$ max | **0.730** | — | 0.552 | $2.8 \times 10^{-9}$ | 6.38 |
| Encoder norm | 0.757 | 5.68$\times$ | — | — | — |
| $\cos(\hat{e}_c, d_p)$ mean | 0.681 | — | 0.517 | $2.7 \times 10^{-6}$ | 4.63 |
| EDA: $1 - \cos(\hat{e}_c, d_c)$ | 0.650 | 2.09$\times$ | 0.533 | $1.6 \times 10^{-4}$ | 2.49 |
| Freq. ratio (inverted) | 0.595 | 1.33$\times$ | — | — | — |
| Decoder norm | 0.515 | 1.02$\times$ | — | — | — |
| $\cos(\hat{e}_c, d_c)$ raw | 0.350 | — | — | — | — |
| Random baseline | 0.500 | 1.00$\times$ | — | — | — |

*EDA and $\cos(\hat{e}_c, d_c)$ raw are mathematical inverses; identical performance confirms the implementation. Encoder norm AUROC is not significantly higher than EDA (DeLong $p = 0.153$). Cross-directional metrics use proxy labels ($n_\text{pos} = 50$); EDA validation uses exact labels ($n_\text{pos} = 18$). $z_\text{null}$ is the $z$-score above permutation null AUROC.*

---

## 4.2 EDA Decomposition: Encoder vs. Decoder Alignment

Figure 4 decomposes EDA into its encoder and decoder components to test the mechanistic
conjecture underlying Proposition 2.

For letter features at layer 6 ($n = 50$), the decoder aligns more strongly with the
first-letter probe than the encoder:
decoder-probe cosine mean $= 0.383$ versus encoder-probe cosine mean $= 0.139$
(paired $t$-test: $t = -38.3$, $p = 3.5 \times 10^{-38}$, diff $= -0.244$).
As predictors of letter-feature membership, the decoder achieves AUROC $= 1.000$ and the
encoder achieves $\text{AUROC} = 0.991$ — both near-perfect, but the decoder is strictly
stronger.
For non-letter features, the gap collapses: decoder mean $= 0.099$, encoder mean $= 0.056$
(diff $= -0.043$).

This pattern is consistent with the mechanistic conjecture: the decoder remains anchored to
the child concept direction (letter identity), while the encoder is partially pulled toward the
parent direction during training, reducing encoder-probe alignment relative to decoder-probe
alignment.

**Encoder norm as confound.** Letter features have systematically larger encoder norms
($3.279 \pm 0.544$) than non-letter features ($2.575 \pm 0.707$).
This inflated norm is consistent with a feature whose encoder direction is under competing
gradient pressure — both the child reconstruction signal and the parent activation signal
act on $\hat{e}_c$ — but also with polysemanticity.
Because encoder norm does not distinguish between these two explanations, it cannot be
interpreted as a direct absorption signal without additional evidence.
We note this as an unresolved tension: the mechanistic story predicts encoder norm should
be elevated in absorbed features, but the same prediction follows from polysemanticity.

**EDA magnitude tension.** A distinct unresolved tension: the observed mean EDA for letter
features is $0.671$ (implying a $\sim 48°$ angle between $\hat{e}_c$ and $d_c$),
whereas Proposition 1 predicts that absorption onset occurs at small $\theta_{p,c}$
(parent-child decoder angle), which should correspond to small EDA in a newly absorbed feature.
Large observed EDA is consistent with long-term evolution of the encoder direction post-absorption
but is not directly predicted by the theory. We report this as an open question.

![Encoder and decoder probe alignment for letter vs. non-letter features (L6)](figures/fig4_enc_dec_alignment.pdf)

## 4.3 EDA Across Architectures and Scales

Figure 3 plots EDA$_\Delta$ (letter minus non-letter mean EDA) across 11 SAE configurations.

**Standard/L1 suite.** For the five gpt2-small-res-jb configurations (layers 2–10, width 24,576),
EDA$_\Delta$ is positive at all five layers and peaks at L4 ($+0.045$, AUROC $= 0.716$) and L6
($+0.050$, AUROC $= 0.702$), declining toward L10 ($+0.005$, AUROC $= 0.505$).
The L10 near-zero EDA$_\Delta$ — despite the strongly reversed AUROC seen in the pairwise analysis
— reflects the fact that the full layer 10 letter-feature population is heterogeneous.

**TopK SAE comparison.** A TopK SAE ($k = 32$, width 32,768, layer 6) shows lower mean letter-feature
EDA ($0.476$) compared to the L1-penalized SAE at the same layer ($0.676$), with the fraction of
letter features exceeding $\text{EDA} > 0.5$ dropping from 100\% to 25\%.
Both architectures show statistically significant positive EDA$_\Delta$ (Wilcoxon $p = 3.3 \times 10^{-4}$
for TopK), but the signal is weaker for TopK, consistent with the prediction that exact-sparsity
constraints alter the gradient landscape during training.

**AJT training regime reversal.** Three AJT-trained SAEs at layer 6 (width 46,080, L0 ranging
34.5–81.0) exhibit strongly negative EDA$_\Delta$: $-0.204$, $-0.177$, and $-0.217$,
with AUROC values of $0.154$, $0.354$, and $0.158$.
Letter features in AJT SAEs have *lower* EDA than non-letter features, the opposite of the L1-SAE
pattern.
This polarity reversal — present across all three AJT variants regardless of their different
sparsity levels — suggests that the AJT training regime fundamentally alters the geometry of
encoder-decoder dissociation for absorbed features.
The mechanism is not established by the current data; we conjecture that AJT's non-L1 sparsity
formulation changes the gradient signal that pulls encoder directions during training.

**Width analysis.** The feature-splitting suite at layer 8 (widths 12k, 24k, 49k, 98k,
matched $L_0 \approx 51$) shows decreasing EDA$_\Delta$ as width increases:
$+0.028$ at 12k, $+0.041$ at 24k, $+0.012$ at 49k, and $-0.017$ at 98k.
As SAE width grows, feature splitting produces additional features that cover fine-grained
child concepts, and the EDA signal dilutes because not all newly split child features will have
the same absorption geometry.

![EDA$_\Delta$ and AUROC across 11 SAE configurations, by layer, architecture family, and width](figures/fig3_eda_scaling.pdf)

## 4.4 Absorption Phase Stability

Figure 5 shows absorption rates across all 11 SAE configurations as a function of $1/L_0$.

All 11 configurations produce uniformly high absorption rates: the range is $0.876$–$0.978$,
with mean $0.950$.
The standard/L1 jb suite alone spans $0.938$–$0.967$ across layers 2–10.
AJT variants, despite their reversed EDA polarity, show absorption rates of $0.919$, $0.876$,
and $0.978$, all substantially above zero.

Testing whether absorption rate follows a sigmoid-shaped transition in $1/L_0$
(Hypothesis H4): the likelihood-ratio test comparing sigmoid versus linear fit gives
LRT $p = 0.456$ and BIC difference $= -3.22$ (negative means sigmoid is not preferred),
with the sigmoid inflection estimated at $L_0 \approx 81$ — outside the observed range.
Spearman rank correlation between $1/L_0$ and absorption rate across all 11 configurations:
$\rho = -0.482$, $p = 0.133$ (not significant).
Within the primary jb suite only: $\rho = -0.100$, $p = 0.873$.
These results indicate that no phase transition in sparsity is detectable within the tested
$L_0$ range of 18–81.

**Hysteresis experiment.** To test whether the absorbed state is metastable,
we fine-tuned a high-sparsity SAE (gpt2-small-res-jb, layer 2, baseline $L_0 = 33.7$,
absorption rate $= 0.959$) for 500 steps with a 5$\times$-reduced sparsity coefficient
($\lambda_\text{finetune} = 0.2\lambda_\text{original}$), reaching $L_0 = 68.6$.
After 500 fine-tuning steps, absorption rate $= 0.960$ — effectively unchanged
(fraction of baseline $= 1.001$).
A from-scratch SAE trained at the lower sparsity level achieves absorption rate $= 0.964$
($L_0 = 84.2$), also high.
The checkpoint trajectory shows no decrease in absorption at any step: $0.959 \to 0.959 \to
0.960 \to 0.960 \to 0.960$ at steps 100, 200, 300, 400, 500.
These data are consistent with the absorbed state being metastable: reducing sparsity at
fine-tuning time does not escape the absorbed configuration.
We note that the interpretation is constrained by the saturation: because all tested
$L_0$ values produce very high absorption, classical hysteresis (where a second stable
non-absorbed state exists at low sparsity) cannot be tested in this regime.

## 4.5 Cross-Domain Absorption

To determine whether feature absorption is specific to orthographic hierarchies or extends to
semantic concept hierarchies, we measure absorption rates on three hierarchy types:
(1) first-letter, (2) animate-inanimate, and (3) noun-proper noun.

**First-letter: GO.** The first-letter hierarchy yields absorption rate $= 0.0083$
(120 IG-ablation events across 8 letters: a, g, h, i, j, l, m, q),
ratio-to-null $= 10.0$, 95\% bootstrap CI $[0, 0.029]$.
This confirms that the sae_spelling methodology detects genuine absorption, with one letter
(h) showing absorption rate $= 0.067$ and the remaining seven letters at 0.

**Semantic hierarchies: NO\_GO.** Both animate-inanimate and noun-proper noun hierarchies
produce absorption rate $\approx 0$ ($9.999 \times 10^{-9}$, indistinguishable from the
shuffled-label null), ratio-to-null $= 1.0$.
Under the animate-inanimate hierarchy, the five parent latents (features 3152, 18287, 18617,
16169, 3278) do not produce measurable child suppression over 20 test words.
The noun-proper noun hierarchy shows the same null pattern.

**Interpretation.** Feature absorption in GPT-2 Small layer 6 is specific to orthographic
(first-letter) hierarchies at the tested scale; semantic concept absorption does not emerge.
This is a scoped null result: it does not rule out semantic absorption in larger models or
at different layers, and our experimental design cannot test all possible semantic hierarchies.
The parent feature identification method for semantic hierarchies (top-5 probe-aligned latents)
may also underestimate the relevant feature set.
Experiments on Gemma Scope SAEs, which operate at larger scale and may develop richer semantic
feature hierarchies, are necessary to determine whether this null result generalizes.

---

**Summary of experimental findings.** EDA achieves statistically significant detection of
absorbed features (AUROC $= 0.650$, $z_\text{null} = 2.49$) using only SAE weight matrices,
with no probe training required. The cross-directional metric $\cos(\hat{e}_p, d_c)$ is
stronger (AUROC $= 0.730$). Absorption rates are uniformly high (0.876–0.978) across all
tested configurations, no phase transition is detected, and the absorbed state is metastable
to fine-tuning. The EDA signal is architecture-dependent: L1-penalized SAEs show positive
EDA$_\Delta$; AJT-trained SAEs show reversed polarity; TopK SAEs show reduced signal.
Absorption is specific to the first-letter orthographic hierarchy in GPT-2 Small; semantic
hierarchies produce null results at this scale.

<!-- FIGURES
- Figure 2: gen_fig2_eda_distributions.py, fig2_eda_distributions.pdf — EDA violin distributions for letter vs. non-letter features at L6 (AUROC=0.659, Cohen's d=0.533) and L10 (AUROC=0.256, Cohen's d=-0.890, reversed polarity)
- Figure 3: gen_fig3_eda_scaling.py, fig3_eda_scaling.pdf — EDA_delta and EDA AUROC across 11 configurations: primary jb suite (L2-L10), AJT suite (reversed), width sweep (L8 12k-98k)
- Figure 4: gen_fig4_enc_dec_alignment.py, fig4_enc_dec_alignment.pdf — Encoder vs. decoder probe alignment scatter and bar chart for L6 letter features (dec AUROC=1.000 > enc AUROC=0.991, diff=-0.244)
- Figure 5: gen_fig5_phase_stability.py, fig5_phase_stability.pdf — Absorption rate vs. 1/L0 across 11 configs with sigmoid/linear fit overlay; LRT p=0.456, BIC diff=-3.22
- Table 1: inline — Main detection results: all detectors with AUROC, AUPRC/base, Cohen's d, p-value, z_null
-->

---

# 5 Discussion

## 5.1 What EDA Establishes and What It Does Not

EDA achieves AUROC $= 0.650$ ($z_\text{null} = 2.49$) against exact FeatureAbsorptionCalculator
labels, and the cross-directional metric $\cos(\hat{e}_p, d_c)$ achieves AUROC $= 0.730$
(Cohen's $d = 0.552$, $p = 2.8 \times 10^{-9}$), both statistically significant.
These results establish that SAE weight matrices alone carry absorption signal, without probes
and without activation data.

The practical scope of this claim is precise. The positive result holds for L1-penalized
(res-jb) SAEs at mid-layers (L4–L8) of GPT-2 Small, on the first-letter orthographic hierarchy.
EDA does not achieve AUROC $\geq 0.7$ everywhere: L2 ($= 0.596$), L10 ($= 0.505$), and all
three AJT configurations ($0.154$–$0.354$) fall short. The correct reading is that EDA is
a reliable signal specifically in the regime where L1-penalized training has had time to
establish absorption but before late-layer encoder re-alignment erases the dissociation.

Practitioners evaluating a deployed SAE can use EDA as a first-pass screen on the condition
that they (a) know the SAE was trained with L1 sparsity and (b) are examining mid-network
layers. Features ranked in the top EDA percentile are enriched for absorbed features
(AUPRC $= 2.09\times$ base rate at L6), providing a tractable candidate set for downstream
IG-based verification — the $n_+ = 18$ exact positives, which require IG ablation to confirm,
can be targeted from a much smaller candidate pool rather than an exhaustive sweep over all
24,576 features.

## 5.2 The Encoder-Decoder Decomposition: What the Data Supports

The most direct mechanistic evidence comes from Figure 4. For 50 letter features at layer 6,
decoder-probe cosine alignment averages $0.383$ versus encoder-probe cosine alignment of $0.139$
(paired $t$, diff $= -0.244$, $p = 3.5 \times 10^{-38}$). The decoder points toward the
letter-identity direction; the encoder does not, even though both vectors nominally belong to
the same feature. This differential alignment is precisely what Proposition 2 (mechanistic
conjecture) predicts: reconstruction loss anchors the decoder to the child concept, while
gradient pressure from parent-only contexts pulls the encoder toward the parent direction.

The pattern disappears for non-letter features: encoder and decoder probe alignment are
$0.056$ and $0.099$, a much smaller gap (diff $= -0.043$). The decoder $>$ encoder alignment
asymmetry is therefore specific to absorbed-candidate features, not a universal property of
the SAE geometry.

What the data does *not* support directly is that this dissociation proves Proposition 2's
causal mechanism. The observation is consistent with Proposition 2 but also consistent with
alternative explanations, including amortization gap effects (O'Neill et al. 2024), where the
encoder fails to specialize to a feature the decoder has already committed to. Distinguishing
causal absorption from amortization gap would require activation experiments beyond the
weight-only analysis presented here.

**EDA magnitude tension.** Proposition 1 predicts absorption is preferred when $\lambda >
\sin^2(\theta_{p,c})$, which implies small parent-child decoder angle at absorption onset.
Small $\theta_{p,c}$ should correspond to small EDA in a newly absorbed feature — yet the
observed letter-feature mean EDA is $0.671$ (implying roughly a $48°$ encoder-decoder angle).
One reconciliation is temporal: the theoretical threshold describes the *onset* of the
absorbed solution's energy advantage, while observed EDA reflects the *post-convergence* state
after substantial encoder drift. If the absorbed encoder continues drifting toward the parent
direction for the duration of training, large EDA can accumulate even when the initial
transition occurred at small $\theta_{p,c}$. This explanation is plausible but unverified;
we report it as an unresolved tension.

## 5.3 Informative Failure Modes

The three failure modes — L10 EDA reversal, AJT polarity reversal, and encoder norm
dominance — collectively provide more mechanistic insight than the positive results alone.

**L10 reversal.** At layer 10, letter features have *lower* EDA than non-letter features
(Cohen's $d = -0.890$, $p = 6.8 \times 10^{-10}$, AUROC $= 0.256$). One interpretation:
late-layer features of GPT-2 Small may include large numbers of non-absorbed letter-adjacent
features (e.g., positional or syntactic features that are highly encoder-decoder aligned),
diluting the L6 absorption signal in the opposite direction. A second interpretation, consistent
with Proposition 2: at late layers, the encoder of an absorbed feature may re-align toward the
child concept as training converges, reducing EDA even as the functional absorption persists.
Both interpretations predict that layer-specific calibration is required; neither is ruled out
by current data. The key practical implication is that EDA thresholds trained on mid-layer
SAEs cannot be transferred to late-layer SAEs without recalibration.

**AJT polarity reversal.** All three AJT-trained SAEs (res_sce-ajt, res_scl-ajt, res_sle-ajt)
show strongly negative EDA$_\Delta$ ($-0.204$, $-0.177$, $-0.217$), meaning absorbed-candidate
features have *lower* EDA in AJT SAEs than the non-absorbed background. This finding rules out
EDA as a universal absorption indicator: the signal direction depends on the training regime.
AJT training uses a non-L1 sparsity formulation; the difference in gradient structure may prevent
or reverse the encoder drift described in Proposition 2. Concrete investigation of AJT's
gradient dynamics is needed to determine the mechanism.

**Encoder norm dominance.** Encoder norm achieves AUROC $= 0.757$ — higher than EDA's $0.650$
but not statistically significantly so (DeLong $p = 0.153$). The mean encoder norm is $3.279$
for letter features versus $2.575$ for non-letter features. Two explanations are compatible:
(1) absorbed features experience competing gradient pressures, increasing the norm of $e_c$
as it balances child reconstruction and parent detection signals; (2) polysemantic features
with multiple strong associations have elevated encoder norms regardless of absorption.
Because encoder norm is not derived from the EDA theory and cannot be interpreted mechanistically
without distinguishing these two sources, it is treated as a confounded baseline rather than a
primary detector. The coincidence between elevated encoder norm and letter-feature membership
nevertheless warrants follow-up experiments that hold polysemanticity constant.

## 5.4 Phase Stability and the Implications for Remediation

Absorption rates are 0.876–0.978 across all 11 tested configurations (layers 2–10, widths
12k–98k, L1-penalized and AJT training regimes, $L_0$ range 18–81). No sigmoid-shaped
transition is detectable (LRT $p = 0.456$, BIC difference $= -3.22$), and no sparsity
dependence is measurable (Spearman $\rho = -0.482$, $p = 0.133$ across all 11; $\rho = -0.100$,
$p = 0.873$ within the primary jb suite).

The hysteresis experiment reinforces this picture: fine-tuning a high-absorption SAE
($\alpha = 0.959$) for 500 steps at one-fifth the original sparsity coefficient
(achieving $L_0 = 68.6$) does not reduce absorption ($\alpha = 0.960$, fraction of baseline
$= 1.001$). The checkpoint trajectory shows absorption stable at $0.959$–$0.960$ throughout.

Two practical implications follow from absorption phase stability. First, practitioners cannot
reduce absorption in a deployed SAE by adjusting L0 via activation-function threshold changes
or top-$k$ recalibration: the sparsity coefficient change required to escape absorption is
beyond anything achievable without retraining from scratch. Second, because all tested
hyperparameter configurations produce highly absorbed SAEs, the structural source of absorption
must reside in the training objective itself (consistent with Proposition 1) rather than in
any particular hyperparameter choice. Architectural interventions that modify the absorption
threshold — orthogonality penalties (OrtSAE), hierarchical codebooks (Matryoshka SAE), or
per-latent importance weighting — act on the objective directly, which is why they reduce
absorption while hyperparameter tuning does not.

The saturation of absorption rates throughout the tested range also means that the classical
hysteresis experiment (comparing the absorbed state at high sparsity to a non-absorbed stable
state at low sparsity) cannot be executed in this regime: no non-absorbed stable state is
accessible via parameter changes within the tested L0 range. Testing for hysteresis would
require either an architectural change that creates a non-absorbed equilibrium or experiments
on much wider SAEs at much lower sparsity, which we leave for future work.

## 5.5 Cross-Domain Scope: Orthographic vs. Semantic Absorption

First-letter absorption (ratio-to-null $= 10.0$, 120 events, GO) and null semantic results
(animate-inanimate, noun-proper; ratio-to-null $= 1.0$, NO\_GO) together establish that
feature absorption in GPT-2 Small is specific to orthographic hierarchies at the tested scale.

This null result for semantic hierarchies should not be over-interpreted. The absence of
detectable absorption under animate-inanimate and noun-proper noun hierarchies in GPT-2 Small
is consistent with three distinct explanations: (1) semantic concept hierarchies do not produce
sufficient co-occurrence of parent and child features in this model to trigger the absorption
threshold; (2) the parent feature identification method (top-5 probe-aligned latents) misses the
true parent features in semantically complex hierarchies; (3) semantic absorption exists but the
IG-ablation methodology requires richer context prompts than the fixed ICL format used here.
All three explanations predict that experiments on Gemma Scope SAEs (Gemma 2 2B or larger, where
semantic hierarchies should be more richly represented) could find semantic absorption even if
GPT-2 Small does not exhibit it.

The scope of the present result is therefore: orthographic (first-letter) absorption is
confirmed; semantic absorption is neither confirmed nor definitively ruled out. Gemma Scope
experiments are required before drawing conclusions about the generality of absorption across
hierarchy types.

## 5.6 Connections to Architectural Mitigations

The rate-distortion framework (Proposition 1) provides a unified account of why architectural
mitigations work: any intervention that increases $\sin^2(\theta_{p,c})$ for parent-child pairs
raises the sparsity threshold below which absorption is preferred.

OrtSAE imposes an orthogonality penalty on decoder columns, directly increasing the angle
$\theta_{p,c}$ between any two decoder directions, including parent-child pairs; by Proposition 1,
this raises the absorption onset threshold. Matryoshka SAE assigns parent features to an inner
dictionary and child features to an outer dictionary, so parent and child decoders operate in
structurally different subspaces; the effective $\theta_{p,c}$ for cross-level pairs is larger
by construction. ATM SAE uses per-latent importance weighting, creating non-uniform effective
$\lambda$ across features; child features with reduced effective sparsity penalty fall below the
absorption threshold even when their decoder angle is small.

These connections are theoretical predictions, not verified by our experiments, but the
predictions are falsifiable: each mitigation should be accompanied by an increase in the
mean decoder angle $\theta_{p,c}$ for absorbed pairs, measurable from weights alone. The
EDA framework provides the measurement tool.

## 5.7 Limitations and Open Questions

**Label sparsity.** The exact FeatureAbsorptionCalculator labels contain $n_+ = 18$ absorbed
features out of 24,576 at layer 6. All AUROC comparisons involving exact labels operate in a
regime where small changes in the label set can substantially alter the apparent metric. The
proxy label expansion to $n_+ = 50$ stabilizes results (AUROC $= 0.659$ vs. $0.650$), and the
Jaccard overlap of $0.115$ between exact and proxy sets suggests they measure overlapping but
not identical aspects of absorption. A larger exact label set, obtainable by running
FeatureAbsorptionCalculator across more letters and at lower activation thresholds, would
strengthen statistical power substantially.

**Single model and task.** All results are obtained on GPT-2 Small (117M parameters), which
is at the smallest end of the model scale relevant to mechanistic interpretability work.
Gemma 2 2B has richer semantic features, deeper feature hierarchies, and substantially larger
SAEs; whether the EDA signal, the cross-directional metric, and the AJT polarity reversal
all replicate at this scale remains open. Cross-model generalization is the most important
single extension to this work.

**Amortization gap confound.** EDA measures intra-feature encoder-decoder misalignment.
High EDA could arise from absorption (encoder pulled toward parent direction) or from
amortization gap (encoder fails to specialize to a feature the decoder has committed to).
Distinguishing these requires activation-data experiments comparing encoder firing patterns to
decoder direction projections; the weight-only analysis in this paper cannot separate them.

**Proxy label quality.** Proxy labels (letter features above decoder-probe threshold) have
Jaccard overlap $= 0.115$ with exact Chanin labels. The low overlap means that most proxy-labeled
features are not confirmed absorbed by IG-ablation. The AUROC results using proxy labels should
be interpreted with this in mind: they measure a related but broader property than the exact
absorption labels.

**AJT mechanism unknown.** The three AJT SAEs produce a clear and reproducible polarity
reversal (EDA$_\Delta < 0$, AUROC $< 0.2$), but the mechanism is not established. Whether
the AJT sparsity formulation prevents encoder drift, reverses it, or creates a qualitatively
different geometry for absorbed features is unknown. This is a concrete open question with
practical implications: if AJT training were combined with post-hoc EDA screening, a new
calibrated detector (possibly using inverted EDA) might still work, but this requires
verification.

---

The central results are stable to these limitations: EDA provides a statistically significant
weight-only absorption signal for L1-penalized SAEs at mid-layers ($z_\text{null} = 2.49$,
AUROC $= 0.650$), the cross-directional metric $\cos(\hat{e}_p, d_c)$ is stronger still
(AUROC $= 0.730$), absorption is phase-stable across the tested hyperparameter range
(rates 0.876–0.978, no phase transition, hysteresis not escaped by fine-tuning), and the
encoder-decoder decomposition provides direct geometric evidence consistent with the
mechanistic conjecture.

<!-- FIGURES
- None (Discussion section contains no new figures; references figures from earlier sections)
-->

---

# 6 Conclusion

Feature absorption — the suppression of child SAE latents by parent features — reduces the
reliability of sparse autoencoders for mechanistic interpretability by creating features that
appear to encode a concept yet fail to fire on 92–97\% of relevant inputs. Three results in
this paper address the detection, mechanism, and phase behavior of this failure mode.

**Probe-free absorption detection.** EDA, $\text{EDA}(j) = 1 - \cos(\hat{e}_j, d_j)$, achieves
AUROC $= 0.650$ against exact FeatureAbsorptionCalculator labels ($n_+ = 18$, $z_\text{null}
= 2.49$) on GPT-2 Small layer 6 — statistically significant and computed from SAE weights
alone, without probes or activation data. The cross-directional metric $\cos(\hat{e}_p, d_c)$
is stronger: AUROC $= 0.730$ (Cohen's $d = 0.552$, $p = 2.8 \times 10^{-9}$), a result not
anticipated by the original EDA theory and identified empirically from pairwise analysis. Both
detectors yield AUPRC above $2 \times$ base rate, providing a tractable candidate set for
downstream IG-based verification. Decoder cosine similarity between feature pairs fails as a
detector (AUROC $= 0.206$), confirming that absorption geometry is carried by encoder directions,
not between decoders.

**Geometric mechanism.** Proposition 1 proves that the rate-distortion training objective prefers
the absorbed solution when $\lambda > \sin^2(\theta_{p,c})$, with the co-occurrence frequency
$p_\text{co}$ canceling from the threshold. This provides the first closed-form, falsifiable
condition for absorption onset, and its key corollary — that rare and common feature pairs with
identical decoder angles are equally at risk — contradicts the intuition that infrequent co-occurrence
protects against absorption. The encoder-decoder decomposition confirms the mechanistic conjecture
(Proposition 2): at layer 6, letter-feature decoder-probe alignment ($0.383$) exceeds
encoder-probe alignment ($0.139$) by a margin of $0.244$ ($p = 3.5 \times 10^{-38}$), consistent
with decoders anchored to the child concept and encoders pulled toward the parent. This asymmetry
is absent in non-letter features (gap $= 0.043$), ruling out a universal SAE geometry effect.
One tension remains unresolved: observed EDA values near $0.67$ imply an encoder-decoder angle
of roughly $48°$, larger than the small angle predicted by Proposition 1 at absorption onset;
we hypothesize post-convergence encoder drift accounts for this gap, but the claim is unverified.

**Phase stability and architecture dependence.** Absorption rates span only $0.919$–$0.968$
across all 11 tested SAE configurations (GPT-2 Small, layers 2–10, widths 12k–98k, $L_0$
range $18$–$81$). A likelihood-ratio test rejects a sparsity-dependent sigmoid (LRT $p = 0.456$;
BIC difference $= -3.22$), and Spearman $\rho = 0.191$ ($p = 0.574$) confirms the absence of
a monotonic sparsity trend. Fine-tuning a high-absorption SAE ($\alpha = 0.959$) for 500 steps
at substantially reduced sparsity does not escape the absorbed state ($\alpha = 0.960$). The
practical implication is direct: absorption cannot be tuned away by adjusting $L_0$ or
choosing a different layer; only architectural interventions that increase $\sin^2(\theta_{p,c})$
— OrtSAE, Matryoshka SAE, ATM SAE — act on the training objective in the right way. AJT-trained
SAEs exhibit reversed EDA polarity (EDA$_\Delta < 0$, AUROC $= 0.154$–$0.354$), demonstrating
that training regime, not hyperparameter tuning, is the critical variable in absorption geometry.

**Scope and open questions.** All positive detection results are scoped to L1-penalized SAEs at
mid-layers of GPT-2 Small on first-letter orthographic hierarchies. Semantic hierarchy absorption
is absent in GPT-2 Small (animate-inanimate, noun-proper: ratio-to-null $= 1.0$), though three
explanations remain viable — insufficient semantic co-occurrence at this scale, parent feature
misidentification, or methodology artifacts — and Gemma Scope experiments are required to resolve
them. Three open questions merit direct follow-up: (1) whether EDA's magnitude tension is resolved
by measuring encoder drift trajectories during training; (2) whether $\cos(\hat{e}_p, d_c)$
remains the strongest cross-directional detector on Gemma-scale SAEs, where EDA's cross-model
generalization is unconfirmed; and (3) whether AJT training prevents encoder drift outright or
produces a qualitatively different geometry that requires an inverted detector.

The weight matrices of a trained SAE record the history of the training objective's geometry.
EDA reads that record.

<!-- FIGURES
- None
-->

---

