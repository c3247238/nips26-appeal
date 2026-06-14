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
