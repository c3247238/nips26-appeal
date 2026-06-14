# 3. Encoder-Decoder Alignment (EDA) as an Absorption Indicator

The Chanin et al. metric requires pre-specified probe directions and activation data---it detects absorption only where the analyst already suspects it. We seek a complementary screening signal computable from SAE weights alone. This section derives Encoder-Decoder Alignment (EDA), establishes a formal lower bound connecting EDA to absorption degree, extends the scalar metric to a directional decomposition (D-EDA) that distinguishes absorption from polysemanticity, and validates both on synthetic data with known ground truth.

## 3.1 Derivation

A Sparse Autoencoder trained with the sparse dictionary learning (SDL) loss, which has a biconvex structure, learns encoder weights $W_e \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ and unit-normed decoder columns $d_j \in \mathbb{R}^{d_{\text{model}}}$ minimizing

$$\mathcal{L}(\theta) = \mathbb{E}_x \left[\|x - \hat{x}\|^2 + \lambda \|z\|_1\right], \quad \text{s.t.} \quad \|d_j\| = 1 \; \forall j. \tag{1}$$

At a global minimum of this biconvex loss, latent $j$'s encoder direction $w_{e,j}$ (row $j$ of $W_e$) aligns with its decoder direction $d_j$: the encoder projects inputs onto the same direction the decoder uses for reconstruction. Absorption disrupts this alignment. When a child latent $c$ absorbs parent latent $j$, the sparsity penalty $\lambda \|z\|_1$ drives the parent activation $z_j$ toward zero on inputs where $z_c > 0$. The encoder direction $w_{e,j}$ rotates away from $d_j$, drifting toward directions that no longer serve latent $j$'s original detection role.

This geometric observation motivates a scalar metric:

$$\text{EDA}(j) = 1 - \cos(w_{e,j}, d_j) = 1 - \frac{w_{e,j} \cdot d_j}{\|w_{e,j}\| \cdot \|d_j\|}. \tag{2}$$

EDA ranges from 0 (perfect alignment) to 2 (perfect anti-alignment). It requires only the SAE weight matrices $W_e$ and $W_d$---no activation data, no probe directions, no model forward passes. Computation takes under one second for a 65k-width SAE.

## 3.2 Formal Lower Bound (Theorem 1)

**Definition ($\delta$-absorption).** Latent $j$ exhibits $\delta$-absorption of child $c$ if the sparsity gradient from child-active inputs displaces $w_{e,j}$ from $d_j$ with magnitude $\delta \geq 0$. Larger $\delta$ corresponds to stronger suppression.

**Theorem 1 (EDA Lower Bound).** *For a Sparse Autoencoder at a partial minimum of the biconvex SDL loss (Equation 1), if latent $j$ exhibits $\delta$-absorption of child $c$, then:*

$$\text{EDA}(j) \geq \frac{\delta^2 \sin^2(\theta_{jc})}{2 + \delta^2}, \tag{3}$$

*where $\theta_{jc}$ is the angle between decoder directions $d_j$ and $d_c$, and $\delta \geq 0$ is the absorption degree as defined above.*

The bound is monotonically increasing in $\delta$: stronger absorption produces larger EDA. When $\delta = 0$ (no absorption) and the SAE is at a global minimum, $\text{EDA}(j) = 0$. The $\sin^2(\theta_{jc})$ factor means the bound is tightest when the parent and child decoder directions are orthogonal and vanishes when $d_j = d_c$ (degenerate case where parent and child represent the same direction).

**Interpretation as a necessary condition.** Theorem 1 establishes that $\text{EDA} > 0$ is *necessary* for absorption at a partial minimum. The converse does not hold---three mechanisms produce nonzero EDA in non-absorbed latents. First, polysemanticity: a latent encoding multiple unrelated concepts requires $w_{e,j}$ to serve as a compromise projection for multiple input distributions, while $d_j$ points along a single reconstruction direction. Second, the amortization gap (O'Neill et al., 2024): even without absorption, the shared encoder may fail to implement the optimal per-latent projection, producing nonzero EDA in healthy latents. Third, training noise: stochastic gradient descent may not converge to exact stationarity. EDA therefore provides an absorption-enriched screening signal, not a definitive diagnosis. Section 3.3 introduces D-EDA to partially disambiguate absorption from polysemanticity.

*Proof sketch.* At a partial minimum, the encoder direction satisfies a first-order stationarity condition with respect to the loss restricted to encoder parameters. Absorption introduces a perturbation: the gradient contribution from child-active inputs pushes $w_{e,j}$ away from $d_j$ by an amount proportional to $\delta$. Projecting this perturbation onto the component perpendicular to $d_j$ yields a residual with magnitude $\geq \delta \sin(\theta_{jc})$. Converting to cosine distance gives the quadratic form in Equation (3). The full proof builds on the partial minimum characterization of Tang et al. (2025).

## 3.3 D-EDA: Directional Decomposition

Scalar EDA measures the magnitude of encoder-decoder misalignment but does not reveal its cause. Directional EDA (D-EDA) decomposes the misalignment residual to distinguish absorption (residual explained by a few child decoder directions) from polysemanticity (residual distributed across many unrelated directions).

**Step 1: Compute the residual.** For latent $j$, project $w_{e,j}$ onto $d_j$ and take the perpendicular component:

$$r_j = w_{e,j} - \frac{w_{e,j} \cdot d_j}{\|d_j\|^2} \, d_j. \tag{4}$$

By construction, $r_j \perp d_j$. If $\text{EDA}(j) = 0$, then $\|r_j\| = 0$; large EDA implies large $\|r_j\|$.

**Step 2: Sparse projection onto the decoder dictionary.** Decompose $r_j$ in the basis of decoder columns via sparse regression:

$$r_j \approx \sum_k \beta_k \, d_k, \quad \text{minimizing } \|r_j - W_d \beta\|^2 + \mu \|\beta\|_1. \tag{5}$$

The sparse coefficient vector $\beta \in \mathbb{R}^{d_{\text{SAE}}}$ identifies which decoder directions explain the encoder's deviation from $d_j$.

**Step 3: Classify the residual.** The absorption signature is a sparse $\beta$ ($\|\beta\|_0$ small) with the active components $k$ satisfying $\cos(d_k, d_j) > 0.1$---the residual is explained by a few decoder directions geometrically related to the absorbed latent. The polysemanticity signature is a dense $\beta$ ($\|\beta\|_0$ large) with active components distributed across unrelated directions. We define the D-EDA absorption indicator as the ratio of variance explained by the top-3 decoder components to total residual variance. The absorbing source set is $S_j = \{k : |\beta_k| \text{ significant} \wedge \cos(d_k, d_j) > 0.1\}$.

**Limitation.** D-EDA does not outperform scalar EDA on Gemma Scope SAEs (Section 4). One exception: on GPT-2 Small layer 10, D-EDA achieves AUROC = 0.762 (95% CI: [0.686, 0.830]) where scalar EDA achieves only 0.336. The sparse projection in Equation (5) becomes ill-conditioned when $d_{\text{SAE}} \gg d_{\text{model}}$ (e.g., $65536 \gg 2304$ for Gemma Scope 65k SAEs), because the decoder dictionary is highly overcomplete and the regression has many near-degenerate solutions. Appendix B provides a conditioning analysis.

## 3.4 Synthetic Validation (SynthSAEBench)

Before testing EDA on real SAEs---where ground-truth absorption labels are noisy and scarce---we validate it under controlled conditions using a synthetic benchmark we constructed with known ground-truth absorption labels (SynthSAEBench; construction described in Appendix A).

**Setup.** Five synthetic SAEs are constructed, each with $d_{\text{model}} = 64$, $d_{\text{SAE}} = 500$ features, and 100 features designated as absorbed (absorption injected by rotating encoder directions away from their decoder directions toward a randomly selected child decoder direction). Absorption degree $\delta$ is drawn uniformly in $[0.3, 1.5]$.

As shown in Figure 2, EDA achieves perfect discrimination on this synthetic data. Across all 5 trials, AUROC = 1.000 and best-threshold F1 = 0.974. Absorbed latents have median EDA = 0.837, while non-absorbed latents have median EDA = 0.069---a $12\times$ separation ratio. The non-unity F1 reflects a small number of borderline cases where low-$\delta$ absorbed latents overlap with the most misaligned non-absorbed latents.

![SynthSAEBench validation: (a) ROC curve showing perfect discrimination (AUROC = 1.000); (b) EDA distributions for absorbed (median = 0.84) vs. non-absorbed (median = 0.07) synthetic latents. Five trials, 500 features each, 100 absorbed per trial.](figures/fig2_synthsae.pdf)

**Random direction baseline.** On a real Gemma Scope SAE (L12-16k, $d_{\text{SAE}} = 16384$), mean EDA for actual decoder directions is 0.214, while mean EDA computed against 100 random unit vectors substituted as decoder directions is 1.000 $\pm$ 0.002. Real encoder-decoder pairs are $4.7\times$ more aligned than chance, confirming EDA measures a genuine structural property of trained SAEs.

**Threshold sensitivity.** A $\pm$10% perturbation of the EDA threshold (around the median of 0.205) shifts the absorption rate classification by 19.8%---moderate sensitivity within the pre-registered 30% acceptability criterion. This motivates using percentile-based thresholds rather than fixed absolute values in subsequent analyses.

## 3.5 EDA vs. Baselines

EDA is evaluated against three baselines:

- **Decoder cosine similarity baseline.** Uses $1 - \cos(d_j, d_{\text{nearest}})$ as a proxy, measuring how isolated a decoder direction is in the dictionary. This tests whether decoder-only geometry suffices or the encoder direction carries additional signal.
- **Shuffled EDA null.** Randomly permutes the encoder-decoder pairing, destroying any learned alignment signal while preserving marginal distributions.
- **Dead feature indicator.** Flags latents with activation frequency $< 10^{-5}$. A simple filter that identifies latents that never fire, which trivially includes some absorbed latents.

EDA outperforms all baselines. The strongest comparison is at L5-16k: EDA achieves AUROC = 0.698 vs. decoder cosine similarity at 0.302---a gap of +0.396 AUROC (DeLong $p \approx 0$, 95% CI on difference: [0.276, 0.512]). The shuffled null achieves AUROC = 0.565 at the same configuration, confirming that the directional pairing between $w_{e,j}$ and $d_j$---not just the magnitude of $w_{e,j}$---carries the absorption signal. Full detection results across 8 SAE configurations appear in Section 4.

<!-- FIGURES
- Figure 1: fig1_absorption_mechanism_desc.md -- Two-panel diagram: (a) feature absorption mechanism with parent-child suppression; (b) EDA and D-EDA vector geometry
- Figure 2: gen_fig2_synthsae.py, fig2_synthsae.pdf -- SynthSAEBench validation: ROC curve (AUROC = 1.000) and EDA distribution violin plot (absorbed vs. non-absorbed)
-->
