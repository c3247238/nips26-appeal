# 2. Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders have become the dominant tool for mechanistic interpretability of large language models. An SAE learns an overcomplete dictionary $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$ such that activations $a \in \mathbb{R}^{d_{\text{model}}}$ decompose into sparse latent codes $z \in \mathbb{R}^{d_{\text{dict}}}_{\geq 0}$ via $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$, with reconstruction $\hat{a} = W_{\text{dec}} z + b_{\text{dec}}$. The sparsity constraint ensures that each latent activates on only a small fraction of inputs, yielding interpretable features that correspond to human-understandable concepts.

Bricken et al. (2023) demonstrated that SAEs trained on transformer activations produce monosemantic features---latents that respond to single concepts such as "month names" or "DNA sequences"---thereby addressing the superposition problem, where neural networks represent more features than they have dimensions. Cunningham et al. (2023) scaled this approach to larger models and showed that SAE features support circuit analysis: specific latents can be identified as components of computational subgraphs that implement tasks such as indirect object identification. Templeton et al. (2024) extended SAEs to frontier models with millions of latents, validating the paradigm at scale.

The field has invested heavily in tooling and standardization. SAELens (Bloom, 2024) provides a unified library for training and analyzing SAEs across architectures. SAEBench (Karvonen et al., 2025) offers standardized benchmarking for absorption, dead neurons, and other pathologies. These efforts have made SAEs the default approach for unsupervised feature discovery in language models.

## 2.2 Feature Absorption

Feature absorption, first identified by Chanin et al. (2024), is a failure mode where general (parent) features fail to fire when more specific (child) features are present. In a canonical example, a latent representing "starts with any letter" may not activate when "starts with Q" fires, even though the input satisfies both conditions. The parent feature is not destroyed---its decoder direction $W_{\text{dec}}[i]$ remains intact---but its encoder activation is suppressed.

Chanin et al. proposed a differential correlation metric for detecting absorption. For a parent feature $i$ and child feature $j$, the metric compares the correlation between their activations before and after ablating $j$. A drop in correlation indicates that $j$ was capturing activation that should have belonged to $i$. SAEBench (Karvonen et al., 2025) standardized this metric, enabling cross-model comparison.

Architectural solutions have emerged to reduce absorption. Matryoshka SAEs (Bussmann et al., 2025) use hierarchical dictionary structure with nested subspaces, reducing the probability that a child feature can fully capture a parent's activation. OrtSAE (Korznikov et al., 2025) enforces orthogonality constraints on decoder directions, directly limiting the correlations that enable absorption. HSAE (hierarchical SAE) and ATM (adaptive threshold matching) similarly modify training objectives. All these approaches require retraining the SAE from scratch.

A critical gap remains: no existing work explains *why* absorption occurs or *which* features are at risk before running absorption metrics. The differential correlation metric detects absorption post hoc; practitioners must evaluate every feature pair, a process that scales as $O(d_{\text{dict}}^2)$. The architectural solutions reduce absorption but do not provide a mechanistic theory connecting the phenomenon to SAE structure.

## 2.3 The Locally Competitive Algorithm

The Locally Competitive Algorithm (LCA), proposed by Rozell et al. (2008), solves sparse coding through a dynamical system with lateral inhibition. The LCA state $u \in \mathbb{R}^{d_{\text{dict}}}$ evolves according to:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$$

where $b = W_{\text{enc}}^T x$ is the feedforward input, $G \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{dict}}}$ is the inhibition matrix, and $T(u) = \max(0, u)$ is the threshold function. The inhibition matrix governs competitive dynamics: a large positive $G_{ij}$ means that when neuron $j$ is active, it suppresses neuron $i$.

Rozell et al. showed that for a dictionary $W_{\text{dec}}$, the natural choice is $G = W_{\text{dec}}^T W_{\text{dec}}$, because the inhibition term $G \cdot a$ then equals $W_{\text{dec}}^T (W_{\text{dec}} a)$, which is the projection of the reconstructed signal back into the latent space. This choice ensures that the dynamics converge to a sparse representation that optimally reconstructs the input under an $\ell_1$ sparsity penalty.

The LCA has approximately 2,000 citations and has been applied to image denoising, compressed sensing, and neural population coding. However, *no prior work connects LCA to sparse autoencoders for language model interpretability*. The structural correspondence between $G = W_{\text{dec}}^T W_{\text{dec}}$ and the SAE decoder correlation matrix has not been articulated in the SAE literature. Section 3.1 formalizes this correspondence and derives its implications for feature absorption.

## 2.4 Competitive Dynamics in Neural Networks

Competitive dynamics appear throughout neural computation. In biological networks, lateral inhibition enhances contrast and selectivity in sensory processing: active neurons suppress neighbors, sharpening receptive fields (Hartline & Ratliff, 1957). In deep learning, softmax attention implements a form of competition where tokens vie for normalized weight. Winner-take-all (WTA) circuits, used in sparse coding and clustering, explicitly select the most active unit while suppressing others (Makhzani & Frey, 2015).

The connection between decoder correlations and competition has been noted in passing. Schubert et al. (2023) observed that highly correlated decoder directions can cause feature splitting, where a single concept is represented by multiple latents. Lieberum et al. (2023) discussed decoder correlation as a source of interference in SAE reconstructions. Neither work framed these correlations as an inhibition matrix or connected them to the LCA framework.

Our contribution is to make this connection explicit and exploit it. We show that $W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing a mechanistic explanation for absorption as competitive suppression. We construct the first local inhibition graph from decoder correlations and test whether it predicts absorption pairs. The framework is entirely training-free, computed from pretrained weights, and scales to million-latent SAEs via top-$k$ neighbor sparsification.

<!-- FIGURES
- None
-->
