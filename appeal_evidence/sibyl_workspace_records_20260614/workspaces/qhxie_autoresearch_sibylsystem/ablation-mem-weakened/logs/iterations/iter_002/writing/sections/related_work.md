# 2. Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders have become the dominant unsupervised approach for decomposing neural network activations into human-interpretable features. An SAE reconstructs activations $a \in \mathbb{R}^{d_{\text{model}}}$ as $\hat{a} = W_{\text{dec}} \cdot \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}) + b_{\text{dec}}$, where $W_{\text{enc}} \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{model}}}$ is the encoder, $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$ is the decoder, and $d_{\text{dict}} \gg d_{\text{model}}$ creates an overcomplete representation. The foundational premise, rooted in the superposition hypothesis (Elhage et al., 2022), is that neural networks represent more features than they have dimensions by encoding them in overlapping, approximately orthogonal directions. SAEs unpack this superposition into sparse, monosemantic features through dictionary learning with sparsity constraints.

Bricken et al. (2023) demonstrated the first large-scale SAE feature extraction from language models, showing that learned latents correspond to interpretable concepts such as legal language, DNA sequences, and grammatical structures. Templeton et al. (2024) scaled this approach to Claude 3 Sonnet, extracting millions of features and validating them through sparse probing and steering interventions. Marks et al. (2024) used feature steering to identify sparse feature circuits---causal subgraphs of model computation---demonstrating that SAE features can be manipulated to alter model behavior with high precision.

These applications depend on a critical assumption: that SAE features are reliable. A feature that appears interpretable when inspected in isolation must also behave predictably when used for steering, probing, or circuit analysis. The credibility of the entire SAE paradigm rests on this assumption, and recent work has challenged it.

## 2.2 Feature Absorption

Feature absorption is a failure mode where a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. Chanin et al. (2024) formally identified absorption and proved that it is a logical consequence of the sparsity objective under hierarchical feature structure. Their detection method uses differential correlation: a child $c$ is flagged as absorbing parent $f$ if the correlation between their activations, conditioned on the parent concept being present, exceeds a threshold after ablating the child.

Chanin et al. validated absorption across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families, establishing it as a cross-architecture phenomenon. SAEBench (Karvonen et al., 2025) subsequently adopted the Chanin et al. differential correlation metric as a standardized benchmark alongside sparsity, reconstruction error, and explained variance. The metric is now reported routinely in SAE evaluations.

Despite this attention, a critical gap remains: **no existing work provides a mechanistic theory that explains why absorption happens or identifies which features are at risk before running absorption metrics.** Researchers can detect absorption after the fact, but they cannot predict it, explain its structure, or repair it without retraining. This gap is the motivation for our work.

## 2.3 The Locally Competitive Algorithm

Rozell et al. (2008) proposed the Locally Competitive Algorithm (LCA) for sparse coding, where neurons compete via lateral inhibition. The LCA dynamics are:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$$

where $u$ is the membrane potential, $b$ is the feedforward input, $G$ is the inhibition matrix governing competitive interactions between neurons, and $T(u) = \max(0, u)$ is the threshold function. In LCA, neurons with strong feedforward input suppress neighboring neurons via the inhibition matrix $G$, producing sparse activation patterns.

The LCA framework has received approximately 2,000 citations across neuroscience, signal processing, and compressed sensing, yet **no prior work connects LCA to sparse autoencoders for language model interpretability.** The structural correspondence we develop in Section 3---that $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs---has not been articulated in the SAE literature. This connection provides the theoretical foundation for our local inhibition graph.

## 2.4 Competitive Dynamics in Neural Networks

Competitive dynamics appear throughout neural network architectures. Lateral inhibition in biological neural networks enhances contrast and selectivity in sensory processing (retina, cochlea). Softmax competition in attention mechanisms allocates limited capacity across positions. Winner-take-all (WTA) circuits in deep learning enforce sparse activation through explicit competition.

In the context of SAEs, competitive dynamics have been addressed primarily through architectural constraints rather than mechanistic analysis. Matryoshka SAEs (Bussmann et al., 2025) use nested dictionary structures to reduce absorption by encoding broader concepts at coarser granularities. OrtSAE (Korznikov et al., 2025) enforces decoder orthogonality, reducing absorption by 65% by preventing feature overlap. HSAE (Luo et al., 2026) explicitly models hierarchical structure in the SAE architecture. All three approaches target absorption reduction through structural modification, but none explains the mechanism that causes absorption or provides a training-free diagnostic.

Our contribution is distinct: we connect decoder correlations to competitive suppression via the LCA framework, providing a mechanistic explanation that is exact (not metaphorical) and yields a training-free diagnostic tool.

## 2.5 Training-Free SAE Analysis

A growing body of work analyzes pretrained SAEs without retraining. SAEBench (Karvonen et al., 2025) provides standardized evaluation metrics for pretrained SAEs across architectures. GemmaScope (Lieberum et al., 2024) releases comprehensive pretrained JumpReLU SAEs for community analysis. Recent "sanity check" studies show that frozen random baselines achieve comparable performance to trained SAEs on several metrics, raising questions about whether SAE training captures meaningful structure or merely fits activation statistics.

Our local inhibition graph is entirely training-free: it is computed from pretrained decoder weights in a single pass. This aligns with the practical constraint that researchers often work with publicly available pretrained SAEs and cannot retrain for every analysis.

<!-- FIGURES
- None
-->
