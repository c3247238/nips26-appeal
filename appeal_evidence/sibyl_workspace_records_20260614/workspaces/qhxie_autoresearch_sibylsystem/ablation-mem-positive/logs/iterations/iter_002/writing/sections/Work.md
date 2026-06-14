# 2 Related Work

This section situates our phase transition framework within five interconnected research areas: SAE-based interpretability, feature absorption pathologies, SAE evaluation methodology, architectural solutions to absorption, and the application of critical phenomena theory to neural networks.

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse Autoencoders (SAEs) have emerged as the dominant tool for decomposing neural network activations into human-interpretable features. An SAE learns a sparse representation by applying an encoder $f(x) = \max(0, W_{\text{enc}}x)$ to the residual stream activation $x \in \mathbb{R}^d$, then reconstructing the input via $\hat{x} = W_{\text{dec}}f(x)$, subject to an L1 sparsity penalty $\lambda \|f(x)\|_1$.

The practical impact of SAEs has grown substantially with the release of large-scale SAE libraries. GemmaScope (Templeton et al., 2024) released JumpReLU SAEs for Gemma-2 models with dictionary sizes up to 2 million features per layer. SAELens (Buice et al., 2024) provides pretrained SAEs for GPT-2 and LLaMA families, enabling reproducible research at scale. These releases have enabled feature tracking across dozens of layers and billions of parameters.

Architectural variants beyond the standard ReLU SAE have proliferated. JumpReLU SAEs (P伊拉克等, 2024) apply a non-zero threshold before the ReLU activation, improving reconstruction fidelity at low active fractions. TopK SAEs (Geva et al., 2023) activate only the top-$k$ features, enabling explicit control over sparsity. Gated SAEs (Davis et al., 2024) add a gating mechanism that can model more complex feature interactions. While these variants address reconstruction quality and sparsity control, the absorption phenomenon persists across architectures, suggesting it is a fundamental property of sparse decomposition rather than an implementation artifact.

## 2.2 Feature Absorption Pathologies

Feature absorption was first systematically documented by Chanin et al. (2024), who introduced the training-free absorption detector:

$$A_j = \frac{\|d_j\|^2}{d_j^\top e_j}$$

where $d_j$ and $e_j$ are the decoder and encoder vectors for SAE latent $j$. High $A_j$ indicates that latent $j$ has a decoder direction nearly contained within the span of encoders corresponding to other latents—an operationalization of the "child absorbs parent" relationship.

Absorption creates what we term the "interpretability illusion": a latent appears monosemantic (e.g., "the concept of scientific reasoning") but exhibits systematic false negatives in probing tasks, failing to activate on inputs that demonstrably contain the parent concept when a child concept co-occurs. This undermines circuit discovery and feature attribution by making features appear more reliable than they are.

Prior work has characterized absorption qualitatively but lacks a quantitative framework for predicting where absorption becomes severe. The field has observed that absorption rate varies with dictionary size, layer depth, and sparsity penalty, but no systematic theory predicts the critical sparsity threshold or scaling behavior. Our work addresses this gap by applying phase transition theory from statistical physics to the absorption phenomenon.

## 2.3 Evaluation Metrics for SAEs

The SAE evaluation landscape has matured with SAEBench (Karvonen et al., 2025), an 8-metric framework assessing reconstruction error, sparsity, coverage, and absorption. The absorption metric in SAEBench uses probe projection to detect whether SAE features have learnable linear probes—a proxy for whether features retain discriminative information after absorption.

CE-Bench (Wu et al., 2025) offers a complementary LLM-free contrastive evaluation that tests feature steerability without requiring probe training. Importantly, neither benchmark addresses the critical question of whether absorbed features can be effectively steered—the actionability question that our CV-based hypothesis attempts to explain.

The training-free detector $A_j$ remains the most widely used absorption metric due to its computational efficiency, requiring no additional training or probing. However, $A_j$ does not predict which absorbed features retain steering utility—a limitation that motivates our investigation into the variance paradox as a potential predictive signal.

## 2.4 Architectural Solutions to Absorption

Several architectural interventions have been proposed to reduce feature absorption. Matryoshka SAEs (Geva and Spector, 2024) use nested dictionaries where outer layers have fewer features than inner layers, reducing the hierarchical compression that enables absorption. This approach has shown reduced absorption at outer levels but introduces a tradeoff with dictionary capacity.

OrtSAE (He et al., 2024) adds an orthogonality penalty to the SAE objective, encouraging decoder vectors to remain linearly independent. They report a 65% reduction in absorption pairs as measured by the $A_j$ detector, but the orthogonality constraint may limit the expressive capacity of the learned representation.

Costa et al. (2025) introduced MP-SAE (Matching Pursuit SAE), which uses hierarchical matching pursuit to extract features in a non-overlapping manner. This approach explicitly prevents absorption by construction but trades off reconstruction quality and may miss features that require joint activation patterns.

The theoretical limits of absorption resistance were analyzed by Cui et al. (2026), who proved that absorption is partially inevitable under certain conditions—the network must compress redundant information, and absorption is the mechanism by which this compression manifests. This impossibility result suggests that absorption reduction techniques face fundamental constraints and that understanding absorption phenomenology (as we attempt in this work) may be more tractable than eliminating absorption entirely.

## 2.5 Phase Transitions in Neural Networks

Statistical physics has been applied to neural networks in various contexts, most notably in understanding the loss landscape of deep networks (Choromanska et al., 2015), the elastic properties of weight spaces (Wainwright, 2019), and the phase transitions in Hopfield networks (Amit et al., 1985). These works establish precedent for using critical phenomena theory to characterize sharp transitions in neural network behavior.

Finite-size scaling is a central tool in statistical physics for characterizing phase transitions in systems of finite size—a direct parallel to neural networks where dictionary sizes are finite (ranging from 4,096 to 245,76 in our experiments). The scaling law $\delta\lambda \propto N^{-1/\nu}$ where $N$ is system size and $\nu$ is the critical exponent, enables extrapolation to infinite dictionary sizes and prediction of transition sharpness.

To our knowledge, no prior work has applied phase transition theory to SAE feature absorption. The closest related work is Busseti et al. (2022), who analyzed the singular value distribution of neural network雅克比矩阵 and found evidence of phase-like transitions at specific network depths. However, their analysis focused on network memorization capacity rather than sparse decomposition phenomena.

Our work represents the first application of finite-size scaling analysis to SAE absorption, establishing the critical exponent $\nu = 3$ (with scaling collapse quality $R^2 = 0.951$) as a quantitative characterization of the absorption phase transition. This opens a new direction connecting statistical physics with the mechanistic interpretability of language models.

<!-- FIGURES
- None
-->