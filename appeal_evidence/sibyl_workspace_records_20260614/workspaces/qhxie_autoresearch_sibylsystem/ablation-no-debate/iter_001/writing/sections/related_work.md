# Related Work

## Sparse Autoencoders

Sparse Autoencoders (SAEs) trained on transformer residual stream activations have become a standard tool for decomposing dense model representations into interpretable features. Bricken et al. (2023) demonstrated that SAEs applied to GPT-4 activations recover thousands of monosemantic features -- including features for syntactic structures, semantic concepts, and entities -- that activate selectively to specific patterns in the input. Templeton et al. (2024) extended this approach to Claude, documenting a rich feature universe including emotional states, reasoning processes, and social dynamics.

SAEs typically use a TopK activation function that enforces sparsity by retaining only the k highest-scoring features (Huber, 1962), where k is chosen to produce a target L0 (number of active features). The encoder maps residual stream activations $x \in \mathbb{R}^d$ to sparse feature activations $f \in \mathbb{R}^{d_{sae}}$ via a learned weight matrix $W_{enc}$, and the decoder reconstructs the input via $x' = W_{dec} f + b_{dec}$. Training optimizes the reconstruction loss while encouraging sparse activations.

Recent work has explored alternative sparsity mechanisms beyond TopK, including L1 regularization (Wright et al., 2010), jumps (Sharkey & Sharkey, 2024), and learned top-k mechanisms (Gao et al., 2025). The choice of sparsity mechanism may affect absorption patterns; we use TopK following prior absorption work (Chanin et al., 2024).

## Feature Absorption

The absorption phenomenon was first systematically documented by Chanin et al. (2024), who introduced the concept of "feature absorption" in SAE representations. They defined absorption as occurring when a parent feature's decoder direction is spanned by the decoder directions of multiple child features, allowing children to substitute for the parent in sparse representations.

Their methodology measured absorption via single-feature ablation: activate the parent feature by providing appropriate input, then ablate individual child features and measure whether the parent's reconstruction quality degrades. If ablating a child significantly degrades parent reconstruction, that child is considered to be absorbing the parent.

However, this single-child ablation approach suffers from a saturation problem: ablating one child allows remaining children to reconstruct the parent, producing absorption rates of 1.0 regardless of whether the SAE has learned structured representations. This saturation is a core motivation for our multi-child proportional ablation methodology.

Gao et al. (2025) proposed MultiScale SAEs that learn features at multiple resolutions, potentially addressing absorption by decomposing concepts at different granularities. Their approach is complementary to our work: rather than fixing absorption, MultiScale SAEs may reduce its impact by capturing parent-child relationships across scales.

## Sanity Checks for Interpretability

A growing body of work questions whether standard interpretability benchmarks and metrics capture genuine mechanistic properties rather than statistical artifacts.

Korznikov et al. (2026) demonstrated that random baselines -- SAEs with untrained or shuffled weights -- recover similar feature counts and activation statistics to trained SAEs on standard interpretability benchmarks. Their "sanity checks" framework establishes that demonstrating a phenomenon requires showing it is not present in random baselines. This finding motivates our use of multiple baseline conditions: random decoder, shuffled features, and permuted encoder.

Song et al. (2025) proposed a feature consistency position that argues interpretability features should be validated against independent methods before being used for downstream applications. They note that high explained variance does not imply feature identity, a concern that applies directly to absorption metrics.

Korznikov et al. (2025) introduced OrtSAE, which uses geometry-based regularization to improve feature quality. Their orthogonalization approach modifies the decoder structure to reduce feature co-activation, potentially reducing absorption as a side effect. We do not use OrtSAE in this study, but acknowledge it as a potential mitigation strategy.

## Steering and Feature Sensitivity

A central question for absorption is whether it is merely epistemic (a measurement artifact of how we represent features) or causal (an active mechanism degrading downstream performance). Steering experiments provide a methodology for testing causal relationships.

O'Brien et al. (2024) demonstrated that SAE features can be used for targeted steering -- adding scaled directions to model activations to evoke or suppress specific behaviors. They found that steering sensitivity varies across features, with some features responding strongly to interventions while others are resistant. Our work extends this by testing whether resistance to steering correlates with absorption status.

Tian et al. (2025) proposed methods for measuring feature sensitivity that enable quantitative comparison across features. They showed that sensitivity scores vary with feature type and training configuration. We use their framework to measure whether absorbed features have systematically lower sensitivity than non-absorbed features.

The distinction between epistemic and causal failure is important for interpretability methodology. An epistemic failure means our measurement tools are inadequate; a causal failure means the phenomenon actively degrades model behavior. Our steering experiments (Section 6.3) test this distinction directly.

## Competitive Exclusion and Ecological Analogies

The ecological analogy between feature absorption and biological competitive exclusion has been proposed as a theoretical framework for predicting when absorption occurs (Korznikov et al., 2025; Kalmykov & Kalmykov, 2012). The competitive exclusion principle states that two species competing for the same resources cannot coexist at constant population values; applied to SAEs, this suggests that features with overlapping decoder subspaces cannot both be active simultaneously.

Blumenthal & Mehta (2023) studied the geometry of ecological coexistence using Lotka-Volterra-style models, providing mathematical tools for analyzing competitive dynamics. We draw on their geometric framework to test whether absorption follows competitive exclusion patterns.

However, the ecological analogy may not directly apply to SAE feature dynamics. Features do not "compete" in the same biological sense; rather, they are jointly optimized to minimize reconstruction loss. Whether this optimization produces competitive exclusion patterns is an empirical question that our frequency-absorption correlation analysis addresses (Section 6.2).

## Synthesis

This work sits at the intersection of SAE methodology, absorption measurement, and causal validation. We build on the foundational work of Bricken et al. (2023) and Chanin et al. (2024) while addressing the sanity-check concerns raised by Korznikov et al. (2026). Our multi-child proportional ablation methodology resolves the saturation problem in prior absorption measurement, and our steering experiments test whether absorption is causal or epistemic. This provides a more rigorous foundation for understanding absorption's implications for SAE-based interpretability.

<!-- FIGURES
- None
-->
