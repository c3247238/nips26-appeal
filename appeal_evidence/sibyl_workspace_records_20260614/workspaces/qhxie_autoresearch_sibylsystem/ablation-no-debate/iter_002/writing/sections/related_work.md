# 4. Background and Related Work

## 4.1 Sparse Autoencoders and the Superposition Problem

Sparse autoencoders decompose dense neural activations into sparse, interpretable feature directions via a bottleneck architecture. An SAE maps residual stream activations $x \in \mathbb{R}^d$ to sparse features $f \in \mathbb{R}^{d_{sae}}$ through an encoder $W_{enc}$ and reconstructs via a decoder $W_{dec}$: $x \approx W_{dec} \, \sigma(W_{enc} x + b_{enc}) + b_{dec}$, where $\sigma$ is a sparsity-inducing nonlinearity such as TopK activation. The expansion ratio $d_{sae} / d_{model}$ typically ranges from 4x to 64x, creating an overcomplete dictionary in which each latent dimension ideally corresponds to a monosemantic concept.

The theoretical justification for SAEs rests on the superposition hypothesis: neural networks represent more features than dimensions by packing concepts into non-orthogonal directions. Bricken et al. (2023/2025) demonstrated the feasibility of million-feature SAEs on Claude 3 Sonnet, establishing that scale alone can improve feature recovery. However, precision/recall analysis in that work revealed persistent imbalances---not all learned features correspond to ground-truth concepts, and not all ground-truth concepts are recovered.

## 4.2 Feature Absorption: Definition and Prior Characterization

Feature absorption occurs when child features (subordinate concepts) substitute for parent features (superordinate concepts) in the sparse representation, leaving the parent latent inactive despite the parent concept being present in the input. Chanin et al. (2024/2025) provided the first systematic definition and proved via toy models that hierarchy combined with sparsity optimization inevitably causes absorption. Their absorption rate metric---computed as the fraction of parent true positives where children compensate---has become the standard evaluation criterion. The authors validated absorption on hundreds of LLM SAEs and showed it is loss-minimizing: when parent and child features are geometrically aligned, suppressing the parent and routing activation through children reduces the sparsity penalty without sacrificing reconstruction.

Marks et al. (2025/2026) formalized the optimization landscape through a unified theory of sparse dictionary learning. Their framework identifies piecewise biconvexity and spurious minima as the structural cause: hierarchical data induces absorbing partial minima in the loss surface, and standard training converges to these minima because they offer locally optimal sparsity-reconstruction trade-offs. Feature anchoring---initializing dictionary elements near ground-truth directions---improves recovery across architectures, but the theory does not decompose absorption into encoder versus decoder contributions.

Cui et al. (2025) derived closed-form solutions for SAEs under idealized conditions and showed that full feature recovery is achievable only under extreme sparsity (near-zero $L_0$), a regime incompatible with practical interpretability. Their analysis confirms that partial recovery---where some features absorb others---is the generic outcome under realistic constraints.

## 4.3 Architecture Solutions and Their Limitations

Multiple architectural innovations target absorption directly. Matryoshka SAE (Bussmann et al., 2025) trains nested dictionaries simultaneously: smaller inner dictionaries learn general features, while larger outer dictionaries specialize. This explicit multi-level structure reduces absorption from 0.49 to 0.05 on standard benchmarks, though at approximately 50% computational overhead and slightly worse reconstruction. Feature hedging---a related pathology where correlated features merge in narrow SAEs---can still occur in the inner levels (Chanin & Garriga-Alonso, 2025).

HSAE (Luo et al., 2026) constructs a feature forest by jointly training a series of SAEs with explicit parent-child structural constraints. A dedicated structural loss plus random perturbation during training substantially outperforms flat baselines on absorption metrics, particularly at larger scales. OrtSAE (Korznikov et al., 2025) imposes an orthogonality penalty on latents, achieving 65% absorption reduction with minimal training overhead. AdaptiveK SAE (Till, 2025) allocates dynamic per-input sparsity and reports the best absorption scores on SAEBench despite training on 2,000x less data than competitors.

Oursland (2026) proposed a decoder-free SAE trained from first principles, eliminating the decoder entirely and thereby removing one potential source of absorption. However, all these approaches treat absorption as a joint encoder-decoder phenomenon and do not empirically decompose which component drives the effect.

## 4.4 Evaluation Benchmarks and Metrics

SAEBench (Karvonen et al., 2025) introduced an eight-metric benchmark (absorption, sparse probing, auto-interpretability, RAVEL, unlearning, SCR, TPP, cross-entropy loss) evaluated on over 200 SAEs. A critical finding from this benchmark is that proxy metrics---reconstruction quality and $L_0$ sparsity---do not reliably predict practical interpretability performance. An SAE with low reconstruction error may still exhibit high absorption and poor feature recovery.

SynthSAEBench (Chanin et al., 2026) provides ground-truth synthetic data with 16,000 features, explicit hierarchy, controlled correlation, and superposition. This benchmark revealed that Matching Pursuit SAEs exploit superposition noise without learning true features, and no existing architecture achieves perfect performance. Feature sensitivity (Hu et al., 2025) adds a complementary evaluation dimension: the fraction of LLM-generated similar texts that activate a given feature. The authors found that sensitivity declines with SAE width, suggesting that scale introduces reliability trade-offs distinct from absorption.

## 4.5 Research Gap

Despite this progress, no prior work empirically decomposes absorption into encoder and decoder contributions. The factorial design we introduce in Section 5.4 directly addresses this gap. Existing theoretical frameworks (Marks et al., 2025/2026; Cui et al., 2025) treat the SAE as a monolithic optimization problem; our decomposition reveals that the encoder and decoder play asymmetric roles, with implications for where mitigation efforts should focus.

<!-- FIGURES
- None
-->
