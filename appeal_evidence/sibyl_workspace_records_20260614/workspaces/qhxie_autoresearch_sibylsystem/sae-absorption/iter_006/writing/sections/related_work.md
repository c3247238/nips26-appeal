# 2 Background and Related Work

## 2.1 Sparse Autoencoders in Mechanistic Interpretability

An SAE maps a residual stream activation $x \in \mathbb{R}^{d_{\text{model}}}$ to a sparse latent vector $z \in \mathbb{R}^{d_{\text{SAE}}}$ via encoder $W_e$ and reconstructs $\hat{x} = W_d z$, where decoder columns $d_j$ are unit-normalized. Two architecture families dominate current practice. L1-ReLU SAEs (Cunningham et al., 2023; Bricken et al., 2023) apply a soft L1 penalty: $z_j = \text{ReLU}(w_{e,j}^\top x + b_{e,j})$, producing graded activation suppression. JumpReLU SAEs (Rajamanoharan et al., 2024) impose a hard per-latent threshold $\theta_j$:
$$z_j = (w_{e,j}^\top x + b_{e,j}) \cdot \mathbb{1}[w_{e,j}^\top x + b_{e,j} > \theta_j]$$
trained via the straight-through estimator. The $L_0$ operating point---the number of non-zero latents per forward pass---controls the sparsity-fidelity tradeoff. Gemma Scope (Lieberum et al., 2024) provides 400+ open JumpReLU SAEs on Gemma 2 2B/9B/27B with configurable $L_0$, making it the primary testbed for absorption research on modern architectures.

SAEs have been scaled to frontier models (Templeton et al., 2024; Gao et al., 2024) and applied to circuit analysis (Lindsey et al., 2025), but theoretical guarantees remain limited. Leask et al. (2025) show SAE features are neither canonical nor atomic. Cui et al. (2025) prove recovery fails unless features are extremely sparse. These results motivate direct validation of what SAE-derived metrics actually capture.

## 2.2 Feature Absorption

Chanin et al. (2024) formalize feature absorption: a parent latent $z_p$ (e.g., "starts-with-A words") drops to $z_p = 0$ on parent-positive inputs when a child latent $z_c$ (e.g., "starts-with-A proper nouns") is active ($z_c > 0$). Their measurement protocol trains $k$-sparse logistic regression probes on SAE latent activations, identifies SAE latents whose decoder directions align with the probe (cosine similarity $\geq \tau_{\cos}$), and computes the false-negative rate on parent-positive inputs. Across hundreds of SAEs in SAEBench (Karvonen et al., 2025), absorption rates of 15--35% are reported on first-letter spelling---the only task on which absorption has been measured. All published results use GPT-2 Small with L1-ReLU SAEs; no study has validated the Chanin metric on JumpReLU SAEs.

Tang et al. (2025) show that the biconvex optimization landscape of sparse dictionary learning contains partial minima where absorption is locally optimal. O'Neill et al. (2024) identify a structural cause: the amortization gap in feedforward encoding prevents the encoder from recovering all dictionary atoms. Chanin and Garriga-Alonso (2025) demonstrate that incorrect $L_0$ causes SAEs to learn systematically wrong features, with too-low $L_0$ triggering hedging---correlated features merged due to insufficient capacity. Hedging produces false negatives that mimic absorption but arise from information spreading rather than competitive suppression. This hedging-versus-hierarchy-driven-absorption distinction is central to our confound decomposition (Section 4.2).

## 2.3 Architectural Mitigations

Matryoshka SAE (Bussmann et al., 2025; ICML 2025) trains nested dictionaries that preserve parent features at smaller sizes. OrtSAE (Korznikov et al., 2025) enforces orthogonality via a pairwise cosine penalty. ATM-SAE (Li et al., 2025) applies adaptive temporal masking, reporting absorption scores of 0.007 versus 0.140 for TopK and 0.011 for JumpReLU on Gemma 2 2B. KronSAE (2025) exploits Kronecker factorization. MP-SAE (Costa et al., 2025) uses iterative encoding. Masked regularization (Narayanaswamy et al., 2026) disrupts co-occurrence patterns during training.

All six mitigations assume the Chanin metric measures hierarchy-driven competitive exclusion. None validates the metric on JumpReLU SAEs, the architecture whose hard-threshold activation dynamics differ most from the L1-ReLU SAEs on which the metric's thresholds ($\tau_{\cos} \geq 0.025$, $\tau_{\text{mag}} \geq 1.0$) were calibrated.

## 2.4 Rate-Distortion Theory and Successive Refinement

The successive refinement theorem (Equitz and Cover, 1991) states that a source is successively refinable---encodable in stages without information loss---if and only if the descriptions form a Markov chain. Applied to SAE features: when $X \to f_{\text{child}} \to f_{\text{parent}}$ is Markov, the conditional mutual information (CMI) $I(X; f_{\text{parent}} \mid f_{\text{child}}) = 0$ and the parent carries no unique information beyond the child. Absorbing the parent is information-theoretically lossless. When CMI is positive, absorption destroys unique information that the parent encodes about the input.

No prior work has connected successive refinement to SAE feature absorption. This connection provides the theoretical basis for our rate-distortion diagnostic (Section 6): features with low CMI are cheap to absorb under sparsity pressure, while features with high CMI resist absorption because suppressing them incurs an information cost.

Section 3 describes the experimental methodology for our three-pronged study.

<!-- FIGURES
- None
-->
