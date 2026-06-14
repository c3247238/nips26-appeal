# 2 Background and Related Work

## 2.1 Sparse Autoencoders in Mechanistic Interpretability

A Sparse Autoencoder (SAE) maps a residual stream activation $x \in \mathbb{R}^{d_{\text{model}}}$ to a sparse latent vector $z \in \mathbb{R}^{d_{\text{SAE}}}$ via encoder weights $W_e$ and reconstructs $\hat{x} = W_d z$, where each decoder column $d_j$ is unit-normalized (Cunningham et al., 2023; Bricken et al., 2023). Two architecture families dominate current practice:

**L1-ReLU SAEs** apply a continuous L1 penalty on latent activations: $z_j = \text{ReLU}(w_{e,j}^\top x + b_{e,j})$, producing graded suppression that scales smoothly with penalty strength (Cunningham et al., 2023; Bricken et al., 2023). Scaling to frontier models demonstrated that SAEs recover safety-relevant and semantically coherent features (Templeton et al., 2024; Gao et al., 2024).

**JumpReLU SAEs** impose a hard per-latent threshold $\theta_j$:
$$z_j = (w_{e,j}^\top x + b_{e,j}) \cdot \mathbb{1}[w_{e,j}^\top x + b_{e,j} > \theta_j]$$
trained via the straight-through estimator (Rajamanoharan et al., 2024). The binary activation regime---fire or zero---differs qualitatively from L1-ReLU's graded suppression. The $L_0$ operating point, the number of non-zero latents per forward pass, directly controls the sparsity--fidelity tradeoff. Gemma Scope (Lieberum et al., 2024) provides 400+ open JumpReLU SAEs on Gemma 2 2B/9B/27B with configurable $L_0$, making it the largest public testbed for SAE evaluation.

Theoretical limitations constrain what SAEs can reliably recover. Leask et al. (2025) show that SAE features are neither canonical (different training runs yield different feature sets) nor atomic (meta-SAEs decompose individual features into sub-features). Cui et al. (2025) prove that recovery of ground-truth features fails unless features are extremely sparse. Engels et al. (2025) discover irreducible multi-dimensional features (e.g., circular representations for days and months) that one-dimensional SAE latents cannot faithfully represent. These results motivate direct empirical validation of what SAE-derived metrics actually capture, rather than assuming interpretable features correspond to ground-truth concepts.

## 2.2 Feature Absorption

Chanin et al. (2024; NeurIPS 2025 Oral) define feature absorption: a parent latent $z_p$ (e.g., encoding "starts-with-A") drops to $z_p = 0$ on parent-positive inputs when a child latent $z_c$ (e.g., encoding a specific word token) is active ($z_c > 0$). The measurement protocol trains $k$-sparse logistic regression probes ($k = 5$) on SAE latent activations, identifies candidate SAE latents whose decoder columns have cosine similarity $\cos(d_j, v_p) \geq 0.025$ to the probe direction $v_p$, and computes the false-negative rate---the fraction of parent-positive tokens where all $k$ probe-associated latents are inactive. Across hundreds of SAEs evaluated in SAEBench (Karvonen et al., 2025; ICML 2025), absorption rates of 15--35% are reported on the first-letter spelling task. All published absorption measurements use GPT-2 Small with L1-ReLU SAEs; no study has validated the Chanin metric on JumpReLU SAEs.

Two theoretical accounts explain why absorption arises. Tang et al. (2025) show that the biconvex optimization landscape of sparse dictionary learning contains partial minima where absorption is locally optimal---the encoder settles into a configuration where suppressing the parent saves one unit of $L_0$ at negligible reconstruction cost. O'Neill et al. (2024) identify the amortization gap: feedforward encoding cannot recover all dictionary atoms, so the encoder systematically fails to activate parents when children already account for most of the reconstruction.

A distinct but observationally similar phenomenon is feature hedging (Chanin & Garriga-Alonso, 2025): when the $L_0$ operating point is too low, the SAE merges correlated features, spreading parent information across many latents rather than concentrating it in a single parent latent. Hedging produces false negatives that mimic absorption---the parent latent does not fire---but the mechanism is capacity starvation rather than competitive suppression. The false negatives from hedging resolve when $L_0$ is increased (i.e., more latents are permitted to fire), whereas absorption-driven false negatives persist across $L_0$ values. This hedging-versus-competitive-exclusion distinction is central to our confound decomposition (Section 3.4) and constitutes the primary empirical finding of Section 4.

Tian et al. (2025) frame feature absorption as a special case of poor feature sensitivity: many features that appear monosemantic on their top activating examples nonetheless fail to activate on semantically similar inputs. This broader failure mode subsumes absorption and suggests that recall deficits in SAE features extend beyond hierarchical parent-child relationships.

## 2.3 Architectural Mitigations

The absorption finding has prompted a wave of architectural interventions, each proposing encoder modifications to reduce absorption while maintaining reconstruction quality:

- **Matryoshka SAE** (Bussmann et al., 2025; ICML 2025) trains nested dictionaries of increasing size simultaneously, organizing features hierarchically so that smaller dictionaries learn general concepts. SAEBench absorption score: 0.03 versus 0.29 for BatchTopK. Chanin et al. (2025) subsequently show that the inner levels trade absorption for hedging, with a compound multiplier of $\sim$0.75 achieving the best balance.
- **OrtSAE** (Korznikov et al., 2025) enforces orthogonality via a pairwise cosine penalty on decoder columns, reducing absorption by $\sim$65% relative to BatchTopK.
- **ATM-SAE** (Li et al., 2025) applies adaptive temporal masking with per-latent importance scoring, achieving an absorption score of 0.007 versus 0.140 for TopK and 0.011 for JumpReLU on Gemma 2 2B---the lowest reported absorption scores to date.
- **KronSAE** (2025) exploits Kronecker factorization to reduce parameter count and absorption fraction via structured correlation exploitation.
- **Masked regularization** (Narayanaswamy et al., 2026) disrupts co-occurrence patterns during training by randomly masking tokens, reducing absorption across SAE architectures.
- **MP-SAE** (Costa et al., 2025) uses iterative encoding to refine latent representations, partially recovering missed features.

All six mitigations benchmark against the Chanin absorption metric and assume it measures hierarchy-driven competitive exclusion. None validates the metric on JumpReLU SAEs, the architecture whose hard-threshold activation dynamics differ most from the L1-ReLU SAEs on which the metric's thresholds ($\tau_{\cos} \geq 0.025$, $\tau_{\text{mag}} \geq 1.0$) were calibrated. If the metric conflates hedging with competitive exclusion on JumpReLU architectures, the reported mitigation gains may reflect changes in false-negative rates rather than genuine reductions in hierarchy-driven absorption. Our metric audit (Section 4) tests this possibility directly.

## 2.4 Rate-Distortion Theory and Successive Refinement

The successive refinement theorem (Equitz and Cover, 1991) establishes that a source is successively refinable---encodable in stages without information loss---if and only if the stage descriptions form a Markov chain. Applied to SAE features: if $X \to f_{\text{child}} \to f_{\text{parent}}$ is Markov, the conditional mutual information (CMI) $I(X; f_{\text{parent}} \mid f_{\text{child}}) = 0$ and the parent carries no unique information about the input beyond what the child already encodes. Absorbing such a parent is information-theoretically lossless.

When CMI $> 0$, the parent encodes information about $X$ that the child does not capture. Absorbing the parent under sparsity pressure destroys this unique information---a quantifiable cost measured in bits. The magnitude of CMI thus provides a theoretical criterion for absorption susceptibility: features with low CMI are cheap to absorb (little information is lost), while features with high CMI resist absorption because suppressing them incurs a larger information cost.

Two technical choices constrain CMI estimation in the SAE context. First, $I(X; f_{\text{parent}} \mid f_{\text{child}})$ must be estimated in a finite-dimensional subspace of the decoder, since $d_{\text{model}} = 2{,}304$ for Gemma 2 2B renders direct estimation unreliable. We use a $k$-nearest-neighbor estimator (Kraskov et al., 2004) in a $d' = 10$ dimensional decoder subspace. Second, the estimate depends on probe quality: at $L_0 = 82$, only 10 of 25 first-letter probes pass the F1 $> 0.85$ quality gate, confounding the CMI--absorption relationship with probe accuracy. Section 6 reports this confound transparently and tests robustness via replication at $L_0 = 22$ where all 25 probes achieve F1 = 1.0.

No prior work has connected successive refinement theory to SAE feature absorption. This connection provides the theoretical basis for the rate-distortion diagnostic explored in Section 6, though as we report there, the diagnostic does not survive replication with perfect probes.

Section 3 describes the experimental methodology for the three-pronged study: metric audit (Q1), $L_0$ phase transition (Q2), and CMI diagnostic (Q3).

<!-- FIGURES
- None
-->
