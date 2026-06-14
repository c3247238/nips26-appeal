# 2. Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders decompose polysemantic LLM activations into an overcomplete sparse basis of approximately monosemantic directions. Given an input activation $\mathbf{x} \in \mathbb{R}^d$, an SAE computes latent activations $\mathbf{z} \in \mathbb{R}^m$ (where $m \gg d$) and reconstructs $\hat{\mathbf{x}} = W_d \mathbf{z} + \mathbf{b}_d$. The sparsity objective ensures that $L_0$ (the expected number of active latents per input) is small relative to $m$, so that each active latent can be interpreted as a single semantic direction.

The architecture family has diversified rapidly. Bricken et al. (2023) demonstrated the first large-scale monosemantic decomposition on a 1-layer transformer using L1-penalized SAEs. Gao et al. (2024) proposed TopK SAEs that select exactly $k$ highest-activating latents per input, yielding clean scaling laws on GPT-4 with up to 16M latents. Rajamanoharan et al. (2024a) introduced Gated SAEs, which decouple feature detection from magnitude estimation to reduce L1-induced shrinkage bias. Rajamanoharan et al. (2024b) proposed JumpReLU SAEs, which train $L_0$ directly via the straight-through estimator and achieve state-of-the-art reconstruction fidelity on Gemma 2 9B. Google DeepMind released Gemma Scope (Lieberum et al., 2024), comprising 400+ open-source JumpReLU SAEs across all layers of Gemma 2 2B, 9B, and 27B with dictionary widths from 1k to 1M -- the primary evaluation target for absorption research.

Evaluation has standardized around SAEBench (Karvonen et al., 2025), which scores 200+ open-source SAEs on eight metrics: sparse probing, RAVEL disentanglement (TPP), spurious correlation removal (SCR), unlearning, and absorption, among others. A key SAEBench finding motivates our work: proxy metrics such as CE loss recovered and $L_0$ do not reliably predict practical downstream performance (Karvonen et al., 2025, Table 3), raising the question of whether absorption -- a structural failure mode rather than a proxy metric -- is a better quality predictor.

## 2.2 Feature Absorption: Definition and Prior Measurements

Chanin et al. (2024) formalized feature absorption. The phenomenon occurs when a parent SAE latent $j$ (e.g., "starts with S") fails to fire on tokens where a more specific child latent $c$ (e.g., "September") activates instead, because the SAE's sparsity objective favors the child's single-latent encoding over the parent-plus-child two-latent encoding. Their measurement protocol trains linear probes to identify ground-truth feature directions, uses $k$-sparse probing to find feature splits, identifies false-negative tokens (where all split latents for a feature fail to fire but the probe correctly classifies the input), and applies integrated-gradients ablation to attribute false negatives to specific absorbing latents. Absorption is detected when the highest-ablation-effect latent has cosine similarity exceeding $\tau_{\text{cos}} = 0.025$ with the parent probe direction and dominance ratio exceeding $\tau_{\text{dom}} = 1.0$ over the second-highest.

On the first-letter spelling task across mid-layer Gemma Scope SAEs, Chanin et al. report absorption rates of 15--35%, with the rate increasing at wider dictionaries and lower $L_0$. Their toy model proves that absorption is loss-optimal when the sparsity penalty exceeds $\sin^2(\theta_{p,c})$, where $\theta_{p,c}$ is the decoder angle between parent and child directions. Absorption appears in every tested SAE architecture (L1, TopK, JumpReLU) and across multiple model families (Gemma 2, Llama 3.2, Qwen2).

Tian et al. (2025) frame absorption as a special case of poor feature *sensitivity*: a latent that activates selectively on its target concept but fails on similar inputs has low sensitivity. Their scalable evaluation reveals that many features rated as interpretable by activation-example inspection have poor recall, consistent with the partial absorption phenomenon.

Two properties of the existing measurement base limit the field's understanding. First, all absorption measurements use a single evaluation task -- the first-letter spelling task -- a deterministic hierarchy with an unusually sharp parent-child structure (each token has exactly one first letter). Whether absorption rates generalize to fuzzier, semantically richer hierarchies (knowledge taxonomies, safety-relevant features) is unknown, a limitation explicitly noted by Chanin et al. and at least four subsequent papers (Karvonen et al., 2025; Korznikov et al., 2025; Li et al., 2025; SynthSAEBench, 2026). Second, no study has tested whether absorption scores predict downstream SAE quality after controlling for confounds, leaving the assumed causal chain (less absorption implies better interpretability) empirically unvalidated.

## 2.3 Mitigation Approaches

Multiple architectural interventions reduce absorption, though no unified account explains why they work.

Bussmann et al. (2025) proposed Matryoshka SAEs, which train nested dictionaries of increasing width simultaneously. The nested structure allocates general features to smaller inner dictionaries and specific features to larger outer ones, directly reducing the parent-child competition that drives absorption. Matryoshka SAEs achieve the best absorption scores in SAEBench while maintaining competitive reconstruction, though inner dictionary levels suffer from feature hedging (Chanin et al., 2025).

Korznikov et al. (2025) enforce pairwise orthogonality on decoder columns via a cosine similarity penalty (OrtSAE), reducing absorption by 65% relative to standard SAEs with linear computational overhead. Orthogonality decreases $\cos(\mathbf{p}, \mathbf{d}_j)$ between parent probe directions and child decoder directions, directly targeting the geometric condition in the Chanin et al. toy model.

Li et al. (2025) introduced Adaptive Temporal Masking (ATM), which dynamically scores per-latent importance based on activation magnitude, frequency, and reconstruction contribution. ATM achieves the lowest reported absorption score: 0.0068 on Gemma 2 2B, compared to 0.1402 for TopK and 0.0114 for JumpReLU.

Narayanaswamy et al. (2026) proposed masked regularization, which randomly masks high-frequency tokens during SAE training to disrupt the co-occurrence patterns that enable absorption. The approach improves out-of-distribution robustness across SAE architectures.

A unified theoretical framework (arXiv:2512.05534) casts all sparse dictionary learning methods as a piecewise biconvex optimization problem and identifies stable partial minima where absorbed features are trapped. The proposed remedy, feature anchoring, restores identifiability in synthetic benchmarks but has not been validated for absorption reduction on real LLM SAEs.

These diverse mechanisms -- nesting, orthogonality, temporal masking, token masking, anchoring -- all reduce absorption but operate through different pathways. Our scaling surface analysis (Section 3.3) provides complementary evidence by mapping which regions of the (width, $L_0$) hyperparameter space exhibit high absorption, independent of architecture choice.

## 2.4 The Unresolved Confound Problem

The most consequential open question about absorption is whether it is a genuine causal driver of SAE quality degradation or an epiphenomenon of correlated hyperparameters.

Chanin et al. (2024) report that absorption correlates with downstream quality ($r = -0.595$ across 54 Gemma Scope SAEs), but their analysis does not control for $L_0$. In the Gemma Scope collection, all high-absorption SAEs are 1M width with low $L_0$ (16--58), while all low-absorption SAEs are 16k or 65k width with high $L_0$ (137--297). Prior partial correlations controlled for log(width) and layer but not $L_0$, leaving open the possibility that absorption is simply a proxy for sparsity level -- in which case the entire mitigation research program is misdirected.

The confound is sharpened by Chanin and Garriga-Alonso (2025), who show that incorrect $L_0$ (which is common in open-source SAEs) causes feature hedging, a distinct failure mode where correlated features are merged into a single latent. Feature hedging and absorption both manifest as features failing to fire where they should, but they have different causes: hedging arises from insufficient dictionary capacity, while absorption arises from hierarchical feature structure. No study has systematically disentangled the two in observational SAE data.

DeepMind's safety research team publicly deprioritized SAE research in 2025 after finding that dense linear probes dramatically outperform 1-sparse SAE probes on harmful intent detection. Their report identifies feature absorption as a key culprit but provides no systematic quantification of the absorption-performance link.

No prior work applies formal causal inference methods to SAE evaluation. The SAE community relies on bivariate or partially controlled correlations between architectures, without the mediation analysis, propensity matching, or sensitivity analysis that are standard in epidemiological and social science research for establishing causal claims from observational data. Our Phase 1 analysis (Section 3.1) introduces these methods to SAE evaluation: partial correlation with $L_0$ control, Baron-Kenny mediation analysis, and Rosenbaum sensitivity bounds. Our Phase 2 (Section 3.2) tests cross-domain generalizability on knowledge hierarchies. Our Phase 3 (Section 3.3) constructs the first joint (width, $L_0$) absorption scaling surface with formal interaction testing across 420 SAEs.

<!-- FIGURES
- None
-->
