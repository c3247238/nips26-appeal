# 1. Introduction

Sparse Autoencoders (SAEs) decompose neural network activations into sparse combinations of learned features, enabling researchers to inspect and steer individual computational units inside language models (Bricken et al., 2023; Cunningham et al., 2023). SAEBench confirms that dictionary learning produces interpretable features across hundreds of SAEs, multiple models, and architectures (Karvonen et al., 2025). Yet one failure mode persistently undermines SAE reliability: **feature absorption**.

Feature absorption occurs when a parent latent --- one encoding a general concept --- is suppressed to zero on inputs where a more specific child latent is active, even though the parent concept is present (Chanin et al., 2024). A concrete example (Figure 1a): a latent trained to detect "starts-with-A words" (parent) fails to fire whenever a "starts-with-A proper nouns" latent (child) is active. The child's activation absorbs the parent's reconstruction contribution. This suppression is a systematic consequence of sparsity pressure: the biconvex sparse dictionary learning (SDL) loss rewards solutions where one latent handles a subset of inputs that two latents previously shared (Tang et al., 2025). The practical consequence is severe --- any feature-based causal analysis or steering experiment that reads parent latent activations will silently produce incorrect conclusions on inputs where a child latent is active.

Despite sustained attention, including architectural responses such as OrtSAE, Matryoshka SAE, KronSAE, and masked regularization (Narayanaswamy et al., 2026; Costa et al., 2025), three gaps remain open.

**Gap 1 --- Detection requires foreknowledge.** The Chanin et al. metric requires pre-specified probe directions: one must already know which parent-child hierarchy to test. Auditing all $d_{\text{SAE}}$ latents in a deployed SAE is intractable under this regime.

**Gap 2 --- Generalizability is assumed, not tested.** Every published absorption measurement uses the first-letter spelling task. Whether absorption extends to entity types, safety-relevant concepts, or other semantic hierarchies has never been empirically tested.

**Gap 3 --- No actionable taxonomy.** All absorbed latents are treated as a single category. A latent whose parent concept was never allocated in the dictionary (a training-time coverage failure) cannot be fixed by the same means as a latent whose parent decoder direction exists but whose encoder was trained away from it (an inference-time encoder suppression).

This paper addresses all three gaps with a unified, training-free framework. Our contributions:

1. **Encoder-Decoder Alignment (EDA)**, the first weight-only absorption screening metric. EDA measures the angular mismatch between each latent's encoder direction $w_{e,j}$ and decoder direction $d_j$ (Figure 1b). We derive a formal lower bound from biconvex optimization theory (Theorem 1): for a SAE at a partial minimum with $\delta$-absorption, $\text{EDA}(j) \geq \delta^2 \sin^2(\theta_{jc}) / (2 + \delta^2)$. EDA achieves AUROC = 0.776 at L12-16k (Gemma Scope), AUROC = 0.629 at GPT2-L6 with exact Chanin et al. labels, and outperforms the decoder cosine baseline by +0.396 AUROC (DeLong $p \approx 0$). EDA requires only SAE weight matrices --- no activation data.

2. **First cross-domain absorption characterization.** We extend absorption measurement to three RAVEL entity-attribute hierarchies: city-continent, city-country, and city-language. All 18 SAE-hierarchy measurements (6 configs $\times$ 3 hierarchies) exceed the $3\times$ random baseline; intra-RAVEL Spearman $\rho$ = 0.924 confirms coherent absorption rankings. Absorption is not a first-letter artifact.

3. **Three-subtype absorption taxonomy.** We classify absorbed latents into early absorption (parent decoder direction absent; ~72--75%), late absorption (parent direction present, encoder suppressed; ~13%), and partial absorption (context-dependent failure; ~13%). The dominant failure mode is dictionary-coverage failure at training time, not encoder suppression at inference time --- redirecting remediation strategy from encoder correction toward dictionary allocation. The late > early EDA ordering is robust across all five tested thresholds (Kruskal-Wallis $p$ = 0.0002 at L12-65k).

We additionally report Inference-Time Absorption Correction (ITAC), a proof-of-concept that achieves 3.14% mean false-negative reduction for late-type latents (best case: 22.7%) without degrading reconstruction quality (FVU change: $-4.23$%), and a negative scaling result: wider SAEs absorb more at any $L_0$ (H6 falsified).

Section 2 establishes the theoretical framework underlying EDA, connecting encoder-decoder geometry to absorption degree through biconvex optimization theory. Section 3 derives EDA and its formal lower bound. Section 4 validates EDA on Gemma Scope and GPT-2 SAEs. Section 5 presents the cross-domain anatomy. Section 6 introduces the taxonomy. Section 7 consolidates the findings and their implications.

<!-- FIGURES
- Figure 1: fig1_absorption_mechanism_desc.md — Two-panel conceptual diagram: (a) feature absorption mechanism with parent/child latent suppression example, (b) EDA geometry showing encoder-decoder angular mismatch for healthy vs. absorbed latents
- None (no code-generated figures in this section)
-->
