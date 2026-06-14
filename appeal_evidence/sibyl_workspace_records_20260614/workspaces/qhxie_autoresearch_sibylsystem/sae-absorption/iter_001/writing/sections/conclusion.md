# 8. Conclusion

Three contributions address three open gaps in understanding feature absorption in Sparse Autoencoders.

**A weight-only screening metric.** Encoder-Decoder Alignment ($\text{EDA}(j) = 1 - \cos(w_{e,j}, d_j)$) detects absorption from SAE weight matrices alone. Theorem 1 establishes a formal lower bound: EDA increases monotonically with absorption degree $\delta$. Empirical validation confirms regime-specific reliability --- AUROC = 0.776 at L12-16k (Gemma Scope), 0.629 at GPT2-L6 (exact Chanin et al. labels), and +0.396 AUROC over the decoder cosine baseline (DeLong $p \approx 0$). Within polysemantic latent regions, where absorption concentrates, EDA reaches AUROC = 0.922. The metric is not universal: it is strongest at mid-layers in 16k-width SAEs and degrades at deeper layers and wider dictionaries, a regime dependency that itself reveals where absorption's geometric signature is strongest.

**Absorption generalizes beyond first-letter.** All 18 RAVEL entity-attribute measurements (city-continent, city-country, city-language $\times$ 6 SAE configs) exceed the 3$\times$ random baseline. Intra-RAVEL Spearman $\rho$ = 0.924 confirms coherent absorption rankings across hierarchy types. Absorption is not a first-letter artifact --- it occurs wherever hierarchical parent-child feature structure exists in the residual stream.

**Early-absorption dominance reframes remediation strategy.** The three-subtype taxonomy --- early (decoder-absent), late (encoder-suppressed), partial (context-dependent) --- reveals that 72--75% of absorbed latents are early-type (Kruskal-Wallis $p$ = 0.0002 at L12-65k). These latents correspond to parent features the SAE dictionary never allocated. The dominant failure mode is a training-time dictionary coverage gap, not an inference-time encoder misalignment. ITAC confirms this framing: it achieves only 3.14% mean FN reduction because it is structurally limited to the ~13% late-type minority. Wider dictionaries do not compensate either (H6 falsified: partial $\rho$(width, absorption | $L_0$) = +0.37). The practical prescription is direct: architectural innovations that improve dictionary coverage --- Matryoshka SAE, targeted regularization, hierarchically-aware training objectives --- address the root cause for three-quarters of absorbed latents.

Future work should prioritize replicating the cross-domain analysis with probes trained directly on Gemma 2 2B, developing dictionary-coverage-aware training objectives, and extending EDA validation to alternative SAE architectures (OrtSAE, MP-SAE). The EDA metric, three-subtype taxonomy, and cross-domain evaluation code are released as an open SAEBench extension.

<!-- FIGURES
- None
-->
