# 5. Discussion

## 5.1 TopK Sparsity as the Operative Mechanism

The 78.0% absorption reduction from TopK sparsity (Cohen's $d = 5.51$) far exceeds any other tested component. To understand why TopK dominates, we examine the mechanism.

Absorption occurs when parent and child features co-occur and the SAE's sparsity constraint forces the parent to be suppressed in favor of the children. The Baseline SAE uses an L1 sparsity penalty, which encourages but does not enforce sparsity. The resulting $L_0 = 964$ active latents per sample (out of $m = 2048$) means nearly half the dictionary co-activates, creating ample opportunity for parent-child competition. TopK, by contrast, enforces a hard limit of $k = 50$ active latents. With only 50 slots, the SAE must be selective about which features to represent, and the explicit constraint prevents the dense co-activation that enables absorption.

The positive absorption--L0 correlation ($r \approx +0.93$ across $n = 4$ variants, $p = 0.067$) is consistent with this interpretation. Baseline ($L_0 = 964$, $A = 0.252$), TopK ($L_0 = 50$, $A = 0.056$), and Orthogonality ($L_0 = 550$, $A = 0.245$) all fall on the same line. This suggests that absorption may be primarily a function of sparsity level, not architectural novelty: more active latents mean more opportunities for parent-child competition. TopK achieves low absorption not because top-k activation is architecturally special, but because it enforces extreme sparsity.

If this correlation is confirmed by a dedicated sparsity-sweep experiment (varying $k$ within a single architecture), it would suggest reframing the research question. Instead of asking "which architecture reduces absorption?" the field should ask: (1) what is the causal mechanism linking sparsity to absorption? (2) what is the optimal sparsity level for minimizing absorption while maintaining reconstruction? and (3) do other sparsity-control mechanisms (e.g., adaptive TopK, learned thresholds) achieve similar effects?

## 5.2 Why Orthogonality Fails

The orthogonality penalty achieves near-perfect reconstruction (MSE $\approx 3 \times 10^{-5}$) but negligible absorption reduction (2.7%, $d = 0.14$). This is a striking dissociation: the component that most improves reconstruction does nothing for absorption.

The explanation lies in the operative constraint. Absorption is driven by the activation sparsity pattern---how many latents co-fire when parent and child features are present---not by the geometric coherence of decoder directions. The orthogonality penalty affects decoder geometry (encouraging $W_{\text{dec}}^\top W_{\text{dec}} \approx I$), but it does not change the fact that 550 latents co-activate per sample. With 550 active latents, parent-child competition still occurs; the decoder directions are simply more orthogonal.

This finding has practical implications. Korznikov et al. (2025) reported 65% absorption reduction from orthogonality penalties, but their measurement used the SAEBench probe-based protocol on real LLMs. Our ground-truth synthetic result suggests that the orthogonality effect may be mediated by probe geometry rather than genuine absorption reduction. The orthogonality penalty may improve probe separability without changing the underlying activation pattern that drives absorption.

## 5.3 The Missing Trade-off

Chanin et al. (2025) demonstrated that absorption and hedging sit on a Pareto frontier: architectures that reduce absorption increase hedging, and vice versa. Matryoshka SAEs, for example, reduce absorption but increase hedging in inner dictionary levels.

Our data do not replicate this trade-off. Hedging scores are invariant across all tested variants (~0.24), with no statistically detectable differences. We see three possible explanations:

1. **Synthetic data lacks correlation structure.** Real LLM features exhibit rich correlational structure beyond hierarchical containment. Synthetic features are generated from a tree-structured co-occurrence model that may not produce the correlations that trigger hedging.

2. **Hedging emerges at different scales.** The pilot used 1,024 features; the full experiment uses 16,384. Hedging may require larger dictionaries or different feature densities to manifest.

3. **The hedging metric is insensitive.** Our hedging score follows Chanin et al. (2025), but alternative formulations might detect differences our metric misses.

We acknowledge this as a limitation. The absence of a trade-off does not contradict Chanin et al. (2025); it merely indicates that our synthetic setting does not capture the phenomenon. Future work should test hedging on real LLM features or more complex synthetic correlation structures.

## 5.4 Implications for Architecture Design

Our findings yield three concrete recommendations for practitioners and architecture researchers.

**First, prioritize explicit sparsity control.** If absorption reduction is the goal, TopK sparsity is the most effective single component tested, with an effect size roughly five times larger than multi-scale decomposition and roughly forty times larger than orthogonality. The implementation is simple (replace L1 penalty with top-k activation) and the computational overhead is minimal.

**Second, multi-scale decomposition may still help, but its effect is unconfirmed.** The pilot MultiScale result (75.3% reduction, $d \approx 1.1$) is promising, but it comes from a pilot (1,024 features, 1 replicate) that is not directly comparable to the full experiment (16,384 features, 5 replicates). The full experiment is needed before any ranking can be established. Even if confirmed, the effect would be secondary to TopK. Multi-scale dictionaries may offer other benefits (feature organization, interpretability) that justify their complexity, but absorption reduction is not the primary one.

**Third, orthogonality penalties add compute overhead without absorption benefit.** The 4--11% compute overhead reported by Korznikov et al. (2025) is not justified by absorption reduction in our ground-truth test. Orthogonality may still improve reconstruction or probe separability, but practitioners should not expect it to reduce absorption.

## 5.5 Limitations

Our study has five principal limitations.

**Incomplete variant set.** Only 3 of 6 variants have full 5-replicate data (Baseline, TopK, Orthogonality). MultiScale, Gating, and Full Matryoshka are represented by pilot data only. The component ranking may change when all variants are completed.

**Synthetic data may not match real LLM feature structure.** SynthSAEBench-16k uses binary features with tree-structured co-occurrence. Real LLM features are continuous, context-dependent, and embedded in a residual stream with nonlinear interactions. The component ranking on synthetic data may not transfer to real LLMs.

**MCC metric fails to discriminate.** All variants show MCC ~0.21--0.22, including the Random control. This is expected behavior for Hungarian matching on overcomplete dictionaries, but it means we cannot use MCC to validate feature recovery quality.

**Single sparsity level for TopK.** We test $k = 50$ only. The optimal $k$ for absorption-reconstruction trade-offs is unknown and may vary by dataset and model size.

**Hedging invariance may reflect synthetic data limitations.** As discussed in Section 5.3, the absence of a hedging trade-off may be a property of our synthetic setting rather than a genuine finding about architecture.

## 5.6 Future Work

Four directions would strengthen and extend this study.

**Complete remaining variants.** Full 5-replicate experiments for MultiScale, Gating, and Full Matryoshka would enable definitive component ranking and ANOVA across all six conditions.

**Real-LLM validation.** Training the top-performing component (TopK) on Pythia-160M or Gemma-2-2B activations and measuring first-letter absorption via SAEBench would test synthetic-to-real transfer. If the component ranking replicates, the synthetic benchmark is validated as a proxy.

**Sparsity sweep.** Varying $k \in \{10, 25, 50, 100, 200\}$ for TopK SAEs would characterize the absorption-sparsity relationship and identify the optimal sparsity level.

**Rate-distortion theoretical framing.** Absorption can be framed as an information-theoretic necessity: given a rate constraint (sparsity level), some parent-feature information must be lost. Multi-scale dictionaries effectively expand the rate budget by allowing features to be represented at multiple resolutions. A formal rate-distortion analysis could derive bounds on absorption as a function of sparsity and hierarchy depth.

From these implications, we distill the key takeaways.

<!-- FIGURES
- Figure 2: (referenced from Results) — Absorption rate by variant
- Figure 3: (referenced from Results) — Absorption vs. L0 sparsity
- Figure 4: (referenced from Results) — Effect sizes
- Figure 5: (referenced from Results) — Pareto frontier
- None (no new figures generated in this section)
-->
