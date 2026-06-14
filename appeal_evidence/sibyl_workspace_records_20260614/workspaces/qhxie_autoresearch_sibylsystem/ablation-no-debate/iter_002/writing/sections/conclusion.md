# 8. Conclusion

## 8.1 Summary of Findings

We decomposed feature absorption in sparse autoencoders through a controlled 2x2 factorial design and found that the encoder is the primary driver. The encoder effect ($E_{enc} = 0.843 \pm 0.082$) is approximately 80 times larger than the decoder effect ($E_{dec} = 0.011 \pm 0.015$), and the decoder effect is statistically indistinguishable from zero ($t = 0.71$, $p = 0.48$). This finding is robust across 5 random seeds ($t = 36.04$, $p = 3.85 \times 10^{-10}$), monotonic with parent-child cosine similarity (0.416 at cos = 0.5 to 0.544 at cos = 0.8, ANOVA $p < 10^{-10}$), and generalizes perfectly to unseen hierarchical patterns from the same distribution (Pearson $r = 0.998$, $p = 1.44 \times 10^{-4}$).

A surprising ablation reveals that lower sparsity increases absorption --- opposite to naive expectation. $L_0 = 20$ yields absorption of 0.552, while $L_0 = 50$ yields 0.419 (ANOVA $F = 4342.17$, $p < 10^{-10}$). We interpret this as a capacity-pressure mechanism: with fewer active features, the encoder overloads each feature with more concepts, and parent-child geometric alignment makes parents natural candidates for suppression.

We also report two important negative results. Steering interventions do not differentially affect absorbed features: the sensitivity ratio $s_{ratio} = 0.776$ ($p = 0.273$), indistinguishable from 1.0. Safety-critical features are not disproportionately absorbed compared to matched non-safety controls: 0.967 versus 0.968 (Mann-Whitney $p = 0.989$). Absorption is a universal geometric property, not an intervention target or a safety-specific pathology.

## 8.2 Implications

These findings reframe absorption from a controllable training artifact to a fundamental structural property of encoder learning on hierarchical data. The practical implication is clear: mitigation efforts should focus on the encoder, not the decoder. Decoder-side modifications such as orthogonality penalties (Korznikov et al., 2025) or nested dictionaries (Bussmann et al., 2025) address symptoms; encoder regularization that discourages parent-child co-activation addresses the cause.

The capacity-pressure finding complicates sparsity tuning. Practitioners face a trade-off between interpretability (lower $L_0$ produces sparser, more readable feature sets) and absorption fidelity (higher $L_0$ preserves parent features at the cost of denser representations). Simply increasing $L_0$ reduces absorption gradually --- a 150% increase in active features (from 20 to 50) reduces absorption by only 0.133 points --- so sparsity alone is not a lever for eliminating absorption.

The null result on safety specificity carries a positive implication for SAE-based safety analysis. If safety-critical features were disproportionately absorbed, practitioners would face a systematic blind spot: the features most relevant for monitoring dangerous behaviors would be the least likely to activate when those behaviors occur. The finding that absorption is uniform across semantic categories suggests that SAE-based safety monitoring is not systematically compromised, even though individual features may still fail to activate due to absorption in specific contexts.

## 8.3 Limitations and Future Work

Five limitations bound the scope of our conclusions.

**Synthetic data scope.** All factorial decomposition, multi-seed validation, and ablation experiments use synthetic data with $d_{model} = 128$. Real LLM residual streams operate at much higher dimensions (768 to 12,288) with more complex feature geometry. The H_Safe and held-out generalization experiments partially address this by testing on real GPT-2 SAEs and unseen synthetic hierarchies, but a full validation of encoder dominance on real-model SAEs at scale remains future work.

**Post-hoc criterion revision.** The original H_Mech pass criteria ($B \approx D$ and $C \approx A$) failed on 14 of 15 runs because Condition D consistently showed lower absorption than Condition B. We revised the criteria to encoder effect $> 0.5$ and decoder effect $< 0.1$ after observing the data. While these revised criteria directly test the paper's core claim, the revision was not pre-registered and should be interpreted as exploratory validation.

**Single architecture.** All experiments use TopK SAEs. JumpReLU, Matryoshka, and other architectures may exhibit different encoder-decoder dynamics. Marks et al. (2025) showed that the optimization landscape varies across sparsity mechanisms, and our findings may not transfer to architectures with fundamentally different activation functions.

**Metric inconsistency.** Three absorption metrics (cosine similarity, overlap fraction, Jaccard index) are used across experiments. While all capture the same conceptual phenomenon, their numerical scales differ and direct comparison requires caution.

**Same-distribution generalization.** The held-out test splits data from the same generative process. The near-perfect correlation ($r = 0.998$) demonstrates stability across samples but does not test generalization to new hierarchy geometries (e.g., training on cosine similarity 0.5 and testing on 0.8).

Future work should pursue four directions. First, replication on real language model SAEs at production scale (Gemma Scope, Claude SAEs) is essential to validate encoder dominance outside the synthetic regime. Second, encoder regularization experiments --- penalizing parent-child co-activation during training --- would directly test whether encoder-side intervention can reduce absorption. Third, cross-architecture comparison (TopK vs. JumpReLU vs. Matryoshka) would test whether encoder dominance generalizes across sparsity mechanisms. Fourth, out-of-distribution generalization --- training on one hierarchy geometry and testing on another --- would characterize the limits of absorption as a stable representational property.

<!-- FIGURES
- None
-->
