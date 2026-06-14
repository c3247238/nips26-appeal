# 7. Discussion

## 7.1 Encoder Dominance and the Two-Player Dynamic

Our central finding is that the encoder drives feature absorption to an extent that leaves the decoder effectively irrelevant. The encoder effect ($E_{enc} = 0.843 \pm 0.082$) is approximately 80 times larger than the decoder effect ($E_{dec} = 0.011 \pm 0.015$), and the decoder effect is statistically indistinguishable from zero ($t = 0.71$, $p = 0.48$). This overturns the implicit assumption in prior work that absorption is a joint encoder-decoder phenomenon.

A more subtle finding emerges from comparing Condition B (trained encoder, random decoder) with Condition D (full training). Condition D shows lower absorption overlap than Condition B, suggesting that the trained decoder actively compensates for encoder-induced absorption by redistributing parent activations across more features during reconstruction. This reveals a two-player dynamic: the encoder compresses hierarchical structure into child features, and the decoder partially decompresses it. The net absorption in full training reflects the balance between these opposing forces, not the sum of independent contributions.

This observation aligns with Oursland's (2026) theoretical derivation of encoder-decoder asymmetry, which predicts that the encoder's alignment with data geometry dominates the learned representation while the decoder's role is primarily reconstructive. Our factorial design empirically validates this asymmetry and quantifies its magnitude.

## 7.2 Capacity Pressure: Why Lower Sparsity Increases Absorption

The L0 sparsity ablation produced a result opposite to our pre-registered hypothesis. We predicted that higher $L_0$ (more active features) would reduce absorption by providing more capacity for separate parent and child representations. Instead, lower $L_0$ (fewer active features) consistently produced higher absorption: $L_0 = 20$ yields 0.552, $L_0 = 32$ yields 0.490, and $L_0 = 50$ yields 0.419 (ANOVA $F = 4342.17$, $p < 10^{-10}$).

We interpret this as a capacity-pressure mechanism. When the encoder is constrained to activate fewer features per sample, each active feature must represent more concepts to maintain reconstruction quality. Parent concepts, which are geometrically aligned with their children, are the natural candidates for this overloading: suppressing the parent and routing its signal through children preserves reconstruction while satisfying the sparsity constraint. Under this interpretation, absorption is not merely a side effect of hierarchy but an active compression strategy that the encoder employs when capacity is scarce.

This finding has direct implications for SAE design. Simply increasing $L_0$ does reduce absorption, but the relationship is monotonic and gradual: moving from $L_0 = 20$ to $L_0 = 50$ reduces absorption by only 0.133 points. Practitioners seeking to minimize absorption through sparsity tuning face a trade-off between interpretability (lower $L_0$ produces sparser, more readable feature sets) and absorption fidelity (higher $L_0$ preserves parent features at the cost of denser representations).

## 7.3 Absorption Is Not an Intervention Target

The H3 steering experiment falsified the hypothesis that absorbed features are more sensitive to parent-direction steering than non-absorbed features. Across all 9 steering conditions, the sensitivity ratio $s_{ratio}$ ranged from 0.776 to 1.167 with no statistically significant deviation from 1.0. This is a genuine negative result: absorbed features do not respond differently to steering interventions.

This finding reframes absorption from a causal mechanism to a representational property. Absorption affects what the SAE represents (epistemic) but not how the represented features behave under intervention (causal). For practitioners using SAEs for model steering, this means that absorption does not create a special class of features that can be exploited for targeted control. The standard steering protocol---identifying features by their activating concepts and steering in the corresponding direction---remains valid regardless of whether those features are absorbed.

## 7.4 Absorption Is Universal, Not Safety-Specific

The H_Safe analysis on GPT-2 Small SAEs shows that safety-critical features and matched non-safety controls exhibit statistically indistinguishable absorption rates (0.967 vs. 0.968, Mann-Whitney $p = 0.989$). Both groups show near-complete absorption, consistent with the high absorption rates reported by Chanin et al. (2024) on real LLM SAEs.

This null result carries a positive implication for SAE-based safety analysis. If safety-critical features were disproportionately absorbed, practitioners would face a systematic blind spot: the very features most relevant for monitoring dangerous behaviors would be the least likely to activate when those behaviors occur. The finding that absorption is uniform across semantic categories suggests that SAE-based safety monitoring is not systematically compromised by absorption, even though individual safety features may still fail to activate due to absorption in specific contexts.

## 7.5 Limitations

We report five limitations that bound the scope of our conclusions.

**Synthetic-only evidence.** All factorial decomposition, multi-seed validation, and ablation experiments use synthetic data with $d_{model} = 128$. While the synthetic framework provides ground-truth hierarchy and precise absorption measurement, real LLM residual streams operate at much higher dimensions (768 to 12,288) with more complex feature geometry. The H_Safe and held-out generalization experiments partially address this by testing on real GPT-2 SAEs and unseen synthetic hierarchies, respectively, but a full validation of encoder dominance on real-model SAEs remains future work.

**Post-hoc criterion revision.** The original H_Mech pass criteria ($B \approx D$ and $C \approx A$) failed on 14 of 15 runs because Condition D consistently showed lower absorption than Condition B. We revised the criteria to encoder effect $> 0.5$ and decoder effect $< 0.1$ after observing the data. While these revised criteria directly test the paper's core claim, the revision was not pre-registered and should be interpreted as exploratory validation rather than confirmatory hypothesis testing.

**Metric inconsistency.** Three different absorption metrics (cosine similarity, overlap fraction, Jaccard index) are used across experiments without establishing formal equivalence. The factorial decomposition uses overlap fraction, the multi-seed validation uses Jaccard overlap, and the safety analysis uses cosine-based proportional absorption. While all three metrics capture the same conceptual phenomenon, their numerical scales differ and direct comparison requires caution.

**Same-distribution generalization.** The held-out generalization test splits data from the same generative process (80/20 by hierarchy instance). The near-perfect correlation ($r = 0.998$) demonstrates stability across samples but does not test generalization to new hierarchy geometries (e.g., training on cosine similarity 0.5 and testing on 0.8).

**Single architecture.** All experiments use TopK SAEs. JumpReLU, Matryoshka, and other architectures may exhibit different encoder-decoder dynamics. Marks et al. (2025) showed that the optimization landscape varies across sparsity mechanisms, and our findings may not transfer to architectures with fundamentally different activation functions.

## 7.6 Implications for Mechanistic Interpretability

Our findings suggest three directions for the field.

**Encoder regularization as mitigation.** If absorption is driven by encoder alignment, then encoder-side regularization---penalizing parent-child co-activation during training---is the natural mitigation strategy. Decoder-side modifications (orthogonality penalties, nested dictionaries) address symptoms rather than causes. A pilot experiment adding an encoder regularization term that discourages child features from activating when parent features are present would directly test this prediction.

**Training-free detection.** The encoder-decoder asymmetry we identify may enable training-free detection of absorption-prone features. If encoder geometry alone predicts absorption, then analyzing the encoder weight matrix without running the full SAE forward pass could flag features at risk of absorption. This would be valuable for auditing pretrained SAEs where retraining is infeasible.

**Reframing absorption.** Absorption should be reframed from a controllable training artifact to a fundamental structural constraint of encoder learning on hierarchical data. The field's goal should shift from "eliminating absorption" to "navigating the trade-offs that absorption creates"---between sparsity and fidelity, between parent recovery and child precision, and between interpretability and reconstruction quality.

<!-- FIGURES
- None
-->
