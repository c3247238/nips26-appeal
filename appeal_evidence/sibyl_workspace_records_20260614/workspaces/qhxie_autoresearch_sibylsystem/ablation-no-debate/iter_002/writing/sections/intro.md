# 1. Introduction

## 1.1 Problem Statement

Sparse autoencoders (SAEs) decompose transformer residual stream activations into sparse, interpretable feature representations via a bottleneck architecture: $x \approx W_{dec} \, \sigma(W_{enc} x + b_{enc}) + b_{dec}$. By learning an overcomplete dictionary of feature directions, SAEs promise to map the dense geometry of neural activations onto human-interpretable concepts---from syntactic constructs to semantic categories---enabling mechanistic interpretability at scale (Bricken et al., 2023; Templeton et al., 2024). Million-feature SAEs trained on production language models have recovered thousands of monosemantic features, establishing the empirical feasibility of this agenda.

A structural challenge undermines this promise: **feature absorption**. When a parent concept is present in the input, its dedicated feature may remain inactive because child features (subordinate concepts) capture the relevant signal and activate instead (Chanin et al., 2024). The parent feature is effectively absorbed by its children, making it an unreliable indicator of the parent concept's involvement in the model's computation. Chanin et al. proved that absorption is loss-minimizing under hierarchical data: when parent and child features are geometrically aligned, suppressing the parent and routing activation through children reduces the sparsity penalty without sacrificing reconstruction. Marks et al. (2025) formalized this through a unified theory of sparse dictionary learning, identifying spurious minima in the optimization landscape where hierarchical data induces absorbing partial minima.

Despite this theoretical progress, a fundamental mechanistic question remains unanswered: **which component of the SAE drives absorption?** The encoder $W_{enc}$ maps activations to sparse features; the decoder $W_{dec}$ reconstructs activations from features. Prior work treats the SAE as a monolithic optimization problem and does not empirically decompose absorption into encoder versus decoder contributions. This gap matters because mitigation strategies depend on where the problem originates. If the encoder drives absorption, encoder-side regularization is the natural intervention. If the decoder drives absorption, decoder-side modifications such as orthogonality penalties or nested dictionaries (Bussmann et al., 2025; Korznikov et al., 2025) address the right component.

## 1.2 The Mechanistic Gap

The prevailing implicit assumption is that absorption emerges from joint encoder-decoder optimization. Marks et al. (2025) analyze the SAE loss landscape as a whole; Cui et al. (2025) derive closed-form solutions for the full encoder-decoder system; architectural interventions modify both components simultaneously. Oursland (2026) proposed eliminating the decoder entirely, but this is a radical redesign rather than a decomposition.

No prior work isolates the encoder and decoder contributions. A 2x2 factorial design---crossing random versus trained encoder with random versus trained decoder---can answer this question directly. If the trained encoder alone produces absorption comparable to full training, while the trained decoder alone produces baseline-level absorption, then absorption is encoder-driven. If both components contribute, the decomposition quantifies their relative magnitudes.

## 1.3 Research Questions

This paper addresses five questions through controlled experiments on synthetic hierarchical data with ground-truth parent-child relationships:

1. **Does the encoder or the decoder drive feature absorption?** We decompose absorption via a 2x2 factorial design crossing encoder state (random/trained) with decoder state (random/trained).

2. **Is absorption robust across random seeds and hierarchy strengths?** We validate across 5 seeds with stochastic hierarchy generation and vary parent-child cosine similarity across $\{0.5, 0.67, 0.8\}$.

3. **Does absorption generalize to unseen hierarchical patterns?** We test whether absorption rates on training hierarchies predict absorption on held-out hierarchies from the same generative distribution.

4. **Can absorbed features be exploited for steering interventions?** We test whether absorbed features show differential sensitivity to parent-direction steering compared to non-absorbed features.

5. **Are safety-critical features disproportionately absorbed?** We compare absorption rates between safety-relevant and matched non-safety features in real GPT-2 SAEs.

## 1.4 Contributions

Our experiments yield six findings:

1. **Encoder-driven absorption**: A 2x2 factorial decomposition reveals that the encoder effect ($E_{enc} = 0.843 \pm 0.082$) is approximately 80 times larger than the decoder effect ($E_{dec} = 0.011 \pm 0.015$). The decoder effect is statistically indistinguishable from zero ($t = 0.71$, $p = 0.48$).

2. **Robustness across seeds**: Trained SAEs show consistently high absorption (0.477 $\pm$ 0.022) versus random baselines (0.033 $\pm$ 0.011) across 5 seeds ($t = 36.04$, $p = 3.85 \times 10^{-10}$).

3. **Dose-response with hierarchy strength**: Absorption increases monotonically with parent-child cosine similarity: 0.416 at cos = 0.5, 0.501 at cos = 0.67, 0.544 at cos = 0.8 (ANOVA $F = 4718.81$, $p < 10^{-10}$).

4. **Capacity-pressure effect**: Lower sparsity ($L_0 = 20$) produces higher absorption (0.552) than higher sparsity ($L_0 = 50$, 0.419), opposite to naive expectation. We interpret this as a capacity-pressure mechanism: with fewer active features, the encoder overloads each feature with more concepts.

5. **Negative result on steering**: Absorbed features do not show differential sensitivity to parent-direction steering (sensitivity ratio $s_{ratio} = 0.776$, $p = 0.273$). Absorption is a representational property, not an intervention target.

6. **Negative result on safety specificity**: Safety-critical features and matched non-safety controls exhibit statistically indistinguishable absorption rates (0.967 vs. 0.968, Mann-Whitney $p = 0.989$). Absorption is universal across semantic categories.

These findings reframe absorption from a joint encoder-decoder artifact to a fundamental structural property of encoder learning on hierarchical data, guiding future work toward encoder-side mitigation strategies.

<!-- FIGURES
- None
-->
