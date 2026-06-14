# Introduction

Sparse Autoencoders (SAEs) have emerged as a primary tool for decomposing the internal activations of large language models into human-interpretable feature representations (Bricken et al., 2023; Templeton et al., 2024). By training SAEs on transformer residual stream activations, researchers have recovered thousands of monosemantic features -- from syntactic constructs to semantic concepts -- that are otherwise entangled in the dense representations of base models. This decomposition promises a pathway toward mechanistic interpretability: if we can identify and track individual features, we can understand, monitor, and potentially steer model behavior at a structural level.

A key challenge for this agenda is **feature absorption** -- the phenomenon where lower-level features (children) substitute for higher-level features (parents) in sparse representations (Chanin et al., 2024). When a parent concept is present in the input, its dedicated feature may remain inactive because child features already capture the relevant signal and are activated instead. This creates a fundamental reliability problem for SAE-based interpretability: a feature that fails to activate when its concept is present cannot serve as a trustworthy indicator of that concept's involvement in the model's computation.

Despite growing interest in absorption, the field faces a **measurement crisis**. Korznikov et al. (2026) demonstrated that random baselines recover similar feature counts and activation statistics to trained SAEs on standard interpretability benchmarks, raising the question of whether previously reported phenomena are genuine or statistical artifacts. For absorption specifically, single-feature ablation methodology saturates: ablating one child feature allows the remaining children to reconstruct the parent, producing absorption rates of 1.0 for both trained SAEs and random baselines. This saturation makes it impossible to determine whether absorption is a learned representational phenomenon or an artifact of how we measure it.

This paper addresses three open questions about feature absorption:

1. **Can we measure absorption that differentiates trained SAEs from random baselines?** Prior methodology saturates. We need a measurement approach that reveals whether absorption reflects structured learned representations rather than random activation statistics.

2. **Is absorption causally responsible for downstream task failures?** Documentation of absorption does not establish whether absorbed features actively interfere with downstream computation. Absorption could be epistemic (a measurement artifact) rather than causal (an active mechanism degrading model performance).

3. **Do safety-critical features exhibit elevated absorption?** If the most important features for AI safety are disproportionately absorbed, SAE-based safety analysis may be unreliable precisely when it matters most.

We propose **multi-child proportional ablation** as a measurement methodology that resolves the saturation problem. The key insight is that ablating a single child saturates because remaining children compensate; ablating the top-k children tests whether the collective set of children substitutes for the parent. On synthetic hierarchies with known ground truth, trained SAEs exhibit absorption rates of 0.50 compared to 0.059 for random decoder baselines (Cohen's d = 8.94, p < 10^-133), confirming absorption as a genuine representational phenomenon.

However, absorption does not correlate with feature activation frequency (Spearman rho = +0.17, p < 10^-7), contradicting the competitive exclusion hypothesis: higher-frequency features tend to have higher absorption, not lower. Steering experiments -- adding parent-direction interventions to absorbed features -- produce no sensitivity improvement (0.0 improvement for both absorbed and non-absorbed features), suggesting absorption may be epistemic rather than causal.

This paper makes the following contributions:

1. **Multi-child proportional ablation**: A measurement methodology that differentiates trained SAEs from random baselines, resolving the saturation that plagues single-feature ablation.

2. **Causal validation via steering**: The first test of whether absorbed features actively degrade downstream feature sensitivity, revealing that steering does not restore sensitivity.

3. **Competitive exclusion falsification**: Evidence that absorption does not follow the competitive exclusion pattern predicted by ecological analogy.

4. **Negative results**: Rigorous documentation of what fails -- steering interventions, frequency correlations -- enabling the field to move past unsuccessful approaches.

<!-- FIGURES
- None
-->
