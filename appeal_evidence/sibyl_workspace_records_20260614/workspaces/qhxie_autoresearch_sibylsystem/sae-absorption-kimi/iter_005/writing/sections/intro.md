# Abstract

Sparse autoencoders (SAEs) suffer from feature absorption, a failure mode where parent features are suppressed in favor of their children. Prior work reports absorption reductions from architectural innovations including multi-scale decomposition, orthogonality penalties, and gating mechanisms, but no study isolates which specific component drives the effect. We present the first component-isolated causal analysis, training six SAE variants on SynthSAEBench-16k ground-truth synthetic hierarchies and measuring absorption directly from known parent-child relationships. Our key finding is that TopK sparsity---not multi-scale decomposition or orthogonality---is the dominant driver of absorption reduction, with an effect size (Cohen's $d = 5.51$) an order of magnitude larger than any other tested component. An exploratory observation of a positive absorption--L0 sparsity correlation ($r \approx +0.93$ across $n = 4$ variants, $p = 0.067$) suggests that explicit sparsity control, not architectural novelty, may be the operative mechanism: higher L0 sparsity (more active latents) is associated with higher absorption. With only $n = 4$ points, this correlation is preliminary and requires confirmation. These results motivate further research into sparsity--absorption coupling.

# 1. Introduction

## 1.1 Feature Absorption as a Central Pathology

Sparse autoencoders (SAEs) have become the dominant approach for decomposing neural network activations into interpretable features. By learning an overcomplete dictionary of latent directions, SAEs aim to recover monosemantic representations---features that each encode a single human-interpretable concept---from the polysemantic superposition that pervades transformer residual streams (Bricken et al., 2023; Cunningham et al., 2023). The promise is substantial: if SAEs succeed, mechanistic interpretability can move from hand-crafted circuit tracing to automated feature discovery at scale.

Yet SAEs suffer from well-documented failure modes. Feature absorption, first characterized analytically by Chanin et al. (2024), is among the most consequential. When a parent feature (e.g., "animal") and its child features (e.g., "dog," "cat") co-occur in the training data, the SAE's sparsity loss incentivizes allocating representational capacity to the children at the parent's expense. The parent's information is not merely distributed across children; it is actively suppressed. Chanin et al. proved that this phenomenon is analytically incentivized by the L1 sparsity penalty for hierarchical features with parent-child containment structure.

The practical implications are severe. If SAEs absorb parent features, downstream interpretability tools that rely on SAE latents will miss high-level concepts entirely. A probe searching for "animal" features in a language model's SAE representation might find only "dog" and "cat" latents, with no latent encoding the general category. This undermines the core premise of SAE-based interpretability: that the latent basis captures the full conceptual structure of the model's internal representations.

## 1.2 The Absorption-Reduction Literature: A Component-Isolation Gap

Given the theoretical importance of absorption, the community has pursued architectural innovations that reduce it. The results are impressive:

- **Matryoshka SAEs** (Bussmann et al., 2025) report an order-of-magnitude absorption reduction (from 0.49 to 0.05) on the first-letter task, combining multi-scale dictionary decomposition, batch TopK sparsity, and hierarchical loss weighting.
- **OrtSAE** (Korznikov et al., 2025) achieves a 65% absorption reduction by adding decoder orthogonality penalties, with 4--11% compute overhead.
- **HSAE** (Zhan et al., 2026) enforces explicit tree-structured constraints on the dictionary.
- **Gated SAEs** (Rajamanoharan et al., 2024) decouple the detection path from the magnitude path, improving reconstruction-sparsity trade-offs.

Each paper reports full-architecture improvements. Matryoshka SAEs combine at least three distinct components; OrtSAE adds orthogonality to an existing architecture; Gated SAEs introduce a new activation mechanism. None of these studies isolate which specific component drives the reported absorption reduction. Is it the multi-scale decomposition? The explicit TopK sparsity? The orthogonality penalty? The gating mechanism? Or some interaction among them?

This question matters because the components differ radically in implementation complexity and computational cost. TopK sparsity is a one-line change to the activation function. Multi-scale decomposition requires nested dictionaries and hierarchical loss terms. Orthogonality penalties add a regularization term with hyperparameter tuning. If TopK sparsity alone achieves most of the absorption reduction, then the community's investment in more complex architectures may be misdirected.

## 1.3 The Measurement Crisis Motivating This Study

Our prior work (iterations 2--4 of this research project) revealed fatal anomalies in probe-based absorption metrics on real LLMs that make causal component isolation impossible with existing measurement tools:

**Co-occurrence confound.** Non-hierarchy correlated word pairs produce higher "absorption" scores than true semantic hierarchies ($\bar{A}_{\text{NH}} = 0.331$ vs. $\bar{A}_{\text{SH}} = 0.235$; paired t-test: $t = -4.748$, $p = 0.003$). This proves the metric detects correlation, not containment structure.

**Ceiling effect.** All probe AUROCs on residual activations equal 1.0, collapsing the absorption formula's numerator to a degenerate quantity. The metric has no headroom to discriminate conditions.

**Model dependence.** GPT-2 small shows near-zero semantic-hierarchy absorption (0.000--0.003) where Pythia-160M shows substantial values (0.064--0.359). The same metric on the same task yields qualitatively different results across base models.

**Geometric dominance.** A Random-SAE control with permuted decoder achieves semantic-hierarchy absorption comparable to trained SAEs (0.175 vs. 0.064--0.359), suggesting the metric captures base-model geometry rather than learned SAE structure.

These findings mean real-LLM probe-based absorption metrics cannot support causal claims about architectural components. Any observed difference between Matryoshka and Standard could reflect geometric confounds, probe artifacts, or co-occurrence sensitivity rather than genuine architectural improvement.

## 1.4 Pivot to Ground-Truth Synthetic Data

SynthSAEBench-16k (Chanin & Garriga-Alonso, 2026) provides an escape hatch. With 16,384 ground-truth features (10,884 hierarchical) organized into 128 root trees of depth 3 and branching factor 4, absorption can be measured directly from known parent-child relationships. No probes. No AUROCs. No ceiling effects. The absorption rate is simply the fraction of parent features subsumed by their children, computed from the ground-truth feature structure.

This enables causal component isolation: we train six SAE variants that differ by exactly one architectural component, measure ground-truth absorption on each, and attribute effects to the component that changed. The design is a classic ablation study applied to SAE architecture, with the critical advantage that the ground-truth metric is unambiguous.

## 1.5 Research Questions

This paper is guided by three research questions:

**RQ1 (Component causality).** Which specific architectural component is the primary driver of absorption reduction? We test six variants: Baseline ReLU, +TopK sparsity, +MultiScale dictionaries, +Orthogonality penalties, +Gating, and +Full Matryoshka (all components combined).

**RQ2 (Component ranking).** What is the ordering of components by effect size on absorption reduction, feature recovery, and reconstruction quality?

**RQ3 (Trade-off structure).** Do absorption-reducing components introduce new pathologies (hedging, reconstruction loss, compute overhead)?

**Scope note.** This paper reports partial results: 3 of 6 variants have full 5-replicate data (Baseline, TopK, Orthogonality), 1 has pilot data (+MultiScale), and 2 are pending (+Gating, +Full Matryoshka). The component ranking is provisional and may change when the full variant set is completed. We flag this limitation prominently throughout and discuss its implications in Section 5.5.

## 1.6 Contributions

Our study makes four contributions:

1. **First component-isolated causal analysis of SAE absorption-reduction mechanisms.** By varying one component at a time on ground-truth synthetic data, we can attribute effects to specific architectural choices rather than full-architecture bundles.

2. **First ground-truth validation of absorption-reduction claims on synthetic hierarchies.** All prior absorption measurements use probe-based metrics on real LLMs. We measure absorption directly from known parent-child relationships, eliminating probe artifacts.

3. **Evidence that TopK sparsity---not multi-scale decomposition or orthogonality---is the dominant driver.** TopK reduces absorption by 78% (Cohen's $d = 5.51$), an order of magnitude larger than any other tested component. Orthogonality achieves only 2.7% reduction ($d = 0.14$).

4. **An exploratory observation of a positive absorption--L0 sparsity correlation** ($r \approx +0.93$ across $n = 4$ variants, 95% CI: $[+0.87, +1.00]$, $p = 0.067$). Higher L0 sparsity (more active latents) is associated with higher absorption. With only $n = 4$ points and 2 degrees of freedom, this correlation is exploratory, not a primary contribution. Confirmation requires the full 6-variant set and a dedicated sparsity-sweep experiment. If confirmed, it would suggest that explicit sparsity control---not architectural novelty---is the operative mechanism.

From the motivation, we turn to the experimental design that enables causal component isolation.

<!-- FIGURES
- None
-->
