# 1. Introduction

## 1.1 Feature Absorption as a Central Pathology

Sparse autoencoders (SAEs) have become the dominant approach for decomposing neural network activations into interpretable features. By learning an overcomplete dictionary of latent directions, SAEs aim to recover monosemantic representations---features that each encode a single human-interpretable concept---from the polysemantic superposition that pervades transformer residual streams (Bricken et al., 2023; Cunningham et al., 2023). The promise is substantial: if SAEs succeed, mechanistic interpretability can move from hand-crafted circuit tracing to automated feature discovery at scale.

Yet SAEs suffer from well-documented failure modes. Feature absorption, first characterized analytically by Chanin et al. (2024), is among the most consequential. When a parent feature (e.g., "animal") and its child features (e.g., "dog," "cat") co-occur in the training data, the SAE's sparsity loss incentivizes allocating representational capacity to the children at the parent's expense. The parent's information is not merely distributed across children; it is actively suppressed. Chanin et al. proved that this phenomenon is incentivized by sparsity loss for hierarchical features, though our results suggest the metric may also respond to non-hierarchical correlations.

The practical implications are severe. If SAEs absorb parent features, downstream interpretability tools that rely on SAE latents will miss high-level concepts entirely. A probe searching for "animal" features in a language model's SAE representation might find only "dog" and "cat" latents, with no latent encoding the general category. This undermines the core premise of SAE-based interpretability: that the latent basis captures the full conceptual structure of the model's internal representations.

## 1.2 The First-Letter Benchmark and Its Limitations

Given the theoretical importance of absorption, the community needed a standardized measurement. SAEBench (Karvonen et al., 2025) provided one, incorporating an absorption evaluator based on Chanin et al.'s first-letter classification task. The task constructs 26 parent-child hierarchies defined by character-level properties: a parent feature like "starts with S" with children like "short," "small," and "smart." A ground-truth logistic probe trained on base-model residual activations provides an upper-bound baseline; probes on full SAE latents and top-$k$ sparse latents measure information loss. The absorption score $A_{\text{full}}$ quantifies the maximum relative accuracy drop across these three conditions:

$$A_{\text{full}} = \max\left(0, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{sae}}}{\text{acc}_{\text{resid}}}, \frac{\text{acc}_{\text{resid}} - \text{acc}_{\text{k-sparse}}}{\text{acc}_{\text{resid}}}\right)$$

where $\text{acc}_{\text{resid}}$ is the probe accuracy on base-model residual activations, $\text{acc}_{\text{sae}}$ is accuracy on full SAE latents, and $\text{acc}_{\text{k-sparse}}$ is accuracy on top-$k$ SAE latents.

The first-letter benchmark has genuine strengths. Ground-truth labels are available without human annotation. Causal ablations are tractable because the hierarchies are cleanly defined. The task isolates a specific structural property---parent-child containment---that theory predicts should trigger absorption. These virtues have made first-letter absorption one of several standardized evaluations in SAEBench, and architecture papers routinely report it as a primary metric (Bussmann et al., 2025; Rajamanoharan et al., 2024).

But the benchmark also has a critical limitation: it is an artificial task with no clear relationship to the semantic hierarchies that motivate absorption theory. Chanin et al. (2024) themselves noted that "finding examples of feature absorption unrelated to character identification" remains open future work. The first-letter hierarchies are defined by orthographic properties, not semantic ones. Whether absorption scores on "starts with S" generalize to "animal $\to$ dog" is an empirical question that has never been tested. Without such a test, architecture comparisons that rank SAEs by first-letter absorption may be optimizing for a metric that does not reflect behavior on real conceptual hierarchies.

## 1.3 Research Questions

This paper conducts the first construct-validity study of the SAEBench absorption metric. We adapt the first-letter evaluation protocol to semantic hierarchies extracted from WordNet (Miller, 1995) and test whether the resulting scores correlate with, and are specific to, hierarchical structure. Our analysis is guided by three research questions, which we test via three hypotheses (H1--H3, Section 2.6):

**RQ1 (Construct Validity):** Do first-letter absorption scores correlate with semantic-hierarchy absorption scores across diverse SAE architectures? If the metric measures a general absorption phenomenon, architectures that score high on first-letter hierarchies should also score high on semantic hierarchies.

**RQ2 (Hierarchy Specificity):** Is the metric specific to hierarchical features, or does it detect absorption-like behavior in semantically correlated but non-hierarchical pairs? A valid absorption metric should score higher on parent-child hierarchies than on synonym or co-occurrence pairs that lack containment structure.

**RQ3 (Robustness):** How stable is the correlation across feature-splitting thresholds ($\tau_{\text{fs}}$) in k-sparse probing? We treat this as a secondary robustness check rather than a primary research question.

## 1.4 Contributions

Our study makes four contributions:

1. **First construct-validity test on semantic hierarchies derived from WordNet.** We provide the first empirical test of whether the dominant SAE absorption metric generalizes from artificial first-letter hierarchies to real semantic hierarchies.

2. **Evidence of hierarchy-specificity failure.** All eight architectures show higher non-hierarchy absorption than semantic-hierarchy absorption (mean $\bar{A}_{\text{NH}} = 0.331$ vs. $\bar{A}_{\text{SH}} = 0.235$; paired t-test: $t = -4.748$, $p = 0.003$), rejecting the hypothesis that the metric is specific to hierarchical structure. This rejection holds under our experimental conditions; we discuss alternative explanations, including template-induced spurious correlations, in Section 4.2.

3. **Random-SAE control anomaly.** On semantic hierarchies, a Random-SAE control achieves scores comparable to trained SAEs (0.175 vs. 0.064--0.359 across architectures), suggesting that the semantic-hierarchy adaptation of the metric may capture geometric artifacts rather than learned structure. This degeneracy is task-specific: the first-letter task does distinguish trained from random SAEs, with Random scoring near-zero (0.030) while TopK scores 0.576.

4. **Open-source replication materials.** We release the WordNet hierarchy dataset, evaluation code, and per-architecture scores to enable community replication and extension.

The implications extend beyond this specific metric. Our findings reveal a methodological blind spot in a widely adopted benchmark and suggest that domain-specific absorption metrics---validated for hierarchy specificity and sensitivity to training---are needed before absorption scores can guide architecture selection for real-world interpretability tasks.

We now describe the measurement protocol that adapts the SAEBench absorption evaluator to semantic hierarchies, non-hierarchical controls, and a randomized baseline.

<!-- FIGURES
- None
-->
