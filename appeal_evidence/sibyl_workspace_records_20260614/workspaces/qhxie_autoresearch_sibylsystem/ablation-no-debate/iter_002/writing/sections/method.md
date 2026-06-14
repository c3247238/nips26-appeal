# 5. Methodology

We design controlled experiments to decompose feature absorption into encoder and decoder contributions, validate robustness across random seeds, test intervention hypotheses, and characterize absorption under varying structural conditions. All experiments use synthetic hierarchical data with ground-truth parent-child relationships, enabling precise measurement of absorption rates without reliance on human-interpretable feature labels.

## 5.1 Synthetic Hierarchy Generation

Our synthetic data generator constructs three-level feature hierarchies with known structure. Each hierarchy contains one parent feature, two child features, and four grandchild features per child (eight grandchildren total). The generator produces residual stream activations $x \in \mathbb{R}^{d}$ where $d = 128$ by sampling from a Gaussian mixture model conditioned on hierarchy membership.

Parent-child cosine similarity is configurable across three levels: $\{0.5, 0.67, 0.8\}$. Grandchildren are pairwise orthogonal with cosine similarity in the range $[-0.03, 0.03]$. We add stochastic noise ($\sigma_{noise} = 0.1$) to hierarchy generation to ensure variance across random seeds. Each experimental configuration uses 5 hierarchies per seed with 10,000 samples per hierarchy for training and 2,000 for validation.

The ground-truth structure enables direct measurement of absorption: because parent and child features are explicitly defined in the generative model, we can compute the overlap between parent-active and child-active token sets without ambiguity.

## 5.2 SAE Training

We train TopK sparse autoencoders with $d_{model} = 128$ and $d_{sae} = 4096$ (32x expansion). The encoder $W_{enc} \in \mathbb{R}^{d_{sae} \times d}$ and decoder $W_{dec} \in \mathbb{R}^{d \times d_{sae}}$ are initialized with Xavier uniform weights. The bias terms $b_{enc}$ and $b_{dec}$ are initialized to zero.

Training uses Adam with learning rate $1 \times 10^{-3}$ and batch size 256 for 5,000 steps. The sparsity objective targets $L_0 \in \{20, 32, 50\}$ active features per sample, controlled via TopK activation. We train on 50,000 synthetic activations and validate on 10,000 held-out samples.

For the factorial decomposition (Section 5.4), we implement four training conditions by freezing subsets of parameters:
- **Condition A**: Neither encoder nor decoder trained (random initialization baseline)
- **Condition B**: Encoder trained, decoder frozen at initialization
- **Condition C**: Decoder trained, encoder frozen at initialization
- **Condition D**: Both encoder and decoder trained (full training)

## 5.3 Multi-Child Proportional Absorption Metric

Our primary metric, multi-child proportional absorption rate $\alpha_{abs}$, measures the fraction of parent activation routed through child latents. For a parent feature $p$ with children $c_1, c_2, ..., c_k$, we compute:

$$\alpha_{abs} = \frac{|\{x : f_c(x) > 0\} \cap \{x : f_p(x) > 0\}|}{|\{x : f_p(x) > 0\}|}$$

where $f_c(x) > 0$ indicates that at least one child feature is active on input $x$, and $f_p(x) > 0$ indicates the parent feature is active. This Jaccard overlap formulation captures the intuitive notion of substitution: when the parent is active, how often are children active instead.

We report absorption as a decimal rate in $[0, 1]$. A random SAE baseline with Xavier-initialized weights shows expected absorption near 0.03, reflecting chance overlap between independently initialized feature detectors.

## 5.4 2x2 Factorial Design

The central experiment decomposes absorption into encoder and decoder contributions through a 2x2 factorial crossing encoder state (random vs. trained) with decoder state (random vs. trained). Figure 7 illustrates the design.

![Conceptual diagram of the 2x2 factorial decomposition. Four quadrants show encoder-decoder combinations: A (random/random), B (trained/random), C (random/trained), D (trained/trained). Arrows indicate absorption pathways from parent to child features through the encoder and decoder subnetworks.](figures/figure_7_factorial_design.pdf)

The encoder effect $E_{enc} = \alpha(B) - \alpha(A)$ isolates the contribution of encoder alignment with hierarchical structure. The decoder effect $E_{dec} = \alpha(C) - \alpha(A)$ isolates the contribution of decoder geometry. The interaction $E_{int} = \alpha(D) - \alpha(B) - \alpha(C) + \alpha(A)$ captures non-additive encoder-decoder dynamics.

We validate across 5 seeds and 3 $L_0$ levels, yielding 15 independent runs. A run confirms the encoder-driven hypothesis if the encoder effect exceeds 0.5 and the decoder effect falls below 0.1. These thresholds reflect the pilot finding that the encoder effect (0.191) was large while the decoder effect (0.000) was negligible.

## 5.5 Steering Intervention Protocol

To test whether absorbed features are more sensitive to steering interventions, we implement a directional steering protocol. For each seed, we identify absorbed features (those with $\alpha_{abs} > 0.3$ and parent-child overlap $> 0.5$) and non-absorbed features (those with $\alpha_{abs} < 0.1$).

We steer feature activations toward parent directions by adding a scaled direction vector: $f_{steered} = f_{baseline} + \alpha \cdot v_{parent}$, where $\alpha \in \{0.5, 1.0, 2.0, 5.0\}$ is the steering coefficient and $v_{parent}$ is the parent feature's activation direction. We measure feature sensitivity as the change in reconstruction quality when the feature is ablated before and after steering.

The primary metric is the sensitivity ratio $s_{ratio} = s_{abs} / s_{non}$ at $\alpha = 2.0$. A ratio significantly greater than 1.0 would indicate that absorbed features are more responsive to parent-direction steering.

## 5.6 Safety-Critical Feature Analysis

We test whether safety-critical features show elevated absorption compared to non-safety features using pretrained Gemma Scope SAEs. We load the `gemma-scope-2b-pt-res` release via SAELens, targeting layer 12 with $d_{sae} = 16384$.

Feature selection proceeds in two stages. First, we identify 20 safety-relevant features by querying Neuronpedia for annotations related to deception, jailbreaking, harm, or manipulation. Second, we match each safety feature with a non-safety control feature from the same layer, matched by activation frequency distribution to control for baseline activity differences.

We measure absorption via the proportional method adapted to real SAEs: for each feature, we identify its top-activating tokens and test whether child-like sub-features (features with correlated activation patterns) substitute for the parent feature on those tokens. The Mann-Whitney U test compares absorption distributions between safety and control groups.

## 5.7 Ablation Schedule

We conduct two structural ablations to characterize how absorption varies with hierarchy strength and sparsity level.

**Hierarchy strength ablation**: We vary parent-child cosine similarity across $\{0.5, 0.67, 0.8\}$ while holding $L_0 = 32$ constant. Higher similarity should increase absorption if the mechanism depends on geometric alignment between parent and child features.

**L0 sparsity ablation**: We vary the target number of active features across $\{20, 32, 50\}$ while holding similarity at 0.67. This tests whether sparsity level modulates absorption. Our initial hypothesis predicted higher $L_0$ (more active features) would reduce absorption by distributing parent activation across more features; the results show the opposite pattern (Section 6.6).

**Held-out generalization**: We split synthetic data 80/20 by hierarchy instance (not by sample), training on 4 hierarchies and testing on 1 per seed. Perfect correlation between train and test absorption indicates that absorption generalizes to unseen hierarchical patterns from the same generative distribution.

<!-- FIGURES
- Figure 7: figure_7_factorial_design_desc.md, figure_7_factorial_design.pdf — Conceptual diagram of the 2x2 factorial decomposition showing encoder-decoder combinations and absorption pathways
- Table 1: inline — 2x2 factorial design conditions (A, B, C, D) with encoder/decoder states and expected absorption patterns
- None
-->
