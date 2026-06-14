# Methodology

We propose multi-child proportional ablation, a measurement methodology that resolves the saturation problem in prior single-feature ablation approaches. Our experimental design comprises four components: synthetic hierarchy generation, SAE training with Korznikov-style baselines, multi-child proportional absorption measurement, and steering intervention for causal validation.

## 5.1 Synthetic Hierarchy Generation

We construct synthetic feature hierarchies with known ground truth geometry to enable precise absorption measurement. Following the structure proposed in prior work on hierarchical feature learning, each hierarchy $H$ contains three levels:

- **Level 0 (Parent)**: A single high-level feature $p$
- **Level 1 (Children)**: Two child features $c_1, c_2$ that decompose the parent
- **Level 2 (Grandchildren)**: Four grandchildren $g_1, g_2, g_3, g_4$ per child, where $g_1$ and $g_2$ constitute $c_1$, and $g_3$ and $g_4$ constitute $c_2$

The geometric constraints are:

| Constraint | Value |
|------------|-------|
| $cos(p, c_1) = cos(p, c_2)$ | 0.67 |
| $cos(g_i, g_j)$ for $i \neq j$ | $-0.03$ to $0.03$ |
| $cos(g_1, g_2) = cos(g_3, g_4)$ | 1.0 |

The parent-children cosine similarity of 0.67 means the parent is a weighted sum of its children, consistent with the absorption hypothesis. Grandchildren are nearly orthogonal to each other but fully span their parent, ensuring the hierarchy is well-defined.

We generate 5 hierarchies per random seed, with 10,000 samples per hierarchy in the full experiment. The pilot used 100 samples per seed for rapid validation.

## 5.2 SAE Training

We train TopK Sparse Autoencoders on the synthetic activation data. The SAE decomposes residual stream activations $x \in \mathbb{R}^d$ into sparse features $f \in \mathbb{R}^{d_{sae}}$:

$$x = W_{enc} \cdot f + b_{enc}, \quad f = \text{TopK}(W_{dec} \cdot x + b_{dec})$$

where TopK retains only the $k$ highest-scoring features. We use the SAELens framework for implementation.

The architecture configuration is:

| Parameter | Value |
|-----------|-------|
| $d_{model}$ | 512 |
| $d_{sae}$ | 4096 |
| Expansion factor | 8x |
| L0 targets | {16, 32, 64} |
| Training steps | 20,000 |
| Learning rate | 1e-3 |
| Batch size | 4096 |
| Seeds | {42, 43, 44, 45, 46} |

This yields 15 trained SAE configurations (3 L0 targets x 5 seeds).

## 5.3 Baseline Methods

Following Korznikov et al. (2026), we employ three baseline methods to test whether absorption is specific to trained representations or a generic property of sparse decomposition:

1. **Random Decoder ($SAE_{rand}$)**: Xavier-initialized decoder weights with no training. The encoder is also random. This baseline has the same architecture as trained SAEs but no learned structure.

2. **Shuffled Features ($SAE_{shuff}$)**: The same activations and trained encoder, but feature indices are randomly permuted. This breaks feature identity while preserving activation statistics and encoder structure.

3. **Permuted Encoder ($SAE_{perm}$)**: A trained SAE with encoder weights randomly shuffled. This preserves decoder structure but breaks encoder-feature correspondence.

These baselines test three distinct null hypotheses: (a) absorption requires trained decoder weights, (b) absorption requires correct feature indexing, and (c) absorption requires encoder-decoder correspondence.

## 5.4 Multi-Child Proportional Ablation

### The Saturation Problem

Single-feature ablation saturates at 1.0 for both trained SAEs and random baselines. When we ablate one child feature, the remaining child can still reconstruct the parent because both children activate in the parent's presence. This saturation prevents differentiation between trained and baseline SAEs.

### Multi-Child Ablation

Our key insight is that ablating top-k children simultaneously tests whether children collectively substitute for the parent. For the absorption rate $abs_k$ after ablating top-k children:

$$abs_k = \frac{act(p | c_1, ..., c_k \text{ ablated})}{act(p)}$$

When $k=1$, both trained and random SAEs saturate at $abs_1 \approx 1.0$. When $k=5$ (ablating all children in our hierarchy), trained SAEs show $abs_5 = 0.50$ while random decoder shows $abs_5 = 0.059$.

### Proportional Variant

We also compute proportional absorption, which captures asymmetry in child contributions. For each child $c_i$, the proportional contribution is:

$$prop_i = \frac{cos(W_{dec}[p], W_{dec}[c_i])}{\sum_j cos(W_{dec}[p], W_{dec}[c_j])}$$

The proportional variance $var(prop)$ measures how asymmetrically children substitute for the parent. Trained SAEs show $var(prop) = 0.115$ versus $0.004$ for random decoder, indicating more structured absorption patterns.

## 5.5 Steering Intervention

To test whether absorption is causal (active interference) or epistemic (merely a measurement limitation), we conduct steering interventions on absorbed features.

### Identifying Absorbed Features

We classify features as absorbed if their proportional absorption exceeds 0.5 (top quartile of the distribution). In our pilot, this identified 7 absorbed features and 1,014 non-absorbed features.

### Parent Direction Reconstruction

We reconstruct the parent direction from children's decoder subspace:

$$parent\_dir = proj(span(W_{dec}[c_1], ..., W_{dec}[c_k]), W_{dec}[p])$$

This direction represents what the SAE would activate if the parent were present, computed purely from the children's geometry.

### Steering Protocol

We apply steering by adding scaled directions to activations:

$$x_{steered} = x + \alpha \cdot parent\_dir$$

with $\alpha \in \{0.05, 0.1, 0.15, 0.2, 0.25\}$. We measure feature sensitivity before and after steering on held-out text samples.

### Sensitivity Measurement

Feature sensitivity is measured as the change in feature activation when the corresponding concept appears in the input. Higher sensitivity indicates the feature more reliably tracks its intended concept.

## 5.6 Statistical Analysis

For hypothesis testing:

- **H1 (Multi-child absorption)**: Two-sample t-test comparing trained SAE vs. random decoder absorption rates across 5 seeds
- **H2 (Frequency correlation)**: Spearman rank correlation between feature activation frequency and absorption rate
- **H3 (Steering effectiveness)**: Paired t-test comparing sensitivity before vs. after steering for absorbed vs. non-absorbed features

Pass thresholds are pre-registered: H1 requires $p < 0.05$ and delta > 0.15; H2 requires $\rho < -0.3$ and $p < 0.01$; H3 requires $p < 0.01$ with absorbed features showing greater improvement than non-absorbed.

![Multi-Child Proportional Ablation Procedure](figures/fig5_method_architecture.pdf)

<!-- FIGURES
- Figure 5: fig5_method_architecture_desc.md — Architecture diagram illustrating the 3-level synthetic hierarchy and multi-child ablation procedure
- None
-->
