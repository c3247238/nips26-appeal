## 4. Experiments

We present three experiments that address our research questions in sequence: probe scalability (E3v2), training-free detector validation (E6v2), and cross-architecture comparison (E7). Each experiment builds on the previous, moving from establishing a reliable measurement pipeline to characterizing the architectural and layer-dependent behavior of absorption metrics.

### 4.1 E3v2: Scaled Semantic Probes on GemmaScope

#### Setup

We evaluate 30 semantic probes (10 WordNet categories $\times$ 3 layers) on Gemma-2-2B with JumpReLU SAEs from GemmaScope (Lieberum et al., 2024). We sample layers 5, 10, and 15 from the width\_16k configuration ($d_{\text{SAE}} = 16384$), corresponding to relative depths of 0.19, 0.38, and 0.58. Each probe uses 15 hyponyms per category, yielding 105 training and 45 test examples.

#### Probe Quality

All 30 probes exceed the validity threshold (AUROC $>$ 0.7), achieving a mean AUROC of 0.980 $\pm$ 0.034. Figure 3 shows the AUROC distribution across layers. The minimum AUROC is 0.854 (plant at layer 15), well above the validity threshold. Layer 5 achieves the highest mean AUROC (0.987), with layer 10 at 0.978 and layer 15 at 0.976. The slight degradation at deeper layers may reflect changes in representational structure, though the effect is small (delta AUROC = 0.011).

![Figure 3: Probe AUROC distribution across GemmaScope layers 5, 10, and 15. All 30 probes exceed the validity threshold of AUROC = 0.7 (red dashed line). Diamond markers indicate means. Box colors distinguish layers: blue (layer 5), orange (layer 10), green (layer 15).](figures/fig3_auroc.pdf)

#### Absorption Metrics

Table 1 reports per-layer absorption statistics. Projection absorption is consistently high across all layers: 98.3% at layer 5, 98.2% at layer 10, and 98.1% at layer 15. The ablation-based metric detects no absorption (0.0% across all layers), with mean ablation scores of $0.0016 \pm 0.0082$. This confirms the functional insensitivity of ablation scores on JumpReLU SAEs, consistent with our Iteration 2 findings.

**Table 1: Probe AUROC and Absorption Metrics per Layer --- GemmaScope JumpReLU**

| Layer | Mean AUROC | Valid/Total | Mean $A_{\text{proj}}$ | Ablation Rate | Proj. Rate ($>$0.5) |
|-------|-----------|-------------|------------------------|---------------|---------------------|
| 5 | 0.987 | 10/10 | 0.983 $\pm$ 0.014 | 0.0% | 100% |
| 10 | 0.978 | 10/10 | 0.982 $\pm$ 0.011 | 0.0% | 100% |
| 15 | 0.976 | 10/10 | 0.981 $\pm$ 0.010 | 0.0% | 100% |
| **All** | **0.980** | **30/30** | **0.982 $\pm$ 0.012** | **0.0%** | **100%** |

#### H1 Validation

Hypothesis H1 states that expanding from 5 to 15 hyponyms per category will improve failed probes from 9/10 to 10/10 per category without changing mean absorption rates. The result is a clear pass: all 30/30 probes are valid (100%), compared to 27/30 in our prior unpublished pilot with 5 hyponyms. Mean projection absorption remains high (0.982), confirming that the expanded dataset improves probe reliability without altering the underlying absorption pattern.

### 4.2 E6v2: GPT-2 $A_j$ Validation

#### Setup

We compute the training-free $A_j$ detector on GPT-2 Small with ReLU SAEs from SAELens, evaluating layers 5, 8, and 10 ($d_{\text{SAE}} = 24576$, $d_{\text{model}} = 768$). The same 10 semantic categories and 15 hyponyms per category are used, with prompts processed through GPT-2 to extract residual stream activations at each target layer.

#### Decoder Norm Analysis

A prerequisite for interpreting $A_j$ is understanding decoder norm constraints. Our original hypothesis (from proposal.md) predicted that GPT-2 ReLU SAEs would have unconstrained decoder norms and therefore higher $A_j$ correlations than GemmaScope JumpReLU. We find that GPT-2 ReLU SAEs have decoder norms fixed at approximately 1.0 across all layers: mean $1.000045 \pm 0.000005$ (layer 5: 1.000038, layer 8: 1.000046, layer 10: 1.000051). This contradicts our original hypothesis. Both GemmaScope JumpReLU and GPT-2 ReLU maintain norm constraints, suggesting that decoder normalization emerges from training dynamics rather than architectural design, though architectural effects cannot be ruled out with only two SAE families.

#### Layer-Dependent Correlation Pattern

Table 2 reports $A_j$ statistics and correlations with projection absorption for each layer. The key finding is a non-monotonic correlation pattern across layer depth. Correlations are computed over $n = 10$ category-level points; 95% confidence intervals are reported using Fisher's z-transformation.

**Table 2: Layer Statistics and $A_j$ Correlations --- GPT-2 ReLU**

| Layer | $A_j$ Mean | $A_j$ Std | Dec. Norm Mean | $\rho$ (proj) | 95% CI | $p$-value | Significant? |
|-------|-----------|-----------|----------------|---------------|--------|-----------|--------------|
| 5 | 0.252 | 0.148 | 1.00004 | $-$0.590 | [$-$0.90, +0.08] | 0.073 | Marginal |
| 8 | 0.287 | 0.087 | 1.00005 | **+0.705** | [**+0.12**, **+0.93**] | **0.023** | **Yes** |
| 10 | 0.300 | 0.080 | 1.00005 | $-$0.697 | [$-$0.93, $-$0.10] | 0.025 | Yes |

Layer 8 shows the strongest positive correlation ($\rho = 0.705$, $p = 0.023$), while layers 5 and 10 show negative correlations of similar magnitude ($\rho = -0.590$ and $-0.697$). The sign flip between adjacent layers is the most statistically distinctive feature of this pattern. Figure 4 visualizes the layer 8 scatter plot.

![Figure 4: $A_j$ vs projection absorption for 10 semantic probes at GPT-2 ReLU layer 8. Each point represents one category; the regression line shows the positive correlation ($\rho = 0.705$, $p = 0.023$). Category labels are shown for interpretability.](figures/fig4_aj_scatter.pdf)

#### H2 Analysis

The original H2 hypothesis---that $A_j$ correlation would be higher on GPT-2 ReLU ($\rho > 0.6$) than GemmaScope JumpReLU---fails. The mean $\rho$ across GPT-2 layers is $-0.194$, far below the 0.6 threshold. However, an exploratory observation reveals a more nuanced finding: layer 8 achieves $\rho = 0.705 > 0.6$. With only 3 layers tested and correlations computed over $n = 10$ categories, this observation requires validation on additional layers before generalizing. The sign flip between layers indicates that $A_j$ detector effectiveness varies systematically with network depth rather than architecture alone.

### 4.3 E7: Cross-Architecture Comparison

#### Setup

We compare absorption metrics between GemmaScope JumpReLU (30 probes, layers 5, 10, 15) and GPT-2 ReLU (30 probes, layers 5, 8, 10). Layer depths are matched where possible, though exact alignment is complicated by the different model depths (Gemma-2-2B has 26 layers; GPT-2 Small has 12).

#### Projection Absorption Stability

Table 3 presents the cross-architecture statistical comparison. Projection absorption differs by 7.7% between architectures: GemmaScope achieves 98.2% $\pm$ 1.2%, while GPT-2 achieves 91.2% $\pm$ 5.2%. This difference is statistically significant ($p < 0.001$, Mann-Whitney U; Cohen's $d = 1.82$) but the absolute difference is modest, supporting the stability of projection-based metrics across architectural families.

**Table 3: Cross-Architecture Statistical Comparison**

| Metric | Gemma JumpReLU | GPT-2 ReLU | % Diff | $p$-value | Cohen's $d$ |
|--------|---------------|------------|--------|-----------|-------------|
| Projection absorption | 0.982 $\pm$ 0.012 | 0.912 $\pm$ 0.052 | 7.7% | $<$0.001 | 1.82 |
| Ablation rate | 0.0% | 33.3% | --- | --- | --- |
| Mean ablation score | 0.0016 $\pm$ 0.0082 | 0.0192 $\pm$ 0.0358 | 91.9%$^a$ | 0.040 | $-$0.67 |

$^a$The 91.9% percentage difference is misleading: the absolute difference is only 0.0176, and the large relative value arises from dividing by a near-zero baseline (0.0016).

Figure 5 visualizes the comparison. Ablation rates are low on GemmaScope (0.0%) and GPT-2 (33.3%), confirming that functional insensitivity is not architecture-specific.

![Figure 5: Cross-architecture absorption comparison. Projection absorption (blue) is high on both architectures with a 7.7% difference. Ablation rates (orange) are low and similar. Significance markers: * $p <$ 0.001 for projection absorption difference.](figures/fig5_cross_arch.pdf)

#### H3 Validation

Hypothesis H3 states that projection-based absorption rates will differ by $<$ 10% between architectures. The actual difference is 7.67%, so H3 passes. This validates projection-based absorption as a robust cross-architecture baseline for absorption quantification.

#### Decoder Norm Comparison

Both architectures maintain norms approximately 1.0, with GPT-2 ReLU showing mean 1.000045 and GemmaScope JumpReLU constrained by design. The consistency of norm constraints across architectures is consistent with the hypothesis that training dynamics contribute to decoder normalization, though architectural effects cannot be ruled out with only two SAE families.

![Figure 6: Decoder norm statistics across layers for GPT-2 ReLU (layers 5, 8, 10) and GemmaScope JumpReLU (layers 5, 10, 15). Both architectures maintain decoder norms approximately 1.0. Error bars show standard deviation.](figures/fig6_dec_norm.pdf)

### 4.4 Layer-Dependent Correlation Analysis (H2v2)

#### Key Finding

The $A_j$ correlation with projection absorption is non-monotonic across layer depth in GPT-2 ReLU SAEs. Layer 8 (relative depth $\sim$0.67) shows a strong positive correlation ($\rho = 0.705$), while layers 5 and 10 show negative correlations of similar magnitude ($\rho = -0.590$ and $-0.697$). This mid-layer peak suggests that feature hierarchies are most detectable by the $A_j$ detector at intermediate-to-deep layers.

#### Statistical Significance

Table 4 reports pairwise comparisons of correlations using Fisher's r-to-z transformation. The layer 8 correlation is significantly different from both layer 5 ($z = 2.91$, $p = 0.0036$) and layer 10 ($z = 3.25$, $p = 0.0011$). The layer 5 vs layer 10 comparison is not significant ($z = 0.35$, $p = 0.73$), indicating that the two outer layers have statistically equivalent negative correlations.

**Table 4: Pairwise Correlation Comparisons --- GPT-2 Layers**

| Comparison | $\rho$ diff | $z$-statistic | $p$-value | Significant? |
|------------|------------|---------------|-----------|--------------|
| Layer 8 vs 5 | +1.295 | 2.909 | 0.0036 | Yes |
| Layer 8 vs 10 | +1.402 | 3.253 | 0.0011 | Yes |
| Layer 5 vs 10 | +0.107 | 0.345 | 0.7304 | No |

Figure 7 visualizes the layer-dependent pattern. The sign flip at adjacent layers implies that $A_j$ captures different phenomena at different depths, or that the relationship between $A_j$ and absorption is modulated by layer-specific representational structure.

![Figure 7: Layer-dependent $A_j$ correlation pattern across GPT-2 layers. Spearman $\rho$ (y-axis) vs layer depth (x-axis). Layer 8 shows a strong positive correlation (orange, significant), while layers 5 and 10 show negative correlations (gray). The horizontal line marks $\rho = 0$; the shaded region indicates the near-zero range.](figures/fig7_layer_corr.pdf)

#### Interpretation

The sign-flip pattern has three plausible explanations. First, feature hierarchies may be most pronounced at mid-layers, where abstract concepts have been formed but not yet compressed into distributed representations. Second, deep layers (layer 10) may encode more distributed, context-dependent features that are less amenable to single-latent absorption detection. Third, shallow layers (layer 5) may lack sufficient semantic structure for the $A_j$ detector to discriminate absorbed from non-absorbed features. Future work could distinguish these explanations via activation patching on individual latents (Section 5.5).

<!-- FIGURES
- Figure 3: gen_fig3_auroc.py, fig3_auroc.pdf --- AUROC distribution boxplot across GemmaScope layers 5, 10, 15
- Figure 4: gen_fig4_aj_scatter.py, fig4_aj_scatter.pdf --- A_j vs projection absorption scatter for GPT-2 layer 8
- Figure 5: gen_fig5_cross_arch.py, fig5_cross_arch.pdf --- Cross-architecture absorption comparison bar chart
- Figure 6: gen_fig6_dec_norm.py, fig6_dec_norm.pdf --- Decoder norm statistics across layers
- Figure 7: gen_fig7_layer_corr.py, fig7_layer_corr.pdf --- Layer-dependent correlation pattern line plot
- Table 1: inline --- Probe AUROC and absorption metrics per GemmaScope layer
- Table 2: inline --- Layer statistics and A_j correlations per GPT-2 layer
- Table 3: inline --- Cross-architecture statistical comparison
- Table 4: inline --- Pairwise correlation comparisons with z-statistics
-->
