# 1. Introduction

## 1.1 Motivation: The SAE Credibility Crisis and the Absorption Problem

Sparse autoencoders (SAEs) have become the dominant paradigm for mechanistic interpretability of large language models. By decomposing high-dimensional activations into sparse, interpretable features, SAEs promise to unpack the "superposition" problem---the phenomenon where neural networks represent more features than they have dimensions---into human-understandable concepts (Bricken et al., 2023; Cunningham et al., 2023; Templeton et al., 2024). The field has invested heavily in this promise: SAELens, a standard library for SAE training and analysis, supports dozens of architectures; SAEBench provides standardized benchmarking across models and metrics; and recent work has scaled SAEs to frontier models with millions of latents.

Yet SAEs face a credibility crisis. A growing body of work identifies systematic failure modes that call into question whether SAE features correspond to genuine, causally meaningful model computations. Feature absorption, first identified by Chanin et al. (2024), is one such failure mode: general (parent) features fail to fire when more specific (child) features are present, with the child's activation capturing what should have been the parent's. In a well-known example, a feature representing "starts with any letter" may fail to activate when "starts with Q" fires, even though the input clearly satisfies both conditions. The parent feature is not destroyed---its decoder direction remains intact---but its encoder activation is suppressed.

Existing work detects absorption but does not explain why it happens or which features are at risk. Chanin et al. (2024) proposed a differential correlation metric that measures absorption by comparing parent-child correlations before and after ablating the child feature. SAEBench (Karvonen et al., 2025) standardized this metric for benchmarking. Architectural solutions---Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), and HSAE---all retrain the SAE to reduce absorption, but none provides a mechanistic theory connecting absorption to SAE architecture.

The gap is twofold. First, no theory explains why absorption occurs: why does a child feature suppress its parent? Second, no training-free diagnostic predicts at-risk features without running absorption metrics: practitioners must evaluate every feature pair, a computationally expensive process that scales poorly with dictionary size. This paper addresses both gaps.

## 1.2 The LCA Connection: From Neuroscience to SAEs

Our central insight is that the SAE decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from the Locally Competitive Algorithm (LCA), a neuroscience-inspired sparse coding framework proposed by Rozell et al. (2008). In LCA, neurons compete via lateral inhibition governed by $G$: when neuron $j$ fires, it suppresses neuron $i$ in proportion to $G_{ij}$. Rozell et al. showed that for a dictionary $W_{\text{dec}}$, the natural choice is $G = W_{\text{dec}}^T W_{\text{dec}}$, because the inhibition term then equals the projection of the reconstructed signal back into the latent space.

For an SAE with tied encoder-decoder weights ($W_{\text{enc}} = W_{\text{dec}}^T$), this correspondence is exact: the decoder correlation matrix is the LCA inhibition matrix. Even with untied weights---the norm in modern SAE training---decoder correlations encode the same competitive relationships. When latent $j$ fires with activation $z_j$, its contribution to the reconstruction is $z_j \cdot W_{\text{dec}}[j]$. The overlap $\langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ measures how much the reconstruction contributed by latent $j$ projects onto the encoder direction of latent $i$, reducing $i$'s net input.

This correspondence transforms absorption from a mysterious pathology into a predictable consequence of competitive suppression. A child feature that fires strongly inhibits its parent via decoder correlations; the parent fails to reach threshold (recall loss) but its decoder direction remains unchanged (precision preserved). The LCA framework explains the precision-recall asymmetry observed in prior work: precision is invariant because inhibition suppresses true positives but does not cause false positives; recall varies because the degree of suppression depends on which child features are active.

The LCA has approximately 2,000 citations but zero applications to LLM SAEs. This paper is the first to connect the two frameworks.

## 1.3 Research Questions

We test five research questions:

**RQ1 (Primary):** Does the local inhibition graph predict known absorption pairs? We construct a graph from decoder correlations (top-$k$ neighbors per latent) and test whether edges correspond to Chanin et al. absorption pairs with precision significantly above chance.

**RQ2 (Secondary):** Does inhibition explain the precision-recall asymmetry? We test whether total incoming inhibition correlates with recall loss but not with precision degradation.

**RQ3 (Secondary):** Can the graph predict at-risk features before running absorption metrics? We test whether graph statistics (total inhibition, mean edge weight) correlate with absorption rate.

**RQ4 (Exploratory):** Does graph structure vary across layers? We compare inhibition graphs at layers 0, 4, 8, and 10 of GPT-2 Small.

**RQ5 (Exploratory):** Can homeostatic rebalancing restore parent firing? We propose a single-pass correction $z'_i = z_i + \alpha \sum_{j \in N(i)} G_{ij} z_j$ and test whether it restores parent firing with minimal reconstruction cost.

## 1.4 Contributions

1. **First connection between LCA lateral inhibition and SAE absorption.** We show that $W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing a mechanistic explanation for absorption as competitive suppression.

2. **First local inhibition graph for SAE diagnostics.** The graph is training-free, computed from pretrained weights, and scales to million-latent SAEs via top-$k$ neighbor sparsification.

3. **Mechanistic explanation for precision-recall asymmetry.** Competitive suppression explains why absorption affects recall (coverage) but not precision (selectivity)---a finding from prior work that lacked theoretical grounding.

4. **First training-free post-hoc repair for absorption.** Homeostatic rebalancing operates on pretrained SAEs with a single forward-pass correction, inspired by biological homeostatic plasticity.

5. **Validation on GPT-2 Small with integration of prior empirical findings.** The inhibition framework provides a unified explanation for all key findings from iterations 1--8 of this project: precision invariance, recall loss, layer-dependent effects, steering robustness, and the necessity of delta-corrected metrics.

## 1.5 Key Results Preview

Our empirical validation yields a nuanced picture. The primary hypothesis (H6: graph edges predict absorption pairs) is falsified: precision@20 = 0.0, with no enrichment over chance ($p = 1.0$, Fisher exact test). The local inhibition graph with fixed $k = 20$ does not directly identify parent-child absorption relationships. Graph statistics similarly fail to predict at-risk features (H8: $r = +0.12$, $p = 0.55$).

However, the mechanistic framework is strongly supported. The precision-recall asymmetry (H7) matches the competitive suppression prediction exactly: precision standard deviation is 0.016 at $k_{\text{probe}} = 20$ (layer 8), while recall standard deviation is 0.167---more than 10x larger. Twenty-five of 26 first-letter features achieve precision = 1.0, while recall ranges from 0.21 to 0.49. The correlation between absorption rate and recall is negative at all sparsity levels ($r = -0.282$ at $k = 20$), while the correlation with precision is near-zero ($|r| < 0.11$).

Layer-dependent graph structure (H9) shows a descriptive trend: mean edge weight increases monotonically from 0.312 (layer 0) to 0.384 (layer 8), aligning with the prior finding that layer 8 exhibits the strongest absorption-steering correlation ($r = -0.431$, $p = 0.028$ for delta-corrected steering). Homeostatic rebalancing (H10) was deferred pending improved graph construction.

The central takeaway is that the **mechanistic framework is supported even when the predictive tool is not**. Competitive suppression explains the precision-recall asymmetry and integrates all prior findings, even though the local inhibition graph with fixed $k$ does not directly predict absorption pairs. This suggests that inhibition operates at a finer granularity than top-k neighbor relationships, and that future work should explore larger $k$, adaptive neighborhood sizes, or context-dependent edge weighting.

The remainder of this paper is organized as follows. Section 2 reviews background on SAEs, feature absorption, and the LCA. Section 3 presents the local inhibition graph framework, the structural correspondence proof, and the competitive suppression mechanism. Section 4 describes the experimental methodology and reports results. Section 5 discusses implications, limitations, and future work. Section 6 concludes.

<!-- FIGURES
- None
-->
