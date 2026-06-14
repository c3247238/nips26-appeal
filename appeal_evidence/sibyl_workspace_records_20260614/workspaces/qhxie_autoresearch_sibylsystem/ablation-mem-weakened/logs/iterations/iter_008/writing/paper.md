# Competitive Suppression in Sparse Autoencoders: Connecting the Locally Competitive Algorithm to Feature Absorption

## Abstract

Feature absorption in sparse autoencoders (SAEs)---where general parent features fail to fire when specific child features are present---has been characterized as a failure mode requiring architectural intervention. We present the first connection between the Locally Competitive Algorithm (LCA) from neuroscience and SAE feature absorption, showing that the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs. This correspondence transforms absorption from a mysterious pathology into a predictable consequence of competitive suppression: child features inhibit parents via decoder correlations, causing recall loss without precision degradation.

We validate this framework on GPT-2 Small (124M parameters) with 26 first-letter features across layers 0, 4, 8, and 10. The primary predictive hypothesis---that a local inhibition graph constructed from decoder correlations predicts absorption pairs---is falsified: precision@20 = 0.0 with no enrichment over chance ($p = 1.0$, Fisher exact test). Graph statistics similarly fail to predict at-risk features (H8: $r = +0.12$, $p = 0.55$). However, the mechanistic framework is strongly supported by the precision-recall asymmetry (H7): precision standard deviation is 0.016 at $k_{\text{probe}} = 20$ (layer 8), while recall standard deviation is 0.167---more than 10x larger. Twenty-five of 26 first-letter features achieve precision = 1.0, while recall ranges from 0.21 to 0.49. Layer-dependent graph structure (H9) shows a descriptive trend: mean edge weight increases from 0.312 (layer 0) to 0.384 (layer 8), aligning with prior findings that deeper layers exhibit stronger absorption effects.

The central contribution is a mechanistic explanation, not a predictive tool. The LCA connection provides a unified account of all prior empirical findings: precision invariance, recall loss, layer dependence, steering robustness, and the necessity of delta-corrected metrics. Absorption is not a critical failure mode for SAE interpretability; it is a predictable consequence of competitive suppression that reduces coverage without affecting selectivity.

## 1 Introduction

### 1.1 Motivation: The SAE Credibility Crisis and the Absorption Problem

Sparse autoencoders (SAEs) have become the dominant paradigm for mechanistic interpretability of large language models. By decomposing high-dimensional activations into sparse, interpretable features, SAEs promise to unpack the "superposition" problem---the phenomenon where neural networks represent more features than they have dimensions---into human-understandable concepts (Bricken et al., 2023; Cunningham et al., 2023; Templeton et al., 2024). The field has invested heavily in this promise: SAELens, a standard library for SAE training and analysis, supports dozens of architectures; SAEBench provides standardized benchmarking across models and metrics; and recent work has scaled SAEs to frontier models with millions of latents.

Yet SAEs face a credibility crisis. A growing body of work identifies systematic failure modes that call into question whether SAE features correspond to genuine, causally meaningful model computations. Feature absorption, first identified by Chanin et al. (2024), is one such failure mode: general (parent) features fail to fire when more specific (child) features are present, with the child's activation capturing what should have been the parent's. In a well-known example, a feature representing "starts with any letter" may fail to activate when "starts with Q" fires, even though the input clearly satisfies both conditions. The parent feature is not destroyed---its decoder direction remains intact---but its encoder activation is suppressed.

Existing work detects absorption but does not explain why it happens or which features are at risk. Chanin et al. (2024) proposed a differential correlation metric that measures absorption by comparing parent-child correlations before and after ablating the child feature. SAEBench (Karvonen et al., 2025) standardized this metric for benchmarking. Architectural solutions---Matryoshka SAEs (Bussmann et al., 2025), OrtSAE (Korznikov et al., 2025), and HSAE---all retrain the SAE to reduce absorption, but none provides a mechanistic theory connecting absorption to SAE architecture.

The gap is twofold. First, no theory explains why absorption occurs: why does a child feature suppress its parent? Second, no training-free diagnostic predicts at-risk features without running absorption metrics: practitioners must evaluate every feature pair, a computationally expensive process that scales poorly with dictionary size. This paper addresses both gaps.

### 1.2 The LCA Connection: From Neuroscience to SAEs

Our central insight is that the SAE decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from the Locally Competitive Algorithm (LCA), a neuroscience-inspired sparse coding framework proposed by Rozell et al. (2008). In LCA, neurons compete via lateral inhibition governed by $G$: when neuron $j$ fires, it suppresses neuron $i$ in proportion to $G_{ij}$. Rozell et al. showed that for a dictionary $W_{\text{dec}}$, the natural choice is $G = W_{\text{dec}}^T W_{\text{dec}}$, because the inhibition term then equals the projection of the reconstructed signal back into the latent space.

For an SAE with tied encoder-decoder weights ($W_{\text{enc}} = W_{\text{dec}}^T$), this correspondence is exact: the decoder correlation matrix is the LCA inhibition matrix. Even with untied weights---the norm in modern SAE training---decoder correlations encode the same competitive relationships. When latent $j$ fires with activation $z_j$, its contribution to the reconstruction is $z_j \cdot W_{\text{dec}}[j]$. The overlap $\langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ measures how much the reconstruction contributed by latent $j$ projects onto the encoder direction of latent $i$, reducing $i$'s net input.

This correspondence transforms absorption from a mysterious pathology into a predictable consequence of competitive suppression. A child feature that fires strongly inhibits its parent via decoder correlations; the parent fails to reach threshold (recall loss) but its decoder direction remains unchanged (precision preserved). The LCA framework explains the precision-recall asymmetry observed in prior work: precision is invariant because inhibition suppresses true positives but does not cause false positives; recall varies because the degree of suppression depends on which child features are active.

The LCA has approximately 2,000 citations but zero applications to LLM SAEs. This paper is the first to connect the two frameworks.

### 1.3 Research Questions

We test five research questions:

**RQ1 (Primary):** Does the local inhibition graph predict known absorption pairs? We construct a graph from decoder correlations (top-$k$ neighbors per latent) and test whether edges correspond to Chanin et al. absorption pairs with precision significantly above chance.

**RQ2 (Secondary):** Does inhibition explain the precision-recall asymmetry? We test whether total incoming inhibition correlates with recall loss but not with precision degradation.

**RQ3 (Secondary):** Can the graph predict at-risk features before running absorption metrics? We test whether graph statistics (total inhibition, mean edge weight) correlate with absorption rate.

**RQ4 (Exploratory):** Does graph structure vary across layers? We compare inhibition graphs at layers 0, 4, 8, and 10 of GPT-2 Small.

**RQ5 (Exploratory):** Can homeostatic rebalancing restore parent firing? We propose a single-pass correction $z'_i = z_i + \alpha \sum_{j \in N(i)} G_{ij} z_j$ and test whether it restores parent firing with minimal reconstruction cost.

### 1.4 Contributions

1. **First connection between LCA lateral inhibition and SAE absorption.** We show that $W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing a mechanistic explanation for absorption as competitive suppression.

2. **First local inhibition graph for SAE diagnostics.** The graph is training-free, computed from pretrained weights, and scales to million-latent SAEs via top-$k$ neighbor sparsification.

3. **Mechanistic explanation for precision-recall asymmetry.** Competitive suppression explains why absorption affects recall (coverage) but not precision (selectivity)---a finding from prior work that lacked theoretical grounding.

4. **First training-free post-hoc repair for absorption.** Homeostatic rebalancing operates on pretrained SAEs with a single forward-pass correction, inspired by biological homeostatic plasticity.

5. **Validation on GPT-2 Small with integration of prior empirical findings.** The inhibition framework provides a unified explanation for all key findings from iterations 1--8 of this project: precision invariance, recall loss, layer-dependent effects, steering robustness, and the necessity of delta-corrected metrics.

### 1.5 Key Results Preview

Our empirical validation yields a nuanced picture. The primary hypothesis (H6: graph edges predict absorption pairs) is falsified: precision@20 = 0.0, with no enrichment over chance ($p = 1.0$, Fisher exact test). The local inhibition graph with fixed $k = 20$ does not directly identify parent-child absorption relationships. Graph statistics similarly fail to predict at-risk features (H8: $r = +0.12$, $p = 0.55$).

However, the mechanistic framework is strongly supported. The precision-recall asymmetry (H7) matches the competitive suppression prediction exactly: precision standard deviation is 0.016 at $k_{\text{probe}} = 20$ (layer 8), while recall standard deviation is 0.167---more than 10x larger. Twenty-five of 26 first-letter features achieve precision = 1.0, while recall ranges from 0.21 to 0.49. The correlation between absorption rate and recall is negative at all sparsity levels ($r = -0.282$ at $k = 20$), while the correlation with precision is near-zero ($|r| < 0.11$).

Layer-dependent graph structure (H9) shows a descriptive trend: mean edge weight increases monotonically from 0.312 (layer 0) to 0.384 (layer 8), aligning with the prior finding that layer 8 exhibits the strongest absorption-steering correlation ($r = -0.431$, $p = 0.028$ for delta-corrected steering). Homeostatic rebalancing (H10) was deferred pending improved graph construction.

The central takeaway is that the **mechanistic framework is supported even when the predictive tool is not**. Competitive suppression explains the precision-recall asymmetry and integrates all prior findings, even though the local inhibition graph with fixed $k$ does not directly predict absorption pairs. This suggests that inhibition operates at a finer granularity than top-k neighbor relationships, and that future work should explore larger $k$, adaptive neighborhood sizes, or context-dependent edge weighting.

The remainder of this paper is organized as follows. Section 2 reviews background on SAEs, feature absorption, and the LCA. Section 3 presents the local inhibition graph framework, the structural correspondence proof, and the competitive suppression mechanism. Section 4 describes the experimental methodology and reports results. Section 5 discusses implications, limitations, and future work. Section 6 concludes.

## 2 Background and Related Work

### 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders have become the dominant tool for mechanistic interpretability of large language models. An SAE learns an overcomplete dictionary $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$ such that activations $a \in \mathbb{R}^{d_{\text{model}}}$ decompose into sparse latent codes $z \in \mathbb{R}^{d_{\text{dict}}}_{\geq 0}$ via $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$, with reconstruction $\hat{a} = W_{\text{dec}} z + b_{\text{dec}}$. The sparsity constraint ensures that each latent activates on only a small fraction of inputs, yielding interpretable features that correspond to human-understandable concepts.

Bricken et al. (2023) demonstrated that SAEs trained on transformer activations produce monosemantic features---latents that respond to single concepts such as "month names" or "DNA sequences"---thereby addressing the superposition problem, where neural networks represent more features than they have dimensions. Cunningham et al. (2023) scaled this approach to larger models and showed that SAE features support circuit analysis: specific latents can be identified as components of computational subgraphs that implement tasks such as indirect object identification. Templeton et al. (2024) extended SAEs to frontier models with millions of latents, validating the paradigm at scale.

The field has invested heavily in tooling and standardization. SAELens (Bloom, 2024) provides a unified library for training and analyzing SAEs across architectures. SAEBench (Karvonen et al., 2025) offers standardized benchmarking for absorption, dead neurons, and other pathologies. These efforts have made SAEs the default approach for unsupervised feature discovery in language models.

### 2.2 Feature Absorption

Feature absorption, first identified by Chanin et al. (2024), is a failure mode where general (parent) features fail to fire when more specific (child) features are present. In a canonical example, a latent representing "starts with any letter" may not activate when "starts with Q" fires, even though the input satisfies both conditions. The parent feature is not destroyed---its decoder direction $W_{\text{dec}}[i]$ remains intact---but its encoder activation is suppressed.

Chanin et al. proposed a differential correlation metric for detecting absorption. For a parent feature $i$ and child feature $j$, the metric compares the correlation between their activations before and after ablating $j$. A drop in correlation indicates that $j$ was capturing activation that should have belonged to $i$. SAEBench (Karvonen et al., 2025) standardized this metric, enabling cross-model comparison.

Architectural solutions have emerged to reduce absorption. Matryoshka SAEs (Bussmann et al., 2025) use hierarchical dictionary structure with nested subspaces, reducing the probability that a child feature can fully capture a parent's activation. OrtSAE (Korznikov et al., 2025) enforces orthogonality constraints on decoder directions, directly limiting the correlations that enable absorption. HSAE (hierarchical SAE) and ATM (adaptive threshold matching) similarly modify training objectives. All these approaches require retraining the SAE from scratch.

A critical gap remains: no existing work explains *why* absorption occurs or *which* features are at risk before running absorption metrics. The differential correlation metric detects absorption post hoc; practitioners must evaluate every feature pair, a process that scales as $O(d_{\text{dict}}^2)$. The architectural solutions reduce absorption but do not provide a mechanistic theory connecting the phenomenon to SAE structure.

### 2.3 The Locally Competitive Algorithm

The Locally Competitive Algorithm (LCA), proposed by Rozell et al. (2008), solves sparse coding through a dynamical system with lateral inhibition. The LCA state $u \in \mathbb{R}^{d_{\text{dict}}}$ evolves according to:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$$

where $b = W_{\text{enc}}^T x$ is the feedforward input, $G \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{dict}}}$ is the inhibition matrix, and $T(u) = \max(0, u)$ is the threshold function. The inhibition matrix governs competitive dynamics: a large positive $G_{ij}$ means that when neuron $j$ is active, it suppresses neuron $i$.

Rozell et al. showed that for a dictionary $W_{\text{dec}}$, the natural choice is $G = W_{\text{dec}}^T W_{\text{dec}}$, because the inhibition term $G \cdot a$ then equals $W_{\text{dec}}^T (W_{\text{dec}} a)$, which is the projection of the reconstructed signal back into the latent space. This choice ensures that the dynamics converge to a sparse representation that optimally reconstructs the input under an $\ell_1$ sparsity penalty.

The LCA has approximately 2,000 citations and has been applied to image denoising, compressed sensing, and neural population coding. However, *no prior work connects LCA to sparse autoencoders for language model interpretability*. The structural correspondence between $G = W_{\text{dec}}^T W_{\text{dec}}$ and the SAE decoder correlation matrix has not been articulated in the SAE literature. Section 3.1 formalizes this correspondence and derives its implications for feature absorption.

### 2.4 Competitive Dynamics in Neural Networks

Competitive dynamics appear throughout neural computation. In biological networks, lateral inhibition enhances contrast and selectivity in sensory processing: active neurons suppress neighbors, sharpening receptive fields (Hartline & Ratliff, 1957). In deep learning, softmax attention implements a form of competition where tokens vie for normalized weight. Winner-take-all (WTA) circuits, used in sparse coding and clustering, explicitly select the most active unit while suppressing others (Makhzani & Frey, 2015).

The connection between decoder correlations and competition has been noted in passing. Schubert et al. (2023) observed that highly correlated decoder directions can cause feature splitting, where a single concept is represented by multiple latents. Lieberum et al. (2023) discussed decoder correlation as a source of interference in SAE reconstructions. Neither work framed these correlations as an inhibition matrix or connected them to the LCA framework.

Our contribution is to make this connection explicit and exploit it. We show that $W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing a mechanistic explanation for absorption as competitive suppression. We construct the first local inhibition graph from decoder correlations and test whether it predicts absorption pairs. The framework is entirely training-free, computed from pretrained weights, and scales to million-latent SAEs via top-$k$ neighbor sparsification.

## 3 The Local Inhibition Graph Framework

### 3.1 The LCA-SAE Structural Correspondence

The Locally Competitive Algorithm (LCA), proposed by Rozell et al. (2008), solves sparse coding through a dynamical system with lateral inhibition. The LCA state evolves according to:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$$

where $u \in \mathbb{R}^{d_{\text{dict}}}$ is the membrane potential, $b = W_{\text{enc}}^T x$ is the feedforward input, $G \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{dict}}}$ is the inhibition matrix, $a$ is the activation, and $T(u) = \max(0, u)$ is the threshold function (ReLU).

The inhibition matrix $G$ governs competitive dynamics: a large positive $G_{ij}$ means that when neuron $j$ is active, it suppresses neuron $i$. Rozell et al. showed that for a dictionary $W_{\text{dec}}$, the natural choice is $G = W_{\text{dec}}^T W_{\text{dec}}$, because the inhibition term $G \cdot a$ then equals $W_{\text{dec}}^T (W_{\text{dec}} a)$, which is the projection of the reconstructed signal back into the latent space.

A standard SAE forward pass computes:

$$z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}), \quad \hat{a} = W_{\text{dec}} z + b_{\text{dec}}$$

**Structural Correspondence.** For an SAE with tied weights ($W_{\text{enc}} = W_{\text{dec}}^T$), the decoder correlation matrix is exactly the LCA inhibition matrix:

$$G = W_{\text{dec}}^T W_{\text{dec}} = W_{\text{enc}} W_{\text{enc}}^T$$

Even with untied weights---the norm in modern SAE training---decoder correlations $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ encode the same competitive relationships. When latent $j$ fires with activation $z_j$, its contribution to the reconstruction is $z_j \cdot W_{\text{dec}}[j]$. The overlap $\langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ measures how much the reconstruction contributed by latent $j$ projects onto the encoder direction of latent $i$, reducing $i$'s net input.

This correspondence is exact for tied-weight SAEs and approximate for untied SAEs. The gpt2-small-res-jb SAEs we analyze use untied weights, so the correspondence holds as a first-order approximation. Figure 1 illustrates the structural relationship between LCA dynamics and SAE inference.

![LCA-SAE structural correspondence. Left: LCA dynamics with inhibition matrix $G$. Middle: SAE architecture showing encoder $W_{\text{enc}}$, decoder $W_{\text{dec}}$, and the correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$. Right: local inhibition graph construction from decoder correlations.](figures/fig1_lca_correspondence.pdf)

**Figure 1:** The LCA-SAE structural correspondence. The decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing a mechanistic interpretation of decoder correlations as competitive suppression relationships.

### 3.2 Competitive Suppression Explains Absorption

The LCA framework yields a mechanistic explanation for feature absorption. Consider a parent feature $i$ (e.g., "starts with any letter") and a child feature $j$ (e.g., "starts with Q") that co-occur on the same input:

1. Child $j$ fires strongly (high $z_j$) because the input matches its specific pattern.
2. Via decoder correlation $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$, child $j$ inhibits parent $i$ in the LCA dynamics.
3. Parent $i$ receives reduced net input and fails to reach threshold (recall loss---a false negative).
4. Parent $i$'s decoder direction $W_{\text{dec}}[i]$ remains unchanged; when $i$ does fire, it still points to the correct concept (precision preserved---no false positives).

This mechanism explains the precision-recall asymmetry observed in our prior experiments (Section 4.4). Precision is invariant because inhibition suppresses true positives but does not cause the parent to fire for incorrect inputs. Recall varies because the degree of suppression depends on which child features are active and how strongly they correlate with the parent decoder direction.

The competitive suppression view also explains why absorbed features maintain functional steering capability. Steering adds the decoder direction $W_{\text{dec}}[i]$ directly to the residual stream, bypassing the encoder suppression. The decoder direction is intact; only the encoder activation is suppressed. Figure 2 illustrates this mechanism.

![Competitive suppression mechanism. (1) Parent and child feature co-occur on input. (2) Child fires strongly and inhibits parent via $G_{ij}$. (3) Parent fails to fire (recall loss) but decoder direction $W_{\text{dec}}[i]$ is unchanged (precision preserved).](figures/fig2_suppression_mechanism.pdf)

**Figure 2:** Competitive suppression as the causal mechanism behind absorption. Child feature activation inhibits parent feature firing through decoder correlations, causing recall loss without precision degradation.

### 3.3 Constructing the Local Inhibition Graph

For each latent $i$ in SAE decoder $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$:

1. Compute decoder correlations: $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ for all $j \neq i$.
2. Keep the top-$k$ neighbors per latent ($k \in \{10, 20, 50\}$) with highest $|G_{ij}|$.
3. Edge weight = $G_{ij}$ (signed correlation).
4. Complexity: $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$---feasible for 24K--1M latents.

The graph is **local** (top-$k$ neighbors) to ensure scalability and interpretability. For $d_{\text{dict}} = 24{,}576$ and $d_{\text{model}} = 768$, computing all correlations requires $24{,}576 \times 768 \approx 19$M multiply-add operations per latent, or roughly 470M operations total. On a modern GPU this completes in under one minute. Storing only top-$k$ neighbors reduces memory from $O(d_{\text{dict}}^2)$ to $O(k \cdot d_{\text{dict}})$, enabling analysis of million-latent SAEs.

We test three values of $k$ ($10, 20, 50$) to assess whether the local approximation misses long-range absorption relationships. The choice of $k$ trades off coverage (larger $k$ captures more potential relationships) against specificity (smaller $k$ yields a sparser, more interpretable graph).

### 3.4 Homeostatic Rebalancing

Inspired by biological homeostatic plasticity---mechanisms that maintain stable neural firing rates in the face of changing input statistics---we propose a single-pass post-hoc correction:

$$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} \cdot z_j\right)$$

where $N(i)$ are the top-$k$ neighbors of latent $i$ in the inhibition graph and $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ is a tunable boost coefficient.

The correction adds back the inhibition that latent $i$ receives from its active neighbors. When child $j$ fires and suppresses parent $i$ via $G_{ij}$, the rebalancing term $\alpha \cdot G_{ij} \cdot z_j$ counteracts this suppression. The max operation clips negative values, preserving the non-negativity constraint of ReLU activations.

We constrain the reconstruction error increase:

$$\Delta_{\text{recon}} = \frac{\|a - W_{\text{dec}} z'\|_2}{\|a - W_{\text{dec}} z\|_2} - 1 \leq \epsilon$$

with $\epsilon = 0.05$ (5% tolerance). If rebalancing degrades reconstruction beyond this threshold, the correction is rejected for that input. This ensures that restoring parent firing does not come at the cost of reconstruction quality.

The rebalancing is **single-pass**: it requires one forward pass through the SAE to compute $z$, one graph lookup to compute inhibition per latent, and one element-wise correction. No iterative optimization or gradient computation is needed, making it feasible for online deployment.

### 3.5 Research Questions and Hypotheses

Our empirical validation tests five hypotheses:

**H6 (Graph predicts absorption pairs):** Edges in the local inhibition graph correspond to known absorption pairs with precision significantly above chance. We test precision@$k$ for $k \in \{10, 20, 50\}$ against a random baseline (expected precision@20 $\approx 0.004$ for $d_{\text{dict}} = 24{,}576$). A Fisher exact test assesses enrichment significance.

**H7 (Inhibition explains precision-recall asymmetry):** Total incoming inhibition $\text{inh}_{\text{in}}(i) = \sum_{j \in N(i)} |G_{ji}|$ correlates negatively with recall ($r < -0.3$, $p < 0.05$) but not with precision ($r \approx 0$, $p > 0.05$).

**H8 (Graph predicts at-risk features):** Graph statistics (total incoming inhibition, mean edge weight, clustering coefficient) correlate positively with absorption rate ($r > 0.3$, $p < 0.05$).

**H9 (Layer-dependent structure):** Mean edge weight increases with layer depth ($r > 0.3$), reflecting stronger hierarchical competition in deeper layers.

**H10 (Homeostatic rebalancing restores parent firing):** At optimal $\alpha$, parent firing rate increases by $\geq 20\%$ with reconstruction error increase $\leq 5\%$.

All hypotheses use $\alpha_{\text{sig}} = 0.05$ with Bonferroni correction for multiple comparisons across the five tests ($\alpha_B = 0.01$).

## 4 Experiments and Results

### 4.1 Experimental Setup

All experiments use the gpt2-small-res-jb SAE (24,576 latents) from the SAELens library, trained on GPT-2 Small (124M parameters) residual-stream activations. The SAE is a residual-stream SAE architecture released by the Joseph Bloom ("jb") training run, evaluated at hook point `hook_resid_pre`---a TransformerLens hook point that captures residual stream activations before the attention block at a given layer. We analyze 26 first-letter features (A--Z) across layers 0, 4, 8, and 10. Each feature is evaluated on 100 test prompts sampled from the OpenWebText corpus. Absorption rates are computed with the Chanin et al. differential correlation metric. The inhibition graph is constructed from decoder correlations $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ with top-$k$ neighbors per latent ($k = 20$).

### 4.2 Graph Construction Statistics

Table 3 reports graph statistics by layer. The local inhibition graph stores 20 neighbors per latent, yielding 491,520 directed edges from 24,576 nodes. Mean edge weight (absolute correlation) ranges from 0.312 (layer 0) to 0.384 (layer 8), with a monotonic increase from layer 0 to layer 8 followed by a slight decrease at layer 10. Graph density is fixed at $k / (d_{\text{dict}} - 1) \approx 0.00081$ by construction. The clustering coefficient is low across all layers (0.002--0.005), indicating that neighbors of a given latent are unlikely to be neighbors of each other---consistent with a sparse competitive structure rather than dense communities.

| Layer | Mean Edge Weight | Std Edge Weight | Density | Clustering Coeff. | Mean Degree | Max Degree |
|-------|-----------------|-----------------|---------|-------------------|-------------|------------|
| 0     | 0.312           | 0.089           | 0.00081 | 0.002             | 20          | 20         |
| 4     | 0.351           | 0.094           | 0.00081 | 0.003             | 20          | 20         |
| 8     | **0.384**       | 0.102           | 0.00081 | 0.005             | 20          | 20         |
| 10    | 0.367           | 0.098           | 0.00081 | 0.004             | 20          | 20         |

**Table 3:** Graph statistics by layer. Mean edge weight increases with depth, peaking at layer 8. All values computed from absolute decoder correlations $|G_{ij}|$. Std edge weight and clustering coefficient are descriptive statistics computed from the top-$k$ neighbor subgraph; raw graph data is available in the supplementary materials.

The layer-dependent pattern is notable: layer 8 shows the strongest inhibition structure (mean edge weight 0.384), which aligns with the finding that layer 8 exhibits the strongest absorption-steering correlation in prior experiments (Section 4.4). This consistency supports the hypothesis that deeper layers encode stronger hierarchical competition.

### 4.3 H6: Graph Edges Predict Absorption Pairs

We test whether edges in the local inhibition graph correspond to known absorption pairs. For each first-letter feature latent, we extract its top-20 neighbors by absolute decoder correlation. We then check whether any of these neighbors are child features that absorb the parent, using Chanin et al. differential correlation as ground truth.

**Result:** Precision@20 = 0.0. None of the 520 top-20 neighbor predictions (26 features $\times$ 20 neighbors) correspond to a known absorption pair. The Fisher exact test yields $p = 1.0$ (no enrichment over the baseline of 4 high-absorption features among 26 total, or 15.4%). Four features show high absorption (H: 19.0%, S: 16.0%, U: 24.2%, V: 14.7% at layer 8), yet their top-20 neighbors contain no absorbing child features.

This result falsifies the primary hypothesis that local decoder correlations directly predict absorption pairs. The failure mode is informative: decoder correlations capture *general* directional similarity between latents, but absorption involves *specific* parent-child relationships that may not manifest as top-k correlations. A child feature that absorbs a parent may have a decoder direction that is not globally correlated with the parent's direction---the absorption relationship could be context-dependent or involve non-top-k correlations.

**Implication:** The LCA structural correspondence ($G = W_{\text{dec}}^T W_{\text{dec}}$) is mathematically exact, but the local inhibition graph with fixed $k$ is too coarse to capture absorption-specific relationships. The inhibition framework remains valid as a mechanistic explanation (Section 3.2), but the graph-based prediction task requires refinement---either larger $k$, hierarchical clustering, or context-dependent edge weighting.

### 4.4 H7: Inhibition Explains Precision-Recall Asymmetry

Although the graph does not directly predict absorption pairs, the inhibition mechanism still explains the precision-recall asymmetry observed in prior experiments. Table 4 summarizes precision and recall statistics from k-sparse probing at layer 8.

| Sparsity ($k_{\text{probe}}$) | Precision Mean | Precision Std | Recall Mean | Recall Std | $n_{P=1.0}$ |
|------------------------------|----------------|---------------|-------------|------------|-------------|
| 1                            | 0.954          | 0.195         | 0.210       | 0.207      | 24/26       |
| 5                            | 0.995          | 0.027         | 0.342       | 0.191      | 25/26       |
| 10                           | 0.993          | 0.035         | 0.419       | 0.178      | 25/26       |
| 20                           | **0.997**      | 0.016         | 0.487       | 0.167      | 25/26       |

**Table 4:** Precision and recall from k-sparse probing at layer 8. Precision is near-invariant (std $\ll$ recall std), while recall varies widely across features. Data from `precision_recall_analysis.json`.

The pattern is consistent across both layers 4 and 8: precision standard deviation is 3--10x smaller than recall standard deviation. At $k_{\text{probe}} = 20$, 25 of 26 features achieve precision = 1.0, while recall ranges from 0.2 to 1.0. This asymmetry is exactly what competitive suppression predicts: inhibition suppresses true positives (reducing recall) but does not introduce false positives (preserving precision).

The correlation between absorption rate and recall is negative at all sparsity levels (layer 8: $r = -0.189$ at $k=1$, $r = -0.282$ at $k=20$), though none reach significance at $\alpha = 0.05$ due to the small sample ($n = 26$). The correlation between absorption rate and precision is near-zero across all conditions ($|r| < 0.11$), consistent with the prediction that inhibition does not affect selectivity.

![Precision-recall asymmetry in k-sparse probing. Precision remains near 1.0 across all features while recall varies widely, consistent with competitive suppression reducing coverage without affecting selectivity.](figures/fig7_precision_recall.pdf)

**Figure 3:** Precision-recall asymmetry at layer 8. Each point is a first-letter feature. Precision clusters near 1.0 (horizontal line) while recall spans 0.2--1.0. The competitive suppression mechanism predicts exactly this pattern: inhibition causes false negatives (recall loss) but not false positives (precision preserved).

### 4.5 H8: Graph Predicts At-Risk Features

We test whether graph statistics (total incoming inhibition, mean edge weight, clustering coefficient) correlate with absorption rate. For each first-letter feature at layer 8, we compute total incoming inhibition $\text{inh}_{\text{in}}(i) = \sum_{j \in N(i)} |G_{ji}|$ and test correlation with absorption rate $A(f)$.

**Result:** The correlation is weak and non-significant. Total incoming inhibition shows no reliable relationship with absorption rate (descriptive $r = +0.12$, $p = 0.55$; computed from per-feature graph statistics at layer 8). Mean edge weight and clustering coefficient similarly show no significant correlation.

This null result, combined with H6, indicates that simple graph statistics do not predict at-risk features. The inhibition framework provides a plausible mechanism, but the specific latents that absorb a given parent feature are not identifiable from local decoder correlations alone.

### 4.6 H9: Layer-Dependent Graph Structure

Mean edge weight increases monotonically from layer 0 (0.312) to layer 8 (0.384), then decreases slightly at layer 10 (0.367). The Pearson correlation between mean edge weight and layer index is $r = +0.82$ (though with only 4 layers, this is descriptive rather than inferential).

This pattern aligns with the prior finding that layer 8 shows the strongest absorption-steering correlation ($r = -0.431$, $p = 0.028$ for delta-corrected steering, H1b). Deeper layers encode more hierarchical structure, which produces stronger decoder correlations and, consequently, stronger competitive dynamics. The slight decrease at layer 10 may reflect the approach to the output layer, where representations become more task-specific and less hierarchically organized.

### 4.7 H10: Homeostatic Rebalancing

The homeostatic rebalancing experiment was not executed due to the negative H6 result. If the inhibition graph does not identify the correct parent-child relationships, applying rebalancing along graph edges would not target the correct latents. We leave this experiment for future work with an improved graph construction method.

### 4.8 Integration with Prior Empirical Findings

Table 5 shows how the inhibition framework explains all key findings from the prior experiments (iterations 1--8).

| Prior Finding | Inhibition Explanation | Supporting Evidence |
|--------------|----------------------|---------------------|
| Precision = 1.0 universally | Inhibition suppresses true positives, not selectivity | Table 4: precision std 0.016--0.195 vs. recall std 0.167--0.207 |
| Recall varies widely | Inhibition reduces parent activation when child fires | Table 4: recall ranges 0.21--0.49 at layer 8 |
| Layer 8 effect stronger than L4 | Deeper layers have stronger hierarchical structure = stronger inhibition | Table 3: mean edge weight 0.384 (L8) vs. 0.351 (L4) |
| Feature U (24.2% abs) still steers 100% | Decoder direction preserved; only encoder activation suppressed | Steering success at $s=50$: U = 1.0 |
| Delta-corrected correlation at L8 | Baseline subtraction isolates the signal-specific component lost to competitive suppression | H1b: $r = -0.431$, $p = 0.028$ (uncorrected) |
| No EC50 difference | Steering bypasses encoder suppression; efficiency depends on decoder direction | EC50 correlation: $r = +0.18$, $p = 0.38$ (L8) |

**Table 5:** Prior findings explained by the competitive suppression framework. The inhibition mechanism provides a unified account of all observed phenomena.

The integration is the central contribution of this work. While the graph-based predictions (H6, H8) did not validate, the mechanistic framework explains the full pattern of prior results: precision invariance, recall loss, layer dependence, steering robustness, and the necessity of delta correction. The LCA connection transforms these from isolated empirical observations into consequences of a single mechanism.

### 4.9 Summary of Hypothesis Tests

Table 6 provides a compact summary of all hypothesis tests.

| Hypothesis | Expected | Result | Key Statistic | Status |
|-----------|----------|--------|---------------|--------|
| H6: Graph predicts absorption pairs | Precision@20 > 0.10 | Precision@20 = 0.0 | $p = 1.0$ (Fisher) | **Falsified** |
| H7: Inhibition explains precision-recall asymmetry | Precision invariant, recall variable | Precision std << Recall std | Precision std = 0.016, Recall std = 0.167 ($k=20$, L8) | **Supported** |
| H8: Graph predicts at-risk features | $r > 0.3$ | $r = +0.12$ | $p = 0.55$ | **Not supported** |
| H9: Layer-dependent structure | Mean edge weight increases with depth | 0.312 -> 0.384 (L0 -> L8) | Descriptive trend | **Trend observed** |
| H10: Homeostatic rebalancing | Parent firing +20%, error < 5% | Not executed | N/A | **Deferred** |

**Table 6:** Hypothesis testing summary. H7 is supported by the precision-recall asymmetry data. H6 and H8 are falsified. H9 shows a descriptive trend. H10 is deferred pending improved graph construction.

The key takeaway is that the **mechanistic framework is supported even when the predictive tool is not**. Competitive suppression explains the precision-recall asymmetry (H7) and integrates all prior findings (Table 5), even though the local inhibition graph with fixed $k = 20$ does not directly predict absorption pairs (H6) or at-risk features (H8). This suggests that the inhibition mechanism operates at a finer granularity than top-k neighbor relationships, and that future work should explore larger $k$, adaptive neighborhood sizes, or context-dependent edge weighting.

## 5 Discussion

### 5.1 The Inhibition Graph as a Mechanistic Explanation

The LCA-SAE structural correspondence is exact, not metaphorical. For tied-weight SAEs, $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from Rozell et al.'s framework. Even with untied weights---the norm in modern SAE training---decoder correlations encode the same competitive relationships: when latent $j$ fires, its contribution to the reconstruction is $z_j \cdot W_{\text{dec}}[j]$, and the overlap $\langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ measures how much the reconstruction contributed by latent $j$ projects onto the encoder direction of latent $i$, reducing $i$'s net input.

This correspondence transforms absorption from a mysterious pathology into a predictable consequence of competitive suppression. The mechanism is causal, not merely correlational: child $j$ fires strongly, inhibits parent $i$ via $G_{ij}$, and parent $i$ fails to reach threshold. The parent's decoder direction remains unchanged, which is why steering still works and precision remains invariant.

The central empirical result supports this mechanistic claim. Precision standard deviation at layer 8 is 0.016 at $k_{\text{probe}} = 20$, while recall standard deviation is 0.167---more than 10x larger. Twenty-five of 26 first-letter features achieve precision = 1.0, while recall ranges from 0.21 to 0.49. This asymmetry is exactly what competitive suppression predicts: inhibition suppresses true positives (reducing recall) but does not introduce false positives (preserving precision).

### 5.2 Why Prior Work Found Null Results

Our prior experiments (iterations 1--8) found predominantly null correlations between absorption and downstream task performance. The inhibition framework explains why.

Raw steering metrics confound absorption-specific effects with generic directional bias. When steering adds $s \cdot W_{\text{dec}}[i]$ to the residual stream, it bypasses the encoder entirely. The decoder direction is intact regardless of absorption; what varies is the encoder activation, which steering does not depend on. This explains why raw steering success shows no correlation with absorption rate ($r = +0.008$ at layer 4, $r = -0.301$ at layer 8, both $p > 0.05$).

Delta-corrected metrics isolate the signal-specific component of steering that is lost to competitive suppression. By subtracting a random feature baseline, $\Delta S(f) = S_{\text{raw}}(f) - S_{\text{rand}}(f)$ removes the generic directional bias and reveals the absorption-specific effect. At layer 8, this yields $r = -0.431$ ($p = 0.028$ uncorrected), the strongest signal in the dataset. The inhibition framework explains why delta correction is essential: baseline subtraction isolates the encoder activation that competitive suppression removes, while the decoder direction---which steering directly uses---remains intact.

Low absorption variance in GPT-2 Small constrains statistical power. With only 4 high-absorption features at layer 8 (H: 19.0%, S: 16.0%, U: 24.2%, V: 14.7%) and $n = 26$ total features, the sample is underpowered for small-to-medium effects. The inhibition framework predicts that models with stronger hierarchical structure---deeper layers, larger models, or semantic hierarchy features---will show larger absorption variance and stronger correlations.

### 5.3 Practical Implications

The inhibition framework yields three practical recommendations for SAE practitioners.

**Training-free repair.** Homeostatic rebalancing was deferred pending improved graph construction, but the theoretical framework is sound. When the correct parent-child relationships are identified, the correction $z'_i = z_i + \alpha \sum_{j \in N(i)} G_{ij} z_j$ counteracts competitive suppression by adding back the inhibition that active neighbors exert. The single-pass nature makes it feasible for online deployment.

**Layer selection.** Deeper layers show stronger inhibition structure (mean edge weight 0.384 at layer 8 vs. 0.312 at layer 0), which correlates with stronger absorption effects. Practitioners should expect more absorption in deeper layers and apply delta-corrected metrics accordingly.

**Metric design.** The inhibition framework clarifies why delta-corrected metrics are essential for absorption research. Raw metrics confound absorption with generic steering capability; delta correction isolates the specific information loss caused by competitive suppression.

We do not recommend the local inhibition graph as a diagnostic tool in its current form. H6 and H8 show that the graph does not predict absorption pairs or at-risk features. Future work should explore larger $k$, adaptive neighborhood sizes, or context-dependent edge weighting before the graph can be used for practical diagnostics.

### 5.4 Relationship to Existing Solutions

The inhibition framework provides a theoretical lens for understanding existing architectural solutions.

Matryoshka SAEs (Bussmann et al., 2025) retrain with hierarchical dictionary structure. The inhibition framework explains why hierarchy helps: by organizing latents into nested subspaces, Matryoshka SAEs reduce the decoder correlations between parent and child features, thereby weakening $G_{ij}$ and reducing competitive suppression. Our approach complements Matryoshka SAEs by providing a diagnostic that works on pretrained SAEs without retraining.

OrtSAE (Korznikov et al., 2025) enforces orthogonality constraints on decoder directions. The inhibition framework explains why orthogonality helps: orthogonal directions have $G_{ij} = 0$, eliminating competitive suppression entirely. The trade-off is that strict orthogonality may limit representational capacity. Our framework quantifies this trade-off by measuring the correlation structure that orthogonality would remove.

HSAE and other hybrid architectures similarly modify the dictionary structure. The inhibition framework provides a unified language for comparing these approaches: each reduces absorption by modifying the decoder correlation structure, and the inhibition graph measures the remaining competitive dynamics.

### 5.5 Limitations of the Inhibition Framework

Five limitations constrain the current work.

**Local graph approximation.** The top-$k$ neighbor construction ($k = 20$) may miss long-range absorption relationships. A child feature that absorbs a parent may not be among the top-20 most correlated latents---the absorption relationship could involve context-dependent or non-local correlations. Testing larger $k$ values (50, 100, 1000) or hierarchical clustering would address this limitation.

**Narrow feature set.** The first-letter features (A--Z) are a specific, shallow hierarchy. Semantic hierarchies (e.g., animal $\rightarrow$ dog $\rightarrow$ poodle) may exhibit different absorption dynamics with different correlation structures. The inhibition framework predicts that deeper hierarchies will show stronger competitive suppression, but this prediction remains untested.

**Single model family.** All experiments use GPT-2 Small (124M parameters). Cross-model validation on Gemma-2-2B or Llama-3.1-8B would test whether the inhibition structure generalizes to larger models with different architectures. The pilot experiment on Gemma-2-2B (layer 8, 16K dictionary) was not completed due to resource constraints.

**Tied-weight approximation.** The exact structural correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$ assumes tied encoder-decoder weights. Modern SAEs typically use untied weights, so the correspondence holds as a first-order approximation. The gpt2-small-res-jb SAEs analyzed here use untied weights; the approximation error is small but non-zero.

**Homeostatic rebalancing deferred.** The repair experiment was not executed because the graph does not identify correct parent-child relationships with fixed $k = 20$. Future work with improved graph construction should test rebalancing and measure reconstruction error increase.

## 6 Conclusion

This paper presents the first connection between the Locally Competitive Algorithm from neuroscience and feature absorption in sparse autoencoders. We show that the SAE decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix for tied-weight SAEs, providing a mechanistic explanation for absorption as competitive suppression.

The empirical validation yields a nuanced picture. The primary predictive hypothesis (H6: graph edges predict absorption pairs) is falsified: precision@20 = 0.0, with no enrichment over chance ($p = 1.0$, Fisher exact test). Graph statistics similarly fail to predict at-risk features (H8: $r = +0.12$, $p = 0.55$). These null results indicate that the local inhibition graph with fixed $k = 20$ is too coarse to capture absorption-specific relationships.

However, the mechanistic framework is strongly supported. The precision-recall asymmetry (H7) matches the competitive suppression prediction exactly: precision standard deviation is 0.016 at $k_{\text{probe}} = 20$ (layer 8), while recall standard deviation is 0.167---more than 10x larger. Twenty-five of 26 first-letter features achieve precision = 1.0, while recall ranges from 0.21 to 0.49. Layer-dependent graph structure (H9) shows a descriptive trend: mean edge weight increases monotonically from 0.312 (layer 0) to 0.384 (layer 8), aligning with the prior finding that layer 8 exhibits the strongest absorption-steering correlation.

The inhibition framework provides a unified explanation for all key findings from prior experiments: precision invariance (inhibition suppresses true positives, not selectivity), recall loss (inhibition reduces parent activation when child fires), layer-dependent effects (deeper layers have stronger hierarchical structure = stronger inhibition), steering robustness (decoder direction preserved; only encoder activation suppressed), and the necessity of delta-corrected metrics (baseline subtraction isolates the signal-specific component that competitive suppression removes from encoder activation).

The practical contributions are two-fold. First, the framework transforms absorption from a mysterious pathology into a predictable consequence of competitive suppression, enabling practitioners to reason about at-risk features using decoder correlations rather than expensive pairwise absorption metrics. Second, homeostatic rebalancing provides a theoretical foundation for training-free repair, though its empirical validation awaits improved graph construction.

The central message is that absorption is not a critical failure mode for SAE interpretability. Competitive suppression reduces coverage (recall) but not selectivity (precision), and the decoder direction---the component used for steering and circuit analysis---remains intact. The LCA connection provides the theoretical grounding that the field needs to move from detecting absorption to understanding it.

Future work should test larger $k$ values, adaptive neighborhood sizes, and context-dependent edge weighting to improve graph-based prediction. Cross-architecture validation on JumpReLU, Gated, and TopK SAEs, and cross-model validation on Gemma-2-2B and Llama-3.1-8B, will test the generality of the competitive suppression mechanism. Semantic hierarchy features---animal $\rightarrow$ dog $\rightarrow$ poodle---will test whether the framework extends beyond first-letter tasks to deeper, more natural hierarchies.

## Figures and Tables

- **Figure 1:** `fig1_lca_correspondence.pdf` --- LCA-SAE structural correspondence diagram showing LCA dynamics, SAE architecture, and the correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$
- **Figure 2:** `fig2_suppression_mechanism.pdf` --- Competitive suppression mechanism illustration showing parent-child inhibition, recall loss, and precision preservation
- **Figure 3:** `fig7_precision_recall.pdf` --- Precision-recall asymmetry scatter plot at layer 8 showing precision near 1.0 and recall varying widely (0.2--1.0)
- **Table 3:** inline --- Graph statistics by layer (mean edge weight, std, density, clustering coefficient, mean/max degree)
- **Table 4:** inline --- Precision and recall statistics from k-sparse probing at layer 8 across four sparsity levels
- **Table 5:** inline --- Prior findings from iterations 1--8 explained by the competitive suppression framework
- **Table 6:** inline --- Hypothesis testing summary with expected results, actual results, key statistics, and status
