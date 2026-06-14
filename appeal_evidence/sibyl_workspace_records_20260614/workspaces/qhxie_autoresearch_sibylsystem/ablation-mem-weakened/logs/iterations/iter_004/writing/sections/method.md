# 3. The Local Inhibition Graph Framework

## 3.1 The LCA-SAE Structural Correspondence

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

## 3.2 Competitive Suppression Explains Absorption

The LCA framework yields a mechanistic explanation for feature absorption. Consider a parent feature $i$ (e.g., "starts with any letter") and a child feature $j$ (e.g., "starts with Q") that co-occur on the same input:

1. Child $j$ fires strongly (high $z_j$) because the input matches its specific pattern.
2. Via decoder correlation $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$, child $j$ inhibits parent $i$ in the LCA dynamics.
3. Parent $i$ receives reduced net input and fails to reach threshold (recall loss---a false negative).
4. Parent $i$'s decoder direction $W_{\text{dec}}[i]$ remains unchanged; when $i$ does fire, it still points to the correct concept (precision preserved---no false positives).

This mechanism explains the precision-recall asymmetry observed in our prior experiments (Section 4.4). Precision is invariant because inhibition suppresses true positives but does not cause the parent to fire for incorrect inputs. Recall varies because the degree of suppression depends on which child features are active and how strongly they correlate with the parent decoder direction.

The competitive suppression view also explains why absorbed features maintain functional steering capability. Steering adds the decoder direction $W_{\text{dec}}[i]$ directly to the residual stream, bypassing the encoder suppression. The decoder direction is intact; only the encoder activation is suppressed. Figure 2 illustrates this mechanism.

![Competitive suppression mechanism. (1) Parent and child feature co-occur on input. (2) Child fires strongly and inhibits parent via $G_{ij}$. (3) Parent fails to fire (recall loss) but decoder direction $W_{\text{dec}}[i]$ is unchanged (precision preserved).](figures/fig2_suppression_mechanism.pdf)

**Figure 2:** Competitive suppression as the causal mechanism behind absorption. Child feature activation inhibits parent feature firing through decoder correlations, causing recall loss without precision degradation.

## 3.3 Constructing the Local Inhibition Graph

For each latent $i$ in SAE decoder $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$:

1. Compute decoder correlations: $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ for all $j \neq i$.
2. Keep the top-$k$ neighbors per latent ($k \in \{10, 20, 50\}$) with highest $|G_{ij}|$.
3. Edge weight = $G_{ij}$ (signed correlation).
4. Complexity: $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$---feasible for 24K--1M latents.

The graph is **local** (top-$k$ neighbors) to ensure scalability and interpretability. For $d_{\text{dict}} = 24{,}576$ and $d_{\text{model}} = 768$, computing all correlations requires $24{,}576 \times 768 \approx 19$M multiply-add operations per latent, or roughly 470M operations total. On a modern GPU this completes in under one minute. Storing only top-$k$ neighbors reduces memory from $O(d_{\text{dict}}^2)$ to $O(k \cdot d_{\text{dict}})$, enabling analysis of million-latent SAEs.

We test three values of $k$ ($10, 20, 50$) to assess whether the local approximation misses long-range absorption relationships. The choice of $k$ trades off coverage (larger $k$ captures more potential relationships) against specificity (smaller $k$ yields a sparser, more interpretable graph).

## 3.4 Homeostatic Rebalancing

Inspired by biological homeostatic plasticity---mechanisms that maintain stable neural firing rates in the face of changing input statistics---we propose a single-pass post-hoc correction:

$$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} \cdot z_j\right)$$

where $N(i)$ are the top-$k$ neighbors of latent $i$ in the inhibition graph and $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ is a tunable boost coefficient.

The correction adds back the inhibition that latent $i$ receives from its active neighbors. When child $j$ fires and suppresses parent $i$ via $G_{ij}$, the rebalancing term $\alpha \cdot G_{ij} \cdot z_j$ counteracts this suppression. The max operation clips negative values, preserving the non-negativity constraint of ReLU activations.

We constrain the reconstruction error increase:

$$\Delta_{\text{recon}} = \frac{\|a - W_{\text{dec}} z'\|_2}{\|a - W_{\text{dec}} z\|_2} - 1 \leq \epsilon$$

with $\epsilon = 0.05$ (5% tolerance). If rebalancing degrades reconstruction beyond this threshold, the correction is rejected for that input. This ensures that restoring parent firing does not come at the cost of reconstruction quality.

The rebalancing is **single-pass**: it requires one forward pass through the SAE to compute $z$, one graph lookup to compute inhibition per latent, and one element-wise correction. No iterative optimization or gradient computation is needed, making it feasible for online deployment.

## 3.5 Research Questions and Hypotheses

Our empirical validation tests five hypotheses:

**H6 (Graph predicts absorption pairs):** Edges in the local inhibition graph correspond to known absorption pairs with precision significantly above chance. We test precision@$k$ for $k \in \{10, 20, 50\}$ against a random baseline (expected precision@20 $\approx 0.004$ for $d_{\text{dict}} = 24{,}576$). A Fisher exact test assesses enrichment significance.

**H7 (Inhibition explains precision-recall asymmetry):** Total incoming inhibition $\text{inh}_{\text{in}}(i) = \sum_{j \in N(i)} |G_{ji}|$ correlates negatively with recall ($r < -0.3$, $p < 0.05$) but not with precision ($r \approx 0$, $p > 0.05$).

**H8 (Graph predicts at-risk features):** Graph statistics (total incoming inhibition, mean edge weight, clustering coefficient) correlate positively with absorption rate ($r > 0.3$, $p < 0.05$).

**H9 (Layer-dependent structure):** Mean edge weight increases with layer depth ($r > 0.3$), reflecting stronger hierarchical competition in deeper layers.

**H10 (Homeostatic rebalancing restores parent firing):** At optimal $\alpha$, parent firing rate increases by $\geq 20\%$ with reconstruction error increase $\leq 5\%$.

All hypotheses use $\alpha_{\text{sig}} = 0.05$ with Bonferroni correction for multiple comparisons across the five tests ($\alpha_B = 0.01$).

<!-- FIGURES
- Figure 1: fig1_lca_correspondence_desc.md — LCA-SAE structural correspondence diagram showing LCA dynamics, SAE architecture, and graph construction pipeline
- Figure 6: fig6_suppression_mechanism_desc.md — Competitive suppression mechanism illustration showing parent-child inhibition and precision-recall asymmetry
- None (other visual elements)
-->
