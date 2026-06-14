# 2. Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders decompose high-dimensional neural activations into sparse, interpretable feature representations through dictionary learning with sparsity constraints (Elhage et al., 2022). The SAE forward pass is:

$$z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}), \quad \hat{a} = W_{\text{dec}} z + b_{\text{dec}}$$

where $a \in \mathbb{R}^{d_{\text{model}}}$ is the input activation, $z \in \mathbb{R}^{d_{\text{dict}}}_{\geq 0}$ is the sparse latent representation, and $\hat{a}$ is the reconstruction. The rate-distortion objective is:

$$\min_{W_{\text{enc}}, W_{\text{dec}}} \mathbb{E}\left[\|a - \hat{a}\|_2^2 + \lambda \|z\|_1\right]$$

SAEs have enabled circuit analysis (O'Neill et al., 2024), feature steering (Templeton et al., 2024), and model editing applications. The dominant architectures include standard ReLU+L1 SAEs, TopK SAEs (Gao et al., 2024), JumpReLU SAEs (Rajamanoharan et al., 2024), and Gated SAEs. Dictionary sizes range from 16K to 16M latents, scaling to GPT-4 (Adamek et al., 2025).

## 2.2 Feature Absorption

**Definition**: Feature absorption occurs when a general (parent) feature $f_{\text{parent}}$ fails to fire because a specific (child) feature $f_{\text{child}}$ captures the activation. The child feature "absorbs" the parent direction in activation space.

Chanin et al. (2024) formalized this through the **differential correlation metric**: compare the correlation between parent and child features before and after ablating the child. A drop in correlation indicates absorption. They proved that absorption is a logical consequence of sparsity loss under hierarchical feature co-occurrence: when parent and child features co-occur frequently, the SAE's sparsity objective incentivizes merging their directions to reduce L0 activation.

**Detection**: The absorption rate $A(f)$ for feature $f$ is the fraction of child prompts where the parent latent does NOT fire but the child DOES. Mean absorption rates in GPT-2 Small range from 2.1-3.9% across layers, with maximum 24.2% (Feature U at L8).

**Architectural responses**: Matryoshka SAEs (Bussmann et al., 2025) use nested multi-level dictionaries to reduce absorption from 0.49 to 0.05. OrtSAE (Korznikov et al., 2025) enforces decoder orthogonality, reducing absorption by 65%. Balance Matryoshka (Chanin et al., 2025) addresses both absorption and hedging failure modes. ATM (Li et al., 2025) achieves ~40% absorption reduction via adaptive temporal masking.

However, these architectural fixes assume absorption is harmful. We question this assumption.

## 2.3 Rate-Distortion Theory and Optimal Compression

The rate-distortion perspective on SAEs treats them as lossy compressors trading reconstruction fidelity (distortion $D = \|a - \hat{a}\|_2^2$) against sparsity (rate $R = \|z\|_0$ or L1 relaxation). Chanin et al.'s Proposition 2 proves that under hierarchical co-occurrence, absorption minimizes sparsity loss. Our extension: absorption is not merely tolerated—it is the optimal strategy for minimizing rate while preserving decoder alignment.

**The key asymmetry**: The decoder direction $d_f = W_{\text{dec}}[:, f]$ is a column of $W_{\text{dec}}$ and is not directly affected by which encoder activations fire. This means precision (whether the feature fires correctly when the concept is present) is preserved even when recall (coverage of the concept) is reduced. The parent feature's decoder direction remains accurate; only its encoder activation is suppressed.

This explains the precision-recall asymmetry we observe: precision = 1.0 universally at k >= 5 (decoder alignment preserved), while recall varies 0.05-1.0 (encoder coverage reduced).

## 2.4 Prior Work on SAE Reliability

Several recent works question whether SAEs reliably capture meaningful features:

**Korznikov et al. (2026)** ("Sanity Checks"): Random and frozen SAEs achieve comparable performance to trained SAEs on downstream tasks, raising questions about whether SAE training produces meaningful structure or merely selects random directions.

**Wang et al. (ICLR 2026)**: Weak correlation (tau_b ~ 0.3) between interpretability scores and steering utility, suggesting better interpretability does not imply better practical performance.

**Kantamneni et al. (ICLR 2025)**: SAEs do not consistently outperform strong non-SAE baselines on probing tasks.

Our H10 result (trained SAE shows lower absorption than random SAE, 0.034 vs 0.278) addresses this directly: trained SAEs do learn different structure from random initialization, but the learned structure appears to reduce rather than eliminate absorption.

## 2.5 Inhibition Graph Hypothesis (Falsified in This Work)

The **Locally Competitive Algorithm (LCA)** (Rozell et al., 2008) proposes that lateral inhibition via $G = W_{\text{dec}}^T W_{\text{dec}}$ governs competitive dynamics in sparse coding. Under tied weights ($W_{\text{enc}} = W_{\text{dec}}^T$), this gives an exact structural correspondence: $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$.

We hypothesized that decoder correlations could predict absorption pairs: if feature $i$ absorbs feature $j$, their decoder directions should be similar (high $G_{ij}$) due to competition dynamics. We constructed an inhibition graph with top-20 neighbors per first-letter feature and validated against ground truth absorption pairs.

Result: precision@20 = 0.0 (0/520 predictions correct), enrichment = 0.0x, Fisher p = 1.0. The structural correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$ does not translate into predictive power. This falsification suggests either (1) the SAE uses untied weights, breaking the correspondence; (2) absorption is driven by encoder dynamics not decoder geometry; or (3) the Chanin metric captures something other than competitive suppression.

## 2.6 Summary and Positioning

Our work differs from prior approaches in three key ways:

1. **Training-free analysis**: We analyze existing pretrained SAEs rather than training new architectures. All experiments are completed—no new compute needed.

2. **Downstream-focused**: While prior work measures absorption rates, we test whether absorption actually harms steering, probing, and other practical applications.

3. **Rigorous controls**: We apply multiple comparison correction (Bonferroni and BH-FDR), include random baseline comparisons, and use cross-layer validation. Most prior absorption studies report raw correlations without correction.

Our contribution is not a new SAE architecture but a systematic empirical challenge to the assumption that absorption is a failure mode requiring architectural mitigation.

<!-- FIGURES
- None
-->