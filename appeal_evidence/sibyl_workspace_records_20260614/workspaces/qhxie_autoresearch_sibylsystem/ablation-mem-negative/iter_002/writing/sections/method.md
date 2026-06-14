# 3. Methodology

## 3.1 Definitions and Metrics

**Feature Collision.** Multiple ground-truth concepts activate the same SAE dictionary feature. Formally, for concept set $\mathcal{C}$ and feature activation map $\phi: \mathcal{C} \rightarrow \{0,1\}^{d_{\text{SAE}}}$, a collision occurs when $|\{c \in \mathcal{C} : \phi_i(c) = 1\}| > 1$ for feature $i$.

**Collision Rate (CR).** $\text{CR} = \frac{|\{i : \exists c_1 \neq c_2, \phi_i(c_1) = \phi_i(c_2) = 1\}|}{d_{\text{SAE}}}$.

**Absorption.** Following Chanin et al. [2024], absorption measures parent-feature suppression of child features under co-occurrence. This requires hierarchy labels and is our gold-standard metric where available.

**Suppression Signal.** We introduce this concept as the defining characteristic of absorption: when parent feature $p$ and child feature $c$ co-occur, the activation of $c$ is suppressed relative to its expected activation without $p$. Formally:
$$\Delta_{\text{supp}}(c, p) = \mathbb{E}[\phi_c \mid \phi_p = 0] - \mathbb{E}[\phi_c \mid \phi_p = 1, \text{co-occur}]$$
Positive $\Delta_{\text{supp}}$ indicates absorption: the child's activation is lower when the parent is present.

## 3.2 The UAD Method (Tested and Failed)

We describe the Unsupervised Absorption Detector (UAD) as proposed in prior work, which we subsequently test and find ineffective.

UAD operates in three steps:
1. **Co-occurrence Matrix Construction:** For each feature $i$, compute $C_{ij} = \mathbb{E}[z_i z_j]$ across a corpus sample of 10,000 tokens.
2. **Hierarchical Clustering:** Cluster features by co-occurrence similarity using Ward linkage into $n_c = 50$ clusters.
3. **Collision Detection:** Features in the same cluster with high mutual activation (top 10\% of co-occurrence values) are flagged as potential collisions.

The intuition is that absorbed features should co-occur frequently and thus cluster together. Our experiments test whether this intuition holds.

## 3.3 Dynamic Feature De-Absorption (DFDA)

DFDA is an inference-time compensation module. For each absorbed feature $i$, a small MLP (2-layer, 16 hidden units, ReLU activation, $\sim$97 parameters per pair) learns to predict the residual activation that would have been present without absorption:

$$\hat{r}_i = \text{MLP}_{\text{comp}}(z_{\neg i})$$

where $z_{\neg i}$ is the activation vector excluding feature $i$. The compensated reconstruction is $\hat{x}_{\text{comp}} = W_{\text{dec}}(z + \hat{r})$. Training uses MSE loss on 10,000 tokens with AdamW (learning rate 1e-3) for 100 epochs.

## 3.4 Random Baseline

To anchor all F1 claims, we compute a random baseline: randomly select the same number of feature pairs as UAD detects, and compute F1 against known collision labels. This baseline is essential because a non-trivial F1 can arise by chance when the detection set is large.
