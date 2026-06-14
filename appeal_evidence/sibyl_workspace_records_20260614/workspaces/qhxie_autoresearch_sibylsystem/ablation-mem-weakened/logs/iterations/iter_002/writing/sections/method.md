# 3. The Local Inhibition Graph Framework

## 3.1 The LCA--SAE Structural Correspondence

Rozell et al. (2008) proposed the Locally Competitive Algorithm (LCA) for sparse coding, where neurons compete via lateral inhibition. The LCA dynamics are:

$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$$

where $u \in \mathbb{R}^{d_{\text{dict}}}$ is the membrane potential, $b = W_{\text{enc}}^T x$ is the feedforward input, $G \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{dict}}}$ is the inhibition matrix, $a$ is the activation after thresholding, and $T(u) = \max(0, u)$ is the threshold function (ReLU).

An SAE forward pass computes:

$$z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}), \quad \hat{a} = W_{\text{dec}} z + b_{\text{dec}}$$

**Theorem (Structural Correspondence).** For an SAE with tied weights ($W_{\text{enc}} = W_{\text{dec}}^T$), the decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the inhibition matrix from the LCA framework.

*Proof.* With tied weights, $W_{\text{enc}} = W_{\text{dec}}^T$, so $G = W_{\text{dec}}^T W_{\text{dec}} = W_{\text{enc}} W_{\text{enc}}^T$. The SAE ReLU is the LCA threshold function $T(u) = \max(0, u)$. The SAE forward pass $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$ approximates the LCA steady-state where $du/dt = 0$, yielding $u = b - G \cdot a$ and $a = T(u)$. Therefore, $z \approx a$ and the SAE computes the LCA fixed point. $\square$

Even with untied weights---the standard case for trained SAEs---the structural correspondence holds approximately: decoder correlations $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ still encode competitive relationships between latents, because the decoder directions that reconstruct the input must compete to explain the same variance.

**Implication.** Decoder correlations are not merely statistical patterns. They encode competitive suppression relationships: high $G_{ij}$ means latent $j$ suppresses latent $i$ when both are active.

![The LCA--SAE structural correspondence. Left: LCA dynamics with inhibition matrix $G$. Center: SAE architecture showing $W_{\text{enc}}$, $W_{\text{dec}}$, and the correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$. Right: local inhibition graph construction from top-k correlated neighbors per latent.](figures/fig1_lca_correspondence.pdf)

**Figure 1:** The LCA--SAE structural correspondence and inhibition graph construction pipeline. The decoder correlation matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ is exactly the LCA inhibition matrix, enabling a mechanistic understanding of absorption as competitive suppression.

## 3.2 Competitive Suppression Explains Absorption

The structural correspondence yields a mechanistic explanation for feature absorption. When parent feature $i$ and child feature $j$ co-occur in an input:

1. Child $j$ fires strongly (high activation $z_j$).
2. Via decoder correlation $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$, child $j$ inhibits parent $i$.
3. Parent $i$ fails to fire (recall loss---a false negative).
4. Parent's decoder direction $W_{\text{dec}}[i]$ is unchanged (precision preserved---no false positives).

This explains the precision--recall asymmetry observed in our prior experiments (H5, Section 5.3): precision is invariant because inhibition suppresses true positives but does not cause false positives; recall varies because suppression reduces the number of true positives detected.

![Competitive suppression mechanism. (1) Parent and child feature co-occur. (2) Child fires strongly, inhibits parent via $G_{ij}$. (3) Parent fails to fire (recall loss) but decoder direction remains unchanged (precision preserved). Activation bars show the before/after inhibition effect.](figures/fig6_suppression_mechanism.pdf)

**Figure 2:** Competitive suppression is the causal mechanism behind absorption's precision--recall asymmetry. When child $j$ fires, it inhibits parent $i$ via decoder correlation $G_{ij}$, causing recall loss without affecting precision.

## 3.3 Constructing the Local Inhibition Graph

For each latent $i$ in SAE decoder $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$:

1. Compute decoder correlations: $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$ for all $j \neq i$.
2. Keep top-k neighbors per latent ($k \in \{10, 20, 50\}$) with highest $|G_{ij}|$.
3. Edge weight = $G_{ij}$ (signed correlation).
4. Complexity: $O(k \cdot d_{\text{dict}} \cdot d_{\text{model}})$---feasible for 24K--1M latents on a single GPU.

The graph is **local** (top-k neighbors) to ensure scalability and interpretability. For GPT-2 Small ($d_{\text{dict}} = 24{,}576$, $d_{\text{model}} = 768$), computing all correlations takes approximately 2 seconds on an A100; extracting top-k neighbors takes negligible additional time.

## 3.4 Homeostatic Rebalancing

Inspired by biological homeostatic plasticity, we propose a single-pass post-hoc correction that boosts parent activations by the inhibition they receive from active neighbors:

$$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} \cdot z_j\right)$$

where $N(i)$ are the top-k neighbors of latent $i$ in the inhibition graph and $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ is a tunable boost coefficient. The correction is constrained by reconstruction error:

$$\|a - W_{\text{dec}} z'\|_2 \leq (1 + \epsilon) \|a - W_{\text{dec}} z\|_2$$

with $\epsilon = 0.05$ (5% tolerance). If the constraint is violated, $\alpha$ is reduced until the constraint is satisfied.

---

# 4. Methodology

## 4.1 Overview

Our approach is entirely training-free. We analyze pre-trained SAEs using SAELens, TransformerLens, and custom evaluation code. No SAE training is required. The experimental pipeline has six phases:

| Phase | Hypothesis | Task |
|-------|-----------|------|
| 1 | --- | Construct inhibition graph per layer |
| 2 | H6 | Validate graph edges against absorption pairs |
| 3 | H7 | Test inhibition--recall/precision correlations |
| 4 | H8 | Predict at-risk features from graph statistics |
| 5 | H9 | Compare graph structure across layers |
| 6 | H10 | Evaluate homeostatic rebalancing |

## 4.2 Model and SAE Configuration

We use GPT-2 Small ($M$, 124M parameters, 12 layers, $d_{\text{model}} = 768$) with the gpt2-small-res-jb SAE release ($d_{\text{dict}} = 24{,}576$ latents). We test four layer indices: $l \in \{0, 4, 8, 10\}$, all at the hook_resid_pre hook point. Layer 0 provides a near-input baseline; layers 4, 8, and 10 sample the mid-to-late network where hierarchical feature structure increases.

## 4.3 Phase 1: Graph Construction

For each layer $l$, we load the pre-trained SAE via SAELens and compute $G = W_{\text{dec}}^T W_{\text{dec}}$. We extract top-k neighbors per latent for $k \in \{10, 20, 50\}$. We record: edge weights, graph density $\rho_{\mathcal{G}}$, mean clustering coefficient $\text{CC}_{\mathcal{G}}$, and mean edge weight $\bar{G}$.

## 4.4 Phase 2: Validation Against Absorption Pairs (H6)

**Ground truth.** We use Chanin et al.'s absorption detection on 26 first-letter features ($\mathcal{L} = \{A, B, \ldots, Z\}$) from our prior experiments. For each feature $f \in \mathcal{L}$, we identify the parent latent $\phi(f)$ that maximally activates on tokens starting with $f$. For each absorption pair $(\phi(f), c)$ where child $c$ absorbs parent $\phi(f)$, we check if $c \in N(\phi(f))$.

**Metrics.** We compute precision@k, recall@k, and AUPR for $k \in \{10, 20, 50\}$. We test enrichment vs. a random baseline (shuffle latent indices; expected precision@20 $\approx 0.004$) using a Fisher exact test. We also test against a non-absorbed control: for each parent latent, we identify the top-k most correlated neighbors that are *not* absorption pairs, and compare their edge weights to true absorption pairs.

**Falsification.** H6 is not supported if precision@20 $\leq 0.05$ (the structural correspondence fails to predict absorption pairs above a lenient threshold).

## 4.5 Phase 3: Precision--Recall Asymmetry Test (H7)

For each first-letter feature $f$ at layers 4 and 8, we compute total incoming inhibition:

$$\text{inh}_{\text{in}}(f) = \sum_{j \in N(\phi(f))} |G_{j, \phi(f)}|$$

We test two correlations using Pearson $r$:

1. $\text{inh}_{\text{in}}(f)$ vs. recall at $k_{\text{probe}} = 5$ (predicted: negative).
2. $\text{inh}_{\text{in}}(f)$ vs. precision at $k_{\text{probe}} = 5$ (predicted: none).

Recall and precision data come from our prior k-sparse probing experiments (Section 5.3). H7 is supported if $r(\text{inh}, \text{recall}) < -0.3$ with $p < 0.05$ and $r(\text{inh}, \text{precision})$ is non-significant ($p > 0.05$).

## 4.6 Phase 4: At-Risk Feature Prediction (H8)

For each first-letter feature latent $\phi(f)$, we compute graph statistics: $\text{inh}_{\text{in}}(f)$, $\text{inh}_{\text{out}}(f)$, mean edge weight to neighbors, and maximum edge weight. We test Pearson correlation between each statistic and absorption rate $A(f)$. We compare top-quartile vs. bottom-quartile features by total inhibition. H8 is supported if $r > 0.3$ with $p < 0.05$.

## 4.7 Phase 5: Layer-Dependent Structure (H9)

We construct graphs for all four layers and compare mean edge weight $\bar{G}$, density $\rho_{\mathcal{G}}$, and clustering coefficient $\text{CC}_{\mathcal{G}}$. We test correlation between each statistic and layer depth using Pearson $r$. H9 is supported if $r(\bar{G}, l) > 0.3$.

## 4.8 Phase 6: Homeostatic Rebalancing (H10)

For test prompts (100 per feature, drawn from the same vocabulary list as prior experiments with seed 42), we compute original latents $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}})$. We apply rebalancing with $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$, clip negative values, and enforce the reconstruction constraint. We measure: parent firing rate before/after, reconstruction error change $\Delta_{\text{recon}}$, and the Pareto frontier of absorption improvement vs. error increase. H10 is supported if parent firing increases by $>20\%$ and $\Delta_{\text{recon}} < 5\%$ at some $\alpha$.

## 4.9 Software and Reproducibility

All experiments use Python 3.12 with SAELens (SAE loading), TransformerLens (model hooks), PyTorch (tensor operations), NumPy/SciPy (statistics), and Matplotlib (visualization). The random seed is fixed at 42. All SAEs are from publicly available releases. Code and evaluation protocol are released with the paper.

<!-- FIGURES
- Figure 1: fig1_lca_correspondence_desc.md, fig1_lca_correspondence.pdf — Architecture diagram showing LCA dynamics, SAE structure, and graph construction pipeline
- Figure 2: fig6_suppression_mechanism_desc.md, fig6_suppression_mechanism.pdf — Flow diagram showing competitive suppression mechanism with activation bars
- Table 1: inline — Six-phase experimental pipeline overview
-->