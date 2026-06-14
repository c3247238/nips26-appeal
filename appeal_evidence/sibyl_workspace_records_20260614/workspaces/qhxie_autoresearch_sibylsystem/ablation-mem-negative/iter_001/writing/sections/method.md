# Methodology

### 3.1 Definitions and Metrics

**Feature Collision.** Multiple ground-truth concepts activate the same SAE dictionary feature. Formally, for concept set $\mathcal{C}$ and feature activation map $\phi: \mathcal{C} \rightarrow \{0,1\}^{d_{\text{SAE}}}$, a collision occurs when $|\{c \in \mathcal{C} : \phi_i(c) = 1\}| > 1$ for feature $i$.

**Collision Rate (CR).** $\text{CR} = \frac{|\{i : \exists c_1 \neq c_2, \phi_i(c_1) = \phi_i(c_2) = 1\}|}{d_{\text{SAE}}}$.

**Absorption Rate (AR).** Following Chanin et al. [2024], AR measures parent-feature suppression of child features under co-occurrence. We use this as the gold-standard metric where hierarchy labels are available.

**Specificity Score.** $\text{Spec}_i = \frac{\max_c \mathbb{1}[\phi_i(c)]}{\sum_c \mathbb{1}[\phi_i(c)]}$, measuring how exclusive a feature is to a single concept.

**Unsupervised Detection F1.** For UAD, precision $P = \frac{|\text{detected} \cap \text{true collisions}|}{|\text{detected}|}$, recall $R = \frac{|\text{detected} \cap \text{true collisions}|}{|\text{true collisions}|}$.

### 3.2 Hypotheses

We test six hypotheses:
- **H1**: Collision rates differ across SAE architectures (JumpReLU vs. TopK).
- **H2**: Higher collision rates causally impair downstream sparse probing accuracy.
- **H3**: Collision rate increases monotonically with sparsity level ($k$).
- **H4**: Collision rate varies systematically with layer depth.
- **H5**: Absorbed feature pairs can be detected without ground-truth hierarchies (UAD).
- **H6**: Absorbed features can be recovered via lightweight residual compensation (DFDA).

### 3.3 Cross-Architecture Absorption Benchmark (CAAB)

**Models.** GPT-2 Small (124M parameters, 12 layers) as the primary model. Gemma-2-2B was planned as a secondary model but GemmaScope experiments were blocked by API issues; all reported results use GPT-2 Small.

**SAE Architectures.** JumpReLU (pretrained from GemmaScope, $d_{\text{SAE}}$ = 24,576) and TopK (trained from scratch, $d_{\text{SAE}}$ = 16,384, $k$ = 50).

**Concept Sets.** 26 first-letter concepts (a--z) providing natural parent-child hierarchies (e.g., parent = "starts with vowel", children = {a, e, i, o, u}).

**Evaluation Protocol.** (1) Extract SAE features for each concept via maximum activation; (2) Compute collision rate via feature activation overlap; (3) Measure absorption rate via parent-child suppression; (4) Evaluate downstream sparse probing accuracy.

### 3.4 Unsupervised Absorption Detector (UAD)

UAD requires no ground-truth hierarchies. It operates in three steps:
1. **Co-occurrence Matrix Construction:** For each feature $i$, compute $C_{ij} = \mathbb{E}[z_i z_j]$ across a corpus sample of 10,000 tokens.
2. **Hierarchical Clustering:** Cluster features by co-occurrence similarity using Ward linkage into $n_c = 50$ clusters.
3. **Collision Detection:** Features in the same cluster with high mutual activation (top 10% of co-occurrence values) are flagged as potential collisions.

### 3.5 Dynamic Feature De-Absorption (DFDA)

DFDA is an SAE-retraining-free inference-time module. For each absorbed feature $i$, a small MLP (2-layer, 16 hidden units, ReLU activation, $\sim$97 parameters per pair) learns to predict the residual activation that would have been present without absorption:

$$\hat{r}_i = \text{MLP}_{\text{comp}}(z_{\neg i})$$

where $z_{\neg i}$ is the activation vector excluding feature $i$. The compensated reconstruction is $\hat{x}_{\text{comp}} = W_{\text{dec}}(z + \hat{r})$. Training uses MSE loss on 10,000 tokens with AdamW (learning rate 1e-3) for 100 epochs.

### 3.6 Statistical Analysis

We report Spearman rank correlation ($\rho_S$) with exact sample sizes. P-values are computed via permutation tests. Effect sizes are interpreted using Cohen's conventions: small ($|r|$ = 0.1), medium ($|r|$ = 0.3), large ($|r|$ = 0.5). With $n$ = 6 layers or $k$ values, the study has approximately 20% power to detect a medium effect size ($r$ = 0.5) at $\alpha$ = 0.05.

---
