# Notation Table

This document defines all mathematical symbols and notation used throughout the paper. All section writers must reference this file for consistency.

---

## Sets and Spaces

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathcal{D}$ | Training dataset (OpenWebText samples) | $N$ sequences |
| $\mathcal{V}$ | Vocabulary of tokens | $|\mathcal{V}|$ tokens |
| $\mathcal{F}$ | Set of SAE features (latent dimensions) | $|\mathcal{F}| = d_{\text{SAE}}$ |
| $\mathcal{C}$ | Set of concept tokens tested (numbers, punctuation, case) | $|\mathcal{C}|$ concepts |
| $\mathcal{H}$ | Set of concept hierarchies | $\{\text{numbers}, \text{punctuation}, \text{case}\}$ |
| $\mathbb{R}$ | Real numbers | -- |
| $\mathbb{R}^d$ | $d$-dimensional real vector space | -- |

---

## Model Components

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $f_\theta$ | Language model with parameters $\theta$ | -- |
| $\text{SAE}$ | Sparse autoencoder | encoder + decoder |
| $W_{\text{enc}}$ | SAE encoder weight matrix | $d_{\text{SAE}} \times d_{\text{model}}$ |
| $W_{\text{dec}}$ | SAE decoder weight matrix | $d_{\text{model}} \times d_{\text{SAE}}$ |
| $b_{\text{enc}}$ | SAE encoder bias | $d_{\text{SAE}}$ |
| $b_{\text{dec}}$ | SAE decoder bias | $d_{\text{model}}$ |
| $d_{\text{model}}$ | Model hidden dimension | 768 (GPT-2 Small) |
| $d_{\text{SAE}}$ | SAE latent dimension | 24,576 (gpt2-small-res-jb) |
| $L$ | Number of model layers | 12 |
| $h_{\ell}$ | Residual stream activation at layer $\ell$ | $\mathbb{R}^{d_{\text{model}}}$ |

---

## Feature Activations

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $z_i$ | Activation of SAE feature $i$ (latent) | scalar, $z_i \geq 0$ |
| $z \in \mathbb{R}^{d_{\text{SAE}}}$ | Full SAE activation vector | $\mathbb{R}^{d_{\text{SAE}}}$ |
| $\text{topK}(c)$ | Set of top-K activating features for concept $c$ | $\subseteq \mathcal{F}$, $|\text{topK}(c)| = K$ |
| $K$ | Number of top features considered | $K = 5$ (default) |
| $a_{i,t}$ | Activation of feature $i$ on token $t$ | scalar |
| $\bar{a}_i$ | Mean activation of feature $i$ across corpus | scalar |
| $\sigma_i$ | Standard deviation of activation for feature $i$ | scalar |
| $\phi_{ij}$ | Phi coefficient (Pearson correlation of binary activations) between features $i$ and $j$ | $[-1, 1]$ |

---

## Absorption and Hierarchy

| Symbol | Definition | Range |
|--------|-----------|-------|
| $c$ | A concept token (e.g., "three", ".") | -- |
| $c_p$ | Parent concept (broad category, e.g., "number") | -- |
| $c_c$ | Child concept (specific instance, e.g., "three") | -- |
| $\mathcal{A}(c)$ | Set of features that absorb concept $c$ | $\subseteq \mathcal{F}$ |
| $A(c_1, c_2)$ | Absorption rate between concepts $c_1$ and $c_2$ | $[0, 1]$ |
| $\text{CR}(c_1, c_2)$ | Collision rate (top-K overlap) between concepts | $[0, 1]$ |
| $\mathcal{GT}$ | Ground truth absorption pairs | $\subseteq \mathcal{C} \times \mathcal{C}$ |

---

## Metrics

| Symbol | Definition | Formula |
|--------|-----------|---------|
| $\text{Jaccard}(S_1, S_2)$ | Jaccard similarity of two sets | $\frac{|S_1 \cap S_2|}{|S_1 \cup S_2|}$ |
| $A(c_1, c_2)$ | Absorption rate (Jaccard of top-K sets) | $\text{Jaccard}(\text{topK}(c_1), \text{topK}(c_2))$ |
| $\text{CR}(c_1, c_2)$ | Collision rate (synonym for absorption rate in this paper) | $A(c_1, c_2)$ |
| $\rho$ | Spearman rank correlation coefficient | $[-1, 1]$ |
| $r$ | Pearson correlation coefficient | $[-1, 1]$ |
| $p$ | Statistical p-value | $[0, 1]$ |
| $\text{CI}_{95}$ | 95% confidence interval | $[L, U]$ |
| $P$ | Precision | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ |
| $R$ | Recall | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ |
| $F_1$ | F1 score (harmonic mean of precision and recall) | $2 \cdot \frac{P \cdot R}{P + R}$ |
| $\text{TP}$ | True positives | count |
| $\text{FP}$ | False positives | count |
| $\text{FN}$ | False negatives | count |
| $\text{TN}$ | True negatives | count |

---

## UAD Pipeline

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\text{UAD}$ | Unsupervised Absorption Detection method | From Chanin et al. (2024) |
| $\mathcal{F}_{\text{dead}}$ | Set of dead features (near-zero variance) | filtered out |
| $M_{\text{cooc}}$ | Co-occurrence matrix (phi coefficients) | $|\mathcal{F}| \times |\mathcal{F}|$ |
| $\mathcal{C}_{\text{clusters}}$ | Set of clusters from hierarchical clustering | $|\mathcal{C}_{\text{clusters}}| = 50$ |
| $\mathcal{P}_{\text{detected}}$ | Set of detected absorption pairs | $\subseteq \mathcal{F} \times \mathcal{F}$ |
| $N_{\text{clusters}}$ | Number of clusters | 50 |
| $N_{\text{features}}$ | Number of features analyzed | 504 |

---

## Subscripts and Superscripts Convention

| Pattern | Meaning | Example |
|---------|---------|---------|
| $i, j$ | Feature indices | $z_i, z_j$ |
| $t$ | Token index | $a_{i,t}$ |
| $n$ | Sequence/sample index | $x_n$ |
| $\ell$ | Layer index | $h_\ell$ |
| $c$ | Concept index | $\text{topK}(c)$ |
| $k$ | Top-K rank index | $z^{(k)}$ (k-th largest activation) |
| $*$ | Optimal/best value | $F_1^*$ |
| $\hat{\cdot}$ | Estimated/predicted value | $\hat{y}$ |

---

## Constants and Hyperparameters

| Symbol | Value | Description |
|--------|-------|-------------|
| $K$ | 5 | Top-K features per concept |
| $N_{\text{clusters}}$ | 50 | Number of clusters in UAD |
| $N_{\text{seq}}$ | 1,000 | Number of sequences in dataset |
| $N_{\text{bootstrap}}$ | 1,000 | Bootstrap resamples for CI |
| $\text{seed}$ | 42 | Random seed (all experiments) |
| $\text{probe\_threshold}$ | 0.1 | Activation threshold for feature detection |

---

## Abbreviations Used in Math Context

| Abbreviation | Expansion | Context |
|-------------|-----------|---------|
| SAE | Sparse Autoencoder | model architecture |
| UAD | Unsupervised Absorption Detection | method name |
| GT | Ground Truth | evaluation |
| TP/FP/FN/TN | True/False Positives/Negatives | classification metrics |
| CI | Confidence Interval | statistics |
| res-jb | residual JumpReLU | SAE architecture variant |

---

## Important Distinctions

1. **Absorption rate vs Collision rate**: In this paper, these are the same quantity: $A(c_1, c_2) = \text{CR}(c_1, c_2) = \text{Jaccard}(\text{topK}(c_1), \text{topK}(c_2))$. The two terms are used interchangeably to emphasize different aspects (absorption = theoretical concept, collision = operational measurement).

2. **Operationalization vs Proxy**: We use "operationalization" to mean "how we define and measure absorption in practice." We do NOT claim collision rate is an independent "proxy" validated against external ground truth. Both metrics are computed from the same top-K feature sets.

3. **Token-level vs Sequence-level**: "Token-level mutual exclusivity" means two features never activate on the same token position. "Sequence-level" would allow co-occurrence within the same sequence but at different positions.

4. **Feature vs Concept**: A "feature" is a dimension in the SAE latent space (e.g., feature 24189). A "concept" is a semantic token or idea (e.g., "four"). Features may encode concepts; concepts may be encoded by multiple features.
