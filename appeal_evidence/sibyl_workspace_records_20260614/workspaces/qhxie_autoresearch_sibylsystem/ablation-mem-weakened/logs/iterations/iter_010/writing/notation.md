# Notation Table

## Inputs and Data

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $x$ | Input token sequence | $x \in \mathcal{V}^T$ where $\mathcal{V}$ is vocabulary, $T$ is sequence length |
| $\mathcal{D}$ | Dataset of test prompts | $|\mathcal{D}| = 2600$ (100 prompts $\times$ 26 features) |
| $h^{(l)}$ | Hidden activation at layer $l$ | $h^{(l)} \in \mathbb{R}^{d_{\text{model}}}$ |
| $a$ | Input activation to SAE (typically $h^{(l)}$) | $a \in \mathbb{R}^{d_{\text{model}}}$ |
| $d_{\text{model}}$ | Model hidden dimension | $d_{\text{model}} = 768$ (GPT-2 Small) |

## SAE Components

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\text{SAE}$ | Sparse autoencoder | Encoder + decoder pair |
| $W_{\text{enc}}$ | SAE encoder matrix | $W_{\text{enc}} \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{model}}}$ |
| $W_{\text{dec}}$ | SAE decoder matrix | $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{dict}}}$ |
| $b_{\text{enc}}$ | SAE encoder bias | $b_{\text{enc}} \in \mathbb{R}^{d_{\text{dict}}}$ |
| $b_{\text{pre}}$ | SAE pre-encoder bias | $b_{\text{pre}} \in \mathbb{R}^{d_{\text{dict}}}$ |
| $b_{\text{dec}}$ | SAE decoder bias | $b_{\text{dec}} \in \mathbb{R}^{d_{\text{model}}}$ |
| $d_{\text{dict}}$ | SAE dictionary size | $d_{\text{dict}} = 24{,}576$ (GPT-2 Small res-jb) |
| $z$ | SAE latent activations (post-ReLU) | $z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}) \in \mathbb{R}^{d_{\text{dict}}}_{\geq 0}$ |
| $\hat{h}$ | SAE reconstruction | $\hat{h} = W_{\text{dec}} z + b_{\text{dec}} \in \mathbb{R}^{d_{\text{model}}}$ |
| $f_i$ | Feature $i$ (latent dimension) | $f_i \in \{1, \ldots, d_{\text{dict}}\}$ |
| $z_i$ | Activation of feature $i$ | $z_i \in \mathbb{R}_{\geq 0}$ |
| $d_i$ | Decoder direction for feature $i$ | $d_i = W_{\text{dec}}[:, i] \in \mathbb{R}^{d_{\text{model}}}$ |

## Rate-Distortion Framework

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $R$ | Rate (sparsity cost) | $R = \|z\|_0$ or $R = \|z\|_1$ (L1 relaxation) |
| $D$ | Distortion (reconstruction error) | $D = \|a - \hat{h}\|_2^2$ |
| $\lambda$ | Rate-distortion trade-off parameter | Controls sparsity-reconstruction balance |
| $\mathcal{R}(D)$ | Rate-distortion function | Minimum rate achievable for distortion $\leq D$ |
| $p_{\text{cooc}}$ | Parent-child co-occurrence probability | $p_{\text{cooc}} = P(\text{parent fires} \mid \text{child fires})$ |

## Absorption Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $A(f)$ | Absorption rate of feature $f$ | $A(f) \in [0, 1]$ |
| $A$ | Absorption rate (generic) | $A \in [0, 1]$ |
| $C_{\text{parent}}(f)$ | Parent feature activation correlation | Pearson correlation |
| $C_{\text{child}}(f)$ | Child feature activation correlation | Pearson correlation |
| $\Delta C(f)$ | Differential correlation (Chanin et al.) | $\Delta C(f) = C_{\text{parent}}(f) - C_{\text{child}}(f)$ |
| $\mathcal{F}_{\text{HIGH}}$ | High absorption feature set | $A(f) > 0.10$ |
| $\mathcal{F}_{\text{MED}}$ | Medium absorption feature set | $0.05 \leq A(f) \leq 0.10$ |
| $\mathcal{F}_{\text{LOW}}$ | Low absorption feature set | $A(f) < 0.05$ |

## Inhibition Graph (Falsified Framework)

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $G$ | Decoder correlation matrix | $G = W_{\text{dec}}^T W_{\text{dec}} \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{dict}}}$ |
| $G_{ij}$ | Decoder correlation between latents $i$ and $j$ | $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle \in \mathbb{R}$ |
| $N(i)$ | Top-k neighbors of latent $i$ | $N(i) \subseteq \{1, \ldots, d_{\text{dict}}\}$, $|N(i)| = k$ |
| $k$ | Number of neighbors per latent | $k = 20$ (default) |

## Graph Prediction Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $\text{P}@k$ | Precision at k | Fraction of top-k neighbors that are true absorption pairs |
| $\text{R}@k$ | Recall at k | Fraction of absorption pairs found in top-k neighbors |
| $\text{AUPR}$ | Area under precision-recall curve | $\text{AUPR} \in [0, 1]$ |
| $\text{TP}$ | True positives | Correctly predicted absorption pairs |
| $\text{FP}$ | False positives | Non-absorption pairs predicted as absorption |
| $\text{FN}$ | False negatives | Absorption pairs missed |

## Steering Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $s$ | Steering strength | $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$ |
| $S(f, s)$ | Steering success rate for feature $f$ at strength $s$ | $S(f, s) \in [0, 1]$ |
| $S_{\text{raw}}(f)$ | Raw steering success (at $s=50$) | $S_{\text{raw}}(f) \in [0, 1]$ |
| $S_{\text{rand}}(f)$ | Random baseline steering success | $S_{\text{rand}}(f) \in [0, 1]$ |
| $\Delta S(f)$ | Delta-corrected steering success | $\Delta S(f) = S_{\text{raw}}(f) - S_{\text{rand}}(f)$ |
| $\text{EC50}(f)$ | Median effective steering strength | $\text{EC50}(f) \in \mathbb{R}_{>0}$ |

## Probing Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $k_{\text{probe}}$ | Sparsity level for k-sparse probe | $k_{\text{probe}} \in \{1, 5, 10, 20\}$ |
| $\text{F1}(f, k)$ | F1 score for feature $f$ at sparsity $k$ | $\text{F1} \in [0, 1]$ |
| $P(f, k)$ | Precision for feature $f$ at sparsity $k$ | $P \in [0, 1]$ |
| $R(f, k)$ | Recall for feature $f$ at sparsity $k$ | $R \in [0, 1]$ |

## Random SAE Baseline (H10)

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $W_{\text{dec}}^{\text{rand}}$ | Random decoder matrix | Orthonormal random initialization |
| $W_{\text{enc}}^{\text{rand}}$ | Random encoder matrix | Independent random initialization |
| $A_{\text{trained}}(f)$ | Absorption rate on trained SAE | |
| $A_{\text{random}}(f)$ | Absorption rate on random SAE | |

## Correlation and Statistical Metrics

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $r$ | Pearson correlation coefficient | $r \in [-1, 1]$ |
| $\rho$ | Spearman rank correlation coefficient | $\rho \in [-1, 1]$ |
| $p$ | p-value (two-tailed) | $p \in [0, 1]$ |
| $R^2$ | Coefficient of determination | $R^2 \in [0, 1]$ |
| $\beta$ | Linear regression slope | $\beta \in \mathbb{R}$ |
| $t$ | t-statistic | $t \in \mathbb{R}$ |
| $d$ | Cohen's d (effect size) | $d \in \mathbb{R}$; small=0.2, medium=0.5, large=0.8 |
| $n$ | Sample size | $n = 26$ features (first-letter) |
| $\alpha_{\text{sig}}$ | Significance threshold | $\alpha_{\text{sig}} = 0.05$ |
| $\alpha_B$ | Bonferroni-corrected threshold | $\alpha_B = 0.05 / N_{\text{tests}}$ |
| $q_{\text{BH}}$ | Benjamini-Hochberg FDR q-value | $q_{\text{BH}} \in [0, 1]$ |
| $\text{CV}$ | Coefficient of variation | $\text{CV} = \sigma / |\mu|$ |

## Model and Layer Indices

| Symbol | Definition |
|--------|-----------|
| $l$ | Layer index | $l \in \{0, 4, 8, 10\}$ |
| $L$ | Total number of layers tested | $L = 4$ |
| $M$ | Primary model | GPT-2 Small (124M parameters) |
| $\theta$ | Model parameters | $\theta \in \mathbb{R}^{\|\theta\|}$ |

## Sets and Indices

| Symbol | Definition |
|--------|-----------|
| $\mathcal{L}$ | Set of letters | $\mathcal{L} = \{A, B, \ldots, Z\}$ |
| $|\mathcal{L}|$ | Number of first-letter features | $|\mathcal{L}| = 26$ |
| $i, j$ | Latent indices | $i, j \in \{1, \ldots, d_{\text{dict}}\}$ |
| $\phi(f)$ | Parent latent ID for feature $f$ | $\phi(f) \in \{1, \ldots, d_{\text{dict}}\}$ |

## Key Equations

**SAE forward pass:**
$$z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}), \quad \hat{a} = W_{\text{dec}} z + b_{\text{dec}}$$

**Rate-distortion objective:**
$$\min_{W_{\text{enc}}, W_{\text{dec}}} \mathbb{E}\left[\|a - \hat{a}\|_2^2 + \lambda \|z\|_1\right]$$

**Absorption detection (Chanin et al.):**
$$A(f) = \frac{\text{number of child features absorbing } f}{\text{total number of child features}}$$

**Steering intervention:**
$$h^{(l)}_{\text{steered}} = h^{(l)} + s \cdot d_f$$

**Delta-corrected steering:**
$$\Delta S(f) = S_{\text{feature}}(f) - S_{\text{random}}(f)$$

**F1 score:**
$$\text{F1} = 2 \cdot \frac{P \cdot R}{P + R}$$

**Decoder correlation (inhibition graph):**
$$G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$$

**Precision@k:**
$$\text{P}@k = \frac{|\{j \in N_k(i) : (i,j) \text{ is absorption pair}\}|}{k}$$
