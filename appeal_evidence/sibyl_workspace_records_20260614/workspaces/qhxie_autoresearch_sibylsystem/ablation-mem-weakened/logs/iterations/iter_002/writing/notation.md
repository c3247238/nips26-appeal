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

## LCA Framework

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $u$ | Membrane potential (LCA) | $u \in \mathbb{R}^{d_{\text{dict}}}$ |
| $b$ | Feedforward input (LCA) | $b = W_{\text{enc}}^T x \in \mathbb{R}^{d_{\text{dict}}}$ |
| $G$ | Inhibition matrix (LCA) | $G = W_{\text{dec}}^T W_{\text{dec}} \in \mathbb{R}^{d_{\text{dict}} \times d_{\text{dict}}}$ |
| $G_{ij}$ | Inhibition from latent $j$ to latent $i$ | $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle \in \mathbb{R}$ |
| $T$ | Threshold function | $T(u) = \max(0, u)$ (ReLU) |
| $\tau$ | Time constant (LCA dynamics) | $\tau \in \mathbb{R}_{>0}$ |
| $\mathcal{G}$ | Local inhibition graph | $\mathcal{G} = (V, E)$ where $V = \{1, \ldots, d_{\text{dict}}\}$ |
| $N(i)$ | Top-k neighbors of latent $i$ in $\mathcal{G}$ | $N(i) \subseteq V$, $|N(i)| = k$ |
| $k$ | Number of neighbors per latent | $k \in \{10, 20, 50\}$ |

## Inhibition Graph Metrics

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $\text{deg}(i)$ | Degree of node $i$ | $\text{deg}(i) = |N(i)| = k$ (fixed for local graph) |
| $\text{inh}_{\text{in}}(i)$ | Total incoming inhibition | $\text{inh}_{\text{in}}(i) = \sum_{j \in N(i)} |G_{ji}|$ |
| $\text{inh}_{\text{out}}(i)$ | Total outgoing inhibition | $\text{inh}_{\text{out}}(i) = \sum_{j \in N(i)} |G_{ij}|$ |
| $\bar{G}$ | Mean edge weight | $\bar{G} = \frac{1}{|E|} \sum_{(i,j) \in E} |G_{ij}|$ |
| $\rho_{\mathcal{G}}$ | Graph density | Fraction of possible edges present |
| $C(i)$ | Local clustering coefficient | $C(i) = \frac{2 \cdot |\{(j,k) \in N(i) \times N(i) : (j,k) \in E\}|}{k(k-1)}$ |
| $\text{CC}_{\mathcal{G}}$ | Mean clustering coefficient | $\text{CC}_{\mathcal{G}} = \frac{1}{d_{\text{dict}}} \sum_i C(i)$ |

## Absorption Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $A(f)$ | Absorption rate of feature $f$ | $A(f) \in [0, 1]$ |
| $A$ | Absorption rate (generic) | $A \in [0, 1]$ |
| $C_{\text{parent}}(f)$ | Parent feature activation correlation | Pearson correlation |
| $C_{\text{child}}(f)$ | Child feature activation correlation | Pearson correlation |
| $\Delta C(f)$ | Differential correlation | $\Delta C(f) = C_{\text{parent}}(f) - C_{\text{child}}(f)$ |
| $\mathcal{F}_{\text{HIGH}}$ | High absorption feature set | $A(f) > 0.10$ |
| $\mathcal{F}_{\text{MED}}$ | Medium absorption feature set | $0.05 \leq A(f) \leq 0.10$ |
| $\mathcal{F}_{\text{LOW}}$ | Low absorption feature set | $A(f) < 0.05$ |

## Graph Prediction Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $\text{P}@k$ | Precision at k | Fraction of top-k neighbors that are true absorption pairs |
| $\text{R}@k$ | Recall at k | Fraction of absorption pairs found in top-k neighbors |
| $\text{AUPR}$ | Area under precision-recall curve | $\text{AUPR} \in [0, 1]$ |
| $\text{TP}$ | True positives | Correctly predicted absorption pairs |
| $\text{FP}$ | False positives | Non-absorption pairs predicted as absorption |
| $\text{FN}$ | False negatives | Absorption pairs missed |

## Steering Metrics (from prior experiments)

| Symbol | Definition | Range |
|--------|-----------|-------|
| $s$ | Steering strength | $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$ |
| $S(f, s)$ | Steering success rate for feature $f$ at strength $s$ | $S(f, s) \in [0, 1]$ |
| $S_{\text{raw}}(f)$ | Raw steering success (at $s=50$) | $S_{\text{raw}}(f) \in [0, 1]$ |
| $S_{\text{rand}}(f)$ | Random baseline steering success | $S_{\text{rand}}(f) \in [0, 1]$ |
| $\Delta S(f)$ | Delta-corrected steering success | $\Delta S(f) = S_{\text{raw}}(f) - S_{\text{rand}}(f)$ |
| $\text{EC50}(f)$ | Median effective steering strength | $\text{EC50}(f) \in \mathbb{R}_{>0}$ |

## Probing Metrics (from prior experiments)

| Symbol | Definition | Range |
|--------|-----------|-------|
| $k_{\text{probe}}$ | Sparsity level for k-sparse probe | $k_{\text{probe}} \in \{1, 5, 10, 20\}$ |
| $\text{F1}(f, k)$ | F1 score for feature $f$ at sparsity $k$ | $\text{F1} \in [0, 1]$ |
| $P(f, k)$ | Precision for feature $f$ at sparsity $k$ | $P \in [0, 1]$ |
| $R(f, k)$ | Recall for feature $f$ at sparsity $k$ | $R \in [0, 1]$ |

## Homeostatic Rebalancing

| Symbol | Definition | Range |
|--------|-----------|-------|
| $\alpha$ | Rebalancing boost coefficient | $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ |
| $z'_i$ | Rebalanced activation of latent $i$ | $z'_i = \max(0, z_i + \alpha \cdot \text{inh}_i)$ |
| $\text{inh}_i$ | Inhibition received by latent $i$ | $\text{inh}_i = \sum_{j \in N(i)} G_{ij} \cdot z_j$ |
| $\epsilon$ | Reconstruction error tolerance | $\epsilon = 0.05$ (5%) |
| $\Delta_{\text{recon}}$ | Relative reconstruction error increase | $\Delta_{\text{recon}} = \frac{\|a - W_{\text{dec}} z'\|_2}{\|a - W_{\text{dec}} z\|_2} - 1$ |
| $\Delta_{\text{fire}}$ | Change in parent firing rate | $\Delta_{\text{fire}} = \text{fire}_{\text{after}} - \text{fire}_{\text{before}}$ |

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

**LCA dynamics (Rozell et al.):**
$$\tau \cdot \frac{du}{dt} = -u + (b - G \cdot a), \quad a = T(u)$$

**SAE forward pass:**
$$z = \text{ReLU}(W_{\text{enc}} a + b_{\text{pre}}), \quad \hat{a} = W_{\text{dec}} z + b_{\text{dec}}$$

**Structural correspondence (tied weights):**
$$G = W_{\text{dec}}^T W_{\text{dec}} = W_{\text{enc}} W_{\text{enc}}^T$$

**Inhibition graph construction:**
$$N(i) = \underset{j \neq i, |J| = k}{\arg\max_J} \sum_{j \in J} |G_{ij}|$$

**Homeostatic rebalancing:**
$$z'_i = \max\left(0, \; z_i + \alpha \sum_{j \in N(i)} G_{ij} z_j\right)$$

**Absorption detection (Chanin et al.):**
$$A(f) = \frac{\text{number of child features absorbing } f}{\text{total number of child features}}$$

**Steering intervention:**
$$h^{(l)}_{\text{steered}} = h^{(l)} + s \cdot d_f$$

**Delta-corrected steering:**
$$\Delta S(f) = S_{\text{feature}}(f) - S_{\text{random}}(f)$$

**F1 score:**
$$\text{F1} = 2 \cdot \frac{P \cdot R}{P + R}$$
