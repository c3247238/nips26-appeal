# Notation Table

## Mathematical Symbols

### Inputs and Data
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $x$ | Input activation vector from model residual stream | $x \in \mathbb{R}^{d_{\text{model}}}$ |
| $\mathcal{D}$ | Evaluation dataset (OpenWebText) | -- |
| $N$ | Number of samples in dataset | $N = 1{,}000$ (full experiments) |
| $c$ | A semantic concept (e.g., first letter 'a') | $c \in \mathcal{C}$ |
| $\mathcal{C}$ | Set of all concepts evaluated | $|\mathcal{C}| = 26$ (first letters a-z) |
| $t$ | Input token | -- |
| $\mathcal{T}$ | Token sequence | -- |

### Model Components
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $\theta$ | Language model parameters (frozen) | -- |
| $f_{\theta}$ | Language model (GPT-2 Small) | -- |
| $d_{\text{model}}$ | Model hidden dimension | $d_{\text{model}} = 768$ |
| $L$ | Number of model layers | $L = 12$ |
| $l$ | Layer index | $l \in \{0, 1, \ldots, L-1\}$ |

### SAE Components
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $\text{SAE}$ | Sparse Autoencoder | -- |
| $W_{\text{enc}}$ | SAE encoder weight matrix | $W_{\text{enc}} \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ |
| $W_{\text{dec}}$ | SAE decoder weight matrix | $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$ |
| $b_{\text{enc}}$ | SAE encoder bias | $b_{\text{enc}} \in \mathbb{R}^{d_{\text{SAE}}}$ |
| $b_{\text{dec}}$ | SAE decoder bias | $b_{\text{dec}} \in \mathbb{R}^{d_{\text{model}}}$ |
| $d_{\text{SAE}}$ | SAE dictionary size | $d_{\text{SAE}} = 24{,}576$ |
| $z$ | SAE latent representation | $z \in \mathbb{R}^{d_{\text{SAE}}}$ |
| $\hat{x}$ | SAE reconstruction of input activation | $\hat{x} \in \mathbb{R}^{d_{\text{model}}}$ |
| $f_i$ | The $i$-th SAE dictionary feature | $f_i \in \mathbb{R}^{d_{\text{model}}}$ |
| $\phi_i(x)$ | Activation of feature $f_i$ on input $x$ | $\phi_i(x) \in \mathbb{R}_{\geq 0}$ |

### UAD: Co-Occurrence and Clustering
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $A$ | Feature activation matrix | $A \in \mathbb{R}^{N \times d_{\text{SAE}}}$ |
| $C$ | Co-occurrence matrix: $C_{ij} = \sum_n \mathbb{1}[A_{ni} > 0] \cdot \mathbb{1}[A_{nj} > 0]$ | $C \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{SAE}}}$ |
| $R$ | Phi coefficient correlation matrix | $R \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{SAE}}}$ |
| $\phi_{ij}$ | Phi coefficient between features $i$ and $j$ | $\phi_{ij} \in [-1, 1]$ |
| $n_c$ | Number of clusters for hierarchical clustering | $n_c = 50$ |
| $\mathcal{H}$ | Hierarchical clustering model (Ward linkage) | -- |
| $L_{\text{link}}$ | Linkage matrix from hierarchical clustering | -- |
| $\mathcal{P}_{\text{cand}}$ | Set of candidate absorbed (parent, child) pairs | -- |

### Absorption and Suppression
| Symbol | Definition | Range |
|--------|-----------|-------|
| $p$ | Parent feature index | $p \in \{0, \ldots, d_{\text{SAE}}-1\}$ |
| $c$ | Child feature index | $c \in \{0, \ldots, d_{\text{SAE}}-1\}$ |
| $\mathbb{P}(p \mid c)$ | Probability parent fires given child fires | $[0, 1]$ |
| $\mathbb{P}(p \mid \neg c)$ | Probability parent fires given child does not fire | $[0, 1]$ |
| $\Delta_{\text{supp}}$ | Suppression signal: $\mathbb{P}(p \mid \neg c) - \mathbb{P}(p \mid c)$ | $[-1, 1]$ |
| $\rho_{\text{cooc}}$ | Co-occurrence strength (symmetric) | $[0, 1]$ |

### DFDA: Compensation Network
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $\text{MLP}_{\text{comp}}$ | Compensation multi-layer perceptron | -- |
| $W_1$ | First layer weight matrix | $W_1 \in \mathbb{R}^{64 \times 1}$ |
| $W_2$ | Second layer weight matrix | $W_2 \in \mathbb{R}^{1 \times 64}$ |
| $b_1$ | First layer bias | $b_1 \in \mathbb{R}^{64}$ |
| $b_2$ | Second layer bias | $b_2 \in \mathbb{R}$ |
| $z_c$ | Child feature activation | $z_c \in \mathbb{R}$ |
| $z_p$ | Parent feature activation | $z_p \in \mathbb{R}$ |
| $\hat{r}_p$ | Predicted parent residual: $\hat{r}_p = \text{MLP}_{\text{comp}}(z_c)$ | $\hat{r}_p \in \mathbb{R}$ |
| $z_p^{\text{comp}}$ | Compensated parent activation: $z_p^{\text{comp}} = z_p + \hat{r}_p$ | $z_p^{\text{comp}} \in \mathbb{R}$ |

### Evaluation Metrics
| Symbol | Definition | Range |
|--------|-----------|-------|
| $\text{TP}$ | True positives: same-cluster pairs that are supervised collisions | $\text{TP} \in \mathbb{N}$ |
| $\text{FP}$ | False positives: same-cluster pairs that are not collisions | $\text{FP} \in \mathbb{N}$ |
| $\text{FN}$ | False negatives: collision pairs not in same cluster | $\text{FN} \in \mathbb{N}$ |
| $\text{Prec}$ | Precision: $\text{TP} / (\text{TP} + \text{FP})$ | $[0, 1]$ |
| $\text{Rec}$ | Recall: $\text{TP} / (\text{TP} + \text{FN})$ | $[0, 1]$ |
| $\text{F1}$ | F1 score: $2 \cdot \text{Prec} \cdot \text{Rec} / (\text{Prec} + \text{Rec})$ | $[0, 1]$ |
| $\text{MSE}$ | Mean squared error: $\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$ | $\mathbb{R}_{\geq 0}$ |
| $\Delta_{\text{MSE}}$ | MSE improvement: $(\text{MSE}_{\text{base}} - \text{MSE}_{\text{comp}}) / \text{MSE}_{\text{base}}$ | $\mathbb{R}$ |
| $\rho_S$ | Spearman rank correlation coefficient | $[-1, 1]$ |
| $p$ | Statistical significance (p-value) | $[0, 1]$ |
| $\text{CI}_{95}$ | 95% bootstrap confidence interval | -- |

### Set and Function Notation
| Symbol | Definition |
|--------|-----------|
| $\mathbb{R}$ | Real numbers |
| $\mathbb{R}_{\geq 0}$ | Non-negative real numbers |
| $\mathbb{R}_{>0}$ | Positive real numbers |
| $\mathbb{N}$ | Natural numbers (including 0) |
| $\mathbb{1}[\cdot]$ | Indicator function (1 if condition true, 0 otherwise) |
| $\|\cdot\|_2$ | L2 (Euclidean) norm |
| $\odot$ | Element-wise (Hadamard) product |
| $\mathbb{E}[\cdot]$ | Expected value |
| $\text{Var}(\cdot)$ | Variance |
| $\sigma(\cdot)$ | Sigmoid function |
| $\text{ReLU}(x)$ | Rectified linear unit: $\max(0, x)$ |

## Abbreviations Used in Equations
| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| UAD | Unsupervised Absorption Detection |
| DFDA | Dynamic Feature De-Absorption |
| MSE | Mean Squared Error |
| HAC | Hierarchical Agglomerative Clustering |
| MLP | Multi-Layer Perceptron |
| ReLU | Rectified Linear Unit |
| TP | True Positive |
| FP | False Positive |
| FN | False Negative |
| AUROC | Area Under Receiver Operating Characteristic |
| PMI | Pointwise Mutual Information |
| MI | Mutual Information |
