# Notation Table

## Mathematical Symbols

### Inputs and Data
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $x$ | Input activation vector from model residual stream | $x \in \mathbb{R}^{d_{\text{model}}}$ |
| $\mathcal{D}$ | Training/evaluation dataset (OpenWebText) | -- |
| $N$ | Number of samples in dataset | $N = 10{,}000$ (full), $1{,}000$ (pilot) |
| $c$ | A semantic concept (e.g., first letter 'a') | $c \in \mathcal{C}$ |
| $\mathcal{C}$ | Set of all concepts evaluated | $|\mathcal{C}| = 26$ (first letters) |

### Model Components
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $\theta$ | Model parameters (frozen during SAE evaluation) | -- |
| $f_{\theta}$ | Language model (Gemma-2-2B or GPT-2 Small) | -- |
| $d_{\text{model}}$ | Model hidden dimension | $d_{\text{model}} = 2304$ (Gemma-2-2B), $768$ (GPT-2 Small) |
| $L$ | Number of model layers | $L = 18$ (Gemma-2-2B), $12$ (GPT-2 Small) |
| $l$ | Layer index | $l \in \{0, 1, \ldots, L-1\}$ |

### SAE Components
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $\text{SAE}$ | Sparse Autoencoder | -- |
| $W_{\text{enc}}$ | SAE encoder weight matrix | $W_{\text{enc}} \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ |
| $W_{\text{dec}}$ | SAE decoder weight matrix | $W_{\text{dec}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$ |
| $b_{\text{enc}}$ | SAE encoder bias | $b_{\text{enc}} \in \mathbb{R}^{d_{\text{SAE}}}$ |
| $b_{\text{dec}}$ | SAE decoder bias | $b_{\text{dec}} \in \mathbb{R}^{d_{\text{model}}}$ |
| $d_{\text{SAE}}$ | SAE dictionary size (number of features) | $d_{\text{SAE}} \in \{16{,}384, 24{,}576, 3{,}072\}$ |
| $z$ | SAE latent representation (pre-activation) | $z \in \mathbb{R}^{d_{\text{SAE}}}$ |
| $\hat{x}$ | SAE reconstruction of input activation | $\hat{x} \in \mathbb{R}^{d_{\text{model}}}$ |
| $k$ | TopK sparsity parameter (number of active features) | $k \in \{10, 25, 50, 100, 200\}$ |
| $f_i$ | The $i$-th SAE dictionary feature (column of $W_{\text{dec}}$) | $f_i \in \mathbb{R}^{d_{\text{model}}}$ |
| $\phi_i(x)$ | Activation of feature $f_i$ on input $x$ | $\phi_i(x) \in \mathbb{R}_{\geq 0}$ |

### Sparsity and Activation
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $\text{L0}$ | L0 "norm": mean number of active features per token | $\text{L0} \in \mathbb{R}_{\geq 0}$ |
| $\text{TopK}(z, k)$ | Hard top-k activation: keep top $k$ values, zero others | -- |
| $\text{JumpReLU}(z)$ | JumpReLU gating: $\mathbb{1}[z > \theta] \odot z$ | -- |
| $\lambda$ | Sparsity penalty coefficient (for L1 SAEs) | $\lambda \in \mathbb{R}_{>0}$ |

### Absorption Metrics
| Symbol | Definition | Range |
|--------|-----------|-------|
| $\text{CR}$ | Collision rate: fraction of concepts sharing features | $\text{CR} \in [0, 1]$ |
| $\text{AR}$ | Absorption rate: fraction of absorbed parent-child pairs | $\text{AR} \in [0, 1]$ |
| $\text{Spec}(f, c)$ | Specificity score for feature $f$ on concept $c$ | $\text{Spec} \in \mathbb{R}_{\geq 0}$ |
| $\text{UF}$ | Number of unique features (distinct feature indices) | $\text{UF} \in \mathbb{N}$ |
| $\text{DFR}$ | Dead feature ratio: fraction of never-active features | $\text{DFR} \in [0, 1]$ |

### Downstream Task Metrics
| Symbol | Definition | Range |
|--------|-----------|-------|
| $\text{MSE}$ | Mean squared error: $\frac{1}{N}\sum_{i=1}^N \|x_i - \hat{x}_i\|_2^2$ | $\text{MSE} \in \mathbb{R}_{\geq 0}$ |
| $\text{Acc}_{\text{train}}$ | Sparse probing training accuracy | $[0, 1]$ |
| $\text{Acc}_{\text{test}}$ | Sparse probing test accuracy | $[0, 1]$ |
| $\text{AUROC}$ | Area under ROC curve for concept detection | $[0, 1]$ |
| $\rho_S$ | Spearman rank correlation coefficient | $[-1, 1]$ |
| $p$ | Statistical significance (p-value) | $[0, 1]$ |

### UAD and DFDA
| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $\text{UAD}$ | Unsupervised Absorption Detector | -- |
| $\text{DFDA}$ | Dynamic Feature De-Absorption module | -- |
| $C$ | Co-occurrence matrix (feature activations) | $C \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{SAE}}}$ |
| $n_c$ | Number of clusters for hierarchical clustering | $n_c = 50$ |
| $\text{MLP}_{\text{comp}}$ | Compensation MLP in DFDA | $<1\%$ of SAE params |
| $\Delta_{\text{MSE}}$ | MSE improvement from DFDA | $\Delta_{\text{MSE}} \in \mathbb{R}$ |

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

## Abbreviations Used in Equations
| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| CAAB | Cross-Architecture Absorption Benchmark |
| UAD | Unsupervised Absorption Detection |
| DFDA | Dynamic Feature De-Absorption |
| MSE | Mean Squared Error |
| L0 | L0 pseudo-norm (count of non-zeros) |
| AUROC | Area Under Receiver Operating Characteristic |
