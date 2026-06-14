# Notation Table

## Model and Architecture

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $M$ | Language model (Gemma-2-2B or GPT-2 Small) | -- |
| $L$ | Number of layers in the model | scalar |
| $\ell$ | Layer index | $\ell \in \{1, \dots, L\}$ |
| $d_{\text{model}}$ | Hidden dimension of the model | $d_{\text{model}} = 768$ (GPT-2), 2304 (Gemma-2-2B) |
| $d_{\text{SAE}}$ | SAE dictionary size (number of latents) | $d_{\text{SAE}} = 16384$ (GemmaScope width_16k), 24576 (GPT-2) |

## SAE Components

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $W_{\text{enc}}$ | SAE encoder weight matrix | $\mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ |
| $W_{\text{dec}}$ | SAE decoder weight matrix | $\mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$ |
| $b_{\text{enc}}$ | SAE encoder bias | $\mathbb{R}^{d_{\text{SAE}}}$ |
| $b_{\text{dec}}$ | SAE decoder bias | $\mathbb{R}^{d_{\text{model}}}$ |
| $d_j$ | $j$-th decoder column (decoder vector for latent $j$) | $\mathbb{R}^{d_{\text{model}}}$ |
| $e_j$ | $j$-th encoder row (encoder vector for latent $j$) | $\mathbb{R}^{d_{\text{model}}}$ |
| $z$ | SAE latent activation vector | $\mathbb{R}^{d_{\text{SAE}}}$ |
| $z_j$ | Activation of latent $j$ | scalar |
| $\text{L0}$ | Average number of non-zero latents per token | scalar |

## Activations and Representations

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $x$ | Model activation (residual stream) at a given layer | $\mathbb{R}^{d_{\text{model}}}$ |
| $\hat{x}$ | SAE reconstruction of activation | $\mathbb{R}^{d_{\text{model}}}$ |
| $f(x)$ | SAE encoding function: $f(x) = \text{topk}(W_{\text{enc}} x + b_{\text{enc}})$ or $\text{ReLU}(\cdot)$ | $\mathbb{R}^{d_{\text{SAE}}}$ |
| $g(z)$ | SAE decoding function: $g(z) = W_{\text{dec}} z + b_{\text{dec}}$ | $\mathbb{R}^{d_{\text{model}}}$ |

## Semantic Probes

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathcal{C}$ | Set of semantic categories, $|\mathcal{C}| = 10$ | set |
| $c$ | Individual semantic category (e.g., "animal") | element of $\mathcal{C}$ |
| $\mathcal{H}_c$ | Set of hyponyms for category $c$, $|\mathcal{H}_c| = 15$ | set |
| $h$ | Individual hyponym (e.g., "dog") | string |
| $w$ | Linear probe weight vector for a category | $\mathbb{R}^{d_{\text{SAE}}}$ |
| $w_j$ | $j$-th component of probe weight vector | scalar |
| $j^*$ | Index of the top-weighted latent: $j^* = \arg\max_j |w_j|$ | scalar |

## Absorption Metrics

| Symbol | Definition | Range |
|--------|-----------|-------|
| $A_{\text{abl}}$ | Ablation score: accuracy difference before/after ablating top latent | $[-1, 1]$ |
| $A_{\text{proj}}$ | Projection absorption: fraction of probe weight captured by top latent | $[0, 1]$ |
| $R_{\text{proj}}$ | Projection ratio: $1 - A_{\text{proj}}$ | $[0, 1]$ |
| $A_j$ | Training-free detector: $A_j = \|d_j\|^2 / (d_j^\top e_j)$ where $e_j$ is the $j$-th encoder row | $[0, \infty)$ |
| $\tau_{\text{abs}}$ | Absorption threshold for ablation metric (default: 0.05) | scalar |
| $\tau_{\text{proj}}$ | Absorption threshold for projection metric (default: 0.5) | scalar |

## Statistical Measures

| Symbol | Definition |
|--------|-----------|
| $\rho$ | Spearman rank correlation coefficient |
| $p$ | p-value for statistical significance test |
| $d$ | Cohen's d (effect size) |
| $U$ | Mann-Whitney U statistic |
| $z$ | z-statistic for Fisher's r-to-z transformation |
| $\text{AUROC}$ | Area Under the Receiver Operating Characteristic curve |
| $\mu$ | Mean |
| $\sigma$ | Standard deviation |

## Architecture-Specific Notation

| Symbol | Definition |
|--------|-----------|
| $\text{JumpReLU}$ | JumpReLU SAE architecture (GemmaScope) |
| $\text{ReLU}$ | Standard ReLU SAE architecture (GPT-2 via SAELens) |
| $\text{TopK}$ | Top-K SAE architecture |
| $\|\cdot\|$ | L2 norm (Euclidean norm) |
| $\|\cdot\|_1$ | L1 norm |
| $\cos(\cdot, \cdot)$ | Cosine similarity |

## Sets and Indices

| Symbol | Definition |
|--------|-----------|
| $\mathcal{D}$ | Training dataset for probes |
| $\mathcal{D}_{\text{test}}$ | Test dataset for probes |
| $N$ | Number of probes (total) |
| $n$ | Number of probes per layer or architecture |
| $i$ | Sample index |
| $j$ | Latent/feature index |
| $k$ | Category index |
