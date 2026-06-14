# Notation Table: CV-Based Actionability Decomposition in Absorbed SAE Features

## Mathematical Symbols

### Model and SAE Parameters

| Symbol | Definition | Dimensionality |
|--------|------------|----------------|
| $\mathcal{D}$ | Training dataset (text corpus) | - |
| $x \in \mathbb{R}^d$ | Residual stream activation vector | $d = 768$ for GPT-2 |
| $L$ | Number of transformer layers | 12 for GPT-2 Small |
| $l$ | Specific layer index | $l \in \{0, 1, ..., 11\}$ |
| $\theta$ | SAE encoder/decoder weight matrix | - |
| $d_j$ | Decoder vector for SAE latent $j$ | $\mathbb{R}^d$ |
| $e_j$ | Encoder vector for SAE latent $j$ | $\mathbb{R}^d$ |
| $W_{enc}, W_{dec}$ | SAE encoder and decoder weight matrices | - |

### SAE Architecture

| Symbol | Definition | Notes |
|--------|------------|-------|
| $N$ | Number of SAE latents (dictionary size) | Also written as $d_{sae}$ |
| $d_{in}$ | Input dimension | $d_{in} = 768$ for GPT-2 |
| $d_{sae}$ | SAE latent dimension (dictionary size) | ~16k latents in experiments |
| $\lambda$ | L1 sparsity penalty coefficient | Control parameter |
| $f(x) = \max(0, W_{enc}x)$ | SAE encoder activation function | ReLU |
| $\hat{x} = W_{dec}f(x)$ | SAE reconstruction | - |

### Steering and Actionability

| Symbol | Definition | Notes |
|--------|------------|-------|
| $\tau$ | Steering strength | Applied at $\pm 3, \pm 5, \pm 7$ |
| $AUROC$ | Area under ROC curve | Detection metric, 98.2% in Basu et al. |
| $\Delta_{logit}$ | Logit change at target token | Primary steering effectiveness measure |
| $\Delta_{logit}^{abs}$ | Absolute logit change | $|\Delta_{logit}|$ for effect magnitude |

### Coefficient of Variation (CV)

| Symbol | Definition | Formula |
|--------|------------|---------|
| $CV$ | Coefficient of variation | $CV = \sigma / \mu$ |
| $\mu$ | Mean activation magnitude | Across contexts |
| $\sigma$ | Standard deviation of activation magnitude | Across contexts |
| $CV_{absorbed}$ | CV for absorbed features | ~7.33 in experiments |
| $CV_{non-absorbed}$ | CV for non-absorbed features | ~0.01 in experiments |
| $CV_{high}$ | High CV threshold | CV > 1.0 |
| $CV_{low}$ | Low CV threshold | CV $\le 1.0$ |

### Absorption Metrics

| Symbol | Definition | Formula |
|--------|------------|---------|
| $A_j$ | Training-free absorption detector | $A_j = \|d_j\|^2 / (d_j^\top e_j)$ |
| $\alpha$ | Absorption rate (fraction of features absorbed) | $\alpha = n_{absorbed} / N$ |
| $n_{absorbed}$ | Number of absorbed features | - |
| $threshold_{abs}$ | Absorption classification threshold | absorption_score > 0.5 |

### Causal Mediation

| Symbol | Definition | Notes |
|--------|------------|-------|
| $R_{parent}$ | Parent feature recovery (%) | From activation patching |
| $f_{child}$ | Child feature activation | Zeroed in ablation |
| $r_{context}$ | Context-sensitive routing | For high-CV features |
| $r_{bypass}$ | Stable bypass routing | For low-CV features |

### Decoder Orthogonality (H6)

| Symbol | Definition | Notes |
|--------|------------|-------|
| $cos(d_j, d_k)$ | Decoder cosine similarity | Between features j and k |
| $\bar{c}_j$ | Mean cosine similarity of feature j | With all other decoder vectors |
| $orthogonality_j$ | Orthogonality score for feature j | $1 - \bar{c}_j$ (higher = more orthogonal) |

### Statistical Terms

| Symbol | Definition | Notes |
|--------|------------|-------|
| $r$ | Pearson correlation coefficient | For CV-steering correlation |
| $R^2$ | Coefficient of determination | For regression fit quality |
| $p$ | p-value for statistical significance | $p < 0.01$ for significance |
| $t$ | t-statistic | For Welch's t-test |
| $\rho$ | Spearman rank correlation | Non-parametric alternative |
| $n$ | Sample size | Number of features or measurements |

---

## Greek Letters Reference

| Symbol | Name | Usage |
|--------|------|-------|
| $\lambda$ | lambda | Sparsity penalty coefficient |
| $\sigma$ | sigma | Standard deviation |
| $\mu$ | mu | Mean value |
| $\tau$ | tau | Steering strength |
| $\Delta$ | delta | Logit change (difference) |
| $\alpha$ | alpha | Absorption rate |
| $\rho$ | rho | Correlation coefficient |

---

## Superscripts and Subscripts

| Symbol | Meaning |
|--------|---------|
| $CV_{absorbed}$ | Coefficient of variation for absorbed features |
| $CV_{non-absorbed}$ | Coefficient of variation for non-absorbed features |
| $\Delta_{logit}$ | Logit change at target token |
| $\lambda_c$ | Critical lambda (for phase transition context) |

---

## Mathematical Notation Style

- Use italic for mathematical symbols: $m$, $\lambda$, $\chi$
- Use upright for text: SAE, GPT-2
- Greek letters preferred for physical quantities: $\lambda$ (lambda), $\chi$ (chi), $\tau$ (tau)
- Vector notation: $\mathbb{R}^d$ for d-dimensional real vectors
- Set notation: $\in$ for element membership
- Use $\times$ for multiplication: $\sigma \times \mu^{-1}$
- Scientific notation: $5 \times 10^{-5}$ for small numbers

---

## Abbreviations

| Abbreviation | Expansion |
|--------------|-----------|
| SAE | Sparse Autoencoder |
| ReLU | Rectified Linear Unit |
| CV | Coefficient of Variation |
| GPT | Generative Pre-Trained Transformer |
| LLM | Large Language Model |
| SAELens | Sparse Autoencoder Lens (library) |
| GemmaScope | Google's SAE release for Gemma models |
| AUROC | Area Under Receiver Operating Characteristic curve |
| BH | Benjamini-Hochberg (correction for multiple comparisons) |

---

## Preferred Terminology

| Preferred | Avoid |
|-----------|-------|
| "steering effect" | "steering utility" |
| "robust absorbed" | "absorbing parent" |
| "fragile absorbed" | "non-steerable feature" |
| "actionability paradox" | "steering failure" (for Basu et al.) |
| "variance paradox" | "CV reversal" |
| "context-sensitive routing" | "variable activation path" |
| "bypass routing" | "stable path" |
| "CV-based decomposition" | "CV classification" |
