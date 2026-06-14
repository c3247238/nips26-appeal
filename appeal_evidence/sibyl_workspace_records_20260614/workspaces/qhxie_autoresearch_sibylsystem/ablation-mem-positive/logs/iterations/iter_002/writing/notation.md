# Notation Table: Phase Transitions in SAE Feature Absorption

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
| $d_{sae}$ | SAE latent dimension (dictionary size) | $\{6144, 12288, 24576\}$ in experiments |
| $\lambda$ | L1 sparsity penalty coefficient | Control parameter, $\lambda \in [1e-5, 5e-2]$ |
| $\lambda_c$ | Critical sparsity threshold | Phase transition point at $\approx 5 \times 10^{-5}$ |
| $\lambda_{eff}$ | Effective sparsity per layer | Layer-dependent critical value |
| $f(x) = \max(0, W_{enc}x)$ | SAE encoder activation function | ReLU |
| $\hat{x} = W_{dec}f(x)$ | SAE reconstruction | - |

### Phase Transition Physics Analogy

| Symbol | Definition | Physical Analogy |
|--------|------------|-------------------|
| $m$ | Order parameter: mean absorption rate | Magnetization |
| $m(\lambda)$ | Absorption rate as function of sparsity | Order parameter curve |
| $\chi = dm/d\lambda$ | Susceptibility: rate of absorption change | Magnetic susceptibility |
| $\chi_{max}$ | Maximum susceptibility (peak at critical point) | $\chi_{max} = 11.19$ |
| $\chi_{ratio}$ | Ratio of peak susceptibility to average | $\chi_{ratio} = \chi_{max} / \bar{\chi}$ |
| $\delta\lambda$ | Transition width | Width of phase transition region |
| $\nu$ | Critical exponent | Governs scaling: $\delta\lambda \propto N^{-1/\nu}$, $\nu = 3$ |
| $N$ | Number of latents (dictionary size) | System size |

### Absorption Metrics

| Symbol | Definition | Formula |
|--------|------------|---------|
| $A_j$ | Training-free absorption detector | $A_j = \|d_j\|^2 / (d_j^\top e_j)$ |
| $\alpha$ | Absorption rate (fraction of features absorbed) | $\alpha = n_{absorbed} / N$ |
| $n_{absorbed}$ | Number of absorbed features | - |
| $r$ | Pearson correlation coefficient | See statistical symbols |
| $CV$ | Coefficient of variation | $CV = \sigma / \mu$ |
| $\mu$ | Mean activation magnitude | - |
| $\sigma$ | Standard deviation of activation magnitude | - |

### Co-occurrence Formula (H5)

| Symbol | Definition | Notes |
|--------|------------|-------|
| $S_{revised}$ | Revised co-occurrence score | $S_{revised} = cos(d_j, d_k) \cdot \log(f_j / f_k) \cdot (1 - \rho_j \rho_k)$ |
| $cos(d_j, d_k)$ | Decoder cosine similarity | Between features j and k |
| $f_j, f_k$ | Feature activation frequencies | - |
| $\rho_j, \rho_k$ | Normalized activation suppressions | - |

### Graph Topology (H6)

| Symbol | Definition | Notes |
|--------|------------|-------|
| $G = (V, E)$ | Absorption graph | Nodes = SAE features, Edges = absorption relationships |
| $|V| = N$ | Number of nodes (SAE latents) | - |
| $|E|$ | Number of edges (absorption pairs) | - |
| $C$ | Number of connected components | Order parameter candidate (falsified) |
| $S_{giant}$ | Giant component size | Largest connected component |
| $k_{mean}$ | Mean degree (edges per node) | - |

### Statistical Terms

| Symbol | Definition | Notes |
|--------|------------|-------|
| $r$ | Pearson correlation coefficient | $r = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2}\sqrt{\sum(y_i - \bar{y})^2}}$ |
| $R^2$ | Coefficient of determination | For scaling collapse quality, $R^2 = 0.951$ |
| $p$ | p-value for statistical significance | $p < 0.01$ for significance |
| $t$ | t-statistic | For t-test comparison of CV values |
| $\rho$ | Spearman rank correlation | Non-parametric alternative to Pearson |

### Experiment Configuration

| Symbol | Definition | Value |
|--------|------------|-------|
| $n_{samples}$ | Number of text samples | 1000 for most experiments |
| $seed$ | Random seed | 42 (fixed for reproducibility) |
| $threshold$ | Absorption classification threshold | $\lambda = 0.001$ |
| $hook_name$ | TransformerLens hook point | e.g., "blocks.6.hook_resid_pre" |

### Steering and Actionability

| Symbol | Definition | Notes |
|--------|------------|-------|
| $\tau$ | Steering strength | Applied at $\pm 3, \pm 5, \pm 7$ |
| $AUROC$ | Area under ROC curve | Detection metric, 98.2% in Basu et al. |
| $\Delta_{logit}$ | Logit change at target token | Steering effectiveness measure |

---

## Greek Letters Reference

| Symbol | Name | Usage |
|--------|------|-------|
| $\lambda$ | lambda | Sparsity penalty coefficient, control parameter |
| $\chi$ | chi | Susceptibility, $dm/d\lambda$ |
| $\nu$ | nu | Critical exponent for finite-size scaling |
| $\alpha$ | alpha | Absorption rate |
| $\rho$ | rho | Correlation coefficient, or normalized activation |
| $\sigma$ | sigma | Standard deviation |
| $\mu$ | mu | Mean value |
| $\epsilon$ | epsilon | Small perturbation |

---

## Superscripts and Subscripts

| Symbol | Meaning |
|--------|---------|
| $\lambda_c$ | Critical lambda (critical point) |
| $\chi_{max}$ | Maximum susceptibility |
| $\chi_{ratio}$ | Ratio of peak to average susceptibility |
| $N^{-1/\nu}$ | Inverse power scaling with critical exponent |
| $d_{sae}$ | SAE latent dimension |
| $d_{in}$ | Input dimension |
| $CV_{absorbed}$ | Coefficient of variation for absorbed features |
| $CV_{non-absorbed}$ | Coefficient of variation for non-absorbed features |

---

## Mathematical Notation Style

- Use italic for mathematical symbols: $m$, $\lambda$, $\chi$
- Use upright for text: SAE, GPT-2
- Greek letters preferred for physical quantities: $\lambda$ (lambda), $\chi$ (chi), $\nu$ (nu)
- Vector notation: $\mathbb{R}^d$ for d-dimensional real vectors
- Set notation: $\in$ for element membership
- Use $\times$ for multiplication: $\lambda \times N^{1/\nu}$
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

---

## Preferred Terminology

| Preferred | Avoid |
|-----------|-------|
| "feature absorption" | "absorption phenomenon" |
| "quasi-critical behavior" | "sharp transition" (when chi_ratio < 3) |
| "variance paradox" or "CV reversal" | "failed H4" |
| "layer saturation" | "layer heterogeneity" (for saturated case) |
| "critical exponent" | "scaling exponent" |
| "sparsity penalty" | "L1 regularization" (in SAE context) |
| "dictionary size" | "latent count", "number of features" |
| "giant component" | "main component" |
| "actionability paradox" | "steering failure" |
