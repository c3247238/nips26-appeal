# Notation Table: Quantifying Feature Absorption in Sparse Autoencoders

## Model and SAE Architecture

| Symbol | Definition | Dimensionality |
|--------|------------|-----------------|
| $x \in \mathbb{R}^{d_{model}}$ | Residual stream activation vector at a given token and layer | $d_{model}=768$ for GPT-2 small |
| $\hat{x} \in \mathbb{R}^{d_{model}}$ | SAE reconstruction of $x$ | Same as $x$ |
| $W_{enc} \in \mathbb{R}^{d_{sae} \times d_{model}}$ | SAE encoder weight matrix | $d_{sae} \in \{2048, 4096, 8192, 16384, 24576\}$ |
| $W_{dec} \in \mathbb{R}^{d_{model} \times d_{sae}}$ | SAE decoder weight matrix (transpose of encoder) | $d_{model} \times d_{sae}$ |
| $b_{enc} \in \mathbb{R}^{d_{sae}}$ | SAE encoder bias vector | $d_{sae}$ |
| $b_{dec} \in \mathbb{R}^{d_{model}}$ | SAE decoder bias vector | $d_{model}$ |
| $f \in \{1, \ldots, d_{sae}\}$ | Latent feature index | Scalar |
| $act_f \in \mathbb{R}$ | Activation value of latent $f$ for a given token | Scalar |
| $d_{sae}$ | SAE dictionary (bottleneck) size | Scalar |
| $L0$ | L0 norm: number of non-zero latents per token | Scalar |

## Absorption Score Computation

| Symbol | Definition | Type |
|--------|------------|------|
| $\mathcal{A}_f$ | Set of activating tokens for latent $f$ (tokens where $act_f > \tau$) | Set of token indices |
| $\tau$ | Activation threshold: 1% of max activation of $f$ across corpus | Scalar |
| $C_{f,t}$ | Top-5 co-firing latents for latent $f$ at token $t$ | Set of 5 latent indices |
| $x_t$ | Original residual stream activation at token $t$ | Vector, $\mathbb{R}^{d_{model}}$ |
| $x_{partial}$ | Partial reconstruction using $f$ and its top-5 co-firers | Vector, $\mathbb{R}^{d_{model}}$ |
| $x_{partial} = W_{dec}[f] \cdot act_f + \sum_{c \in C_{f,t}} W_{dec}[c] \cdot act_c$ | Partial reconstruction formula | Scalar reconstruction |
| $VE_f$ | Variance explained by co-firers for a single token | Scalar in $[0, 1]$ |
| $VE_f = 1 - \frac{\text{var}(x_t - x_{partial})}{\text{var}(x_t)}$ | Variance explained formula | Scalar |
| $A_f$ | Absorption score for latent $f$ (also denoted $absorption(f)$) | Scalar in $[0, 1]$ |
| $A_f = \frac{1}{\|\mathcal{A}_f\|} \sum_{t \in \mathcal{A}_f} \mathbb{1}[VE_f(t) > 0.8]$ | Fraction of activating tokens where co-firers explain >80% of variance | Scalar |

## Layer-Dependent Absorption Results

| Symbol | Definition | Value |
|--------|------------|-------|
| $A_{layer4}$ | Absorption rate (% >0.5) at layer 4 | 49.3% (CONFIRMED) |
| $A_{layer8}$ | Absorption rate (% >0.5) at layer 8 | 0.19% (FALSIFIED) |
| $L0_{layer4}$ | Mean L0 at layer 4 | 37.8 |
| $L0_{layer8}$ | Mean L0 at layer 8 | 71.9 |
| $r_S$ | Spearman rank correlation between L0 and absorption across layers | 0.086 (p=0.872) |
| $r_P$ | Pearson correlation between L0 and absorption across layers | -0.073 (p=0.891) |

## H3 Sparsity Results (Inverted-U Pattern)

| Layer | Mean L0 | Mean Absorption | % > 0.5 |
|-------|---------|-----------------|---------|
| 0 | 18.9 | 0.229 | 19.5% |
| 2 | 29.1 | 0.470 | 45.5% |
| 4 | 37.8 | 0.503 | 49.3% |
| 6 | 57.0 | 0.430 | 41.0% |
| 8 | 71.9 | 0.305 | 20.9% |
| 10 | 56.0 | 0.287 | 17.3% |

## H2 Token Frequency Correlation (PENDING - Critical Path)

| Symbol | Definition | Status |
|--------|------------|--------|
| $f_{token}$ | Token log-frequency in corpus | To be computed |
| $r_{S}^{(H2)}$ | Spearman correlation between token frequency and absorption | PENDING |
| $Q_1, Q_2, Q_3, Q_4$ | Quartile bins by median token frequency | PENDING |
| $A_{Q_1}$ | Mean absorption in lowest-frequency quartile | PENDING |
| $A_{Q_4}$ | Mean absorption in highest-frequency quartile | PENDING |

## H6 Perfect-Score Latent Investigation (PENDING - Critical Path)

| Symbol | Definition | Status |
|--------|------------|--------|
| $\mathcal{P}_f$ | Set of activating token positions for latent $f$ | To be computed |
| $p_{consistency}$ | Position consistency score (fraction activating at same position) | PENDING |
| $N_{perfect}$ | Number of perfect-score latents ($A_f = 1.0$) | 8 observed |
| $T_{perfect}$ | Activating token count per perfect-score latent | 100 (0.78% of corpus) |

## H5 Dictionary Size Results

| Symbol | Definition | Value |
|--------|------------|-------|
| $\mathcal{D}_2$ | Dictionary size 2,048 | Mean absorption: 0.0268, %>0.5: 2.25% |
| $\mathcal{D}_8$ | Dictionary size 8,192 | Mean absorption: 0.0067, %>0.5: 0.56% |
| $\mathcal{D}_{24}$ | Dictionary size 24,576 | Mean absorption: 0.0022, %>0.5: 0.19% |

## H4 Circuit Faithfulness Results

| Symbol | Definition | Value |
|--------|------------|-------|
| $F_{raw}$ | Faithfulness (raw residual patching) | 0.400 |
| $F_{sae}$ | Faithfulness (SAE all latents) | 0.289 |
| $F_{low}$ | Faithfulness (SAE low-absorption latents) | 0.000 |
| $F_{high}$ | Faithfulness (SAE high-absorption latents) | 0.000 |
| $\Delta F_{raw-sae}$ | Faithfulness loss from SAE bottleneck | 0.111 (11.1 pp) |

## Experiment Parameters

| Symbol | Definition | Value |
|--------|------------|-------|
| $N_{seq}$ | Number of sequences in analysis corpus | 1,024 (full) / 100 (pilot) |
| $T_{seq}$ | Tokens per sequence | 128 |
| $N_{total} = N_{seq} \times T_{seq}$ | Total tokens in analysis corpus | 131,072 (full) / 12,800 (pilot) |
| $L$ | Model layers audited | $\{0, 2, 4, 6, 8, 10\}$ |
| $d_{model}$ | Model hidden dimension | 768 (GPT-2 small) |
| $seed$ | Random seed for reproducibility | 42 |

## Statistical Measures

| Symbol | Definition | Type |
|--------|------------|------|
| $\mathbb{1}[cond]$ | Indicator function (1 if condition true, 0 otherwise) | Scalar |
| $\text{var}(\cdot)$ | Variance across a set of values | Scalar |
| $\text{mean}(\cdot)$ | Arithmetic mean | Scalar |
| $\text{median}(\cdot)$ | Median value | Scalar |
| $r_S$ | Spearman rank correlation coefficient | Scalar in $[-1, 1]$ |
| $r_P$ | Pearson correlation coefficient | Scalar in $[-1, 1]$ |
| $pp$ | Percentage points (for faithfulness differences) | Scalar |

## Abbreviations

| Abbreviation | Expansion |
|--------------|-----------|
| SAE | Sparse Autoencoder |
| SAELens | Sparse Autoencoder Lens (library) |
| MLP | Multi-Layer Perceptron |
| residual stream | Skip-connection pathway in a Transformer |
| d_model | Model hidden dimension |
| d_sae | SAE dictionary (bottleneck) size |
| L0 | L0 norm: count of non-zero entries |
| L1 | L1 norm: sum of absolute values (sparsity penalty) |
| pp | Percentage points |
| FALSIFIED | Hypothesis result when empirical evidence contradicts prediction |
| CONFIRMED | Hypothesis result when empirical evidence supports prediction |
| NOT TESTED | Hypothesis not run due to NO-GO project decision |
| UNINFORMATIVE | Hypothesis test yielded no meaningful signal (both conditions = 0) |
| NO-GO | Project decision to not proceed with full experiments |