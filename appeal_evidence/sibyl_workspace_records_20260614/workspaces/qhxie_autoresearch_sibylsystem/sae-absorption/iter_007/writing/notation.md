# Notation Table

All mathematical symbols and notation used in the paper. Section writers, critics, and the editor reference this file for consistency.

---

## Model and Architecture

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $x$ | Residual stream activation | $x \in \mathbb{R}^{d_{\text{model}}}$ |
| $\hat{x}$ | SAE reconstruction of $x$ | $\hat{x} \in \mathbb{R}^{d_{\text{model}}}$ |
| $d_{\text{model}}$ | Model hidden dimension | Scalar; 2304 for Gemma 2 2B, 768 for GPT-2 Small |
| $d_{\text{SAE}}$ | SAE dictionary width (number of latents) | Scalar; 16,384 or 65,536 |
| $W_e$ | SAE encoder weight matrix | $W_e \in \mathbb{R}^{d_{\text{SAE}} \times d_{\text{model}}}$ |
| $W_d$ | SAE decoder weight matrix | $W_d \in \mathbb{R}^{d_{\text{model}} \times d_{\text{SAE}}}$ |
| $d_j$ | $j$-th decoder column (unit-normalized) | $d_j \in \mathbb{R}^{d_{\text{model}}}$, $\|d_j\| = 1$ |
| $z$ | SAE latent activation vector | $z \in \mathbb{R}^{d_{\text{SAE}}}$ |
| $z_j$ | Activation of $j$-th SAE latent | Scalar $\geq 0$ |
| $b_{e,j}$ | Encoder bias for latent $j$ | Scalar |
| $\theta_j$ | JumpReLU per-latent activation threshold | Scalar $> 0$ |
| $L_0$ | Number of non-zero latents per forward pass (operating point) | Scalar; tested at $\{22, 41, 82, 176\}$ |

## Absorption Measurement

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $z_p$ | Parent SAE latent (general feature, e.g., "starts-with-A") | Scalar activation |
| $z_c$ | Child SAE latent (specific feature, e.g., "word-specific token") | Scalar activation |
| $v_p$ | Probe direction (learned weight vector of $k$-sparse logistic regression) | $v_p \in \mathbb{R}^{d_{\text{SAE}}}$ |
| $k$ | Sparsity of logistic regression probe (number of selected latents) | Scalar; $k = 5$ throughout |
| $\tau_{\cos}$ | Cosine similarity threshold for candidate feature identification | Scalar; default 0.025 |
| $\tau_{\text{mag}}$ | Magnitude gap threshold for absorption confirmation | Scalar; default 1.0 |
| $\cos(d_j, v_p)$ | Cosine similarity between decoder column $j$ and probe direction | Scalar $\in [-1, 1]$ |

## Rates and Statistics

| Symbol | Definition | Notes |
|--------|-----------|-------|
| Absorption rate | Fraction of parent-positive tokens where all $k$ probe-associated latents are inactive | Per-letter and aggregate |
| FN | False negative: token correctly classified by probe but all $k$ associated latents inactive | Count or rate |
| F1 | Probe classification F1 score per parent class | Quality gate: F1 > 0.85 |
| $\rho_s$ | Spearman rank correlation coefficient | Used for CMI-absorption and LOO analyses |
| Cohen's $d$ | Standardized mean difference between absorbed and non-absorbed groups | Effect size |
| CV | Coefficient of variation (std/mean) | Used for threshold sensitivity, cross-layer stability |
| CI | 95% bootstrap confidence interval | 10,000 resamples, seed = 42 |

## Information-Theoretic Quantities

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $I(X; f_p \mid f_c)$ | Conditional mutual information: unique information parent encodes about $X$ given child | Scalar $\geq 0$ (bits) |
| CMI | Shorthand for conditional mutual information | |
| $d'$ | Subspace dimension for CMI estimation | Scalar; primary $d' = 10$ |
| $k_{\text{nn}}$ | Number of nearest neighbors in $k$-NN MI estimator | Scalar; primary $k_{\text{nn}} = 5$ |

## Confound Decomposition

| Symbol | Definition | Notes |
|--------|-----------|-------|
| Permissive hedging | Token ceases to be FN at any higher $L_0$ | Upper bound on hedging; 98.6% at $L_0=22$ |
| Strict hedging | At least 1 of $k$ parent-associated latents fires at $L_0=176$ | Lower bound on hedging; 6.2% at $L_0=22$ |
| Persistent core word | Token classified as FN at all 4 tested $L_0$ values $\{22, 41, 82, 176\}$ | 8 words identified |
| Hierarchy-driven FN | FN that persists across all $L_0$ values (not resolved by relaxing sparsity) | 8--9 tokens (see footnote on count discrepancy) |

## Controls

| Symbol | Definition | Expected Behavior |
|--------|-----------|-------------------|
| C1 (Random probe) | Probe trained on random direction | Near-zero absorption (~9.2%) |
| C2 (Shuffled labels) | Probe trained on permuted class assignments | Should be $\leq$ measured; INVERTED in all 5 domains |
| C3 (Dense probe) | All-feature logistic regression | High F1 (0.929), demonstrates signal in activations |
| C4 (Untrained SAE) | Pre-training decoder columns | 0% absorption (correct null) |

## Cross-Domain Hierarchies

| Hierarchy | Parent | Child | Source |
|-----------|--------|-------|--------|
| First-letter | Letter (A--Z) | Word token | English word list |
| City-country | Country | City | RAVEL dataset |
| City-continent | Continent | City | RAVEL dataset |
| City-language | Language | City | RAVEL dataset |
| Animal-class | Taxonomic class | Animal | WordNet |
