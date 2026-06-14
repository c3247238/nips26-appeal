# Notation Table

> All mathematical symbols and notation used in the paper.  
> Section writers must reference this file for consistency.

---

## Inputs and Data

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $\mathcal{D}$ | Training dataset (OpenWebText samples) | $N$ sequences |
| $x$ | Input token sequence | $x \in \mathcal{V}^L$ |
| $x_i$ | $i$-th token in sequence | $x_i \in \mathcal{V}$ |
| $\mathcal{V}$ | Vocabulary | $|\mathcal{V}| = 50,257$ (GPT-2) |
| $L$ | Sequence length | $L \leq 128$ |
| $N$ | Number of sequences | $N = 1,000$ |

## Model and SAE

| Symbol | Definition | Dimensionality |
|--------|-----------|----------------|
| $f_\theta$ | Language model (GPT-2 Small) | $\theta \in \mathbb{R}^{124\text{M}}$ |
| $h^{(l)}$ | Hidden representation at layer $l$ | $h^{(l)} \in \mathbb{R}^d$ |
| $d$ | Hidden dimension | $d = 768$ (GPT-2 Small) |
| $\text{SAE}$ | Sparse autoencoder | encoder + decoder |
| $W_{\text{enc}}$ | SAE encoder weight matrix | $W_{\text{enc}} \in \mathbb{R}^{m \times d}$ |
| $W_{\text{dec}}$ | SAE decoder weight matrix | $W_{\text{dec}} \in \mathbb{R}^{d \times m}$ |
| $m$ | SAE dictionary size | $m = 24,576$ |
| $b_{\text{enc}}$ | SAE encoder bias | $b_{\text{enc}} \in \mathbb{R}^m$ |
| $b_{\text{dec}}$ | SAE decoder bias | $b_{\text{dec}} \in \mathbb{R}^d$ |
| $\phi$ | SAE feature activation (post-ReLU) | $\phi \in \mathbb{R}^m$ |
| $\phi_j$ | Activation of feature $j$ | $\phi_j \in \mathbb{R}_{\geq 0}$ |

## Feature Sets and Concepts

| Symbol | Definition |
|--------|-----------|
| $\mathcal{F}$ | Set of all SAE features | $|\mathcal{F}| = m$ |
| $\mathcal{F}_{\text{active}}$ | Subset of active features analyzed | $|\mathcal{F}_{\text{active}}| = 500$ |
| $\mathcal{F}_{\text{dead}}$ | Dead features (near-zero variance) | $\mathcal{F}_{\text{dead}} \subset \mathcal{F}$ |
| $c$ | A concept (e.g., "three", ".") | $c \in \mathcal{C}$ |
| $\mathcal{C}$ | Set of all concepts in a hierarchy | |
| $\mathcal{C}_{\text{num}}$ | Number concepts | $\{ \text{one}, \text{two}, \ldots, \text{eight} \}$ |
| $\mathcal{C}_{\text{punct}}$ | Punctuation concepts | $\{ \text{.}, \text{,}, \text{!}, \text{?}, \text{;}, \text{:}, \text{"}, \text{'} \}$ |
| $\mathcal{C}_{\text{case}}$ | Case concepts | $\{ \text{a}, \text{A}, \ldots, \text{z}, \text{Z} \}$ |
| $\text{topK}(c)$ | Top-K activating features for concept $c$ | $\text{topK}(c) \subset \mathcal{F}$ |
| $K$ | Top-K parameter | $K = 5$ (default) or $K = 10$ |

## Absorption and Ground Truth

| Symbol | Definition |
|--------|-----------|
| $p$ | Parent concept (e.g., "number") | |
| $c_1, c_2$ | Child concepts (e.g., "three", "four") | |
| $\mathcal{A}(c)$ | Absorption feature set for concept $c$ | $\mathcal{A}(c) \subset \mathcal{F}$ |
| $\rho_{\text{abs}}(c_1, c_2)$ | True absorption rate between $c_1$ and $c_2$ | $\rho_{\text{abs}} \in [0, 1]$ |
| $\rho_{\text{coll}}(c_1, c_2)$ | Collision rate between $c_1$ and $c_2$ | $\rho_{\text{coll}} \in [0, 1]$ |

## UAD Pipeline

| Symbol | Definition |
|--------|-----------|
| $\Phi$ | Co-occurrence matrix (phi coefficients) | $\Phi \in \mathbb{R}^{|\mathcal{F}_{\text{active}}| \times |\mathcal{F}_{\text{active}}|}$ |
| $\phi_{ij}$ | Phi coefficient between features $i$ and $j$ | $\phi_{ij} \in [-1, 1]$ |
| $k$ | Number of clusters | $k = 50$ |
| $\mathcal{C}_1, \ldots, \mathcal{C}_k$ | Cluster partition of features | $\bigcup_{i=1}^k \mathcal{C}_i = \mathcal{F}_{\text{active}}$ |
| $\mathcal{P}_{\text{UAD}}$ | Pairs detected by UAD | $\mathcal{P}_{\text{UAD}} = \bigcup_{i=1}^k \{ (f_a, f_b) : f_a, f_b \in \mathcal{C}_i, a \neq b \}$ |
| $\mathcal{P}_{\text{GT}}$ | Ground truth absorption pairs | $\mathcal{P}_{\text{GT}} \subset \mathcal{F} \times \mathcal{F}$ |

## Metrics

| Symbol | Definition | Formula |
|--------|-----------|---------|
| $\text{Jaccard}(A, B)$ | Jaccard similarity | $\frac{|A \cap B|}{|A \cup B|}$ |
| $\text{Prec}$ | Precision | $\frac{|\mathcal{P}_{\text{UAD}} \cap \mathcal{P}_{\text{GT}}|}{|\mathcal{P}_{\text{UAD}}|}$ |
| $\text{Rec}$ | Recall | $\frac{|\mathcal{P}_{\text{UAD}} \cap \mathcal{P}_{\text{GT}}|}{|\mathcal{P}_{\text{GT}}|}$ |
| $F_1$ | F1 score | $2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ |
| $\text{TP}$ | True positives | $|\mathcal{P}_{\text{UAD}} \cap \mathcal{P}_{\text{GT}}|$ |
| $\text{FP}$ | False positives | $|\mathcal{P}_{\text{UAD}} \setminus \mathcal{P}_{\text{GT}}|$ |
| $\text{FN}$ | False negatives | $|\mathcal{P}_{\text{GT}} \setminus \mathcal{P}_{\text{UAD}}|$ |
| $r_s$ | Spearman rank correlation | |
| $r_p$ | Pearson correlation | |
| $\text{CI}_{95}$ | 95% confidence interval | Bootstrap percentile method |

## Statistical Notation

| Symbol | Definition |
|--------|-----------|
| $n$ | Sample size | |
| $B$ | Number of bootstrap samples | $B = 1,000$ |
| $H_0$ | Null hypothesis | |
| $p$ | p-value (reported for reference only) | |

## Algorithm Parameters

| Symbol | Definition | Value |
|--------|-----------|-------|
| $\tau_{\text{dead}}$ | Dead feature variance threshold | near-zero |
| $\text{linkage}$ | Clustering linkage method | Ward (default), single (ablation) |
| $\text{seed}$ | Random seed | 42 |

---

## Naming Conventions

- Use **feature absorption** (not "absorbed features" as noun phrase)
- Use **co-occurrence clustering** (hyphenated)
- Use **collision rate** (not "feature collision")
- Use **top-K** (not "top k" or "TopK")
- Use **ground truth** (not "ground-truth" as adjective before noun, but "ground-truth labels" is acceptable)
- Use **phi coefficient** (lowercase phi, not "Phi")
- Use **hierarchical clustering** (not "hierarchic clustering")
- Use **Jaccard similarity** (not "Jaccard index" unless specifically referring to the index form)
- Use **mutual exclusivity** (not "mutual exclusion")
- Use **token-level** (hyphenated when used as modifier: "token-level activations")
