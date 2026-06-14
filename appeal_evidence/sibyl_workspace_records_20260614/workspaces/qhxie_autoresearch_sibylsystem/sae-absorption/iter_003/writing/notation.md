# Mathematical Notation Table

Established notation for all paper sections. Section writers, critics, and the editor must reference this file for consistency.

---

## 1. Model and Activation Notation

| Symbol | Definition | Dimensionality | Notes |
|---|---|---|---|
| `x` | Residual stream activation at a given layer | `x ∈ ℝ^d_model` | `d_model = 768` for GPT-2-small |
| `L` | Transformer layer index (0-indexed) | integer | Primary experiments at L6 |
| `d_model` | Residual stream dimension | integer | 768 (GPT-2-small) |
| `T` | Number of tokens in dataset | integer | 10,000 (absorption); 200,000 (co-occurrence) |

---

## 2. SAE Architecture Notation

| Symbol | Definition | Dimensionality | Notes |
|---|---|---|---|
| `d_SAE` | SAE dictionary size (number of features/latents) | integer | 24,576 (Standard/L1); 32,768 (TopK-32k) |
| `W_enc` | Encoder weight matrix | `W_enc ∈ ℝ^{d_SAE × d_model}` | Row j is the encoder direction for latent j |
| `W_enc_j` | j-th row of encoder matrix (encoder direction for latent j) | `W_enc_j ∈ ℝ^{d_model}` | |
| `b_enc` | Encoder bias vector | `b_enc ∈ ℝ^{d_SAE}` | |
| `W_dec` | Decoder weight matrix (dictionary atoms) | `W_dec ∈ ℝ^{d_model × d_SAE}` | Column j is decoder direction for latent j |
| `d_j` | Decoder direction for latent j (j-th column of W_dec) | `d_j ∈ ℝ^{d_model}` | Unit-normalized in standard SAE training |
| `b_dec` | Decoder bias vector | `b_dec ∈ ℝ^{d_model}` | Pre-activation bias subtracted from input |
| `z` | SAE latent activation vector | `z ∈ ℝ^{d_SAE}` | Sparse; most entries zero |
| `z_j` | Activation of the j-th SAE latent | scalar | `z_j ≥ 0` after ReLU |
| `K` | Target number of active features (TopK or matched-L0) | integer | `K = 53` in OMP oracle experiment |

### SAE Encoding Variants

| Symbol | Definition |
|---|---|
| `z = f_FF(x)` | Feedforward encoding: `z = ReLU(W_enc (x - b_dec) + b_enc)` |
| `z = f_OMP(x, K)` | OMP oracle encoding: Orthogonal Matching Pursuit with K steps on dictionary W_dec |
| `L0(z)` | Sparsity of activation vector z: number of non-zero entries |
| `L̄0` | Mean L0 across a dataset (for matching sparsity across conditions) |

---

## 3. Absorption Notation

| Symbol | Definition | Notes |
|---|---|---|
| `𝒜` | Set of absorbed SAE latents (gold-standard labels) | `\|𝒜\| = 18` (GPT-2 L6 Standard, IG labels) |
| `𝒩` | Set of non-absorbed SAE latents | `\|𝒩\| = d_SAE - \|𝒜\|` |
| `i` | Index of parent (absorbing) latent | Higher frequency; fires when child should fire |
| `j` | Index of child (absorbed) latent | Lower frequency; fails to fire on target tokens |
| `AR(c)` | Absorption rate for hierarchy level / letter `c` | Fraction of tokens where child feature fails to fire despite probe positive |
| `AR_FF` | Absorption rate under feedforward encoding | Baseline condition |
| `AR_OMP` | Absorption rate under OMP oracle encoding | Test condition |
| `r_OMP` | OMP reduction ratio: `(AR_FF - AR_OMP) / AR_FF` | `r_OMP = 0.0` in our experiment |

---

## 4. Detection Score Notation

| Symbol | Definition | Formula | Notes |
|---|---|---|---|
| `enc_norm(j)` | Encoder weight norm of latent j | `enc_norm(j) = ‖W_enc_j‖₂` | Primary detector |
| `EDA(j)` | Encoder-Decoder Angle for latent j | `EDA(j) = 1 - cos(W_enc_j / ‖W_enc_j‖, d_j)` | Baseline detector from iter_001/002 |
| `f_j` | Activation frequency of latent j | `f_j = P(z_j > 0)` over the dataset | Estimated empirically |
| `𝒮_j` | Activation set of latent j | `𝒮_j = {t : z_j^(t) > 0}` | Set of token indices where j activates |
| `O_jaccard(j, i)` | Jaccard overlap between latents j and i | `O_jaccard(j,i) = \|𝒮_j ∩ 𝒮_i\| / \|𝒮_j ∪ 𝒮_i\|` | Co-occurrence detector component |
| `O_jaccard(j)` | Maximum Jaccard overlap for latent j | `max_{i: f_i > 3f_j} O_jaccard(j, i)` | Per-latent co-occurrence score |
| `A_cooccur(j, i)` | Directed co-occurrence asymmetry | `P(z_j > 0 \| z_i > 0) / P(z_i > 0 \| z_j > 0)` | Bounded by `f_j/f_i` when `f_i > 3f_j` |
| `ARS_v2(j)` | Revised Absorption Risk Score | `enc_norm(j) × A_cooccur(j)` | Tested; does NOT improve over enc_norm alone |
| `AFS(j)` | Absorption Fingerprint Score (original proposal) | `O(j,i) × A(j,i) × cos²(d_i, d_j)` | Falsified; not used in paper |

---

## 5. Evaluation Metrics

| Symbol | Definition | Notes |
|---|---|---|
| AUROC | Area Under the Receiver Operating Characteristic Curve | Primary ranking metric |
| AUPRC | Area Under the Precision-Recall Curve | More informative at extreme class imbalance; report alongside AUROC |
| P@k | Precision at top-k (k=50, 100, 500) | Practical retrieval metric |
| `n_pos` | Number of positive (absorbed) examples | 18 (L6 IG labels); 77 (L6 TopK-32k labels) |
| `n_neg` | Number of negative (non-absorbed) examples | `d_SAE - n_pos` |
| `d` | Cohen's d effect size | `(μ_absorbed - μ_all) / σ_all`; 0.971 (Standard), 1.235 (TopK-32k) |
| CI₉₅ | 95% bootstrap confidence interval | Report for all AUROC values |
| DeLong | DeLong AUROC comparison test (z-statistic, p-value) | For paired AUROC comparisons |

---

## 6. Width Recovery Notation

| Symbol | Definition | Notes |
|---|---|---|
| `d_narrow` | Width of narrow SAE | 24,576 (Standard/L1) |
| `d_wide` | Width of wider SAE | 32,768 (TopK-32k) |
| `cos_sim(d_j^narrow, d_k^wide)` | Cosine similarity between decoder directions | Used to find recovered features |
| `θ_rec` | Recovery threshold cosine similarity | `θ_rec = 0.80` |
| `𝒜_rec` | Set of recovered absorbed features | `𝒜_rec = {j ∈ 𝒜 : max_k cos_sim(d_j^narrow, d_k^wide) > θ_rec}` |

---

## 7. Abbreviations

| Abbreviation | Full Form |
|---|---|
| SAE | Sparse Autoencoder |
| LM | Language Model |
| GPT-2 | Generative Pre-trained Transformer 2 (Radford et al., 2019) |
| L6 | Transformer layer 6 (0-indexed) |
| IG | Integrated Gradients |
| OMP | Orthogonal Matching Pursuit |
| EDA | Encoder-Decoder Angle |
| ARS | Absorption Risk Score |
| AFS | Absorption Fingerprint Score |
| AUROC | Area Under the ROC Curve |
| AUPRC | Area Under the Precision-Recall Curve |
| MI | Mechanistic Interpretability |
| SDL | Sparse Dictionary Learning |
| L1 | L1 regularization (sparsity penalty) |
| TopK | Top-K activation function (retain K largest activations) |
| ReLU | Rectified Linear Unit |
| CI | Confidence Interval |
| FF | Feedforward (encoding condition label) |

---

## 8. Model Identifiers (for reproducibility)

| Identifier | Description | Source |
|---|---|---|
| `gpt2-small-res-jb` | GPT-2-small Standard/L1 SAE, all layers | Bloom et al. via SAELens |
| `gpt2-small-resid-post-v5-32k` | GPT-2-small TopK-32k SAE, residual stream | EleutherAI via SAELens |
| `blocks.6.hook_resid_pre` | Hook name for GPT-2 L6 residual stream (pre-MLP) | TransformerLens notation |
| `blocks.6.hook_resid_post` | Hook name for GPT-2 L6 residual stream (post-MLP) | TransformerLens notation |
