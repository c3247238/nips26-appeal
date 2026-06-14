# Glossary

## Core Technical Terms

**Feature absorption**
: A pathology in Sparse Autoencoders where parent features in semantic hierarchies are subsumed by their child features under sparsity pressure, creating gaps in feature coverage.

**Sparse Autoencoder (SAE)**
: An autoencoder trained to reconstruct neural network activations using a sparse latent representation, used for mechanistic interpretability.

**Mechanistic interpretability**
: The study of understanding how neural networks compute their outputs by decomposing them into interpretable components.

**L0 sparsity**
: The average number of active (non-zero) latent features per input token. Lower L0 means sparser representations.

**True L0**
: L0 computed after removing dead latents (latents that never activate). Provides a sparsity measure for the active sub-dictionary only.

**Dead latent**
: A latent feature that never activates above a threshold across the entire evaluation dataset, indicating it has not learned a useful representation.

**Feature recovery**
: The degree to which SAE latents align with ground-truth features, typically measured via optimal assignment (Hungarian algorithm).

**Ground-truth feature**
: In synthetic data, a feature with known semantics and hierarchical relationships, enabling exact measurement of SAE performance without approximation.

## Architecture Terms

**Baseline L1 (ReLU+L1 SAE)**
: Standard SAE architecture using ReLU activation and L1 regularization on latent activations to induce sparsity. Also called "soft sparsity" because L1 penalizes but does not strictly bound the number of active features.

**TopK SAE**
: SAE architecture that explicitly selects only the top-k latent activations per input, setting all others to zero. Enforces "hard sparsity" with a fixed, deterministic L0.

**Matryoshka SAE**
: SAE architecture with nested multi-scale dictionaries that learn features at multiple resolutions. Inner levels act as narrower SAEs.

**OrtSAE (Orthogonal Sparse Autoencoder)**
: SAE architecture that enforces chunk-wise orthogonality constraints on decoder weights via a penalty term, claimed to reduce absorption.

**Gated SAE**
: SAE architecture with separate gate and magnitude pathways for better sparsity control.

**JumpReLU SAE**
: SAE architecture with learnable per-feature activation thresholds (jump discontinuities) for improved reconstruction fidelity.

## Sparsity and Regularization Terms

**Hard sparsity**
: Sparsity enforced by explicitly selecting a fixed number of active features (e.g., TopK k-selection). Contrast with soft sparsity.

**Soft sparsity**
: Sparsity induced by penalizing non-zero activations (e.g., L1 regularization) without strictly bounding the number of active features. Also called "L1-regularized sparsity."

**Feature hedging**
: A phenomenon where correlated features share a single latent direction rather than being assigned independent representations, common in narrow SAEs.

**Overcomplete dictionary**
: An SAE where the number of latents ($d_{\text{sae}}$) exceeds the number of ground-truth features ($n$). Our experiments use $d_{\text{sae}} = 2048$ with $n = 1024$.

## Evaluation Terms

**Matthews Correlation Coefficient (MCC)**
: A correlation coefficient between predicted and true binary classifications, ranging from -1 (perfect disagreement) to +1 (perfect agreement). Used here to measure feature recovery via optimal latent-to-feature matching.

**Explained variance (EV)**
: The proportion of variance in the input data explained by the reconstruction: $1 - \text{Var}(x - \hat{x}) / \text{Var}(x)$.

**Reconstruction MSE**
: Mean squared error between input activations and SAE reconstructions, measuring how well the SAE preserves activation information.

**Mutual coherence**
: Maximum absolute inner product between pairs of normalized dictionary atoms, measuring dictionary redundancy.

## Statistical Terms

**Welch's t-test**
: A two-sample t-test that does not assume equal variances, used for comparing absorption rates across SAE variants.

**Cohen's d**
: A standardized effect size measure: the difference between two means divided by a pooled standard deviation.

**Pearson correlation**
: A measure of linear correlation between two variables, ranging from -1 to +1.

**Bonferroni correction**
: A method for adjusting significance thresholds when performing multiple comparisons, controlling the family-wise error rate.

**ANOVA**
: Analysis of Variance; a statistical test for differences among group means.

**95% confidence interval (CI)**
: An interval estimate that, under repeated sampling, would contain the true parameter value 95% of the time.

## Experimental Terms

**L0-matching protocol**
: Our proposed methodology of tuning the L1 penalty ($\lambda$) to match the L0 sparsity of a target architecture before comparing absorption rates.

**Dose-response study**
: An experimental design where a single variable (here, $\lambda$) is systematically varied to observe its effect on outcomes (absorption and MCC), analogous to dose-response curves in pharmacology.

**Synthetic hierarchical data**
: Artificially generated data with known parent-child feature hierarchies, enabling exact ground-truth measurement of absorption without probe-based approximations.

**Seed**
: A random number generator initialization value. We use 5 seeds (42, 123, 456, 789, 1011) for statistical replication.

## Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| LLM | Large Language Model |
| MCC | Matthews Correlation Coefficient |
| MSE | Mean Squared Error |
| EV | Explained Variance |
| ANOVA | Analysis of Variance |
| CI | Confidence Interval |
| RQ | Research Question |
| SAEBench | Sparse Autoencoder Benchmark |
| CE-Bench | Contrastive Evaluation Benchmark |
| GBA | Group Bias Adaptation |
| HSAE | Hierarchical Sparse Autoencoder |
| ReLU | Rectified Linear Unit |
| PCA | Principal Component Analysis |
| MD5 | Message-Digest Algorithm 5 (hash function) |
| JSON | JavaScript Object Notation |
| GPU | Graphics Processing Unit |

## Preferred Phrasing

| Preferred | Avoid |
|-----------|-------|
| fine-tuning | finetuning |
| few-shot | few shot |
| L0-matched | L0 matched (without hyphen) |
| absorption rate | absorption (when referring to the metric) |
| feature recovery | feature reconstruction (when referring to MCC) |
| ground-truth | ground truth (as adjective: ground-truth features) |
| dead latent rate | dead latents (use "percentage" or "rate" for clarity) |
| sparsity level | sparsity (when L0 is implied) |
| k-selection | k-selection (not "K-selection") |
| lambda sweep | lambda sweep (not "lambda-sweep") |
| dose-response | dose-response (hyphenated) |
| statistically indistinguishable | numerically similar (unless formal test reported) |
| does not support | falsifies (reserve "falsifies" for sensitive metrics with genuine nulls) |
