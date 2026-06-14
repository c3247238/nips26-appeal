# Glossary

## Core Concepts

**Feature Absorption**
: A pathology in sparse autoencoders where a general (parent) feature is subsumed by a more specific (child) feature during sparse optimization, causing the parent latent to have systematic false negatives. First systematically studied by Chanin et al. (2024).

**Feature Hedging**
: A related SAE pathology where correlated features are merged together in narrow SAEs, producing latents that activate on multiple unrelated concepts. Identified by Chanin et al. (2025a).

**Mechanistic Interpretability**
: The subfield of AI research focused on reverse-engineering neural networks to understand their internal computations at the level of individual features and circuits.

**Sparse Autoencoder (SAE)**
: An autoencoder with a sparsity constraint on the bottleneck layer, trained to decompose neural network activations into a sparse set of interpretable features. The dominant unsupervised approach for feature extraction in mechanistic interpretability.

## SAE Architectures

**JumpReLU SAE**
: An SAE architecture that applies a non-zero threshold before the ReLU activation, improving reconstruction fidelity. Used in GemmaScope. Introduced by Rajamanoharan et al. (2024).

**ReLU SAE**
: The standard SAE architecture using ReLU activation: $z = \text{ReLU}(W_{\text{enc}} x + b_{\text{enc}})$. Used in GPT-2 SAEs via SAELens.

**TopK SAE**
: An SAE architecture that explicitly selects the top-$k$ latents by activation magnitude, providing direct sparsity control. Introduced by Gao et al. (2024).

**Gated SAE**
: An SAE variant that addresses systematic underestimation from the L1 sparsity penalty by gating the activation. Introduced by Rajamanoharan et al. (2024).

## Metrics and Evaluation

**Ablation Score**
: The accuracy difference between a probe on the full model and on the model with the top latent ablated. A near-zero score suggests the latent is absorbed (the model compensates). Known to be functionally insensitive.

**Projection Absorption**
: The fraction of a linear probe's weight vector captured by its top-weighted latent: $A_{\text{proj}} = |w_{j^*}| / \sum_j |w_j|$. A high value indicates the feature is concentrated in a single latent.

**Projection Ratio**
: The complement of projection absorption: $R_{\text{proj}} = 1 - A_{\text{proj}}$. Measures how much of the probe weight is distributed across other latents.

**A_j (Training-Free Detector)**
: A metric computed from SAE weights without probe training: $A_j = \|d_j\|^2 / (d_j^\top e_j)$ where $e_j$ is the $j$-th encoder row. Proposed by Chanin et al. (2024) as a training-free proxy for absorption.

**AUROC (Area Under the Receiver Operating Characteristic Curve)**
: A measure of probe classification quality. AUROC > 0.7 indicates a valid probe; AUROC = 0.5 is random chance.

**L0 (Sparsity Level)**
: The average number of non-zero latents per token in an SAE. A key hyperparameter controlling the sparsity-fidelity trade-off.

## Datasets and Models

**GemmaScope**
: Google's open-source SAE suite for Gemma 2 models, providing pretrained SAEs for every layer and sublayer with 16k/65k/131k widths using JumpReLU architecture (Lieberum et al., 2024).

**SAELens**
: An open-source library providing pretrained SAEs for various models including GPT-2, with standardized loading and analysis interfaces.

**WordNet**
: A lexical database of semantic relations between words, organized into synsets (synonym sets). Used in this work to construct semantic probe categories and hyponym hierarchies.

**Semantic Probe**
: A linear classifier trained on SAE activations to detect whether a given semantic concept is present in the input. Used to measure whether a concept is "absorbed" into a single latent.

## Statistical Terms

**Spearman Rank Correlation ($\rho$)**
: A non-parametric measure of rank correlation between two variables. Used in this work to assess the relationship between A_j and projection absorption without assuming linearity.

**Cohen's d**
: A standardized measure of effect size: the difference between two means divided by the pooled standard deviation. Values: 0.2 (small), 0.5 (medium), 0.8 (large).

**Mann-Whitney U Test**
: A non-parametric test comparing two independent samples. Used when normality assumptions are not met.

**Fisher's r-to-z Transformation**
: A transformation converting correlation coefficients to approximately normal z-scores, enabling comparison of correlations from different samples.

## Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| LLM | Large Language Model |
| LRH | Linear Representation Hypothesis |
| MI | Mechanistic Interpretability |
| AUROC | Area Under the Receiver Operating Characteristic Curve |
| ReLU | Rectified Linear Unit |
| MLP | Multi-Layer Perceptron |
| SAEBench | Sparse Autoencoder Benchmark (Karvonen et al., 2025) |
| CE-Bench | Contrastive Evaluation Benchmark (Peng et al., 2025) |
| OrtSAE | Orthogonal Sparse Autoencoder (Korznikov et al., 2025) |
| MP-SAE | Matching Pursuit Sparse Autoencoder (Costa et al., 2025) |
| WSAE | Weighted Sparse Autoencoder (Cui et al., 2026) |
| ATM | Adaptive Temporal Masking (Li & Ren, 2025) |

## Preferred Phrasing

- "fine-tuning" (not "finetuning")
- "few-shot" (not "few shot")
- "training-free" (not "training free")
- "cross-architecture" (not "cross architecture")
- "decoder norm" (not "decoder norm constraint")
- "projection-based" (not "projection based")
- "ablation-based" (not "ablation based")
- "layer-dependent" (not "layer dependent")
- "mid-layer" (not "mid layer")
- "feature absorption" (not "absorption phenomenon" when used as noun phrase)
- "semantic probe" (not "probe" alone when first introduced)
- "hyponym" (a word with a more specific meaning than another, e.g., "dog" is a hyponym of "animal")
- "hypernym" (a word with a broader meaning, e.g., "animal" is a hypernym of "dog")
