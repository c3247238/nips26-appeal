# Glossary

Unified terminology definitions for all paper sections. Writers, critics, and the editor must reference this file for consistency.

---

## Core Concepts

| Term | Definition | Notes |
|------|-----------|-------|
| **Feature absorption** | A failure mode where a general (parent) SAE feature fails to activate when a specific (child) feature co-occurs, because the SAE achieves better sparsity by encoding the parent's information into the child's decoder direction. | Preferred phrasing: "feature absorption" (not "absorption" alone when ambiguous). First use: define fully. |
| **Feature hedging** | A failure mode where an SAE merges correlated features into a single latent due to insufficient dictionary width, causing systematic false negatives for the individual features. | Distinct from absorption: hedging is caused by width limitations, absorption by hierarchical feature structure. |
| **Feature suppression** | The observed effect of absorption during activation patching: zeroing a child feature allows the parent feature's probe prediction to recover. Used when describing the causal mechanism without committing to a specific mechanistic explanation. | Preferred over "competitive exclusion" in empirical claims. |
| **Superposition** | The phenomenon where neural networks encode more features than they have neurons by representing features as overlapping linear directions. | Standard usage from Elhage et al. (2022). |
| **Monosemanticity** | The property of a neuron or SAE latent responding to a single, interpretable concept. | Preferred: "monosemanticity" (not "mono-semanticity"). |
| **Polysemanticity** | The property of a neuron responding to multiple unrelated concepts. The complement of monosemanticity. | Preferred: "polysemanticity" (not "poly-semanticity"). |

## SAE Architecture Terms

| Term | Definition | Notes |
|------|-----------|-------|
| **Sparse autoencoder (SAE)** | An autoencoder trained with a sparsity-inducing objective to decompose neural activations into an overcomplete set of approximately monosemantic features. | Abbreviation: SAE. Define on first use. |
| **Latent** | A single learned feature direction in the SAE dictionary. Synonymous with "SAE feature" or "dictionary element." | Preferred: "latent" (consistent with SAEBench/SAELens terminology). Acceptable: "SAE feature." Avoid: "neuron" (reserved for model neurons). |
| **Dictionary** | The set of all decoder directions in an SAE. The dictionary size $M$ is the number of latents. | Preferred: "dictionary" (not "codebook"). |
| **JumpReLU SAE** | SAE architecture using the JumpReLU activation function, which applies a learnable threshold before ReLU. Trains $L_0$ directly via straight-through estimator. | Rajamanoharan et al. (2024). The Gemma Scope SAEs use this architecture. |
| **BatchTopK SAE** | SAE architecture that selects the top-$k$ latents per batch (rather than per sample), allowing variable sparsity per input while maintaining a target average $L_0$. | Bussmann (2024). Basis for Matryoshka SAE. |
| **Matryoshka SAE** | SAE trained with nested prefix losses at increasing dictionary sizes, creating a natural feature hierarchy where smaller dictionaries learn general features. | Bussmann et al. (2025). Named after Matryoshka nesting dolls. |
| **$L_0$** | The expected number of active (non-zero) latents per input token. A measure of SAE sparsity. | Always formatted as $L_0$ in math mode. Not "L0" in running text. |

## Feature Hierarchy Terms

| Term | Definition | Notes |
|------|-----------|-------|
| **Parent feature** | A general feature in a hierarchy that applies to a broad class (e.g., "starts with s"). | |
| **Child feature** | A specific feature that implies the parent (e.g., "snake" implies "starts with s"). | |
| **Feature hierarchy** | A directed graph where child features imply parent features. In an absorption context, child features can absorb parent features' activation. | |
| **First-letter hierarchy** | The syntactic hierarchy mapping tokens to their initial letter. 26 classes, near-uniform distribution. The canonical absorption evaluation task. | Preferred: "first-letter" (hyphenated). |
| **RAVEL hierarchies** | Entity-attribute hierarchies from the RAVEL dataset: city-continent (6 classes), city-country (80 classes), city-language (50 classes). | Always specify which RAVEL hierarchy. |

## Measurement Terms

| Term | Definition | Notes |
|------|-----------|-------|
| **Absorption rate** | The fraction of feature classes exhibiting at least one absorbed false negative, measured via integrated-gradients attribution on SAE false negatives. | Based on Chanin et al. (2024) metric. |
| **False negative (FN)** | A token where the linear probe correctly classifies the raw activation but incorrectly classifies the SAE-reconstructed activation. | In the context of absorption measurement. Not general ML usage. |
| **Integrated gradients (IG)** | An attribution method that integrates the gradient along a straight path from a baseline (zero) to the actual activation, assigning importance to each SAE latent for a given classification. | Sundararajan et al. (2017). Abbreviated: IG. |
| **Activation patching** | An interventional method where a specific SAE latent's activation is set to zero, and the downstream effect on probe classification is measured. Provides causal (not merely correlational) evidence. | Preferred: "activation patching" (not "causal tracing" or "interchange intervention"). |
| **Recovery rate** | In activation patching: the fraction of absorbed tokens where zeroing the child feature causes the probe to correctly classify the parent. | |
| **Strict hedging** | A false negative that resolves at higher $L_0$ specifically because the parent feature reactivates. | Our three-way decomposition. |
| **Compensatory resolution** | A false negative that resolves at higher $L_0$ because other (non-parent) features add sufficient information, not because the parent feature itself recovers. | Our three-way decomposition. |
| **Persistent false negative** | A false negative that does not resolve even at substantially higher $L_0$. | Our three-way decomposition. |

## Detector Abbreviations

| Term | Definition | Notes |
|------|-----------|-------|
| **GAS** | Geometric Absorption Score. An unsupervised metric based on decoder-activation co-occurrence mismatch. Negative result in this paper (rho=0.116). | |
| **CMI** | Conditional Mutual Information. An information-theoretic measure tested as an absorption predictor. Negative result (rho=0.044). | |
| **Absorption Tax** | The theoretical minimum additional $L_0$ cost $T(G)$ for absorption-free representation of a feature hierarchy $G$. Ranking predictions not supported empirically (rho=-0.20). | Capitalize "Absorption Tax" as a proper noun when referring to our formulation. |

## Datasets and Models

| Term | Definition | Notes |
|------|-----------|-------|
| **Gemma 2 2B** | Google's 2-billion parameter language model from the Gemma 2 family. | Preferred: "Gemma 2 2B" (not "Gemma-2-2B" or "Gemma2 2B"). |
| **Gemma Scope** | Google DeepMind's collection of 400+ open JumpReLU SAEs trained on all layers of Gemma 2 models. | Lieberum et al. (2024). |
| **SAEBench** | A comprehensive 8-metric evaluation benchmark for SAEs, including an absorption task. | Karvonen et al. (2025). Preferred: "SAEBench" (not "SAE Bench" or "SAE-Bench"). |
| **RAVEL** | A dataset of entity-attribute pairs for evaluating representation quality. Used here for city-continent, city-country, and city-language hierarchies. | Dataset: `hij/ravel` on HuggingFace. |
| **SAELens** | The standard library for training and evaluating SAEs, integrated with TransformerLens. | Bloom et al. (2024). Preferred: "SAELens" (not "sae-lens" in text). |
| **TransformerLens** | A library for mechanistic interpretability providing hook-based activation caching and editing for 50+ transformer models. | Preferred: "TransformerLens" (CamelCase). |

## Statistical Terms

| Term | Definition | Notes |
|------|-----------|-------|
| **Bootstrap CI** | Confidence interval computed by resampling with replacement. We use 10,000 resamples throughout. | Always specify "95% bootstrap CI." |
| **Cohen's $d$** | Standardized effect size for two-group comparisons. | Report to 2 decimal places. |
| **Kruskal-Wallis test** | Non-parametric one-way ANOVA on ranks. Used for comparing absorption rates across hierarchy types or architectures. | Preferred over parametric ANOVA given non-normal distributions. |
| **Wilcoxon signed-rank test** | Non-parametric paired test. Used for activation patching (child recovery vs control recovery within each word). | |
| **Permutation test** | Non-parametric test using random permutation of labels to construct the null distribution. Used for pairwise cross-domain comparisons. | |

## Style Preferences

| Preferred | Avoid |
|-----------|-------|
| fine-tuning | finetuning, fine tuning |
| few-shot | few shot, fewshot |
| first-letter | first letter (when used as modifier) |
| cross-domain | cross domain (when used as modifier) |
| layer-dependent | layer dependent (when used as modifier) |
| $L_0$ (in math mode) | L0 (in running text) |
| sparse autoencoder (SAE) | Sparse Auto-Encoder, sparse auto-encoder |
| latent | dictionary atom, code, SAE neuron |
| absorption rate | absorption score, absorption metric |
| Gemma 2 2B | gemma-2-2b (except in code) |
| Gemma Scope | GemmaScope, Gemma-Scope |
| SAEBench | SAE Bench, SAE-Bench |
| SAELens | sae-lens (except in code/citations) |
