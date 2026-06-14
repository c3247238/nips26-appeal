# Glossary

This document unifies key terminology definitions used throughout the paper. All section writers must reference this file for consistency.

---

## Core Technical Terms

### Sparse Autoencoder (SAE)
An autoencoder with a sparsity constraint on the latent representation, trained to reconstruct activations from a neural network's hidden layers. In mechanistic interpretability, SAEs decompose high-dimensional activations into sparse, interpretable features. Each latent dimension typically corresponds to a human-interpretable concept.

### Feature Absorption
A failure mode in SAEs where a parent feature (encoding a broad concept) is suppressed when child features (encoding sub-concepts) are present. The SAE allocates its activation budget to the more specific child features, making the parent appear weak or dead. This creates interpretability illusions: the parent feature exists in the SAE but fails to activate on arbitrary positive examples. Defined by Chanin et al. (2024).

### Token-Level Mutual Exclusivity
The property that two features never activate on the same token position. In tested token-disjoint hierarchies (numbers, punctuation), absorption features exhibit this property because child concepts (e.g., "three" and "four") never appear at the same position.

### Token-Disjoint Hierarchy
A concept hierarchy where child concepts never co-occur at the same token position. Example: number tokens ("one" through "eight") are token-disjoint because a single token position cannot be both "three" and "four". Contrast with semantic hierarchies (e.g., "animal" and "dog") where children may co-occur in natural text.

### Co-occurrence Clustering
A clustering method that groups features based on how often they activate on the same tokens. The UAD method uses phi coefficients (Pearson correlation of binary activations) as the similarity measure for hierarchical clustering.

---

## Method-Specific Terms

### Unsupervised Absorption Detection (UAD)
A training-free method proposed by Chanin et al. (2024) for detecting absorbed feature pairs without supervised ground truth. Pipeline: (1) dead feature filtering, (2) phi coefficient co-occurrence matrix computation, (3) hierarchical clustering, (4) specificity filtering, (5) pair extraction from clusters.

### Collision Rate
The Jaccard similarity of top-K activating feature sets for two concepts: $\text{CR}(c_1, c_2) = \text{Jaccard}(\text{topK}(c_1), \text{topK}(c_2))$. In this paper, collision rate and absorption rate are the same quantity computed from the same data. Used as an operationalization of absorption.

### Operationalization
The process of defining a theoretical concept (feature absorption) in terms of concrete, measurable operations (top-K feature overlap). An operationalization is "internally consistent" if it produces stable, expected patterns across diverse cases. Distinguished from "proxy validation," which would require an independent ground truth.

### Top-K Features
For a given concept token, the K SAE features with the highest activation magnitudes when that token is processed. Default K = 5 in this paper.

### Absorption Rate
The proportion of features that absorb one concept and also absorb another concept. Computed as Jaccard similarity of the top-K feature sets. Synonymous with collision rate in this paper.

### Phi Coefficient
The Pearson correlation coefficient computed on binary activation vectors. In UAD, phi measures whether two features tend to activate on the same tokens. Range: [-1, 1].

### Dead Feature
An SAE feature with near-zero activation variance across the corpus. UAD filters these out before clustering.

---

## Model and Architecture Terms

### GPT-2 Small
A 124M parameter transformer language model with 12 layers and 768 hidden dimensions. Used as the base model for all experiments.

### gpt2-small-res-jb
A pretrained SAE for GPT-2 Small layer 8 (residual stream post), trained with JumpReLU activation. d_SAE = 24,576. Available via SAELens.

### JumpReLU
A variant of the ReLU activation function used in some SAEs. Features a learned threshold below which activations are exactly zero.

### Residual Stream
The main information pathway in a transformer, where layer outputs are added to the input via residual connections. The SAE operates on post-layer residual stream activations.

### Hook Point
A named location in a neural network where activations can be intercepted and analyzed. TransformerLens provides hook points like `blocks.8.hook_resid_post`.

---

## Evaluation Terms

### Ground Truth (GT)
In this paper, the set of concept pairs known to be in an absorption relationship based on our operationalization (top-K feature overlap). Not an independently verified truth but a consistently defined standard for evaluation.

### True Positive (TP)
A concept pair correctly identified as absorbed by the detection method.

### False Positive (FP)
A concept pair incorrectly flagged as absorbed by the detection method.

### False Negative (FN)
A concept pair that is actually absorbed but missed by the detection method.

### Precision
The fraction of detected pairs that are true positives: TP / (TP + FP).

### Recall
The fraction of true absorption pairs that are detected: TP / (TP + FN).

### F1 Score
The harmonic mean of precision and recall: 2 * (P * R) / (P + R).

### Bootstrap 95% CI
A confidence interval computed by resampling the data with replacement 1,000 times and computing the metric on each resample. The 2.5th and 97.5th percentiles form the 95% CI.

### Spearman Rank Correlation ($\rho$)
A non-parametric measure of rank correlation between two variables. Range [-1, 1]. Used to assess the relationship between collision rate and absorption rate without assuming linearity.

---

## Preferred Phrasing

| Preferred | Avoid | Reason |
|-----------|-------|--------|
| fine-tuning | finetuning | Standard spelling |
| few-shot | few shot | Compound adjective |
| ground truth | ground-truth (as noun) | Hyphenate only as modifier |
| co-occurrence | cooccurrence | Standard spelling |
| token-level | token level (as modifier) | Hyphenate as compound adjective |
| token-disjoint | token disjoint | Hyphenate as compound adjective |
| operationalization | proxy (for collision rate) | Critical distinction in this paper |
| internal consistency | validation (for collision rate) | Critical distinction in this paper |
| tested token-disjoint hierarchies | all hierarchies | Honest scoping |
| features that fire | features that activate | Either acceptable; be consistent within a section |
| SAE feature | latent dimension | Use "feature" for interpretability context |
| absorption pair | absorbed pair | Use "absorption pair" for the relationship |

---

## Abbreviations and Expansions

| Abbreviation | Expansion | First Use |
|-------------|-----------|-----------|
| SAE | Sparse Autoencoder | Spell out on first use, then SAE |
| UAD | Unsupervised Absorption Detection | Spell out on first use, then UAD |
| GT | Ground Truth | Spell out on first use, then GT (in methods/results only) |
| CI | Confidence Interval | Spell out on first use |
| TP/FP/FN/TN | True/False Positive/Negative | Spell out on first use |
| LLM | Large Language Model | Spell out on first use |
| res-jb | residual JumpReLU | Explain on first use with SAE name |
| d_SAE | SAE latent dimension | Define in notation table, use freely |
| d_model | Model hidden dimension | Define in notation table, use freely |

---

## Domain-Specific Terms (for Non-Specialist Readers)

### Mechanistic Interpretability
The subfield of AI research that aims to reverse-engineer neural networks to understand how they compute. SAEs are a primary tool in this field.

### Interpretability Illusion
A situation where an SAE feature appears to track a meaningful concept but actually fails to activate on all instances of that concept. Feature absorption creates such illusions.

### Activation Patching (Causal Intervention)
A technique from mechanistic interpretability where activations at specific locations are manually modified (e.g., zeroed out) to test causal dependencies between components.

### Decoder Weight
The weight matrix in an SAE that maps from the latent space back to the activation space. Each column corresponds to the direction in activation space associated with one feature.

---

## Terms to Avoid

| Avoid | Use Instead | Reason |
|-------|-------------|--------|
| "novel" (unqualified) | specific contribution description | Banned pattern |
| "groundbreaking" | -- | Banned pattern |
| "significantly outperforms" | exact numbers (e.g., "improves F1 by X") | Banned pattern |
| "proxy validation" (for collision rate) | "internal consistency of operationalization" | Critical distinction |
| "UAD is useless" | "UAD fails for token-disjoint hierarchies" | Honest, scoped |
| "absorption features ARE mutually exclusive" | "absorption features EXHIBIT token-level mutual exclusivity" | Scoped, evidence-based |
| "manual inspection of 50 false positives" | -- | Fabricated claim; removed |
| "to the best of our knowledge" | -- | Banned pattern |
| "moreover", "furthermore" | direct transition or omit | Banned pattern |
| "it is worth noting that" | omit or rephrase | Banned pattern |
