# Glossary

> Unified terminology definitions for all section writers.  
> All subsequent sections must use the preferred phrasing listed here.

---

## Core Technical Terms

### Feature Absorption
The suppression of a parent feature by child features in a sparse autoencoder, occurring when hierarchical concepts co-occur in training data. Creates interpretability illusions where a latent appears to track a concept but fails to activate on arbitrary positive examples. Coined by Chanin et al. (2024).

- **Preferred**: "feature absorption" (noun), "absorbed" (adjective: "absorbed parent feature")
- **Avoid**: "feature suppression" (too generic), "absorption phenomenon" (redundant)

### Sparse Autoencoder (SAE)
An autoencoder with a sparsity constraint on the latent representation, trained to decompose neural network activations into interpretable features. Uses an overcomplete dictionary (typically 10-100x the hidden dimension) with ReLU or JumpReLU activation.

- **Preferred**: "sparse autoencoder" (first use), "SAE" (subsequent uses)
- **Avoid**: "sparse auto-encoder" (outdated spelling)

### Mechanistic Interpretability
The study of reverse-engineering neural networks to understand the algorithms they implement at a mechanistic level. SAEs are currently the dominant tool in this field.

- **Preferred**: "mechanistic interpretability"
- **Avoid**: "mechanistic interpretability research" (redundant), "MI" (abbreviation not established)

### Co-occurrence Clustering
A clustering method that groups features based on how often they activate on the same inputs (tokens). In UAD, uses phi coefficients as similarity measure and hierarchical clustering with Ward linkage.

- **Preferred**: "co-occurrence clustering" (hyphenated)
- **Avoid**: "cooccurrence clustering" (no hyphen), "occurrence clustering" (wrong)

### Collision Rate
The Jaccard overlap of top-K activating SAE features between two concepts. Serves as a proxy metric for the true absorption rate. Computed as |shared features| / |union of features|.

- **Preferred**: "collision rate" (lowercase except at sentence start)
- **Avoid**: "feature collision rate" (redundant), "collision metric" (imprecise)

### Token-Level Mutual Exclusivity
The property that absorption features for different child concepts fire on disjoint sets of tokens. For example, the feature absorbing "three" never activates on "four". This is the root cause of UAD's failure.

- **Preferred**: "token-level mutual exclusivity" (hyphenated when modifying noun)
- **Avoid**: "mutual exclusion" (wrong term), "disjoint activation" (less precise)

---

## Method-Specific Terms

### Unsupervised Absorption Detection (UAD)
The specific method tested in this paper, proposed as a training-free alternative to supervised absorption detection. Pipeline: feature selection -> phi coefficient matrix -> hierarchical clustering -> pair extraction -> dead feature filtering.

- **Preferred**: "UAD" (after first use), "Unsupervised Absorption Detection" (first use)
- **Avoid**: "the UAD method" (redundant), "unsupervised absorption detection" (generic, not the method name)

### Phi Coefficient
A measure of association for two binary variables, equivalent to Pearson correlation on dichotomous data. In UAD, computed from binarized feature activation patterns.

- **Preferred**: "phi coefficient" (lowercase phi)
- **Avoid**: "Phi coefficient" (capitalized), "phi correlation" (less common)

### Dead Feature
An SAE feature with near-zero activation variance across the dataset, indicating it is effectively unused. UAD filters these out before clustering.

- **Preferred**: "dead feature"
- **Avoid**: "inactive feature" (different meaning: may activate on rare inputs), "zero feature"

### Top-K Feature Overlap
The set intersection of the K most strongly activating features for two concepts. Used to define both collision rate and true absorption rate.

- **Preferred**: "top-K feature overlap" (K capitalized)
- **Avoid**: "top-k overlap" (inconsistent), "K-top overlap" (wrong order)

### Ground Truth Absorption Rate
The true degree of absorption between two child concepts, defined as the Jaccard similarity of their absorption feature sets. Distinguished from collision rate, which is a proxy.

- **Preferred**: "true absorption rate" (shorter), "ground truth absorption rate" (explicit)
- **Avoid**: "actual absorption" (vague), "real absorption rate"

---

## Model and Architecture Terms

### GPT-2 Small
OpenAI's 124M parameter transformer language model. 12 layers, 768 hidden dimensions, 12 attention heads.

- **Preferred**: "GPT-2 Small" (first use), "GPT-2" (subsequent, if unambiguous)
- **Avoid**: "GPT2" (no hyphen), "GPT 2" (space instead of hyphen)

### gpt2-small-res-jb
A pretrained SAE for GPT-2 Small layer 8, released by Joseph Bloom (JB) via SAELens. Dictionary size 24,576, trained on residual stream activations.

- **Preferred**: "gpt2-small-res-jb" (lowercase, code-formatted)
- **Avoid**: "GPT2-Small-Res-JB" (capitalized), "the JB SAE" (too informal)

### Residual Stream
The main data pathway in a transformer where information flows from input to output, with skip connections around each sublayer (attention and MLP).

- **Preferred**: "residual stream"
- **Avoid**: "residual connection" (refers to the skip connection, not the stream)

### SAELens
An open-source library for training and analyzing sparse autoencoders on language models. Provides pretrained SAEs and analysis tools.

- **Preferred**: "SAELens" (camelCase)
- **Avoid**: "SAE-Lens" (hyphenated), "Saelens" (lowercase L)

---

## Statistical Terms

### Spearman Rank Correlation
A non-parametric measure of rank correlation between two variables. Used as the primary effect size measure in this paper.

- **Preferred**: "Spearman r" (in text), "Spearman rank correlation" (first use)
- **Avoid**: "Spearman's rho" (different notation convention), "rank correlation" (too generic)

### Bootstrap Confidence Interval
A confidence interval computed by resampling with replacement and computing percentiles. We use B = 1,000 bootstrap samples with the percentile method.

- **Preferred**: "95% CI" (in tables), "95% confidence interval" (in text)
- **Avoid**: "bootstrap CI" (without specifying method), "confidence interval" (without coverage level)

### Jaccard Similarity
The size of the intersection divided by the size of the union of two sets. Used to compute both collision rate and true absorption rate.

- **Preferred**: "Jaccard similarity" or "Jaccard overlap"
- **Avoid**: "Jaccard index" (unless specifically referring to the index form), "Jaccard coefficient" (less common in ML)

---

## Abbreviations

| Abbreviation | Expansion | First Use Rule |
|-------------|-----------|----------------|
| SAE | Sparse Autoencoder | Expand on first use in each major section |
| UAD | Unsupervised Absorption Detection | Expand on first use |
| GT | Ground Truth | Expand on first use; prefer "ground truth" in text |
| TP | True Positive | Expand on first use in Results |
| FP | False Positive | Expand on first use in Results |
| FN | False Negative | Expand on first use in Results |
| CI | Confidence Interval | Expand on first use |
| MLP | Multi-Layer Perceptron | Expand on first use |
| ReLU | Rectified Linear Unit | No expansion needed (standard) |

---

## Proposed Alternative Methods (For Discussion Section)

### Decoder Weight Similarity
Cosine similarity between SAE decoder weight vectors. Proposed as a theoretically sound alternative to co-occurrence for detecting absorption.

- **Preferred**: "decoder weight similarity"
- **Avoid**: "decoder similarity" (ambiguous: could mean output similarity)

### Causal Intervention
Experimental manipulation of model activations (e.g., activation patching, ablation) to establish causal relationships between features.

- **Preferred**: "causal intervention"
- **Avoid**: "intervention" (too generic), "causal analysis" (broader)

### Activation Patching
A specific causal intervention technique where activations from one forward pass are patched into another to measure causal effects.

- **Preferred**: "activation patching"
- **Avoid**: "activation replacement" (less standard)

---

## Writing Style Notes

### Negative Result Framing
- Use "fails" or "does not detect" (direct, honest)
- Avoid "underperforms" (too soft), "struggles to" (anthropomorphizes)
- Use "indistinguishable from random" (precise statistical claim)
- Avoid "performs poorly" (vague)

### Precision in Claims
- Use exact numbers: "F1 = 0.00048" not "F1 near zero"
- Use confidence intervals: "r = 0.87, 95% CI [0.780, 0.938]"
- Report sample sizes: "n = 56 pairs"
- Report exact counts: "1 true positive out of 4,155 detected pairs"

### Avoid These Phrases
- "To the best of our knowledge" (banned per writing rules)
- "Significantly outperforms" (use exact numbers)
- "Groundbreaking" / "game-changing" / "revolutionary" (banned)
- "Moreover" / "Furthermore" / "It is worth noting that" (banned filler)
- "Novel" (unless quantified with specific novelty claim)
