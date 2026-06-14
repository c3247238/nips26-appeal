# Glossary

This document defines key terminology used throughout the paper to ensure consistency.

---

## Core SAE Terminology

### Sparse Autoencoder (SAE)
A neural network that learns to reconstruct its input through a sparse bottleneck layer, decomposing complex activations into monosemantic feature representations.

**Preferred**: "SAE" (abbreviation) or "Sparse Autoencoder" (first use)
**Not**: "sparse auto-encoder", "sparse AE"

### Feature
A single dimension in the SAE's latent space, learned through training to correspond to a human-interpretable concept in the model's representations.

**Preferred**: "feature" (singular), "features" (plural)
**Not**: "neuron" (confusing with model neurons), "latent unit"

### Residual Stream
The accumulated activation signal flowing through transformer's skip connections.

**Preferred**: "residual stream"
**Not**: "residual stream activations" (redundant), "skip connection output"

### Encoder / Decoder
The encoder maps from residual stream to sparse features; the decoder reconstructs the residual stream from features.

**Preferred**: "encoder" and "decoder" as nouns
**Not**: "encoding" or "decoding" when referring to weights

---

## Feature Absorption

### Feature Absorption
The phenomenon where child features (sub-components) substitute for parent features (higher-level concepts) in sparse representations, making the parent feature inactive despite the concept being present in the input.

**Preferred**: "feature absorption" or simply "absorption"
**Not**: "absorption phenomenon", "absorbing features"

### Parent Feature
A higher-level feature that could be decomposed into multiple child features.

**Preferred**: "parent feature" or "parent"
**Not**: "superordinate feature", "parent concept"

### Child Feature
A lower-level feature that forms part of a parent's decomposition.

**Preferred**: "child feature" or "child"
**Not**: "sub-feature", "child concept"

### Grandchild Feature
Features at the lowest level of a hierarchy, directly constituting child features.

**Preferred**: "grandchild feature" or "grandchild"
**Not**: "sub-child feature", "leaf feature"

### Multi-Child Proportional Ablation
An ablation methodology that ablates multiple child features simultaneously and measures the proportional contribution of each child to the parent's reconstruction.

**Preferred**: "multi-child proportional ablation"
**Not**: "k-child ablation" (vague)

### Absorption Rate
The fraction of a parent feature's activation that can be explained by its children's combined activation after the parent is ablated.

**Preferred**: "absorption rate" (as decimal, e.g., 0.50)
**Not**: "absorption percentage" (mixes decimal/percentage)

---

## Encoder-Driven Mechanism

### Encoder Effect
The component of absorption attributable to the trained encoder's learned alignment with hierarchical features. Measured as the difference in absorption between trained-encoder and random-encoder conditions.

**Preferred**: "encoder effect" or "encoder alignment effect"
**Not**: "encoder contribution" (ambiguous)

### Decoder Effect
The component of absorption attributable to the trained decoder's geometric structure. Measured as the difference in absorption between trained-decoder and random-decoder conditions.

**Preferred**: "decoder effect" or "decoder geometry effect"
**Not**: "decoder contribution" (ambiguous)

### Decoder Disentanglement
The observation that a trained decoder reduces absorption compared to a random decoder when paired with a trained encoder, suggesting the decoder learns to redistribute parent activations across more features.

**Preferred**: "decoder disentanglement"
**Not**: "decoder compensation" (less precise)

### Capacity Pressure
The mechanism by which fewer active features (lower L0) force each feature to represent more concepts, leading to higher absorption rates.

**Preferred**: "capacity pressure"
**Not**: "feature overload" (informal)

### 2x2 Factorial Design
An experimental design crossing two factors (encoder: random/trained, decoder: random/trained) to decompose absorption into encoder and decoder contributions.

**Preferred**: "2x2 factorial" or "factorial decomposition"
**Not**: "four-way comparison"

---

## Measurement Methodology

### Top-K Activation
A sparsity mechanism where only the k highest-scoring features are active.

**Preferred**: "TopK activation" or "top-k activation"
**Not**: "Top-k sparsity", "k-top activation"

### L0
The number of active (non-zero) features in a sparse representation.

**Preferred**: "L0" (as in "L0 target of 32")
**Not**: "L0 norm" (L0 is not a norm), "sparsity level"

### Jaccard Overlap
The intersection-over-union measure used as the primary absorption metric, computed as the overlap between parent-active and child-active token sets.

**Preferred**: "Jaccard overlap" or "overlap"
**Not**: "Jaccard similarity" (different normalization)

### Saturation
The condition where ablation fails to reduce reconstruction quality because other features compensate.

**Preferred**: "saturation" or "saturated"
**Not**: "maxed out", "plateaued"

---

## Statistical Terms

### Effect Size (Cohen's d)
A standardized measure of the magnitude of difference between two groups.

**Preferred**: "Cohen's d" or "effect size"
**Not**: "effect size d", "d value"

### Mann-Whitney U Test
A non-parametric test comparing distributions between two independent groups.

**Preferred**: "Mann-Whitney U test" or "Mann-Whitney"
**Not**: "Wilcoxon rank-sum" (synonym but less common)

### Falsified
When a hypothesis fails to meet its pre-registered pass criteria.

**Preferred**: "falsified" or "not supported"
**Not**: "disproved" (too strong), "failed" (ambiguous)

---

## Baseline Methods

### Random SAE Baseline
An SAE with Xavier-initialized weights and no training.

**Preferred**: "random SAE baseline" or "Random SAE"
**Not**: "untrained SAE" (may imply partially trained)

### Shuffled Features Baseline
Baseline where feature indices are randomly permuted, breaking feature identity while preserving activation statistics.

**Preferred**: "shuffled features baseline" or "Shuffled Features"
**Not**: "feature permutation baseline"

---

## Steering and Intervention

### Steering
Adding a direction vector (scaled by alpha) to activations to test feature causality.

**Preferred**: "steering" (noun and verb)
**Not**: "intervention" (too generic), "direction injection"

### Feature Sensitivity
The degree to which a feature responds to its intended concept in the input.

**Preferred**: "sensitivity" or "feature sensitivity"
**Not**: "selectivity" (different meaning), "responsiveness"

### Sensitivity Ratio
The ratio of sensitivity between absorbed and non-absorbed features under steering intervention.

**Preferred**: "sensitivity ratio"
**Not**: "relative sensitivity" (vague)

---

## Safety Analysis

### Safety-Critical Feature
A feature whose activation is relevant to AI safety concerns (e.g., deception, jailbreaking, harm, manipulation).

**Preferred**: "safety-critical feature" or "safety-relevant feature"
**Not**: "dangerous feature" (value-laden), "unsafe feature"

### Gemma Scope
A collection of SAEs trained on Google's Gemma 2B model, available via HuggingFace.

**Preferred**: "Gemma Scope"
**Not**: "GemmaScope" (capital S), "gemma-scope"

### Neuronpedia
An open platform for exploring and annotating SAE features.

**Preferred**: "Neuronpedia"
**Not**: "neuronpedia" (lowercase)

---

## Negative Results

### Negative Result
An experimental outcome where a hypothesis is falsified; scientifically valuable finding.

**Preferred**: "negative result" or "null finding"
**Not**: "failed experiment" (implies error), "inconclusive result"

### Null Result
A finding of no statistically significant difference between groups or conditions.

**Preferred**: "null result"
**Not**: "zero effect" (implies true zero rather than insufficient evidence)

---

## Writing Conventions

### Hyphenation
- "feature absorption" (no hyphen, noun-noun compound)
- "top-k" (hyphenated when used as adjective)
- "multi-child" (hyphenated)
- "safety-critical" (hyphenated)
- "encoder-driven" (hyphenated)

### Capitalization
- "Sparse Autoencoder" (first use only)
- "SAE" thereafter
- "Methodology", "Experiments", "Discussion" (section headings)
- "Figure 1", "Table 1" (figure/table references)

### Number Formatting
- Absorption rates: decimal (0.50), not percentage (50%)
- p-values: scientific notation (3.85e-10), not decimal (0.000...)
- Effect sizes: one decimal (d=8.9), two decimals for small values

### Citation Format
- Use arXiv IDs when available (e.g., Chanin et al., 2024)
- Include full citations in References section
