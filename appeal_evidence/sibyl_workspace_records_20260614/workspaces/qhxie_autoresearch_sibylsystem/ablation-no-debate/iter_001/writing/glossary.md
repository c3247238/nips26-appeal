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
An ablation methodology that ablates top-k child features simultaneously and measures the proportional contribution of each child to the parent's reconstruction.

**Preferred**: "multi-child proportional ablation"
**Not**: "k-child ablation" (vague)

### Absorption Rate
The fraction of a parent feature's activation that can be explained by its children's combined activation after the parent is ablated.

**Preferred**: "absorption rate" (as decimal, e.g., 0.50)
**Not**: "absorption percentage" (mixes decimal/percentage)

---

## Geometric vs. Learned Decomposition

### Geometric Contribution
The component of absorption attributable to decoder geometry (structural alignment), independent of training. Measured as absorption when encoder is random but decoder is trained.

**Preferred**: "geometric contribution"
**Not**: "structural contribution" (ambiguous)

### Learned Contribution
The additional absorption attributable to encoder alignment during training. Measured as the difference between full training and geometric-only conditions.

**Preferred**: "learned contribution" or "encoder alignment effect"
**Not**: "training contribution" (confusing)

### 2x2 Factorial Design
An experimental design crossing two factors (encoder: random/trained, decoder: random/trained) to decompose absorption into geometric and learned components.

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

### Single-Child Ablation
Ablating a single feature to measure its contribution to reconstruction.

**Preferred**: "single-child ablation"
**Not**: "single ablation", "one-feature ablation"

### Multi-Child Ablation
Ablating multiple (top-k) features simultaneously.

**Preferred**: "multi-child ablation"
**Not**: "multiple ablation", "k-feature ablation"

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

### Random Decoder Baseline
An SAE with Xavier-initialized decoder weights and no training.

**Preferred**: "random decoder baseline" or "Random Decoder"
**Not**: "untrained SAE" (encoder is also random)

### Shuffled Features Baseline
Baseline where feature indices are randomly permuted, breaking feature identity while preserving activation statistics.

**Preferred**: "shuffled features baseline" or "Shuffled Features"
**Not**: "feature permutation baseline"

### Permuted Encoder Baseline
A trained SAE with encoder weights randomly shuffled, breaking encoder-feature correspondence.

**Preferred**: "permuted encoder baseline" or "Permuted Encoder"
**Not**: "encoder shuffle baseline"

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

### Parent Direction
The reconstruction direction of a parent feature, computed as the projection of children's decoder subspace onto the parent's decoder.

**Preferred**: "parent direction" or "parent_dir"
**Not**: "parent vector", "parent subspace"

### Alpha (Steering Coefficient)
The scalar multiplier applied to the steering direction.

**Preferred**: "alpha" or "steering coefficient"
**Not**: "steering strength" (vague), "direction magnitude"

---

## Ecological Analogy

### Competitive Exclusion
The ecological principle that two species competing for the same resources cannot coexist at constant population values.

**Preferred**: "competitive exclusion" or "competitive exclusion principle"
**Not**: "exclusion principle"

### Niche Overlap
The degree to which two features occupy similar representational space.

**Preferred**: "niche overlap"
**Not**: "feature overlap", "representation similarity"

### Lotka-Volterra Model
A mathematical model describing predator-prey dynamics, proposed as analogy for feature competition.

**Preferred**: "Lotka-Volterra model" or "Lotka-Volterra"
**Not**: "LV model", "L-V equations"

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

### Downstream Task
A task that depends on correct feature activation for its output.

**Preferred**: "downstream task"
**Not**: "downstream application", "subsequent task"

---

## Negative Results

### Negative Result
An experimental outcome where a hypothesis is falsified; scientifically valuable finding.

**Preferred**: "negative result" or "null finding"
**Not**: "failed experiment" (implies error), "inconclusive result"

### Epistemic Failure
A limitation in knowledge or understanding, as opposed to active interference.

**Preferred**: "epistemic" (adjective)
**Not**: "epistemological" (philosophy of knowledge)

### Causal Failure
Active interference where one factor directly causes degradation in another.

**Preferred**: "causal" (adjective)
**Not**: "causally" (adverb, when "causal" suffices)

---

## Writing Conventions

### Hyphenation
- "feature absorption" (no hyphen, noun-noun compound)
- "top-k" (hyphenated when used as adjective)
- "multi-child" (hyphenated)
- "safety-critical" (hyphenated)

### Capitalization
- "Sparse Autoencoder" (first use only)
- "SAE" thereafter
- "Methodology", "Experiments", "Discussion" (section headings)
- "Figure 1", "Table 1" (figure/table references)

### Number Formatting
- Absorption rates: decimal (0.50), not percentage (50%)
- p-values: scientific notation (3.16e-133), not decimal (0.000...)
- Effect sizes: one decimal (d=8.9), two decimals for small values

### Citation Format
- Use arXiv IDs when available (e.g., Chanin et al., 2024)
- Include full citations in References section
