# Glossary: Quantifying Feature Absorption in Sparse Autoencoders

## Core Concepts

### Feature Absorption
**Definition**: The phenomenon where a latent feature in a Sparse Autoencoder is functionally redundant with other co-firing features -- its activating tokens can be well-reconstructed by those co-firing neighbors. Also called "feature collapse" or "feature suppression" in some literature, though these terms are not strictly synonymous.
**Preferred term**: "feature absorption" (absorption for short).
**Not preferred**: "feature collapse" (too generic), "feature suppression" (implies active inhibition rather than co-occurrence).

### Absorption Score
**Definition**: A scalar in [0, 1] measuring what fraction of a latent's activating tokens have their residual-stream variance explained (>80% threshold) by the latent's top-5 co-firing neighbors. Score=1.0 means the feature is fully absorbed; score=0.0 means it is independent.
**Preferred term**: "absorption score" or "score" (in context), abbreviated $A_f$.
**Not preferred**: "absorption metric" (score is the metric output).

### Sparse Autoencoder (SAE)
**Definition**: An autoencoder trained with an L1 sparsity penalty on the bottleneck representation, forcing most latents to be zero for any given input. Used in mechanistic interpretability to decompose residual-stream or MLP-out activations into human-interpretable features.
**Preferred term**: "SAE" (abbreviation) or "Sparse Autoencoder" (first mention).
**Not preferred**: "sparse auto-encoder" (hyphenation).

### Residual Stream
**Definition**: The skip-connection pathway in a Transformer that carries the accumulated representation from layer to layer. SAE features are typically extracted from this stream.
**Preferred term**: "residual stream" (lowercase "residual").
**Not preferred**: "Residual Stream" (capitalized), "residual" (alone is ambiguous).

### Dictionary Size / Bottleneck Size
**Definition**: The number of latent features in an SAE's learned representation ($d_{sae}$). Larger dictionaries can represent more features but may introduce redundancy.
**Preferred term**: "dictionary size" or "$d_{sae}$".
**Not preferred**: "latent dimension" (confusing with model dimension), "codebook size" (overused in other domains).

### Co-firing / Co-firers
**Definition**: Two or more latents that are simultaneously active on the same token. Co-firing is the basis for computing absorption: if a feature's tokens are covered by its co-firers, the feature is "absorbed."
**Preferred term**: "co-firing" (verb/adjective), "co-firers" (noun referring to the co-firing latents).
**Not preferred**: "co-occurring" (less precise), "feature correlation" (correlation is a specific statistical measure).

## Experimental Terms

### Absorption Rate
**Definition**: The fraction (or percentage) of latents in an SAE that have absorption score >0.5 (or some threshold). Used to summarize prevalence across a layer or SAE configuration.
**Preferred term**: "absorption rate" or "% latents >0.5" or "$A_{layer}$".
**Not preferred**: "absorption percentage" (rate is not percentage; use "absorption rate of 0.19%").

### Falsification / Falsified
**Definition**: The process of testing a hypothesis by attempting to disprove it. A hypothesis is "falsified" when empirical evidence contradicts its prediction. This is a scientific virtue, not a failure.
**Preferred term**: "falsified" (adjective), "falsification" (noun).
**Not preferred**: "disproved" (stronger claim than warranted by non-significant results), "failed" (connotation of error rather than scientific result).

### NO-GO (Decision)
**Definition**: A project-level decision to not proceed with full experiments, made when pilot results indicate the hypotheses are wrong or the experimental design is flawed. Opposite of "GO" decision.
**Preferred term**: "NO-GO", "go/no-go".
**Not preferred**: "failed" (the project did not fail; the hypothesis did).

### Critical Path
**Definition**: The sequence of pending analyses or experiments that must be completed for the paper to advance beyond pilot stage. Currently H2 (token frequency correlation) and H6 (perfect-score latent investigation) are on the critical path, both requiring ~2h CPU analysis on existing layer 4 data.
**Preferred term**: "critical path", "on the critical path".
**Not preferred**: "blocking issue" (implies obstruction rather than prerequisite), "pending analysis" (less specific).

### Pilot Experiment
**Definition**: A small-scale experiment (100 sequences, single layer, quick runtime ~10 min) designed to rapidly test whether a hypothesis has any chance of being confirmed before committing to full-scale experiments.
**Preferred term**: "pilot experiment" or "pilot".
**Not preferred**: "preliminary experiment" (vague), "test run" (sounds ad-hoc).

### Faithfulness
**Definition**: In circuit analysis, the degree to which activation patching at a specific layer/position restores the clean-vs-corrupted behavior. Higher faithfulness means the patched location is causally important for the task.
**Preferred term**: "faithfulness" (scalar in [0, 1]).
**Not preferred**: "causal importance" (related but not identical), "reliability" (overloaded).

### Activation Patching
**Definition**: A causal interference technique where activations from a clean run are replaced ("patched") into a corrupted run at specific layers and positions, to measure the causal role of those locations. Also called "intervention patching" or "trace patching."
**Preferred term**: "activation patching" or just "patching."
**Not preferred**: "activation replacement" (less precise), "logit patching" (only for output layer).

### Cumulative Subsampling
**Definition**: In H5, a method for simulating different dictionary sizes from a single trained SAE ($d_{sae}$=24,576). Smaller dictionaries are created by selecting a subset of latents, prioritizing "absorbable" latents (those with highest absorption scores in the full SAE) to ensure they are not excluded from smaller dictionaries.
**Preferred term**: "cumulative subsampling" or "cumulative subsample."
**Not preferred**: "random subsampling" (implies uniform random selection), "dictionary pruning" (implies modifying trained weights).

### Inverted-U Pattern
**Definition**: A non-monotonic relationship where absorption first increases with a parameter (layer depth) then decreases, producing a peak at intermediate values. Observed in H3 where absorption peaks at layer 4 (49.3%) and declines in deeper layers (layer 8: 20.9%, layer 10: 17.3%).
**Preferred term**: "inverted-U pattern" or "inverted-U relationship."
**Not preferred**: "U-shaped curve" (misleading -- this is the inverse of a U).

### Layer-Dependent (Absorption)
**Definition**: The phenomenon where absorption prevalence varies systematically with model layer depth, not uniform across all layers. The central finding of this paper: absorption peaks at mid-layers (layer 4: 49.3%) and is nearly absent at deeper layers (layer 8: 0.19%).
**Preferred term**: "layer-dependent absorption" or "layer-dependent inverted-U pattern."
**Not preferred**: "layer-specific absorption" (less precise about the non-uniform nature), "non-uniform absorption" (vague).

## Model and Dataset Terms

### GPT-2 Small
**Definition**: The 124M-parameter GPT-2 model (12 layers, $d_{model}$=768, 12 attention heads). The base model used for SAE feature extraction in this study.
**Preferred term**: "GPT-2 small" (not "GPT2" or "gpt2").
**Not preferred**: "GPT-2" alone (could mean any size), "gpt2-small" (inconsistent casing).

### gpt2-small-res-jb
**Definition**: The SAELens pretrained residual-stream SAE for GPT-2 small, trained by Jesse B. (jb) on the Pile. The specific SAE release used for all experiments.
**Preferred term**: "gpt2-small-res-jb" (exact identifier from SAELens).
**Not preferred**: "the SAE" (too vague), "SAELens SAE" (redundant).

### Pile / pile-uncopyrighted
**Definition**: The EleutherAI Pile dataset, a 825GB text corpus used for language model training. The "uncopyrighted" subset excludes content with restrictive licenses. Our analysis corpus is drawn from this subset.
**Preferred term**: "pile-uncopyrighted" (for the dataset subset).
**Not preferred**: "The Pile" (when referring to the subset), "Pile" alone (ambiguous with other uses).

### L0 Norm
**Definition**: The count of non-zero elements in a vector. For SAEs, mean L0 is the average number of active latents per token. Used as a proxy for sparsity level.
**Preferred term**: "L0" (pronounced "L-zero", not "L-oh").
**Not preferred**: "sparsity count" (non-standard), "number of active latents."

## Writing Style Terms

### Filler Words to Avoid
**Banned openings** (rewrite on sight):
- "In recent years, X has attracted increasing attention..."
- "Sparse Autoencoders have become a cornerstone..."
- "Understanding X is crucial for..."

**Banned transitions**: "Moreover", "Furthermore", "It is worth noting that", "Additionally"

**Banned hype**: "groundbreaking", "game-changing", "revolutionary", "novel" (unless followed by a specific quantified claim)

**Preferred alternatives**: Lead with the result or number. Use "however", "but", "yet" for contrast. Use "specifically", "in particular" for emphasis.

### Rounding Rule
Never round away meaningful precision. If a result is 0.19%, do not write "less than 1%." If Spearman r=0.086, do not write "essentially zero."

### Negative Results Reporting
Report negative results explicitly without softening. "H1 was falsified: absorption rate was 0.19% (95% CI: [0.12%, 0.26%]), far below the hypothesized threshold of 20%." Do not write "H1 showed unexpected patterns" or "H1 requires further investigation."

### Perfect-Score Latents
**Preferred term**: "perfect-score latents" or "8 latents with $A_f=1.0$."
**Not preferred**: "suspicious latents" (implies wrongdoing), "anomalous latents" (too vague), "the 8 perfect-score outliers" (outlier implies statistical artifact; these may be genuine).

## Domain-Specific Abbreviations

| Abbreviation | Expansion | Notes |
|--------------|-----------|-------|
| SAE | Sparse Autoencoder | Core object of study |
| LLM | Large Language Model | Base model class |
| MLP | Multi-Layer Perceptron | Transformer component |
| d_model | Model hidden dimension | 768 for GPT-2 small |
| d_sae | SAE dictionary/bottleneck size | Variable (2K-16K+) |
| pp | Percentage points | For differences |
| GO/NO-GO | Experiment continuation decision | Project-level |
| CI | Confidence interval | Statistical |
| FALSIFIED | Hypothesis result when evidence contradicts prediction | Scientific virtue |
| CONFIRMED | Hypothesis result when evidence supports prediction | Positive result |
| NOT TESTED | Hypothesis not run due to NO-GO decision | Skipped |
| UNINFORMATIVE | Hypothesis test yielded no meaningful signal | Both conditions = 0 |