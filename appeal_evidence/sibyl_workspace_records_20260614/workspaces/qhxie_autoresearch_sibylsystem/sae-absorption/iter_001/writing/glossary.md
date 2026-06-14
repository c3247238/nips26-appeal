# Glossary: Anatomy of Feature Absorption in Sparse Autoencoders

Unified terminology definitions for all paper writers, critics, and editors.
Use the preferred phrasing listed here. Do not introduce synonyms without adding them here first.

---

## Core Concepts

**Sparse Autoencoder (SAE)**
A neural network trained to reconstruct model activations via a sparse code; used in mechanistic interpretability to decompose residual stream activations into interpretable learned features.
Preferred: "Sparse Autoencoder (SAE)" on first use, "SAE" thereafter.
Do NOT use: "sparse auto-encoder", "dictionary learning network".

**feature absorption** (lower-case noun phrase)
The failure mode where a parent latent's activation is suppressed to zero on inputs where a more specific child latent is active, even though the parent concept is present in the input.
Preferred: "feature absorption" or just "absorption" after first definition.
Do NOT use: "feature suppression" (different meaning), "absorption failure", "absorption artifact".

**parent latent / parent feature**
A latent representing a more general concept that subsumes the child concept (e.g., "starts-with-A words" is parent to "starts-with-A proper nouns").
Preferred: "parent latent" or "parent feature". Use consistently; do not alternate.

**child latent / child feature**
A latent representing a more specific concept whose activation suppresses the parent latent.
Preferred: "child latent" or "child feature".

**absorption rate**
Fraction of input tokens (or RAVEL entities) for which a parent latent has zero activation despite the parent concept being present.
Preferred: "absorption rate". Report as a decimal fraction (e.g., 0.011) or percentage (1.1%), specify which.

---

## EDA and D-EDA

**Encoder-Decoder Alignment (EDA)**
The metric $\text{EDA}(j) = 1 - \cos(w_{e,j}, d_j)$, measuring the angular mismatch between the encoder direction and decoder direction for latent $j$.
Preferred: "EDA" after first definition as "Encoder-Decoder Alignment (EDA)".
Do NOT use: "encoder-decoder misalignment score", "EDA score" (just "EDA").

**Directional EDA (D-EDA)**
The extension of EDA that decomposes the encoder residual into the decoder dictionary to distinguish absorption from polysemanticity.
Preferred: "D-EDA" or "Directional EDA (D-EDA)" on first use.
Do NOT use: "directional EDA metric", "DEDA" (no hyphen).

**probe direction**
A unit-normed vector in residual stream space obtained by training a linear probe to detect a specific concept (e.g., the parent feature class).
Preferred: "probe direction". Not "probe vector" (inconsistent with linear probe terminology), not "direction vector".

**encoder direction**
Row $j$ of the SAE encoder weight matrix: $w_{e,j}$. The direction in residual stream space that latent $j$'s encoder projects input onto.
Preferred: "encoder direction".

**decoder direction**
Column $j$ of the SAE decoder weight matrix: $d_j$. The direction in residual stream space along which latent $j$ contributes to the reconstruction.
Preferred: "decoder direction".

---

## Three-Subtype Taxonomy

**early absorption**
The absorption subtype where the parent feature was never allocated in the SAE dictionary: $\max_k \cos(d_k, v_p) < \tau$. The encoder cannot fire the parent latent because no decoder direction approximates the parent concept.
Preferred: "early absorption" (noun phrase). Adjective form: "early-absorbed", "early-type latent".
Do NOT use: "Type I absorption", "missing-feature absorption".

**late absorption**
The absorption subtype where the parent decoder direction exists in the dictionary ($\max_k \cos(d_k, v_p) \geq \tau$) but the encoder is suppressed on parent-positive inputs. The feature was learned but the encoder was trained away from it.
Preferred: "late absorption". Adjective form: "late-absorbed", "late-type latent".
Do NOT use: "Type II absorption", "encoder-suppressed absorption".

**partial absorption**
The absorption subtype where the parent direction exists and the latent fires on some but not all parent-positive inputs — context-dependent, selective suppression.
Preferred: "partial absorption". Adjective form: "partial-absorbed", "partial-type latent".
Do NOT use: "Type III absorption", "selective absorption" (ambiguous).

**early-absorption dominance**
The empirical finding that early absorption constitutes ~72–75% of all absorbed latents, making dictionary-coverage failure the dominant cause of absorption.
Preferred: "early-absorption dominance" as a noun phrase when referring to the finding.

---

## ITAC

**Inference-Time Absorption Correction (ITAC)**
A training-free post-hoc correction method that recovers suppressed parent activations by using the D-EDA decomposition to estimate the missing parent contribution at inference time.
Preferred: "ITAC" after first definition as "Inference-Time Absorption Correction (ITAC)".
Do NOT use: "inference-time correction", "ITAC correction" (redundant).

---

## Evaluation and Datasets

**Chanin et al. metric**
The supervised absorption metric from Chanin et al. (2024): requires pre-specified probe directions and activation data; measures false negative rate of parent latents on parent-positive inputs.
Preferred: "Chanin et al. metric" or "supervised absorption metric".
Do NOT use: "canonical metric" (too vague), "ground-truth metric" (it is not ground truth — it is a supervised approximation).

**RAVEL**
Relational Attribute Verbalization for Entity (probing) dataset; contains city-continent, city-country, city-language, and other entity-attribute pairs. Available at `hij/ravel` on HuggingFace.
Preferred: "RAVEL" (all caps, as an acronym). Use "RAVEL entity-attribute hierarchies" when specificity is needed.

**RAVEL hierarchy**
One specific parent-child feature pair from RAVEL (e.g., city-continent = parent "continent", child "city").
Preferred: "RAVEL hierarchy". Use "hierarchy" as shorthand after first definition in context.

**SAEBench**
The benchmark suite for evaluating SAE quality across multiple metrics including absorption, published by Karvonen et al. (2025).
Preferred: "SAEBench". Do NOT use "SAE Bench" (two words).

**Gemma Scope**
The publicly released collection of Sparse Autoencoders for Gemma 2 2B, from Google DeepMind.
Preferred: "Gemma Scope". Full citation on first use.

**SynthSAEBench**
Synthetic SAE benchmark with known ground-truth absorption labels from arXiv:2602.14687; used to validate EDA's lower bound theorem.
Preferred: "SynthSAEBench".

---

## Mechanistic Interpretability Context

**mechanistic interpretability**
The research agenda of reverse-engineering the computations performed by neural networks at the level of individual weights and activations.
Preferred: "mechanistic interpretability". Abbreviate as "MI" only after explicit introduction.

**polysemanticity**
The property of a latent encoding multiple unrelated concepts simultaneously, as opposed to a monosemantic latent that encodes a single concept.
Preferred: "polysemanticity". Adjective: "polysemantic". Do NOT use "polysemantics" (not a noun).

**monosemanticity**
The property of a latent encoding a single, interpretable concept.
Preferred: "monosemanticity". Adjective: "monosemantic".

**biconvex SDL loss**
The Sparse Dictionary Learning loss used to train SAEs, which is convex separately in encoder parameters and decoder parameters but not jointly.
Preferred: "biconvex SDL loss". On first use: "sparse dictionary learning (SDL) loss, which has a biconvex structure".

**local minimum / partial minimum**
A point in the biconvex loss landscape where neither encoder nor decoder updates reduce loss, but the joint optimum has not been reached. Absorption is a consequence of settling at such a partial minimum.
Preferred: "partial minimum" (more specific to biconvex structure); "local minimum" is acceptable.

---

## Statistical Terms

**bootstrap CI / bootstrap confidence interval**
Confidence interval estimated by resampling the data (10,000 resamples, seed = 42 throughout this paper).
Preferred: "95% bootstrap CI" or "95% CI" after the method is defined.

**AUROC**
Area Under the Receiver Operating Characteristic curve. Values close to 1.0 indicate strong discrimination; 0.5 is chance.
Preferred: "AUROC". Do NOT use "AUC" without the "ROC" qualifier.

**Spearman rho / Spearman rank correlation**
Non-parametric rank correlation; reported as $\rho_s$.
Preferred: "Spearman $\rho$" on first use; "$\rho$" thereafter in context. Do NOT use "Pearson correlation" where Spearman is intended.

**DeLong test**
Statistical test for comparing two AUROC values from the same sample; used to compare EDA vs. decoder cosine baseline.
Preferred: "DeLong test". Full citation on first use.

**Mann-Whitney U test**
Non-parametric test for comparing EDA distributions between two groups (e.g., absorbed vs. non-absorbed latents).
Preferred: "Mann-Whitney U test". Not "Wilcoxon rank-sum test" (same test, different naming convention — use Mann-Whitney).

**Kruskal-Wallis test**
Non-parametric test for comparing EDA distributions across three or more groups (subtype classification).
Preferred: "Kruskal-Wallis test".

**Cohen's d**
Standardized effect size for comparing two group means: $d = (\mu_1 - \mu_2) / \sigma_{\text{pooled}}$.
Preferred: "Cohen's $d$" on first use; "$d$" in context.

---

## Polysemanticity Stratification

**feature density**
Fraction of input tokens on which a latent has non-zero activation; used as a proxy for polysemanticity (high density = polysemantic, low density = monosemantic). Sourced from SAEBench.
Preferred: "feature density". Do NOT use "activation frequency" (not established).

**median split**
Division of all SAE latents into two equal halves by feature density; above-median latents = polysemantic half, below-median = monosemantic half. Used for EDA stratification analysis in Section 4.3.
Preferred: "median split (by feature density)". Report the median density value when relevant.

**polysemantic regime**
The set of SAE latents with feature density above the median; characterized by higher EDA AUROC (0.922 vs. 0.643 at L12-16k).
Preferred: "polysemantic latent regions" or "polysemantic regime". Do NOT use "dense feature space".

---

## Writing Conventions

- Numbers: use Arabic numerals for all numeric results (0.776, not "zero point seven seven six").
- Percentages: "72.3%" not "72.3 percent" or "72%".
- Decimal precision: report AUROC to 3 decimal places; absorption rates to 3 significant figures; p-values as "p = 0.004" (not "p < 0.05" unless the exact value is unavailable).
- Equations: number all displayed equations; refer as "Equation (1)" in text.
- Figure references: always cite figure before it appears: "As shown in Figure 3, ..."
- Table references: always capitalize: "Table 1", "Table 2".
- Model names: "Gemma 2 2B" (not "Gemma-2-2B"); "GPT-2 Small" (not "GPT2 Small" or "GPT-2").
- SAE notation: "16k SAE" means $d_{\text{SAE}} = 16384$; "65k SAE" means $d_{\text{SAE}} = 65536$.
