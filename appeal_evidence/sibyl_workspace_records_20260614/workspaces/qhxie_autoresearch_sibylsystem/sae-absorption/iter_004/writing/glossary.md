# Glossary

Unified terminology for the paper. All section writers must use these exact terms and abbreviations.

---

## Core Concepts

| Term | Definition | Preferred Form |
|------|-----------|----------------|
| Sparse autoencoder | A dictionary learning model that decomposes neural network activations into sparse linear combinations of learned features (latents). | SAE (plural: SAEs) |
| Feature absorption | The systematic failure of an SAE to represent a general feature because its information is captured by one or more specific features that co-occur with it in the training corpus. | "feature absorption" (not "absorption phenomenon" or "feature suppression") |
| Latent | A single learned feature direction in the SAE dictionary. Synonymous with "SAE feature" or "dictionary element." | "latent" in formal contexts; "feature" acceptable when unambiguous |
| Superposition | The phenomenon where a neural network represents more features than it has dimensions, encoding them as non-orthogonal directions. Distinct from absorption. | "superposition" (not "polysemanticity," which refers to a neuron responding to multiple unrelated concepts) |
| Mechanistic interpretability | The subfield of AI safety research that reverse-engineers the internal computations of neural networks to understand their behavior. | "mechanistic interpretability" (abbreviation: "mech interp" only in informal contexts, never in the paper) |

## Feature Types

| Term | Definition | Preferred Form |
|------|-----------|----------------|
| Parent feature | A general feature in a feature hierarchy (e.g., "first letter = A") that is susceptible to absorption by more specific child features. | "parent feature" or "parent latent" |
| Child feature | A specific feature (e.g., "the token 'April'") that absorbs information from a parent feature through co-activation. | "child feature" or "child latent" or "absorbing latent" |
| First-letter task | The canonical absorption evaluation task: whether the SAE correctly represents the "starts with letter X" feature for each letter A--Z. | "first-letter task" (hyphenated) |

## Absorption Types (Our Taxonomy)

| Term | Definition | Preferred Form |
|------|-----------|----------------|
| Type I absorption | Full, single-latent absorption: one child latent accounts for >80% of the parent's suppression, with Chanin metric > 0.5. | "Type I (full) absorption" |
| Type II absorption | Partial absorption: the parent latent activates at <50% of its expected magnitude on relevant tokens, but no single child dominates. | "Type II (partial) absorption" |
| Type III absorption | Distributed absorption: top-3 children collectively explain the parent's suppression (DAS(k=3) > 0.6), but no single child qualifies as Type I. | "Type III (distributed) absorption" |
| Comprehensive absorption rate | The fraction of features showing any type of absorption (Type I + II + III). | "comprehensive absorption rate" |

## Methods and Metrics

| Term | Definition | Preferred Form |
|------|-----------|----------------|
| Competition coefficient | The Lotka-Volterra-inspired quantity $\alpha_{ij} = \sigma_{ij} \cdot (f_j / f_i)$ that quantifies competitive pressure between two SAE latents. | "competition coefficient $\alpha_{ij}$" (not "LV coefficient" or "alpha score") |
| Niche overlap | The normalized co-activation rate $\sigma_{ij}$ between two latents, analogous to ecological niche overlap. | "niche overlap $\sigma_{ij}$" |
| Distributed Absorption Score | DAS(P, k): a measure of how much parent P's firing is explained by its top-k children. | "DAS" (always followed by the $k$ value, e.g., "DAS($k=3$)") |
| Chanin metric | The ground-truth absorption rate from sae-spelling (Chanin et al., 2024), which uses pre-specified probe directions to identify absorbed features. | "Chanin metric" or "sae-spelling absorption rate" |
| Probe gap | The difference in ROC-AUC between a dense linear probe on the residual stream and a 1-sparse SAE probe, measuring information loss due to SAE encoding. | "probe gap" |

## SAE Architectures

| Term | Definition | Preferred Form |
|------|-----------|----------------|
| Gemma Scope | Google's collection of pre-trained SAEs for the Gemma 2 model family, available at multiple layers and widths via SAELens. | "Gemma Scope SAEs" |
| JumpReLU SAE | An SAE variant using a JumpReLU activation function for cleaner feature selection. | "JumpReLU SAE" |
| Matryoshka SAE | An SAE with nested structure enabling hierarchical feature representation. | "Matryoshka SAE" |
| OrtSAE | An SAE with orthogonality regularization on decoder columns to reduce feature absorption. | "OrtSAE" |

## Evaluation Benchmarks

| Term | Definition | Preferred Form |
|------|-----------|----------------|
| SAEBench | A standardized benchmark suite (Karvonen et al., 2025) that evaluates SAEs across absorption, sparse probing, RAVEL, SCR, and unlearning tasks. | "SAEBench" (one word, capital B) |
| RAVEL | Representation Analysis via Entity-Level probing; measures whether SAE features preserve entity-level distinctions. | "RAVEL" |
| SCR | Spurious Correlation Removal; measures whether SAE features enable targeted removal of spurious associations. | "SCR" |
| TPP | Token Prediction Probing; the SAEBench proxy metric for RAVEL performance. | "TPP" |

## Tools and Libraries

| Term | Definition | Preferred Form |
|------|-----------|----------------|
| SAELens | A Python library for loading, training, and analyzing SAEs, supporting multiple pre-trained SAE collections. | "SAELens" (capital L) |
| TransformerLens | A library for mechanistic interpretability that provides hook-based access to transformer internals. | "TransformerLens" (capital L) |
| sae-spelling | A library implementing the first-letter absorption measurement protocol from Chanin et al. (2024). | "`sae-spelling`" (lowercase, monospace in text) |

## Statistical Terms

| Term | Definition | Preferred Form |
|------|-----------|----------------|
| Bonferroni correction | A multiple-testing correction that divides the significance threshold by the number of tests. | "Bonferroni correction" (not "Bonferroni adjustment") |
| Partial correlation | Correlation between two variables after removing the linear effect of control variables. | "partial correlation" |
| HC3 standard errors | Heteroskedasticity-consistent standard errors (MacKinnon & White, 1985). | "HC3 robust standard errors" |

## Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| LV | Lotka-Volterra |
| PMI | Pointwise Mutual Information |
| DAS | Distributed Absorption Score |
| AIC | Akaike Information Criterion |
| AUC | Area Under the (ROC) Curve |
| SCR | Spurious Correlation Removal |
| TPP | Token Prediction Probing |
| OWT | OpenWebText |
