# Glossary

Unified terminology for the paper. All section writers must use these exact phrasings. Deviations are errors.

---

## Core Concepts

| Term | Definition | Notes |
|------|-----------|-------|
| **Feature absorption** | A failure mode where an SAE latent fails to activate on inputs it should represent because a more specific co-occurring latent subsumes its role via the sparsity objective. | Always "absorption", never "feature suppression" or "feature occlusion". |
| **Feature splitting** | The complementary phenomenon to absorption: a single concept is represented by multiple SAE latents that partition the input space. | Always "splitting", never "fragmentation". |
| **Feature hedging** | A failure mode where an SAE merges correlated features into a single latent because the dictionary is too narrow, producing a "hedge" that partially represents both features. | Distinct from absorption: hedging is a capacity problem, absorption is a hierarchy problem. |
| **Superposition** | The phenomenon where neural networks encode more features than they have dimensions by using overlapping, nearly orthogonal directions. | Use "superposition" for the general phenomenon; "polysemanticity" for the per-neuron manifestation. |
| **Polysemanticity** | The property of a single neuron activating for multiple unrelated concepts due to superposition. | Adjective form: "polysemantic". |
| **Monosemanticity** | The property of a neuron or SAE latent activating for a single, coherent concept. | Adjective form: "monosemantic". |

## SAE Architecture Terms

| Term | Definition | Notes |
|------|-----------|-------|
| **Sparse autoencoder (SAE)** | A neural network that decomposes activation vectors into a sparse, overcomplete basis of interpretable features. | Abbreviation: SAE (defined at first use). Always "sparse autoencoder", not "sparse auto-encoder". |
| **SAE latent** | A single feature direction in the SAE's dictionary (one column of the decoder matrix). | Prefer "latent" over "feature" when referring to the SAE's learned directions, to distinguish from ground-truth features. |
| **Decoder direction** | The unit vector corresponding to one latent in the SAE's decoder weight matrix. | Use $\mathbf{d}_j$ notation. |
| **Dictionary width** | The number of latents in an SAE (dimension of the latent space $m$). | Always "width", not "size" or "capacity". |
| **L0 (sparsity)** | The expected number of active latents per input token. | Always styled as "L0" (capital L, zero), not "l0" or "$\ell_0$" in running text. Use $L_0$ in equations. |
| **TopK SAE** | SAE architecture that selects exactly the $k$ highest-activating latents per input. | Capitalize as "TopK", not "topk" or "Top-K". |
| **JumpReLU SAE** | SAE architecture with a learnable per-latent activation threshold, trained via straight-through estimator. | One word: "JumpReLU", not "Jump-ReLU" or "Jump ReLU". |
| **Gated SAE** | SAE architecture that decouples feature detection from magnitude estimation using a gating mechanism. | |
| **Matryoshka SAE** | SAE architecture that trains nested dictionaries of increasing size, organizing features hierarchically. | |
| **Dead feature** | An SAE latent that never activates on any input in the evaluation set. | "Dead feature", not "dormant feature" or "inactive feature". |

## Evaluation Terms

| Term | Definition | Notes |
|------|-----------|-------|
| **SAEBench** | A comprehensive benchmark suite for evaluating SAEs across 8 metrics, covering 200+ open-source SAEs. | One word: "SAEBench", not "SAE-Bench" or "SAE Bench". |
| **Sparse probing** | An evaluation method that trains k-sparse linear probes on SAE latent activations to measure how well SAE features capture target concepts. | "Sparse probing", not "sparse probe" when referring to the method. |
| **Spurious correlation removal (SCR)** | A SAEBench metric measuring how well ablating a single SAE latent removes a spurious correlation from model behavior. | Abbreviation: SCR. |
| **RAVEL** | An evaluation benchmark measuring how well SAE features disentangle attributes in entity representations. | All caps. TPP = True Positive Proportion (the primary RAVEL metric used in this paper). |
| **Absorption rate** | The fraction of tested features that exhibit absorption: features where all dedicated split latents fail to fire but a non-split latent with high ablation effect is detected. | Symbol: $R_{\text{abs}}$. |
| **False-negative token** | A token where all k-sparse split latents for a given feature fail to activate, despite the linear probe correctly classifying the token. | Hyphenate: "false-negative", not "false negative" when used as adjective. |

## Statistical Methods

| Term | Definition | Notes |
|------|-----------|-------|
| **Partial correlation** | Pearson correlation between two variables after controlling for (linearly removing) the effects of one or more confounders. | Always specify the covariates in context. |
| **Suppression effect** | A phenomenon where adding a covariate to a regression strengthens (rather than weakens) the association between predictor and outcome, because the covariate was masking the true relationship. | Also called "suppression variable effect". |
| **Baron-Kenny mediation** | A four-step procedure for testing whether variable M mediates the effect of X on Y: (1) X predicts Y; (2) X predicts M; (3) M predicts Y controlling for X; (4) X's effect on Y weakens when M is added. | Hyphenate: "Baron-Kenny", not "Baron and Kenny". Full mediation: c' = 0; partial mediation: c' reduced but nonzero. |
| **Rosenbaum sensitivity analysis** | A method for quantifying how robust a matched-pair finding is to unmeasured confounders, reported as the Gamma parameter at which significance is lost. | "Gamma" (capital G) when referring to the Rosenbaum parameter; $\Gamma$ in equations. |
| **Generalized additive model (GAM)** | A regression model that replaces linear terms with smooth nonparametric functions, allowing flexible nonlinear relationships. | Abbreviation: GAM. |
| **Tensor interaction** | In a GAM, a smooth function of two variables that captures their joint effect beyond additive contributions. | Written as $\text{ti}(x_1, x_2)$ in GAM formulas. |
| **BCa bootstrap** | Bias-corrected and accelerated bootstrap method for constructing confidence intervals. | Always "BCa", not "BCA" or "Bca". |

## Data Sources

| Term | Definition | Notes |
|------|-----------|-------|
| **Gemma Scope** | Google DeepMind's open-source collection of 400+ JumpReLU SAEs trained on Gemma 2 (2B, 9B, 27B) across all layers and widths 1k--1M. | Two words: "Gemma Scope", not "GemmaScope". |
| **Gemma 2 2B** | Google's 2-billion parameter language model from the Gemma 2 family. | "Gemma 2 2B", not "Gemma-2-2B" in running text. |
| **GPT-2 Small** | OpenAI's 124M parameter transformer language model. | "GPT-2 Small", not "GPT2-small" or "gpt2". |
| **RAVEL dataset** | A dataset of entity-attribute pairs (e.g., cities with country, continent, language) used for evaluating feature disentanglement. | Context-dependent: "RAVEL" can refer to the dataset or the metric. Clarify when ambiguous. |
| **SAELens** | The standard open-source library for SAE training and evaluation. | One word: "SAELens", not "SAE-Lens" or "SAE Lens". |
| **TransformerLens** | The standard open-source library for mechanistic interpretability with hook-based activation access. | One word: "TransformerLens". |

## Abbreviations (alphabetical)

| Abbreviation | Expansion |
|-------------|-----------|
| ATM | Adaptive Temporal Masking |
| CI | Confidence Interval |
| GAM | Generalized Additive Model |
| LLM | Large Language Model |
| MLP | Multi-Layer Perceptron |
| OLS | Ordinary Least Squares |
| SAE | Sparse Autoencoder |
| SCR | Spurious Correlation Removal |
| SE | Standard Error |
| STE | Straight-Through Estimator |
| TPP | True Positive Proportion |
| VIF | Variance Inflation Factor |
