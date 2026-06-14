# Glossary

Unified terminology for the paper. All section writers, critics, and the editor must follow these definitions and preferred phrasings.

## Core Concepts

| Term | Definition | Preferred phrasing |
|------|-----------|-------------------|
| **Feature absorption** | The systematic failure of a general (parent) SAE feature to fire when a specific (child) feature capturing a subset of the parent's domain is active. | "feature absorption" (not "absorption" alone in first use; after first use, "absorption" is acceptable) |
| **Sparse autoencoder (SAE)** | A neural network that decomposes dense activation vectors into sparse, interpretable features via an overcomplete dictionary. | "sparse autoencoder" on first use, then "SAE" |
| **Parent feature** | An SAE feature representing a general concept (e.g., "starts with S", "in Europe") that should fire for all instances of that concept. | "parent feature" (not "general feature" or "superordinate feature") |
| **Child feature** | An SAE feature representing a specific instance or subset of a parent concept (e.g., "the word Saturday", "the city Paris"). | "child feature" (not "specific feature" or "subordinate feature") |
| **Feature hierarchy** | A structured relationship between parent and child features, where the parent's semantic domain subsumes the child's. | "feature hierarchy" (not "concept hierarchy" or "feature tree") |
| **Competitive exclusion** | The mechanism by which a child feature suppresses the parent feature during SAE encoding, preventing the parent from firing. | "competitive exclusion" (borrowed from ecology; define on first use) |
| **Monosemanticity** | The property of an SAE feature responding to a single, coherent concept. Absorption threatens monosemanticity by creating systematic non-firing patterns. | "monosemanticity" (not "mono-semanticity") |

## Measurement and Methods

| Term | Definition | Preferred phrasing |
|------|-----------|-------------------|
| **Absorption rate** | The fraction of inputs where the probe correctly classifies raw activations but incorrectly classifies SAE-reconstructed activations (false negative rate attributable to SAE encoding). | "absorption rate" (not "absorption score" or "absorption fraction") |
| **False negative (FN)** | An input where the probe predicts the correct parent class from raw activations but the wrong class from SAE activations. The core unit of absorption measurement. | "false negative" or "FN" |
| **Probe** | A linear classifier (logistic regression) trained on residual stream activations to predict a categorical property (e.g., first letter, continent). | "probe" (not "classifier" or "linear head") |
| **Quality gate** | A minimum probe F1 threshold required for results to be considered reliable. Strict: F1 > 0.90; relaxed: F1 >= 0.80. | "quality gate" (always specify strict or relaxed) |
| **Activation patching** | An interventional method that modifies specific SAE feature activations (e.g., zeroing a child feature) and measures downstream effects on probe predictions. | "activation patching" (not "causal tracing" or "activation intervention") |
| **Recovery rate** | The fraction of false negatives where zeroing a feature restores the correct probe prediction. | "recovery rate" (not "restoration rate") |
| **Integrated gradients** | An attribution method used to identify which SAE features contribute to a probe's prediction, enabling parent-child pair identification. | "integrated gradients" or "integrated-gradients attribution" |

## Hedging and Decomposition

| Term | Definition | Preferred phrasing |
|------|-----------|-------------------|
| **Hedging** | A phenomenon where a feature distributes its activation across multiple related features instead of concentrating on the most relevant one. Related to but distinct from absorption. | "hedging" (not "feature splitting" or "feature spreading") |
| **Strict absorbed** | A false negative where the main parent feature does not fire at all -- a genuine feature gap in the SAE dictionary. | "strict absorbed" or "strictly absorbed" |
| **Compensatory FN** | A false negative where the main parent feature fires but other features interfere with the probe's prediction on SAE-reconstructed activations. | "compensatory" (not "soft absorbed" or "partial absorbed") |
| **Persistent FN** | A residual false negative that does not fit strict absorbed or compensatory categories (e.g., probe boundary errors). | "persistent" |

## Benign vs. Pathological

| Term | Definition | Preferred phrasing |
|------|-----------|-------------------|
| **Pathological absorption** | An absorption instance where ablating the parent direction from the child feature's decoder causes a significant logit change (> threshold), indicating the model relies on this information. | "pathological absorption" (not "harmful absorption") |
| **Benign absorption** | A hypothesized absorption instance where the parent feature is computationally redundant and its absence causes minimal downstream effect. Our experiments found 0% benign absorption. | "benign absorption" (note: this term should always appear with "hypothesized" or "putative" since the hypothesis was falsified) |

## SAE Architectures

| Term | Definition | Preferred phrasing |
|------|-----------|-------------------|
| **JumpReLU** | An SAE architecture using a learnable threshold activation function. Part of Gemma Scope. | "JumpReLU" (capital J, capital R, capital LU) |
| **BatchTopK** | An SAE architecture that selects the top-k activations per batch, enforcing sparsity. From SAEBench. | "BatchTopK" (capital B, capital T, capital K) |
| **Matryoshka** | An SAE architecture with nested dictionary structure. From SAEBench. | "Matryoshka" (capital M) |
| **Gemma Scope** | Google DeepMind's collection of pre-trained SAEs for Gemma 2 models. | "Gemma Scope" (two words, both capitalized) |
| **SAEBench** | A benchmark suite for evaluating SAEs, providing standardized pre-trained SAEs. | "SAEBench" (one word, capital S, capital B) |

## Datasets and Models

| Term | Definition | Preferred phrasing |
|------|-----------|-------------------|
| **Gemma 2 2B** | Google DeepMind's 2-billion parameter language model. The base model for all experiments. | "Gemma 2 2B" (not "Gemma-2-2B" or "gemma-2-2b" in text; use `google/gemma-2-2b` in code references) |
| **RAVEL** | Resolved Attribute Value Estimation for Language models. A dataset of entity-attribute pairs (city-country, city-continent, city-language) with validated probes. | "RAVEL" (all caps; cite Huang et al., 2024) |
| **sae-spelling** | The codebase from Chanin et al. (2024) for measuring absorption on the first-letter spelling task. | "`sae-spelling`" (monospace, lowercase with hyphen) |
| **TransformerLens** | A library for mechanistic interpretability that provides hook-based access to transformer internals. | "TransformerLens" (one word, capital T, capital L) |
| **SAELens** | A library for training and analyzing sparse autoencoders, built on TransformerLens. | "SAELens" (one word, capital S, capital L) |

## Statistical and Analytical Terms

| Term | Definition | Preferred phrasing |
|------|-----------|-------------------|
| **Bootstrap CI** | A 95% confidence interval computed from 10,000 bootstrap resamples. | "bootstrap 95% CI" (always specify 95% and resample count on first use) |
| **Bonferroni correction** | A multiple-comparison correction that divides the significance threshold by the number of tests. | "Bonferroni correction" (not "Bonferroni adjustment") |
| **Kruskal-Wallis test** | A non-parametric test for differences across multiple groups, used for cross-hierarchy comparisons. | "Kruskal-Wallis test" (not "KW test" in text) |
| **Wilcoxon signed-rank test** | A non-parametric paired test, used for activation patching recovery rate comparisons. | "Wilcoxon signed-rank test" (not "Wilcoxon test" alone) |
| **Cohen's $d$** | A standardized effect size measure (difference in means divided by pooled standard deviation). | "Cohen's $d$" (always italicize $d$) |

## Negative Result Terminology

| Term | Definition | Preferred phrasing |
|------|-----------|-------------------|
| **GAS (Geometric Absorption Score)** | A proposed unsupervised metric based on decoder-activation geometry mismatch. Failed as an absorption detector (rho=0.116). | "Geometric Absorption Score (GAS)" on first use, then "GAS" |
| **CMI (Conditional Mutual Information)** | An information-theoretic measure of feature dependence. Failed to predict absorption (rho=0.044). | "conditional mutual information (CMI)" on first use, then "CMI" |
| **Absorption Tax $T(G)$** | A theoretical construct: the minimum additional $L_0$ cost for absorption-free representation of hierarchy $G$. Quantitative predictions failed. | "Absorption Tax" (capitalized; define $T(G)$ formally in notation) |
| **Rate-distortion predictors** | A three-factor model (cosine similarity, co-occurrence, reconstruction importance) for predicting per-pair absorption probability. Model R^2=0.088. | "rate-distortion predictor model" (not "three-factor model" alone) |

## Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse autoencoder |
| FN | False negative |
| GAS | Geometric Absorption Score |
| CMI | Conditional mutual information |
| CI | Confidence interval |
| ANOVA | Analysis of variance |
| AUROC | Area under the receiver operating characteristic curve |
| LOO-CV | Leave-one-out cross-validation |
| LLM | Large language model |
| MI | Mechanistic interpretability |
