# Glossary

## Core Technical Terms

**Sparse Autoencoder (SAE)**
An autoencoder with a sparsity constraint on the latent representation, trained to reconstruct neural network activations while using only a small fraction of dictionary features per input. The standard tool for mechanistic interpretability.

**Feature Absorption**
The phenomenon where a single SAE dictionary feature represents multiple semantically distinct concepts, typically because a high-frequency (parent) concept suppresses the representation of a low-frequency (child) concept. Also called *feature collision* in this paper when measured via shared feature indices.

**Feature Collision**
A measurable proxy for absorption: when multiple ground-truth concepts (e.g., first letters 'a', 'i', 'o') activate the same SAE feature index. Collision rate = (# concepts sharing features) / (# total concepts).

**Parent-Child Feature Pair**
A hierarchical relationship where a general concept (parent, e.g., "vowel") and a specific concept (child, e.g., "letter 'a'") are semantically related. In absorption, the parent feature dominates and the child feature is suppressed.

**Superposition**
The phenomenon where neural networks represent more features than they have dimensions by encoding features in non-orthogonal directions. Absorption is a form of superposition at the SAE level.

**Polysemanticity**
When a single neuron (or SAE feature) responds to multiple unrelated inputs. Related to but distinct from absorption: polysemanticity implies unrelated concepts, while absorption implies hierarchical/related concepts.

## SAE Architecture Terms

**JumpReLU**
An SAE architecture that uses a gating mechanism (ReLU with a learned threshold jump) to control feature activation. Achieves state-of-the-art reconstruction quality. Used in GemmaScope.

**TopK SAE**
An SAE architecture that enforces sparsity by keeping only the top-k highest activations and zeroing all others. The k parameter directly controls sparsity level.

**BatchTopK**
A variant of TopK that selects the top-k features across an entire batch rather than per sample, allowing adaptive sparsity.

**Gated SAE**
An SAE architecture with a separate gating network that determines which features to activate.

**Matryoshka SAE**
An SAE with a nested dictionary structure where features are organized in hierarchical groups of varying granularity.

**Dictionary Size ($d_{\text{SAE}}$)**
The number of features in the SAE latent space. Typical values: 16K, 24K, 65K. Larger dictionaries can reduce absorption but increase computation.

**L0 Sparsity**
The mean number of active (non-zero) SAE features per input token. Not a true norm but a count statistic. Lower L0 means sparser representations.

**Dead Feature**
An SAE dictionary feature that never activates on any input from the evaluation dataset. High dead feature ratios indicate poor dictionary utilization.

## Evaluation Terms

**Sparse Probing**
Training a linear classifier on SAE feature activations to predict a semantic concept (e.g., "does this token start with 'a'?"). Measures how well SAE features encode concepts.

**Steering Efficacy**
The degree to which modifying SAE feature activations changes model behavior in a targeted direction (e.g., increasing sentiment positivity).

**Feature Attribution**
Mapping model predictions back to SAE features to understand which features influenced a decision. Methods include integrated gradients and attention-based attribution.

**Reconstruction MSE**
Mean squared error between the original model activation and the SAE reconstruction. Lower MSE means the SAE preserves more information.

**Loss Recovered**
The fraction of model loss explained by the SAE reconstruction relative to the original activation. A normalized measure of reconstruction quality.

**Specificity Score**
For a feature-concept pair, the ratio of activation on the target concept to activation on other concepts. Higher specificity means the feature is more selective.

## Method-Specific Terms

**Cross-Architecture Absorption Benchmark (CAAB)**
Our standardized evaluation protocol for comparing absorption rates across SAE architectures under controlled conditions.

**Unsupervised Absorption Detection (UAD)**
Our method for detecting absorbed feature pairs without ground-truth parent feature labels, using co-occurrence clustering.

**Dynamic Feature De-Absorption (DFDA)**
Our SAE-retraining-free method for recovering absorbed parent feature activations at inference time using a lightweight compensation network. Note: DFDA trains a small MLP (97 parameters per pair) but does not retrain the SAE itself.

**Co-occurrence Matrix**
A matrix $C$ where $C_{ij}$ counts how often features $i$ and $j$ activate together on the same input. Used in UAD for clustering.

**Hierarchical Clustering**
An unsupervised clustering method that builds a tree of nested clusters. Used in UAD to group potentially absorbed features.

**Residual Compensation Network**
A small MLP ($<1\%$ of SAE parameters) in DFDA that predicts parent feature activations from child feature activations.

## Model and Dataset Terms

**Gemma-2-2B**
Google's 2-billion parameter language model (Gemma 2 series). Planned as a secondary model but experiments were blocked by API issues.

**GPT-2 Small**
OpenAI's 124-million parameter language model (12 layers, 768 hidden dim). Primary model for all reported experiments.

**TransformerLens**
A library for mechanistic interpretability research that provides hooks into transformer internals for activation extraction.

**SAELens**
A library for training and evaluating sparse autoencoders on language model activations. Provides pretrained SAEs (GemmaScope) and training utilities.

**GemmaScope**
A collection of pretrained JumpReLU SAEs for Gemma models, released by Google DeepMind. We use the canonical 24K-width SAEs.

**OpenWebText**
An open replication of OpenAI's WebText dataset, used for training and evaluating language models. Our primary evaluation dataset.

**First-Letter Features**
SAE features that detect whether a token starts with a specific letter (a-z). A standard concept set from Chanin et al. (2024) used for interpretability evaluation.

## Statistical Terms

**Spearman Rank Correlation ($\rho_S$)**
A non-parametric measure of rank correlation between two variables. Used when linear correlation assumptions may not hold.

**Partial Correlation**
The correlation between two variables while controlling for one or more confounding variables. Used in our causal analysis to control for reconstruction quality.

**P-value**
The probability of observing the data (or more extreme) under the null hypothesis. $p < 0.05$ is the conventional threshold for statistical significance.

**Effect Size**
The magnitude of a relationship independent of sample size. We report Spearman r as an effect size measure.

## Preferred Phrasing

| Preferred | Avoid | Notes |
|-----------|-------|-------|
| feature absorption | feature collision | Use "absorption" for the general phenomenon; "collision" only as a measurable proxy |
| sparse autoencoder | SAE (after first use) | Spell out on first mention, abbreviate thereafter |
| fine-tuning | finetuning | Hyphenated form |
| few-shot | few shot | Hyphenated when used as adjective |
| downstream task | downstream | Always specify which task |
| reconstruction quality | reconstruction | Be specific: MSE, loss recovered, etc. |
| pre-trained | pretrained | Hyphenated form |
| ground truth | ground-truth | Hyphenated when used as adjective |
| top-k | TopK | Capitalize when referring to the architecture; lowercase for the operation |
| JumpReLU | Jump-ReLU | No hyphen in the architecture name |

## Abbreviations

| Abbreviation | Full Form | First Use |
|-------------|-----------|-----------|
| SAE | Sparse Autoencoder | Section 1 |
| CAAB | Cross-Architecture Absorption Benchmark | Section 1 |
| UAD | Unsupervised Absorption Detection | Section 1 |
| DFDA | Dynamic Feature De-Absorption | Section 1 |
| MSE | Mean Squared Error | Section 3 |
| L0 | L0 pseudo-norm (count of non-zeros) | Section 3 |
| AUROC | Area Under Receiver Operating Characteristic | Section 3 |
| ReLU | Rectified Linear Unit | Section 3 |
| MLP | Multi-Layer Perceptron | Section 3 |
| GPU | Graphics Processing Unit | Section 3 |
| API | Application Programming Interface | Section 3 |
