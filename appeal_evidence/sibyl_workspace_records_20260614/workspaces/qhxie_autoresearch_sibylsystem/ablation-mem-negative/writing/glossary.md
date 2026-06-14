# Glossary

## Core Technical Terms

**Sparse Autoencoder (SAE)**
An autoencoder with a sparsity constraint on the latent representation, trained to reconstruct neural network activations while using only a small fraction of dictionary features per input. The standard tool for mechanistic interpretability. Spell out on first mention, abbreviate to SAE thereafter.

**Feature Absorption**
The phenomenon where a broad, interpretable parent feature is suppressed by a more specific child feature when both co-occur. The SAE merges the parent into the child's latent to increase sparsity while maintaining reconstruction, creating an interpretability illusion where the parent feature fails to activate on arbitrary positive examples. Defined by Chanin et al. (2024).

**Feature Collision**
A measurable proxy for absorption: when multiple ground-truth concepts (e.g., first letters 'a', 'i', 'o') activate the same SAE feature index. Collision rate = (# concepts sharing features) / (# total concepts). Used as validation ground truth for UAD since true absorption requires supervised detection.

**Parent-Child Feature Pair**
A hierarchical relationship where a general concept (parent, e.g., "starts with S") and a specific concept (child, e.g., "short") are semantically related. In absorption, the child feature dominates and the parent feature is suppressed.

**Super-Absorber**
An SAE feature that absorbs multiple related concepts simultaneously (e.g., feature 18486 absorbing 5 first-letter concepts). Suggests multi-level hierarchical consolidation beyond binary parent-child pairs.

**Superposition**
The phenomenon where neural networks represent more features than they have dimensions by encoding features in non-orthogonal directions. Absorption is a form of superposition at the SAE level.

**Polysemanticity**
When a single neuron (or SAE feature) responds to multiple unrelated inputs. Related to but distinct from absorption: polysemanticity implies unrelated concepts, while absorption implies hierarchical/related concepts.

## SAE Architecture Terms

**JumpReLU**
An SAE architecture that uses a gating mechanism (ReLU with a learned threshold jump) to control feature activation. Achieves state-of-the-art reconstruction quality. Used in GemmaScope. No hyphen in the architecture name.

**TopK SAE**
An SAE architecture that enforces sparsity by keeping only the top-k highest activations and zeroing all others. The k parameter directly controls sparsity level. Capitalize when referring to the architecture.

**Dictionary Size ($d_{\text{SAE}}$)**
The number of features in the SAE latent space. Our experiments use $d_{\text{SAE}} = 24{,}576$.

**L0 Sparsity**
The mean number of active (non-zero) SAE features per input token. Not a true norm but a count statistic. Lower L0 means sparser representations.

**Dead Feature**
An SAE dictionary feature that never activates on any input from the evaluation dataset. High dead feature ratios indicate poor dictionary utilization.

## Absorption-Specific Terms

**Suppression Signal**
The defining characteristic of absorption: when the child feature activates, the parent feature is systematically suppressed. Formally: $\mathbb{P}(\text{parent fires} \mid \text{child fires}) \ll \mathbb{P}(\text{parent fires} \mid \text{child absent})$. This asymmetric signal is what distinguishes absorption from mere semantic correlation.

**Correlation (Semantic)**
Two features that fire together frequently because their concepts co-occur in text. A symmetric relationship: if A correlates with B, then B correlates with A. Detectable by co-occurrence clustering. **Not the same as absorption.**

**Absorption Signature**
The characteristic suppression pattern of an absorbed parent: low conditional probability $\mathbb{P}(\text{parent} \mid \text{child})$ despite high $\mathbb{P}(\text{parent} \mid \text{child absent})$. This asymmetric pattern is structurally distinct from correlation (symmetric, high co-occurrence).

## UAD-Specific Terms

**Unsupervised Absorption Detection (UAD)**
Our tested method for detecting absorbed feature pairs without ground-truth parent feature labels or supervised probe directions, using co-occurrence clustering on phi coefficient affinity matrices. **Achieves F1 = 0.007, no better than random.** Presented as a tested-and-failed approach that provides methodological insight.

**Co-Occurrence Matrix ($C$)**
A matrix where $C_{ij}$ counts how often features $i$ and $j$ activate together on the same input. Computed as $C = A^T A$ where $A$ is the binary feature activation matrix. Detects symmetric correlation, not asymmetric suppression.

**Phi Coefficient ($\phi$)**
A measure of association between two binary variables, equivalent to the Pearson correlation for dichotomous data. Used in UAD to normalize co-occurrence counts, accounting for feature activation frequency differences. A symmetric measure.

**Hierarchical Agglomerative Clustering (HAC)**
An unsupervised clustering method that builds a tree of nested clusters by iteratively merging the closest pair of clusters. UAD uses Ward linkage, which minimizes variance within clusters.

**Same-Cluster Pair**
A pair of SAE features assigned to the same cluster by HAC. In UAD, same-cluster pairs are candidate absorbed (parent, child) pairs. In practice, 99.6% are false positives (semantic correlations, not absorption).

## DFDA-Specific Terms

**Dynamic Feature De-Absorption (DFDA)**
Our training-free method for recovering absorbed parent feature activations at inference time using a lightweight residual compensation network. Does not retrain the SAE. Reported as exploratory work with preliminary positive results (21% MSE improvement on parent-positive examples).

**Residual Compensation Network**
A small MLP (97 parameters per pair) in DFDA that predicts parent feature activations from child feature activations. The predicted residual is added to the parent feature's SAE output.

**Child-Dominant Example**
An input where the child feature activates but the parent feature does not (or activates weakly). Used for training the DFDA compensation MLP.

**Parent-Positive Example**
An input where the parent feature should activate according to ground truth, regardless of child feature presence. The correct evaluation setting for DFDA. DFDA's 21% improvement was measured on parent-positive examples.

## Evaluation Terms

**Sparse Probing**
Training a linear classifier on SAE feature activations to predict a semantic concept (e.g., "does this token start with 'a'?"). Measures how well SAE features encode concepts.

**Reconstruction MSE**
Mean squared error between the original model activation and the SAE reconstruction. Lower MSE means the SAE preserves more information.

**Specificity Score**
For a feature-concept pair, the ratio of activation on the target concept to activation on other concepts. Higher specificity means the feature is more selective.

**Random Baseline**
A null model where feature pairs are selected at random. In our experiments, random selection achieves F1 = 0.0075, statistically indistinguishable from UAD's F1 = 0.007. The random baseline anchors all F1 claims.

## Model and Dataset Terms

**GPT-2 Small**
OpenAI's 124-million parameter language model (12 layers, 768 hidden dim). Primary model for all reported experiments.

**TransformerLens**
A library for mechanistic interpretability research that provides hooks into transformer internals for activation extraction.

**SAELens**
A library for training and evaluating sparse autoencoders on language model activations. Provides pretrained SAEs (GemmaScope) and training utilities.

**OpenWebText**
An open replication of OpenAI's WebText dataset, used for training and evaluating language models. Our primary evaluation dataset.

**First-Letter Features**
SAE features that detect whether a token starts with a specific letter (a-z). A standard concept set from Chanin et al. (2024) used for interpretability evaluation.

## Statistical Terms

**Spearman Rank Correlation ($\rho_S$)**
A non-parametric measure of rank correlation between two variables. Used when linear correlation assumptions may not hold.

**P-value**
The probability of observing the data (or more extreme) under the null hypothesis. $p < 0.05$ is the conventional threshold for statistical significance.

**Bootstrap Confidence Interval**
A confidence interval computed by resampling the data with replacement. Used to estimate uncertainty in F1 scores.

**Permutation Test**
A non-parametric statistical test that shuffles labels to construct a null distribution. Used to test whether UAD F1 significantly exceeds random chance.

## Preferred Phrasing

| Preferred | Avoid | Notes |
|-----------|-------|-------|
| feature absorption | feature collision | Use "absorption" for the general phenomenon; "collision" only as a measurable proxy |
| sparse autoencoder | SAE (after first use) | Spell out on first mention, abbreviate thereafter |
| pre-trained | pretrained | Hyphenated form |
| ground truth | ground-truth | Hyphenated when used as adjective |
| fine-tuning | finetuning | Hyphenated form |
| few-shot | few shot | Hyphenated when used as adjective |
| downstream task | downstream | Always specify which task |
| reconstruction quality | reconstruction | Be specific: MSE, loss recovered, etc. |
| top-k | TopK | Capitalize when referring to the architecture; lowercase for the operation |
| JumpReLU | Jump-ReLU | No hyphen in the architecture name |
| co-occurrence | cooccurrence | Hyphenated form |
| unsupervised | non-supervised | "Unsupervised" is the standard term |
| suppression signal | suppression pattern | Either is acceptable; be consistent within a section |
| negative result | failed experiment | "Negative result" is the standard scientific term |
| methodological critique | failure analysis | Either is acceptable |

## Abbreviations

| Abbreviation | Full Form | First Use |
|-------------|-----------|-----------|
| SAE | Sparse Autoencoder | Section 1 |
| UAD | Unsupervised Absorption Detection | Section 1 |
| DFDA | Dynamic Feature De-Absorption | Section 1 |
| MSE | Mean Squared Error | Section 3 |
| HAC | Hierarchical Agglomerative Clustering | Section 3 |
| MLP | Multi-Layer Perceptron | Section 3 |
| ReLU | Rectified Linear Unit | Section 3 |
| TP | True Positive | Section 3 |
| FP | False Positive | Section 3 |
| FN | False Negative | Section 3 |
| AUROC | Area Under Receiver Operating Characteristic | Section 3 |
| GPU | Graphics Processing Unit | Section 3 |
| PMI | Pointwise Mutual Information | Section 3 |
| MI | Mutual Information | Section 3 |
| CI | Confidence Interval | Section 4 |
