# Glossary

## Core Technical Terms

**Feature Absorption**
: A failure mode in sparse autoencoders where a parent feature's information is lost when child features are active, because the SAE allocates representational capacity to children at the parent's expense. First analytically characterized by Chanin et al. (2024).

**Sparse Autoencoder (SAE)**
: A neural network that learns to compress high-dimensional activations into a sparse latent representation, typically via an encoder-decoder architecture with sparsity constraints. Preferred phrasing: "sparse autoencoder" (not "sparse auto-encoder").

**Superposition**
: The phenomenon where a neural network represents more features than it has dimensions by encoding them in non-orthogonal directions. A central challenge that SAEs aim to address.

**Polysemanticity**
: The property of a single neuron or latent dimension encoding multiple unrelated concepts. The opposite of monosemanticity.

**Monosemanticity**
: The ideal property where each latent dimension encodes a single interpretable concept. SAEs aim to achieve this via sparsity.

**Residual Stream**
: The main information pathway in a transformer, where layer outputs are added to the input via skip connections. Denoted as the sequence of hidden states passed between layers.

## Metric and Evaluation Terms

**K-Sparse Probing**
: A diagnostic technique where a linear probe is trained on only the top-k most active SAE latents, measuring whether the most relevant features alone suffice for a classification task. In this work, k = 10.

**Ground-Truth Probe**
: A logistic regression classifier trained on base-model activations with known labels, serving as an upper-bound baseline for measuring information loss in SAE representations.

**Feature-Splitting Threshold (tau_fs)**
: The threshold used in k-sparse probing to determine how many latents to retain. A lower threshold retains fewer latents, making the probe more selective. This work tests tau_fs in {0.01, 0.03, 0.05}.

**Absorption Score**
: The SAEBench metric quantifying how much parent-feature information is lost in SAE representations, computed as the maximum relative accuracy drop between residual, full-SAE, and k-sparse probes. Range: [0, 1], where 0 = no absorption, 1 = complete absorption.

**AUROC**
: Area Under the Receiver Operating Characteristic curve. A probe-quality metric measuring discriminative performance independent of classification threshold. Range: [0.5, 1.0], where 1.0 = perfect discrimination.

## Architecture Terms

**Standard SAE**
: The baseline architecture with ReLU activation, trained with L1 sparsity penalty. Also called "ReLU SAE" in some literature.

**TopK SAE**
: An SAE that enforces sparsity by keeping only the k largest activations and zeroing the rest, replacing the L1 penalty with a hard constraint.

**BatchTopK SAE**
: A variant of TopK that selects the top-k activations across an entire batch rather than per-sample, improving feature diversity.

**GatedSAE**
: An SAE with a gating mechanism that separately controls the magnitude and direction of latent activations, reducing interference between features.

**JumpReLU SAE**
: An SAE using a jumping ReLU activation that introduces a threshold jump, allowing more flexible nonlinear feature boundaries.

**PAnneal SAE**
: An SAE trained with penalty annealing, gradually reducing the sparsity penalty strength during training to balance reconstruction and sparsity.

**Matryoshka SAE**
: A nested SAE architecture that learns features at multiple scales, allowing variable capacity allocation. Named after Russian nesting dolls.

## Benchmark and Dataset Terms

**SAEBench**
: A standardized evaluation framework for sparse autoencoders, providing canonical metrics including absorption, feature density, and reconstruction fidelity. Preferred phrasing: "SAEBench" (one word, capital S, B).

**SAELens**
: An open-source library for training and analyzing sparse autoencoders, providing pretrained SAE checkpoints and evaluation tools.

**WordNet**
: A lexical database of English words organized into synonym sets (synsets) linked by semantic relations including hypernymy (is-a). Used in this work to extract semantic hierarchies.

**Hypernym**
: A word with a broad meaning that encompasses more specific words (hyponyms). E.g., "animal" is a hypernym of "dog".

**Hyponym**
: A word with a specific meaning that falls under a broader category (hypernym). E.g., "dog" is a hyponym of "animal".

## Statistical Terms

**Construct Validity**
: The degree to which a measurement instrument actually measures the theoretical construct it claims to measure. In this work: whether first-letter absorption scores measure a general absorption phenomenon or an artifact of the specific task.

**Hierarchy Specificity**
: The property of a metric being sensitive to hierarchical structure specifically, as opposed to general semantic correlation. A hierarchy-specific absorption metric should score higher on parent-child pairs than on semantically related non-hierarchical pairs.

**Bootstrap Confidence Interval**
: A non-parametric method for estimating the uncertainty of a statistic by resampling the data with replacement. This work uses B = 10,000 bootstrap samples.

**Pearson Correlation (r)**
: A measure of linear correlation between two variables, ranging from -1 (perfect negative) to +1 (perfect positive).

**Paired t-test**
: A statistical test comparing the means of two related samples (e.g., the same SAE architectures measured on two different tasks).

## Abbreviations

| Abbreviation | Expansion | Preferred Usage |
|-------------|-----------|----------------|
| SAE | Sparse Autoencoder | Always spell out on first use in each section |
| LLM | Large Language Model | Spell out on first use |
| SAEBench | SAE Benchmark | Always capitalized, no space |
| SAELens | SAE Lens | Always capitalized, no space |
| ReLU | Rectified Linear Unit | No need to spell out (standard) |
| AUROC | Area Under the Receiver Operating Characteristic curve | Spell out on first use |
| CI | Confidence Interval | Spell out on first use |
| GPT | Generative Pre-trained Transformer | No need to spell out (standard) |
| NLP | Natural Language Processing | No need to spell out (standard) |
| ML | Machine Learning | No need to spell out (standard) |

## Terminology Preferences

| Preferred | Avoid | Reason |
|-----------|-------|--------|
| fine-tuning | finetuning | Standard ML spelling |
| few-shot | few shot | Compound adjective |
| first-letter | first letter | Compound modifier |
| semantic-hierarchy | semantic hierarchy | Compound modifier before noun |
| non-hierarchy | nonhierarchy | Hyphenated compound |
| k-sparse | k sparse | Compound modifier |
| ground-truth | ground truth | Hyphenated when used as modifier |
| pre-trained | pretrained | Both accepted; be consistent |
| state-of-the-art | state of the art | Hyphenated when used as modifier |
