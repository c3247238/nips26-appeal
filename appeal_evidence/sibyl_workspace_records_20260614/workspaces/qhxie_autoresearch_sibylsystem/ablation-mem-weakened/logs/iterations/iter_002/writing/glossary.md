# Glossary

## Core Technical Terms

**Sparse Autoencoder (SAE)**
An unsupervised neural network that learns to encode high-dimensional activations into a sparse overcomplete representation and decode them back. Used in mechanistic interpretability to discover interpretable features in language models.

**Feature Absorption**
A failure mode of SAEs where a general (parent) feature fails to fire and its activation is instead captured by more specific (child) features. First identified and measured by Chanin et al. (2024).

**Locally Competitive Algorithm (LCA)**
A neuroscience-inspired sparse coding algorithm proposed by Rozell et al. (2008) where neurons compete via lateral inhibition. The inhibition matrix $G = W_{\text{dec}}^T W_{\text{dec}}$ governs competitive dynamics.

**Local Inhibition Graph**
A graph constructed from SAE decoder correlations where each latent is a node and edges connect top-k correlated neighbors. Edge weights are signed decoder correlations $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$.

**Competitive Suppression**
The mechanism by which a strongly firing latent inhibits correlated latents via decoder correlations, causing them to fail to activate. This is the proposed explanation for feature absorption.

**Homeostatic Rebalancing**
A single-pass post-hoc correction that boosts parent feature activations by the inhibition they receive from active neighbors: $z'_i = z_i + \alpha \cdot \sum_{j \in N(i)} G_{ij} z_j$.

## Measurement Terms

**Absorption Rate**
The fraction of child features that absorb a parent feature's activation. Range: [0, 1]. A rate of 0.242 means 24.2% of child features show absorption behavior.

**Differential Correlation**
The absorption detection metric from Chanin et al. (2024) that compares the correlation between a parent feature and a child feature before and after ablating the child feature. A drop in correlation indicates absorption.

**Decoder Correlation**
The inner product between two decoder directions: $G_{ij} = \langle W_{\text{dec}}[i], W_{\text{dec}}[j] \rangle$. In the LCA framework, this is the inhibition from latent $j$ to latent $i$.

**Precision@k**
The fraction of top-k neighbors in the inhibition graph that are true absorption pairs. Used to validate graph edges against ground truth absorption data.

**Recall@k**
The fraction of true absorption pairs that are found in the top-k neighbors of the parent latent.

**Total Incoming Inhibition**
The sum of absolute edge weights from all neighbors to a given latent: $\sum_{j \in N(i)} |G_{ji}|$. Predictor of at-risk features.

**Total Outgoing Inhibition**
The sum of absolute edge weights from a given latent to all its neighbors: $\sum_{j \in N(i)} |G_{ij}|$.

**Steering Success Rate**
The percentage of test prompts where steering increases the probability of the target concept. Range: [0, 1].

**Delta-Corrected Steering**
Steering success after subtracting a random feature baseline: $\Delta S = S_{\text{feature}} - S_{\text{random}}$. Isolates absorption-specific effects from generic SAE steering effects.

**k-Sparse Probe**
A linear classifier constrained to use at most k non-zero weights. Tests whether a small set of SAE latents is sufficient for a classification task.

**Precision (in probing)**
The fraction of predicted positives that are true positives. Precision = 1.0 means no false positives -- the feature only fires when the concept is present.

**Recall (in probing)**
The fraction of true positives that are predicted. Recall < 1.0 means the feature misses some instances of the concept.

**EC50 (Median Effective Concentration)**
In dose-response analysis, the steering strength at which 50% of the maximum steering effect is achieved. Lower EC50 indicates higher steering efficiency.

## Statistical Terms

**Pearson Correlation (r)**
A measure of linear correlation between two variables. Range: [-1, 1]. r = -0.431 indicates a moderate negative linear relationship.

**Spearman Rank Correlation (rho)**
A non-parametric measure of rank correlation. More robust to outliers than Pearson. Range: [-1, 1].

**AUPR (Area Under Precision-Recall Curve)**
A metric for imbalanced classification that averages precision across all recall levels. Higher AUPR indicates better prediction quality.

**Fisher Exact Test**
A statistical significance test for contingency tables, used to test whether graph edges are enriched for absorption pairs vs. random chance.

**Bonferroni Correction**
A multiple comparisons correction that divides the significance threshold by the number of tests. Controls the family-wise error rate.

**Benjamini-Hochberg (BH) Procedure**
A less conservative multiple comparisons correction that controls the false discovery rate rather than the family-wise error rate.

**Cohen's d**
A standardized measure of effect size. d = 0.2 (small), 0.5 (medium), 0.8 (large). d = 1.26 indicates a very large effect.

## Model and Architecture Terms

**GPT-2 Small**
An 85M parameter (124M including embeddings) transformer language model by OpenAI (2019). Hidden dimension: 768. Layers: 12. Used as the primary model in this study.

**res-jb SAE**
A residual-stream SAE architecture from the SAELens library, trained on GPT-2 Small activations. Dictionary size: 24,576. Released by the "jb" (Joseph Bloom) training run.

**Hook Point (hook_resid_pre)**
A TransformerLens hook point that captures residual stream activations before the attention block at a given layer.

**Decoder Direction**
The column of the SAE decoder matrix corresponding to a specific feature. Used as the steering vector: $d_f = W_{\text{dec}}[:, f]$.

**Parent Feature**
A general SAE feature that represents a broad concept (e.g., "starts with any letter"). In absorption, the parent feature fails to fire when a more specific child feature is present.

**Child Feature**
A specific SAE feature that represents a narrower concept (e.g., "starts with Q"). In absorption, the child feature captures activation that should belong to the parent feature.

**Tied Weights**
An SAE where the encoder is the transpose of the decoder: $W_{\text{enc}} = W_{\text{dec}}^T$. Ensures the structural correspondence $G = W_{\text{dec}}^T W_{\text{dec}}$ is exact.

**Activation Redistribution**
The mental model that absorption represents a reallocation of activation from parent to child features, rather than destruction of the parent feature. Supported by the finding that absorbed features maintain perfect precision and functional steering capability.

## Graph Theory Terms

**Graph Density**
The fraction of possible edges that are present in the graph. For a local graph with fixed degree k, density = $k / (d_{\text{dict}} - 1)$.

**Clustering Coefficient**
A measure of local connectivity: the fraction of a node's neighbors that are also neighbors of each other. High clustering indicates local communities.

**Centrality**
A measure of node importance in a graph. Degree centrality counts connections; betweenness centrality measures shortest-path traffic.

## Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| LCA | Locally Competitive Algorithm |
| LLM | Large Language Model |
| MI | Mechanistic Interpretability |
| EC50 | Half Maximal Effective Concentration |
| F1 | F1 Score (harmonic mean of precision and recall) |
| AUPR | Area Under Precision-Recall Curve |
| TP | True Positive |
| FP | False Positive |
| FN | False Negative |
| ReLU | Rectified Linear Unit |
| R^2 | Coefficient of Determination |
| p | p-value (statistical significance) |
| q | q-value (FDR-adjusted p-value from Benjamini-Hochberg procedure) |
| n | Sample size |
| k | Number of neighbors in local graph or sparsity level in probe |
| s | Steering strength |
| alpha | Rebalancing boost coefficient |
| SAELens | Library for training and analyzing SAEs |
| TransformerLens | Library for mechanistic interpretability of transformers |
| BH-FDR | Benjamini-Hochberg False Discovery Rate |

## Preferred Phrasing

- "fine-tuning" (not "finetuning")
- "few-shot" (not "few shot")
- "downstream task" (not "down-stream task")
- "baseline-corrected" (hyphenated when used as adjective)
- "delta-corrected" (hyphenated when used as adjective)
- "first-letter feature" (hyphenated)
- "k-sparse" (hyphenated)
- "feature-specific" (hyphenated when used as adjective)
- "absorption rate" (two words, no hyphen)
- "steering effectiveness" (two words)
- "sparse probing" (two words)
- "mechanistic interpretability" (two words, not hyphenated)
- "inhibition graph" (two words)
- "competitive suppression" (two words)
- "homeostatic rebalancing" (two words)
- "cross-model" (hyphenated when used as adjective)
- "cross-layer" (hyphenated when used as adjective)
- "training-free" (hyphenated when used as adjective)

## Domain-Specific Terms for Non-Expert Readers

**Superposition**
The phenomenon where neural networks represent more features than they have dimensions by using nearly orthogonal directions. SAEs aim to unpack superposition into interpretable features.

**Monosemantic vs. Polysemantic**
A monosemantic neuron/feature responds to a single interpretable concept. A polysemantic neuron responds to multiple unrelated concepts. SAEs aim to produce monosemantic features.

**Circuit**
In mechanistic interpretability, a subgraph of the model's computation that implements a specific task (e.g., indirect object identification).

**Residual Stream**
The main information pathway in a transformer, where hidden states are passed from layer to layer with skip connections.

**Activation Patching**
A causal intervention technique where activations from one forward pass are patched into another to test causal hypotheses about model behavior.

**Sparse Coding**
A representation learning framework where data is represented as a sparse linear combination of basis vectors (dictionary atoms). SAEs implement sparse coding with learned dictionaries.

**Lateral Inhibition**
A neural mechanism where active neurons suppress neighboring neurons, enhancing contrast and selectivity. Common in sensory processing (e.g., retina, cochlea).

**Homeostatic Plasticity**
Biological mechanisms that maintain stable neural activity levels over time. Inspired our rebalancing approach: when a neuron is suppressed by neighbors, boost it to restore baseline firing.
