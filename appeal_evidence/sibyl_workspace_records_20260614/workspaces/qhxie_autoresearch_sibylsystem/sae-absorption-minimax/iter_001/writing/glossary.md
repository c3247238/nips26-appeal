# Glossary

## Key Terms

### Absorption (Feature Absorption)
A structural failure mode of Sparse Autoencoders where hierarchical LLM features cause parent features to be subsumed by their children. The parent feature fails to fire independently, even when its conceptual content is present in the input.

### Absorbed Feature
A feature that has been subsumed by its child features in the SAE dictionary. An absorbed feature may fire whenever any of its child features fires, reducing its monosemanticity and interpretability.

### ATM (Adaptive Temporal Masking)
A mitigation method for feature absorption that uses adaptive temporal masking to detect and protect high-importance features during SAE training (Li & Ren, 2025).

### AUC (Area Under ROC Curve)
A metric for discriminability that measures the ability to distinguish between classes. AUC = 0.5 indicates random performance; AUC = 1.0 indicates perfect discrimination.

### Causally Influential Feature
A feature whose decoder direction, when added to the residual stream, reliably changes model outputs. High steering sensitivity indicates strong causal influence.

### Chanin Absorption Score
The supervised absorption metric proposed by Chanin et al. (2024), measuring whether a parent feature fires when its child features do not. Higher scores indicate more severe absorption.

### Cross-Entropy Loss (CE Loss)
A loss function measuring the difference between two probability distributions, commonly used to evaluate language model perplexity and SAE reconstruction quality.

### Decoder Direction
The column of the SAE decoder matrix corresponding to a specific feature. Adding a scaled decoder direction to the residual stream constitutes a steering intervention.

### d_model
The hidden dimension size of the base language model. For GPT-2 Small, d_model = 768; for Gemma-2B, d_model = 2048.

### d_sae
The dictionary size (number of features) of a Sparse Autoencoder. Larger d_sae enables finer-grained feature decomposition but increases computational cost.

### Feature Sensitivity
**Activation sensitivity** (Tian et al., 2025): how reliably a feature activates on texts similar to its activating examples. This is NOT what we measure.
**Steering sensitivity** (our measure): how much model outputs change when we add a feature's decoder direction to the residual stream.

### First-Letter Probe
A supervised absorption measurement method for GPT-2 that identifies features activated by words starting with specific letters and measures whether the feature fires when the letter pattern is present.

### Gini Absorption
An absorption metric computed using Gini impurity on feature activation patterns. Used in pilot_h2 as an alternative to the Chanin absorption score.

### Hierarchical Feature Structure
A property of LLM representations where concepts form parent-child relationships (e.g., "math" is parent of "algebra", "geometry").

### Hubs / Hub Features
Features that are geometrically central in the residual stream — directions that participate in many concept representations. Our hypothesis: absorbed features function as hub features with high residual stream leverage.

### JumpReLU SAE
A variant of SAE using a jump threshold activation function: $f(x) = x \cdot \mathbb{1}[x > \tau]$. Features below the threshold are inactive.

### L0 Sparsity
The number of non-zero activations in a Sparse Autoencoder, computed as $L_0 = \sum_f \mathbb{1}[act_f > 0]$. Lower L0 indicates sparser representations.

### L1 Regularization
An L1 penalty ($\lambda \sum_f |act_f|$) applied during SAE training to encourage sparsity. Higher $\lambda$ produces sparser feature activations.

### Language Model (LM)
A neural network trained to predict the next token given preceding context. Examples: GPT-2 Small (117M parameters), Gemma-2B (2B parameters).

### Leakage (Feature Leakage)
When a child feature's activation pattern is encoded in the parent feature's activations, causing the parent to appear active even when its conceptual content is absent.

### Matryoshka SAE
An SAE architecture that learns features at multiple granularities simultaneously, potentially reducing absorption by sharing representations across scales (Bussmann et al., 2025).

### Mean Squared Error (MSE)
A reconstruction quality metric: $\mathcal{L}_{recon} = \|x - \hat{x}\|^2$. Lower MSE indicates better reconstruction.

### Monosemantic Feature
A feature that responds to a single, interpretable concept. The goal of SAE training is to decompose LLM representations into monosemantic features.

### OrtSAE (Orthogonal SAE)
A mitigation method that penalizes high cosine similarity between feature directions during training, reducing absorption by 65% on Gemma-2B (Korznikov et al., 2025).

### Parent Feature
In hierarchical LLM representations, a feature encoding a more general concept (e.g., "math") that subsumes more specific concepts (e.g., "algebra", "geometry").

### Phantom Feature
An absorbed feature that appears to encode a concept but fires only when its child features do, failing to provide independent causal influence.

### Pre-trained SAE
An SAE trained on a corpus (typically Pile or Pile-uncopyrighted) before downstream use. We use pre-trained SAEs from the SAELens release.

### Residual Stream
The accumulated residual connections in a transformer, representing the token representation at each layer. SAEs decompose the residual stream into interpretable features.

### SAE (Sparse Autoencoder)
A neural network that learns a sparse decomposition of inputs: encoder maps to sparse activations, decoder reconstructs the original input. Used to interpret LLM residual streams.

### SAELens
An open-source library for training and analyzing Sparse Autoencoders, providing pre-trained SAE releases and utilities for interpretability research.

### Steering Intervention
An experimental technique that adds a scaled decoder direction to the residual stream, measuring the causal effect on model outputs. Also called "activation steering" or "direction steering."

### Steering Sensitivity
The magnitude of model output change resulting from a steering intervention. Measured as logit change or probability shift.

### TopK SAE
A variant of SAE that activates exactly the k highest-scoring features. Enforces deterministic sparsity at the cost of reconstruction quality.

### UAS (Unsupervised Absorption Score)
Our proposed training-time absorption monitor: $UAS(f) = \alpha \cdot cos\_sim\_variance(f) + \beta \cdot freq\_skewness(f)$. Computed without labeled probes.

### Unsupervised Metric
A metric computed using only feature geometry and activation statistics, without requiring labeled probes or downstream tasks.

## Abbreviations

| Abbreviation | Expansion |
|--------------|-----------|
| ATM | Adaptive Temporal Masking |
| AUC | Area Under ROC Curve |
| CE | Cross-Entropy |
| d_model | Model Hidden Dimension |
| d_sae | SAE Dictionary Size |
| GPT-2 | Generative Pre-trained Transformer 2 |
| H1-H5 | Hypothesis 1-5 |
| L0 | L-zero (count of non-zero activations) |
| L1 | L-one (sparsity regularization) |
| LM | Language Model |
| LLM | Large Language Model |
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| OrtSAE | Orthogonal Sparse Autoencoder |
| SAE | Sparse Autoencoder |
| SAELens | Sparse Autoencoder Lens (library) |
| Spearman r | Spearman Rank Correlation Coefficient |
| UAS | Unsupervised Absorption Score |

## Preferred Phrasing

| Use | Avoid |
|-----|-------|
| fine-tuning | finetuning |
| few-shot | few shot |
| pre-trained | pretrained |
| ground truth | groundtruth |
| steer | drive, manipulate (in casual contexts) |
| feature absorption | absorption (when unambiguous) |
| steering sensitivity | steering effect (when discussing magnitude) |
| residual stream | residual stream (standard term) |
| decoder direction | direction vector, feature direction |
| absorption reduction | absorption mitigation (when referring to reduction metrics) |
