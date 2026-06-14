# Glossary

## Core Terms

**Sparse Autoencoder (SAE)**
An unsupervised neural network that learns a sparse overcomplete representation of input data. In mechanistic interpretability, SAEs decompose neural network activations into interpretable features. Consists of an encoder, a sparsity-inducing nonlinearity (typically ReLU), and a decoder.

**Feature Absorption**
A failure mode of SAEs where a general (parent) feature fails to fire when its concept is present, and its activation is instead captured by more specific (child) features. For example, a "starts with A" feature may be absorbed by "starts with Apple" or "starts with Ant" features.

**Mechanistic Interpretability (MI)**
The subfield of AI research that seeks to reverse-engineer neural networks into human-understandable algorithms. Goal: understand what a model computes and how, at the level of individual neurons, circuits, and features.

**Feature Steering**
An interpretability technique where a feature direction (from an SAE decoder) is added to model activations during forward pass to influence model outputs. Used to test whether a feature causally influences model behavior.

**Sparse Probing**
Training a linear classifier on a small subset (k) of SAE latents to detect whether a specific concept is present in the input. Measures how well a feature encodes its concept in isolation.

## SAE Architecture Terms

**Latent / Dictionary Element**
A single unit in the SAE's hidden layer. After training, each latent typically corresponds to an interpretable concept. Also called "feature" or "neuron" in SAE literature.

**Dictionary Size**
The number of latents in the SAE hidden layer. Denoted $n_{\text{latent}}$. Larger dictionaries can represent more concepts but may increase absorption. Our study uses 24,576 latents (GPT-2) and 32,768 latents (Pythia).

**Residual Stream**
The main activation pathway in a transformer model. SAEs are typically applied to residual stream activations at specific layers. Denoted $a_\ell$ for layer $\ell$.

**hook_resid_pre**
A TransformerLens hook point capturing residual stream activations before the attention and MLP sublayers at a given layer. Our GPT-2 experiments use hook_resid_pre at layers 0, 4, 8, and 10.

**hook_resid_post**
A TransformerLens hook point capturing residual stream activations after the attention and MLP sublayers at a given layer. Our Pythia-70M experiment uses hook_resid_post at layer 4.

## Absorption-Specific Terms

**Parent Feature**
A general feature that should fire for a broad concept (e.g., "starts with A"). In absorption, the parent feature fails to activate despite the concept being present.

**Child Feature**
A more specific feature that captures a subset of the parent's concept (e.g., "starts with Apple"). In absorption, child features "steal" activation from the parent.

**Absorption Rate**
The fraction of child features that absorb a parent feature's activation, as measured by the Chanin et al. differential correlation method. Denoted $A(f) \in [0, 1]$. Higher values indicate more absorption.

**Differential Correlation**
The Chanin et al. (2024) metric for detecting absorption. Measures the correlation between a parent feature's activation and each child feature's activation, conditioned on the parent concept being present.

## Experimental Terms

**First-Letter Feature**
An SAE latent that activates when the next token starts with a specific letter (A-Z). Used as a canonical test case for absorption because letters have clear hierarchical structure (e.g., "A" -> "Apple", "Ant", "April").

**Steering Strength**
The scalar multiplier $s$ applied to a feature direction when steering. Higher strengths produce stronger effects but may cause degradation. We test $s \in \{1.0, 2.0, 5.0, 10.0, 20.0, 50.0\}$.

**Raw Steering Success Rate**
The fraction of test prompts where steering increases the probability of the target concept (e.g., words starting with the target letter), without any baseline correction. Denoted $S(f, s) \in [0, 1]$.

**Random Feature Baseline**
A control condition where randomly selected SAE latents (rather than feature-specific latents) are used for steering. Measures the baseline steering effect from arbitrary directions. Denoted $S_{\text{rand}}(s)$.

**Delta Steering Success**
The difference between feature-specific steering success and random baseline steering success: $\Delta S(f, s) = S(f, s) - S_{\text{rand}}(s)$. This controls for baseline steering effects and isolates the true feature-specific contribution.

**Probability Lift**
The average increase in target token probability when steering is applied, relative to the unsteered baseline. Denoted $\Delta P(f)$.

**EC50 (Median Effective Steering Strength)**
The steering strength at which 50% of the maximum steering success is achieved. Computed by linear interpolation between dose points bracketing 50% success. Used to test whether absorption affects steering efficiency (H4).

**k-Sparse Probe**
A linear classifier trained on only the $k$ most active SAE latents for a given task. Measures whether a small set of features can reliably detect a concept. We test $k \in \{1, 5, 10, 20\}$.

**F1 Score**
The harmonic mean of precision and recall: $\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$. Used to evaluate probing accuracy.

**Precision (in probing)**
The fraction of predicted positives that are true positives. In our context: when the probe predicts a letter, how often is it correct? Denoted $\text{Prec} \in [0, 1]$.

**Recall (in probing)**
The fraction of true positives that are predicted. In our context: of all tokens starting with the target letter, what fraction does the probe correctly identify? Denoted $\text{Rec} \in [0, 1]$.

## Statistical Terms

**Pearson Correlation (r)**
A measure of linear correlation between two variables. $r \in [-1, 1]$. We use Pearson r to test the linear relationship between absorption rate and task performance.

**Spearman Rank Correlation ($\rho$)**
A non-parametric measure of rank correlation. Robust to non-linear relationships and outliers. We report Spearman $\rho$ alongside Pearson r.

**Coefficient of Variation (CV)**
The ratio of standard deviation to absolute mean: $\text{CV} = \sigma / |\mu|$. Used to test consistency of degradation coefficients across layers (H3). The absolute value on the denominator is necessary because regression slopes can have opposite signs, making the raw mean near zero and the raw CV negative and uninterpretable. The Method section (Section 4.6) defines $\text{CV} = \sigma / |\mu|$ with the same rationale.

**p-value**
The probability of observing the data (or more extreme) under the null hypothesis. We use $p < 0.05$ as the significance threshold.

**R-squared ($R^2$)**
The proportion of variance in the dependent variable explained by the independent variable. $R^2 \in [0, 1]$. Low $R^2$ indicates weak explanatory power.

**Cohen's d**
A standardized measure of effect size, defined as the difference between two means divided by the pooled standard deviation. $d < 0.2$ = small, $0.2 \leq d < 0.8$ = medium, $d \geq 0.8$ = large. We report Cohen's d for the random baseline validation.

## Model and Dataset Terms

**GPT-2 Small**
A 124M-parameter (85M non-embedding) transformer language model by OpenAI. Our primary model. 12 layers, 768 hidden dimensions, 12 attention heads.

**Pythia-70M**
A 70M-parameter (19M non-embedding) transformer language model by EleutherAI. Used for cross-model validation. 6 layers, 512 hidden dimensions, 8 attention heads.

**res-jb SAEs**
A family of pre-trained SAEs for GPT-2 Small released by Joseph Bloom (via SAELens). Trained with standard SAE architecture on residual stream activations. Dictionary size: 24,576 latents.

**res-sm SAEs**
A family of pre-trained SAEs for Pythia models released via SAELens. Dictionary size: 32,768 latents for Pythia-70M.

**SAELens**
An open-source library for loading, analyzing, and visualizing pre-trained SAEs. Provides standardized interfaces for SAE evaluation.

**TransformerLens**
An open-source library for mechanistic interpretability of transformer models. Provides hook-based activation caching and steering capabilities.

**SAEBench**
A standardized benchmarking framework for SAE evaluation. Includes absorption, sparsity, reconstruction, and other metrics.

## Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| SAE | Sparse Autoencoder |
| MI | Mechanistic Interpretability |
| LLM | Large Language Model |
| ReLU | Rectified Linear Unit |
| F1 | F1 Score (harmonic mean of precision and recall) |
| CV | Coefficient of Variation |
| HF | HuggingFace |
| GPU | Graphics Processing Unit |
| MLP | Multi-Layer Perceptron |
| EC50 | Median Effective Concentration/Strength |

## Preferred Phrasing

- "fine-tuning" (not "finetuning")
- "few-shot" (not "few shot")
- "pre-trained" (not "pretrained")
- "downstream" (not "down-stream")
- "first-letter" (hyphenated when used as modifier: "first-letter features")
- "feature absorption" (not "absorption phenomenon" or "absorption effect" alone)
- "steering effectiveness" (not "steering performance" or "steering quality")
- "sparse probing" (not "sparsity probing" or "probe sparsity")
- "null result" (not "negative result" when referring to absence of effect)
- "not supported" (not "rejected" or "falsified" for hypotheses)
- "delta steering" (not "corrected steering" or "normalized steering")
- "random baseline" (not "random control" or "baseline control")
- "cross-model validation" (not "cross-validation" when referring to testing on a different model)
