# 2. Background and Related Work

## 2.1 Sparse Autoencoders in Mechanistic Interpretability

Sparse Autoencoders (SAEs) have become a standard tool for decomposing neural network activations into human-interpretable features. An SAE maps a model's residual stream activation $x \in \mathbb{R}^d$ to a sparse latent vector $f \in \mathbb{R}^{d_{\text{sae}}}$ via an encoder, then reconstructs $\hat{x} = W_{\text{dec}} f + b_{\text{dec}}$ with an L1 sparsity penalty that encourages most entries of $f$ to be zero. Standard SAEs use a ReLU or gated encoder, column-normalized decoder weights, and separate encoder/decoder biases (Bricken et al., 2023; Anthropic, 2023). SAELens (Bloomfield et al., 2024) provides a library for loading pretrained SAEs, enabling reproducible evaluation across dictionary sizes and training configurations.

SAEs trained on residual stream activations decompose the flow of information between transformer layers. The residual stream at layer $\ell$ sits between the attention and MLP sublayers, making it a natural intervention point for circuit-level analysis (Elhage et al., 2021). Applications include feature probing (identifying latents that respond to specific semantic concepts), circuit discovery (determining which attention heads or MLP neurons participate in a behavior), and concept localization (mapping features to their preferred tokens). SAE quality is typically assessed along three axes: reconstruction fidelity (does $\hat{x}$ faithfully reconstruct $x$?), sparsity (how many latents are active per token?), and interpretability (do individual latents correspond to recognizable features?). Existing SAE quality metrics do not measure feature absorption — the phenomenon this paper investigates.

## 2.2 Feature Interaction Phenomena in Dictionaries

A central challenge in dictionary learning for neural networks is that feature representations are not guaranteed to be linearly independent. **Superposition** (Elhage et al., 2022) describes the phenomenon where a single neuron or dictionary direction represents multiple features simultaneously, because the number of distinct features exceeds the representational budget. This creates polysemantic neurons that respond to multiple unrelated concepts depending on context. Sharkey et al. (2023) empirically characterized superposition in GPT-2 small using probing and correlation analyses, finding that dictionary features often exhibit correlated activation patterns. Templeton (2025) studied the internal geometry of SAE dictionaries, identifying structured clustering in activation space.

These phenomena are related to but distinct from absorption. **Feature absorption** refers specifically to one feature's variance being redundantly encoded by other features — the absorbed feature contributes little independent signal to reconstruction. Superposition, by contrast, describes a geometric property: multiple features occupying nearby directions in activation space. Two features may superpose without either being absorbed — both contribute independently to reconstruction but their directions are non-orthogonal. Absorption is the more operationally severe failure mode: if feature $f$ is fully absorbed by co-firers $c \in \text{top5}(f)$, then $f$ carries no additional information beyond what is recoverable from those co-firers alone. Section 8 (H5) investigates the relationship between dictionary size and absorption rates.

## 2.3 Causal Analysis with SAEs

A growing body of work uses activation patching (also called activation intervention or path patching) to establish causal relationships between model components and behaviors. The standard paradigm runs a "clean" prompt that produces a target output and a "corrupted" prompt that deviates from it, then patches activations from the clean run into the corrupted run to measure how much the output recovers (Wang et al., 2022; Meng et al., 2022). The **faithfulness** of an intervention is measured as the fraction of the clean-to-corrupted logit difference ($\Delta_{\text{logit}}$) restored by patching:

$$\text{faithfulness} = \frac{\Delta_{\text{patch}}}{\Delta_{\text{logit}}}$$

A central open problem in mechanistic interpretability is whether SAE latents are causally meaningful — whether patching a specific latent affects model behavior in proportion to that latent's semantic importance. This question is sharpened by the possibility of absorption: if some latents redundantly encode the same information, patching those latents would produce correlated effects that complicate causal inference. Conversely, if absorption is rare, most latents are approximately independent and SAE-based circuit analysis may be more reliable than feared.

The faithfulness of SAE-based patching depends on the quality of the SAE reconstruction. If absorption causes systematic biases in the decoder — for instance, if absorbed features cause the decoder to underweight certain semantic directions — then patching through the SAE would degrade faithfulness relative to raw residual patching. Section 7 (H4) tests this directly by comparing faithfulness across raw residual patching, full SAE patching, and patching restricted to high-absorption or low-absorption latent subsets.

## 2.4 Summary of Positioning

Prior work has established that SAEs produce sparse, interpretable features, that feature representations in neural networks exhibit superposition and related interaction effects, and that activation patching enables causal circuit analysis. However, no prior work has provided a systematic quantification of absorption prevalence, a validated metric for measuring it, or an evaluation of whether absorption degrades downstream causal analyses. This paper contributes a reproducible absorption metric and the first multi-hypothesis empirical characterization of absorption across layers, dictionary sizes, and sparsity levels in GPT-2 small.

<!-- FIGURES
- None
-->
