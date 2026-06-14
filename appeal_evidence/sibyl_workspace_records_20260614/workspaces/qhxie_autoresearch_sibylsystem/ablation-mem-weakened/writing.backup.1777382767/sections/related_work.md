# 2. Background and Related Work

## 2.1 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders have emerged as the dominant unsupervised approach for decomposing neural network activations into human-interpretable features. An SAE reconstructs activations $a \in \mathbb{R}^d$ as $\hat{a} = W_{\text{dec}} \cdot f(W_{\text{enc}} \cdot a + b_{\text{pre}})$, where $f(\cdot)$ is a sparsity-inducing nonlinearity (typically ReLU), $W_{\text{enc}} \in \mathbb{R}^{n_{\text{latent}} \times d}$ is the encoder, and $W_{\text{dec}} \in \mathbb{R}^{d \times n_{\text{latent}}}$ is the decoder. The dictionary size $n_{\text{latent}}$ is typically much larger than $d$, creating an overcomplete representation where each latent ideally corresponds to a single interpretable concept.

SAEs enable several downstream interpretability tasks. Feature steering (Marks et al., 2024; Rimsky et al., 2024) adds a feature direction to model activations during the forward pass to test causal influence. Sparse probing (Templeton et al., 2024) trains linear classifiers on SAE latents to detect concepts. Circuit analysis traces how features compose to produce model behavior. Model editing modifies specific features to alter model outputs. These applications depend on the assumption that SAE features are reliable---that a feature which appears interpretable in isolation will behave predictably when used downstream.

While SAEs enable these applications, their reliability is contested. Korznikov et al. (2026) showed that random baseline SAEs match trained SAEs on standard metrics (explained variance, sparsity), raising fundamental questions about whether SAE features capture meaningful structure or merely fit activation statistics.

## 2.2 Feature Absorption

Feature absorption is a failure mode where a general (parent) feature fails to fire on positive examples, and its activation is instead captured by more specific (child) features. Chanin et al. (2024) formally identified absorption and provided strong evidence that hierarchical feature structure---where broad concepts decompose into narrower sub-concepts---causes the phenomenon. Their detection method uses differential correlation: a child $c$ is flagged as absorbing parent $f$ if the correlation between their activations, conditioned on the parent concept being present, exceeds a threshold.

Chanin et al. validated the phenomenon across hundreds of LLM SAEs spanning Gemma, Llama, Pythia, and Qwen model families. SAEBench (Karvonen et al., 2025) subsequently adopted the Chanin et al. differential correlation metric as a standardized benchmark alongside sparsity, reconstruction error, and explained variance. The metric is now reported routinely in SAE evaluations.

Despite this attention, a critical gap remains: no existing work quantifies whether absorption degrades the interpretability tasks that motivate SAE research. The field has optimized absorption metrics without establishing their relationship to task performance.

## 2.3 Downstream Interpretability Tasks

**Feature steering** tests whether a feature causally influences model behavior by adding its decoder direction to activations during inference. Marks et al. (2024) used steering to identify sparse feature circuits in language models. Rimsky et al. (2024) showed that steering can induce specific behaviors with high precision. The effectiveness of steering depends on whether the feature direction $W_{\text{dec}}[\phi(f)]$ reliably captures the target concept---precisely the property that absorption undermines.

**Sparse probing** trains linear classifiers on a small subset of SAE latents to detect whether a concept is present. Templeton et al. (2024) used sparse probing to scale interpretability to large models. The k-sparse constraint ($\|w_k\|_0 \leq k$) isolates whether a small set of features can reliably detect a concept. If a parent feature is absorbed, the probe must rely on child features or correlated latents, potentially reducing accuracy.

Neither steering nor probing has been systematically correlated with absorption rates. Prior work treats these tasks and absorption as separate evaluation dimensions. Without this bridge, architectural innovations targeting absorption reduction lack empirical justification for their design objective, and practitioners cannot determine whether absorption metrics should influence feature selection for downstream tasks.

## 2.4 Architectural Responses to Absorption

The identification of absorption has motivated several architectural innovations. Matryoshka SAEs (Bussmann et al., 2025) use a hierarchical dictionary structure where broader concepts are encoded at coarser granularities, reporting reduced absorption rates compared to standard SAEs. OrtSAE (Korznikov et al., 2026) enforces orthogonal decomposition to prevent feature overlap, achieving lower absorption through architectural constraint. HSAE (Luo et al., 2026) explicitly models hierarchical structure in the SAE architecture, directly addressing the root cause of absorption.

All three approaches target absorption reduction as a primary objective, yet none of them quantify the task-level impact of the failure mode they target. If absorption does not significantly degrade steering or probing, the field may be over-investing in solutions to a non-problem. This study tests that assumption directly.

<!-- FIGURES
- None
-->
