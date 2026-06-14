# 2 Related Work

## 2.1 Sparse Autoencoders for Interpretability

Sparse Autoencoders (SAEs) have emerged as a primary tool for decomposing neural network activations into human-interpretable features. An SAE learns a sparse dictionary that decomposes the residual stream representation $x \in \mathbb{R}^d$ into $N$ latent features, where sparsity is enforced via an L1 penalty $\lambda$. The encoder computes activations $f(x) = \max(0, W_{enc}x)$, and the reconstruction is $\hat{x} = W_{dec}f(x)$.

Recent work has scaled SAE training to million-feature dictionaries on large language models. The GemmaScope release (Hubert et al., 2025) provides JumpReLU SAEs at multiple layers for Gemma-2 models. SAELens (Bricken et al., 2023) provides pretrained SAEs for GPT-2 and other models, enabling feature analysis without retraining. These releases have established SAEs as a standard tool for mechanistic interpretability research.

SAE features have been shown to correspond to interpretable concepts such as tokens, phrases, and semantic roles. Feature extraction at scale enables analysis of entire model activations rather than isolated circuits.

## 2.2 Feature Absorption in SAEs

Feature absorption is a pathological phenomenon where a general (parent) feature is subsumed by a more specific (child) feature during sparse optimization. When both parent and child concepts are present in an input, only the child feature activates, suppressing the parent's effective activation.

Chanin et al. (2024) introduced the training-free absorption detector

$$A_j = \frac{\|d_j\|^2}{d_j^\top e_j}$$

where $d_j$ and $e_j$ are the decoder and encoder vectors for feature $j$. High values of $A_j$ indicate absorption, as the decoder direction aligns poorly with its encoder output due to child feature interference.

Absorption creates systematic false negatives in feature attribution. A feature that appears inactive may in fact be absorbed, active through its child, and contributing to model behavior. This undermines SAE-based interpretability: detected features may not be independently steerable.

## 2.3 The Actionability Paradox

Basu et al. (2026) demonstrated a striking disconnect between SAE detection and steering utility. Their clinical domain features achieved 98.2% AUROC for detection but 0% output change when used for steering. This "actionability paradox" suggests that measuring absorption does not predict what we can actually do with SAE features.

The paradox has cast doubt on SAE-based interpretability for intervention tasks. Near-perfect feature detection appears to create an "interpretability illusion" where features are identified but not causally effective. Basu et al.'s findings suggest that absorption may render features permanently non-steerable.

The actionability paradox motivates our research question: are all absorbed features uniformly non-steerable, or is there heterogeneity within the absorbed population?

## 2.4 Evaluation Metrics for SAEs

SAEBench (Karvonen et al., 2025) established an 8-metric framework for evaluating SAEs, including the probe projection metric that measures how well SAE features predict model behavior. CE-Bench (Kusingo et al., 2025) provides LLM-free contrastive evaluation without requiring behavioral probes.

These evaluation frameworks have clarified what good SAE performance looks like on detection tasks. However, they primarily measure representation quality rather than actionability. The gap between Basu et al.'s 98.2% AUROC and 0% steering utility illustrates that detection metrics do not translate directly to intervention capability.

## 2.5 Architectural Solutions to Absorption

Several architectural modifications have been proposed to reduce absorption. OrtSAE (Sharkey et al., 2025) adds an orthogonality penalty to the training objective, reducing absorption by 65% in experiments. Matryoshka SAEs (Sok rotov et al., 2025) use nested dictionaries to capture features at multiple resolutions. MP-SAE (Costa et al., 2025) applies Matching Pursuit for hierarchical feature extraction that is resistant to absorption.

These approaches address absorption at the source by modifying SAE training. In contrast, our work takes a different approach: rather than preventing absorption, we investigate how to predict which absorbed features remain steerable. This is complementary to architectural solutions—the same prediction capability could help prioritize features for retraining or select intervention targets from existing SAEs.

## 2.6 Variance and Feature Steerability

The relationship between activation variance and feature behavior has been noted in prior work. Attributes with high variance across contexts tend to be more context-sensitive and specialized (Wilkus et al., 2025). This suggests that features with high coefficient of variation may route through different channels than low-variance features.

We build on this intuition by connecting CV to steering actionability specifically. Our finding that high-CV absorbed features are more steerable than low-CV absorbed features suggests that CV captures something about the causal routing of absorbed features—specifically, whether they route through context-sensitive channels (high-CV) or stable bypass channels (low-CV).

<!-- FIGURES
- None
-->