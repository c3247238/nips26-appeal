# 1. Introduction

Sparse Autoencoders (SAEs) have become a standard tool in mechanistic interpretability for decomposing the residual stream of Large Language Models (LLMs) into human-interpretable features. By training autoencoders with L1 sparsity penalties on model activations, researchers obtain dictionaries of latent features that purportedly correspond to distinct conceptual elements. This decomposition enables circuit analysis, concept localization, and feature steering. However, a critical assumption underlying these applications is that SAE features are **independent** -- that each latent captures a unique aspect of the input that cannot be fully explained by other co-firing latents.

**Feature absorption** is the failure mode where this assumption breaks down. A latent feature is "absorbed" when its activating tokens can be well-reconstructed by other simultaneously active latents. In this case, the absorbed feature adds no independent explanatory signal: its "concept" is already captured by its co-firers. If absorption is widespread, SAE-based interpretability methods lose reliability: circuit tracing through absorbed latents double-counts causal effects, feature steering at absorbed locations produces ambiguous interventions, and concept localization conflates multiple features into one readout.

Despite its theoretical importance, feature absorption has received only qualitative treatment in the literature. Prior work identifies it as a potential concern but does not systematically quantify its prevalence, its relationship to model depth, or its dependence on SAE configuration parameters such as dictionary size and sparsity. This gap matters: without baseline measurements, the interpretability community has no principled basis for deciding when absorption is a practical concern.

We conduct the first systematic falsification study of feature absorption in pretrained SAEs. We evaluate five hypotheses about absorption prevalence and its effects, using a training-free analytical approach applied to the `gpt2-small-res-jb` SAE trained on GPT-2 small. Our methodology computes an **absorption score** for each latent: the fraction of its activating tokens where top-5 co-firing neighbors explain more than 80% of the residual-stream variance. This score directly measures the degree to which a feature's representational role is redundant with its co-activations.

Our central finding is a **100x gap** between hypothesized and observed absorption rates. Hypothesis H1 predicted that over 20% of mid-to-deep layer latents would show absorption scores above 0.5. We measured 0.19% -- well below even the falsification threshold of 10%. Across all five hypotheses, the evidence consistently points toward absorption being a rare phenomenon in GPT-2 small residual-stream SAEs, not the widespread failure mode assumed in prior qualitative discussions.

These negative results carry a constructive implication: SAE features in the residual stream of GPT-2 small are largely independent. The core assumption motivating SAE-based interpretability -- that each latent captures a distinct, non-redundant aspect of the representation -- holds in practice for this model and SAE configuration. This finding does not rule out absorption in other architectures, larger models, or MLP-out SAE configurations, but it establishes a meaningful baseline against which future claims of widespread absorption must be compared.

## 1.1 Related Work and Background

SAEs were introduced to interpretability by Cunningham et al. and subsequently scaled by Bricken et al. through the SAELens library, which released pretrained SAE checkpoints for GemmaScope and GPT-2 small. The standard SAE training objective combines a reconstruction loss with an L1 sparsity penalty:

$$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|f\|_1$$

where $x$ is the residual-stream input, $\hat{x}$ is the reconstruction, and $f$ is the sparse latent activation. The L1 penalty drives most latents toward zero for any given input, producing a decomposition where each active latent is presumed to represent an independent feature.

Feature absorption as a concept appears in qualitative discussions of SAE failure modes. Engelcke et al. and others note that co-firing latents may represent the same underlying concept, but the magnitude of this overlap has not been measured systematically. Other documented failure modes include dead latents (never activate), encoder collapse (uniform activations), and feature suppression (one latent actively inhibiting others). Absorption is distinct from these: it concerns representational redundancy among actively firing latents, not activation absence or uniform encoder behavior.

## 1.2 Summary of Contributions

We summarize our contributions as follows:

1. **First systematic quantification of feature absorption** in pretrained SAEs, using a principled metric that measures representational redundancy among co-firing latents.

2. **Five-hypothesis falsification study** covering absorption prevalence (H1), token-frequency correlation (H2), sparsity trade-offs (H3), circuit faithfulness impact (H4), and dictionary size effects (H5).

3. **Training-free analytical methodology** that can be applied to any pretrained SAE without retraining, enabling rapid hypothesis evaluation.

4. **Evidence that absorption is rare** in GPT-2 small residual-stream SAEs (0.19% prevalence at >0.5 threshold), contradicting prior qualitative assumptions of widespread redundancy.

## 1.3 Limitations

We focus exclusively on GPT-2 small (124M parameters, 12 layers, d_model=768) and the `gpt2-small-res-jb` residual-stream SAE from SAELens. Results may differ for larger models (Llama, Gemma, Mistral), MLP-out SAE configurations, or SAE training runs with different hyperparameters. The absorption score metric uses an 80% variance-explained threshold that is necessarily arbitrary; alternative thresholds could yield different prevalence estimates, though the 100x gap between hypothesis and observation suggests the qualitative conclusion is robust. Our analysis corpus of 1,024 sequences (131,072 tokens) may miss extremely rare high-absorption features, though the consistency of near-zero rates across layers argues against this concern.

## 1.4 Paper Structure

The remainder of this paper is organized as follows. Section 2 reviews related work on SAEs in interpretability, feature reliability metrics, and negative results methodology. Section 3 describes our experimental setup, the absorption score metric, and the five hypotheses tested. Section 4 presents results for each hypothesis with supporting tables and figures. Section 5 discusses the implications of our negative findings, the suspicious latents requiring investigation, and the limitations of our study. Section 6 concludes with a summary and future directions.

<!-- FIGURES
- None
-->
