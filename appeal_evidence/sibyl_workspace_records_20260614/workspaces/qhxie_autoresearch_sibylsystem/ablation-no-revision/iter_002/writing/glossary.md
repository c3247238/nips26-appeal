# Glossary

Key terminology definitions used throughout this paper. Preferred phrasing is indicated where variants exist.

## Core Concepts

**Feature Absorption** -- A phenomenon in Sparse Autoencoders (SAEs) where a latent feature's activation variance is primarily explained by the activations of other latents (its co-firers), rather than representing independent semantic content. Distinguished from superposition: absorption specifically means the feature contributes no independently reconstructable variance. Also called "redundant encoding" in some literature, distinct from "feature collision" (which refers to distinct features sharing a direction).

**Sparse Autoencoder (SAE)** -- An autoencoder trained with an L1 sparsity penalty to produce sparse, interpretable latent representations of neural network activations. The SAE maps a $d_{\text{model}}=768$ residual stream to a $d_{\text{sae}} \in \{2048, 8192, 24576\}$ sparse latent space. The encoder has weight matrix $W_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}} \times d}$ and bias $b_{\text{enc}} \in \mathbb{R}^{d_{\text{sae}}}$. The decoder has weight matrix $W_{\text{dec}} \in \mathbb{R}^{d \times d_{\text{sae}}}$ and bias $b_{\text{dec}} \in \mathbb{R}^d$.

**Residual Stream** -- The flowing activation vector between transformer layers. SAEs in this work are applied to the residual stream of GPT-2 small at layers $\ell \in \{0, 2, 4, 6, 8, 10\}$.

**Decoder Weight Matrix ($W_{\text{dec}}$)** -- The learned matrix that maps SAE latents back to the residual stream dimension. Each column corresponds to one latent's reconstruction direction.

## Absorption Score

**Absorption Score ($A_f$)** -- A metric quantifying the extent to which a latent feature's activating tokens can be reconstructed by its top-5 co-firing latents. $A_f = 1$ means the feature contributes no independent variance (fully absorbed); $A_f = 0$ means it is fully independent.

**Activating Tokens** -- Tokens where a latent's activation exceeds 1% of its maximum activation across the corpus. These are the tokens "associated with" the feature.

**Co-firing Latents** -- Latents that are simultaneously active on the same tokens as a target latent. Used to detect absorption because absorbed features fire alongside their absorbers.

**Per-token Variance Explained (RVE)** -- The fraction of residual-stream variance at a given token explained by a partial reconstruction using feature $f$ and its top-5 co-firers. Used as the basis for the absorption score.

**Always-on Features** -- Latent features that fire on >90% of corpus tokens. 38 such latents were identified in the pilot experiments. These are excluded from the primary analysis as they trivially co-fire with all other features.

**Bimodal Distribution** -- A distribution with two distinct peaks. At layer 4, absorption scores are bimodal: 25.1% of latents score $A_f=1.0$ (fully absorbed) and 34.2% score $A_f=0.0$ (fully independent), with the remaining 40.7% distributed between. This contrasts with layer 8 where 99.4% of latents score exactly 0.0.

## Sparsity

**L0 Norm** -- The number of non-zero activations for a given input. Reported as the mean L0 across all tokens in the analysis corpus. L0 is a property of activation patterns on data, not a direct measure of the training L1 penalty $\lambda$.

**L1 Sparsity Penalty ($\lambda$)** -- The coefficient on the L1 term in the SAE training loss. Higher $\lambda$ encourages sparser representations. SAEs in this work do not expose explicit $\lambda$ values; we proxy sparsity via measured L0.

## Downstream Evaluation

**Activation Patching** -- A causal intervention technique where activations from a clean run are patched into a corrupted run to measure which model components are causally important for a behavior. Also called "activation intervention" or "path patching."

**Faithfulness** -- A metric measuring how well a causal analysis (e.g., activation patching) recovers the ground-truth causal structure. Here: fraction of the clean-to-corrupted logit difference restored by patching.

**Circuit Discovery** -- The task of identifying which components (attention heads, MLP neurons, or SAE latents) are causally responsible for a model behavior. Circuit tracing via activation patching is the standard approach in mechanistic interpretability.

## Dictionary and Model Terms

**Dictionary Size ($d_{\text{sae}}$)** -- The number of latent features in an SAE. Larger dictionaries can represent more distinct features but may introduce fragmentation or redundancy.

**Random Dictionary Control** -- A baseline SAE where the decoder weights are random Gaussian columns, normalized per column. Used to establish a null distribution for the absorption score. Real SAEs should show higher absorption than random controls if the metric detects genuine structure.

**SAELens** -- The library used to load pretrained SAEs (`sae_lens`>=0.5.0). We use the `gpt2-small-res-jb` release, which provides residual-stream SAEs for all 12 layers of GPT-2 small.

## Experimental Terms

**Pilot Experiment** -- A small-scale experiment (100 sequences, single layer) designed to test feasibility and approximate effect sizes before committing to full-scale experiments (1,024 sequences, multi-layer). This paper reports pilot-scale experiments; full-scale replication remains a commitment for future work.

**Falsification Threshold** -- A pre-specified boundary that, if crossed by experimental results, rejects the hypothesis. For H1: <10% prevalence would falsify the >20% claim at layer 8.

**Inverted-U Pattern** -- A non-monotonic relationship where a quantity rises to a peak at intermediate values then declines. Observed in H3: absorption peaks at layer 4 (mid-depth: 49.3%) rather than increasing monotonically with sparsity or depth.

**Omitted Experiment** -- An experiment that was planned but not executed. Not the same as a null result; an omitted experiment yields no evidence for or against the hypothesis.

## Layer-Dependent Absorption

**Layer-Dependent** -- A property of absorption where the prevalence and intensity vary systematically across model layers. Absorption peaks at mid-layers (layer 4: 49.3%) and declines at both shallow (layer 0: 19.5%) and deeper layers (layer 8: 0.19%, layer 10: 17.3%). This is the central empirical finding of the paper.

## Preferred Terminology

| Preferred | Avoid |
|-----------|-------|
| fine-tuning | finetuning |
| few-shot | few shot |
| large language model (LLM) | LLM (in text, spell out first use) |
| activation (not "activations" as singular) | activations (as singular noun) |
| logit difference | logit diff |
| GPT-2 small | gpt2-small |
| hook point | hook |
| per-layer | layer-wise (when describing across layers) |
| statistical significance | significance (without "statistical") |
| co-firer | cofirer (in text) |
| partial reconstruction | partial-reconstruction |
| $A_f > 0.5$ (absorbed) | "clearly absorbed" without threshold |
| $A_f = 0.0$ (independent) | "completely independent" (use the notation) |
| inconclusive | "falsified" when experiment design was flawed |
| confirmed at layer 4 | "not falsified at layer 4" |
| uninformative | "falsified" for H4 (experiment design flaw, not null result) |