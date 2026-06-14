# Related Work

## Discrete Diffusion Language Models and Inference Engineering

Discrete diffusion language models (dLLMs) extend diffusion-style iterative generation to discrete text, with early foundations such as reparameterized discrete diffusion and ratio-estimation-based formulations establishing the feasibility of non-autoregressive text denoising. As these models matured, the literature increasingly shifted from basic feasibility questions to inference-time engineering: how to schedule denoising, how to allocate compute, and how to improve generation quality without retraining the backbone.

Our work belongs to this inference-engineering tradition, but in a narrower sense than most method papers. We do not propose a new dLLM family or a new training pipeline. Instead, we study whether an observer-side uncertainty signal can justify a bounded test-time intervention claim when compared against carefully designed controls.

## Test-Time Scaling, Guided Decoding, and Memory Augmentation

Recent dLLM-adjacent work has made inference-time scaling a first-class object of study. Examples include hierarchical search and self-verification for discrete diffusion models, confidence-guided direct decoding, and persistent memory augmentation for iterative denoising. These lines collectively show that the community is no longer satisfied with plain denoising schedules; it is now actively exploring how additional computation, guidance, or state can be used to improve reasoning at test time.

Our contribution differs from these lines in two ways. First, we focus on a training-free setting and avoid heavier search or learned memory augmentation. Second, our central question is attribution: if a small gain appears, can we show that it comes from entropy-routed compute rather than from a generic frontier budget, a sampler change, or an execution mismatch? This makes our paper closer to an audit of a mechanism claim than to a maximally optimized performance paper.

## Calibration, Uncertainty, and Failure Prediction

There is a long tradition of using uncertainty-like signals to support decision making in machine learning systems. However, calibration and failure-prediction work has repeatedly warned that an informative confidence signal does not automatically translate into a useful downstream intervention rule. This caution is especially relevant in our setting. Earlier iteration-4 screening already suggested that entropy may be informative without being a valid semantic controller: entropy-guided revision did not cleanly separate from random targeting under matched compute.

The present paper follows that warning rather than fighting it. We do not attempt to rescue entropy as a semantic targeting rule. Instead, we ask whether entropy can be repurposed into a narrower role, namely routing and stopping additional compute. In this sense, the paper sits between uncertainty estimation and test-time compute allocation.

## Sampler-Centric and Runtime-Centric Attribution

Recent sampler-centric evaluation work has emphasized that method claims in diffusion systems are often confounded by decoding choices, runtime contracts, or backend differences. That critique is directly relevant to dLLM small-gain papers. Once the deltas become small, the scientific burden shifts toward transparent execution envelopes, sham controls, and explicit runtime lineage.

This is precisely the niche our work aims to fill. The main value of our result is not that the absolute GSM8K score is high, but that a modest gain survives a matched fixed-frontier sham under a unified contract. We therefore position the paper as a bounded contribution on attribution discipline for training-free dLLM intervention.

<!-- FIGURES
- None
-->
