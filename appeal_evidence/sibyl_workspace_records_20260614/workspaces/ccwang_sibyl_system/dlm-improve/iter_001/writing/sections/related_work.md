# 2. Related Work

## 2.1 Diffusion language models and the move to inference-time research

Early discrete and masked diffusion language models established that denoising can serve as a viable language-modeling paradigm. Subsequent scaling work, including LLaDA and Dream, shifted the center of gravity from architecture design to inference-time use. Once DLMs became competitive enough to warrant serious deployment and benchmarking, the most immediate research question became how to decode them faster and better without retraining.

Our paper sits squarely in that inference-time regime. It does not introduce a new diffusion backbone or training objective. Instead, it studies how training-free revision methods should be compared and interpreted.

## 2.2 Training-free acceleration and selective computation

A substantial body of recent work studies inference acceleration for DLMs. Fast-dLLM and related systems use approximate caching or parallel decoding to reduce cost. DCD-style methods change the timing of token commitment. SureLock and similar approaches skip computation once tokens look converged. Other schedulers adjust which positions are unmasked together or how entropy is bounded during decoding.

These papers establish that runtime policy matters, but they typically evaluate under method-specific cost views. In contrast, our contribution is not another acceleration mechanism. It is an honest-compute protocol showing that nominal step counts alone can obscure meaningful changes in actual NFE, latency, throughput, and Pareto conclusions.

## 2.3 Revision, remasking, and fragile-token control

Another line of work studies revision itself: after an initial draft, a method revisits positions judged uncertain, brittle, or otherwise worth changing. Probe-based baselines such as CORE-like methods, entropy-guided remasking, and instability-guided revision all belong to this family. Their shared intuition is that some tokens deserve more compute than others.

What prior work has not cleanly separated is the difference between an observer and a controller. A signal may be useful for finding fragile positions without yielding a strong intervention once turned into a policy. Our signal audit addresses exactly that gap by comparing calibration, entropy, and instability under a common observer-controller framework.

## 2.4 Calibration and uncertainty in generative systems

Calibration is a classical topic in classification and autoregressive language modeling, where the central question is whether predicted confidence matches correctness. DLMs complicate this because denoising is iterative and locally confident tokens can still participate in globally wrong trajectories. Earlier drafts of this project emphasized calibration as a method ingredient. The current evidence supports a narrower role: calibration is diagnostically informative, but not the basis of a persuasive deployed controller.

This repositioning is one of the paper's main differences from prior confidence-based DLM work. We treat calibration, entropy, and instability first as measurement objects and only second as control candidates.

## 2.5 Closest distinction to prior work

The closest prior papers propose new revision or acceleration heuristics. Our paper instead asks what conclusions survive once methods are compared under honest compute, observer and controller roles are factorized, and transfer is checked across tasks with different structural demands. That is why we describe the contribution as a compute-normalized diagnostic study rather than a new dominant algorithm.

<!-- FIGURES
- None
-->
