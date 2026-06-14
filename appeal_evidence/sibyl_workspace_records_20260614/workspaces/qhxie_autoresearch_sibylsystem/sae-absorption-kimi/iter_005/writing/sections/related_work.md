# 2. Related Work

## 2.1 From Superposition to Sparse Autoencoders

The theoretical foundation for SAEs rests on the superposition hypothesis articulated by Elhage et al. (2022): neural networks represent more features than they have dimensions by encoding them as non-orthogonal linear combinations of activations. This overcomplete representation is necessary because the number of meaningful concepts in language far exceeds the residual-stream dimension, but it creates polysemanticity---individual neurons that respond to multiple unrelated stimuli---making direct interpretation intractable.

Bricken et al. (2023) demonstrated that SAEs could recover monosemantic features from a one-layer transformer, sparking rapid expansion of the field. Cunningham et al. (2023) scaled SAE training to larger models, and by 2024--2025 the literature spans multiple architectures (ReLU, TopK, JumpReLU, Gated, Matryoshka), comprehensive benchmarks, and applications beyond language (vision, proteins, RNA). Templeton et al. (2024) extracted millions of interpretable features from Claude 3 Sonnet, demonstrating that SAEs can operate at frontier-model scale.

The central tension in this line of work is between reconstruction fidelity and interpretability. SAEs optimize a trade-off between reconstructing the original activations and enforcing sparsity in the latent representation. The sparsity-reconstruction Pareto frontier has become the standard way to compare architectures, but Karvonen et al. (2025) showed that gains on this frontier do not reliably predict downstream interpretability outcomes. Our work extends this skepticism to a specific architectural question: which component on the frontier actually drives absorption reduction?

## 2.2 Feature Absorption and Its Mitigation

Chanin et al. (2024) identified feature absorption as a structural failure mode analytically incentivized by sparsity loss. When parent and child features co-occur, the SAE allocates capacity to children at the parent's expense, creating "holes" in feature coverage. Their analytical proof applies to hierarchical features with parent-child containment structure, and their empirical validation used first-letter classification tasks.

Subsequent work has pursued two directions: mitigation and characterization. Bussmann et al. (2025) introduced Matryoshka SAEs, which learn nested dictionaries of increasing size and report reduced absorption rates from 0.49 to 0.05 on the first-letter task. Korznikov et al. (2025) enforced orthogonality constraints (OrtSAE), reducing absorption by 65% with 4--11% compute overhead. Zhan et al. (2026) proposed hierarchical SAEs with explicit tree-structured constraints. Rajamanoharan et al. (2024) introduced JumpReLU and Gated SAEs, which improve reconstruction fidelity and sparsity trade-offs but do not directly target absorption.

A critical counterpoint is Chanin's (2025) finding of feature hedging---the opposite failure mode where reconstruction loss drives correlated features to share latents rather than split them. Matryoshka SAEs, while reducing absorption, increase hedging in inner dictionary levels. This reveals that absorption and hedging sit on a Pareto frontier: no known architecture simultaneously minimizes both. Our finding that hedging is invariant across tested variants (~0.24) adds a new observation to this trade-off space, though we caution that synthetic data may not trigger hedging as real LLM features do.

## 2.3 Benchmarks and Metric Validation

SAEBench (Karvonen et al., 2025) standardized eight evaluations for SAE comparison, including absorption, sparse probing, automated interpretability, loss recovery, and feature disentanglement. The absorption evaluation adapts Chanin et al.'s protocol: 26 first-letter hierarchies, ground-truth logistic probes, k-sparse probing with feature-splitting detection, and a composite absorption score. SAEBench has become the dominant community benchmark, with 200+ SAEs evaluated and an interactive leaderboard at neuronpedia.org/sae-bench.

Despite its centrality, the absorption metric has not undergone construct-validation testing on real semantic hierarchies. Our prior work (iterations 2--4) was the first to test this, finding that the metric fails hierarchy specificity (non-hierarchies score higher than hierarchies) and produces degenerate scores on a Random-SAE control. These findings motivated our pivot to ground-truth synthetic data, where measurement is unambiguous.

The broader issue of proxy metric validity has received increasing attention. Kantamneni et al. (2025) showed that SAEs do not consistently outperform strong non-SAE baselines on downstream sparse probing tasks, questioning whether SAE-based features carry genuine utility beyond interpretability appeal. Lieberum et al. (2024) found that feature interpretability ratings from automated methods correlate weakly with human judgments. These findings align with our motivation: metrics that appear meaningful in controlled settings may not support causal claims without ground-truth validation.

## 2.4 Synthetic Evaluation for Mechanistic Interpretability

Toy models and synthetic data have a long history in mechanistic interpretability. Elhage et al. (2022) used synthetic superposition models to derive theoretical predictions about feature geometry. Lieberum et al. (2023) trained SAEs on synthetic data with known feature structure to validate dictionary learning. SynthSAEBench (Chanin & Garriga-Alonso, 2026) extends this tradition to 16,384 ground-truth features with hierarchical structure, built into the SAELens library for community use.

The role of synthetic benchmarks is to validate claims before real-LLM deployment. If an architectural component fails to reduce absorption on ground-truth synthetic data---where measurement is direct and unambiguous---it is unlikely to succeed on real LLMs, where measurement is confounded by probe artifacts and geometric variation. Our component-isolated design leverages this logic: we test causal claims on synthetic data first, establishing a foundation for real-LLM validation in future work.

## 2.5 Positioning This Work

Our study occupies the intersection of three literatures: SAE failure-mode characterization, architectural ablation, and synthetic benchmark validation. Unlike architecture papers that propose new SAE variants and report absorption scores (Bussmann et al., 2025; Korznikov et al., 2025), we do not introduce a new method. Unlike theoretical work that analyzes absorption as an optimization property, we do not derive new bounds or guarantees. Instead, we ask a methodological question: which component of existing architectures actually drives absorption reduction?

This question is timely because absorption has become a primary criterion for architecture comparison. Papers routinely report first-letter absorption as a key result, and SAEBench's interactive leaderboard ranks SAEs partly by this metric. If the reported improvements come primarily from a single component (e.g., TopK sparsity) that is already widely available, then the community's investment in more complex architectures may be inefficient. Our component-isolated design provides the first evidence for answering this question.

We now describe the measurement protocol that enables causal component isolation on SynthSAEBench-16k.

<!-- FIGURES
- None
-->
