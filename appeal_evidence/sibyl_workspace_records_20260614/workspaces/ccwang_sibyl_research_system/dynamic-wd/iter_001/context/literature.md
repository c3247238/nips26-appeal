# Literature Survey Report

**Research Topic**: Dynamic Alignment-Aware Weight Decay for Non-Convex SGD: From Worst-Case Alignment to Cumulative Contraction -- Theory and Algorithms
**Survey Date**: 2026-03-17
**arXiv Search Keywords**: ["weight decay alignment SGD non-convex", "adaptive weight decay gradient alignment loss landscape", "decoupled weight decay AdamW convergence non-convex", "gradient alignment weight regularization convergence", "cautious weight decay adaptive weight decay scheduled weight decay", "contraction SGD nonconvex convergence rate", "weight decay role understanding deep learning", "implicit regularization SGD weight decay nonconvex", "AlphaDecay AdaDecay module-wise weight decay layer-wise weight decay", "weight decay pitfalls gradient norm", "weight decay scaling layerwise robust"]
**Web Search Keywords**: ["adaptive weight decay SGD non-convex optimization alignment state of the art 2025", "weight decay regularization deep learning theory convergence 2024 2025 survey", "gradient alignment weight decay SGD convergence non-convex benchmark", "cautious weight decay ICLR 2026 alignment gradient direction optimizer", "cumulative contraction SGD non-convex optimization convergence theory 2024 2025", "AlphaDecay module-wise weight decay LLM heavy-tailed spectral alignment 2025", "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD CVPR 2025"]

## 1. Field Overview

Weight decay is one of the most broadly used techniques in deep learning optimization, yet its theoretical understanding in the non-convex setting has undergone a paradigm shift in 2024-2026. The classical view treated weight decay as explicit L2 regularization, but a growing body of work demonstrates that weight decay in modern deep learning acts primarily as a **training dynamics modifier** -- stabilizing optimization, controlling weight norms, and interacting with learning rate schedules and stochastic noise -- rather than as a classical regularizer. D'Angelo et al. (NeurIPS 2024) provide a unifying perspective: for both vision models and LLMs, weight decay is never useful as explicit regularization but instead changes the training dynamics in desirable ways. Sun et al. (CVPR 2025) further show that weight decay does not accelerate SGD convergence but provides the first theoretical proof of its generalization benefit in non-convex optimization.

A second major trend is the move from **uniform to adaptive/selective weight decay**. Cautious Weight Decay (CWD, ICLR 2026) applies decay only when the parameter sign aligns with the optimizer update direction, preserving the original loss objective. AlphaDecay (NeurIPS 2025) assigns module-wise decay strengths based on spectral heavy-tailedness. Selective Projection Decay (SPD) expands or contracts the search space per layer based on loss reduction consistency. These methods share a common theme: the **alignment between the weight decay force and the optimization trajectory** is critical, and uniform decay is suboptimal.

The theoretical foundations for non-convex SGD convergence are maturing rapidly. The optimal O(epsilon^{-4}) rate for finding stationary points is well established, with recent work relaxing assumptions (expected smoothness variants, relaxed step-size conditions). However, most convergence analyses treat weight decay as a fixed hyperparameter. The gap between **adaptive/alignment-aware weight decay practice** and **non-convex convergence theory** remains wide -- no existing work provides a unified theoretical framework that (1) dynamically adjusts weight decay based on gradient-weight alignment, (2) proves convergence under cumulative contraction conditions weaker than worst-case alignment, and (3) validates on modern architectures. This is precisely the research gap our proposed work targets.

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | Cautious Weight Decay | arXiv 2510.12402 / ICLR 2026 | 2025/2026 | Sign-alignment-based selective decay; bilevel optimization interpretation; sliding-mode behavior on stationary manifold | Only binary sign alignment (on/off); no cumulative alignment theory; limited to coordinate-wise analysis |
| 2 | Investigating the Role of Weight Decay in Enhancing Nonconvex SGD | CVPR 2025 | 2025 | First theoretical proof of WD generalization benefit in non-convex; convergence theory for SGDW under weak assumptions | Shows WD slows convergence; does not propose adaptive WD; worst-case alignment bounds only |
| 3 | Why Do We Need Weight Decay in Modern Deep Learning? | arXiv 2310.04415 / NeurIPS 2024 | 2023/2024 | Unifying perspective: WD as dynamics modifier, not regularizer; loss stabilization mechanism for SGD; bias-variance tradeoff for LLMs | Empirical focus; no formal convergence rate improvement from dynamic WD |
| 4 | AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs | arXiv 2506.14562 / NeurIPS 2025 | 2025 | Spectral-density-guided module-wise decay; HT-SR theory integration | Heuristic decay assignment; no per-iteration adaptation; limited to LLM pre-training |
| 5 | Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks | arXiv 2305.17212 | 2023 | Weight decay induces rotational equilibrium; balanced rotation across layers; explains AdamW vs Adam+L2 | Focuses on angular dynamics; does not formalize alignment-based decay scheduling |
| 6 | Adam-family Methods with Decoupled Weight Decay in Deep Learning | arXiv 2310.08858 | 2023 | Convergence framework for Adam-family with decoupled WD; shows framework asymptotically approximates SGD | Non-adaptive WD; fixed decay coefficient |
| 7 | Correction of Decoupled Weight Decay | arXiv 2512.08217 | 2025 | Derives WD proportional to gamma^2 for stable weight norm; Total Update Contribution analysis | Focuses on scaling rules, not alignment-based adaptation |
| 8 | Weight Norm Control | arXiv 2311.11446 | 2023 | Generalizes WD to target-norm control (AdamWN); WD as special case with target=0 | Fixed target norm; no gradient-alignment sensitivity |
| 9 | Rethinking Weight Decay for Robust Fine-Tuning of Foundation Models (SPD) | arXiv 2411.01713 | 2024 | Selective Projection Decay: layer-wise penalty based on loss reduction consistency | Fine-tuning focused; no convergence theory for non-convex from-scratch training |
| 10 | Optimizer choice matters for the emergence of Neural Collapse | arXiv 2602.16642 | 2026 | Proves decoupled WD prevents Neural Collapse in adaptive optimizers; optimizer-dependent NC dynamics | Focused on NC phenomenon; does not study generalization or convergence rates |
| 11 | GALA: Gradient Alignment-based Learning Rate Adaptation | arXiv 2506.08419 | 2025 | Gradient alignment for adaptive LR (not WD); online learning framework; data-adaptive convergence rate for normalized SGD | Adapts LR not WD; no weight-gradient alignment analysis |
| 12 | Robust Layerwise Scaling Rules by Proper Weight Decay Tuning | arXiv 2510.15262 | 2025 | WD scaling rule for muP width transfer; singular-value spectrum scales as sqrt(eta/lambda) | Scaling rule focus; not per-iteration alignment-based |
| 13 | How to set AdamW's weight decay as you scale model and dataset size | arXiv 2405.13698 | 2024 | WD as EMA timescale; optimal timescale constant in epochs across scales | Provides scaling intuition but no alignment-aware adaptation |
| 14 | Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training | arXiv 2505.13738 | 2025 | AdamW timescale constant; optimal lambda scales linearly with batch size; power law in D/N | Scaling laws; no per-step adaptation |
| 15 | Weight Decay may matter more than muP for LR Transfer | arXiv 2510.19093 | 2025 | WD stabilizes update dynamics across widths better than muP after initial training; challenges muP assumptions | Empirical; no theoretical framework for dynamic WD |
| 16 | Low-rank bias, weight decay, and model merging | arXiv 2502.17340 | 2025 | L2 regularization induces parameter-gradient alignment, norm preservation, low-rank bias at stationary points | Static analysis at stationary points; no dynamic/trajectory-level alignment theory |
| 17 | Decoupled Weight Decay for Any p Norm | arXiv 2404.10824 | 2024 | Generalizes decoupled WD to Lp norms; sparsification via p<1 without gradient divergence | Fixed decay schedule; no alignment awareness |
| 18 | ThermoLion: SNR-gated optimizer with Momentum Alignment | arXiv 2512.01881 | 2025 | Dynamic SNR gating between exploration/exploitation; Momentum Alignment detects constructive interference | Vision-centric; alignment for momentum not WD; no formal convergence theory |
| 19 | A Function Centric Perspective on Flat and Sharp Minima | arXiv 2510.12451 | 2025 | Sharp minima under regularization (SAM, WD) can generalize better; function complexity governs geometry | Challenges flatness=generalization; does not propose new WD methods |
| 20 | Demystifying the Myths and Legends of Nonconvex Convergence of SGD | ICLR 2024 | 2024 | Epsilon-stationary point exists in final iterates (not just anywhere); stronger convergence result | Does not address WD specifically |

## 3. SOTA Methods and Benchmarks

### Current Best Methods for Weight Decay
| Method | Type | Key Feature | Venue |
|--------|------|-------------|-------|
| **Cautious Weight Decay (CWD)** | Coordinate-wise selective | Sign-alignment mask; drop-in for AdamW/Lion/Muon | ICLR 2026 |
| **AlphaDecay** | Module-wise adaptive | HT-SR spectral density guided | NeurIPS 2025 |
| **SPD** | Layer-wise selective | Loss-reduction consistency | arXiv 2024 |
| **AdaDecay** | Parameter-wise adaptive | Gradient-norm proportional via sigmoid | Earlier work |
| **Weight Norm Control (AdamWN)** | Target-norm | Generalizes WD to non-zero target | arXiv 2023 |
| **Scheduled Weight Decay (SWD)** | Time-varying | Closes SGD-Adam generalization gap | OpenReview |

### Standard Benchmarks
- **Vision**: CIFAR-10/100, ImageNet (ResNet, ViT, VGG)
- **Language**: LLM pre-training (60M-2B params), OpenWebText, C4
- **Evaluation metrics**: Final validation loss/perplexity, test accuracy, convergence speed, gradient norm stability, weight norm trajectory
- **Theoretical metrics**: Convergence rate to epsilon-stationary point, generalization bound gap

### Convergence Theory Benchmarks
- Non-convex smooth: O(epsilon^{-4}) for stationary points (optimal, well-established)
- Under PL condition: O(epsilon^{-1})
- SGDW convergence: established under weaker-than-standard assumptions (Sun et al. CVPR 2025)
- CWD with Adam: O(T^{-1/2}) on squared gradient norm (Chen et al. ICLR 2026)

## 4. Identified Research Gaps

- **Gap 1: No continuous alignment-based weight decay theory.** CWD uses binary sign alignment (decay or not). No method provides a continuous, gradient-magnitude-aware alignment score that modulates decay strength smoothly. The transition from worst-case alignment assumptions to cumulative/average-case alignment conditions is unexplored.

- **Gap 2: Cumulative contraction framework is missing.** Existing convergence analyses for SGDW use per-iteration worst-case bounds. A cumulative contraction approach -- where the decay benefit is measured over trajectory segments rather than individual steps -- could yield tighter convergence rates, especially in regions where alignment is intermittent.

- **Gap 3: No unified theory connecting alignment-aware WD to convergence rate improvement.** Sun et al. (CVPR 2025) show WD helps generalization but slows convergence. CWD (ICLR 2026) preserves the loss objective but does not claim faster convergence. Can alignment-aware WD simultaneously improve convergence speed and generalization?

- **Gap 4: Parameter-gradient alignment dynamics along the training trajectory are understudied.** Kuzborskij et al. (2025) show alignment at stationary points under L2 regularization. The trajectory-level alignment dynamics -- how alignment evolves during training and how to exploit this evolution -- remain uncharacterized.

- **Gap 5: Interaction between alignment-aware WD and learning rate schedules.** The WD-LR interaction is well-studied for fixed WD (EMA timescale, gamma^2 scaling), but how alignment-aware dynamic WD interacts with warmup, cosine decay, and other LR schedules is unknown.

- **Gap 6: Lack of practical dynamic WD methods with convergence guarantees for SGD.** Most adaptive WD methods (AlphaDecay, SPD) target Adam-family optimizers. SGD-specific alignment-aware WD with proven non-convex convergence rates is absent.

## 5. Available Resources

### Open-source Code
- **CWD**: One-line modification, described in paper (no separate repo needed; drop-in for any optimizer)
- **AlphaDecay**: https://github.com/hed-ucas/AlphaDecay (PyTorch, NeurIPS 2025)
- **SPD (Selective Projection Decay)**: https://github.com/GT-RIPL/Selective-Projection-Decay (PyTorch)
- **Why Do We Need Weight Decay**: https://github.com/tml-epfl/why-weight-decay (PyTorch, NeurIPS 2024)
- **GALA (Gradient Alignment LR Adaptation)**: Described in arXiv 2506.08419 (reference for alignment-based adaptation patterns)
- **MoFaSGD**: https://github.com/pmahdavi/MoFaSGD (low-rank momentum factorized SGD)
- **IMU-1**: https://huggingface.co/thepowerfuldeez/imu1_base (training recipe with CWD + NorMuon)

### Datasets
- CIFAR-10/100: Standard torchvision
- ImageNet: Standard benchmark for vision optimization
- OpenWebText / C4: Standard for LLM pre-training experiments
- Rosenbrock function variants: For synthetic non-convex optimization validation

### Pretrained Models / Baselines
- Standard PyTorch optimizers: SGD, Adam, AdamW (built-in)
- Lion optimizer: https://github.com/google/automl/tree/master/lion
- Muon optimizer: Referenced in CWD paper
- NanoGPT: https://github.com/karpathy/nanoGPT (lightweight LLM training testbed)

## 6. Implications for Idea Generation

### Directions Worth Exploring
1. **Continuous alignment score for WD modulation**: Replace CWD's binary sign mask with a smooth, cosine-similarity-based alignment coefficient between gradient and weight vectors. This preserves the intuition (decay when aligned with optimization direction) while enabling finer control and differentiable analysis.

2. **Cumulative contraction convergence theory**: Define a trajectory-level contraction condition that averages alignment over windows of K iterations. This is analogous to the "expected smoothness" relaxation for gradient variance -- moving from worst-case per-step to amortized guarantees. Could yield O(epsilon^{-4}/kappa) rates where kappa measures average alignment quality.

3. **SGD-specific dynamic WD**: Most recent WD innovations target Adam-family. SGD's simpler update structure (no second moment) may admit cleaner theoretical analysis of alignment dynamics. The CVPR 2025 result (WD helps generalization but slows convergence) provides a baseline to improve upon.

4. **Alignment-aware WD + LR co-scheduling**: Design joint WD-LR schedules where WD increases when alignment is poor (more regularization needed) and decreases when alignment is strong (let optimization proceed). This extends the EMA timescale framework to be alignment-adaptive.

### Saturated Directions (Avoid)
- Simple scaling rules for fixed WD (gamma^2, batch-size linear) -- well-covered by recent work
- Binary sign-based masking for WD -- CWD (ICLR 2026) is definitive
- Module-wise static WD assignment -- AlphaDecay (NeurIPS 2025) covers this

### Cross-Domain Analogies with Potential
- **Federated learning gradient conflict**: Layer-wise gradient alignment detection (FedLAG, FedTAIL) uses similar alignment concepts for aggregation decisions; could inspire per-layer WD alignment thresholds
- **Multi-task gradient alignment**: Dherin (2023) shows gradient alignment as implicit regularization in multi-task settings; analogous contraction analysis may apply
- **Online learning for hyperparameter adaptation**: GALA's online learning framework for LR adaptation could be directly applied to WD adaptation
- **Control theory / sliding mode**: CWD's sliding-mode interpretation on the stationary manifold suggests deeper connections to control-theoretic optimization frameworks

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| CWD (one-line mask in optimizer) | High | MIT (ICLR paper) | **Extend** | Start from CWD's sign-alignment framework; replace binary mask with continuous alignment score |
| AlphaDecay (GitHub) | Medium | Apache-2.0 | **Compose** | Borrow spectral analysis tools for module-wise alignment characterization; combine with our per-step adaptation |
| why-weight-decay (GitHub) | Medium | MIT | **Adopt** | Use their experimental framework (ResNet/ViT on CIFAR/ImageNet, LLM on OpenWebText) as evaluation baseline |
| GALA (arXiv) | Medium | N/A | **Extend** | Adapt online learning framework (Follow-the-Regularized-Leader) from LR adaptation to WD adaptation |
| NanoGPT | High | MIT | **Adopt** | Lightweight LLM testbed for quick theoretical validation experiments |
| PyTorch SGD/AdamW | High | BSD-3 | **Extend** | Implement custom optimizer class inheriting from PyTorch base; minimal code change |
| Sun et al. CVPR 2025 theory | High | N/A | **Build** | Reference their SGDW convergence proof structure; extend with cumulative alignment conditions |

**Priority reusable components**:
- **Evaluation framework**: Reuse why-weight-decay's experimental setup (vision + LLM benchmarks, weight/gradient norm tracking, generalization gap measurement)
- **Baseline optimizers**: Standard PyTorch + CWD one-line modification + AlphaDecay
- **Theoretical proof template**: Sun et al. CVPR 2025 convergence analysis for SGDW as starting point; extend Lemmas to incorporate alignment-dependent decay rates
- **Alignment monitoring**: Implement cosine similarity tracking between gradient and weight vectors per layer, per iteration -- this is the core diagnostic for the proposed method
