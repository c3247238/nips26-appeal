# Literature Survey Report

**Research Topic**: Unified Dynamic Weight Decay Framework -- Unifying WD Scheduling, Alignment-Aware WD, Decoupled WD, and Norm-Matched WD into a Theoretical Framework with Standardized Evaluation Metrics (Budget Equivalence Metric, Coupling Stability Index, Alignment Informativeness Score)
**Survey Date**: 2026-03-18 (Updated: added 7 new papers from 2025-2026)
**arXiv Search Keywords**: ["weight decay scheduling dynamic adaptive", "decoupled weight decay AdamW regularization", "weight decay norm alignment gradient neural network deep learning", "cautious weight decay", "weight norm control Loshchilov", "weight decay layer-wise module-wise parameter-specific", "Scion Muon optimizer weight decay", "weight decay orthogonal radial tangential optimizer", "weight decay Vision Transformer ImageNet CIFAR", "understanding scheduling weight decay", "NOVAK unified adaptive optimizer weight decay", "gradient-to-weight ratio weight decay training dynamics", "CVPR 2025 weight decay nonconvex SGD convergence", "spectral dynamics weights rank minimization weight decay", "LLM pretraining WD schedule cosine warmup-stable-decay"]
**Web Search Keywords**: ["weight decay scheduling deep learning state of the art 2025", "decoupled weight decay AdamW regularization survey 2024 2025", "adaptive weight decay neural network training benchmark comparison", "cautious weight decay CWD optimizer 2025 paper", "weight norm control Loshchilov weight decay target norm optimizer", "weight decay deep learning open source implementation benchmark GitHub 2024 2025", "understanding scheduling weight decay Xie NeurIPS 2023 SWD optimizer", "Loshchilov Hutter decoupled weight decay regularization AdamW ICLR 2019", "NOVAK unified adaptive optimizer 2026", "gradient-to-weight ratio weight decay training dynamics Defazio 2025", "CVPR 2025 weight decay nonconvex SGD Sun", "spectral dynamics of weights rank minimization 2408.11804", "WSD warmup-stable-decay LLM training weight decay 2025 2026"]

## 1. Field Overview

Weight decay (WD) is one of the most universally applied techniques in deep learning optimization, yet the 2023-2026 period has witnessed a profound paradigm shift in understanding its role and designing its application strategy. The classical view treated WD as explicit L2 regularization -- a simple penalty that shrinks weights toward zero. However, a growing body of theoretical and empirical work demonstrates that WD in modern deep learning acts primarily as a **training dynamics modifier**: stabilizing optimization, controlling weight norms, balancing effective learning rates across layers, and interacting with stochastic noise -- rather than as a classical regularizer. D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415) provide a landmark unifying perspective: for both vision models (ResNets) and LLMs, weight decay is never useful as explicit regularization but instead changes the training dynamics in desirable ways via the "loss stabilization mechanism" for SGD and the "bias-variance tradeoff" for near-one-epoch LLM training.

This re-understanding has spawned multiple independent research threads, each proposing a distinct lens for how WD should be dynamically controlled: (1) **WD scheduling** -- adjusting WD strength over the course of training (SWD by Xie et al., log-time WD scheduling by Ferbach et al.); (2) **alignment-aware WD** -- conditioning WD on the geometric relationship between weights and optimizer updates (CWD by Chen et al., AdamO by Chen et al.); (3) **decoupled WD** -- separating WD from gradient scaling in adaptive optimizers (the foundational AdamW by Loshchilov & Hutter, and extensions to Lp norms, Huber decay, and differential privacy); (4) **norm-matched WD** -- targeting specific weight norm levels or spectral properties rather than blindly shrinking to zero (Weight Norm Control by Loshchilov, AlphaDecay by He et al.). A fifth emergent thread studies the **implicit structural effects** of WD: inducing low-rank weight matrices (Galanti et al., Kobayashi et al.), low-rank attention layers in transformers, and neural collapse geometry.

Despite this rich landscape, no existing work provides a **unified theoretical framework** that encompasses all these approaches, reveals their mathematical connections, and offers standardized metrics for comparing them. Each approach addresses a different facet of the same underlying question -- "how should weight decay interact with the training trajectory?" -- but uses different assumptions, different formulations, and different evaluation criteria. This is the gap our proposed Unified Dynamic Weight Decay Framework aims to fill.

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | **Decoupled Weight Decay Regularization** (Loshchilov & Hutter) | ICLR 2019 | 2019 | Foundational: showed L2 regularization != WD in adaptive optimizers; proposed AdamW; 10,000+ citations | Fixed WD coefficient; no scheduling or alignment awareness |
| 2 | **Why Do We Need Weight Decay in Modern Deep Learning?** (D'Angelo et al.) | NeurIPS 2024 / arXiv:2310.04415 | 2023/2024 | Unifying perspective: WD as dynamics modifier (loss stabilization for SGD, bias-variance tradeoff for LLMs); WD prevents bfloat16 loss divergence | Empirical focus; no formal convergence rate improvement from dynamic WD |
| 3 | **Cautious Weight Decay (CWD)** (Chen et al.) | ICLR 2026 / arXiv:2510.12402 | 2025 | Sign-alignment-based selective decay; bilevel Pareto-optimal interpretation; sliding-mode behavior; one-line drop-in for AdamW/Lion/Muon | Binary sign alignment only; no continuous modulation; no cumulative alignment theory |
| 4 | **Weight Norm Control (AdamWN)** (Loshchilov) | arXiv:2311.11446 | 2023 | Generalizes decoupled WD to target-norm control (target=0 is standard WD); any training run can be challenged by AdamWN with scheduled target norm | Fixed target norm; no gradient-alignment sensitivity; limited experimental validation |
| 5 | **AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs** (He et al.) | arXiv:2506.14562 | 2025 | Spectral-density-guided module-wise decay using HT-SR theory; scales 60M-1B params | Heuristic decay assignment; no per-iteration adaptation; LLM-specific |
| 6 | **Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks** (Kosson et al.) | arXiv:2305.17212 | 2023 | WD induces rotational equilibrium; balanced average rotation across layers/neurons; explains AdamW > Adam+L2 | Angular dynamics focus; does not formalize alignment-based scheduling |
| 7 | **On the Overlooked Pitfalls of Weight Decay (SWD)** (Xie et al.) | arXiv:2011.11152 / NeurIPS 2023 (OpenReview) | 2020/2023 | First practical WD scheduler; gradient-norm-aware dynamic WD; closes SGD-Adam generalization gap on CIFAR | Early work; limited theoretical foundation; one scheduling heuristic |
| 8 | **Correction of Decoupled Weight Decay** (Chou) | arXiv:2512.08217 | 2025 | Derives WD proportional to gamma^2 for stable weight norm; Total Update Contribution (TUC) analysis; validates on Scion optimizer | Focuses on scaling rules, not alignment-based adaptation |
| 9 | **Decoupled Orthogonal Dynamics (AdamO)** (Chen, Yuan, Zhang) | arXiv:2602.05136 | 2026 | Identifies "Radial Tug-of-War" conflict between WD and gradient; decouples radial (norm) and tangential (direction) dynamics; SGD-style norm control + Adam tangential | New (Feb 2026); limited large-scale validation; complex implementation |
| 10 | **Adam-family Methods with Decoupled Weight Decay** (Ding et al.) | arXiv:2310.08858 | 2023 | Convergence framework for Adam-family with decoupled WD; shows framework asymptotically approximates SGD; proposes AdamD | Non-adaptive WD; fixed decay coefficient |
| 11 | **Rethinking Weight Decay for Robust Fine-Tuning (SPD)** (Tian et al.) | arXiv:2411.01713 | 2024 | Selective Projection Decay: layer-wise penalty based on loss reduction consistency; expands/contracts per-layer search space | Fine-tuning focused; no from-scratch convergence theory |
| 12 | **Implicit Bias of AdamW: l_inf Norm Constrained Optimization** (Xie & Li) | arXiv:2404.04454 | 2024 | AdamW implicitly performs l_inf constrained optimization; connects to Frank-Wolfe algorithm | Full-batch setting only; does not extend to dynamic WD |
| 13 | **Decoupled Weight Decay for Any p Norm** (Outmezguine & Levi) | arXiv:2404.10824 | 2024 | Generalizes decoupled WD to Lp norms; enables sparsification via p<1 without gradient divergence | Fixed decay schedule; no alignment or scheduling awareness |
| 14 | **AdamHD: Decoupled Huber Decay** (Guo & Fan) | arXiv:2511.14721 | 2025 | Replaces L2 penalty with Huber regularizer; bounded decay gradients; 10-15% faster convergence; sparser weights | New penalty form but still non-adaptive scheduling; GPT-focused |
| 15 | **How to set AdamW's weight decay as you scale** (Wang & Aitchison) | arXiv:2405.13698 | 2024 | WD as EMA timescale; optimal timescale constant in epochs across model/dataset scales; muP-WD interaction | Provides scaling intuition but no alignment-aware adaptation |
| 16 | **Weight Decay may matter more than muP for LR Transfer** (Kosson et al.) | arXiv:2510.19093 | 2025 | WD stabilizes update dynamics across widths better than muP after initial training; challenges muP assumptions | Empirical; no theoretical framework for dynamic WD |
| 17 | **Logarithmic-time Schedules (ADANA)** (Ferbach et al.) | arXiv:2602.05298 | 2026 | Log-time schedules for beta1, beta2, and WD; logarithmic WD alone yields significant improvements; 40% compute efficiency gain | Complex scheduling; optimizer-specific (AdamW/AdEMAMix variants) |
| 18 | **Weight Decay Improves Language Model Plasticity** (Han et al.) | arXiv:2602.11137 | 2026 | Larger WD during pretraining produces more plastic (adaptable) models; WD encourages linearly separable representations | LLM-specific; plasticity not directly connected to training convergence |
| 19 | **OUI: Overfitting-Underfitting Indicator for WD selection** (Fernandez-Hernandez et al.) | arXiv:2504.17160 | 2025 | Novel OUI metric for monitoring WD quality during training; converges faster than traditional metrics; validation-free WD selection | Diagnostic tool only; does not propose adaptive WD algorithm |
| 20 | **SGD and Weight Decay Secretly Minimize Rank** (Galanti et al.) | arXiv:2206.05794 | 2022 | SGD + WD induces low-rank bias in weight matrices; stronger with smaller batch, higher LR, or stronger WD | No connection to dynamic WD; static analysis |
| 21 | **Weight decay induces low-rank attention layers** (Kobayashi et al.) | arXiv:2410.23819 | 2024 | L2 regularization on multiplicative parameters (attention layers) equivalent to nuclear norm; can damage LM performance | Argues for decoupling WD in attention layers from rest; structural insight |
| 22 | **Low-rank bias, weight decay, and model merging** (Kuzborskij & Abbasi-Yadkori) | arXiv:2502.17340 | 2025 | L2 regularization induces parameter-gradient alignment, norm preservation, low-rank bias at stationary points | Static analysis at stationary points; no trajectory-level alignment theory |
| 23 | **PathProx: Proximal Gradient for Weight Decay** (Yang et al.) | arXiv:2210.03069 | 2022 | For ReLU networks, WD objective equivalent to sum of L2 (not squared) norms per neuron; novel proximal gradient algorithm | ReLU-specific; different perspective on WD objective |
| 24 | **Tune without Validation (Twin)** (Brigato & Mougiakakou) | arXiv:2403.05532 | 2024 | Pipeline for tuning LR and WD without validation sets; weight norm strongly correlates with generalization | Tuning method, not adaptive WD; 20 datasets evaluated |
| 25 | **What do near-optimal learning rate schedules look like?** (Naganuma et al.) | arXiv:2603.10301 | 2026 | Most comprehensive study of optimal LR schedule shapes; weight decay has strong effect on optimal schedule shape | LR-focused but reveals critical WD-schedule interaction |
| 26 | **Norm-Hierarchy Transitions in Representation Learning** (Truong & Truong) | arXiv:2603.07323 | 2026 | WD traverses norm hierarchy from shortcut to structured representations; transition delay logarithmic in norm ratio | Theoretical insight into WD dynamics; does not propose adaptive WD |
| 27 | **Muon Optimizes Under Spectral Norm Constraints** (Chen, Li, Liu) | arXiv:2506.15054 | 2025 | Muon with decoupled WD implicitly constrains spectral norm; connects to Lion-K family | Characterizes WD's implicit effect in non-Euclidean optimizers |
| 28 | **Preconditioning for Optimization and Regularization** (Ye) | arXiv:2410.00232 | 2024 | Unified framework: AdamW selects intrinsic parameters for regularization; derives L1-regularization analogue; explains normalization methods | Theoretical framework that touches on unified perspective |
| 29 | **Why Gradients Rapidly Increase Near the End of Training** (Defazio) | arXiv:2506.02285 | 2025 | WD controls gradient-to-weight ratio ‖g‖/‖w‖; all normalized layers converge to the same steady-state ratio ("layer balancing"); explains Adam vs AdamW gap; proposes corrective term for LR-schedule interaction | LLM-specific analysis; largely ignores alignment-aware WD |
| 30 | **NOVAK: Unified Adaptive Optimizer for Deep Neural Networks** (Kavun) | arXiv:2601.07876 | 2026 | Integrates adaptive moment estimation, rectified LR scheduling, decoupled WD, Nesterov momentum, and lookahead into one optimizer; shows coupling WD with α_eff (not α) degrades generalization 4-8pp on CIFAR-100; SOTA vs 14 optimizers on CIFAR/ImageNet | Does not address alignment-aware or scheduled WD; focused on integration rather than theory |
| 31 | **Investigating the Role of Weight Decay in Enhancing Nonconvex SGD** (Sun et al.) | CVPR 2025 | 2025 | First theoretical proof of WD's generalization benefit in nonconvex SGD; proves WD does NOT accelerate convergence but improves generalization; extends to sign-based methods (SignSGD) | Theoretical framework limited to SGD; no analysis of alignment or scheduling strategies |
| 32 | **Approaching Deep Learning through the Spectral Dynamics of Weights** (Yunis et al.) | arXiv:2408.11804 | 2024 | Spectral dynamics (singular value evolution) as a unifying lens; WD promotes rank minimization across architectures; spectral dynamics distinguish memorizing from generalizing networks; connects to lottery tickets and linear mode connectivity | Empirical framework; does not propose adaptive WD algorithm |
| 33 | **Optimal LR Schedules under Functional Scaling Laws** (Ferbach et al.) | arXiv:2602.06797 | 2026 | Sharp phase transition: easy tasks → power decay to zero; hard tasks → WSD (Warmup-Stable-Decay); provides principled evaluation of cosine/linear/WSD schedules; WD-LR schedule co-design framework | LR-schedule focused; WD treated as fixed hyperparameter, not scheduled |
| 34 | **Benchmarking Optimizers for LLM Pretraining** | arXiv:2509.01440 | 2025 | Demonstrates WD's importance across optimizers; high WD with constant LR outperforms AdamW without WD on short horizons; Signum/Lion with high WD beats AdamW+no-WD | Empirical only; no theoretical analysis of WD mechanisms |
| 35 | **AdaDecay: Adaptive Weight Decay for Deep Neural Networks** (Nakamura & Hong) | arXiv:1907.08931 | 2019 | Per-parameter WD via sigmoid of layer-normalized gradient norms; improves generalization on MNIST/CIFAR; no extra backprop cost | Small-scale benchmarks only; outperformed by CPR/AdamW on GPT-2 setting per AlphaDecay 2025 comparison |

## 3. SOTA Methods and Benchmarks

### Taxonomy of Dynamic Weight Decay Methods

**A. WD Scheduling (Time-varying)**
| Method | Mechanism | Venue |
|--------|-----------|-------|
| Scheduled Weight Decay (SWD) | Gradient-norm-aware dynamic WD; reduces WD when gradient norms are large | NeurIPS 2023 |
| ADANA (log-time WD) | Logarithmic-time schedules for WD alongside beta1, beta2 | arXiv 2026 |
| Cosine/Linear WD decay | Standard schedules tied to LR schedule | Practice |
| WSD (Warmup-Stable-Decay) | Stable-phase constant LR + late decay; adopted by DeepSeek-V3; separates decay from total steps | Practice/arXiv 2026 |
| Gradient-to-weight corrective WD | Corrective term compensating for LR-schedule interaction; prevents gradient norm spike at training end | arXiv:2506.02285 |

**B. Alignment-Aware WD (Geometry-sensitive)**
| Method | Mechanism | Venue |
|--------|-----------|-------|
| Cautious Weight Decay (CWD) | Binary sign-alignment mask; decay only when sign(w) matches sign(update) | ICLR 2026 |
| AdamO (Orthogonal Dynamics) | Decouple radial (norm) and tangential (direction) dynamics; eliminate Radial Tug-of-War | arXiv 2026 |
| Selective Projection Decay (SPD) | Layer-wise penalty modulated by loss reduction consistency | arXiv 2024 |

**C. Decoupled WD (Structural separation)**
| Method | Mechanism | Venue |
|--------|-----------|-------|
| AdamW | Decouple WD from gradient scaling in Adam | ICLR 2019 |
| AdamD | Framework for Adam-family with decoupled WD; convergence guarantees | arXiv 2023 |
| Lp-norm decoupled WD | Generalize to any p-norm; sparsification via p<1 | arXiv 2024 |
| AdamHD (Huber Decay) | Replace L2 with Huber penalty; bounded gradients, sparser weights | arXiv 2025 |
| DP-AdamW | Differentially private variant with decoupled WD | arXiv 2025 |

**D. Norm-Matched WD (Target-aware)**
| Method | Mechanism | Venue |
|--------|-----------|-------|
| Weight Norm Control (AdamWN) | Target arbitrary weight norm instead of zero | arXiv 2023 |
| AlphaDecay | Module-wise decay guided by spectral heavy-tailedness (HT-SR) | arXiv 2025 |
| gamma^2 scaling | WD proportional to LR^2 for stable weight norm | arXiv 2025 |
| EMA timescale | Optimal WD derived from EMA timescale constant across scales | arXiv 2024 |

### Benchmarks and Evaluation Practices
- **Vision**: CIFAR-10/100 (ResNet-20, VGG-16-BN), ImageNet (ResNet-50, ViT, DenseNet-BC-100, EfficientNet-B0)
- **Language**: LLM pre-training (60M-2.6B params on OpenWebText, C4, Wikitext-103, Minipile); GPT-2/GPT-3 architectures; Gemma-based models (338M, 986M, 2B)
- **Common metrics**: Test accuracy, validation loss/perplexity, convergence speed (wall-clock and iteration), gradient norm stability, weight norm trajectory
- **Theoretical metrics**: Convergence rate to epsilon-stationary point, generalization bound gap
- **Emerging metrics** (not yet standardized): Weight norm stability across training, spectral density evolution, alignment cosine similarity trajectories, OUI indicator values

### Key Observation: No Standardized Evaluation Framework
Each paper uses a different combination of metrics, making direct comparison difficult. For example:
- CWD reports final loss/accuracy improvements
- AlphaDecay uses perplexity improvements and spectral density analysis
- AdamO reports generalization and stability improvements
- SWD focuses on gradient norm and generalization gap
- Wang & Aitchison focus on optimal WD scaling rules
- NOVAK (arXiv:2601.07876) compares across 14 optimizers on CIFAR/ImageNet but does not control for WD coupling design
- Defazio (arXiv:2506.02285) focuses on gradient-to-weight ratio dynamics but does not compare against alignment-aware methods
- Sun et al. (CVPR 2025) provides convergence theory for SGDW but lacks per-layer or alignment-aware analysis

**This fragmentation motivates our proposed standardized metrics**: Budget Equivalence Metric (normalizing compute), Coupling Stability Index (measuring WD-optimization interaction stability), and Alignment Informativeness Score (quantifying how much the alignment signal improves WD decisions).

## 4. Identified Research Gaps

- **Gap 1: No unified theoretical framework.** The four WD sub-fields (scheduling, alignment-aware, decoupled, norm-matched) each have independent theoretical justifications but no unifying mathematical framework. For example, CWD's bilevel Pareto-optimality interpretation, AdamWN's target-norm control, and the gamma^2 stable-norm scaling all address weight norm dynamics but from incompatible formulations. A unified framework could reveal that these are special cases of a single optimization principle.

- **Gap 2: No standardized evaluation metrics for comparing dynamic WD methods.** Each paper uses different benchmarks, different models, different metrics, and different hyperparameter selection protocols. There is no "Budget Equivalence Metric" that normalizes comparison across different compute budgets, no "Coupling Stability Index" that measures how WD interacts with optimizer state stability, and no "Alignment Informativeness Score" that quantifies the utility of geometric alignment signals for WD decisions.

- **Gap 3: Continuous alignment modulation is unexplored.** CWD (ICLR 2026) uses binary sign alignment (decay or not). AdamO (2026) separates radial/tangential components but uses fixed radial step sizing rules. No method provides a continuous, gradient-magnitude-aware alignment score that smoothly modulates decay strength along the full spectrum from fully aligned to fully opposed.

- **Gap 4: Mathematical connections between sub-approaches are uncharacterized.** It is unclear whether WD scheduling (time-varying lambda(t)) can be derived from an alignment-aware principle, or whether norm-matched WD (target tau) is equivalent to a particular scheduling strategy. The relationship between decoupled WD in adaptive optimizers and alignment-aware WD in SGD remains unformalized.

- **Gap 5: Interaction between dynamic WD and modern optimizer innovations (Muon, Scion, Lion).** The Muon/Scion family uses non-Euclidean LMO-based updates where WD's implicit effect is spectral norm constraining (Chen et al. 2025). How alignment-aware or scheduled WD should be adapted for these emerging optimizers is unknown.

- **Gap 6: Systematic visualization and diagnostic tools.** While individual papers provide ad hoc visualizations (weight norm trajectories, gradient norm plots, spectral density histograms), there is no systematic visualization framework that reveals the core problems across all WD methods and guides practitioners toward optimal strategies.

- **Gap 7: Scale-dependent behavior is poorly understood.** Wang & Aitchison (2024) show optimal WD scales with model and dataset size via EMA timescale. Kosson et al. (2025) show WD matters more than muP for LR transfer at scale. But how alignment-aware or scheduled WD should scale is completely unknown.

- **Gap 8: Gradient-to-weight ratio as unifying lens is unexploited.** Defazio (arXiv:2506.02285) shows that WD drives the gradient-to-weight ratio ‖g‖/‖w‖ of all normalized layers to the same steady-state, providing a clean theoretical explanation for the Adam vs AdamW gap. However, this "layer balancing" insight has not been connected to alignment-aware or scheduled WD methods. A unified framework could formalize how each WD sub-approach affects the gradient-to-weight ratio trajectory.

- **Gap 9: Convergence theory for nonconvex settings is nascent.** Sun et al. (CVPR 2025) provide the first proof of WD's generalization benefit in nonconvex SGD, but show WD does NOT accelerate convergence. How dynamic WD (scheduling or alignment-aware) might achieve faster convergence in nonconvex settings remains an open theoretical question.

- **Gap 10: Spectral dynamics and rank minimization are not integrated into WD scheduling.** Yunis et al. (arXiv:2408.11804) show WD promotes rank minimization via spectral dynamics. This could inspire rank-aware WD scheduling (reduce WD when desired rank is approached, increase WD to accelerate rank compression). No existing method uses spectral rank as a feedback signal for WD scheduling.

## 5. Available Resources

### Open-source Code
- **CWD (Cautious Weight Decay)**: One-line modification, described in paper (arXiv:2510.12402); drop-in for any optimizer
- **AlphaDecay**: https://github.com/hed-ucas/AlphaDecay (PyTorch, module-wise adaptive WD for LLMs)
- **SPD (Selective Projection Decay)**: https://github.com/GT-RIPL/Selective-Projection-Decay (PyTorch, layer-wise WD for fine-tuning)
- **Why Do We Need Weight Decay**: https://github.com/tml-epfl/why-weight-decay (PyTorch, NeurIPS 2024; ResNet/VGG/ViT experiments with comprehensive weight/gradient norm tracking)
- **SWD (Scheduled Weight Decay)**: https://github.com/zeke-xie/stable-weight-decay-regularization (PyTorch, NeurIPS 2023; AdamS optimizer)
- **OUI (Overfitting-Underfitting Indicator)**: https://github.com/AlbertoFdezHdez/OUI (PyTorch; DenseNet/EfficientNet/ResNet WD diagnostic)
- **AdamO (Orthogonal Dynamics)**: Described in arXiv:2602.05136 (not yet publicly released as of March 2026)
- **NanoGPT**: https://github.com/karpathy/nanoGPT (MIT, lightweight LLM training testbed)
- **Lion optimizer**: https://github.com/google/automl/tree/master/lion (Google)
- **Muon/Scion/Gluon**: Referenced in multiple papers; Gluon code at https://github.com/Ocram7/BeyondMuon
- **Clipped Scion**: https://github.com/LIONS-EPFL/ClippedScion (WD via Frank-Wolfe short step)
- **GradientStabilizer**: Described in arXiv:2502.17055 (reduces Adam's sensitivity to WD strength)
- **NOVAK optimizer**: arXiv:2601.07876 (custom CUDA kernels; 3-5x speedup; decoupled WD from rectified LR)
- **Spectral Dynamics codebase**: https://github.com/dyunis/spectral_dynamics (singular value tracking during training across architectures)
- **AdaDecay**: arXiv:1907.08931 (per-parameter sigmoid-scaled WD; reference baseline for gradient-magnitude-aware WD)
- **Gradient-to-weight ratio corrective WD**: arXiv:2506.02285 (Defazio; simple fix for gradient norm spike at training end)

### Datasets
- **CIFAR-10/100**: Standard torchvision (our primary small-scale benchmark)
- **ImageNet-1K**: Standard benchmark for vision optimization (our large-scale benchmark, as specified by project constraints)
- **OpenWebText / Wikitext-103 / C4 / Minipile**: Standard for LLM pre-training experiments (optional extension)

### Pretrained Models / Baselines
- Standard PyTorch optimizers: SGD, Adam, AdamW (built-in)
- PyTorch Lightning: High-level training framework for systematic experiments
- Weights & Biases: Experiment tracking and visualization

## 6. Implications for Idea Generation

### Directions Worth Exploring (High Priority)

1. **Unified mathematical framework connecting all four WD sub-approaches.** Formulate a general dynamic WD update rule: lambda(t, w, g) = f(alignment(w,g), norm(w), schedule(t), target_norm(tau)). Show that CWD, AdamWN, SWD, and standard decoupled WD are special cases with specific choices of f. This is the core theoretical contribution.

2. **Standardized evaluation metrics.** Design and validate:
   - **Budget Equivalence Metric (BEM)**: Normalize all comparisons to equal compute (FLOPs or wall-clock) to prevent unfair comparison between methods that use different training budgets
   - **Coupling Stability Index (CSI)**: Measure the stability of the WD-optimizer coupling (e.g., variance of effective learning rate, oscillation in weight norm trajectory, spectral condition number evolution)
   - **Alignment Informativeness Score (AIS)**: Quantify how much alignment information (cosine similarity between gradient and weight vectors) actually helps improve WD decisions compared to alignment-agnostic baselines

3. **Systematic mathematical derivation linking sub-approaches.** Prove that:
   - WD scheduling is optimal under certain trajectory assumptions derivable from alignment dynamics
   - Norm-matched WD (target tau) is equivalent to a specific alignment-aware strategy with a specific alignment threshold
   - Decoupled WD in Adam is necessary precisely because coupled WD distorts the alignment signal

4. **Large-scale visualization and diagnostic analysis.** Produce comprehensive visualization panels showing weight norm trajectories, gradient-weight alignment evolution, spectral density shifts, effective learning rate dynamics, and coupling stability across all major WD methods on shared benchmarks. This alone would be a significant contribution to the field's understanding.

### Directions to Avoid (Saturated)
- Simple scaling rules for fixed WD (gamma^2, batch-size linear) -- well-covered
- Binary sign-based masking for WD -- CWD (ICLR 2026) is definitive
- Module-wise static WD assignment -- AlphaDecay covers this
- Proposing yet another WD variant without theoretical or comparative framework

### Cross-Domain Analogies with Potential
- **Federated learning gradient conflict**: Layer-wise gradient alignment detection uses similar alignment concepts for aggregation decisions
- **Multi-task gradient alignment**: Gradient alignment as implicit regularization in multi-task settings
- **Control theory / sliding mode**: CWD's sliding-mode interpretation suggests deeper connections to control-theoretic optimization
- **Information geometry**: The relationship between WD and the natural gradient (GRNG by Dash et al. 2026) suggests information-geometric formulations
- **Gradient-to-weight ratio control**: Defazio (2506.02285) shows WD acts as a "layer balancing" controller driving ‖g‖/‖w‖ to a steady state; this suggests a feedback-control interpretation where alignment-aware WD is a more sophisticated controller that conditions the steady state on geometric signal quality
- **Spectral rank as feedback signal**: Yunis et al. (2408.11804) show WD drives rank minimization; a rank-aware WD schedule could close the loop between spectral structure objectives and WD strength, opening a new sub-direction: **spectral-feedback WD scheduling**

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| CWD (one-line mask) | High | MIT | **Extend** | Core alignment-aware baseline; extend binary mask to continuous |
| AlphaDecay (GitHub) | Medium | Apache-2.0 | **Compose** | Borrow spectral analysis for norm-matched WD characterization |
| why-weight-decay (GitHub) | High | MIT | **Adopt** | Use as primary evaluation framework: ResNet/VGG/ViT on CIFAR/ImageNet, weight/gradient norm tracking |
| SWD / AdamS (GitHub) | High | MIT | **Adopt** | Direct baseline for WD scheduling sub-approach |
| OUI (GitHub) | Medium | MIT | **Compose** | Integrate OUI as one component of the CSI metric |
| NanoGPT | Medium | MIT | **Adopt** | Lightweight LLM validation testbed |
| PyTorch SGD/AdamW | High | BSD-3 | **Extend** | Implement unified optimizer class with pluggable WD strategy |
| AdamO theory (arXiv) | High | N/A | **Build** | Reference radial/tangential decomposition; formalize within unified framework |
| Wang & Aitchison EMA analysis | High | N/A | **Build** | Incorporate EMA timescale into norm-matched sub-framework |
| NOVAK (arXiv:2601.07876) | Medium | N/A | **Compose** | Borrow decoupled WD from base-LR (not effective-LR) design insight; reuse ablation methodology |
| Spectral Dynamics (GitHub) | Medium | MIT | **Adopt** | Integrate singular value tracking as diagnostic tool for rank-based WD analysis; supports Gap 10 |
| Defazio gradient-to-weight (arXiv:2506.02285) | High | N/A | **Build** | Incorporate gradient-to-weight ratio as a unified lens metric in the CSI; implement corrective term as an optional WD scheduling variant |
| Sun et al. CVPR 2025 | Medium | N/A | **Build** | Reference convergence theory for nonconvex SGDW; extend to alignment-aware WD convergence analysis |

**Priority reusable components**:
- **Evaluation framework**: Reuse why-weight-decay's experimental setup + SWD's optimizer baselines + OUI's diagnostic tools
- **Baseline methods**: CWD, SWD/AdamS, AlphaDecay, standard AdamW, SGD+WD
- **Visualization toolkit**: Build systematic panels tracking weight norm, gradient-weight alignment (cosine similarity), spectral density, effective LR, all per-layer and aggregated
- **Theoretical foundation**: Build on Loshchilov's Weight Norm Control (general target-norm framework) as the mathematical starting point, extending with alignment and scheduling dimensions
- **Architectures**: ResNet-20 (CIFAR), VGG-16-BN (CIFAR), ResNet-50 (ImageNet), ViT (CIFAR/ImageNet) -- as specified in project constraints
- **Multi-seed protocol**: All experiments with seeds 42, 123, 456 reporting mean +/- std
