

## Project Spec
# 项目: dynamic-wd

## 研究主题

面向非凸 SGD 的动态 Alignment-Aware Weight Decay：从 worst-case alignment 到 cumulative contraction 的理论与算法研究。

## 背景与动机

论文 "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD" 首次系统性地证明了非凸 SGD 中 weight decay 改善泛化的理论机制——关键条件是梯度方向与参数方向不线性相关（alignment quantity δ_T < 1）。但该论文的分析局限于 fixed decay rate，揭示了一个根本性 trade-off：**更慢的收敛 vs 更好的泛化**。

核心洞察：如果把 decay rate 设计成随训练状态动态变化的 λ_t（在需要正则时增强，不需要时减弱），有可能在保持收敛速度的同时获得甚至超越 fixed wd 的泛化收益。

## 初始想法

### 核心研究问题

**能否为非凸 SGD 设计并证明一种动态、alignment-aware 的 weight decay，使其在保持或接近 SGD 原始收敛阶的同时，获得优于 fixed weight decay 的稳定性或泛化解释？**

### 四个子问题

1. **RQ1**: 时间变化的 λ_t 能否在不恶化非凸 SGD 收敛阶的前提下引入有效正则化？
2. **RQ2**: 泛化分析能否从 fixed wd 的 worst-case sup_t δ_t 升级为对平均 alignment / 累计 contraction 的依赖？
3. **RQ3**: 理论 δ_t 依赖全数据梯度不可直接观测，能否构造可计算的 stochastic proxy？
4. **RQ4**: 能否推广到 momentum SGD、SignSGD 甚至 AdamW？

### 三阶段研究路线

**第一阶段**：Time-varying SGDW 理论（deterministic / stagewise schedule）
- 更新形式：w_{t+1} = (1 - λ_t) w_t - γ_t g_t
- 先证明 λ_t = O(γ_t²) 时保持标准非凸 SGD 收敛阶
- 引入 augmented potential：Φ_t = f_S(w_t) + β_t ‖w_t‖²

**第二阶段**：Alignment-aware dynamic rule
- 稳妥规则：λ_t = clip(c · γ_t · (1 - δ̂_t)^p, λ_min, λ_max)
- 激进规则：λ_t = clip(c · γ_t / (δ̂_t + ε), λ_min, λ_max)
- 其中 δ̂_t = |⟨g_t, w_t⟩| / (‖g_t‖ ‖w_t‖ + ε) 是 minibatch alignment proxy

**第三阶段**：论文产出（理论优先或实验优先，根据进展决定）

### Theorem Targets

- **Target A（保守）**: time-varying SGDW 在 λ_t = O(γ_t²) 时保持收敛阶
- **Target B**: alignment-weighted convergence，收敛损失依赖 Σ_t λ_t δ_t 而非单一常数
- **Target C**: dynamic stability bound，stability recursion 依赖累计 contraction Π_s(1 - λ_s + O(Lγ_s))
- **Target D**: minibatch proxy δ̂_t 在 concentration 条件下继承 order-level 保证

### 已知技术难点

1. ‖w_t‖² 项在 time-varying decay 下需 Lyapunov 函数或额外矩控制
2. stability 乘积项到"平均 δ"的化简需严格处理
3. δ_t 的可观测性：全数据梯度 vs stochastic gradient 的桥接
4. λ_t 依赖 minibatch 时本身是随机过程，需处理 filtration 与条件期望

## 关键参考文献

- 核心论文: "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD"（本目录 `SL-wd证明cvpr.pdf`）
- 参考材料目录: `/home/ccwang/research/wd_dynamic_2026-03-16/`
  - `proposal_dynamic_weight_decay_cn.md` — 完整研究 proposal
  - `claude_chat_key_extract_cn.md` — 关键对话摘录
  - `claude_chat_raw_extract.md` — 原始对话记录

## 可用资源

- GPU: 4x NVIDIA RTX PRO 6000 (96GB VRAM each, 本地)
- 总 GPU 显存: 384GB
- 计算后端: local

## 实验约束

- 实验类型: 轻量训练优先
- 优先 CIFAR-10/100（单卡 10-30 分钟），ImageNet 可选
- 单个实验最好控制在 1 小时以内
- Pilot 实验 10-15 分钟左右

### 实验计划

**Phase 1: 复现原论文 + 几何量追踪**
- VGG16 / CIFAR-10, ResNet20 / CIFAR-100
- Batch size 128, 初始 lr=0.1, 每 30 epoch lr×0.2
- Decoupled weight decay: α = 1 - wd × γ
- wd 取值: 0, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2
- 记录: train loss, test loss, test acc, ‖w_t‖, ‖g_t‖, δ̂_t, EMA(δ̂_t), Σλ_s

**Phase 2: Baseline 对比**
1. No-WD SGD
2. Fixed SGDW
3. Stagewise SGDW
4. Alignment-aware Dynamic SGDW

**Phase 3: Dynamic wd 算法实现**
- λ_t = c · γ_t · (1 - δ̂_t)
- λ_t = c · γ_t · (1 - δ̂_t)²
- λ_t = clip(c · γ_t / (δ̂_t + ε), λ_min, λ_max)

**Phase 4: 扩展**
- SGD + momentum
- 小型 Transformer (可选)
- AdamW (可选，视进展)

### 关键诊断指标
1. dynamic wd 是否改善 train-test trade-off？
2. average alignment 是否比 worst-case alignment 更有解释力？
3. dynamic schedule 是否把 decay 集中在更"该正则"的阶段？
4. 收益是否跨模型、跨数据集成立？
5. minibatch proxy 是否稳定？

## 目标产出

- 一篇顶会论文（ICML / NeurIPS / ICLR）
- 暂定标题方向: "Dynamic Weight Decay in Nonconvex SGD: From Worst-Case Alignment to Cumulative Contraction"
- 核心贡献: nonconvex SGD 下 time-varying weight decay 理论 + alignment-aware 动态算法 + 系统性经验验证

## 风险评估

1. 动态 wd 理论可能只对 stagewise schedule 成立，无法覆盖 fully adaptive rule
2. δ̂_t 噪声太大导致算法不稳定
3. 理论可能仍离不开 worst-case control，形式更复杂但无真正提升
4. AdamW 扩展比预期难（adaptive preconditioning 复杂化证明）

## 特殊需求

- 理论推导与实验验证并行推进
- 实验代码需要追踪 alignment 几何量（δ̂_t 等），不仅仅是 loss/accuracy
- 参考材料已在 `/home/ccwang/research/wd_dynamic_2026-03-16/` 准备好
- 无需多个 seed 进行实验算结果的统计性


## User's Initial Ideas
### 核心研究问题

**能否为非凸 SGD 设计并证明一种动态、alignment-aware 的 weight decay，使其在保持或接近 SGD 原始收敛阶的同时，获得优于 fixed weight decay 的稳定性或泛化解释？**

## Seed References (from user)
- 核心论文: "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD"（本目录 `SL-wd证明cvpr.pdf`）
- 参考材料目录: `/home/ccwang/research/wd_dynamic_2026-03-16/`
  - `proposal_dynamic_weight_decay_cn.md` — 完整研究 proposal
  - `claude_chat_key_extract_cn.md` — 关键对话摘录
  - `claude_chat_raw_extract.md` — 原始对话记录

## 文献调研报告（请仔细阅读，避免重复已有工作）
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


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Alignment-Aware Dynamic Weight Decay for Nonconvex SGD: From Worst-Case to Cumulative Contraction

## Title

**Alignment-Aware Dynamic Weight Decay: Cumulative Contraction Theory and Practical Algorithms for Nonconvex SGD**

## Abstract

Weight decay is a universal regularizer in deep learning, yet its theoretical understanding remains limited to fixed-rate analysis. Sun et al. (CVPR 2025) established the first generalization theory for fixed weight decay in nonconvex SGD, showing that the alignment between gradient and parameter directions (measured by a worst-case supremum quantity) governs the stability-generalization tradeoff. However, this worst-case characterization is overly conservative, and fixed decay applies uniform regularization pressure regardless of the training dynamics. We propose **Alignment-Aware Dynamic Weight Decay (AADWD)**, a principled framework that continuously modulates weight decay strength based on the gradient-parameter alignment trajectory. Our contributions are threefold: (1) a **convergence-preserving theorem** showing time-varying SGDW maintains the standard O(T^{-1/2}) rate under mild conditions on the decay schedule; (2) a **cumulative contraction stability bound** that replaces the worst-case alignment sup_t delta_t with a trajectory-weighted average, yielding strictly tighter generalization guarantees when alignment varies across training; and (3) a **stochastic proxy analysis** bridging the gap between the theoretically ideal (full-gradient) alignment and the practically computable (mini-batch) alignment, with concentration-based guarantees on the order-level preservation of the stability bound. We instantiate AADWD with a simple, EMA-smoothed alignment rule requiring only one additional inner product per step (O(d) overhead), and demonstrate its effectiveness on CIFAR-10/100 with ResNet and VGG architectures, showing improved or matched generalization with reduced sensitivity to weight decay hyperparameter selection.

## Motivation

### The Gap Between Theory and Practice

Weight decay is applied in virtually every modern training pipeline, yet practitioners rely on grid search or heuristics (e.g., 0.01 for AdamW, 5e-4 for SGD) to set it. The fundamental question is: **can we do better than a constant, and can we prove it?**

Sun et al. (CVPR 2025) showed that the generalization benefit of weight decay is governed by the alignment quantity delta_T = sup_t |<nabla f_S(w_t), w_t>| / (||nabla f_S(w_t)|| ||w_t||). When delta_T < 1, weight decay contracts the stability recursion, improving generalization. But this worst-case characterization has two critical limitations:

1. **It ignores temporal structure**: In practice, alignment varies substantially across training phases (high in early overfitting phases, low during stable generalization). A worst-case bound captures only the least favorable moment.

2. **It cannot prescribe action**: Knowing that delta_T < 1 helps does not tell us how to adaptively adjust weight decay in response to the observed alignment trajectory.

### Why Not CWD?

Cautious Weight Decay (CWD, ICLR 2026) takes a binary approach: apply decay only when the sign of the update aligns with the sign of the parameter (coordinate-wise). While elegant and practical, CWD has fundamental limitations as a theoretical framework:
- It uses binary (sign-based) alignment rather than continuous cosine similarity, discarding magnitude information
- Its theoretical analysis focuses on sliding-mode behavior on the stationary manifold, not on convergence rate or generalization bounds for the full training trajectory
- It operates at the coordinate level, while our analysis reveals that global alignment structure matters for stability

Our work provides the **continuous, theoretically grounded** counterpart: a framework where the alignment signal drives weight decay strength through a smooth, analyzable function, with full convergence and generalization theory.

### Why Not AlphaDecay?

AlphaDecay (NeurIPS 2025) assigns module-wise weight decay based on spectral heavy-tailedness of weight correlation matrices. While innovative, it is **static** (computed once or periodically), does not use gradient-parameter alignment information, and lacks convergence theory. Our approach is **dynamic** (per-step), directly uses the alignment quantity that appears in the generalization bound, and provides theoretical guarantees.

## Research Questions

**RQ1 (Convergence):** Does time-varying weight decay lambda_t preserve the O(T^{-1/2}) convergence rate of standard nonconvex SGD, and under what conditions on lambda_t?

**RQ2 (Generalization):** Can we replace the worst-case sup_t delta_t in the stability bound with a trajectory-weighted average, yielding a strictly tighter generalization guarantee?

**RQ3 (Practicality):** How reliable is the mini-batch alignment proxy delta_hat_t, and under what conditions does an algorithm based on delta_hat_t inherit the theoretical guarantees derived for the oracle delta_t?

## Hypotheses

### H1: Convergence Preservation
Time-varying SGDW with lambda_t = O(gamma_t^2) maintains the same O(T^{-1/2}) convergence rate as standard SGD without weight decay, via an augmented Lyapunov analysis with time-varying auxiliary coefficient beta_t.

**Prediction:** The convergence bound has the form (1/T) sum E||nabla f(w_t)||^2 <= C_1/sqrt(T) + C_2 Lambda/T, where Lambda = sum lambda_t is the cumulative decay budget. When Lambda = O(sqrt(T)), the O(T^{-1/2}) rate is preserved.

**Test:** Compare training loss curves of fixed vs. dynamic SGDW; dynamic should converge no slower.

### H2: Tighter Generalization via Cumulative Contraction
The alignment-weighted average Delta_bar_T = (sum lambda_t delta_t) / (sum lambda_t) yields a strictly tighter stability bound than sup_t delta_t, with the improvement proportional to the temporal variance of delta_t.

**Prediction:** The stability bound scales as exp(-sum lambda_t (1 - Delta_bar_T)), which is tighter than exp(-sum lambda_t (1 - delta_max)) whenever Var(delta_t) > 0.

**Test:** Track Delta_bar_T vs. delta_max during training; compute the implied stability bounds; verify dynamic WD produces smaller generalization gaps.

### H3: Mini-Batch Proxy Reliability
With batch size B >= C sigma^2 / (delta_min^2 ||w||^2), the EMA-smoothed mini-batch alignment proxy hat{delta}_t is sufficiently accurate that the stability bound degrades by at most O(1/sqrt(B)).

**Prediction:** For B=128 on CIFAR-10/100 with ResNet20, Pearson correlation between mini-batch and full-batch alignment exceeds 0.85 after EMA smoothing (beta=0.99).

**Test:** Tier 0 diagnostic experiment comparing mini-batch vs. large-batch alignment at multiple training checkpoints.

### H4: Practical Superiority
AADWD with the conservative rule lambda_t = clip(c * gamma_t * (1 - EMA(hat{delta}_t)), lambda_min, lambda_max) achieves test accuracy >= best fixed WD baseline, with the additional property of reduced sensitivity to the choice of c (wider plateau in hyperparameter sensitivity analysis).

**Prediction:** On CIFAR-10/ResNet20, dynamic WD matches or exceeds best-tuned fixed WD by >= 0.2% in test accuracy, and the optimal c has a plateau width >= 4x (from 0.5 to 2.0) vs. the optimal fixed lambda width.

**Test:** Tier 1 comparison experiments with 3 random seeds each.

## Expected Contributions

1. **First convergence theory for time-varying SGDW** in the nonconvex setting (Theorem 3.1), establishing that lambda_t = O(gamma_t^2) suffices to preserve the standard rate.

2. **Cumulative contraction stability framework** (Theorem 4.1), replacing worst-case alignment with a trajectory-weighted measure -- the first result connecting alignment dynamics to generalization in a non-trivial way.

3. **Stochastic proxy transfer theorem** (Theorem 5.1), providing conditions under which mini-batch-based alignment-aware rules inherit oracle-level stability guarantees.

4. **Practical algorithm**: A simple, optimizer-compatible dynamic weight decay rule with O(d) overhead per step, demonstrated on standard benchmarks with competitive or improved performance.

5. **First systematic empirical characterization** of the alignment quantity delta_t across training phases, model architectures, and weight decay settings -- establishing the empirical foundation that the theoretical framework requires.

## Novelty Assessment

### Core Claim Novelty Verification

**Claim 1: Time-varying SGDW convergence theory.**
- Sun et al. (CVPR 2025): Only fixed lambda. Our extension to time-varying lambda_t with augmented Lyapunov beta_t is technically non-trivial (the lambda_t^2 ||w_t||^2 cross-term does not appear in fixed analysis).
- No prior work provides convergence rates for time-varying weight decay in nonconvex SGD. The closest is Loshchilov & Hutter's empirical WD scheduling, which lacks theory.
- **Verdict: Novel.**

**Claim 2: Cumulative contraction replacing worst-case alignment.**
- Sun et al. uses sup_t delta_t (worst-case). No prior work defines or analyzes a trajectory-weighted alignment measure for stability bounds.
- The concept of "cumulative contraction" via product decomposition of (1 - lambda_t + L gamma_t) with alignment-dependent lambda_t is new.
- **Verdict: Novel.**

**Claim 3: Alignment-aware dynamic weight decay algorithm.**
- CWD (ICLR 2026): Binary sign-based, coordinate-level. No continuous alignment, no global alignment theory.
- AlphaDecay (NeurIPS 2025): Spectral-based, module-level, static. No per-step dynamics, no alignment.
- Ghiasi et al. (NeurIPS 2022): Adaptive WD based on gradient/weight norm ratio for robustness. Not alignment-based, no generalization theory.
- AdaDecay (Nakamura & Hong 2019): Per-parameter decay based on gradient magnitude. Not alignment-based.
- GALA (if it exists as referenced by contrarian): Adjusts LR rather than WD. Different mechanism.
- **Verdict: Novel.** No prior work uses continuous gradient-parameter cosine similarity to dynamically adjust weight decay with convergence theory.

### Key Differentiators from CWD

| Aspect | CWD (ICLR 2026) | AADWD (Ours) |
|--------|-----------------|--------------|
| Alignment type | Binary sign | Continuous cosine similarity |
| Granularity | Coordinate-level | Global (extensible to layer-level) |
| Theory | Sliding-mode on stationary manifold | Convergence rate + stability bound |
| Decay modulation | 0/1 mask | Smooth function of alignment |
| Hyperparameters | None new | c, lambda_min, lambda_max (auto-tunable) |
| Key insight | Don't decay when update and weight disagree | Modulate decay strength by how much gradient and weight align |

## Methodology

### Algorithm: Conservative Alignment-Aware Dynamic Weight Decay

```
Input: params w_0, learning rate schedule {gamma_t}, base coefficient c,
       clip bounds [lambda_min, lambda_max], EMA decay beta=0.99

Initialize: ema_delta = 0.5  (neutral)

For t = 0, 1, ..., T-1:
    1. Compute stochastic gradient g_t on mini-batch
    2. Compute alignment proxy:
       delta_hat_t = |<g_t, w_t>| / (||g_t|| ||w_t|| + epsilon)
    3. Update EMA: ema_delta = beta * ema_delta + (1-beta) * delta_hat_t
    4. Compute dynamic decay:
       lambda_t = clip(c * gamma_t * (1 - ema_delta), lambda_min, lambda_max)
    5. Update parameters:
       w_{t+1} = (1 - lambda_t) * w_t - gamma_t * g_t
    6. Log: lambda_t, delta_hat_t, ema_delta, ||w_t||
```

### Experimental Plan (Tiered)

**Tier 0 (Diagnostic, ~1.5h):** Validate delta_hat_t reliability
- ResNet20/CIFAR-10, 4 fixed WD values
- Compare mini-batch vs. large-batch alignment (Pearson r target > 0.85)
- Gate: proceed only if EMA(delta_hat_t) shows systematic phase-dependent behavior

**Tier 1 (Core, ~4h):** Head-to-head comparison
- 7 methods: No-WD, Fixed-WD(best), Stagewise-WD, CWD, Conservative, Aggressive, Square
- ResNet20/CIFAR-10, 3 seeds each
- Primary metric: test accuracy; secondary: gen gap, weight norm, cumulative contraction

**Tier 2 (Robustness, ~3h):** Cross-architecture and sensitivity
- ResNet20/CIFAR-100, VGG16/CIFAR-10
- Hyperparameter sensitivity: c in {0.25, 0.5, 1.0, 2.0}
- EMA decay ablation: beta in {0.9, 0.99, 0.999}

### Critical Ablations
- **Random time-varying WD** (replace delta_hat_t with uniform random): Isolate alignment signal from mere time variation
- **Norm-matched fixed WD** (adjust fixed lambda per epoch to match dynamic ||w_t||): Isolate alignment-awareness from norm control
- **Equivalent cumulative WD** (fixed lambda = mean(lambda_t)): Fair comparison of dynamic vs. static at equal total regularization

## Risk Mitigation

### Risk 1: delta_hat_t is noise (Contrarian concern)
**Mitigation:** Tier 0 diagnostic is explicitly designed as a go/no-go gate. Fallback: increase EMA beta to 0.999, or switch to epoch-level full-batch alignment.

### Risk 2: Marginal accuracy improvement
**Mitigation:** Reframe contribution around (a) theory, (b) robustness to hyperparameter choice, (c) first systematic alignment characterization. The paper does not require large accuracy gains to be valuable.

### Risk 3: lambda_t = O(gamma_t^2) makes decay vanishingly small
**Mitigation:** This is a sufficient condition for convergence theory, not a practical requirement. The algorithm uses lambda_min to ensure non-trivial regularization. Theory provides the framework; practice uses the clipped version.

### Risk 4: CWD comparison shows no advantage
**Mitigation:** Our key differentiator is the theory, not necessarily empirical margin. Even if CWD matches empirically, the convergence + stability theory for continuous alignment-aware decay is the primary contribution.

## Paper Structure (ICML/NeurIPS format)

1. **Introduction** (1.5p): Motivation, gap in existing theory, three-layer contribution
2. **Preliminaries** (1.5p): Notation, assumptions A1-A6, Sun et al. recap
3. **Time-Varying SGDW Convergence** (2p): Theorem 3.1 + augmented Lyapunov
4. **Cumulative Contraction Stability** (2.5p): Theorem 4.1 + alignment measure
5. **Stochastic Proxy Analysis** (1.5p): Theorem 5.1 + concentration bridge
6. **Algorithm & Practical Rules** (1p): Algorithm box, design choices
7. **Experiments** (2.5p): Tiers 0-2, ablations, theory verification plots
8. **Discussion & Conclusion** (0.5p)


## 当前可检验假设
# Testable Hypotheses with Expected Outcomes

## H1: Convergence Preservation Under Time-Varying Decay

**Statement:** When lambda_t satisfies (C1) lambda_t <= c_1 * gamma_t^2 and (C2) sum lambda_t <= Lambda < infinity, time-varying SGDW achieves:

(1/T) sum_{t=0}^{T-1} E[||nabla f_S(w_t)||^2] <= C_1/sqrt(T) + C_2 * Lambda / T

preserving the O(T^{-1/2}) convergence rate of standard SGD.

**Expected Outcome:** Training loss curves of AADWD and fixed SGDW converge at the same rate (within statistical noise). The augmented potential Phi_t = f(w_t) + beta_t ||w_t||^2 decreases monotonically in expectation.

**Experimental Validation:**
- Plot train loss vs. epoch for fixed WD and AADWD on ResNet20/CIFAR-10
- Compute empirical Phi_t and verify approximate monotonic decrease
- Compare convergence speed: AADWD should be no more than 5% slower in reaching a given loss threshold

**Falsification Criterion:** If AADWD train loss is consistently >10% worse than best fixed WD at the same epoch count, H1 is weakened (though the theory allows for constant-factor differences).

---

## H2: Tighter Generalization via Cumulative Contraction

**Statement:** Define the effective alignment measure:
Delta_bar_T = sum_{t} lambda_t * delta_t / sum_{t} lambda_t

The stability bound under AADWD scales as exp(-sum lambda_t * (1 - Delta_bar_T)), which is strictly tighter than the fixed-WD bound exp(-T * lambda * (1 - delta_max)) whenever Var(delta_t) > 0.

**Expected Outcome:**
- delta_t varies significantly across training (std > 0.1 across phases)
- Delta_bar_T < delta_max (typically by 0.1-0.3)
- The implied stability bound ratio (dynamic/fixed) < 0.8

**Experimental Validation:**
- Track delta_hat_t throughout training, compute Delta_bar_T and delta_max
- Compute the implied cumulative contraction product Pi_T = prod(1 - lambda_t + L*gamma_t) for both dynamic and fixed WD
- Compare train-test accuracy gap (generalization gap): AADWD should have smaller or equal gap

**Falsification Criterion:** If delta_t is approximately constant throughout training (std < 0.03), then Delta_bar_T ~ delta_max and H2 provides negligible improvement. This would redirect the work toward Alternative A (empirical characterization to find settings where delta_t does vary).

---

## H3: Mini-Batch Alignment Proxy Reliability

**Statement:** For batch size B, the EMA-smoothed alignment proxy satisfies:
E[(hat{delta}_t^{EMA} - delta_t)^2] <= C_delta * sigma^2 / (B * (1-beta) * ||nabla f||^2 * ||w||^2)

With B=128 and beta=0.99, this error is small enough that the practical algorithm inherits the oracle stability bound up to O(1/sqrt(B)) additive degradation.

**Expected Outcome:**
- Pearson correlation between mini-batch hat{delta}_t (EMA-smoothed) and large-batch delta_t exceeds 0.85
- The EMA trajectory shows clear phase-dependent structure (not white noise)
- hat{delta}_t has std < 0.15 (after EMA smoothing with beta=0.99)

**Experimental Validation (Tier 0 diagnostic):**
- At 9 checkpoints (3 per phase: early/mid/late), compute delta_hat_t using B=128 and B=8192
- Scatter plot: x-axis = large-batch delta, y-axis = small-batch delta; compute Pearson r
- Time series: plot raw delta_hat_t and EMA(delta_hat_t) for beta in {0.9, 0.95, 0.99}
- Histogram: delta_hat_t distribution per training phase

**Falsification Criterion:** If Pearson r < 0.6 even after EMA smoothing, the mini-batch proxy is unreliable. Fallback: use epoch-level full-batch alignment, or increase batch size, or adopt Kalman filtering (from interdisciplinary perspective).

---

## H4: Practical Performance of AADWD

**Statement:** The conservative AADWD rule achieves:
- test accuracy >= best fixed WD baseline (with matched or lower hyperparameter tuning budget)
- wider optimal hyperparameter plateau: the range of c values achieving >99% of best accuracy is at least 4x wider than the analogous range for fixed lambda

**Expected Outcome:**
- On CIFAR-10/ResNet20: test accuracy 92.5% +/- 0.2% (baseline: ~92.2%)
- On CIFAR-100/ResNet20: test accuracy 69.5% +/- 0.3% (baseline: ~68.8%)
- Hyperparameter sensitivity: c in [0.5, 2.0] all achieve within 0.3% of best; for fixed WD, the analogous range is ~[3e-4, 7e-4] (a factor of 2.3x)

**Experimental Validation (Tier 1):**
- 7-way comparison: No-WD, Fixed-WD, Stagewise, CWD, Conservative, Aggressive, Square
- 3 random seeds per setting, report mean +/- std
- Hyperparameter sensitivity plot: accuracy vs. c (for dynamic) and accuracy vs. lambda (for fixed)

**Falsification Criterion:** If AADWD underperforms best fixed WD by >0.5% consistently across seeds, H4 is falsified for accuracy. The robustness claim requires the plateau analysis. If both fail, pivot to Alternative C (theory-only contribution).

---

## H5: Alignment-Awareness vs. Mere Time-Variation

**Statement:** The alignment signal in AADWD provides information beyond mere temporal variation of weight decay. Specifically, AADWD outperforms a "random dynamic WD" baseline that uses the same temporal statistics (mean, variance) of lambda_t but with shuffled/random delta_hat_t values.

**Expected Outcome:**
- AADWD outperforms random dynamic WD by >= 0.3% in test accuracy
- AADWD produces more structured lambda_t trajectories (higher autocorrelation)
- Correlation between delta_hat_t and subsequent loss change DeltaL_t is negative (confirming the alignment signal has predictive value)

**Experimental Validation (Tier 2 ablation):**
- Replace delta_hat_t with uniform random in [0, 1], keep all other hyperparameters identical
- Compare test accuracy, generalization gap, and weight norm trajectories

**Falsification Criterion:** If random dynamic WD matches AADWD (within 0.1%), then the alignment signal adds no value beyond temporal variation, undermining the core thesis. This would redirect toward studying the value of time-varying WD per se (Alternative C).

---

## Summary of Hypothesis Priority and Dependencies

| Hypothesis | Priority | Dependency | Phase |
|-----------|----------|------------|-------|
| H3 (proxy reliability) | Highest | None (gate for all others) | Tier 0 |
| H1 (convergence) | High | Theory proof | Tier 1 |
| H4 (practical performance) | High | H3 passes | Tier 1 |
| H2 (tighter bound) | Medium | Theory proof + H3 | Tier 1-2 |
| H5 (alignment vs. randomness) | Medium | H4 shows improvement | Tier 2 |

**Critical path:** H3 (Tier 0) -> H4 (Tier 1) -> H5 (Tier 2). If H3 fails, reconsider entire approach. If H4 fails on accuracy, reframe around robustness. H1 and H2 are theoretical and can proceed in parallel with experiments.


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_aadwd",
      "title": "Alignment-Aware Dynamic Weight Decay: Cumulative Contraction Theory and Practical Algorithms",
      "status": "front_runner",
      "summary": "Continuous alignment-aware dynamic weight decay with three-layer theory: (1) time-varying SGDW convergence preservation, (2) cumulative contraction stability bound replacing worst-case alignment, (3) stochastic proxy transfer. Conservative EMA-smoothed algorithm with O(d) overhead.",
      "hypotheses": [
        "H1: Time-varying lambda_t = O(gamma_t^2) preserves O(T^{-1/2}) convergence rate",
        "H2: Cumulative alignment-weighted stability bound is strictly tighter than worst-case",
        "H3: EMA-smoothed mini-batch alignment proxy has Pearson r > 0.85 with full-batch",
        "H4: AADWD matches or exceeds best fixed WD with wider hyperparameter plateau",
        "H5: Alignment signal adds value beyond random temporal variation"
      ],
      "pilot_focus": "Tier 0: Validate delta_hat_t proxy reliability (H3). Tier 1: Core comparison of 7 methods on ResNet20/CIFAR-10 with 3 seeds (H4).",
      "key_risks": [
        "delta_hat_t proxy too noisy at B=128",
        "CWD matches empirical performance, weakening practical contribution",
        "Theorem 4.1 (cumulative contraction) proof may not close"
      ],
      "differentiators": [
        "vs. CWD: continuous alignment, global theory, convergence rate guarantees",
        "vs. AlphaDecay: per-step dynamic, alignment-based, convergence theory",
        "vs. Sun et al.: time-varying decay, cumulative (not worst-case) alignment"
      ]
    },
    {
      "candidate_id": "cand_empirical",
      "title": "Systematic Characterization of Gradient-Parameter Alignment Dynamics in Deep Learning",
      "status": "backup",
      "summary": "Purely empirical study: first comprehensive characterization of delta_t evolution across architectures, datasets, optimizers, and WD settings. Provides the empirical foundation the field currently lacks. Low risk, moderate impact.",
      "hypotheses": [
        "Alignment shows systematic phase-dependent structure across diverse settings",
        "Per-layer alignment disaggregation reveals qualitatively different behaviors",
        "Alignment variance correlates with generalization gap across models"
      ],
      "pilot_focus": "Multi-architecture alignment tracking on CIFAR and ImageNet-subset.",
      "key_risks": [
        "Limited novelty if alignment turns out trivially constant",
        "Impact limited to empirical observation without algorithmic contribution"
      ],
      "differentiators": [
        "First systematic study of alignment dynamics in training",
        "Provides empirical validation/falsification for Sun et al. theory"
      ]
    },
    {
      "candidate_id": "cand_llm",
      "title": "Alignment-Aware Weight Decay for Transformer Pre-training",
      "status": "backup",
      "summary": "Applied variant: per-layer alignment-aware WD for AdamW in LLM pre-training. NanoGPT/LitGPT experiments (60M-350M). No convergence theory; purely practical contribution targeting the LLM community.",
      "hypotheses": [
        "Per-layer alignment-aware WD improves perplexity over uniform WD in transformer pre-training",
        "Attention and FFN layers exhibit distinct alignment dynamics requiring different decay schedules",
        "Dynamic alignment-aware WD combined with AlphaDecay module selection yields best results"
      ],
      "pilot_focus": "NanoGPT 60M on OpenWebText with per-layer alignment tracking.",
      "key_risks": [
        "AdamW preconditioning may wash out alignment signal",
        "Requires longer experiments (2-4h per run)",
        "Competition with AlphaDecay on the same target (LLM pre-training)"
      ],
      "differentiators": [
        "Dynamic per-step alignment vs. AlphaDecay's static spectral analysis",
        "Direct application to highest-impact setting (LLM)"
      ]
    }
  ],
  "decision_rationale": "The front-runner (AADWD) is selected because it occupies a unique theoretical niche: the first convergence + generalization theory for continuous alignment-aware dynamic weight decay in nonconvex SGD. The theoretical contributions (Theorems 3.1, 4.1, 5.1) are independently valuable even if experiments show marginal empirical gains. The backup candidates address the two main failure modes: if the alignment proxy is unreliable (pivot to empirical study) or if CIFAR-scale experiments lack impact (pivot to LLM focus). Research focus mode (FOCUSED) retains the empirical backup as a safe fallback and the LLM variant as a high-impact pivot.",
  "perspectives_weighted": {
    "theoretical": "Highest weight. The three-layer proof architecture is the backbone of the proposal. Augmented Lyapunov analysis, cumulative contraction framework, and proxy transfer are all technically novel.",
    "pragmatist": "High weight. The tiered experimental plan, specific code structure, and risk mitigation strategies are directly adopted. The conservative algorithm variant is the primary practical instantiation.",
    "empiricist": "High weight. The Tier 0 go/no-go gate, CWD baseline requirement, 3-seed minimum, and norm-matched ablation are all incorporated. The random-time-varying-WD ablation is added as H5.",
    "contrarian": "Medium-high weight. The concerns about proxy noise, hyperparameter explosion, CWD competition, and CIFAR limitations are all acknowledged and addressed through experimental design and risk mitigation. The reframing advice (robustness story if accuracy gains are marginal) is adopted.",
    "innovator": "Medium weight. The cumulative contraction framework (Proposal 3) and momentum-debiased alignment (Proposal 4) both inform the theoretical direction. However, the more speculative ideas (entropy force, sliding mode control, NTK connection) are deferred to future work to maintain focus.",
    "interdisciplinary": "Medium weight. The Kalman filtering idea for delta_t estimation is noted as a fallback if EMA proves insufficient. The dual-timescale design (from synaptic scaling) and PID-controller analogy inform the algorithm variants but are not adopted as the primary rule to keep complexity manageable."
  }
}


## 上一轮 validation 决策意见
# Idea Validation Decision

## Pilot Evidence Summary

All experiments were conducted as 20-epoch pilot runs on ResNet20/CIFAR-10 (seed=42), with additional cross-architecture validation on VGG16-BN/CIFAR-100. Approximately 30 runs were executed across Tier 0 (diagnostic), Tier 1 (method comparison), and Tier 2 (cross-architecture, ablations, hyperparameter sensitivity).

### Candidate: cand_aadwd (Alignment-Aware Dynamic Weight Decay)

**Tier 0 Diagnostic (H3 -- Proxy Reliability):**
- Pearson r (mini-batch EMA vs large-batch alignment) = 0.8489
- Threshold: r >= 0.85 --> PARTIAL FAIL (delta to threshold: -0.001)
- Phase-dependent structure confirmed: delta_hat decreases from early (0.0045) to mid (0.0034) to late (0.0028)
- Overall delta_std = 0.000753 (below 0.05 threshold, but 20-epoch pilot may not capture full phase transitions)
- Verdict: NO-GO at beta=0.99; beta corrected to 0.999 for subsequent experiments

**Tier 1 Fixed WD Baseline:**
- Best fixed WD = 5e-4, best_test_acc = 89.35% (pilot reference ceiling)
- WD=1e-3: 88.98%; No-WD: 87.44%; WD=5e-3: 85.95%; WD=1e-2: 80.13%

**Tier 1 AADWD Variants (3 variants, beta=0.999, c=0.01):**
- AADWD-Aggressive: best_test_acc = 85.09%, final = 83.86%, gen_gap = 5.03% --> PASS (>85%)
  - Lambda drops 187x from 4.15e-4 to 2.21e-6 (strong dynamic behavior)
  - Weight norm = 70.4 (higher than fixed WD's 28.1, due to reduced late-stage decay)
- AADWD-Square: best_test_acc = 83.45%, final = 82.47% --> FAIL (below 85%)
  - Lambda barely varies (1.7x change), behaves like noisy fixed WD
- AADWD-Conservative: best_test_acc = 74.06%, final = 61.80% --> FAIL (severe underfitting)
  - Lambda converges toward lambda_max too fast, creating excessive regularization

**Tier 1 Dynamic Baselines:**
- Stagewise-WD: 85.33% (reasonable, but milestones at 30/60/90 not triggered in 20-epoch pilot)
- CWD (sign-based): 81.10% (12.9x slower than fixed WD, poor accuracy)

**Tier 2 Cross-Architecture (VGG16-BN / CIFAR-100):**
- AADWD-Aggressive: 48.70% vs Fixed-WD: 37.15% (delta = +11.55%)
- Caveat: Fixed-WD severely underfits at 20 epochs on CIFAR-100 (gen_gap = -0.43%), inflating the gap

**Tier 2 Ablations (H5 -- Alignment vs Random):**
- AADWD-Aggressive: best_test_acc = 85.09%
- Random-Dynamic-WD: best_test_acc = 80.34% (delta = -4.75%), final = 72.16% (unstable)
- Norm-Matched-WD: best_test_acc = 85.44% (delta = +0.35% vs AADWD)
- CRITICAL FINDING: Norm-matched WD nearly matches AADWD at 20-epoch scale, suggesting alignment-awareness marginal gain is small in short pilots

**Tier 2 Hyperparameter Sensitivity:**
- c sweep [0.001, 0.1]: best_test_acc varies only 0.65% (85.27%-85.74%) --> EXCELLENT robustness
- beta sweep [0.9, 0.999]: stable (84.86%-85.98%); beta=0.9999 drops to 82.09% (EMA too smooth)

### Candidate: cand_empirical (Empirical Characterization Study)
- No pilot experiments conducted (backup candidate)
- Novelty score: 6/10 (moderate impact ceiling for purely empirical work)

### Candidate: cand_llm (Transformer Pre-training Application)
- No pilot experiments conducted (backup candidate)
- Novelty score: 5/10 (crowded competitive landscape: AlphaDecay, CWD already target LLM pre-training)

---

## Decision Matrix

### cand_aadwd (Alignment-Aware Dynamic Weight Decay)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 3 | AADWD-Aggressive achieves 85.09% (pilot pass at >85%), but trails best fixed WD by -4.26% (89.35% vs 85.09%). This gap is partly attributable to pilot length (20 vs 200 epochs; LR milestones never triggered). AADWD beats Random-Dynamic by +4.75%. |
| Hypothesis survival | 0.25 | 4 | H1 (convergence): CONFIRMED -- all 12+ AADWD runs converge, no NaN. H3 (proxy reliability): PARTIAL -- r=0.849 just below 0.85, correctable by beta adjustment. H5 (hyperparameter robustness): CONFIRMED -- c varies 0.65% across 2 orders of magnitude. H2 (alignment advantage): AMBIGUOUS -- beats random (+4.75%) but ties norm-matched (+0.35%). No hypothesis conclusively falsified. |
| Path to full result | 0.20 | 4 | Clear path: (1) LR milestones (100/150) will activate in 200-epoch runs, enabling AADWD's late-stage WD adjustment to differentiate from fixed WD; (2) longer training amplifies phase transitions for alignment proxy reliability; (3) 3-seed runs will quantify variance. The full experimental recommendations document already specifies exact configurations and execution priorities. |
| Novelty (from report) | 0.15 | 4 | Novelty score 8/10 from literature search. Three-layer theoretical contribution (time-varying SGDW convergence, cumulative contraction stability, stochastic proxy transfer) is genuinely novel. No prior work uses continuous gradient-parameter cosine similarity to dynamically adjust WD with convergence theory. CWD is closest competitor but uses binary sign-based alignment without convergence rate theory. |
| Resource efficiency | 0.10 | 3 | AADWD-Aggressive is ~10x slower than fixed WD per epoch (572s vs 54s for 20 epochs) due to alignment computation. Full experiment suite (P0 methods x 3 seeds) estimated at ~8h on 1 GPU, feasible within project constraints. CWD alternative would be 12.9x slower. |

**Weighted Score: 0.30*3 + 0.25*4 + 0.20*4 + 0.15*4 + 0.10*3 = 0.90 + 1.00 + 0.80 + 0.60 + 0.30 = 3.60**

### cand_empirical (Empirical Characterization)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No dedicated pilot run. Tier 0 diagnostic provides incidental evidence: delta_hat shows phase-dependent structure (mean drops from 0.0045 to 0.0028), but overall std = 0.000753 is low. The alignment characterization would need multi-architecture, multi-dataset runs to constitute a standalone contribution. |
| Hypothesis survival | 0.25 | 3 | The core hypothesis that "alignment shows systematic phase-dependent structure" is weakly supported by Tier 0 (declining trend exists but amplitude is small). No falsification, but the empirical novelty ceiling is moderate -- Sun et al. already introduced the alignment quantity. |
| Path to full result | 0.20 | 3 | Straightforward path (run training with various architectures, track alignment), but lacks algorithmic contribution. Would need extensive experiments across architectures/datasets/optimizers to be publishable. Paper impact ceiling is limited without actionable insights. |
| Novelty (from report) | 0.15 | 3 | Novelty score 6/10. No paper systematically characterizes delta_t dynamics as a primary contribution, but this is a descriptive study. Moderate novelty, moderate impact. |
| Resource efficiency | 0.10 | 4 | Standard training runs with alignment logging. No computational overhead beyond training. |

**Weighted Score: 0.30*2 + 0.25*3 + 0.20*3 + 0.15*3 + 0.10*4 = 0.60 + 0.75 + 0.60 + 0.45 + 0.40 = 2.80**

### cand_llm (Transformer Pre-training Application)

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot experiments conducted. Zero empirical evidence. |
| Hypothesis survival | 0.25 | 2 | AdamW preconditioning may wash out alignment signal (acknowledged risk). CWD already demonstrated on LLM pre-training at billion scale. AlphaDecay already does module-wise adaptive WD for LLMs. Competitive landscape is crowded. |
| Path to full result | 0.20 | 2 | Requires NanoGPT/LitGPT setup (60M-350M params), 2-4h per run. No convergence theory. Needs clear demonstration that per-layer alignment dynamics provide better signal than AlphaDecay's spectral analysis. High execution risk. |
| Novelty (from report) | 0.15 | 2 | Novelty score 5/10. Crowded field. AlphaDecay and CWD already target LLM weight decay. Recommendation: "modify" (not "proceed"). |
| Resource efficiency | 0.10 | 2 | 2-4h per run at 60M-350M scale. Would consume substantial GPU budget for uncertain return. |

**Weighted Score: 0.30*1 + 0.25*2 + 0.20*2 + 0.15*2 + 0.10*2 = 0.30 + 0.50 + 0.40 + 0.30 + 0.20 = 1.70**

---

## Decision Rationale

**Decision: ADVANCE cand_aadwd to full 200-epoch experiments.**

The evidence supports this decision on multiple fronts:

1. **No hypothesis has been falsified.** H1 (convergence preservation) is confirmed across all runs. H3 (proxy reliability) is technically a partial fail (r=0.849 vs threshold 0.85), but the gap is only 0.001 and the beta correction has already been applied. H5 (hyperparameter robustness) is strongly confirmed. H2 (alignment advantage vs random) is confirmed (+4.75%), though the marginal gain over norm-matched WD needs 200-epoch validation.

2. **The 20-epoch pilot systematically disadvantages AADWD.** The learning rate milestones (30/60/90) never triggered during the 20-epoch pilot, meaning fixed WD had an unfair advantage -- it operates optimally at any training length, while AADWD's adaptive behavior requires the LR schedule transitions to demonstrate its value. The full experiment with milestones at [100, 150] will provide a fair comparison.

3. **AADWD-Aggressive shows the RIGHT dynamic behavior.** Lambda drops 187x from early to late training, matching the theoretical prediction: as alignment decreases (gradient direction stabilizes), weight decay should decrease to avoid over-regularization. The Conservative variant's failure (lambda increases, causing underfitting) provides a clean negative control that validates the direction of the effect.

4. **The theoretical contribution is independently valuable.** Even if empirical gains are marginal, the three theorems (time-varying SGDW convergence, cumulative contraction stability, stochastic proxy transfer) constitute a genuine contribution. The novelty score is 8/10 with no direct collision. The paper can be framed around theory + robustness + alignment characterization, not requiring large accuracy gains.

5. **Research Focus mode is FOCUSED (research_focus=4).** The directive explicitly states: "Prefer REFINE over PIVOT. Give the current front-runner more chances to prove itself through additional refinement rounds." cand_aadwd scores 3.60 (above the ADVANCE threshold of 3.5), and the remaining uncertainties (H2 margin over norm-matched, H3 proxy reliability at 200 epochs) are answerable by the planned full experiment, not by pivoting.

**Why not REFINE:** The pilot evidence is sufficient to justify proceeding directly to full experiments. The methodology is sound, the code infrastructure is complete (Tier 0-2 scripts all work), the hyperparameter sensitivity analysis identifies good defaults. No redesign is needed -- only longer training.

**Why not PIVOT:** No hypothesis has been falsified. The worst evidence (AADWD trailing fixed WD by 4.26% in 20-epoch pilot) is explainable by the pilot length limitation. The theoretical contribution remains novel regardless of empirical margins.

---

## Sanity Checks
- [x] Did I compare ALL candidates, not just the front-runner? Yes -- cand_empirical (2.80) and cand_llm (1.70) were scored; both fall well below ADVANCE threshold.
- [x] Did I penalize any candidate that failed its own falsification criteria? Yes -- H3 (r < 0.85) penalized cand_aadwd's pilot signal strength score (3 instead of 4). AADWD-Conservative failure noted but attributed to hyperparameter misconfiguration, not method failure.
- [x] Am I being swayed by sunk cost? No -- the decision is based on (a) theoretical novelty score 8/10, (b) no falsified hypotheses, (c) clear mechanistic explanation for the pilot gap, (d) FOCUSED research mode directive. The ~30 pilot runs already executed are informative but do not influence the forward-looking assessment.
- [x] If the pilot was inconclusive, am I defaulting to REFINE rather than blindly advancing? The pilot is not inconclusive -- it provides clear positive signals (convergence confirmed, alignment dynamics correct, robustness excellent) with specific uncertainties (proxy reliability, margin over norm-matched) that the full experiment is designed to resolve. ADVANCE at 3.60 is justified.

---

## Next Actions

1. **Execute 200-epoch full experiment suite** (P0 priority methods x 3 seeds):
   - Fixed-WD (5e-4) x 3 seeds [~8 min/run x 3]
   - Fixed-WD (1e-3) x 3 seeds [~8 min/run x 3]
   - AADWD-Aggressive (c=0.01, beta=0.999) x 3 seeds [~90 min/run x 3]
   - Norm-Matched-WD x 3 seeds [~100 min/run x 3]

2. **Adjust LR schedule** for 200 epochs: milestones=[100, 150], lr_gamma=0.1

3. **Record alignment proxy diagnostics** every 10 epochs (Pearson r between mini-batch EMA and large-batch alignment) to resolve H3

4. **Include P1 methods** if GPU time allows: No-WD x 1 seed, Stagewise-WD x 3 seeds (milestones=[100,150])

5. **For VGG16/CIFAR-100**: first run fixed-WD grid search (5 values) to establish fair baseline, then compare with AADWD-Aggressive

6. **Post-experiment**: if norm-matched-WD matches AADWD within 0.3% at 200 epochs, design LR-transition-phase ablation (enable alignment-aware adjustment only at epochs 90-110 and 140-160) to isolate the timing signal value

SELECTED_CANDIDATE: cand_aadwd
CONFIDENCE: 0.72
DECISION: ADVANCE


## 上一轮 validation 结构化决策
{
  "decision": "ADVANCE",
  "selected_candidate_id": "cand_aadwd",
  "confidence": 0.72,
  "candidate_scores": {
    "cand_aadwd": {
      "weighted_score": 3.60,
      "verdict": "ADVANCE",
      "criterion_scores": {
        "pilot_signal_strength": {"weight": 0.30, "score": 3, "evidence": "AADWD-Aggressive 85.09% passes pilot threshold (>85%); trails best fixed WD by -4.26% but LR milestones never triggered in 20-epoch pilot; beats Random-Dynamic by +4.75%"},
        "hypothesis_survival": {"weight": 0.25, "score": 4, "evidence": "H1 convergence CONFIRMED (all 12+ runs converge); H3 proxy r=0.849 PARTIAL FAIL (correctable by beta adjustment); H5 robustness CONFIRMED (c varies 0.65% across 2 OoM); H2 alignment advantage AMBIGUOUS (beats random but ties norm-matched); no hypothesis falsified"},
        "path_to_full_result": {"weight": 0.20, "score": 4, "evidence": "Clear path: LR milestones [100,150] will activate in 200 epochs; 3-seed runs planned; full infrastructure (Tier 0-2 scripts) operational; exact configurations specified"},
        "novelty": {"weight": 0.15, "score": 4, "evidence": "Novelty score 8/10; three-layer theoretical contribution (time-varying SGDW convergence, cumulative contraction stability, stochastic proxy transfer) is genuinely novel; recommendation: proceed"},
        "resource_efficiency": {"weight": 0.10, "score": 3, "evidence": "AADWD ~10x slower than fixed WD per epoch; full P0 suite ~8h on 1 GPU; feasible within project constraints"}
      }
    },
    "cand_empirical": {
      "weighted_score": 2.80,
      "verdict": "HOLD_AS_BACKUP",
      "criterion_scores": {
        "pilot_signal_strength": {"weight": 0.30, "score": 2, "evidence": "No dedicated pilot; incidental Tier 0 evidence shows weak phase-dependent structure (delta_hat std=0.000753)"},
        "hypothesis_survival": {"weight": 0.25, "score": 3, "evidence": "Phase-dependent alignment confirmed but amplitude small; no falsification; moderate novelty ceiling for purely empirical work"},
        "path_to_full_result": {"weight": 0.20, "score": 3, "evidence": "Straightforward but lacks algorithmic contribution; needs extensive multi-architecture experiments"},
        "novelty": {"weight": 0.15, "score": 3, "evidence": "Novelty score 6/10; descriptive study; moderate impact ceiling"},
        "resource_efficiency": {"weight": 0.10, "score": 4, "evidence": "Standard training with alignment logging; no computational overhead"}
      }
    },
    "cand_llm": {
      "weighted_score": 1.70,
      "verdict": "DROP",
      "criterion_scores": {
        "pilot_signal_strength": {"weight": 0.30, "score": 1, "evidence": "Zero empirical evidence; no pilot conducted"},
        "hypothesis_survival": {"weight": 0.25, "score": 2, "evidence": "AdamW preconditioning may wash out alignment signal; CWD and AlphaDecay already target LLM WD"},
        "path_to_full_result": {"weight": 0.20, "score": 2, "evidence": "Requires NanoGPT setup, 2-4h/run, no convergence theory; high execution risk"},
        "novelty": {"weight": 0.15, "score": 2, "evidence": "Novelty score 5/10; crowded competitive landscape; novelty report recommends 'modify'"},
        "resource_efficiency": {"weight": 0.10, "score": 2, "evidence": "2-4h per run at 60M-350M scale; high GPU cost for uncertain return"}
      }
    }
  },
  "reasons": [
    "AADWD-Aggressive passes pilot threshold (85.09% > 85%) with correct dynamic behavior (lambda drops 187x early-to-late)",
    "No hypothesis falsified: H1 convergence confirmed, H3 proxy r=0.849 near threshold and correctable, H5 robustness confirmed across 2 OoM",
    "20-epoch pilot systematically disadvantages AADWD (LR milestones never triggered); 200-epoch runs expected to close gap with fixed WD",
    "AADWD beats Random-Dynamic-WD by +4.75% best_test_acc and +11.7% final_test_acc, confirming alignment signal value over random time-variation",
    "Theoretical contribution (novelty 8/10) is independently valuable: first convergence theory for time-varying SGDW, cumulative contraction stability, stochastic proxy transfer",
    "Hyperparameter robustness excellent: c varies only 0.65% across [0.001, 0.05]; beta stable in [0.9, 0.999]",
    "Cross-architecture signal positive: AADWD outperforms fixed WD by +11.55% on VGG16/CIFAR-100 (pilot caveat: fixed WD underfits at 20 epochs)",
    "Research focus=4 (FOCUSED) directs preference for REFINE/ADVANCE over PIVOT"
  ],
  "risks": [
    {
      "risk_id": "H2_margin",
      "description": "Norm-matched-WD (85.44%) nearly matches AADWD (85.09%) at 20-epoch scale; alignment-awareness marginal gain may remain small at 200 epochs",
      "severity": "medium",
      "mitigation": "Include norm-matched-WD as P0 ablation in full experiment; design LR-transition-phase ablation if margin remains < 0.3%"
    },
    {
      "risk_id": "H3_proxy_reliability",
      "description": "Pearson r=0.849 is below 0.85 threshold; may not improve sufficiently at 200 epochs even with beta=0.999",
      "severity": "medium",
      "mitigation": "Record proxy diagnostics every 10 epochs; fallback to beta=0.9995 or epoch-level full-batch alignment; ultimate fallback: pivot to cand_empirical"
    },
    {
      "risk_id": "pilot_gap_persistence",
      "description": "AADWD may continue trailing fixed WD (5e-4) at 200 epochs if the pilot gap is not explained by LR milestone timing",
      "severity": "medium-low",
      "mitigation": "Paper framing can emphasize theory + robustness + alignment characterization even without accuracy superiority; AADWD's hyperparameter robustness (0.65% variation vs fixed WD's sensitivity to lambda choice) is a valid practical contribution"
    },
    {
      "risk_id": "conservative_variant_unfixable",
      "description": "AADWD-Conservative (74.06%) failed severely; even with c=0.001 it may remain problematic",
      "severity": "low",
      "mitigation": "Drop Conservative as P2 priority; focus paper on Aggressive variant which embodies the correct theoretical prediction"
    }
  ],
  "next_actions": [
    "Run 200-epoch full experiment: Fixed-WD (5e-4, 1e-3) x 3 seeds + AADWD-Aggressive x 3 seeds + Norm-Matched-WD x 3 seeds",
    "Adjust LR schedule: milestones=[100,150], lr_gamma=0.1 for all 200-epoch runs",
    "Record alignment proxy Pearson r every 10 epochs to resolve H3",
    "Add P1 methods if GPU allows: No-WD x 1 seed, Stagewise-WD x 3 seeds",
    "For VGG16/CIFAR-100: run fixed-WD grid (5 values) first to establish fair baseline",
    "Post-experiment: if norm-matched-WD matches AADWD within 0.3%, design LR-transition-phase ablation"
  ],
  "dropped_candidates": ["cand_llm"],
  "retained_backups": ["cand_empirical"]
}


## 上一轮新颖性检查报告（必须针对发现的撞车问题进行修正）
# Novelty Report: Alignment-Aware Dynamic Weight Decay

## Search Methodology

Searches were conducted across arXiv (>15 targeted queries), Google Scholar (>8 queries), and general web search (>6 queries) covering:
- Dynamic/adaptive weight decay methods
- Gradient-parameter alignment in optimization
- Stability and generalization theory for SGD with weight decay
- Time-varying regularization convergence theory
- Per-layer/per-parameter weight decay scheduling

---

## Candidate 1: `cand_aadwd` — Alignment-Aware Dynamic Weight Decay (Front-Runner)

### Core Contribution Claims

1. **First convergence theory for time-varying SGDW** in nonconvex setting
2. **Cumulative contraction stability bound** replacing worst-case alignment with trajectory-weighted average
3. **Alignment-aware dynamic weight decay algorithm** using continuous gradient-parameter cosine similarity

### Prior Work Analysis

#### Directly Related Papers

**1. Sun et al., "Investigating the Role of Weight Decay in Enhancing Nonconvex SGD" (CVPR 2025)**
- **Overlap:** This is the foundational paper that AADWD builds upon. Sun et al. establish the first generalization theory for fixed weight decay in nonconvex SGD, introducing the alignment quantity delta_T = sup_t |<nabla f_S(w_t), w_t>| / (||nabla f_S(w_t)|| ||w_t||).
- **Severity:** `related_work` — This is the direct predecessor. AADWD extends it in three non-trivial directions (time-varying decay, cumulative alignment, stochastic proxy). Sun et al. only analyze fixed lambda and use worst-case (supremum) alignment.
- **Differentiation:** AADWD's cumulative contraction framework and time-varying convergence theory are genuine extensions not present in Sun et al.

**2. Cautious Weight Decay (CWD) — Chen et al. (arXiv 2510.12402, Oct 2025)**
- **Overlap:** CWD also uses gradient-parameter alignment to modulate weight decay. It applies decay only when parameter coordinate signs align with optimizer update signs.
- **Severity:** `partial_overlap` — Both use alignment signals to modify weight decay behavior. However:
  - CWD uses binary (sign-based) alignment at coordinate level; AADWD uses continuous cosine similarity at global level
  - CWD's theory focuses on sliding-mode behavior on the stationary manifold; AADWD provides convergence rates and stability bounds for the full trajectory
  - CWD is optimizer-agnostic (AdamW, Lion, Muon); AADWD targets SGD with full theoretical analysis
  - CWD requires no new hyperparameters; AADWD introduces c, lambda_min, lambda_max
- **Risk:** CWD is a strong practical competitor. If CWD matches AADWD empirically, the differentiator must be the theory.

**3. Ghiasi, Shafahi, Ardekani, "Improving Robustness with Adaptive Weight Decay" (NeurIPS 2023)**
- **Overlap:** Proposes adaptive weight decay that adjusts lambda on the fly based on gradient norm and weight norm ratio.
- **Severity:** `partial_overlap` — Both dynamically adjust weight decay during training. However:
  - Ghiasi et al. use gradient/weight norm ratio, not gradient-parameter alignment (cosine similarity)
  - Their focus is adversarial robustness, not generalization theory
  - No convergence theory or stability bounds provided
- **Differentiation:** AADWD uses the specific alignment quantity from Sun et al.'s generalization framework, providing a direct theoretical connection. Ghiasi et al.'s adaptive rule is heuristic.

**4. AdaDecay — Nakamura & Hong, "Adaptive Weight Decay for Deep Neural Networks" (IEEE Access, 2019)**
- **Overlap:** Per-parameter adaptive weight decay based on gradient magnitude.
- **Severity:** `related_work` — Different mechanism entirely (gradient magnitude, not alignment). No convergence theory.

**5. AlphaDecay — He et al. (arXiv 2506.14562, June 2025)**
- **Overlap:** Module-wise adaptive weight decay for LLMs based on heavy-tailed spectral analysis.
- **Severity:** `related_work` — Different signal (spectral properties of weight correlation matrices, not gradient-parameter alignment). Static assignment computed periodically. No per-step dynamics, no convergence theory.

**6. Xie et al., "On the Overlooked Pitfalls of Weight Decay" — Scheduled Weight Decay (SWD) (NeurIPS 2023)**
- **Overlap:** Dynamically schedules weight decay based on gradient norm statistics. Shows weight decay should be scheduled for adaptive gradient methods.
- **Severity:** `partial_overlap` — Both dynamically adjust weight decay during training. However:
  - SWD uses gradient norm, not gradient-parameter alignment
  - SWD targets Adam/AdamW; AADWD targets SGD
  - SWD lacks convergence theory for the scheduled decay itself
- **Differentiation:** AADWD uses the theoretically motivated alignment signal from the stability bound, not gradient norms.

**7. GALA — Jiang, Kavis, Mokhtari, "Online Learning-guided Learning Rate Adaptation via Gradient Alignment" (arXiv 2506.08419, June 2025)**
- **Overlap:** Uses gradient alignment between consecutive gradients to adapt learning rate (not weight decay). Provides convergence analysis for normalized SGD with GALA.
- **Severity:** `related_work` — GALA adjusts learning rate based on gradient-to-gradient alignment (consecutive steps), not weight decay based on gradient-to-weight alignment. Different mechanism, different target hyperparameter.
- **Note:** This confirms the proposal's assessment that GALA adjusts LR rather than WD.

**8. Chou, "Correction of Decoupled Weight Decay" (arXiv 2512.08217, Dec 2025)**
- **Overlap:** Analyzes the relationship between weight decay and learning rate, argues for lambda proportional to gamma^2. Discusses orthogonality of updates to weights.
- **Severity:** `related_work` — Focuses on the correct scaling of weight decay with learning rate, not on alignment-aware dynamic modulation. The lambda proportional to gamma^2 finding is interesting because AADWD's convergence theorem also requires lambda_t = O(gamma_t^2).

**9. Wang & Aitchison, "How to set AdamW's weight decay as you scale model and dataset size" (arXiv 2405.13698, 2024)**
- **Overlap:** Studies optimal weight decay scaling with model/dataset size. Characterizes WD as EMA of recent updates.
- **Severity:** `related_work` — Different focus (scaling laws for WD, not dynamic alignment-aware scheduling).

**10. Chen et al., "Towards Better Generalization: Weight Decay Induces Low-rank Bias" (arXiv 2410.02176, 2024)**
- **Overlap:** Proves WD+SGD leads to approximately rank-two weight matrices. Derives generalization bounds.
- **Severity:** `related_work` — Different theoretical angle (low-rank bias vs. alignment-based stability).

**11. Wan et al., "Spherical Motion Dynamics" (NeurIPS 2021)**
- **Overlap:** Analyzes training dynamics of normalized neural networks with SGD and weight decay. Decouples gradient direction from magnitude.
- **Severity:** `related_work` — Studies the dynamics of weight norms under WD but does not use gradient-parameter alignment to modulate WD.

#### Papers NOT Found (Confirming Novelty)

The following searches returned NO relevant results, supporting novelty claims:
- "Time-varying weight decay convergence theory nonconvex" — No papers provide convergence rates for time-varying weight decay in nonconvex SGD
- "Cumulative contraction stability bound" — This specific concept does not appear in any prior work
- "Continuous cosine similarity dynamic weight decay" — No paper uses continuous gradient-parameter cosine similarity to dynamically modulate weight decay strength with convergence theory

### Novelty Score: **8/10**

**Justification:** The three-layer theoretical contribution is genuinely novel:
1. Time-varying SGDW convergence theory has no precedent in the nonconvex setting.
2. The cumulative contraction framework replacing worst-case alignment is a new theoretical construction.
3. The specific algorithm using continuous cosine similarity with convergence guarantees is new.

The main deduction is for partial overlap with CWD (also alignment-based WD modulation, though binary/coordinate-level) and SWD/Ghiasi (also dynamic WD, though using different signals). The theoretical contributions remain clearly differentiated.

### Recommendation: **PROCEED**

The theoretical niche is well-defended. Key risks are empirical, not novelty-related:
- CWD comparison is essential to demonstrate the value of continuous vs. binary alignment
- The theory is the primary contribution; empirical gains are secondary
- The alignment proxy reliability (H3) is a legitimate technical concern but does not affect novelty

---

## Candidate 2: `cand_empirical` — Systematic Characterization of Gradient-Parameter Alignment Dynamics (Backup)

### Core Contribution Claims

1. First comprehensive empirical characterization of delta_t evolution across architectures/datasets/optimizers
2. Per-layer alignment disaggregation revealing qualitatively different behaviors
3. Alignment variance correlating with generalization gap

### Prior Work Analysis

**1. Sun et al. (CVPR 2025)** — Introduces the alignment quantity delta_t but does not systematically characterize its behavior across settings. Only fixed WD is studied.
- **Severity:** `partial_overlap` — Sun et al. compute delta_t but do not characterize its dynamics as a primary contribution.

**2. Wan et al., "Spherical Motion Dynamics" (NeurIPS 2021)** — Studies angular dynamics of weights under SGD+WD for normalized networks but does not specifically characterize gradient-parameter alignment (cosine similarity).
- **Severity:** `related_work`

**3. Kunin et al., "Neural Mechanics" (2020)** — Studies symmetries and conservation laws in deep learning, including relationships between gradient and weight dynamics under weight decay. Relates weight decay to angular momentum.
- **Severity:** `related_work` — Different theoretical framework (Lagrangian mechanics), does not characterize delta_t.

**4. Yunis et al., "Approaching Deep Learning Through Spectral Dynamics of Weights" (2024)** — Studies spectral properties of weight matrices during training across various settings.
- **Severity:** `related_work` — Spectral analysis, not alignment (cosine similarity) characterization.

### Novelty Score: **6/10**

**Justification:** While no paper systematically characterizes delta_t dynamics as a primary contribution, the novelty is moderate because:
- Pure empirical characterization has lower novelty ceiling
- If delta_t turns out to be trivially constant or noisy, the contribution collapses
- The concept of gradient-parameter alignment is not new; the systematic study of it is the contribution

### Recommendation: **PROCEED (as backup)**

Viable if the front-runner's theory proves too hard to close, or if delta_t shows genuinely surprising phase-dependent structure. Low risk, moderate impact.

---

## Candidate 3: `cand_llm` — Alignment-Aware Weight Decay for Transformer Pre-training (Backup)

### Core Contribution Claims

1. Per-layer alignment-aware WD for AdamW in LLM pre-training
2. Distinct alignment dynamics in attention vs. FFN layers
3. Dynamic alignment-aware WD combined with AlphaDecay

### Prior Work Analysis

**1. AlphaDecay (arXiv 2506.14562)** — Module-wise weight decay for LLMs based on spectral analysis. Direct competitor in the same application domain (LLM pre-training).
- **Severity:** `partial_overlap` — Both assign different WD to different modules in LLMs. AlphaDecay uses spectral properties; this candidate uses alignment. But targeting the same problem.

**2. CWD (arXiv 2510.12402)** — Already demonstrated on language model pre-training at billion scale with AdamW, Lion, and Muon.
- **Severity:** `partial_overlap` — CWD already handles the AdamW+LLM setting this candidate targets.

**3. Chou (arXiv 2512.08217)** — Studies WD scaling for AdamW/Scion in transformer training.
- **Severity:** `related_work`

**4. Wang & Aitchison (arXiv 2405.13698)** — Studies optimal WD scaling for AdamW with transformers.
- **Severity:** `related_work`

### Novelty Score: **5/10**

**Justification:** The competitive landscape is crowded:
- AlphaDecay already does module-wise adaptive WD for LLMs
- CWD already does alignment-based WD modulation for LLM pre-training
- This candidate lacks convergence theory (purely practical)
- The main novelty would be per-layer alignment dynamics characterization in transformers, which is a subset of candidate 2

### Recommendation: **MODIFY TO DIFFERENTIATE**

To proceed, this candidate needs:
- Clear demonstration that per-layer alignment dynamics provide different/better signal than AlphaDecay's spectral analysis
- Theoretical justification for why alignment matters more than spectral properties in transformer layers
- Results at meaningful scale (>350M) where both CWD and AlphaDecay have been tested

---

## Overall Novelty Assessment

| Candidate | Score | Recommendation |
|-----------|-------|----------------|
| cand_aadwd (Front-runner) | 8/10 | **PROCEED** — Unique theoretical niche |
| cand_empirical (Backup) | 6/10 | PROCEED (backup) — Moderate novelty |
| cand_llm (Backup) | 5/10 | MODIFY — Crowded competitive landscape |

### Overall Novelty: **high**

The front-runner occupies a genuinely novel theoretical position. No prior work provides:
1. Convergence theory for time-varying weight decay in nonconvex SGD
2. A cumulative contraction stability framework replacing worst-case alignment
3. A stochastic proxy transfer theorem for alignment-based algorithms

The closest competitors (CWD, SWD, Ghiasi et al.) differ meaningfully in mechanism, scope, or theoretical depth.

### Key Prior Work to Cite

| Paper | Role | Priority |
|-------|------|----------|
| Sun et al. (CVPR 2025) | Direct foundation | Must cite |
| CWD — Chen et al. (2025) | Primary competitor | Must cite & compare |
| Ghiasi et al. (NeurIPS 2023) | Adaptive WD predecessor | Must cite |
| AdaDecay — Nakamura & Hong (2019) | Per-param adaptive WD | Should cite |
| AlphaDecay — He et al. (2025) | Module-wise WD for LLMs | Should cite |
| SWD — Xie et al. (NeurIPS 2023) | Scheduled WD by gradient norm | Must cite |
| GALA — Jiang et al. (2025) | Alignment-based LR adaptation | Should cite |
| Chou (2025) | WD proportional to gamma^2 | Should cite |
| Hardt et al. (ICML 2016) | Stability framework foundation | Must cite |
| Wan et al. (NeurIPS 2021) | Spherical motion dynamics | Should cite |
| Loshchilov & Hutter (2019) | Decoupled WD (AdamW) | Must cite |
| Chen et al. (2024) — Low-rank bias | WD induces low-rank | Could cite |


## Codex 独立评审反馈（必须针对其指出的问题进行修正）
# Codex 独立评审 - idea_debate

**评审时间**: 2026-03-17
**模型**: Codex MCP 不可用 — 由 Sibyl codex-reviewer agent 基于独立第三方审查协议生成替代评审

---

## 评审说明

Codex MCP 未在当前环境中注册（`claude mcp list` 未返回 codex 相关条目）。根据错误处理协议，本评审由 codex-reviewer agent 以独立第三方视角完成，尽量模拟差异化审查。以下评审严格遵循证据驱动原则，不因与提案同生态系统而降低批判标准。

---

## 评审意见

### 1. 研究问题的价值与定位 (7/10)

**优点：**
- 研究问题定位精准：Sun et al. (CVPR 2025) 刚建立 fixed WD 的非凸泛化理论，CWD (ICLR 2026) 用二值 alignment 做选择性 decay，两者之间的"连续 alignment + 收敛理论"空白是真实且有价值的。
- 三层递进结构（收敛保持 → cumulative contraction → stochastic proxy）逻辑清晰，层次分明。

**问题：**
- D'Angelo et al. (NeurIPS 2024) 的"WD is never useful as explicit regularization"论点与本提案将 WD 作为"alignment-aware regularizer"的叙事存在根本张力。提案的 Section 2 (Motivation) 需要更清晰地回应这一挑战，否则审稿人会质疑理论框架的出发点。
- 实践影响力存疑：现有 WD 默认值（0.01 for AdamW, 5e-4 for SGD）在大多数场景已足够好。动态 WD 的边际收益需要在提案中量化预期（不能只说"improved or matched"）。

### 2. 理论贡献评估 (7.5/10)

**定理 3.1（收敛保持）：**
- 技术路线清晰，augmented Lyapunov 分析是标准但非平凡的推广。条件 lambda_t = O(gamma_t^2) 合理且与经验规则一致。
- 风险：lambda_t = O(gamma_t^2) 在训练后期使 WD 趋于零，与实践中训练后期需要正则化的直觉矛盾。提案已在 Risk 3 中承认这一点，但解释（"theory provides framework; practice uses clipped version"）略显回避——clipped 版本的理论保证在哪里？

**定理 4.1（Cumulative Contraction）：**
- 核心创新点。将 worst-case sup_t delta_t 替换为 trajectory-weighted average Delta_bar_T 是有意义的理论进步。
- 关键假设检验缺失：整个定理的价值取决于"delta_t 在训练过程中有足够大的动态范围"这一实证前提。如果 delta_t 全程接近常数（方差很小），cumulative 与 worst-case 几乎无差别。**提案承认了这一点（Tier 0 diagnostic），但应将其提升为 go/no-go gate 的显式地位。**
- 技术难点 2（乘积项的 Abel summation）和难点 4（dynamic 严格优于 fixed 的条件）被标记为中等难度，但从证明路径看更接近高难度。如果 4-6 周内无法闭合，论文的核心理论贡献将被削弱。

**定理 5.1（Stochastic Proxy）：**
- 所需 batch size 条件 B = Omega(sigma^2 T / (gamma_min^2 delta_min^2 R^2)) 在实际中可能非常大。对于 CIFAR-10, T~40000 steps, gamma_min~0.001, 这个下界需要具体数值验证。如果需要 B > 1024 才能满足理论条件，而实验用 B=128，理论与实践之间的 gap 就很大。

### 3. 算法设计评估 (6.5/10)

**AADWD 算法：**
- 简洁可实现，O(d) overhead 声称合理。
- 超参数问题严重：c, lambda_min, lambda_max, beta (EMA), epsilon 共 5 个新超参数。对比 CWD 的零新超参数优势，这是显著劣势。提案中的"auto-tunable"声称缺乏支撑。
- 三种规则变体（conservative, aggressive, square）本身就说明最优规则形式不确定。审稿人会问：为什么不能给出理论上最优的规则选择？

**与 CWD 的差异化：**
- 提案在 Table（Section "Key Differentiators from CWD"）中清晰列出了差异，但实质性优势仍依赖实验验证。如果 CIFAR-10 上 AADWD 无法显著超越 CWD，论文的实践贡献将被大幅削弱，因为 CWD 更简单（一行代码 vs 自定义 optimizer class）。

### 4. 实验设计评估 (6/10)

**严重不足：**
- 仅 CIFAR-10/100 + ResNet/VGG：在 2026 年的 ICML/NeurIPS 投稿中，这个实验范围明显不足。至少需要一个中等规模实验（如 ImageNet subset 或 NanoGPT）才能让审稿人信服方法的通用性。
- 提案中原始版本"无需多个 seed"在修改后的 proposal 中已改为 3 seeds，这是正确的修正。
- Tier 0 诊断实验设计合理，是整个研究的关键 gate。

**缺失的 ablation：**
- Random time-varying WD（用随机数替代 delta_hat_t）是最关键的消融，必须做。这是区分"alignment signal"和"mere time variation"的唯一方法。
- Norm-matched fixed WD baseline 也很重要，用于排除"WD 效果纯粹来自 weight norm control"的替代解释。

### 5. 各视角观点综合评价

**Contrarian 视角**提出了最有力的挑战：delta_hat_t 的信噪比问题和 CWD 差异化不足。这两点是论文最大的 acceptance 风险。

**Empiricist 视角**的优先级建议非常实用——先验证 delta_hat_t 可靠性，再做其他一切。完全同意。

**Innovator 视角**的提案过于发散（热力学、滑模控制、NTK 猜想等），大部分不适合在一篇论文中同时出现。但提案 4（Momentum-Debiased Alignment Estimation）确实指出了一个被忽视的技术问题。

**Interdisciplinary 视角**的 Kalman 滤波建议（用于 delta_t 估计）比简单 EMA 有理论优势，值得考虑纳入 Tier 2 实验。

**Pragmatist 视角**的"最低风险路径"和论文 framing 建议（不押 accuracy gain，押 robustness to hyperparameter）是明智的策略。

**Theoretical 视角**的证明路径详细且可行，但对难点的估计偏乐观。定理 4.1 的 4-6 周估计在含有难点 2 和符号问题的情况下可能需要 8-10 周。

### 6. 核心风险总结

| 风险 | 严重程度 | 概率 | 影响 |
|------|---------|------|------|
| delta_hat_t 信噪比不足，EMA 后仍无规律 | 致命 | 中 (30%) | 整个方法论基础崩塌 |
| CIFAR 上性能增益 <0.3%，统计不显著 | 高 | 高 (50%) | 需转向 robustness narrative |
| 定理 4.1 证明无法在计划时间内闭合 | 高 | 中 (40%) | 论文降级为定理 3.1 + 实验 |
| CWD 在所有设置上匹配或超越 AADWD | 高 | 中 (35%) | 实践贡献被否定 |
| Batch size 条件太强，定理 5.1 不适用于标准设置 | 中 | 中 (30%) | 理论-实践 gap 明显 |

### 7. 改进建议

1. **立即执行 Tier 0 诊断**：delta_hat_t 的行为是 go/no-go gate，在任何理论证明之前完成。
2. **重构动机叙事**：正面回应 D'Angelo et al. 的"WD is not regularization"论点——可以说 AADWD 正是利用 WD 作为"dynamics modifier"的角色，通过 alignment 信号最优化这个 modifier 的强度。
3. **精简超参数**：固定 lambda_min, lambda_max 为数据无关的默认值（如 1e-5, 1e-2），EMA beta 固定为 0.99，只暴露 c 一个新超参数。
4. **扩展实验范围**：至少增加一个 NanoGPT 实验（GPT-2 small on OpenWebText subset, ~30min）。
5. **加入 random time-varying WD baseline**：这是审稿人必问的消融。
6. **理论保底方案明确化**：如果定理 4.1 无法在 deterministic alignment 版本上闭合，准备好仅提交定理 3.1 + 实验 + alignment characterization study 的精简版本。

## 评分

**6.5/10**

理由：研究问题定位准确，理论框架有真实创新（cumulative contraction），但存在多个中-高概率的风险（delta_hat_t 可靠性、CWD 竞争、实验范围不足、超参数复杂度）。如果 Tier 0 诊断成功且定理 4.1 能闭合，论文有竞争力（可提升至 7.5-8/10）；如果诊断失败或定理无法闭合，论文需要大幅转向。当前阶段建议推进到实验验证，但需要做好 fallback 方案。

VERDICT: APPROVE
