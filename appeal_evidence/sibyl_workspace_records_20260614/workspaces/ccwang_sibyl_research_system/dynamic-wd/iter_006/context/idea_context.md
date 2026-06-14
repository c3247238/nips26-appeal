

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


## 文献调研报告（请仔细阅读，避免重复已有工作）
# Literature Survey Report

**Research Topic**: Unified Dynamic Weight Decay Framework -- Unifying WD Scheduling, Alignment-Aware WD, Decoupled WD, and Norm-Matched WD into a Theoretical Framework with Standardized Evaluation Metrics (Budget Equivalence Metric, Coupling Stability Index, Alignment Informativeness Score)
**Survey Date**: 2026-03-19 (Refreshed: comprehensive dual-source search arXiv + Web, 38 core references)
**arXiv Search Keywords**: ["weight decay scheduling dynamic adaptive", "decoupled weight decay AdamW regularization", "weight decay norm alignment gradient neural network deep learning", "cautious weight decay", "weight norm control Loshchilov", "weight decay layer-wise module-wise parameter-specific", "Scion Muon optimizer weight decay", "weight decay orthogonal radial tangential optimizer", "weight decay Vision Transformer ImageNet CIFAR", "understanding scheduling weight decay", "NOVAK unified adaptive optimizer weight decay", "gradient-to-weight ratio weight decay training dynamics", "CVPR 2025 weight decay nonconvex SGD convergence", "spectral dynamics weights rank minimization weight decay", "LLM pretraining WD schedule cosine warmup-stable-decay"]
**Web Search Keywords**: ["weight decay scheduling deep learning state of the art 2025", "decoupled weight decay AdamW regularization survey 2024 2025", "adaptive weight decay neural network training benchmark comparison", "cautious weight decay CWD optimizer 2025 paper", "weight norm control Loshchilov weight decay target norm optimizer", "weight decay deep learning open source implementation benchmark GitHub 2024 2025", "understanding scheduling weight decay Xie NeurIPS 2023 SWD optimizer", "Loshchilov Hutter decoupled weight decay regularization AdamW ICLR 2019", "NOVAK unified adaptive optimizer 2026", "gradient-to-weight ratio weight decay training dynamics Defazio 2025", "CVPR 2025 weight decay nonconvex SGD Sun", "spectral dynamics of weights rank minimization 2408.11804", "WSD warmup-stable-decay LLM training weight decay 2025 2026", "weight decay scheduling deep learning state of the art 2025 2026", "adaptive weight decay neural network training benchmark 2025", "norm-matched weight decay alignment-aware weight decay gradient 2024 2025", "weight decay optimizer survey unified framework SGD Adam 2024 2025 arXiv", "cautious weight decay AlphaDecay module-wise weight decay LLM training 2025", "stable weight decay regularization AdamW instability gradient norm scheduling github 2024", "weight decay learning rate coupling interaction scale-invariant batch normalization 2024 2025", "weight decay transfer learning rate µP maximal update parameterization 2025"]

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
| 36 | **Improving Robustness with Adaptive Weight Decay** (Apple / NeurIPS 2023) | NeurIPS 2023 | 2023 | Auto-tunes WD per iteration for adversarial robustness; 20% relative robustness improvement on CIFAR-100; smaller weight norms; less LR sensitivity | Focused on adversarial robustness; not general training dynamics |
| 37 | **Is your batch size the problem? Revisiting the Adam-SGD gap in language modeling** | arXiv:2506.12543 | 2025 | Shows Adam-SGD gap is largely a batch size / gradient noise phenomenon; critical batch size differs per optimizer; reframes WD's role in optimizer comparison | Does not directly address dynamic WD; provides important context for WD-optimizer interaction |
| 38 | **Normalize-and-Project (NaP)** | NeurIPS 2024 | 2024 | Principled alternative to WD for norm growth control; flexible recipe applicable to any architecture; avoids WD over/under-regularization tradeoff | Alternative to WD rather than a WD method; relevant as comparison baseline |

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
- **Apple Adaptive WD**: NeurIPS 2023 (per-iteration auto-tuned WD for robustness; smaller weight norms baseline)
- **Normalize-and-Project (NaP)**: NeurIPS 2024 (alternative to WD for norm control; useful as comparison baseline)

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


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Research Proposal: Unified Dynamic Weight Decay Framework

## Title

**Lyapunov-Certified Dynamic Weight Decay: A Unified Convergence Framework with Cumulative Alignment Theory**

## Abstract

Weight decay (WD) is ubiquitous in deep learning, yet the field has fragmented into independently developed approaches -- constant, cosine-scheduled, gradient-norm-aware (SWD), alignment-based (CWD), norm-targeted (AdamWN), and spectral-guided (AlphaDecay) -- each with its own convergence analysis, assumptions, and evaluation protocol. We propose the first unified mathematical framework that (1) provides a Lyapunov convergence certificate for general time-varying WD schedules lambda(t), guaranteeing convergence whenever lambda(t) lies within a computable "certified band"; (2) proves that all major WD methods satisfy this certificate as special cases; (3) establishes a cumulative alignment generalization bound that replaces the worst-case alignment quantity of Sun et al. (CVPR 2025) with a trajectory-level analysis, yielding strictly tighter bounds; and (4) derives PMP-WD as the provably optimal schedule within the certified family. Our framework resolves a central puzzle in the literature: why dynamic WD methods rarely outperform constant WD on standard benchmarks. The Lyapunov analysis shows that for architectures with batch normalization, the certified band is narrow, leaving little room for improvement -- a theoretical prediction we validate empirically across CIFAR-10/100 with ResNet-20, VGG-16-BN, and ImageNet with ResNet-50.

## Motivation

The weight decay literature has produced 15+ methods since 2023 (CWD, SWD, AdamWN, AlphaDecay, AdamO, ADANA, CPR, NaP, etc.), each claiming improvements on different benchmarks with different metrics. Practitioners face decision paralysis: which method to use?

Three fundamental questions remain unanswered:

1. **Convergence**: Under what conditions on lambda(t) does SGD with time-varying weight decay converge? CWD (ICLR 2026) proves convergence for binary WD via Lyapunov + LaSalle. Sun et al. (CVPR 2025) prove convergence for fixed WD. No existing work handles general time-varying schedules.

2. **Generalization**: Sun et al.'s generalization bound uses worst-case alignment delta_T = sup_t delta_t. Can we do better with a cumulative alignment analysis that leverages the full alignment trajectory?

3. **Unification**: Are the various WD methods fundamentally different, or special cases of a common principle? If the latter, what determines when one method dominates another?

Our framework addresses all three questions with a coherent theoretical structure grounded in Lyapunov stability theory and uniform stability analysis.

## Research Questions

- **RQ1**: Can we derive computable sufficient conditions (a "certified band") for convergence of SGD with arbitrary time-varying WD?
- **RQ2**: Does cumulative alignment predict generalization better than worst-case alignment?
- **RQ3**: Do all major WD methods satisfy the unified Lyapunov certificate?
- **RQ4**: Is PMP-WD (Pontryagin's Maximum Principle-derived schedule) optimal within the certified family?
- **RQ5**: When and why does alignment-aware WD provide genuine benefit over simple scheduled WD?

## Hypotheses

### H1: Unified Lyapunov Certificate
For V_t = f(w_t) + mu(lambda_t, ||w_t||, t), there exist computable bounds [lambda_min(t), lambda_max(t)] such that any schedule lambda(t) in this band guarantees E[V_T - V_0] <= -Omega(T) (convergence). The certificate uses composite Lyapunov criteria (Hassan Saoud 2025) to handle the time-varying case without requiring LaSalle's invariance principle.

### H2: Cumulative Alignment Bound
The generalization gap satisfies:
gen(A, S) <= (2M/n) * sum_t gamma_t * exp(-sum_{s=t+1}^{T-1} lambda_s(1-delta_s) + O(gamma_s L))
This improves on Sun et al. (CVPR 2025) by replacing sup_t delta_t with the alignment trajectory {delta_t}. When average alignment is low (gradient nearly orthogonal to weights), WD is maximally effective and the bound tightens significantly.

### H3: Subsumption
Constant WD, CWD, cosine schedule, SWD, and PMP-WD all lie within the certified band under their respective standard hyperparameter ranges. Formally:
- Constant WD: lambda_0 satisfies the certificate when lambda_0 * gamma_t <= min(1, alpha_t / (mu_t * ||w_t||^2))
- CWD: Binary mask {0, lambda_0} satisfies the certificate because lambda_t = 0 trivially satisfies the lower bound, and lambda_t = lambda_0 satisfies the upper bound by CWD's sign-alignment condition
- Cosine schedule: satisfies the certificate for lambda_0 <= lambda_max(0)
- SWD: lambda_t = lambda_0 / (1 + c * ||g_t||) satisfies the certificate because reducing lambda when ||g_t|| is large increases the margin to lambda_max

### H4: PMP-WD Optimality
Among all schedules in the certified band, PMP-WD lambda*_t = clip(kappa * (rho* - rho_hat_t)^+, 0, lambda_max(t)) achieves the tightest convergence bound.

### H5: Empirical Alignment Informativeness
On CIFAR-10/100 with standard BN architectures, the alignment signal delta_hat_t has low variance relative to the certified band width, explaining why all WD methods perform similarly (~0.5% spread). On non-BN architectures or under high WD regimes, alignment variance increases and alignment-aware methods differentiate.

## Expected Contributions

1. **First unified Lyapunov convergence certificate for general time-varying WD** (Theorem 1) -- generalizing CWD (ICLR 2026, binary WD), Sun et al. (CVPR 2025, fixed WD), and Ding et al. (fixed decoupled WD in Adam-family).

2. **First cumulative alignment generalization bound** (Theorem 2) -- replacing worst-case delta_T with trajectory-level analysis, yielding strictly tighter bounds when alignment varies across training.

3. **First formal subsumption of existing WD methods as special cases** (Theorem 3) -- constant, CWD, cosine, SWD, PMP-WD all shown to satisfy the same convergence guarantee under standard hyperparameter ranges.

4. **Minibatch alignment concentration bound** (Proposition) -- bridging the gap between theory (population gradients) and practice (minibatch gradients).

5. **Diagnostic metrics** (CSI, AIS) -- Coupling Stability Index and Alignment Informativeness Score that predict when dynamic WD will provide benefit over constant WD.

6. **Comprehensive empirical analysis** -- systematic tracking of alignment, norms, spectral rank, and effective LR across 7+ WD methods on CIFAR-10/100, ImageNet, with proper controls (budget matching, random mask, BN ablation).

## Method

### Theoretical Framework

**Theorem 1 (Lyapunov Certificate for Time-Varying WD):**
Consider SGD with time-varying WD: w_{t+1} = (1 - lambda_t * gamma_t) * w_t - gamma_t * g_t. Define V_t = f(w_t) + mu_t * ||w_t||^2, where mu_t satisfies a backward recursion with terminal condition mu_T = 0. Then V_{t+1} - V_t <= -alpha_t * ||grad f(w_t)||^2 + C * gamma_t^2 * sigma^2 whenever lambda_t in [lambda_min(t), lambda_max(t)].

**Theorem 2 (Cumulative Alignment Generalization Bound):**
Under standard smoothness and bounded variance assumptions, if lambda_t = O(gamma_t^2) and delta_t < 1, the generalization gap depends on the cumulative contraction product prod_s (1 - lambda_s(1-delta_s)) rather than worst-case sup_t delta_t. This recovers Sun et al.'s bound as a special case.

**Theorem 3 (Subsumption):**
All major WD methods (constant, CWD, cosine, SWD, PMP-WD) satisfy the Lyapunov certificate from Theorem 1 as special cases.

**Theorem 4 (PMP-WD Optimality):**
PMP-WD maximizes the convergence rate (tightest bound on E[V_T]) within the certified family.

**Proposition 5 (Minibatch Alignment Concentration):**
Under sub-Gaussian gradient noise with variance sigma^2 and batch size B:
P(|delta_hat_t - delta_t| > epsilon) <= 2 * exp(-B * epsilon^2 * ||grad f||^2 / (C * sigma^2))

### Unified Operator Formulation

All WD methods expressible as: w_{t+1} = (I - Lambda_t) w_t - gamma_t P_t g_t, where Lambda_t = Lambda(w_t, g_t, t; theta) is a positive semidefinite diagonal operator. The effective contraction spectrum eig(I - Lambda_t) characterizes both convergence (Theorem 1) and generalization (Theorem 2). Two methods with matched contraction spectra achieve equivalent performance.

### Spectral Dynamics Complement

Under the Dyson BM model (Olsen et al. ICML 2025), WD modifies the singular value SDE drift. Different WD schedules produce different spectral trajectories; the certified band constrains which trajectories guarantee convergence. This provides the mechanistic "why" that complements the Lyapunov "when."

## Experimental Plan

### Phase 1: CIFAR-10/100 Diagnostic Campaign (Priority 1)

| Experiment | Dataset | Model | Methods | Seeds | Purpose |
|---|---|---|---|---|---|
| Certificate visualization | CIFAR-10 | ResNet-20 | All 7+ | 42,123,456 | Track V_t, verify methods lie within certified band |
| Alignment trajectory | CIFAR-10/100 | ResNet-20, VGG-16-BN | All 7+ | 42,123,456 | Track delta_t, compute bar_delta_T vs sup_t delta_t |
| Budget-matched comparison | CIFAR-10 | ResNet-20 | Fixed, time-only, alignment-aware, CWD, random mask | 42,123,456,789,2024 | Test whether alignment adds value beyond scheduling |
| BN ablation | CIFAR-10 | ResNet-20 (no BN) | Fixed, alignment-aware, CWD | 42,123,456 | Isolate alignment signal from effective LR confound |

Methods: no_wd, constant, cosine_schedule, cwd_hard, swd, half_lambda, random_mask, PMP-WD (new), alignment-adaptive (new)

**Key metrics**: Test accuracy (mean +/- std), Lyapunov V_t trajectory, certified band [lambda_min(t), lambda_max(t)], delta_t trajectory (cumulative and worst-case), ||w_t||, ||g_t||, CSI, AIS.

### Phase 2: Multi-Architecture Validation (Priority 1)

VGG-16-BN/CIFAR-10 with same setup. Purpose: verify certificate band is architecture-independent (up to L and sigma^2).

### Phase 3: Cumulative vs Worst-Case Alignment Grid (Priority 1)

Grid: 6 WD strengths x 4 schedules x 2 architectures x 2 datasets = 96 configurations x 3 seeds. Compute Spearman correlation of bar_delta_T and sup_t delta_t with generalization gap. Partial correlation controlling for WD strength.

### Phase 4: ImageNet Validation (Priority 2)

ResNet-50, 90 epochs, 4 methods (constant, CWD, PMP-WD, alignment-adaptive) x 3 seeds. Estimated: ~72 GPU-hours.

### Phase 5: AdamW Comparison (Priority 2)

All methods under AdamW. Expected: all certified schedules achieve equivalent accuracy (Phi invariance). The certificate band should be narrower under AdamW.

### Falsification Criteria

1. If any known-convergent method (constant, CWD) violates the certified band => certificate is too conservative (relax Theorem 1)
2. If |rho(bar_delta_T, gen_gap)| - |rho(sup_t delta_t, gen_gap)| < 0.05 => cumulative alignment not meaningfully better (H2 weakened)
3. If PMP-WD does not achieve best V_T among certified methods => H4 falsified
4. If both |rho| < 0.3 for alignment measures => alignment framework itself questionable

### Controls

1. **Budget matching**: All alignment-aware methods calibrated to same total sum(lambda_t * ||w_t||^2)
2. **Random mask**: Same sparsity as CWD but random selection (controls for "less decay" confound)
3. **LR schedule**: Identical cosine decay with warmup across all methods
4. **Oracle alignment**: Full-batch delta_t vs minibatch proxy (quantify noise floor)

## Resource Estimate

- Phase 1-2: ~42 GPU-hours (partially completed in iter_003)
- Phase 3: ~48 GPU-hours (96 configs x 3 seeds x 10min)
- Phase 4: ~72 GPU-hours (ImageNet)
- Phase 5: ~21 GPU-hours (AdamW)
- **Total**: ~183 GPU-hours
- **Wall-clock**: ~23 hours with 8x RTX PRO 6000

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Lyapunov certificate too conservative (narrow band) | 30% | Medium | Use composite Lyapunov approach; if band is narrow, this EXPLAINS why constant WD is hard to beat (itself a contribution) |
| PMP-WD does not beat constant WD empirically | 35% | Low | Theory stands regardless; reframe as "constant WD is near-optimal within certified family" |
| Cumulative alignment not better than worst-case | 25% | High | Fall back to certificate + subsumption as primary contribution; cumulative bound becomes secondary |
| ImageNet experiments fail | 20% | Medium | Use CIFAR-100 with deeper models (ResNet-56, WRN-28-10) as intermediate scale |
| Alignment signal uninformative (both |rho| < 0.3) | 15% | High | Pivot to "alignment mirage" narrative: explain why alignment-based WD reduces to implicit LR scheduling under BN |
| BN confound invalidates alignment-based WD motivation | 30% | Medium | This IS a finding -- characterize precisely when alignment is informative vs when it is a proxy for effective LR |

## Novelty Assessment

### Novelty verification via literature search

**Claim 1: Unified Lyapunov certificate for time-varying WD.** Searched "Lyapunov convergence certificate time-varying weight decay" -- no prior work provides this. CWD (ICLR 2026) proves convergence for binary WD via LaSalle. Sun et al. (CVPR 2025) prove convergence for fixed WD. Lion-K (OPT 2025) uses Lyapunov analysis for a specific optimizer family but does not address general WD schedules. **Novelty confirmed.**

**Claim 2: Cumulative alignment generalization bound.** Searched "cumulative alignment generalization bound weight decay SGD" -- Sun et al. (CVPR 2025) provide worst-case alignment bound. No paper extends to cumulative/trajectory-level alignment. Galanti et al. (CPAL 2025) study rank minimization, not alignment-dependent bounds. Holzl et al. (NeurIPS 2025) study gradient-weight alignment empirically but provide no formal bound. **Novelty confirmed.**

**Claim 3: Formal subsumption of WD methods.** Searched "unified framework dynamic weight decay scheduling alignment" -- no unified framework exists. SWD, CWD, AlphaDecay, AdamWN are independently developed with separate analyses. Kosson et al. (2025) unify WD and muP for LR transfer but do not address the full WD method landscape. **Novelty confirmed.**

### Differentiation from closest work

| Prior Work | Their Contribution | Our Differentiation |
|---|---|---|
| Sun et al. (CVPR 2025) | Fixed WD generalization bound with worst-case delta_T | Time-varying WD with cumulative alignment |
| CWD (ICLR 2026) | Lyapunov convergence for binary WD | General time-varying WD Lyapunov certificate |
| SWD (NeurIPS 2023) | Gradient-norm-aware scheduling | Shown as special case of our framework |
| Defazio (2025) | WD as gradient-to-weight ratio controller | We provide formal convergence guarantees for the control perspective |
| Hassan Saoud (2025) | Composite Lyapunov criteria | We apply their tools to the WD problem (novel application) |
| Olsen et al. (ICML 2025) | Spectral dynamics of SGD via Dyson BM | We extend to show how WD modifies spectral drift |

## Addressing Prior Critiques

### From evolution lessons
- **"No deeper theoretical results beyond trivial Proposition 1"**: Addressed by Theorems 1-4 providing genuine theoretical depth (Lyapunov certificate, cumulative alignment bound, subsumption, optimality).
- **"Mechanistic hypothesis needs quantitative verification"**: Addressed by the spectral dynamics complement and comprehensive diagnostic experiments tracking all relevant quantities.

### From contrarian perspective
- **"Alignment signal is noisy/unreliable"**: Directly tested via oracle alignment ablation, random redistribution control, and minibatch concentration bound (Proposition 5).
- **"BN confound"**: Directly tested via BN ablation experiments and theoretical analysis of how BN affects the certified band width.
- **"WD is dead, use CPR/NaP"**: Our framework does not claim a new SOTA WD method. Instead, it provides theoretical understanding of WHEN any WD method works, including the conditions under which alternatives like CPR/NaP would be preferable.

### From pragmatist perspective
- **"Just constant WD works"**: Our framework EXPLAINS this as a prediction (narrow certified band under BN). This transforms a "null result" into a theoretical insight.
- **"Skip spectral/thermodynamic analogies"**: Spectral analysis retained as motivational Section 3 material only, not as core algorithmic contribution. Thermodynamic lens dropped entirely from formal claims.


## 当前可检验假设
# Testable Hypotheses

## H1: Unified Lyapunov Certificate

**Statement**: For V_t = f(w_t) + mu_t * ||w_t||^2 with mu_t satisfying a backward recursion, there exist computable bounds [lambda_min(t), lambda_max(t)] such that any WD schedule lambda(t) in this band guarantees convergence (E[V_T] <= V_0 - Omega(T)).

**Expected outcome**: The certified band is non-trivial (lambda_max - lambda_min > 0.1 * lambda_max at all training phases) and widens during early training (when gradients are large, more WD is tolerated) and narrows during fine-tuning (when stability is critical).

**Falsification**: If the certified band is so narrow that only lambda(t) approximately constant satisfies it, the certificate is too conservative. This outcome is still informative -- it theoretically explains why constant WD is hard to beat.

**Measurement**: Compute lambda_min(t), lambda_max(t) from L, sigma^2, ||w_t||, ||g_t||, delta_t at each epoch. Overlay actual lambda(t) for each method.

---

## H2: Cumulative Alignment Predicts Generalization Better Than Worst-Case

**Statement**: The cumulative average alignment bar_delta_T = (1/T) sum_t delta_t correlates more strongly with generalization gap (train_acc - test_acc) than worst-case alignment sup_t delta_t.

**Expected outcome**: Spearman |rho(bar_delta_T, gen_gap)| > |rho(sup_t delta_t, gen_gap)| by at least 0.1 across the WD strength x schedule grid.

**Falsification**: If the correlation difference is < 0.05, or if both |rho| < 0.3, the cumulative alignment improvement is not empirically supported.

**Measurement**: Grid of 6 WD strengths x 4 schedules x 2 architectures x 2 datasets = 96 configs x 3 seeds. Compute full-batch delta_t every 10 epochs. Bootstrap 95% CI for correlation difference. Partial correlation controlling for WD strength.

---

## H3: Subsumption of Existing Methods

**Statement**: Constant WD, CWD, cosine schedule, SWD, and PMP-WD all lie within the certified band [lambda_min(t), lambda_max(t)] under their standard hyperparameter ranges.

**Expected outcome**: For each method, lambda(t) lies within the certified band for >= 95% of training steps. Occasional violations (< 5% of steps) may occur during phase transitions (LR drops, BN statistics shifts).

**Falsification**: If a known-convergent method (constant, CWD) violates the band for > 20% of training steps, the certificate needs relaxation.

**Measurement**: For each method on each seed, compute the fraction of training steps where lambda(t) in [lambda_min(t), lambda_max(t)].

---

## H4: PMP-WD Optimality

**Statement**: Among all WD schedules in the certified band, PMP-WD achieves the tightest (lowest) final Lyapunov function value V_T.

**Expected outcome**: V_T(PMP-WD) < V_T(constant) < V_T(cosine) < V_T(CWD). The ordering of V_T should correlate with test accuracy on SGD but not necessarily on AdamW (where Phi invariance equalizes performance).

**Falsification**: If another method achieves lower V_T than PMP-WD across all seeds, H4 is falsified (the optimality derivation has an error or the PMP assumptions are violated).

**Measurement**: Track V_t for all methods across training. Report V_T (mean +/- std across seeds) and test accuracy.

---

## H5: Alignment Informativeness Depends on Architecture

**Statement**: The alignment signal delta_hat_t is informative (AIS > threshold) for WD decisions in non-BN architectures but uninformative (AIS ~ 0) in BN architectures where WD primarily controls effective LR.

**Expected outcome**:
- ResNet-20 with BN: alignment variance < 0.1, all WD methods perform within 0.5% (as observed in iter_003)
- ResNet-20 without BN: alignment variance > 0.2, alignment-aware methods differentiate by > 1%
- VGG-16-BN: similar to ResNet-20 with BN (alignment uninformative)

**Falsification**: If alignment-aware WD helps significantly even with BN, the effective-LR interpretation is incomplete. If alignment is uninformative even without BN, the alignment framework has deeper problems.

**Measurement**: AIS (mutual information between delta_hat_t and next-epoch test accuracy change). BN vs no-BN head-to-head comparison.

---

## H6: Minibatch Alignment Proxy Quality

**Statement**: Minibatch alignment delta_hat_t concentrates around population alignment delta_t with deviation O(sigma / (||grad f|| * sqrt(B))).

**Expected outcome**: For batch size B=128 on CIFAR-10/ResNet-20, |delta_hat_t - delta_t| < 0.05 on average. The concentration improves with larger batches and larger gradient norms (early training).

**Falsification**: If average deviation > 0.2, the minibatch proxy is too noisy for practical alignment-aware WD.

**Measurement**: Compute both full-batch (full training set) and minibatch delta_t every 10 epochs for a subset of runs. Report mean and std of |delta_hat_t - delta_t|.

---

## H7: Budget-Matched Alignment Test

**Statement**: Alignment-aware WD achieves statistically significant test accuracy improvement over budget-matched time-only scheduling (same total sum(lambda_t * ||w_t||^2)).

**Expected outcome**: On CIFAR-10/ResNet-20 with BN: no significant difference (p > 0.1). On CIFAR-10/ResNet-20 without BN: significant improvement (p < 0.05, effect > 0.3%).

**Falsification**: If alignment-aware WD shows no benefit even without BN, alignment is not an actionable signal for WD decisions.

**Measurement**: Paired t-test across 5 seeds. Bootstrap 95% CI for accuracy difference. Random redistribution control (same marginal WD distribution, uncorrelated with alignment).

---

## H8: Contraction Spectrum Equivalence

**Statement**: Two WD methods with matched expected contraction spectra E[eig(I - Lambda_t)] achieve the same generalization bound and similar final test accuracy.

**Expected outcome**: A continuous alignment-adaptive rule calibrated to match CWD's average contraction achieves test accuracy within the statistical error bars of CWD.

**Falsification**: If methods with matched spectra achieve significantly different accuracy (> 0.5%, p < 0.05), the contraction spectrum is an insufficient characterization of WD effectiveness.

**Measurement**: Calibrate continuous alignment-adaptive lambda_t to match CWD's average eig(I - Lambda_t). Run head-to-head comparison with 3 seeds.


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_lyapunov_unified",
      "title": "Lyapunov-Certified Dynamic Weight Decay: Unified Convergence Framework with Cumulative Alignment Theory",
      "status": "front_runner",
      "summary": "Unified Lyapunov convergence certificate for general time-varying WD schedules, proving all major WD methods (constant, CWD, cosine, SWD, PMP-WD) are special cases within a certified band. Cumulative alignment generalization bound replacing worst-case delta_T. PMP-WD shown optimal within the certified family. Addresses the key weakness of prior iterations: 'no deeper theoretical results beyond trivial propositions.'",
      "hypotheses": [
        "H1: Computable certified band [lambda_min(t), lambda_max(t)] for convergence",
        "H2: Cumulative alignment predicts generalization better than worst-case (Spearman rho diff > 0.1)",
        "H3: All major WD methods satisfy the Lyapunov certificate as special cases",
        "H4: PMP-WD achieves optimal V_T within certified family",
        "H5: Alignment informativeness depends on BN presence",
        "H6: Minibatch alignment proxy concentrates with O(1/sqrt(B)) rate"
      ],
      "pilot_focus": "Certificate band computation on CIFAR-10/ResNet-20 + alignment trajectory tracking across 7 methods",
      "key_theorems": [
        "Theorem 1: Lyapunov certificate for time-varying WD",
        "Theorem 2: Cumulative alignment generalization bound",
        "Theorem 3: Subsumption (all methods as special cases)",
        "Theorem 4: PMP-WD optimality within certified family",
        "Proposition 5: Minibatch alignment concentration"
      ],
      "strengths": [
        "Addresses the evolution lesson about lacking deeper theoretical results",
        "Subsumption is the unification the paper title promises",
        "Connects Lyapunov theory (when) with spectral dynamics (why)",
        "Low new experiment cost -- same baselines, new theoretical contribution",
        "Resilient to negative empirical results (certificate + subsumption stand regardless)"
      ],
      "risks": [
        "Certificate too conservative (30% probability) -- narrow band means only constant WD works",
        "PMP-WD does not beat constant WD (35% probability) -- reframe as 'constant is near-optimal'",
        "Proofs require strong assumptions that limit practical relevance (20%)"
      ],
      "perspectives_weighted": {
        "innovator": "HIGH -- Lyapunov certificate + spectral dynamics complement adopted as core framework",
        "theoretical": "HIGH -- Cumulative alignment bound and convergence preservation directly adopted",
        "pragmatist": "MEDIUM -- Unified operator formulation adopted as notational framework; CSI/AIS metrics included",
        "contrarian": "HIGH -- BN confound analysis and alignment signal quality audit integrated as key ablations",
        "interdisciplinary": "MEDIUM -- Gain scheduling perspective informs operating region identification; thermodynamic lens dropped",
        "empiricist": "HIGH -- Budget-matched comparison, random mask control, oracle alignment ablation adopted"
      }
    },
    {
      "candidate_id": "cand_alignment_mirage",
      "title": "The Alignment Mirage: When Gradient-Weight Geometry Misleads Dynamic Regularization",
      "status": "backup",
      "summary": "Critical analysis demonstrating that in BN architectures, alignment-based WD is a noisy proxy for effective LR scheduling, not genuine alignment-aware regularization. Proposes simpler effective-LR-aware WD as alternative. First systematic measurement of alignment signal SNR.",
      "hypotheses": [
        "Alignment SNR < 1 in BN networks for most of training",
        "Alignment-aware WD helps only without BN (confound test)",
        "Effective-LR-aware WD matches alignment-aware WD with lower complexity",
        "Random alignment control performs comparably to true alignment in BN networks"
      ],
      "pilot_focus": "BN vs no-BN comparison with alignment tracking on CIFAR-10/ResNet-20",
      "strengths": [
        "Provocative thesis attracts attention",
        "Even negative results are publishable",
        "BN confound never explicitly tested in WD literature",
        "Lower computational cost than main proposal"
      ],
      "risks": [
        "If alignment IS genuinely informative even with BN, thesis collapses",
        "'Alignment mirage' narrative harder to publish at top venues without constructive alternative"
      ]
    },
    {
      "candidate_id": "cand_diagnostic_benchmark",
      "title": "Unifying Dynamic Weight Decay: Comprehensive Diagnostic Benchmark with Standardized Metrics",
      "status": "backup",
      "summary": "Comprehensive empirical benchmark comparing 7+ WD methods with unified PyTorch optimizer, standardized diagnostic metrics (CSI, AIS, BEM), and systematic grid experiments across architectures and datasets. Emphasis on revealing WHY methods converge in performance.",
      "hypotheses": [
        "CSI predicts when WD scheduling helps (correlation > 0.5)",
        "AIS predicts when alignment-awareness helps (correlation > 0.3)",
        "Methods with low CSI difference perform identically",
        "Diagnostic fingerprints are architecture-specific"
      ],
      "pilot_focus": "Extend iter_003 data with diagnostic logging, compute CSI/AIS across existing 7 methods",
      "strengths": [
        "No deep theorems required -- empirical rigor is the contribution",
        "Unified optimizer is standalone community contribution",
        "Reuses most iter_003 data",
        "Comprehensive 'revisiting' papers publish well when done rigorously"
      ],
      "risks": [
        "Reviewers may dismiss as 'just a benchmark' without theoretical depth",
        "CSI/AIS metrics may not correlate with method performance",
        "Requires ImageNet results for credibility (prior failures)"
      ]
    }
  ],
  "synthesis_reasoning": {
    "landscape_analysis": "All 6 perspectives converge on a key empirical finding: constant WD dominates or ties on CIFAR-10/ResNet-20 (91.22% vs 90.7-91.2% for all dynamic methods). The central question is WHY. The innovator and theoretical perspectives provide the answer through Lyapunov certificates and cumulative alignment bounds. The pragmatist demands practical relevance (unified operator, diagnostic metrics). The contrarian challenges the alignment signal (BN confound, noise). The empiricist demands rigorous falsification tests. The interdisciplinary perspective provides the control-theoretic framing.",
    "conflict_resolution": {
      "innovator_vs_pragmatist": "The innovator's spectral-drift-controlled WD (SD-WD) and entropy-production WD (EP-WD) are theoretically beautiful but practically intractable. Pragmatist's concern is valid. Resolution: spectral dynamics retained as motivational theory (Section 3), not as practical algorithm. Thermodynamic lens dropped from formal claims.",
      "theoretical_vs_contrarian": "The theoretical perspective's cumulative alignment bound assumes alignment is informative. The contrarian argues it is a noisy mirage in BN networks. Resolution: BOTH may be correct in different regimes. The BN ablation experiment distinguishes them. The cumulative bound is theoretically valid regardless of practical alignment informativeness.",
      "empiricist_vs_innovator": "The empiricist demands budget-matched controls and random mask ablations. The innovator proposes PMP-WD as optimal. Resolution: PMP-WD is evaluated with the empiricist's rigorous controls. If PMP-WD wins in budget-matched comparison, both perspectives are validated.",
      "pragmatist_vs_interdisciplinary": "The pragmatist worries gain scheduling adds complexity without benefit. The interdisciplinary perspective claims operating regions are natural. Resolution: operating region analysis is included as diagnostic, not as algorithmic component. If regions emerge naturally, the framing is validated."
    },
    "weighting_rationale": "The theoretical and innovator perspectives received highest weight because the evolution lesson explicitly identifies 'no deeper theoretical results' as the key weakness. The contrarian received high weight because the BN confound and alignment noise concerns are legitimate and must be addressed head-on. The empiricist's rigorous evaluation design was adopted wholesale. The pragmatist's unified operator is the notational backbone. The interdisciplinary perspective's control-theoretic framing informs the Lyapunov certificate but is not the lead contribution."
  }
}


## 上一轮新颖性检查报告（必须针对发现的撞车问题进行修正）
# Novelty Report: Unified Dynamic Weight Decay Framework

**Date**: 2026-03-19
**Checker**: sibyl-novelty-checker

---

## Candidate 1: `cand_lyapunov_unified` (Front-runner)

**Title**: Lyapunov-Certified Dynamic Weight Decay: Unified Convergence Framework with Cumulative Alignment Theory

### Novelty Score: 7/10

**Recommendation**: PROCEED with minor repositioning

### Core Claims and Prior Art Analysis

#### Claim 1: Unified Lyapunov Convergence Certificate for General Time-Varying WD

**Prior work found**:
- **CWD (Li et al., ICLR 2026)**: Proves convergence for *binary* WD via Lyapunov + LaSalle invariance principle. Does NOT handle continuous time-varying schedules. Severity: **partial_overlap**.
- **Sun et al. (CVPR 2025)**: Proves generalization bound for *fixed* WD in nonconvex SGD. Does NOT address time-varying WD. Severity: **related_work**.
- **Kondo & Iiduka (arXiv 2508.03105, 2025)**: Novel Lyapunov function for SGDM with dynamic *learning rate* and *batch size* schedules. Does NOT address weight decay schedules. Severity: **related_work** -- uses similar Lyapunov methodology but for different hyperparameters.
- **Hassan Saoud (arXiv 2510.08259, 2025)**: Composite Lyapunov criteria for nonlinear autonomous systems. Applied to inertial gradient systems and primal-dual flows, NOT to weight decay. Severity: **related_work** -- the mathematical tool the proposal plans to apply.
- **Lion-K (OPT 2025 Workshop)**: Lyapunov analysis for LION optimizer family. Does not address general WD schedules. Severity: **related_work**.

**Assessment**: No prior work provides a Lyapunov convergence certificate for *general time-varying* weight decay schedules with a computable "certified band." CWD handles binary WD only; Sun et al. handle fixed WD only; Kondo & Iiduka handle dynamic LR/batch but not WD. The concept of a "certified band" [lambda_min(t), lambda_max(t)] for WD is novel. **Novelty confirmed.**

**Risk**: The Kondo & Iiduka (2025) paper shows that Lyapunov analysis for dynamic hyperparameter schedules is an active area. Someone could extend their framework to WD before this paper is submitted. Time-sensitivity is moderate.

#### Claim 2: Cumulative Alignment Generalization Bound

**Prior work found**:
- **Sun et al. (CVPR 2025)**: Provides worst-case alignment bound using sup_t delta_t. The proposal claims to improve this with cumulative/trajectory-level analysis. Severity: **partial_overlap** -- direct extension of Sun et al.'s bound.
- **Holzl et al. (NeurIPS 2025)**: Gradient-Weight Alignment (GWA) as train-time proxy for generalization. Empirical study of alignment dynamics, NOT formal generalization bounds. Severity: **related_work**.
- **Trajectory-dependent generalization bounds (IJCAI 2025)**: Trajectory-dependent bounds for pairwise learning with phi-mixing. Different setting (pairwise learning, dependent data), not WD-specific. Severity: **related_work**.
- **Gradient flow + LPK bound (arXiv 2506.11357, 2025)**: Trajectory-based generalization via loss path kernels. Different framework (gradient flow, Rademacher complexity). Severity: **related_work**.

**Assessment**: No prior work provides a cumulative alignment-specific generalization bound for SGD with weight decay. The extension from worst-case to cumulative alignment is a meaningful theoretical contribution. However, it is an incremental (though non-trivial) improvement over Sun et al. **Novelty confirmed, but it is an extension rather than a wholly new direction.**

#### Claim 3: Formal Subsumption of WD Methods as Special Cases

**Prior work found**:
- **Newhouse (MIT Thesis, 2025)**: Unified duality-based framework where SGD, AdamW, and Muon are special cases. Focuses on *optimizers*, not WD *scheduling methods*. Severity: **partial_overlap** -- similar unification spirit but different scope (optimizers vs. WD schedules).
- **Emergent Mind survey on Decoupled WD**: Notes that AdamWN, AdamHD, SPD, CWD can be seen as extensions of decoupled WD framework. Informal unification, not formal subsumption with convergence guarantees. Severity: **related_work**.
- **CWD (ICLR 2026)**: Provides convergence guarantees for CWD specifically, not for a unified family.

**Assessment**: No prior work formally proves that constant WD, CWD, cosine schedule, SWD, and PMP-WD all satisfy a single convergence certificate. The Newhouse thesis unifies *optimizers* (SGD/Adam/Muon) under a duality lens, but does not address WD *scheduling variants*. The subsumption claim is novel. **Novelty confirmed.**

#### Claim 4: PMP-WD Optimality

**Prior work found**:
- **PMP for CNN training (arXiv 2504.11647, 2025)**: Applies PMP to train CNNs with L0 regularization. Different application (training algorithm, not WD scheduling). Severity: **related_work**.
- **"Understanding and Scheduling Weight Decay" (OpenReview)**: Proposes WD schedulers from learning dynamics perspective. Does not derive optimal schedule via PMP. Severity: **related_work**.
- **No paper found** that derives an optimal WD schedule using Pontryagin's Maximum Principle.

**Assessment**: Applying PMP to derive the provably optimal WD schedule within a certified family is novel. **Novelty confirmed.**

#### Claim 5: Diagnostic Metrics (CSI, AIS)

**Prior work found**:
- No prior work defines "Coupling Stability Index" or "Alignment Informativeness Score" in the WD context.
- **Holzl et al. (NeurIPS 2025)**: GWA metric is related but serves a different purpose (early stopping, not WD method selection). Severity: **related_work**.

**Assessment**: The specific diagnostic metrics are novel. **Novelty confirmed.**

### Key Collisions Summary

| Prior Work | Overlap | Severity |
|---|---|---|
| CWD (Li et al., ICLR 2026) | Lyapunov convergence for binary WD; sliding-mode analysis | partial_overlap |
| Sun et al. (CVPR 2025) | Generalization bound with worst-case alignment for fixed WD | partial_overlap |
| Kondo & Iiduka (arXiv 2508.03105) | Lyapunov analysis for dynamic hyperparameter schedules (LR/batch, not WD) | related_work |
| Newhouse (MIT Thesis, 2025) | Duality-based unification of optimizers (not WD schedules) | related_work |
| Defazio (arXiv 2506.02285, 2025) | WD as gradient-to-weight ratio controller in normalized layers | related_work |
| Holzl et al. (NeurIPS 2025) | Gradient-Weight Alignment as generalization proxy | related_work |
| Hassan Saoud (arXiv 2510.08259, 2025) | Composite Lyapunov criteria (mathematical tool, not WD application) | related_work |
| Olsen et al. (ICML 2025) | Spectral dynamics of SGD via Dyson BM | related_work |

### Differentiation Notes

The proposal clearly differentiates from closest work:
1. vs. CWD: Generalizes from binary {0, lambda} to arbitrary time-varying lambda(t)
2. vs. Sun et al.: Extends from worst-case to cumulative alignment; from fixed to time-varying WD
3. vs. Kondo & Iiduka: Addresses WD schedules rather than LR/batch schedules
4. vs. Newhouse: Unifies WD scheduling methods rather than optimizer algorithms
5. vs. Defazio: Provides formal convergence guarantees for the control perspective Defazio describes empirically

### Risks to Novelty

1. **Time sensitivity**: Kondo & Iiduka's Lyapunov framework for dynamic schedules (Aug 2025) shows this is an active area. Their extension to WD is straightforward in principle.
2. **CWD extension**: The CWD authors (ICLR 2026) already have Lyapunov + LaSalle machinery. Extending to continuous schedules is a natural next step for them.
3. **Incremental theory risk**: Reviewers may view the cumulative alignment bound as an incremental extension of Sun et al. rather than a fundamental contribution. The bound's practical utility depends on whether cumulative alignment actually predicts generalization better (H2).

---

## Candidate 2: `cand_alignment_mirage` (Backup)

**Title**: The Alignment Mirage: When Gradient-Weight Geometry Misleads Dynamic Regularization

### Novelty Score: 6/10

**Recommendation**: PROCEED ONLY IF main candidate fails; needs stronger constructive contribution

### Prior Art Analysis

**Key prior work found**:
- **Defazio (2025)**: Already shows that in normalized layers, gradients are orthogonal to weights (alignment = 0 by construction), and WD primarily controls effective LR. This substantially anticipates the "alignment mirage" thesis. Severity: **partial_overlap**.
- **"Three Mechanisms of Weight Decay" (Zhanxing Zhu et al.)**: Identifies three distinct mechanisms of WD depending on architecture (BN presence). Severity: **partial_overlap**.
- **"Why Do We Need Weight Decay in Modern Deep Learning?" (NeurIPS 2024)**: Shows WD's key role under BN is controlling effective LR. Severity: **partial_overlap**.
- **"Weight Rescaling" (arXiv 2102.03497)**: Shows BN renders L2 penalty equivalent to adaptive LR adjustment. Severity: **partial_overlap**.
- **Jane Street Blog on L2 and BN**: Popular exposition of BN + WD interaction making L2 into effective LR control.

**Assessment**: The core insight -- that alignment-based WD under BN reduces to effective LR control -- is partially anticipated by Defazio (2025), the three-mechanisms paper, and the NeurIPS 2024 paper. What would be novel is the *systematic measurement of alignment SNR* and the *head-to-head BN ablation* specifically for alignment-aware WD methods (CWD, SWD). However, this is a narrow empirical contribution without deep theory.

### Differentiation Needed

To differentiate from Defazio (2025) and the NeurIPS 2024 paper, the "alignment mirage" candidate would need:
1. Systematic SNR measurement across multiple alignment-aware methods (not just general WD)
2. Formal characterization of when alignment is vs. is not informative (not just BN vs. non-BN)
3. A constructive alternative that outperforms existing methods

---

## Candidate 3: `cand_diagnostic_benchmark` (Backup)

**Title**: Unifying Dynamic Weight Decay: Comprehensive Diagnostic Benchmark with Standardized Metrics

### Novelty Score: 5/10

**Recommendation**: MODIFY -- merge into main candidate's experimental section rather than standalone paper

### Prior Art Analysis

- **SWD (NeurIPS 2023)**: Already provides systematic comparison of WD methods with gradient-norm perspective.
- **CWD (ICLR 2026)**: Compares CWD against AdamW, Lion, Muon at scale (338M-2B parameters).
- **NeurIPS 2024 "Why WD?"**: Systematic comparison of WD mechanisms across architectures.
- **Defazio (2025)**: Compares gradient-to-weight ratio behavior across methods and layers.

**Assessment**: A standalone "benchmark paper" comparing WD methods would face significant reviewability challenges given that SWD, CWD, and the NeurIPS 2024 paper already provide comprehensive comparisons. The CSI/AIS metrics are novel but without theoretical grounding (which the main candidate provides), they risk being seen as ad-hoc. **Better as the experimental backbone of the main candidate.**

---

## Overall Novelty Assessment: **HIGH**

The front-runner candidate (`cand_lyapunov_unified`) has genuinely novel core contributions:
1. The Lyapunov certificate for general time-varying WD is new (no prior work found)
2. The cumulative alignment bound extends Sun et al. non-trivially
3. The formal subsumption of WD methods under a single certificate is new
4. PMP-WD as an optimal schedule is new

The two backup candidates have weaker novelty due to substantial anticipation by Defazio (2025), the NeurIPS 2024 paper, and CWD (ICLR 2026).

### Recommended Citation Strategy

Must cite as primary related work:
1. CWD (Li et al., ICLR 2026) -- closest methodological predecessor
2. Sun et al. (CVPR 2025) -- direct predecessor for generalization bound
3. SWD (Xie et al., NeurIPS 2023) -- key method in the unified family
4. Defazio (arXiv 2025) -- WD as gradient-to-weight controller
5. Hassan Saoud (arXiv 2025) -- composite Lyapunov tool
6. Kondo & Iiduka (arXiv 2025) -- Lyapunov for dynamic schedules
7. Holzl et al. (NeurIPS 2025) -- gradient-weight alignment
8. Olsen et al. (ICML 2025) -- spectral dynamics
9. Newhouse (MIT Thesis, 2025) -- optimizer unification via duality
10. NeurIPS 2024 "Why WD?" -- three mechanisms under BN
