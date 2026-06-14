

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
**Survey Date**: 2026-04-02 (Refreshed: comprehensive dual-source search arXiv + Web, 55+ core references)
**arXiv Search Keywords**: ["weight decay scheduling dynamic adaptive", "decoupled weight decay AdamW regularization", "weight decay norm alignment gradient neural network deep learning", "cautious weight decay", "weight norm control Loshchilov", "weight decay layer-wise module-wise parameter-specific", "Scion Muon optimizer weight decay", "weight decay orthogonal radial tangential optimizer", "weight decay Vision Transformer ImageNet CIFAR", "understanding scheduling weight decay", "NOVAK unified adaptive optimizer weight decay", "gradient-to-weight ratio weight decay training dynamics", "CVPR 2025 weight decay nonconvex SGD convergence", "spectral dynamics weights rank minimization weight decay", "LLM pretraining WD schedule cosine warmup-stable-decay", "correction decoupled weight decay gamma squared", "independent weight decay learning rate transfer muP", "AlphaDecay module-wise weight decay LLM spectral"]
**Web Search Keywords**: ["weight decay scheduling deep learning state of the art 2025 2026", "decoupled weight decay AdamW regularization survey 2024 2025", "adaptive weight decay neural network training benchmark comparison", "cautious weight decay CWD optimizer 2025 2026 paper", "weight norm control Loshchilov weight decay target norm optimizer", "weight decay deep learning open source implementation benchmark GitHub 2024 2025", "understanding scheduling weight decay Xie NeurIPS 2023 SWD optimizer", "norm-matched weight decay alignment-aware weight decay gradient 2024 2025", "weight decay optimizer survey unified framework SGD Adam 2024 2025 arXiv", "cautious weight decay AlphaDecay module-wise weight decay LLM training 2025", "weight decay interaction batch normalization scale invariance neural network 2024 2025", "weight decay ImageNet ResNet training best practices 2025 github", "cautious weight decay CWD optimizer 2026 arxiv", "AlphaDecay layer-wise weight decay LLM spectral 2025 arxiv", "weight decay optimizer comparison Lion Muon AdamW training dynamics 2025", "correction of decoupled weight decay AdamW fix 2025 arxiv", "weight decay Vision Transformer ViT training recipe 2024 2025", "independent weight decay learning rate transfer muP maximal update parameterization 2025", "dynamic weight decay scheduling alignment-aware norm-matched unified framework 2026"]

## 1. Field Overview

Weight decay (WD) is one of the most universally applied techniques in deep learning optimization, yet the 2023-2026 period has witnessed a profound paradigm shift in understanding its role and designing its application strategy. The classical view treated WD as explicit L2 regularization -- a simple penalty that shrinks weights toward zero. However, a growing body of theoretical and empirical work demonstrates that WD in modern deep learning acts primarily as a **training dynamics modifier**: stabilizing optimization, controlling weight norms, balancing effective learning rates across layers, and interacting with stochastic noise -- rather than as a classical regularizer. D'Angelo et al. (NeurIPS 2024, arXiv:2310.04415) provide a landmark unifying perspective: for both vision models (ResNets) and LLMs, weight decay is never useful as explicit regularization but instead changes the training dynamics in desirable ways via the "loss stabilization mechanism" for SGD and the "bias-variance tradeoff" for near-one-epoch LLM training.

This re-understanding has spawned multiple independent research threads, each proposing a distinct lens for how WD should be dynamically controlled: (1) **WD scheduling** -- adjusting WD strength over the course of training (SWD by Xie et al., log-time WD scheduling by Ferbach et al.); (2) **alignment-aware WD** -- conditioning WD on the geometric relationship between weights and optimizer updates (CWD by Chen et al., AdamO by Chen et al.); (3) **decoupled WD** -- separating WD from gradient scaling in adaptive optimizers (the foundational AdamW by Loshchilov & Hutter, and extensions to Lp norms, Huber decay, differential privacy, and the gamma^2 correction by Chou); (4) **norm-matched WD** -- targeting specific weight norm levels or spectral properties rather than blindly shrinking to zero (Weight Norm Control by Loshchilov, AlphaDecay by He et al.). A fifth emergent thread studies the **implicit structural effects** of WD: inducing low-rank weight matrices (Galanti et al., Kobayashi et al.), low-rank attention layers in transformers, and neural collapse geometry.

Additionally, constraint-based approaches like CPR (Franke et al., NeurIPS 2024) reformulate WD as per-parameter-matrix constraint optimization via augmented Lagrangian methods, achieving dynamic and individual adaptation without explicit penalty coefficients. The duality perspective (Newhouse, MIT 2025) offers yet another unifying lens, connecting SGD, Adam, and Shampoo through norm-dependent duality maps that inherently shape how WD operates in each optimizer family.

The 2025-2026 landscape has also seen a major development at the intersection of WD and model scaling: Kosson et al. (NeurIPS 2025, arXiv:2510.19093) demonstrated that **independent weight decay** (decaying by factor 1-lambda rather than 1-eta*lambda) is not merely a helpful addition but a **central mechanism for enabling learning rate transfer** across model widths. Their work fundamentally challenges the role of muP (Maximal Update Parameterization), showing that muP's scaling rules break down after early training and that weight decay is what actually stabilizes update dynamics across scales. This finding has been independently confirmed by multiple groups (Wortsman et al., Blake et al., Wang & Aitchison, Bergsma et al.) and is now incorporated into the ICLR 2025 u-muP framework.

The optimizer landscape has also evolved significantly. While AdamW remains the dominant default (used in GPT-2/3, BERT, and essentially all major LLMs), 2025 saw Muon emerge as the strongest challenger, offering meaningful speedups especially at sub-billion parameter scales, with weight decay being critical for its stability and scalability. A rigorous benchmark of eleven optimizers (arXiv:2509.02046, "Fantastic Pretraining Optimizers") revealed that reported speedups are often overstated against weak baselines -- against well-tuned AdamW, alternative optimizers do not exceed 1.4x speedup. Critically, all the fastest optimizers (Muon, Soap, Kron, Scion) use matrix-based preconditioners, and weight decay interacts differently with these non-Euclidean update rules. Chen et al. (arXiv:2506.15054) showed that Muon with decoupled WD implicitly constrains spectral norm, connecting to the Lion-K family.

Importantly, the interaction between WD and training context is increasingly recognized: Steiner et al. (TMLR 2022) showed that WD in ViT training interacts strongly with data augmentation and dataset size -- careful WD-augmentation co-design can substitute 10x more data. Zhang et al. (arXiv:1810.12281) identified three distinct mechanisms through which WD improves generalization, demonstrating that WD consistently outperforms L2 regularization across SGD, Adam, and K-FAC. Recent work on gradient-weight alignment (Holzl et al., arXiv:2510.25480) and gradient alignment-based LR adaptation (GALA, OpenReview 2025) further demonstrates the potential for alignment-aware dynamic scheduling strategies. The interaction between WD and batch normalization has also received renewed attention: Li & Arora showed that scale invariance from batch normalization reduces WD's effect to merely an effective learning rate scheduler, while competing forces between WD (norm-decreasing) and loss gradients (norm-increasing in scale-invariant layers) create periodic training dynamics. Weight Rescaling (WRS, arXiv:2102.03497) proposes explicit norm rescaling as an alternative.

The practical importance of weight decay configuration is further underscored by the dramatic performance differences observed in ImageNet training recipes. PyTorch's updated ResNet-101 recipe reached 81.9% accuracy (vs. 77.4% with original settings) -- a 4.5% improvement from purely algorithmic changes including decoupled weight decay. The "ResNet Strikes Back" (RSB) work showed that weight decay values of 0.02-0.03 with LAMB/AdamW can push ResNet-50 to ~80.4% on ImageNet, while Google's "Revisiting ResNets" (ResNet-RS) found that *decreasing* weight decay when using stronger regularization improved top-1 accuracy. Importantly, recent analysis by Schaipp (2024) revealed that PyTorch's AdamW implementation does not fully decouple weight decay from learning rate in practice -- when doubling LR, WD should be halved -- highlighting the practical pitfalls of the current decoupled WD paradigm. The new "Correction of Decoupled Weight Decay" work (Chou, arXiv:2512.08217) proposes that the correct scaling is WD proportional to LR^2 for stable weight norms.

For Vision Transformers specifically, weight decay values in practice range from 0.0001 (small-scale) to 0.3 (large-scale ImageNet with NVIDIA NeMo recipes), with the DeiT III meta-recipe emphasizing WD-augmentation co-tuning. A notable finding from "Scaling Vision Transformers" is that different WD strengths benefit the head (strong decay) vs. body (weak decay), suggesting layer/module-specific WD is important beyond LLMs.

Despite this rich landscape, no existing work provides a **unified theoretical framework** that encompasses all these approaches, reveals their mathematical connections, and offers standardized metrics for comparing them. Each approach addresses a different facet of the same underlying question -- "how should weight decay interact with the training trajectory?" -- but uses different assumptions, different formulations, and different evaluation criteria. This is the gap our proposed Unified Dynamic Weight Decay Framework aims to fill.

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | **Decoupled Weight Decay Regularization** (Loshchilov & Hutter) | ICLR 2019 | 2019 | Foundational: showed L2 regularization != WD in adaptive optimizers; proposed AdamW; 10,000+ citations | Fixed WD coefficient; no scheduling or alignment awareness |
| 2 | **Why Do We Need Weight Decay in Modern Deep Learning?** (D'Angelo et al.) | NeurIPS 2024 / arXiv:2310.04415 | 2023/2024 | Unifying perspective: WD as dynamics modifier (loss stabilization for SGD, bias-variance tradeoff for LLMs); WD prevents bfloat16 loss divergence | Empirical focus; no formal convergence rate improvement from dynamic WD |
| 3 | **Cautious Weight Decay (CWD)** (Chen et al.) | ICLR 2026 / arXiv:2510.12402 | 2025/2026 | Sign-alignment-based selective decay; bilevel Pareto-optimal interpretation; sliding-mode behavior; one-line drop-in for AdamW/Lion/Muon; published ICLR 2026 | Binary sign alignment only; no continuous modulation; no cumulative alignment theory |
| 4 | **Weight Norm Control (AdamWN)** (Loshchilov) | arXiv:2311.11446 | 2023 | Generalizes decoupled WD to target-norm control (target=0 is standard WD); any training run can be challenged by AdamWN with scheduled target norm | Fixed target norm; no gradient-alignment sensitivity; limited experimental validation |
| 5 | **AlphaDecay: Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs** (He et al.) | NeurIPS 2025 / arXiv:2506.14562 | 2025 | Spectral-density-guided module-wise decay using HT-SR theory; scales 60M-1B params; best on 6/7 zero-shot benchmarks; code: github.com/hed-ucas/AlphaDecay | Heuristic decay assignment; no per-iteration adaptation; LLM-specific |
| 6 | **Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks** (Kosson et al.) | arXiv:2305.17212 | 2023 | WD induces rotational equilibrium; balanced average rotation across layers/neurons; explains AdamW > Adam+L2 | Angular dynamics focus; does not formalize alignment-based scheduling |
| 7 | **On the Overlooked Pitfalls of Weight Decay (SWD)** (Xie et al.) | arXiv:2011.11152 / NeurIPS 2023 (OpenReview) | 2020/2023 | First practical WD scheduler; gradient-norm-aware dynamic WD; closes SGD-Adam generalization gap on CIFAR | Early work; limited theoretical foundation; one scheduling heuristic |
| 8 | **Correction of Decoupled Weight Decay** (Chou) | arXiv:2512.08217 | 2025/2026 | Derives WD proportional to gamma^2 for stable weight norm; Total Update Contribution (TUC) analysis; validates on Scion optimizer (ScionC); stable weight and gradient norms for AdamC and ScionC | Focuses on scaling rules, not alignment-based adaptation |
| 9 | **Decoupled Orthogonal Dynamics (AdamO)** (Chen, Yuan, Zhang) | arXiv:2602.05136 | 2026 | Identifies "Radial Tug-of-War" conflict between WD and gradient; decouples radial (norm) and tangential (direction) dynamics; SGD-style norm control + Adam tangential | New (Feb 2026); limited large-scale validation; complex implementation |
| 10 | **Adam-family Methods with Decoupled Weight Decay** (Ding et al.) | arXiv:2310.08858 | 2023 | Convergence framework for Adam-family with decoupled WD; shows framework asymptotically approximates SGD; proposes AdamD | Non-adaptive WD; fixed decay coefficient |
| 11 | **Rethinking Weight Decay for Robust Fine-Tuning (SPD)** (Tian et al.) | arXiv:2411.01713 | 2024 | Selective Projection Decay: layer-wise penalty based on loss reduction consistency; expands/contracts per-layer search space | Fine-tuning focused; no from-scratch convergence theory |
| 12 | **Implicit Bias of AdamW: l_inf Norm Constrained Optimization** (Xie & Li) | arXiv:2404.04454 | 2024 | AdamW implicitly performs l_inf constrained optimization; connects to Frank-Wolfe algorithm | Full-batch setting only; does not extend to dynamic WD |
| 13 | **Decoupled Weight Decay for Any p Norm** (Outmezguine & Levi) | arXiv:2404.10824 | 2024 | Generalizes decoupled WD to Lp norms; enables sparsification via p<1 without gradient divergence | Fixed decay schedule; no alignment or scheduling awareness |
| 14 | **AdamHD: Decoupled Huber Decay** (Guo & Fan) | arXiv:2511.14721 | 2025 | Replaces L2 penalty with Huber regularizer; bounded decay gradients; 10-15% faster convergence; sparser weights | New penalty form but still non-adaptive scheduling; GPT-focused |
| 15 | **How to set AdamW's weight decay as you scale** (Wang & Aitchison) | arXiv:2405.13698 | 2024 | WD as EMA timescale; optimal timescale constant in epochs across model/dataset scales; muP-WD interaction | Provides scaling intuition but no alignment-aware adaptation |
| 16 | **Weight Decay may matter more than muP for LR Transfer** (Kosson et al.) | NeurIPS 2025 / arXiv:2510.19093 | 2025 | Independent WD (1-lambda not 1-eta*lambda) central for LR transfer; muP assumptions break down after early training; challenges muP; unifying framework for WD and muP based on relative updates | Empirical; no theoretical framework for dynamic WD |
| 17 | **Logarithmic-time Schedules (ADANA)** (Ferbach et al.) | arXiv:2602.05298 | 2026 | Log-time schedules for beta1, beta2, and WD; logarithmic WD alone yields significant improvements; 40% compute efficiency gain | Complex scheduling; optimizer-specific (AdamW/AdEMAMix variants) |
| 18 | **Weight Decay Improves Language Model Plasticity** (Han et al.) | arXiv:2602.11137 | 2026 | Larger WD during pretraining produces more plastic (adaptable) models; WD encourages linearly separable representations | LLM-specific; plasticity not directly connected to training convergence |
| 19 | **OUI: Overfitting-Underfitting Indicator for WD selection** (Fernandez-Hernandez et al.) | arXiv:2504.17160 | 2025 | Novel OUI metric for monitoring WD quality during training; converges faster than traditional metrics; validation-free WD selection | Diagnostic tool only; does not propose adaptive WD algorithm |
| 20 | **SGD and Weight Decay Secretly Minimize Rank** (Galanti et al.) | arXiv:2206.05794 | 2022 | SGD + WD induces low-rank bias in weight matrices; stronger with smaller batch, higher LR, or stronger WD | No connection to dynamic WD; static analysis |
| 21 | **Weight decay induces low-rank attention layers** (Kobayashi et al.) | arXiv:2410.23819 | 2024 | L2 regularization on multiplicative parameters (attention layers) equivalent to nuclear norm; can damage LM performance | Argues for decoupling WD in attention layers from rest; structural insight |
| 22 | **Low-rank bias, weight decay, and model merging** (Kuzborskij & Abbasi-Yadkori) | arXiv:2502.17340 | 2025 | L2 regularization induces parameter-gradient alignment, norm preservation across layers, low-rank bias at stationary points | Static analysis at stationary points; no trajectory-level alignment theory |
| 23 | **PathProx: Proximal Gradient for Weight Decay** (Yang et al.) | arXiv:2210.03069 | 2022 | For ReLU networks, WD objective equivalent to sum of L2 (not squared) norms per neuron; novel proximal gradient algorithm | ReLU-specific; different perspective on WD objective |
| 24 | **Tune without Validation (Twin)** (Brigato & Mougiakakou) | arXiv:2403.05532 | 2024 | Pipeline for tuning LR and WD without validation sets; weight norm strongly correlates with generalization | Tuning method, not adaptive WD; 20 datasets evaluated |
| 25 | **What do near-optimal learning rate schedules look like?** (Naganuma et al.) | arXiv:2603.10301 | 2026 | Most comprehensive study of optimal LR schedule shapes; weight decay has strong effect on optimal schedule shape | LR-focused but reveals critical WD-schedule interaction |
| 26 | **Norm-Hierarchy Transitions in Representation Learning** (Truong & Truong) | arXiv:2603.07323 | 2026 | WD traverses norm hierarchy from shortcut to structured representations; transition delay logarithmic in norm ratio | Theoretical insight into WD dynamics; does not propose adaptive WD |
| 27 | **Muon Optimizes Under Spectral Norm Constraints** (Chen, Li, Liu) | arXiv:2506.15054 | 2025 | Muon with decoupled WD implicitly constrains spectral norm; connects to Lion-K family | Characterizes WD's implicit effect in non-Euclidean optimizers |
| 28 | **Preconditioning for Optimization and Regularization** (Ye) | arXiv:2410.00232 | 2024 | Unified framework: AdamW selects intrinsic parameters for regularization; derives L1-regularization analogue; explains normalization methods | Theoretical framework that touches on unified perspective |
| 29 | **Why Gradients Rapidly Increase Near the End of Training** (Defazio) | arXiv:2506.02285 | 2025 | WD controls gradient-to-weight ratio ||g||/||w||; all normalized layers converge to the same steady-state ratio ("layer balancing"); explains Adam vs AdamW gap; proposes corrective term for LR-schedule interaction | LLM-specific analysis; largely ignores alignment-aware WD |
| 30 | **NOVAK: Unified Adaptive Optimizer for Deep Neural Networks** (Kavun) | arXiv:2601.07876 | 2026 | Integrates adaptive moment estimation, rectified LR scheduling, decoupled WD, Nesterov momentum, and lookahead into one optimizer; shows coupling WD with alpha_eff (not alpha) degrades generalization 4-8pp on CIFAR-100; SOTA vs 14 optimizers on CIFAR/ImageNet | Does not address alignment-aware or scheduled WD; focused on integration rather than theory |
| 31 | **Investigating the Role of Weight Decay in Enhancing Nonconvex SGD** (Sun et al.) | CVPR 2025 | 2025 | First theoretical proof of WD's generalization benefit in nonconvex SGD; proves WD does NOT accelerate convergence but improves generalization; extends to sign-based methods (SignSGD) | Theoretical framework limited to SGD; no analysis of alignment or scheduling strategies |
| 32 | **Approaching Deep Learning through the Spectral Dynamics of Weights** (Yunis et al.) | arXiv:2408.11804 | 2024 | Spectral dynamics (singular value evolution) as a unifying lens; WD promotes rank minimization across architectures; spectral dynamics distinguish memorizing from generalizing networks; connects to lottery tickets and linear mode connectivity | Empirical framework; does not propose adaptive WD algorithm |
| 33 | **Optimal LR Schedules under Functional Scaling Laws** (Ferbach et al.) | arXiv:2602.06797 | 2026 | Sharp phase transition: easy tasks -> power decay to zero; hard tasks -> WSD (Warmup-Stable-Decay); provides principled evaluation of cosine/linear/WSD schedules; WD-LR schedule co-design framework | LR-schedule focused; WD treated as fixed hyperparameter, not scheduled |
| 34 | **Benchmarking Optimizers for LLM Pretraining** | arXiv:2509.01440 | 2025 | Demonstrates WD's importance across optimizers; high WD with constant LR outperforms AdamW without WD on short horizons; Signum/Lion with high WD beats AdamW+no-WD | Empirical only; no theoretical analysis of WD mechanisms |
| 35 | **AdaDecay: Adaptive Weight Decay for Deep Neural Networks** (Nakamura & Hong) | arXiv:1907.08931 | 2019 | Per-parameter WD via sigmoid of layer-normalized gradient norms; improves generalization on MNIST/CIFAR; no extra backprop cost | Small-scale benchmarks only; outperformed by CPR/AdamW on GPT-2 setting per AlphaDecay 2025 comparison |
| 36 | **Improving Robustness with Adaptive Weight Decay** (Apple / NeurIPS 2023) | NeurIPS 2023 | 2023 | Auto-tunes WD per iteration for adversarial robustness; 20% relative robustness improvement on CIFAR-100; smaller weight norms; less LR sensitivity | Focused on adversarial robustness; not general training dynamics |
| 37 | **Is your batch size the problem? Revisiting the Adam-SGD gap in language modeling** | arXiv:2506.12543 | 2025 | Shows Adam-SGD gap is largely a batch size / gradient noise phenomenon; critical batch size differs per optimizer; reframes WD's role in optimizer comparison | Does not directly address dynamic WD; provides important context for WD-optimizer interaction |
| 38 | **Normalize-and-Project (NaP)** | NeurIPS 2024 | 2024 | Principled alternative to WD for norm growth control; flexible recipe applicable to any architecture; avoids WD over/under-regularization tradeoff | Alternative to WD rather than a WD method; relevant as comparison baseline |
| 39 | **Improving Deep Learning Optimization through Constrained Parameter Regularization (CPR)** (Franke et al.) | NeurIPS 2024 / arXiv:2311.09058 | 2024 | Replaces uniform WD with per-parameter-matrix upper-bound constraints via augmented Lagrangian; only 2 hyperparams; AdamCPR outperforms AdamW on CIFAR-100, ImageNet, GPT-2; dynamic and individual adaptation per matrix | Constraint-based formulation; does not address alignment or scheduling; limited to L2-norm constraint |
| 40 | **Duality, Weight Decay, and Metrized Deep Learning** (Newhouse) | MIT MEng Thesis 2025 | 2025 | Unifies SGD, Adam, Shampoo via duality maps under different norms; connects WD to norm geometry and Muon optimizer; principled framework for optimizer design through dual-vector perspective | Thesis-level work; limited experimental validation; no dynamic WD proposal |
| 41 | **Three Mechanisms of Weight Decay Regularization** (Zhang et al.) | arXiv:1810.12281 | 2018 | Identifies three distinct mechanisms of WD (regularization, convergence, dynamics); WD outperforms L2 for K-FAC; closes generalization gap between 1st and 2nd order optimizers | Pre-dynamic-WD era; no scheduling or alignment awareness |
| 42 | **Layer-wise Pre-weight Decay (LPWD)** (OpenReview submission) | OpenReview / ICLR 2024 | 2024 | Reveals delay defect in standard WD (weights approach negative update, not zero); proposes pre-weight decay to fix timing; layer-wise variant robust to decay rate | Under review; limited scale; no alignment-aware dimension |
| 43 | **How to Train Your ViT? Data, Augmentation, and Regularization** (Steiner et al.) | TMLR 2022 | 2022 | Most comprehensive ablation of WD in ViT training; AugReg can substitute 10x dataset; WD interacts strongly with augmentation and dataset size | Fixed WD; no dynamic scheduling; ViT-specific |
| 44 | **AdamHuberDecay (AdamHD)** (Guo & Fan) | arXiv:2511.14721 | 2025 | Replaces L2 penalty with smooth Huber regularizer; decays quadratically below threshold, linearly above; 10-15% faster convergence on GPT-2/GPT-3 | Already in #14 -- cross-reference for completeness |
| 45 | **GALA: Gradient Alignment-based Learning Rate Adaptation** | OpenReview 2025 | 2025 | Dynamically adjusts LR by tracking alignment between consecutive gradients; FTRL-based adaptive schedule; increases LR when gradients aligned, decreases otherwise | LR-focused but alignment concept directly applicable to WD scheduling |
| 46 | **Gradient-Weight Alignment (GWA) as Train-Time Proxy** (Holzl et al.) | arXiv:2510.25480 | 2025 | Quantifies coherence between per-sample gradients and model weights; effective learning = coherent alignment; predicts optimal early stopping; identifies influential samples | Diagnostic metric; not applied to WD modulation but directly relevant to alignment-aware WD |
| 47 | **A Light Recipe to Train Robust ViTs** (Debenedetti et al.) | arXiv:2209.07399 | 2022 | WD=0.5 (10x standard) dramatically improves adversarial robustness; larger WD reduces generalization gap for robust accuracy; weak augmentation + strong WD outperforms strong augmentation + standard WD | Adversarial robustness focus; validates extreme WD regime as legitimate strategy |
| 48 | **DP-AdamW: Investigating Decoupled Weight Decay and Bias Correction in Private Deep Learning** | ICML 2025 / arXiv:2511.07843 | 2025 | Differentially private variant of AdamW with bias correction for second moment; outperforms DP-SGD, DP-Adam, DP-AdamBC: 15%+ on text, 5% on image, 1% on graph node classification | Privacy-focused; no dynamic or alignment-aware WD; extends decoupled WD to DP setting |
| 49 | **PyTorch AdamW Implementation Subtlety** (Schaipp) | Blog / Feb 2024 | 2024 | Reveals that PyTorch's AdamW does not fully decouple weight decay and learning rate in practice; when doubling LR, WD should be halved; affects hyperparameter tuning and scaling rules | Blog post, not peer-reviewed; practical insight for implementation correctness |
| 50 | **Weight Decay with Tailored Adam on Scale-Invariant Weights for Better Generalization** | IEEE TNNLS | 2024 | For Adam, weight norm increases fast during training, adverse to WD; introduces regularization on adaptive LR + first moment on WD; improves test accuracy by 0.84% (CIFAR-10) and 1.03% (CIFAR-100) with ResNet-50 | Focused on Adam's scale-invariance issue; no alignment-aware dimension |
| 51 | **Fantastic Pretraining Optimizers and Where to Find Them** | arXiv:2509.02046 | 2025 | Rigorous benchmark of 11 optimizers; speedups of new optimizers limited to 1.4x vs well-tuned AdamW; matrix-based optimizers (Muon, Soap, Kron, Scion) consistently outperform scalar-based; WD tuning critical for each optimizer | Benchmarking focus; no WD theory; but essential context for WD-optimizer interactions |
| 52 | **Muon is Scalable for LLM Training** (Moonlight) | arXiv:2502.16982 | 2025 | Key techniques for scaling Muon: adding WD + adjusting per-parameter update scale; WD critical for Muon stability; Muon achieves ~2x compute efficiency vs AdamW; Muon can reuse AdamW's LR and WD | Muon-specific; no WD theory |
| 53 | **On the Periodic Behavior of Neural Network Training with BN and WD** | NeurIPS 2021 | 2021 | WD and BN create competing forces on weight norm (WD decreases, loss gradients increase via scale invariance); creates periodic training dynamics; reveals fundamental WD-normalization interaction | Specific to BN; does not address other normalization types |
| 54 | **Weight Rescaling: Effective and Robust Regularization for DNNs with Batch Normalization** | arXiv:2102.03497 | 2021 | Proposes explicit norm rescaling to unit norm as alternative to WD for scale-invariant networks; prevents large gradients while ensuring sufficient ELR | Alternative to WD; relevant as baseline |
| 55 | **Wide Neural Networks Trained with Weight Decay Provably Exhibit Neural Collapse** | arXiv:2410.04887 | 2024 | WD + gradient descent provably induces neural collapse geometry (balancedness, alignment with weight matrices) in wide networks | Theoretical; specific to wide networks and last-layer geometry |
| 56 | **Rank Minimization, Alignment and Weight Decay** | ICML 2024 | 2024 | Empirically studies singular value evolution across CNNs, LSTMs, Transformers; large singular values grow faster (decreasing effective ranks); WD promotes both rank minimization and alignment between neighboring layers' top singular vectors | Empirical; no dynamic WD proposal; important evidence for spectral-WD connection |

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
| Layer-wise Pre-weight Decay (LPWD) | Fix delay defect by advancing decay before update; layer-wise rate adjustment | OpenReview 2024 |

**B. Alignment-Aware WD (Geometry-sensitive)**
| Method | Mechanism | Venue |
|--------|-----------|-------|
| Cautious Weight Decay (CWD) | Binary sign-alignment mask; decay only when sign(w) matches sign(update); bilevel optimization interpretation | ICLR 2026 |
| AdamO (Orthogonal Dynamics) | Decouple radial (norm) and tangential (direction) dynamics; eliminate Radial Tug-of-War | arXiv 2026 |
| Selective Projection Decay (SPD) | Layer-wise penalty modulated by loss reduction consistency | arXiv 2024 |
| Gradient-Weight Alignment (GWA) | Per-sample gradient-weight coherence as proxy for generalization | arXiv 2025 |
| GALA (Gradient Alignment LR Adaptation) | Alignment between consecutive gradients drives adaptive scheduling | OpenReview 2025 |

**C. Decoupled WD (Structural separation)**
| Method | Mechanism | Venue |
|--------|-----------|-------|
| AdamW | Decouple WD from gradient scaling in Adam | ICLR 2019 |
| AdamD | Framework for Adam-family with decoupled WD; convergence guarantees | arXiv 2023 |
| Lp-norm decoupled WD | Generalize to any p-norm; sparsification via p<1 | arXiv 2024 |
| AdamHD (Huber Decay) | Replace L2 with Huber penalty; bounded gradients, sparser weights | arXiv 2025 |
| DP-AdamW | Differentially private variant with decoupled WD; 15%+ gains on text classification over DP-SGD | ICML 2025 |
| Tailored Adam | Regularize adaptive LR to be WD-friendly; first moment on WD term; +0.84/1.03% on CIFAR-10/100 | IEEE TNNLS 2024 |
| Corrected Decoupled WD (AdamC/ScionC) | WD proportional to LR^2 for stable weight/gradient norms; TUC analysis | arXiv 2025/2026 |
| Independent WD | Decay by 1-lambda (not 1-eta*lambda); critical for LR transfer across widths | NeurIPS 2025 |

**D. Norm-Matched WD (Target-aware)**
| Method | Mechanism | Venue |
|--------|-----------|-------|
| Weight Norm Control (AdamWN) | Target arbitrary weight norm instead of zero | arXiv 2023 |
| AlphaDecay | Module-wise decay guided by spectral heavy-tailedness (HT-SR) | NeurIPS 2025 |
| gamma^2 scaling | WD proportional to LR^2 for stable weight norm | arXiv 2025 |
| EMA timescale | Optimal WD derived from EMA timescale constant across scales | arXiv 2024 |
| CPR (Constrained Parameter Regularization) | Per-parameter-matrix upper-bound constraint via augmented Lagrangian; dynamic per-matrix adaptation | NeurIPS 2024 |

**E. Structural / Implicit Effects of WD**
| Method/Finding | Mechanism | Venue |
|--------|-----------|-------|
| Rank minimization via WD | SGD + WD induces low-rank bias in weight matrices | arXiv 2022 |
| Low-rank attention via WD | L2 on multiplicative params equivalent to nuclear norm | arXiv 2024 |
| Neural collapse via WD | WD + GD provably induces neural collapse geometry | arXiv 2024 |
| Layer balancing | WD drives ||g||/||w|| ratio to steady state across normalized layers | arXiv 2025 |
| Norm-hierarchy transitions | WD traverses from shortcut to structured representations | arXiv 2026 |
| Weight Rescaling (WRS) | Explicit norm rescaling to unit as WD alternative for scale-invariant networks | arXiv 2021 |

### Benchmarks and Evaluation Practices
- **Vision**: CIFAR-10/100 (ResNet-20, VGG-16-BN), ImageNet (ResNet-50, ResNet-101, ViT-S/B/L, DenseNet-BC-100, EfficientNet-B0)
- **Language**: LLM pre-training (60M-2.6B params on OpenWebText, C4, Wikitext-103, Minipile); GPT-2/GPT-3 architectures; Gemma-based models (338M, 986M, 2B)
- **Common metrics**: Test accuracy, validation loss/perplexity, convergence speed (wall-clock and iteration), gradient norm stability, weight norm trajectory
- **Theoretical metrics**: Convergence rate to epsilon-stationary point, generalization bound gap
- **Emerging metrics** (not yet standardized): Weight norm stability across training, spectral density evolution, alignment cosine similarity trajectories, OUI indicator values, gradient-to-weight ratio ||g||/||w|| stability

### ImageNet Training Recipe Benchmark (Weight Decay Impact)

The following table summarizes the impact of weight decay configuration on ImageNet top-1 accuracy across prominent training recipes, providing concrete evidence for the importance of proper WD design:

| Recipe | Architecture | Weight Decay Config | Top-1 | Key WD Insight |
|--------|-------------|-------------------|-------|----------------|
| Original ResNet (2015) | ResNet-50 | 1e-4 (SGD+L2) | ~76% | Standard L2 regularization |
| Bag of Tricks (ResNet-D) | ResNet-50 | Selective (no WD on BN) | ~79% | Removing WD from BN layers improves accuracy |
| RSB A2 (timm) | ResNet-50 | 0.02-0.03 (LAMB/AdamW) | ~79.8% | Decoupled WD with higher coefficient |
| RSB A1 (timm) | ResNet-50 | 0.02-0.03 + more reg | ~80.4% | WD must be co-tuned with augmentation |
| Mosaic "Hot" | ResNet-50 | Decoupled WD (AdamW) | ~80% | 1.77x faster than timm A2 baseline |
| ResNet-RS (Google) | ResNet-50 | Reduced WD + heavy reg | ~82%+ | *Decreasing* WD with stronger regularization improves accuracy |
| CWD (ICLR 2026) | ViT-S/16 | Cautious WD (AdamW) | 79.45% | +0.61% over standard AdamW (78.84%) |
| CWD (ICLR 2026) | ViT-S/16 | Cautious WD (Muon) | 79.91% | +0.56% over standard Muon (79.35%) |
| NVIDIA NeMo ViT-B/16 | ViT-B/16 | WD=0.3 (Adam, 300 ep) | standard | Very high WD for large-scale ViT training |
| Scaling ViTs (Google) | ViT-L+ | Strong head WD + weak body WD | SOTA | Differential WD: strong for head, weak for body |
| DeiT III (Meta) | ViT-B | WD co-tuned with augmentation | 85.7% | Fine-tuned from IN-21K; WD-augmentation recipe critical |
| FFCV fast training | ResNet-50 | Standard SGD WD (1e-4) | ~80% | Progressive resizing + standard WD on non-BN params |

These results demonstrate that WD configuration alone can account for 4-6% accuracy differences on ImageNet, making it one of the most impactful hyperparameters in modern training recipes.

### Optimizer Landscape and WD Interaction (2025-2026)

| Optimizer | WD Best Practice | Key Finding | Source |
|-----------|-----------------|-------------|--------|
| AdamW | WD=1e-2 (standard) | Still dominant default; well-understood | Loshchilov 2019 |
| Lion | WD 3-10x higher than AdamW; LR 3-10x smaller | Sign-based update; WD*LR product matters | Google 2023 |
| Muon | Reuse AdamW's WD; critical for stability | WD + per-param update scale enable Muon scaling | Moonlight 2025 |
| Scion/ScionC | WD prop. to LR^2 for stability | Corrected WD yields stable norms | Chou 2025 |
| Soap/Kron | Matrix-based preconditioners; WD interacts non-trivially | Fastest optimizers all use matrix preconditioners | arXiv:2509.02046 |
| SGD+WD | WD=1e-4 (ImageNet standard) | WD does NOT accelerate convergence but improves generalization | Sun CVPR 2025 |

### Key Observation: No Standardized Evaluation Framework
Each paper uses a different combination of metrics, making direct comparison difficult. For example:
- CWD reports final loss/accuracy improvements
- AlphaDecay uses perplexity improvements and spectral density analysis
- AdamO reports generalization and stability improvements
- SWD focuses on gradient norm and generalization gap
- Wang & Aitchison focus on optimal WD scaling rules
- Kosson et al. focus on LR transfer via relative update analysis
- NOVAK (arXiv:2601.07876) compares across 14 optimizers on CIFAR/ImageNet but does not control for WD coupling design
- Defazio (arXiv:2506.02285) focuses on gradient-to-weight ratio dynamics but does not compare against alignment-aware methods
- Sun et al. (CVPR 2025) provides convergence theory for SGDW but lacks per-layer or alignment-aware analysis
- "Fantastic Pretraining Optimizers" benchmarks 11 optimizers but treats WD as a fixed hyperparameter per optimizer

**This fragmentation motivates our proposed standardized metrics**: Budget Equivalence Metric (normalizing compute), Coupling Stability Index (measuring WD-optimization interaction stability), and Alignment Informativeness Score (quantifying how much the alignment signal improves WD decisions).

## 4. Identified Research Gaps

- **Gap 1: No unified theoretical framework.** The four WD sub-fields (scheduling, alignment-aware, decoupled, norm-matched) each have independent theoretical justifications but no unifying mathematical framework. For example, CWD's bilevel Pareto-optimality interpretation, AdamWN's target-norm control, the gamma^2 stable-norm scaling, and independent WD for LR transfer all address weight norm dynamics but from incompatible formulations. A unified framework could reveal that these are special cases of a single optimization principle.

- **Gap 2: No standardized evaluation metrics for comparing dynamic WD methods.** Each paper uses different benchmarks, different models, different metrics, and different hyperparameter selection protocols. There is no "Budget Equivalence Metric" that normalizes comparison across different compute budgets, no "Coupling Stability Index" that measures how WD interacts with optimizer state stability, and no "Alignment Informativeness Score" that quantifies the utility of geometric alignment signals for WD decisions.

- **Gap 3: Continuous alignment modulation is unexplored.** CWD (ICLR 2026) uses binary sign alignment (decay or not). AdamO (2026) separates radial/tangential components but uses fixed radial step sizing rules. No method provides a continuous, gradient-magnitude-aware alignment score that smoothly modulates decay strength along the full spectrum from fully aligned to fully opposed.

- **Gap 4: Mathematical connections between sub-approaches are uncharacterized.** It is unclear whether WD scheduling (time-varying lambda(t)) can be derived from an alignment-aware principle, or whether norm-matched WD (target tau) is equivalent to a particular scheduling strategy. The relationship between decoupled WD in adaptive optimizers and alignment-aware WD in SGD remains unformalized.

- **Gap 5: Interaction between dynamic WD and modern optimizer innovations (Muon, Scion, Lion).** The Muon/Scion family uses non-Euclidean LMO-based updates where WD's implicit effect is spectral norm constraining (Chen et al. 2025). How alignment-aware or scheduled WD should be adapted for these emerging optimizers is unknown. The "Fantastic Pretraining Optimizers" benchmark shows each optimizer needs very different WD settings, yet no principled method exists to determine these.

- **Gap 6: Systematic visualization and diagnostic tools.** While individual papers provide ad hoc visualizations (weight norm trajectories, gradient norm plots, spectral density histograms), there is no systematic visualization framework that reveals the core problems across all WD methods and guides practitioners toward optimal strategies.

- **Gap 7: Scale-dependent behavior is poorly understood.** Wang & Aitchison (2024) show optimal WD scales with model and dataset size via EMA timescale. Kosson et al. (2025) show WD matters more than muP for LR transfer at scale. But how alignment-aware or scheduled WD should scale is completely unknown. The disconnect between independent WD (needed for scale transfer) and alignment-aware WD (needed for optimization quality) is not addressed by any existing work.

- **Gap 8: Gradient-to-weight ratio as unifying lens is unexploited.** Defazio (arXiv:2506.02285) shows that WD drives the gradient-to-weight ratio ||g||/||w|| of all normalized layers to the same steady-state, providing a clean theoretical explanation for the Adam vs AdamW gap. However, this "layer balancing" insight has not been connected to alignment-aware or scheduled WD methods. A unified framework could formalize how each WD sub-approach affects the gradient-to-weight ratio trajectory.

- **Gap 9: Convergence theory for nonconvex settings is nascent.** Sun et al. (CVPR 2025) provide the first proof of WD's generalization benefit in nonconvex SGD, but show WD does NOT accelerate convergence. How dynamic WD (scheduling or alignment-aware) might achieve faster convergence in nonconvex settings remains an open theoretical question.

- **Gap 10: Spectral dynamics and rank minimization are not integrated into WD scheduling.** Yunis et al. (arXiv:2408.11804) and ICML 2024 work show WD promotes rank minimization via spectral dynamics and alignment between neighboring layers' top singular vectors. This could inspire rank-aware WD scheduling (reduce WD when desired rank is approached, increase WD to accelerate rank compression). No existing method uses spectral rank as a feedback signal for WD scheduling.

- **Gap 11: WD-normalization interaction not formalized for dynamic WD.** The periodic dynamics created by competing WD and batch normalization forces (NeurIPS 2021) are well-documented, but how dynamic WD methods (scheduling, alignment-aware) interact with this periodicity is unknown. Scale invariance from BN reduces WD to an effective LR scheduler, but this analysis assumes constant WD -- the interaction with time-varying or alignment-conditioned WD is unexplored.

## 5. Available Resources

### Open-source Code
- **CWD (Cautious Weight Decay)**: One-line modification, described in paper (arXiv:2510.12402); drop-in for any optimizer; published ICLR 2026
- **AlphaDecay**: https://github.com/hed-ucas/AlphaDecay (PyTorch, module-wise adaptive WD for LLMs; NeurIPS 2025)
- **SPD (Selective Projection Decay)**: https://github.com/GT-RIPL/Selective-Projection-Decay (PyTorch, layer-wise WD for fine-tuning)
- **Why Do We Need Weight Decay**: https://github.com/tml-epfl/why-weight-decay (PyTorch, NeurIPS 2024; ResNet/VGG/ViT experiments with comprehensive weight/gradient norm tracking)
- **SWD (Scheduled Weight Decay)**: https://github.com/zeke-xie/stable-weight-decay-regularization (PyTorch, NeurIPS 2023; AdamS optimizer)
- **OUI (Overfitting-Underfitting Indicator)**: https://github.com/AlbertoFdezHdez/OUI (PyTorch; DenseNet/EfficientNet/ResNet WD diagnostic)
- **AdamO (Orthogonal Dynamics)**: Described in arXiv:2602.05136 (not yet publicly released as of April 2026)
- **CPR (Constrained Parameter Regularization)**: https://github.com/automl/CPR (PyTorch, NeurIPS 2024; AdamCPR optimizer with per-matrix constraint adaptation; pip: `pytorch-cpr`)
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
- **Three Mechanisms of WD**: arXiv:1810.12281 (comprehensive analysis of L2 vs WD across SGD/Adam/K-FAC; useful as theoretical baseline)
- **Awesome-Optimizer**: https://github.com/zoq/Awesome-Optimizer (curated collection of optimizer papers and repos; reference for comprehensive baseline comparison)
- **AdamW-and-SGDW**: https://github.com/loshchil/AdamW-and-SGDW (official AdamW/SGDW implementation by Loshchilov & Hutter)
- **AdaptiveWeightDecayPytorch**: https://github.com/mmbejani/AdaptiveWeightDecayPytorch (AdaDecay PyTorch implementation)
- **Weight-Decay (gd-zhang)**: https://github.com/gd-zhang/Weight-Decay (understanding and scheduling WD experiments)
- **pytorch_optimizer collection**: https://github.com/kozistr/pytorch_optimizer (100+ optimizers including CWD, AdamD, StableAdamW; comprehensive baseline source)
- **Schedule-Free Optimization**: https://github.com/facebookresearch/schedule_free (Meta; schedule-free Adam with configurable WD application point; useful baseline for WD-schedule interaction study)
- **AdamWR (Cosine Restarts)**: https://github.com/mpyrozhok/adamwr (AdamW + cosine annealing with restarts; normalizes WD per restart period)
- **FFCV-ImageNet**: https://github.com/libffcv/ffcv-imagenet (fast ImageNet training in 500 lines; useful for rapid WD ablation experiments on ImageNet)
- **PyLO (Learned Optimizers)**: https://github.com/Belilovsky-Lab/pylo (modular learned optimizer framework with WD/LR schedule integration; HuggingFace compatible)

### Datasets
- **CIFAR-10/100**: Standard torchvision (our primary small-scale benchmark)
- **ImageNet-1K**: Standard benchmark for vision optimization (our large-scale benchmark, as specified by project constraints; local path: `/home/ccwang/dataset/imagenet-1k`)
- **OpenWebText / Wikitext-103 / C4 / Minipile**: Standard for LLM pre-training experiments (optional extension)

### Pretrained Models / Baselines
- Standard PyTorch optimizers: SGD, Adam, AdamW (built-in)
- PyTorch Lightning: High-level training framework for systematic experiments
- Weights & Biases: Experiment tracking and visualization
- timm library: Comprehensive vision model zoo with pre-tuned training recipes

## 6. Implications for Idea Generation

### Directions Worth Exploring (High Priority)

1. **Unified mathematical framework connecting all four WD sub-approaches.** Formulate a general dynamic WD update rule: lambda(t, w, g) = f(alignment(w,g), norm(w), schedule(t), target_norm(tau)). Show that CWD, AdamWN, SWD, and standard decoupled WD are special cases with specific choices of f. This is the core theoretical contribution.

2. **Standardized evaluation metrics.** Design and validate:
   - **Budget Equivalence Metric (BEM)**: Normalize all comparisons to equal compute (FLOPs or wall-clock) to prevent unfair comparison between methods that use different training budgets
   - **Coupling Stability Index (CSI)**: Measure the stability of the WD-optimizer coupling (e.g., variance of effective learning rate, oscillation in weight norm trajectory, spectral condition number evolution, gradient-to-weight ratio stability)
   - **Alignment Informativeness Score (AIS)**: Quantify how much alignment information (cosine similarity between gradient and weight vectors) actually helps improve WD decisions compared to alignment-agnostic baselines

3. **Systematic mathematical derivation linking sub-approaches.** Prove that:
   - WD scheduling is optimal under certain trajectory assumptions derivable from alignment dynamics
   - Norm-matched WD (target tau) is equivalent to a specific alignment-aware strategy with a specific alignment threshold
   - Decoupled WD in Adam is necessary precisely because coupled WD distorts the alignment signal
   - Independent WD (Kosson et al. 2025) and alignment-aware WD can be jointly derived from a common norm-stability principle

4. **Large-scale visualization and diagnostic analysis.** Produce comprehensive visualization panels showing weight norm trajectories, gradient-weight alignment evolution, spectral density shifts, effective learning rate dynamics, and coupling stability across all major WD methods on shared benchmarks. This alone would be a significant contribution to the field's understanding.

5. **Bridge the WD-normalization interaction for dynamic WD.** Extend the periodic dynamics analysis (NeurIPS 2021) to dynamic WD methods -- show how time-varying or alignment-conditioned WD modifies or stabilizes the periodic behavior created by WD-BN interaction.

### Directions to Avoid (Saturated)
- Simple scaling rules for fixed WD (gamma^2, batch-size linear) -- well-covered by Chou (2025), Wang & Aitchison (2024)
- Binary sign-based masking for WD -- CWD (ICLR 2026) is definitive
- Module-wise static WD assignment -- AlphaDecay (NeurIPS 2025) covers this
- Proposing yet another WD variant without theoretical or comparative framework
- Fixed WD hyperparameter tuning methods -- Twin (2024), OUI (2025) already address this
- Privacy-focused WD extensions -- DP-AdamW (ICML 2025) covers this niche

### Cross-Domain Analogies with Potential
- **Federated learning gradient conflict**: Layer-wise gradient alignment detection uses similar alignment concepts for aggregation decisions
- **Multi-task gradient alignment**: Gradient alignment as implicit regularization in multi-task settings
- **Control theory / sliding mode**: CWD's sliding-mode interpretation suggests deeper connections to control-theoretic optimization; the WD-BN periodic dynamics can be analyzed as a limit cycle in a nonlinear dynamical system
- **Information geometry**: The relationship between WD and the natural gradient (GRNG by Dash et al. 2026) suggests information-geometric formulations
- **WD-augmentation co-design**: Steiner et al. showed WD and data augmentation strongly interact in ViT training; a unified framework should model this interaction, especially for ImageNet-scale experiments where WD regime choice dramatically affects results
- **Gradient-to-weight ratio control**: Defazio (2506.02285) shows WD acts as a "layer balancing" controller driving ||g||/||w|| to a steady state; this suggests a feedback-control interpretation where alignment-aware WD is a more sophisticated controller that conditions the steady state on geometric signal quality
- **Spectral rank as feedback signal**: Yunis et al. (2408.11804) and ICML 2024 work show WD drives rank minimization and inter-layer singular vector alignment; a rank-aware WD schedule could close the loop between spectral structure objectives and WD strength, opening a new sub-direction: **spectral-feedback WD scheduling**
- **Duality-based optimizer unification**: Newhouse (2025) connects SGD, Adam, Shampoo via norm-dependent duality; WD in each optimizer family has geometrically different effects -- the unified framework should formalize these differences

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
| CPR / AdamCPR (GitHub) | High | Apache-2.0 | **Compose** | Borrow augmented Lagrangian per-matrix constraint formulation; compare as strong baseline for per-parameter adaptive WD |
| Newhouse Duality Thesis | Medium | N/A | **Build** | Reference duality-based unification of SGD/Adam/Shampoo for theoretical framework connecting WD across optimizer families |
| Awesome-Optimizer (GitHub) | Low | N/A | **Adopt** | Curated reference list for comprehensive baseline comparison and related work coverage |
| pytorch_optimizer collection | High | MIT | **Adopt** | 100+ optimizers with CWD, AdamD, StableAdamW; comprehensive baseline source for multi-optimizer comparison |
| Schedule-Free (Meta) | Medium | BSD | **Compose** | Reference for WD application point (at z vs y); provides schedule-free baseline for WD-schedule interaction study |
| FFCV-ImageNet | Medium | Apache-2.0 | **Adopt** | Fast ImageNet training pipeline; enables rapid WD ablation experiments on ImageNet within time budget |
| GWA alignment metric (arXiv) | Medium | N/A | **Compose** | Borrow gradient-weight coherence metric as component of Alignment Informativeness Score |
| GALA framework (OpenReview) | Medium | N/A | **Compose** | Reference alignment-based adaptive scheduling design; extend from LR to WD domain |
| LPWD pre-weight decay | Medium | N/A | **Build** | Reference delay-defect analysis; compare timing of WD application in unified framework |
| Correction of Decoupled WD (AdamC) | High | N/A | **Compose** | Integrate gamma^2 scaling insight into norm-matched sub-framework; use TUC analysis methodology |
| Independent WD / u-muP | High | N/A | **Build** | Critical for scale-transfer experiments; formalize connection between independent WD and alignment-aware WD |
| Fantastic Pretraining Optimizers | Medium | N/A | **Adopt** | Reference benchmark methodology for multi-optimizer WD comparison; use as evaluation protocol template |

**Priority reusable components**:
- **Evaluation framework**: Reuse why-weight-decay's experimental setup + SWD's optimizer baselines + OUI's diagnostic tools
- **Baseline methods**: CWD, SWD/AdamS, AlphaDecay, standard AdamW, SGD+WD, CPR/AdamCPR
- **Visualization toolkit**: Build systematic panels tracking weight norm, gradient-weight alignment (cosine similarity), spectral density, effective LR, gradient-to-weight ratio, all per-layer and aggregated
- **Theoretical foundation**: Build on Loshchilov's Weight Norm Control (general target-norm framework) as the mathematical starting point, extending with alignment and scheduling dimensions; incorporate Kosson et al.'s relative update framework for scale-transfer analysis
- **Architectures**: ResNet-20 (CIFAR), VGG-16-BN (CIFAR), ResNet-50, ResNet-101 (ImageNet), ViT (CIFAR/ImageNet) -- as specified in project constraints
- **Multi-seed protocol**: All experiments with seeds 42, 123, 456 reporting mean +/- std
- **ImageNet pipeline**: FFCV for fast training; standard PyTorch for reference runs; WD=1e-4 (SGD baseline), WD=1e-2 (AdamW baseline), WD=0.3 (ViT baseline)
