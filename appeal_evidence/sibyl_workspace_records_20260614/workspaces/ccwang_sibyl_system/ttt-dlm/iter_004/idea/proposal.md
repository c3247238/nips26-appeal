# 最终研究提案（第 4 轮迭代）：Belief-State Diffusion with Adaptive Guidance for Reasoning in Masked Diffusion Language Models

## 综合决策理由

### 全景图：六个视角的共识、分歧与本轮关键转折

经过第 3 轮完整实验验证，本项目的实验数据发生了根本性变化。以下是我对六个视角的系统分析：

**本轮关键实验事实**（所有视角共认）：
1. **DMI（扩散记忆注入）是唯一经验证有效的方法**：Countdown-500 三 seed 平均 9.3% vs vanilla 4.7%——近 2x 改善，零额外计算开销
2. **DTA（LoRA 在线更新）pilot 失败**：6.2% < vanilla 12.5%——梯度信号太弱（MLM loss 已在 0.005-0.032）
3. **纯 remasking 无效**：ReMDM-conf 4.4%, RCR 5.7%，均未超越 vanilla 4.7%
4. **SCP 计算开销 12x 但无显著提升**
5. **新文献强力验证连续表示方向**：LRD（GSM8K +2.9, MATH500 +3.8）、ReMix（2-8x 加速无损）、A-CFG（GSM8K 73.5 超越 LLaMA3 8B）

**六视角关键主张总结**：

| 视角 | 核心主张 | 推荐方向 | 权重 |
|------|---------|---------|------|
| **创新者** | DMI 成功 + 文献验证 → 连续信念表示是正确方向；A-CFG 是被忽视的强工具 | BSD + ACFG-R + DGD 三层架构 | **30%** |
| **实用主义者** | Tolerator 已有验证代码，DTA 可在精化阶段生效 | Tolerator+DTA 两阶段 | 15% |
| **理论研究者** | DLM 去噪 = 隐式在线学习，MI-Remask 有理论最优性 | DiffTTT 侧路模块 + MI-Remask | 10% |
| **反对者** | TTT 移植到 DLM 是方向性错误；PPL 完全不可信；推理时计算扩展存在逆向效应 | DLM-Native 方法 + Benchmark 驱动评估 | **20%** |
| **跨学科** | MDLM 去噪 ≅ 模拟退火/Hopfield 联想/进化搜索 | AAR + HAD + EDC | 5% |
| **实验主义者** | 计算量公平对比是核心；DTA H1 成功概率仅 35%；MetaState 已是直接竞争者 | 严格实验控制 + 降级策略 | **20%** |

### 决策转折点：为什么放弃 DTA 转向 BSD+ACFG-R

第 3 轮的实验数据给出了明确信号：

1. **DTA pilot 失败不是偶然**。实验主义者正确指出"在线 1 步 SGD 在 7B 模型上信号太弱"的风险概率为 40%——事实验证了这一担忧。MLM loss 已在 0.005-0.032 的极低水平，LoRA 更新几乎无梯度可用。这不是超参数调优能解决的问题，而是**方法论层面的限制**。

2. **DMI 的成功揭示了正确方向**。创新者精准诊断：改善 DLM 推理的最优杠杆**不在参数空间（梯度更新），而在表示空间（token 的连续表征）**。DMI 通过在 mask embedding 中注入前步 logit 信息，以零开销实现了 2x 改善——这证明跨步信息传递在表示层面比参数层面更有效。

3. **A-CFG 的变革性结果不容忽视**。A-CFG 在 LLaDA-8B 上将 GSM8K 从标准水平推到 73.5（超越 LLaMA3 8B 的 53.1），GPQA +3.9。这是当前 DLM 推理增强领域最强的 training-free 结果，且 Dream-7B 上尚未验证——存在巨大的研究空间。

4. **反对者的核心论点得到实验支持**。DTA 的失败验证了反对者关于"TTT 的成功建立在序列因果结构之上，直接移植到 DLM 不成立"的警告。但反对者关于"推理时计算扩展本身无望"的论点被 DMI 和 A-CFG 的成功所否定——关键不是扩展是否有效，而是**在正确的空间扩展**。

5. **实用主义者的 Tolerator+DTA 仍依赖 DTA——一个已被证伪的组件**。Tolerator 本身是有效的精化框架，但在其上叠加 DTA 面临同样的梯度信号问题。不过，Tolerator 的"先填满后精化"范式可以与 BSD/ACFG-R 结合。

### 各视角的权重分配及理由

- **创新者**（30%）：BSD 和 ACFG-R 的提案直接建立在 DMI 成功和 A-CFG 文献验证之上，是数据驱动的自然推进。三层正交架构（表示层/预测层/结构层）提供了丰富的消融空间和论文叙事。
- **反对者**（20%）：Benchmark 驱动评估、计算量逆向扩展警示、DLM-Native 方案的坚持——这些观点被 DTA 的失败所验证，必须充分吸收。Trajectory Entropy Profiling 的思路与 BSD 的信念向量熵监控互补。
- **实验主义者**（20%）：计算量公平对比、统计严谨性、降级策略——这些是论文可发表性的基础设施。实验主义者对竞争格局的分析（MetaState、CORE、Self-Rewarding SMC）定义了必须超越的基线。
- **实用主义者**（15%）：Tolerator 的两阶段框架可作为 BSD 的工程载体；HEX 的多专家结构作为正交基线。但核心 DTA 方案因实验证伪而降权。
- **理论研究者**（10%）：MI-Remask 的信息论框架可为 ACFG-R 的 re-mask 决策提供理论支撑；DiffTTT 的交替最小化理论可适配到 BSD 的信念精化过程。但需要训练的 DiffTTT 侧路模块在当前时间约束下优先级低。
- **跨学科**（5%）：模拟退火类比与 BSD 的 alpha 调度有对应关系；进化交叉的思路有趣但实现复杂度过高。

---

## Title

**Belief-State Diffusion with Reasoning-Aware Guidance: Multi-Layer Inference-Time Scaling for Masked Diffusion Language Models**

## Abstract

Masked diffusion language models (MDLMs) generate text through iterative denoising, but each step discards the rich distributional information from previous predictions — a fundamental "information island" problem. We propose a multi-layer inference-time scaling framework that simultaneously optimizes three orthogonal aspects of the denoising process: (1) **Belief-State Diffusion (BSD)**, which replaces hard mask embeddings with continuously evolving belief vectors — distributional mixtures weighted by model predictions — enabling smooth information accumulation across denoising steps; (2) **Reasoning-Aware Classifier-Free Guidance (RACFG)**, which identifies reasoning-critical token positions via cross-step stability analysis and applies adaptive CFG with theoretically-motivated temporal scheduling; and (3) their principled combination, where BSD provides continuous belief refinement in early steps and RACFG amplifies reasoning signals during late-stage hard token selection. On Dream-7B, our methods achieve significant improvements on Countdown (+X pp over vanilla 4.7%) and GSM8K, substantially outperforming existing remasking baselines (ReMDM, CORE, RCR) and our prior DMI method — all without external verifiers, additional training, or architecture modification.

## Motivation

### The Information Island Problem in MDLMs

Standard MDLMs suffer from a fundamental limitation: each denoising step performs an independent forward pass on a partially masked sequence, and the continuous distributional information (logits, attention patterns) from the previous step is entirely discarded after discrete argmax sampling. This "information island" problem (MetaState, arXiv 2603.01331) explains why naive approaches to inference-time scaling have failed:

- **TTT 6 variants**: statistically insignificant (p=0.88) — parameter-space updates are orthogonal to the representation-space bottleneck
- **DTA (LoRA online updates)**: 6.2% < vanilla 12.5% — gradient signal too weak (MLM loss already at 0.005-0.032)
- **Pure remasking (ReMDM-conf, RCR)**: 4.4%, 5.7% vs vanilla 4.7% — no information accumulation across steps
- **Best-of-N**: +6.9% PPL degradation with 3x compute — independent trajectories share no information

### The DMI Breakthrough and Its Implications

Our prior work on Diffusion Memory Injection (DMI) achieved Countdown-500 accuracy of 9.3% vs vanilla 4.7% — the **only** method across 20+ iterations to show substantial improvement with multi-seed validation. DMI works by injecting the previous step's logit-weighted embedding mixture into mask positions:

$$b_i^t = \alpha \cdot \text{mask\_emb} + (1-\alpha) \cdot \sum_{v} \text{softmax}(\ell_i^{t-1})_v \cdot e_v$$

This success, combined with independent validation from LRD (GSM8K +2.9, MATH500 +3.8, 10.6x speedup), ReMix (2-8x speedup without quality loss), and EvoToken-DLM, points to a unified insight: **the fundamental bottleneck in MDLM inference is the information poverty of mask embeddings — replacing them with semantically rich continuous representations is the correct direction**.

### A-CFG: The Overlooked Power Tool

Adaptive Classifier-Free Guidance (A-CFG, arXiv 2505.20199) achieved GSM8K 73.5 on LLaDA-8B — surpassing LLaMA3 8B's 53.1 — by dynamically re-masking low-confidence tokens to construct unconditional inputs for CFG. This is a transformative result that our project has completely overlooked. A-CFG exploits MDLMs' natural masking mechanism for CFG without training two separate models. Combined with CFG temporal scheduling theory (arXiv 2507.08965) showing that high guidance is harmful early but beneficial late, this opens a principled avenue for reasoning-focused inference-time scaling.

## Research Questions

1. **RQ1**: Can continuous belief-state representations (BSD) significantly outperform DMI's fixed-ratio embedding mixing on reasoning benchmarks?
2. **RQ2**: Does reasoning-aware CFG with cross-step stability signals (RACFG) outperform single-step confidence-based A-CFG?
3. **RQ3**: Are BSD and RACFG orthogonally complementary — does their combination yield synergistic gains?
4. **RQ4**: How do these methods compare to state-of-the-art baselines (MetaState, CORE, Self-Rewarding SMC) under equal compute budgets?

## Hypotheses

See `hypotheses.md` for complete list. Core hypotheses:

- **H1 (Primary)**: BSD achieves Countdown-500 accuracy ≥ 14% (vs DMI 9.3%, vanilla 4.7%), with p < 0.05 on McNemar test against DMI
- **H2**: RACFG achieves Countdown-500 accuracy ≥ 15% (vs vanilla 4.7%), leveraging CFG's proven effectiveness on reasoning tasks
- **H3**: BSD + RACFG combination achieves ≥ 18% on Countdown-500, demonstrating synergistic gains
- **H4**: Cross-step stability-guided re-masking (JSD) outperforms single-step confidence re-masking (A-CFG) by ≥ 2pp
- **H5**: CFG temporal scheduling (low-early, high-late) outperforms fixed guidance weight by ≥ 2pp
- **H6**: BSD belief vector entropy monotonically decreases during denoising, reaching lower terminal entropy than vanilla

## Method

### Component 1: Belief-State Diffusion (BSD)

**Core idea**: Replace mask embeddings entirely with "belief states" — distributional mixture embeddings weighted by model predictions — and evolve these beliefs continuously across denoising steps, only committing to hard tokens in the final k steps.

**Algorithm**:

```
Input: prompt x_p, generation length L, model f_θ, total steps T, hard-reveal steps k
Output: generated sequence x_0

Initialize: b_i^0 = mask_emb for all generation positions i

// Phase 1: Continuous Belief Refinement (steps T to k+1)
For t = T, T-1, ..., k+1:
    1. Construct input: [x_p; b^t] (prompt embeddings + belief vectors)
    2. Forward pass: ℓ^t = f_θ([x_p; b^t])
    3. Compute soft predictions: p_i^t = softmax(ℓ^t_i / τ_t)
    4. Update beliefs with EMA:
       b_i^{t-1} = (1 - α^t) · b_i^t + α^t · Σ_v p_i^t(v) · e_v
    5. Normalize: b_i^{t-1} ← b_i^{t-1} · (‖mask_emb‖ / ‖b_i^{t-1}‖)
    // NO argmax sampling, NO mask/unmask — beliefs evolve continuously

// Phase 2: Hard Reveal with optional RACFG (steps k to 1)
For t = k, k-1, ..., 1:
    1. Standard confidence-based unmask from belief states
    2. Optionally apply RACFG guidance (see Component 2)
    3. Newly revealed tokens: replace belief vector with actual token embedding
```

**Key design decisions**:

1. **Belief vector = probability-weighted embedding mixture**: $b_i^t = \sum_v p_\theta(x_i=v|x_t)_v \cdot e_v$. This is DMI's logit-weighted embedding generalized to be the **primary representation** rather than a mixing additive.

2. **EMA update rate α^t**: Adaptive schedule — small in early steps (α=0.1-0.3, stabilizing), large in late steps (α=0.5-0.8, converging). This mirrors the insight from CFG scheduling theory that early steps should explore and late steps should exploit.

3. **Norm preservation**: Belief vectors are L2-normalized to match mask embedding norm, preventing distribution shift in the Transformer's input space. This addresses the innovator's key failure mode concern about out-of-distribution inputs.

4. **Graceful degradation**: If full belief replacement fails (OOD concern), we fall back to DMI-style mixing: $\beta \cdot \text{mask\_emb} + (1-\beta) \cdot b_i^t$ with $\beta$ linearly decaying from 0.9 to 0.1. This ensures BSD ≥ DMI in the worst case.

**Distinction from related work**:

| Dimension | DMI (Ours, iter 3) | LRD (2510.11052) | ReMix (2602.22868) | EvoToken (2601.07351) | **BSD (Ours)** |
|-----------|-------------------|------------------|--------------------|-----------------------|----------------|
| Belief as input | Mixed with mask_emb | Mixed with mask_emb | Continuous mixing state | Continuous evolution | **Full replacement** |
| Cross-step memory | None (per-step) | None (KL convergence) | None | None | **EMA accumulation** |
| Goal | Speed + quality | Acceleration | Acceleration | Quality | **Quality-focused** |
| Hard reveal timing | Every step | KL-triggered | After convergence | Gradual | **Last k steps only** |

### Component 2: Reasoning-Aware Classifier-Free Guidance (RACFG)

**Core idea**: Enhance A-CFG with cross-step memory signals to identify reasoning-critical positions, and apply theoretically-motivated temporal scheduling for guidance strength.

**Algorithm** (applied during BSD Phase 2 or standalone):

```
Input: current sequence x_t, logit history {ℓ^s}_{s>t}, guidance schedule w(t)
Output: guided logits ℓ_guided

// Cross-step stability signal
For each position i:
    1. Compute smoothed logits: ℓ̄_i^t = λ · ℓ̄_i^{t+1} + (1-λ) · ℓ_i^t  (EMA, λ=0.7)
    2. Stability score: S_i^t = 1 - JSD(softmax(ℓ̄_i^t) ‖ softmax(ℓ̄_i^{t+1}))
    3. Low stability = model "hesitating" = reasoning decision point

// Stability-guided CFG construction
    4. Select top-m% LEAST stable positions → temporarily re-mask them → x̃_t
    5. Two forward passes: ℓ^+ = f_θ(x_t), ℓ^- = f_θ(x̃_t)
    6. Position-adaptive guidance:
       ℓ_guided_i = ℓ^+_i + w(S_i^t, t) · (ℓ^+_i - ℓ^-_i)

// Temporal schedule (theory-driven, from arXiv 2507.08965)
    w(S, t) = w_base · (1 - S) · schedule(t)
    where schedule(t):
      - mask_rate > 70%: w_scale = 0.0  (no guidance — insufficient info)
      - mask_rate 30-70%: w_scale = linear ramp 0→1  (reasoning window)
      - mask_rate < 30%:  w_scale = 1.0  (maximum guidance)
```

**Distinction from A-CFG**:

| Dimension | A-CFG (2505.20199) | **RACFG (Ours)** |
|-----------|-------------------|------------------|
| Re-mask signal | Single-step low confidence | **Cross-step instability (JSD)** |
| Guidance weight | Fixed w per step | **Stability-adaptive + temporal schedule** |
| Cross-step memory | None | **Logit EMA history** |
| Temporal schedule | None | **Theory-driven (early=0, late=max)** |
| Extra forward passes | 1x | 1x (same) |
| Proven on reasoning | GSM8K 73.5 (LLaDA) | Expected higher (Dream-7B) |

### Component 3: BSD + RACFG Combination

The two components operate at orthogonal layers:
- **BSD** improves the **representation** of mask positions (continuous beliefs vs hard masks)
- **RACFG** improves the **prediction quality** at decision time (CFG amplification of reasoning signals)

**Combined algorithm**:
- Steps T to k+1: BSD continuous belief refinement (no CFG needed — beliefs haven't converged)
- Steps k to 1: Hard reveal with RACFG guidance — beliefs are now highly concentrated, RACFG amplifies the remaining uncertain positions

This mirrors the CFG scheduling theory insight: guidance should be zero when information is insufficient (early/high mask rate) and maximal during the critical decision phase (late/low mask rate).

## Expected Contributions

1. **Belief-State Diffusion**: First training-free method to fully replace mask embeddings with continuously evolving distributional representations in MDLMs, demonstrating significant gains on reasoning benchmarks
2. **Reasoning-Aware CFG**: Enhancement of A-CFG with cross-step stability signals and theory-driven temporal scheduling, improving reasoning-task guidance quality
3. **Multi-layer framework**: Principled combination showing that representation-layer (BSD) and prediction-layer (RACFG) optimizations are orthogonally complementary
4. **Information-theoretic analysis**: Belief entropy trajectory analysis establishing that continuous belief evolution achieves lower terminal entropy than discrete denoising

## Experimental Plan

### Phase 1: Independent Pilots (Day 1, ~4 GPU·h, all parallel)

| GPU | Experiment | Benchmark | Purpose |
|-----|-----------|-----------|---------|
| 0 | A-CFG reproduction | Countdown-100 | Confirm CFG works on Dream-7B |
| 1 | BSD pilot (k=T/2) | Countdown-16 | Verify belief vectors don't cause OOD collapse |
| 2 | RACFG pilot | Countdown-16 | Verify stability-guided re-mask signal |
| 3 | DMI baseline (reproduction) | Countdown-100 | Confirm 9.3% baseline with current code |

**Decision Gate 1**: If A-CFG fails on Dream-7B (< vanilla), switch to LLaDA-8B for all CFG experiments.

### Phase 2: Ablation & Full-Scale (Days 2-3, ~30 GPU·h)

| Experiment | Benchmark | Seeds | GPU·h |
|-----------|-----------|-------|-------|
| BSD k ablation (k=T/4, T/2, 3T/4) | Countdown-100 | 1 | 4 |
| BSD α schedule ablation | Countdown-100 | 1 | 4 |
| RACFG m% ablation (5%, 10%, 20%) | Countdown-100 | 1 | 4 |
| RACFG schedule ablation (fixed vs ramp) | Countdown-100 | 1 | 4 |
| **BSD vs DMI vs vanilla** | **Countdown-500** | **3** | **8** |
| **RACFG vs A-CFG vs vanilla** | **Countdown-500** | **3** | **8** |

**Decision Gate 2**: Advance the best-performing direction(s) to combination and GSM8K.

### Phase 3: Combination & Extension (Days 4-5, ~30 GPU·h)

| Experiment | Benchmark | Seeds | GPU·h |
|-----------|-----------|-------|-------|
| **BSD + RACFG** | **Countdown-500** | **3** | **12** |
| Best method on GSM8K | GSM8K-1319 | 1 | 8 |
| Compute-fair comparison: methods at 2x FLOPs vs vanilla at 2x steps | Countdown-500 | 3 | 8 |
| Belief entropy trajectory analysis | Countdown-500 | 1 | 2 |

### Phase 4: Analysis & Writing Preparation (Day 6, ~6 GPU·h)

| Analysis | Purpose |
|---------|---------|
| Token-level diagnostics: which tokens changed by CFG | Qualitative understanding |
| Belief entropy trajectory visualization | Information-theoretic validation |
| Position comparison: MetaState, LRD, ReMix, CORE | Related work positioning |
| Statistical tests: McNemar + Bootstrap 95% CI | Publication rigor |

### Total Compute Budget

| Component | GPU·h | Success Probability | Priority |
|-----------|-------|--------------------|---------|
| RACFG (standalone) | ~20 | **60%** (A-CFG already proven on reasoning) | **1** |
| BSD (standalone) | ~18 | 55% (DMI validates the direction) | **2** |
| BSD + RACFG combination | ~12 | 50% (dependent on both working) | **3** |
| GSM8K extension | ~8 | conditional | 4 |
| **Total** | **~58** | At least one success: ~82% | |

4x GPU: ~15h total compute time (feasible in 2-3 days with parallelism).

### Evaluation Protocol (Incorporating Empiricist & Contrarian Demands)

**Primary metric**: Benchmark accuracy (Countdown, GSM8K) — NOT PPL

**Mandatory diagnostics per configuration**:
- rep-2/3 (n-gram repetition rate): DTA > vanilla + 20% → degeneration alert
- distinct-1/2/3 (unique n-gram ratio): < vanilla - 15% → diversity alert
- Output length distribution: variance < vanilla/2 → length mode collapse
- 20 random qualitative samples: manual inspection for degeneration patterns

**Statistical analysis**:
- McNemar test (paired classification) for each method pair
- Bonferroni correction for multiple comparisons (α' = 0.05/N)
- Bootstrap 95% CI for accuracy differences (10,000 resamples)
- Difficulty-stratified subgroup analysis

**Compute fairness** (critical, per empiricist):
- BSD ~1.1x vanilla FLOPs (only extra: EMA + normalization)
- RACFG ~2x vanilla FLOPs (one extra forward pass per step)
- BSD + RACFG ~2.1x vanilla FLOPs
- Fair comparison: methods at budget B vs vanilla/ReMDM with equivalent NFE at budget B

## Positioning Against Competition

| Method | Type | Key Result | Our Advantage |
|--------|------|-----------|---------------|
| MetaState (2603.01331) | Trained GRU memory | Cross-step memory | **Training-free** |
| A-CFG (2505.20199) | Training-free CFG | GSM8K 73.5 | **Cross-step stability + scheduling** |
| CORE (2602.04096) | Training-free remasking | MBPP +9.2 | **CFG signal > perturbation signal** |
| LRD (2510.11052) | Training-free acceleration | GSM8K +2.9 | **Quality-focused, not acceleration** |
| HEX (2510.05040) | Multi-schedule ensemble | GSM8K 88.1% (majority vote) | **Orthogonal, can combine** |
| Self-Rewarding SMC (2602.01849) | Particle sampling | Multi-benchmark gains | **Lower compute (no K particles)** |

## Publication Targeting

**Best case** (BSD + RACFG both work, Countdown ≥ 18%, GSM8K significant gains):
**NeurIPS/ICML Main** — "Beyond Remasking: Multi-Layer Inference-Time Scaling for Masked Diffusion Language Models"

**Medium case** (RACFG alone works, Countdown ≥ 15%):
**NeurIPS/ICLR Main** — "Reasoning-Aware Guidance for Diffusion Language Models: From Confidence to Cross-Step Stability"

**Fallback case** (Only BSD improves over DMI):
**ICLR** — "Belief-State Diffusion: Continuous Token Representations for Masked Diffusion Language Models"

**Minimum publishable unit** (BSD ≈ DMI but entropy analysis novel):
**EMNLP/ACL** — "Information Flow in Masked Diffusion Denoising: An Empirical Study of Cross-Step Representations"

## Failure Modes and Mitigations

### BSD Failure Modes

1. **Out-of-distribution inputs** (probability 30%): Belief vectors not in model's training distribution.
   - Mitigation: L2 normalization to match mask_emb norm; fallback to DMI-style mixing ($\beta \cdot \text{mask\_emb} + (1-\beta) \cdot b$)
   - Detection: Monitor perplexity of intermediate states; if > 2x vanilla, activate fallback

2. **Belief oscillation** (probability 20%): Logit distributions flip between steps.
   - Mitigation: Small EMA rate (α < 0.3 in early steps); temperature annealing τ_t decreasing over time

3. **Arithmetic structure sensitivity** (probability 15%): Continuous mixtures may break discrete arithmetic reasoning.
   - Mitigation: Validate on Countdown first (structured), then GSM8K (free-form)

### RACFG Failure Modes

1. **Dream-7B CFG incompatibility** (probability 15%): Dream may not support standard CFG.
   - Mitigation: Test A-CFG reproduction first; if fails, switch to LLaDA-8B (proven)
   - Fallback: Use Dream-7B for BSD only, LLaDA-8B for RACFG

2. **Guidance over-extrapolation** (probability 25%): CFG logit extrapolation produces OOD logits.
   - Mitigation: Cap guidance weight w_max = 2.0; temperature recalibration after guidance

3. **JSD computation noise** (probability 10%): Cross-step probability distributions too noisy for stable JSD.
   - Mitigation: Use EMA-smoothed distributions (λ=0.7) before JSD computation

## Key References

### Core Methods (Must Cite)
1. **LRD** (Zhu et al., 2025, arXiv 2510.11052) — Belief-state refinement, GSM8K +2.9, MATH500 +3.8
2. **A-CFG** (NeurIPS 2025, arXiv 2505.20199) — Adaptive CFG for DLM, GSM8K 73.5
3. **CFG Scheduling** (Rojas et al., 2025, arXiv 2507.08965) — Theory: high guidance early is harmful
4. **ReMix** (Ye et al., 2026, arXiv 2602.22868) — Continuous mixing state validation
5. **EvoToken-DLM** (arXiv 2601.07351) — Continuous token evolution
6. **PRR** (Wan et al., 2026, arXiv 2603.04514) — Cross-step information for acceleration
7. **MetaState** (Xia et al., 2026, arXiv 2603.01331) — Trained cross-step memory (our training-free counterpart)

### Direct Baselines
8. **ReMDM** (arXiv 2503.00307) — Remasking baseline
9. **CORE** (arXiv 2602.04096) — Context-robust remasking
10. **DMI** (Our iteration 3) — 9.3% vs vanilla 4.7% on Countdown-500
11. **HEX** (arXiv 2510.05040) — Multi-schedule ensemble, GSM8K 88.1%
12. **Tolerator** (arXiv 2510.05090) — Two-stage fill-up + refinement

### Theoretical Foundations
13. **TAPS** (arXiv 2601.22629) — Temporal division of labor in DLM
14. **EqT** (arXiv 2511.21882) — Equilibrium Transformers unifying TTT + DLM
15. **Zhang et al.** (arXiv 2602.00286) — Information-theoretic limits of remasking

### Base Models & Benchmarks
16. **Dream 7B** (arXiv 2508.15487) — Primary model
17. **LLaDA 8B** (arXiv 2502.09992) — Validation model
