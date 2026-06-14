

## Project Spec
# 项目: dlm-improve

## 研究主题
在 training-free 条件下改进 Diffusion Language Models (DLMs) 的生成性能，或在维持性能的同时提升 DLMs 的推理速度。

## 背景与动机
Diffusion Language Models（如 MDLM, SEDD, Dream, LLaDA, MMaDA）作为自回归模型的替代范式，具有并行生成、灵活编辑等优势，但在采样质量和推理速度上仍存在明显短板。现有加速方法（如 Block Diffusion, FAST-DLLM V2）虽有进展，但 sampling strategy 仍有较大改进空间。本项目聚焦 training-free 方法，从扩散模型的底层数学原理出发，寻找改进采样过程的理论依据和工程实现。

## 初始想法
- 从扩散过程的数学公式推导出发，分析现有 sampling strategy 的理论缺陷
- 从底层机制（noise schedule、transition kernel、denoising dynamics）寻找改进点
- 探索自适应步数/token 级别的采样策略
- 结合离散扩散过程的特殊性质设计更高效的 sampler
- 研究 confidence-based 或 entropy-based 的动态去噪策略

## 关键参考文献
- MDLM (Masked Diffusion Language Models)
- SEDD (Score Entropy Discrete Diffusion)
- Dream (Diffusion Reasoning with Enhanced Abilities for Machines)
- LLaDA (Large Language Diffusion with Attention)
- MMaDA (Multi-Modal Diffusion with Attention)
- Block Diffusion
- FAST-DLLM V2: Efficient Block-Diffusion LLM
- Continuously Augmented Discrete Diffusion model for Categorical Generative Modeling
- 其他由文献调研阶段补充

## 可用资源
- GPU: 4x NVIDIA RTX PRO 6000
- 服务器: cs8000d (SSH MCP connection: cs8000d, fallback: default)
- 远程路径: /home/ccwang/sibyl_system
- 单个实验如有需要或者资源空闲可尝试多卡并行提升效率
- 实验要考虑大 batch size，以提升训练/推理速度，充分利用显存

## 实验约束
- 实验类型: **优先 training-free**；允许 1 小时内的 LoRA 轻量训练
- 训练任务时间预算: ~1 小时
- 评测任务: **不受时间限制**，按 benchmark 完整跑完
- 统计显著性: **不需要**。不换 seed 多跑，benchmark 合理且无过度下采样时以单次结果为准
- Pilot 采样: **最少 100 条**，不接受 n<100 的 pilot。benchmark 条数本身小于 100 条的除外
- 模型规模: 使用各 DLM 论文的开源预训练模型（通常为中小规模），建议先从小尺寸（0.6B,4B）开始

## 评测策略
- 使用社区通用 benchmark（如各 DLM 论文常用的 text8, One-Billion-Word, lambada, GSM8K 等）
- Pilot 实验可选用小型 benchmark 或 benchmark 子集（>=100 条）
- 正式实验使用完整 benchmark，不做下采样
- 对标模型: 各 DLM 的原始采样策略 + 已有加速方法（Block Diffusion, FAST-DLLM V2 等）

## 目标产出
- **顶会论文**（NeurIPS / ICML / ACL 级别）
- 质量期望: **weak accept 及以上**（borderline+）
- 论文模板: **NeurIPS LaTeX 模板**
- 论文语言: **全英文**
- 图像可视化：可以使用可视化图像来解释和表达含义思想想法的，要尝试使用可视化图像进行表示，不要拘泥于表格和折现柱状曲线图，还要考虑热力图、注意力图等解释模型内部机制的可视化方法

## 方法论要求
- **从原理公式推导**：不仅仅是工程 trick，要有数学/理论支撑
- **底层机制分析**：深入分析扩散过程的 transition dynamics、noise schedule 等
- **理论→实验**：先建立理论框架，再用实验验证理论预测
- 结合实际工程实现，确保方法可复现

## 特殊配置
- 飞书同步: 启用
- Codex 审查: 启用
- 写作模式: parallel（加速写作）


## 文献调研报告（请仔细阅读，避免重复已有工作）
# Iteration 4 Literature Note

## Focus

本轮 literature_search 不做泛化型综述，而是围绕 iteration 3 reflection 中确定的 reviewer-facing gap 做定向补强：

1. 给 training-free / test-time DLM revision 找到更扎实的领域锚点。
2. 给 “signal is informative but does not validate the intervention” 这一论点补 calibration / failure-prediction 文献。
3. 给 runtime-lineage / sampler attribution 补 reviewer-friendly 的 related-work 支撑。
4. 给 manuscript 的最小 bibliography enrichment 提供一组可直接落到 `references.bib` 的候选。

## arXiv track: DLM / revision / test-time scaling

### 1. Foundational discrete diffusion language modeling

- **Zheng et al., 2023 — _A Reparameterized Discrete Diffusion Model for Text Generation_**
  - arXiv: https://arxiv.org/abs/2302.05737
  - Why it matters:
    - 适合作为早期 discrete diffusion text generation 的基础锚点。
    - 可以支撑引言里“DLM 已从可行性走向 inference-time engineering”的前史。

- **Lou et al., 2023 — _Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution_ (SEDD)**
  - arXiv: https://arxiv.org/abs/2310.16834
  - Why it matters:
    - 是离散 diffusion LM 的另一条强基线主线。
    - 若 Related Work 要更像 reviewer-ready package，SEDD 比只列 RADD 更完整。

### 2. Current DLM inference / test-time scaling landscape

- **Bai et al., 2026 — _Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models_**
  - arXiv: https://arxiv.org/abs/2602.01842
  - Web confirmation: arXiv listing surfaced directly in search.
  - Why it matters:
    - 明确说明 discrete diffusion LM 也在走 test-time scaling 路线，而不是只有 autoregressive LLM 在做。
    - 可用于支撑本文“small-gain + inference-time compute reallocation 已经成为真实 reviewer 背景”的论点。

- **Lu et al., 2026 — _Advancing Block Diffusion Language Models for Test-Time Scaling_**
  - arXiv: https://arxiv.org/abs/2602.09555
  - Why it matters:
    - 展示更广义的 block diffusion / adaptive decoding test-time scaling 方向。
    - 有助于把本文放在“test-time intervention landscape”里，而不是只像单一 paper-specific case note。

- **Xia et al., 2026 — _MetaState: Persistent Working Memory for Discrete Diffusion Language Models_**
  - arXiv: https://arxiv.org/abs/2603.01331
  - Web confirmation: arXiv listing surfaced directly in search.
  - Why it matters:
    - 这是 reviewer 很可能会提的 direct neighboring line：通过持久化跨步 memory 改善 dLLM。
    - 即使本文不做 direct experiment，也应在 discussion 中交代：MetaState 属于带训练/微调的 architecture augmentation，与本文 training-free audit template 的约束不同。

### 3. Guidance / control in discrete diffusion

- **Schiff et al., 2024 — _Simple Guidance Mechanisms for Discrete Diffusion Models_**
  - arXiv: https://arxiv.org/abs/2412.10193
  - Why it matters:
    - 直接覆盖“guidance / classifier guidance 如何迁移到 discrete diffusion”。
    - 可用来说明本文不是在提出新的 guidance mechanism，而是在审计一个 training-free revision claim 的 attribution boundary。

### 4. Attribution / sampler-centric caution

- **_Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models_**, 2026
  - arXiv: https://arxiv.org/abs/2602.19619
  - Web confirmation: arXiv listing surfaced directly in search.
  - Why it matters:
    - 这是目前最贴近本文 “runtime / sampler / method attribution 要分开” 的相关工作。
    - 非常适合支持本文的 reviewer-facing runtime-lineage note：headline 结果不能只看 accuracy，还要看 sampler / backend / execution contract。

## arXiv track: calibration / uncertainty / failure prediction

### 1. Calibration methods need not improve downstream trust decisions

- **Zhu et al., 2023 — _Rethinking Confidence Calibration for Failure Prediction_**
  - arXiv: https://arxiv.org/abs/2303.02970
  - Why it matters:
    - 论文核心发现就是：常见 calibration method 对 failure prediction 可能无效甚至有害。
    - 这和本文当前的结论高度同构：observer-side entropy / confidence signal 可以 informative，但不自动转化成 successful controller。
    - 建议在 discussion 里直接用来支撑 “risk marker != validated targeting rule”。

- **Deng et al., 2023 — _Towards A Unified View of Answer Calibration for Multi-Step Reasoning_**
  - arXiv: https://arxiv.org/abs/2311.09101
  - Why it matters:
    - 给 multi-step reasoning 中 answer calibration 的 taxonomy 和 unified view。
    - 可以支持本文在 related work 中把 calibration 放回 “post-processing / answer calibration family” 的正确位置，而不是把它写成已验证的 control principle。

### 2. Training-free guided decoding via confidence-like signals

- **_Enhancing Language Model Factuality via Activation-Based Confidence Calibration and Guided Decoding_**, 2024
  - arXiv: https://arxiv.org/abs/2406.13230
  - Why it matters:
    - 可用作“confidence-like observer signal 被用于 decoding / guidance”的近邻工作。
    - 但本文应明确区分：相关工作说明 observer signal 可被消费，不说明该 signal 在当前 DLM revision setup 中已经被验证为 causal controller。

## Web confirmations and reviewer-facing anchors

通过 Web 搜索补到的最有用锚点不是“更多二手综述”，而是以下几篇当前 reviewer 很可能认识或主动搜索的 arXiv 页面：

- Prism — https://arxiv.org/abs/2602.01842
- Sampler-centric evaluation — https://arxiv.org/abs/2602.19619
- MetaState — https://arxiv.org/abs/2603.01331

这些页面足够作为最小 primary-source anchors，便于后续把 `references.bib` 从“可编译”提升到“reviewer 看到不会觉得太薄”。

## What to cite next in the manuscript

如果下一轮只允许补最小引用集，优先级建议如下：

1. **RADD (2302.05737)**  
   用于 foundational discrete diffusion LM background。

2. **SEDD (2310.16834)**  
   用于补齐离散 diffusion LM 的核心主线，避免只引用自家相邻方法。

3. **Prism (2602.01842)**  
   用于 test-time scaling on dLLMs。

4. **Sampler-Centric Evaluation (2602.19619)**  
   用于 runtime / sampler attribution caution。

5. **MetaState (2603.01331)**  
   用于最直接 neighboring work / reviewer comparison point。

6. **Rethinking Confidence Calibration for Failure Prediction (2303.02970)**  
   用于支撑 “calibration can hurt or fail at downstream trust decisions”。

7. **Towards A Unified View of Answer Calibration for Multi-Step Reasoning (2311.09101)**  
   用于把 calibration/answer post-processing 放回统一 related-work 框架。

8. **ActCab+CoDec (2406.13230)**  
   用于 observer-signal-guided decoding 的近邻支撑。

## Direct writing implications for iteration 4

### 1. Abstract / introduction

- 加一句 scoped caveat：本文证据是 `n=100 audited slice` 的 bounded audit，而不是 benchmark-level population estimate。
- 加一句 literature-facing justification：在 dLLM test-time scaling / revision 已经进入 small-gain regime 的背景下，sham-control-driven reinterpretation 本身就是有价值的结果。

### 2. Related work

建议重写成三段，而不是平铺罗列：

1. **Discrete diffusion LM foundations and inference engineering**  
   RADD, SEDD, LLaDA 1.5, DPad, Prophet, Prism.

2. **Guidance / calibration / uncertainty as observer-side signals**  
   ActCab+CoDec, Unified View of Answer Calibration, Rethinking Confidence Calibration for Failure Prediction.

3. **Attribution and execution-envelope caution**  
   Sampler-Centric Evaluation, plus brief mention of MetaState as a trained neighboring alternative.

### 3. Discussion / limitations

- 明确写：本文不是提出新的 controller family，也不主张 entropy-guided revision 已被验证。
- 明确写：stronger sham control 改写 active-control gain 的解释，这一负面结果本身对 small-gain dLLM evaluation 有价值。

## Bottom line

这轮 literature_search 的结论很明确：下一轮不缺“更多想法”，缺的是**更像 reviewer 认识的 related-work scaffold**。  
最值得补的不是新实验，而是把本文放进下面这条清晰链条里：

`discrete diffusion LM -> test-time scaling / guidance -> calibration / failure prediction caution -> sampler/runtime attribution -> bounded negative-case contribution`

只要这条链写顺，配合 runtime-lineage note 与 artifact-release statement，当前 submission package 就有机会从 `7.2` 继续往 `8+` 推。


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# Iteration 4 Proposal

## Title

Evidence-Aligned Candidate Program For Training-Free DLM Revision

## Abstract

iteration 4 不再把 `MGCD` 写成已经成立的方法胜者，而是把本轮重新定义为一个 **evidence-aligned candidate program**。现有证据仍主要来自 `iter_003` 的负例审计：它支持“observer signal 有信息量”，但不支持“token-wise controller 已成立”。与此同时，iteration 4 的 `proposal.md`、`pilot_summary.md` 与 `summary.md` 一度出现对象漂移：方法主线已 pivot 到新架构，但 pilot artifact 仍沿用 reviewer-polish / no-new-pilot 叙事。基于本轮 6 视角辩论与 3 轮 critique，我们将候选池收束为 `MGCD / BSR / DSG` 三个 serious candidates：`MGCD` 作为当前 front-runner hypothesis，`BSR` 作为复杂度与 repair object 的关键对照，`DSG` 作为轻量 fallback。下一步不是直接进入 full benchmark，而是先执行 screening pilot，在 matched compute 下判别多轨 state、span/boundary object 与单轨 drift gate 三类解释。

## Evidence-Driven Revisions

### 本轮明确变更

1. 不再使用“iteration 4 是 reviewer-polish-only cycle”的旧 framing。
2. 不再把 `MGCD` 写成“当前最优主方案已经成立”。
3. 新增 `cand_bsr`，作为 `MGCD` 的 complexity control 与中间层候选。
4. 保留 `cand_dsg`，避免候选池再次过早塌缩到单一方法。
5. 明确写死：**目前没有 iteration 4 新 pilot 证据**，所以 proposal 只是待验证的 candidate program，而不是实验结论。

### 什么仍未改变

- 当前最强实证仍来自 `iter_003`：entropy / calibration 信号可以作为 observer，但 token-wise revision 没有稳定超越 sham-control。
- 所有 full benchmark 推进都必须等待 screening pilot 完成，并遵守最大 batch size 与加速栈约束。

## Motivation

iteration 3 的核心教训不是“CARD 的措辞需要更保守”，而是：

- `observer signal exists`
- `controller gain is not validated`
- `token-wise revision may be the wrong action object`

因此 iteration 4 的任务不应再是继续包装一个负例论文，也不应直接 all-in 一个未被 pilot 检验的新大方法，而应把问题重写成一组更小、更可证伪的候选：

- 是否需要显式 stateful memory？
- 或者仅仅把 repair object 从 token 提升到 span / boundary 就已足够？
- 又或者单轨 drift gate 已经足够接近最优？

## Research Questions

### RQ1

当前 DLM revision 的失败主要来自 **state representation 不足**，还是来自 **repair object 定义错误**？

### RQ2

在 matched compute 条件下，`stateful consensus`、`boundary-stable span repair`、`single-track drift gate` 三者中，哪一种最先表现出对 `RAND-84` 的真实 separation？

### RQ3

如果新候选出现增益，这个增益是否来自更好的控制对象，而不是隐藏的额外搜索 / 草图 compute？

### RQ4

结构化 repair 是否能减少稳定区域副损伤，而不是通过更大范围重写制造新的 harm？

## Candidate Program

### cand_mgcd

`MGCD: Memory-Gated Consensus Diffusion`

- 角色：当前 `front_runner hypothesis`
- 核心假设：多轨一致性与跨步记忆共同定义了更好的可修复状态
- 最小实现：
  - dual short drafts
  - `locked / contested / bridge` tri-state memory graph
  - contested-island revision
  - memory-gated accept rule
- 当前地位：
  - 可以领跑
  - 不能被写成已胜出
  - 必须先通过 screening gate

### cand_bsr

`BSR: Boundary-Stable Revision`

- 角色：强备选 + complexity control
- 核心假设：真正关键的也许不是 memory，而是 span / boundary repair object
- 最小实现：
  - 单轨历史稳定度
  - island 聚合
  - boundary locking
  - 不引入完整 dual-draft memory graph
- 当前地位：
  - 是本轮必须保留的 serious candidate
  - 用来回答“是否真的需要更复杂 state”

### cand_dsg

`DSG: Drift-Span Gating`

- 角色：轻量备选
- 核心假设：单轨 drift signal 已足以定义可用的 span gate
- 最小实现：
  - 单轨 hidden/logit drift
  - span aggregation
  - 轻量 gate
- 当前地位：
  - 保留为低复杂度 fallback
  - 用来判断是否存在“无需 memory 也可接近最优”的路径

## Current Front-Runner

当前 front-runner 仍是 `cand_mgcd`，但它只比 `BSR` 和 `DSG` 多赢了一点点先验动机，而没有任何 iteration 4 pilot separation。

保留 `MGCD` 领跑的理由：

- 它最直接回应了 `observer -> controller` 断裂背后的 state insufficiency 问题。
- 它能提出比“再做一个更聪明的 entropy score”更结构化的架构解释。

限制条件也必须写清：

- 若 `MGCD` 未优于 `RAND-84`，直接失败。
- 若 `MGCD` 仅与 `BSR` 持平，则说明复杂 memory state 未被证明必要。
- 若 `DSG` 已接近 `MGCD`，则应优先保留更轻解释。

## Hypotheses

### H1: Sham-Control Separation

若新 repair object 真有价值，则至少一个新候选必须在 matched compute 下清晰优于 `RAND-84`。

### H2: State Necessity

若多轨状态确有必要，则 `MGCD` 应明显优于 `BSR` 与 `DSG`。

### H3: Object Sufficiency

若真正关键的是 span / boundary object，而非 memory，那么 `BSR` 应接近或达到 `MGCD` 的效果。

### H4: Hidden-Compute Neutrality

任何方法增益都不能主要由 dual draft 或额外 search compute 解释。

### H5: Harm Containment

结构化 repair 应降低或至少不显著扩大稳定区域副损伤。

## Screening Pilot Plan

### Stage 0: Artifact Alignment

- 重写 `pilot_summary.md` 与 `summary.md`
- 明确 iteration 4 是 `method pivot + screening pilot`
- 清除 reviewer-polish-only 残留

### Stage 1: Minimal Implementations

- `BSR-84`
- `MGCD-84`
- 如资源允许补 `DSG-84`

### Stage 2: Unified Pilot Comparison

- 对照：
  - `RAND-84`
  - `CARD-84`
  - `BSR-84`
  - `MGCD-84`
  - `DSG-84`（可作为第二批）
- benchmark 顺序：
  - 先 `GSM8K audited slice`
  - 通过后再 `MBPP audited slice`

### Required Logging

- exact match
- actual NFE
- latency / tokens_per_sec
- peak_vram_mb
- accepted update ratio
- locked / contested ratio
- mean island size
- repair / harm counts

### Promotion Gates

#### Hard gates

- 未优于 `RAND-84`：不得进入 full benchmark
- evidence bundle 仍不一致：不得进入 planning/full

#### Upgrade gates

- `MGCD > BSR`：才允许把 memory state 写成更强主张
- `BSR > DSG`：才允许把 span / boundary object 写成优于单轨 drift 的解释

## Expected Contributions

### 如果 screening pilot 成功

1. 一个与 `iter_003` 负例诊断一致的 training-free 新候选程序
2. 一个真正可回答“state 是否必要”的 candidate comparison setup
3. 一个 reviewer 更容易接受的 claim ladder：
   - observer
   - state
   - controller

### 如果 screening pilot 失败

1. 更强的否证结果：说明当前 stateful/span-level redesign 仍未越过 sham-control gate
2. 更干净的归因：知道失败发生在 `memory`、`boundary object` 还是 `drift gate`
3. 更少的 story drift：因为 iteration 4 不再把 proposal 写成已验证结论

## What Changed From The Previous Round

- 从“reviewer-facing polish cycle”切换到“method pivot + screening pilot”。
- 从单押 `MGCD` 改成 `MGCD / BSR / DSG` 的 candidate program。
- 从“下一轮该直接验证 MGCD”改成“下一轮先验证哪种控制对象值得继续”。
- 从单一大方法叙事改成 claim ladder + falsification gates。


## 当前可检验假设
# Iteration 4 Hypotheses

## 总原则

本轮不验证“MGCD 已经赢了”，而是验证以下互斥命题哪一个最接近真实。

## H1: Sham-Control Separation Hypothesis

> 若新 revision object 真有价值，则至少一个新候选必须在 matched compute 下清晰优于 `RAND-84`。

### 通过信号

- `MGCD-84`、`BSR-84` 或 `DSG-84` 中至少一个稳定超过 `RAND-84`

### 否证条件

- 所有新候选都未超过 `RAND-84`

## H2: Stateful Consensus Necessity Hypothesis

> 若多轨状态表示是关键，`MGCD-84` 应明显优于 `BSR-84` 与 `DSG-84`。

### 通过信号

- `MGCD-84 > BSR-84`
- `MGCD-84 > DSG-84`

### 否证条件

- `MGCD-84 ≈ BSR-84`
- 或 `MGCD-84 ≈ DSG-84`

## H3: Boundary Object Sufficiency Hypothesis

> 若真正关键的是 span / boundary repair object，而非复杂 memory，`BSR-84` 应接近 `MGCD-84` 并超过 `CARD-84`。

### 通过信号

- `BSR-84 ≈ MGCD-84`
- 且 `BSR-84 > CARD-84`

### 否证条件

- `BSR-84` 无法超过 `CARD-84`
- 或 `BSR-84` 明显落后于 `MGCD-84`

## H4: Single-Track Sufficiency Hypothesis

> 若单轨 drift signal 已足够，`DSG-84` 应接近 `BSR-84` 或 `MGCD-84`。

### 通过信号

- `DSG-84 ≈ BSR-84`
- 或 `DSG-84 ≈ MGCD-84`

### 否证条件

- `DSG-84` 明显落后于 `BSR-84`
- 且明显示弱于 `MGCD-84`

## H5: Hidden-Compute Neutrality Hypothesis

> 新候选的任何增益都不能主要由额外草图/搜索 compute 解释。

### 通过信号

- actual NFE、latency、accepted updates 与 gain 大致一致
- 不存在“质量提升完全被额外 compute 吞掉”的情况

### 否证条件

- 增益主要来自 dual drafts 或隐藏搜索
- matched compute 条件被破坏

## H6: Harm Containment Hypothesis

> 结构化 repair 应降低或至少不显著扩大稳定区域副损伤。

### 通过信号

- harm count 不显著上升
- island size 与 accepted update ratio 仍可控

### 否证条件

- harm count 明显升高
- island 过大导致长跨度重写失控

## Screening Gate

### 进入 MBPP 的条件

- `GSM8K audited slice` 上至少出现一个超过 `RAND-84` 的候选

### 进入 full benchmark 的条件

- H1 未被否证
- H5 未被否证
- evidence bundle 已对齐，不再出现 proposal / summary / pilot drift

### 升级为更强主张的条件

- 只有当 `MGCD-84 > BSR-84` 时，才允许把 memory state 写成更强解释
- 只有当 `BSR-84 > DSG-84` 时，才允许把 boundary object 写成优于单轨 drift 的解释


## 小型实验真实反馈（必须基于这些证据修正 idea，不能忽略负结果）
# Pilot Summary For Iteration 4

## Status

iteration 4 的 screening pilot 已执行完成，当前建议是 `PIVOT`，**不要**进入 full benchmark。

## Headline Result

- `cand_mgcd / MGCD-lite` 在 `GSM8K audited slice` 上 `accuracy=0.22`，相对 `CARD-84` 的 `net_repaired=-5`，相对 `RAND-84` 的 `net_repaired=-4`
- `cand_mgcd / MGCD-lite` 在 `MBPP audited slice` 上 `accuracy=0.0`，相对 `CARD-84` 与 `RAND-84` 都是 `net_repaired=-2`
- `cand_dsg / DSG` 在 `GSM8K audited slice` 上 `accuracy=0.30`，相对 `CARD-84` 的 `net_repaired=-1`，相对 `RAND-84` 的 `net_repaired=0`
- `cand_dsg / DSG` 在 `MBPP audited slice` 上 `accuracy=0.02`，相对 `CARD-84` 与 `RAND-84` 都是 `net_repaired=-1`

## Decision-Relevant Reading

这轮 pilot 给出的不是“新方法已经站住”，而是一个比较干净的负结果：

- `MGCD-lite` 在两个 audited slice 上都没有越过 `RAND-84`
- `DSG` 比 `MGCD-lite` 更轻、更干净，但仍然没有越过 `RAND-84`
- 因此 `H1 / Sham-Control Separation` 被否证，当前候选程序里没有 survivor 可以直接进入 full benchmark

## What This Means

按 `pilot_plan.json` 里写死的 gate：

> 如果 `MGCD-lite` 和 `DSG` 都未能超过 `RAND-84`，就返回 `idea_debate`，而不是启动 full benchmark。

所以 iteration 4 现在不该做的是：

- 不推进 full benchmark
- 不把 `MGCD` 继续保留为 front-runner
- 不把 `DSG` 的“更轻但仍不分离”写成 positive signal

更合理的下一步是：

1. 把这轮结果收束成负例证据包
2. 回到 `idea_debate`
3. 重新审视 `repair object` 本身，必要时改为 `BSR` 或新的候选，而不是继续加预算给 `MGCD / DSG`

## Runtime Notes

- 四个 pilot 都在 `batch_size=50` 下完成；safe batch probe 为 `182`
- `MGCD-lite` 两个 pilot 都跑在 `eager + compile=false`
- `DSG` 两个 pilot 都跑在 `eager + compile=true`
- 执行过程中本地 monitor 有 ghost-running / stale state 漂移，最终以远端 `DONE` marker 作为权威完成信号


## 小型实验结构化信号（供你提炼 go/no-go / confidence / hypothesis status）
{
  "iteration": 4,
  "stage": "pilot_experiments",
  "status": "completed",
  "decision_recommendation": "PIVOT",
  "screening_rule": "If both MGCD-lite and DSG fail to separate from RAND-84, return to idea_debate instead of launching a full benchmark.",
  "advance_candidate_found": false,
  "full_benchmark_allowed": false,
  "shared_runtime": {
    "sample_count": 50,
    "effective_batch_size": 50,
    "probed_safe_batch_size": 182,
    "attention_backend": "eager"
  },
  "candidate_results": [
    {
      "candidate_id": "cand_mgcd",
      "arm_name": "mgcd84",
      "dataset_results": [
        {
          "dataset": "gsm8k",
          "accuracy": 0.22,
          "avg_nfe": 70.22,
          "vs_card84_net_repaired": -5,
          "vs_rand84_net_repaired": -4,
          "compile_enabled": false,
          "result_path": "current/exp/results/mgcd_gsm8k_pilot.remote.json"
        },
        {
          "dataset": "mbpp",
          "accuracy": 0.0,
          "avg_nfe": 84.84,
          "vs_card84_net_repaired": -2,
          "vs_rand84_net_repaired": -2,
          "compile_enabled": false,
          "result_path": "current/exp/results/mgcd_mbpp_pilot.remote.json"
        }
      ],
      "screening_verdict": "failed",
      "failure_reason": "MGCD-lite failed the sham-control gate on both audited slices and underperformed CARD-84 as well."
    },
    {
      "candidate_id": "cand_dsg",
      "arm_name": "dsg84",
      "dataset_results": [
        {
          "dataset": "gsm8k",
          "accuracy": 0.3,
          "avg_nfe": 68.0,
          "vs_card84_net_repaired": -1,
          "vs_rand84_net_repaired": 0,
          "compile_enabled": true,
          "result_path": "current/exp/results/dsg_gsm8k_pilot.json"
        },
        {
          "dataset": "mbpp",
          "accuracy": 0.02,
          "avg_nfe": 68.0,
          "vs_card84_net_repaired": -1,
          "vs_rand84_net_repaired": -1,
          "compile_enabled": true,
          "result_path": "current/exp/results/dsg_mbpp_pilot.remote.json"
        }
      ],
      "screening_verdict": "failed",
      "failure_reason": "DSG was cleaner than MGCD but still never separated from RAND-84 and did not beat CARD-84 on either audited slice."
    }
  ],
  "untested_candidates": [
    {
      "candidate_id": "cand_bsr",
      "status": "not_executed",
      "note": "BSR remained a planned complexity-control candidate but was not run in this screening round."
    },
    {
      "candidate_id": "cand_lmg",
      "status": "deferred",
      "note": "LMG stayed out of the training-free screening stage by design."
    }
  ],
  "gate_checks": {
    "h1_sham_control_separation": "failed",
    "h2_state_necessity": "not_tested",
    "h3_boundary_object_sufficiency": "not_tested",
    "h4_single_track_sufficiency": "failed",
    "h5_hidden_compute_neutrality": "inconclusive",
    "h6_harm_containment": "failed"
  },
  "notes": [
    "MGCD and DSG both consumed real pilot budget and produced negative evidence rather than launch failures.",
    "The local monitor had stale running-state drift during execution; remote DONE markers were used as the authoritative completion signal.",
    "The next loop should return to idea_debate instead of instantiating a full benchmark."
  ]
}


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "selected_candidate_id": "cand_mgcd",
  "selection_status": "front_runner_hypothesis_before_screening",
  "selection_reason": "MGCD is still the strongest architecture-level hypothesis for repairing the observer-to-controller gap, but iteration 4 has no new pilot evidence yet, so it remains only the current front-runner inside a 3-candidate screening program.",
  "evidence_state": {
    "iteration_4_new_pilot_evidence": false,
    "current_issue": "proposal / pilot_summary / summary drift",
    "required_fix": "align artifacts before and during screening pilot"
  },
  "advance_gate": {
    "status": "required_before_full_benchmark",
    "hard_gates": [
      "At least one new candidate beats RAND-84 on matched-compute audited slices",
      "proposal.md, pilot_summary.md, and summary.md are aligned on method-pivot framing",
      "runtime contract uses maximum safe batch size and acceleration stack when available"
    ],
    "upgrade_gates": [
      "Only claim memory necessity if MGCD-84 clearly beats BSR-84",
      "Only claim span/boundary superiority if BSR-84 clearly beats DSG-84"
    ]
  },
  "candidates": [
    {
      "candidate_id": "cand_mgcd",
      "title": "MGCD: Memory-Gated Consensus Diffusion",
      "status": "front_runner",
      "summary": "用 dual draft consensus、tri-state memory graph 与 contested-island revision，把 revision 从 token-wise remask 升级为 stateful span repair。",
      "hypotheses": [
        "Multi-trajectory state is necessary to convert observer signal into controller gain",
        "Structured contested-island repair can beat sham control without hidden-compute confounds"
      ],
      "pilot_focus": "Test whether MGCD beats RAND-84, CARD-84, and BSR-84 under matched compute on audited slices.",
      "strengths": [
        "最直接回应 state insufficiency",
        "最接近真正的架构级修复",
        "training-free 可起步"
      ],
      "risks": [
        "可能退化成昂贵 self-consistency",
        "若只比 BSR 略好，则 memory 必要性不成立",
        "若 separation 主要来自额外 compute，则解释失效"
      ]
    },
    {
      "candidate_id": "cand_bsr",
      "title": "BSR: Boundary-Stable Revision",
      "status": "backup",
      "summary": "不引入完整 memory graph，而是用单轨稳定度、island 聚合与 boundary locking 测试 span-level repair object 是否已足够。",
      "hypotheses": [
        "Span and boundary object may be sufficient even without full memory state",
        "A simpler structured repair path can match or nearly match MGCD"
      ],
      "pilot_focus": "Serve as the complexity-control comparator for MGCD and test whether boundary-stable repair already beats CARD-84.",
      "strengths": [
        "归因更干净",
        "实现成本低于 MGCD",
        "能直接回答 memory 是否必要"
      ],
      "risks": [
        "如果定义不清会退化成 MGCD-lite 话术备选",
        "可能仍不足以超过 RAND-84"
      ]
    },
    {
      "candidate_id": "cand_dsg",
      "title": "DSG: Drift-Span Gating",
      "status": "backup",
      "summary": "仅用单轨 hidden/logit drift 构造 span gate，测试低复杂度 signal 是否已足够接近最优。",
      "hypotheses": [
        "Single-track drift may already provide an adequate action state",
        "Low-complexity gating may approach MGCD/BSR without memory overhead"
      ],
      "pilot_focus": "Check whether a lighter drift-based span gate is already competitive with BSR or MGCD.",
      "strengths": [
        "实现最轻",
        "适合作为 fallback",
        "有助于判断是否需要复杂 state"
      ],
      "risks": [
        "创新性较弱",
        "可能只是 CARD 的 span-level 延长线"
      ]
    }
  ],
  "dropped_candidates": [
    {
      "candidate_id": "cand_lmg",
      "reason": "Deferred from the serious candidate pool until a training-free state signal is validated."
    }
  ]
}


## 上一轮 validation 决策意见
# Idea Validation Decision

## Decision

`PIVOT`

## Why

本轮 screening pilot 已经真正执行完了，而且结论足够明确：当前候选程序里没有 survivor 可以进入 full benchmark。

- `cand_mgcd`
  - `GSM8K audited slice`: `vs_rand84_net_repaired = -4`
  - `MBPP audited slice`: `vs_rand84_net_repaired = -2`
  - 结论：两个 slice 都没有越过 sham-control gate，而且相对 `CARD-84` 也在退步

- `cand_dsg`
  - `GSM8K audited slice`: `vs_rand84_net_repaired = 0`
  - `MBPP audited slice`: `vs_rand84_net_repaired = -1`
  - 结论：它比 MGCD 更轻、更干净，但仍然没有形成真实 separation

按 `current/plan/pilot_plan.json` 的 gate：

> 如果 `MGCD-lite` 和 `DSG` 都未能超过 `RAND-84`，就返回 `idea_debate`，而不是启动 full benchmark。

因此这轮不是 `REFINE current winner`，而是应该承认：

- `cand_mgcd` 已被 screening gate 筛掉
- `cand_dsg` 也没有成为可推进候选
- 当前 front-runner 叙事必须终止

`cand_bsr` 虽然仍可保留为一个下轮讨论对象，但它在这一轮没有执行，因此不能把“也许 BSR 会更好”当成继续给 MGCD / DSG 加预算的理由。

## Required Next Moves

1. 回到 `idea_debate`，围绕 `repair object` 重新组织候选池。
2. 把 `MGCD` 与 `DSG` 的这轮结果作为负例证据包保留下来，不再把它们当成当前 survivor。
3. 如果下一轮要保留 `BSR`，就把它作为新的 object-level candidate 重新立题，而不是延长 MGCD 主线。
4. 在下一轮实验前修复 scheduler / monitor 的 ghost-running 漂移，确保远端完成信号能及时回灌本地状态。


## 上一轮 validation 结构化决策
{
  "decision": "PIVOT",
  "selected_candidate_id": "",
  "confidence": 0.93,
  "reasons": [
    "`cand_mgcd` 已经完成本轮 screening pilot，但在 `GSM8K audited slice` 上相对 `RAND-84` 的 `net_repaired=-4`，在 `MBPP audited slice` 上相对 `RAND-84` 的 `net_repaired=-2`；它在两个 slice 上都没有越过 sham-control gate。",
    "`cand_dsg` 也已经完成本轮 screening pilot，但在 `GSM8K audited slice` 上相对 `RAND-84` 仅 `net_repaired=0`，在 `MBPP audited slice` 上是 `net_repaired=-1`；它比 MGCD 更轻，但仍然没有形成真实 separation。",
    "`current/plan/pilot_plan.json` 已明确写死：如果 `MGCD-lite` 和 `DSG` 都未能超过 `RAND-84`，就返回 `idea_debate`，而不是启动 full benchmark。当前证据正好落在这个 fail branch 上。",
    "`cand_bsr` 在本轮没有执行，因此当前 candidate program 中没有任何一个候选满足 `ADVANCE` 的条件；继续沿着 MGCD front-runner 叙事追加 GPU 预算只会放大错误对象。"
  ],
  "next_actions": [
    "返回 `idea_debate`，把 `cand_mgcd` 从 front-runner 降级为已筛掉候选，并围绕 `repair object` 重新组织候选池。",
    "将 `current/exp/results/pilot_summary.json` 与 `pilot_summary.md` 视为本轮权威 pilot 证据包，后续讨论必须以 `MGCD failed H1`、`DSG failed H1` 为起点。",
    "若保留 `cand_bsr`，应把它作为新的 complexity-control / object-level candidate 重新立题，而不是把它当成 MGCD 的轻量附属物。",
    "在下一轮新实验前修复 scheduler-lock / monitor drift，可审计地记录远端 PID、DONE marker 与本地 experiment_state 的同步，避免再次出现 ghost-running。"
  ],
  "dropped_candidates": [
    "cand_mgcd",
    "cand_dsg"
  ]
}
