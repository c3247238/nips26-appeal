

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
# 文献调研报告（迭代 2）

**研究主题**: 在 training-free 条件下改进 Diffusion Language Models (DLMs) 的生成性能，或在维持性能的同时提升 DLMs 的推理速度  
**调研时间**: 2026-03-12  
**调研方法**: arXiv + Web 双源核验  
**当前用途**: 为 `iter_002` 的 diagnostic / protocol paper 重新建立最新 related-work 地图，并识别与 `benefit_bucket_audit`、`honest compute fairness`、`observer vs controller` 最相关的空白。

## 1. 本轮最重要结论

2025-2026 的 DLM 文献已经明显从“证明 diffusion language modeling 可行”转向“如何在 inference 阶段更快、更稳、更可控”。但这些工作大多沿着三条典型路线推进：

1. **系统加速**：KV cache、prefix/prefix-like commitment、并行调度、吞吐优化  
2. **training-free remasking / revision**：利用 confidence、entropy、spatial-temporal dynamics 或 context perturbation 选择要修复的 token  
3. **受约束或结构化解码**：保证 JSON / regex / code 结构合法

对我们当前论文最关键的观察是：

- 现有工作几乎都把重点放在 **提出新 controller / scheduler / acceleration trick**。
- 很少有工作把 **realized pipeline fairness**、**observer/controller 分离**、**benefit buckets / harmed-fixed-no-effect**、**runtime-lineage discipline** 作为主贡献。
- 这意味着我们的最稳定位仍然不是“再做一个 controller”，而是做一篇 **diagnostic / protocol / failure-analysis paper**。

## 2. 已核验的高相关文献

下面只保留本轮主题最相关、且已经通过 arXiv 或官方 Web 资源核验过的文献。

| 类别 | 题目 | 时间 | 核心点 | 对本项目的直接启示 |
|------|------|------|--------|--------------------|
| 基础模型 | LLaDA | 2025 | 大规模 masked diffusion LLM 的代表工作 | 是当前很多 training-free inference 方法的默认底座 |
| 基础模型 | Dream 7B | 2025 | 7B diffusion LLM，官方仓库公开了 inference API 与 remasking strategy 参数 | 说明 entropy / confidence-based remasking 已经进入实际使用接口 |
| 加速 | FlashDLM | 2025 | FreeCache + guided diffusion，主打训练外加速 | 把 realized systems stack 带入比较，强化了“工程条件会影响结论”这一点 |
| 加速 | DINGO | 2025 | 受约束 decoding，保证 regex / structured output | 说明“结构合法性”是独立维度，不应与一般质量增益混在一起 |
| 修正 | Saber | 2025 | 代码任务上的 adaptive acceleration + backtracking remasking | 说明 code 是 revision 最容易暴露结构风险的场景 |
| 吞吐 | CadLLM | 2025 | confidence-aware calibration，用自适应 block / threshold 提高 throughput | 与我们当前 observer/controller 主题直接相关，但其目标仍是 throughput 而不是诊断 |
| 调度/修正 | STaRR | 2025/2026 | spatial-temporal token dynamics-aware remasking | 是“instability / temporal dynamics”路线的直接代表 |
| 调度 | DCD | 2026 | deferred commitment + confidence-aware sliding window | 代表 uncertainty-aware commitment 路线 |
| 修正 | CORE | 2026 | context perturbation probing 找 brittle token 做 revision | 是我们当前 honest-compute / compute-matched 讨论中最相关的 competitor |
| 调度 | LSP | 2026-03 | Longest Stable Prefix，prefix absorption + KV locality | 非常新，强调 prefix topology 与硬件效率耦合，直接提醒我们“runtime fairness 不只是 NFE” |
| 视角纠偏 | Scaling Beyond Masked Diffusion Language Models | 2026-02 | 指出 perplexity 在 diffusion family 之间并非总是可靠比较轴，speed-quality Pareto 更重要 | 与我们“honest compute changes conclusions”的 framing 高度同频 |
| 不确定性 | Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty | 2025-12 | 用 Denoising Entropy 指导 decoding path 优化 | 表明 entropy-as-signal 正在成为活跃方向，但现有工作仍偏向方法，而非 observer/controller 拆分 |

## 3. 逐篇要点（含链接）

### 3.1 Dream 7B

- 类型：基础模型 / 官方实现
- Web: https://github.com/DreamLM/Dream
- 关键信号：
  - 官方 inference API 已直接暴露 `alg="entropy"`、`maskgit_plus`、`topk_margin` 等 remasking strategy
  - 这说明“training-free remasking strategy”已经是公开可操作接口，不再只是论文概念
- 对我们的意义：
  - 支撑我们把 revision / remasking 当成一个独立研究层，而不是模型内部细节
  - 也提醒我们 related work 里要明确区分“模型能力”与“inference policy”

### 3.2 FlashDLM

- arXiv: https://arxiv.org/abs/2505.21467
- 核心：
  - `FreeCache` 复用跨步稳定 KV
  - `Guided Diffusion` 用轻量 AR 模型监督 token unmasking
  - 声称平均 `12.14x` 端到端加速且几乎不损精度
- 对我们的意义：
  - 这类工作天然把 **系统栈差异** 带入性能结论
  - 反过来加强了我们需要把 `actual NFE`、`latency`、`batch size`、`compile/backend` 一起报告的理由

### 3.3 DINGO

- arXiv: https://arxiv.org/abs/2505.23061
- 核心：
  - 为 diffusion LLM 做 distribution-preserving constrained inference
  - 在 symbolic math / JSON generation 上可显著提升结构合规性
- 对我们的意义：
  - 这类工作证明“结构合法性”可以单独成为目标
  - 也支持我们把 HumanEval 中的 syntax/runtime split 视为结构边界，而不是一般质量指标

### 3.4 Saber

- arXiv: https://arxiv.org/abs/2510.18165
- 核心：
  - 在代码任务上做 adaptive acceleration + backtracking remasking
  - 强调快采样在 code 上容易引起灾难性性能崩塌
- 对我们的意义：
  - 与我们当前的 code boundary 发现同向：code 不是泛化奖励区，而是结构风险放大器

### 3.5 CadLLM

- arXiv: https://arxiv.org/abs/2512.07173
- 核心：
  - training-free confidence-aware calibration
  - 动态控制 block size、step size、threshold，提高 throughput
- 对我们的意义：
  - 这篇最接近我们当前“observer/controller”相关主题
  - 但它仍然把 calibration / confidence 直接当 controller 使用，而不是先分离 observer quality 与 realized gain

### 3.6 STaRR

- arXiv: https://arxiv.org/abs/2601.04205
- 核心：
  - 利用 temporal variance 和 spatial deviance 做动态 remasking
  - 平均 `4.1x` speedup，最高 `8.9x`
- 对我们的意义：
  - 是 instability / temporal-dynamics 路线的代表工作
  - 正好成为我们论证“stronger intervention story does not automatically imply stronger observer evidence”的直接对照对象

### 3.7 DCD

- arXiv: https://arxiv.org/abs/2601.02076
- 核心：
  - deferred commitment + confidence-aware sliding windows
  - 目标是同时改善质量和效率
- 对我们的意义：
  - 属于 uncertainty-aware commitment family
  - 有助于我们 related work 中把“revision/remasking”和“commitment/scheduler”分开写

### 3.8 CORE

- arXiv: https://arxiv.org/abs/2602.04096
- 核心：
  - 用 targeted masked-context perturbation probing 检测 context-brittle token
  - 声称在 reasoning / code 上优于 compute-matched baselines
- 对我们的意义：
  - 是当前最需要认真对标的 revision competitor
  - 也正是我们 honest-compute 讨论里最敏感的 fairness case：因为一旦实现条件不匹配，就很容易把“方法差异”和“系统栈差异”混在一起

### 3.9 Scaling Beyond Masked Diffusion Language Models

- arXiv: https://arxiv.org/abs/2602.15014
- 核心：
  - 明确指出 perplexity 在不同 diffusion family 间未必是可靠比较轴
  - speed-quality Pareto 更有解释力
- 对我们的意义：
  - 这是对我们“honest compute changes conclusions”的最强外部同频支持之一
  - 它不是直接讲 revision，但在方法论上非常支持“不要只看 headline metric”

### 3.10 LSP

- arXiv: https://arxiv.org/abs/2603.05454
- 核心：
  - Longest Stable Prefix scheduler
  - 通过 contiguous prefix absorption 改善 KV locality 与 coherence
  - 在 LLaDA-8B / Dream-7B 上声称最高 `3.4x` 加速
- 对我们的意义：
  - 非常新，而且特别重要，因为它把“scheduler topology”与“hardware efficiency”绑定得更紧
  - 这进一步说明 runtime fairness 不能只写成 nominal steps 或 even actual NFE，至少还要显式说明 batchability / cache locality / compileability

## 4. 本轮 related-work 结构建议

如果下一步要更新论文 related work，建议按下面而不是按“所有方法平铺罗列”的方式组织：

### A. DLM foundations

- SEDD / MDLM / LLaDA / Dream / scaling papers

### B. Training-free acceleration and commitment

- FlashDLM
- DCD
- LSP
- CadLLM

### C. Training-free revision / remasking

- CORE
- STaRR
- Saber
- entropy-guided path optimization

### D. Structured / constrained decoding

- DINGO

## 本轮迭代继承的经验教训（必须吸收，不得忽略）

### 已确认的方向约束

- 不要回到 `TIGER` hero paper 叙事。
- 不要重启 `Calibration-Aware` 命名或把“校准改进”写成主贡献。
- 不要再用 generic “提出一个更强 DLM controller / scheduler” 口吻组织论文。
- 当前最稳的论文定位是：`compute-normalized diagnostic study + protocol / fairness / boundary analysis`。
- `cand_diag` 是当前稳定 front-runner framing；如提出新 idea，必须服务于该 framing，而不是取代它。

### 上一轮 review / reflection 明确指出的硬缺口

- `benefit_bucket_audit.json`：必须把 revision 样本分成 fixed / harmed / no-effect，而不是只讲 aggregate gain。
- `seed_sensitivity_spotcheck.json`：至少对 headline GSM8K pairwise comparison 做最小稳定性检查。
- `canonical_asset_manifest.json`：需要把 figure / table / JSON artifact / runtime metadata 建立一一映射。
- runtime fairness / asset-lineage appendix：要把 actual NFE、latency、batch size、compile/backend、proxy path 写成 reviewer-friendly artifact。
- `observer/controller split` 必须更显式：observer quality 不等于 controller gain，`d(s)` / `g(s)` 的定义要 appendix-grade。

### 本轮 idea debate 的边界要求

- 保持 2-3 个 serious candidates，不要过早塌缩到单一新方法。
- 候选 idea 优先围绕“如何把已有证据封口并转化成更强论文”组织，而不是继续发散找新 remasking trick。
- 如果提出新实验，必须默认采用项目记忆中的速度优化要求：大 batch、flash attention、torch.compile、多 GPU 并行优先。
- cross-task evidence 目前仍是 boundary slices；不要把 MATH500 / HumanEval 小切片写成稳定 regime 结论。
- 版本目录卫生（legacy -> `iter_000`）是迭代边界整理事项，不是本 stage 的主任务。

### 建议优先形成的 2-3 个候选方向

1. **Protocol-first diagnostic paper**
   - 核心：把 honest compute fairness、observer/controller split、asset lineage、benefit buckets 做成主贡献。
   - 目标：不靠新 controller，也能把论文做成 reviewer 难以替代的 protocol / diagnosis 贡献。

2. **Mechanism-first bucket analysis**
   - 核心：围绕 fixed / harmed / no-effect buckets 解释 revision 在不同任务和结构边界下为何帮助或伤害。
   - 目标：把 aggregate story 推进到 mechanism-aware failure taxonomy。

3. **Minimal-controller add-on only if it sharpens the diagnosis**
   - 核心：只接受那种能更干净地区分 observer quality 与 controller gain 的最小控制变量设计。
   - 约束：不能重新把论文拉回 “我们提出了更强方法” 的主叙事。

### E. Our lane

- 我们不是再提一个 controller
- 我们要做的是：**honest compute protocol + observer/controller split + failure buckets / boundary analysis**

## 5. 直接相关的研究空白

### 空白 1：realized pipeline fairness 基本没有被当成主贡献

现有工作常常同时变化：

- scheduler / remasking policy
- batch size
- caching strategy
- compile / backend
- code path / proxy implementation

但论文里往往把最终速度和质量改进直接归因于方法本身。  
这给了我们一个很清楚的位置：**不是再做 controller，而是把 comparison discipline 做对。**

### 空白 2：observer quality 与 controller gain 往往没有被拆开

很多工作默认“好的 uncertainty signal = 好的 control variable”。  
但从我们当前资产看，至少在 tested policies 下，这个等价关系并不自动成立。

### 空白 3：bucket-level mechanism evidence 很少出现

现有工作大多报：

- accuracy
- pass@1
- speedup
- throughput

但很少回答：

- revision 修复了哪些样本？
- 它伤害了哪些原本正确的草稿？
- 哪些样本完全无效？

这正是 `benefit_bucket_audit` 可以让我们与方法论文真正区分开的地方。

### 空白 4：结构边界常被当成 generalization，而不是 stress test

代码任务中的 syntax improvement 往往容易被误解为整体质量改善。  
但 DINGO、Saber、以及我们当前 HumanEval 结果都说明：**结构合法性** 和 **语义/执行正确性** 必须拆开。

## 6. 对当前 iter_002 的直接启示

### 6.1 论文定位

本轮 literature 明确支持我们继续坚持：

- `diagnostic study`
- `evaluation protocol`
- `honest compute`
- `observer vs controller`
- `boundary / failure analysis`

而不建议重新回到：

- hero controller
- generic DLM improvement
- benchmark-standard-setter

### 6.2 下一步最值得补的不是新方法，而是新证据对象

最优先的增量不是再找一个 remasking heuristic，而是补下面这些 artifact：

1. `benefit_bucket_audit.json`
2. `seed_sensitivity_spotcheck.json`
3. `canonical_asset_manifest.json`
4. runtime fairness / asset-lineage appendix

### 6.3 如果后续要加 related-work 对照，优先顺序建议

1. `CORE`
2. `STaRR`
3. `DCD`
4. `LSP`
5. `CadLLM`

原因很简单：

- `CORE` 和我们当前 honest-compute fairness 争议最直接相关
- `STaRR` 最直接对应 instability / temporal-dynamics 路线
- `DCD` / `LSP` 说明 commitment topology 与系统效率是活跃方向
- `CadLLM` 说明 confidence/calibration 已经被直接推进为 throughput controller

## 7. 参考链接（核验来源）

- Dream GitHub: https://github.com/DreamLM/Dream
- Fast-dLLM GitHub: https://github.com/NVlabs/Fast-dLLM
- FlashDLM: https://arxiv.org/abs/2505.21467
- DINGO: https://arxiv.org/abs/2505.23061
- Saber: https://arxiv.org/abs/2510.18165
- CadLLM: https://arxiv.org/abs/2512.07173
- STaRR: https://arxiv.org/abs/2601.04205
- DCD: https://arxiv.org/abs/2601.02076
- CORE: https://arxiv.org/abs/2602.04096
- Scaling Beyond Masked Diffusion Language Models: https://arxiv.org/abs/2602.15014
- LSP: https://arxiv.org/abs/2603.05454
