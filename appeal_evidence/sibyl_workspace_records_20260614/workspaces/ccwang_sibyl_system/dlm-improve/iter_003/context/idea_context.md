
## Iteration-3 当前辩论锚点（必须先读）

以下内容优先级高于后文中仍保留的早期 brainstorming 文案；后文可视为背景库，而不是当前主张。

### 当前控制平面状态

- 当前 stage：`idea_debate`
- 当前 iteration：`3`
- 已完成的 pilot 决策：`idea_validation_decision = PIVOT`
- 当前被选中的候选线：`cand_negative_audit_pivot`

### 已锁定的 pilot 事实

- `CARD-84` 在 `GSM8K` 上相对 compute-matched `DNB-84` 的 `net repaired samples = +7`
- 但 `CARD-84` 相对 sham control `RAND-84` 的 `net repaired samples = +1`
- 该结果未通过我们预先设定的更严格 gate：`gsm8k_card_beats_rand84_by_2_net_repaired = false`
- `MBPP` 没有推翻这次实验，但也没有把 `CARD-84` 从 `RAND-84` 中分离出来
- artifact closure 已成功：四个实验臂可 sample-level join，runtime drift 已修复，claim-to-asset map 已落盘

### 当前辩论不允许做的事

- 不允许再把 `CARD-84` 写成已经成立的正向主方法
- 不允许把 reasoning slice 上的局部信号外推为 general DLM gain
- 不允许重新打开新的 controller family、额外 pilot 分支或“再做一个方法也许就赢了”的逃避叙事
- 不允许把 trajectory 分析包装成缺失主证据的替代品；本轮 trajectory addon 已因负面 pivot 明确跳过

### 当前辩论真正要解决的问题

在 `CARD-84` 没能明显击败 `RAND-84` 之后，这篇论文的最强可发表对象是什么？

允许的主线应围绕以下问题展开：

1. 如何把这次结果写成 reviewer 可接受的 skeptical audit，而不是失败实验的防守性解释？
2. 哪些 claim 仍然可以成立，哪些 wording 必须硬性收缩？
3. 如何把 runtime fairness、matched compute、sham control、per-sample audit、failure taxonomy 组织成一个 protocol-first 贡献？
4. 下一阶段写作时，如何同时保留“reasoning 上曾有信号”与“该信号没有通过更严格负对照”的张力，而不自相矛盾？

### 当前可用的最强正面素材

- 不是“我们找到了更强的 revision controller”
- 而是“更严格的审计协议成功阻止了一个本来很容易被写成正向故事的小增益”
- 以及“current-only artifact bundle 现在足够自包含，能支撑 skeptical read 与复核”

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
# 文献调研报告

**研究主题**: 在 training-free 条件下改进 Diffusion Language Models (DLMs) 的生成性能，或在维持性能的同时提升 DLMs 的推理速度  
**调研时间**: 2026-03-12  
**本轮调研焦点**: 为 iteration 3 重新建立可辩论的文献基础，重点围绕 `bucket-level audit`、`runtime fairness`、`uncertainty/calibration`、`training-free DLM inference` 和 `reviewer-friendly protocol packaging`

**arXiv 搜索关键词**:
- `"diffusion language model" OR DLM OR "discrete diffusion" AND (decoding OR revision OR inference)`
- `("uncertainty" OR calibration OR confidence) AND ("language generation" OR decoding OR reasoning)`
- `("evaluation protocol" OR reproducibility OR auditing OR benchmark) AND ("language model" OR decoding)`
- `LLaDA OR Prophet OR DPad OR "diffusion language models know the answer before" OR "accuracy meets parallelism"`
- `("discrete diffusion" OR "diffusion language model") AND (survey OR benchmark OR evaluation)`

**Web 搜索关键词**:
- `diffusion language models survey 2025 GitHub Awesome-DLMs`
- `LLaDA diffusion language model GitHub`
- `DPad Efficient Diffusion Language Models with Suffix Dropout GitHub`
- `Diffusion Language Models Know the Answer Before Decoding Prophet GitHub`

## 1. 领域现状摘要

2025-2026 的 DLM 研究已经从“DLM 能不能做语言生成”转向“DLM 在什么 regime 下能比 AR 更快、更可控、且不明显掉点”。主流叙事不再只是模型本体，而是 inference-time engineering：减少 denoising steps、利用 early convergence、重构 suffix attention、引入 block/prefix caching、以及用专门 inference engine 把理论并行性变成真实吞吐。代表工作包括 `Prophet`（early commit / early stop）、`DPad`（suffix dropout / sliding window）、`DLM-One`（one-step distillation）和 `Efficient-DLM` / `E2D2`（架构级效率改造）。

与此同时，另一条文献线开始揭示一个对我们更关键的事实：DLM 的 headline accuracy / speed gain 并不足以说明“方法真的更好”。`Is Your Diffusion Sampler Actually Correct?` 这类 sampler-centric 评估工作指出，离散扩散语言模型的采样误差可能和 denoiser 误差混在一起，而常见 aggregate 指标并不能分离 sampler-induced error。这和我们当前项目在 reviewer 处遇到的问题高度一致：如果 runtime regime、coverage、uncertainty scope 没有被讲清楚，那么小幅 gain 很容易被质疑成 implementation artifact、sampling artifact 或 slice artifact。

第三条相关脉络来自 uncertainty / confidence calibration。近两年大量工作开始强调 confidence 不是单一数字，而是与 reasoning depth、evidence access、temporal trajectory 和 token/step-level localization 纠缠在一起。例如 `Don’t Think Twice! Over-Reasoning Impairs Confidence Calibration` 指出更长 reasoning budget 可能让 calibration 变差；`Confidence over Time` 强调 stepwise confidence pattern；`ActCab + CoDec`、`CritiCal` 和 `COREA` 则把 calibration 信号直接用于 decoding 或 routing。关键启示是：observer quality、controller gain 和 realized runtime 应该分开建模与报告，而不能在一个 headline number 里混用。

## 2. 核心参考文献

| 序号 | 标题 | 来源 | 年份 | 核心贡献 | 局限性 |
|------|------|------|------|---------|--------|
| 1 | A Reparameterized Discrete Diffusion Model for Text Generation | arXiv 2302.05737 | 2023 | 早期系统化 discrete diffusion text generation 框架，为后续 dLLM 提供建模基础 | 更偏 foundational model，不能回答 2025-2026 inference efficiency 与 auditability 问题 |
| 2 | LLaDA 1.5: Variance-Reduced Preference Optimization for Large Language Diffusion Models | arXiv 2505.19223 | 2025 | 展示 LLaDA 系列通过 post-training / alignment 在 GSM8K、HumanEval、MBPP 等 benchmark 上提升 | 重点是 preference optimization，不是 training-free inference 或 evaluation protocol |
| 3 | DPad: Efficient Diffusion Language Models with Suffix Dropout | arXiv 2508.14148 | 2025 | training-free suffix pruning / sliding-window 推理优化；报告最高 61.4x speedup | 核心卖点是 speed，缺少 reviewer-friendly uncertainty / fairness accounting |
| 4 | Diffusion Language Models Know the Answer Before Decoding | arXiv 2508.19982 | 2025 | 提出 Prophet，通过 top-2 confidence gap 做 early commit，重构 “何时停止采样” 问题 | 主要关注 early decode convergence，本身并不提供 bucket-level evidence accounting |
| 5 | DLM-One: Diffusion Language Models for One-Step Sequence Generation | arXiv 2506.00290 | 2025 | 用 score distillation 探索 one-step sequence generation，目标是极端加速 | 已不属于 training-free 纯 inference 改动，且 teacher-student distillation 会改变论文定位 |
| 6 | Encoder-Decoder Diffusion Language Models for Efficient Training and Inference | arXiv 2510.22852 | 2025 | 通过 encoder-decoder specialization 改善训练和推理效率，强调 architecture/runtime co-design | 属于架构级方案，不能直接作为我们当前 training-free narrative 的正面对标 |
| 7 | Discrete Diffusion in Large Language and Multimodal Models: A Survey | arXiv 2506.13759 | 2025 | 提供 dLLM / dMLLM 系统综述，覆盖训练、推理、量化、trustworthy issues | survey 宏观全面，但不能替代细粒度 protocol 设计 |
| 8 | A Survey on Diffusion Language Models | arXiv 2508.10875 | 2025 | 从 DLM 全景视角整理模型、inference optimization、multimodal extension 与挑战 | 与上面 survey 有重合，更适合做 taxonomy，不直接解决评审中的 evidence bundling 问题 |
| 9 | Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models | arXiv 2602.19619 | 2026 | 明确指出 sampler-induced error 与 denoiser error 混淆；提出 oracle framework 检查采样正确性 | 仍是 controlled evaluation，不直接给出 reviewer-ready manuscript packaging 方法 |
| 10 | Anchored Diffusion Language Model | arXiv 2505.18456 | 2025 | 通过 anchor token 提升 DLM 建模和生成质量，强调“重要 token”对于 reconstruction 的作用 | 主要是 model design / pretraining story，不是 inference-time protocol story |
| 11 | ActCab + CoDec: Enhancing Language Model Factuality via Activation-Based Confidence Calibration and Guided Decoding | arXiv 2406.13230 | 2024 | 将 calibration signal 用于 guided decoding，证明 confidence signal 可成为 controller input | 研究对象主要是 AR LLM factual QA，不是 DLM；observer-controller transfer 需要谨慎 |
| 12 | Don’t Think Twice! Over-Reasoning Impairs Confidence Calibration | arXiv 2508.15050 | 2025 | 指出 reasoning budget 增长可能恶化 calibration，反对“更多 test-time compute 一定更好” | 不是 DLM-specific，但对 honest-compute framing 非常 relevant |
| 13 | Confidence over Time: Confidence Calibration with Temporal Logic for Large Language Model Reasoning | arXiv 2601.13387 | 2026 | 把 confidence 作为时间序列 / stepwise object 建模，强调 single scalar confidence 不够 | 仍是 AR reasoning-oriented，不提供 DLM-specific bucket / sampler audit |
| 14 | CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration? | arXiv 2510.24505 | 2025 | 区分 uncertainty critique 与 confidence critique，说明 calibration supervision 可以通过自然语言 critique 增强 | 更偏 verbalized confidence / critique training，不是 training-free DLM inference |
| 15 | Confidence-Calibrated Small-Large Language Model Collaboration for Cost-Efficient Reasoning (COREA) | arXiv 2603.03752 | 2026 | 用 calibrated confidence 做 routing/cascade，展示 calibration 与 cost-quality tradeoff 的直接联系 | 是 SLM-LLM collaboration，不是 DLM，但对“confidence 作为 compute allocator”很有启发 |
| 16 | Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models | arXiv 2602.01842 | 2026 | 针对 dLLM 的 test-time scaling、partial remasking、self-verification | 更接近新一代 controller / search 框架，可能是下一轮 idea debate 的重要对标 |

## 3. SOTA 方法与基准

### 3.1 当前方法格局

- **基础 dLLM / DLM 平台**: LLaDA / LLaDA 1.5 / LLaDA-MoE、Dream、Open-dLLM、LLaDA2.0
- **training-free inference acceleration**:
  - `Prophet`: 用 confidence gap 做 early commit / stop
  - `DPad`: 减少 suffix attention 冗余
  - `Prism`: 将 test-time scaling、hierarchical search 和 self-verification 引入 dLLM
  - `dInfer`: 将 block-level parallel decoding、KV-cache reuse 等系统优化落地为 inference engine
- **架构级效率方案**:
  - `DLM-One`: one-step generation
  - `E2D2`: encoder-decoder 分工降低 iterative full-network cost
  - `Efficient-DLM`: 从 AR-to-dLM conversion 入手，强调 accuracy-throughput tradeoff

### 3.2 主流 benchmark

当前 DLM 论文里最常见、也与本项目最相关的 benchmark 仍然是：

- **reasoning / knowledge**: GSM8K, MMLU, AIME, math reasoning suites
- **code generation**: HumanEval, MBPP, LiveCodeBench
- **general instruction / alignment**: IFEval, Arena-Hard, agent/alignment suites
- **long sequence / summarization / seq2seq**: CNN/DailyMail, Gigaword, Arxiv summarization

对本项目而言，`GSM8K + HumanEval/MBPP` 仍是最有价值的组合，因为它们能最清楚地区分：
- revision 是否只在 reasoning 场景稳定有效；
- code 侧 gain 是否被 syntax / formatting / truncation artifact 驱动；
- speedup 与 quality 是否在不同任务上出现不对称 tradeoff。

### 3.3 与我们当前问题最相关的评估信号

本轮文献最直接支持的不是“再发明一个 revision heuristic”，而是以下评估与报告对象：

- **bucket-level outcome decomposition**: fixed / harmed / no-effect 之类的 sample-level accounting
- **runtime-lineage metadata**: backend, compile path, safe batch size, realized steps / NFE, cache regime
- **uncertainty scope**: scalar confidence 不够，需要明确 token-level / step-level / trajectory-level 信号到底在支撑什么结论
- **sampler-centric correctness**: aggregate quality gain 不等于 sampler itself is correct

## 4. 已识别的研究空白

- **空白 1: DLM training-free inference 论文很多，但 reviewer-friendly evidence bundle 很少。**  
  现有工作大多给 speedup / accuracy headline，却很少把 runtime regime、coverage、uncertainty scope、sampler correctness 和 sample-level buckets 合成一个可审计协议。

- **空白 2: DLM 的 confidence / calibration 文献明显少于 AR LLM。**  
  大量 calibration / confidence-decoding 工作建立在 AR reasoning 上，迁移到 DLM 时最关键的问题反而是：confidence 究竟对应 observer、controller，还是 sampler stopping rule。

- **空白 3: sampler-centric evaluation 刚开始出现，但还没有和论文写作协议整合。**  
  `Is Your Diffusion Sampler Actually Correct?` 说明 sampler correctness 是独立问题，但还没有把这一点转化成“论文该如何报告一个小 gain 才可信”的成型 protocol。

- **空白 4: DLM inference efficiency 与 scientific reporting 之间有断层。**  
  速度论文强调 engine / attention / caching；评估论文强调 correctness / calibration；两边很少在同一篇工作里被统一成 claim-to-asset lineage。

- **空白 5: code vs reasoning 的 task-dependent failure mechanism 仍不清楚。**  
  现有文献会报告 benchmark 差异，但对“为什么某种 revision / stop rule 在 GSM8K 可行、在 HumanEval/MBPP 退化”缺少 bucket-level 或 localized uncertainty 解释。

## 5. 可用资源

### 开源代码 / repo

- `inclusionAI/LLaDA2.0`  
  GitHub: https://github.com/inclusionAI/LLaDA2.0  
  用途: 最新大规模 dLLM family、对照其 `CAP` 与 engineering story。

- `inclusionAI/dInfer`  
  GitHub: https://github.com/inclusionAI/dInfer  
  用途: 研究 DLM inference engine、KV-cache reuse、block-level parallel decoding 的系统细节。

- `Crys-Chen/DPad`  
  GitHub: https://github.com/Crys-Chen/DPad  
  用途: training-free speedup baseline；对照 suffix attention 剪枝与 scratchpad 设计。

- `pixeli99/Prophet`  
  GitHub: https://github.com/pixeli99/Prophet  
  用途: early commit / stopping-rule baseline；与本项目的 confidence-based revision/stop 叙事最接近。

- `VILA-Lab/Awesome-DLMs`  
  GitHub: https://github.com/VILA-Lab/Awesome-DLMs  
  用途: iteration 3 持续跟踪 DLM 新论文、benchmark 和 repo 的入口索引。

- `LiQiiiii/DLLM-Survey`  
  GitHub: https://github.com/LiQiiiii/DLLM-Survey  
  用途: discrete diffusion 相关论文集合与 trustworthy / quantization / inference 线索。

- `pengzhangzhi/Open-dLLM`  
  GitHub: https://github.com/pengzhangzhi/Open-dLLM  
  用途: fully open evaluation / inference / checkpoints，对 code benchmark 特别有参考价值。

- `eth-sri/constrained-diffusion`  
  GitHub: https://github.com/eth-sri/constrained-diffusion  
  用途: 结构化 / CFG constrained decoding，可为 code / format-safe generation 提供边界思路。

### 数据集 / benchmark

- GSM8K, MMLU: reasoning / knowledge 主线 benchmark
- HumanEval, MBPP, LiveCodeBench: code generation / code robustness
- CNN/DailyMail, Gigaword, Arxiv summarization: 条件长文本 generation
- survey repo 中收录的 dLLM-specific benchmark list 可作为后续扩展入口

### 预训练模型 / family

- LLaDA / LLaDA 1.5 / LLaDA-MoE / LLaDA2.0
- Dream 系列
- Open-dLLM
- 其他 survey 中收录的 discrete diffusion variants

## 6. 对 Idea 生成的启示

### 给后续 iteration 3 idea debate 的具体建议

- **值得探索 1: 把“方法创新”进一步收缩成“评估协议创新 + 最小 controller change”。**  
  文献显示 speed/accuracy controller 已经很多，下一轮更有胜算的方向是把 reviewer 最缺的 `bucket + runtime + sampler + uncertainty` 四件套做成真正可复用协议。

- **值得探索 2: 把 observer/controller/runtime 明确拆开。**  
  不要再让 calibration signal、revision policy 和 realized execution path 混在一个 headline claim 里；这正是当前 reviewer 持续卡住的点，也是文献还没真正做好包装的空白。

- **值得探索 3: 把 task dependence 当成主问题而不是副现象。**  
  下一轮不应再泛泛追求 “DLM revision works”；更强的方向是解释 why reasoning gains survive but code gains collapse，尤其要区分 truncation / formatting repair 与 genuine reasoning repair。

- **值得探索 4: 向 sampler-centric evaluation 借力。**  
  `Is Your Diffusion Sampler Actually Correct?` 说明 sampler correctness 本身就是 publishable angle。我们可以把当前 honest-compute story 升级成“如何在小 gain setting 下排除 sampler/runtime artifact”的 protocol paper。

- **值得探索 5: 如果要继续做 controller，优先比较 stopping-rule / compute allocation，而不是再发明大而全 revision family。**  
  Prophet、Prism、COREA 这条线都说明 compute allocation / confidence-aware routing 更像当前 frontier。

- **不值得再做 1: 仅凭 aggregate delta 再讲一个更大的 method-forward 故事。**  
  2025-2026 文献已经有大量 speedup/controller 论文，若没有更自包含的 evidence bundle，很容易被视为又一个 benchmark-specific tweak。

- **不值得再做 2: 继续把 calibration 写成泛用增益来源。**  
  现有 calibration literature 说明 confidence 的作用高度依赖 task、budget 和 intervention point。更安全的 framing 是“calibration / uncertainty 是 observer-side evidence，不自动蕴含 controller-side gain”。

## 7. 直接结论（供下一阶段使用）

本轮文献调研支持一个非常明确的策略判断：

1. **第 3 轮不应优先追求新的大 revision 方法。**
2. **第 3 轮最有希望的主线是：把当前项目收缩成一个 DLM diagnostic / evaluation protocol 论文，并用更强的 literature framing 支撑它。**
3. **如果还要做新实验，优先做“runtime / uncertainty / sampler correctness / task dependence”的最小补洞，而不是再扩方法面。**

换句话说，文献现状并不支持我们去跟 DPad / Prophet / Prism 正面拼“又快又强的新 inference trick”；更现实也更有学术辨识度的方向，是把 reviewer 已经指出的 evidence gap 变成 paper 本身的核心贡献。


## 当前综合提案（如已有，请在此基础上迭代，而不是从零开始）
# 从启发式增益到可审计增益：面向 Training-Free DLM 的 Sham-Controlled 最小选择性修订协议

## 摘要

本轮 debate 已经明显收敛：iteration-3 不应再把主问题写成“我们又提出了一个更强的 DLM controller”，而应写成**如何在 training-free DLM 中，把一个小幅 revision gain 变成 reviewer 可接受的、可复核的、边界清晰的证据闭环**。  
因此我选择的单一主线是：**以当前已有最佳证据支持的 `CARD-84` 作为 reference controller，不再扩展新的 controller family；论文主贡献转为一套 protocol-first 的审计框架，并用一个 matched-compute、sham-controlled、cross-task 的决定性 pilot 来回答当前增益究竟来自 targeted revision、extra compute、随机扰动，还是 audited slice 偶然性。**

这条主线吸收了各方最强部分：

- `pragmatist` 提供论文 framing：protocol / audit bundle / artifact closure / runtime fairness
- `empiricist` 提供决定性 pilot：`DNB-64`、`DNB-84`、`CARD-84`、`RAND-84`
- `contrarian` 提供硬约束：不得把 audited-control 结论外推成 general DLM claim；必要时追加 stronger sham tie-breaker
- `theoretical` 提供判定语言：observer / controller / runtime 三层必须分离，且正向 claim 只允许停在“task-conditional controller evidence”
- `interdisciplinary` 提供轻量机制分析：`first-correct`、`stable-correct`、`relapse` 三个 trajectory 指标
- `innovator` 只保留为 discussion 语言资源：若证据真的成立，可用 selective compute / compute eligibility 描述结果，但不能反过来驱动主张

## 选择的主线

### 论文主张

**主张级别：**

在固定 runtime contract、matched compute、sham control 与 audited slice 条件下，检验一个最小 reference controller (`CARD-84`) 是否能在 reasoning 任务上提供可信的 task-conditional gain；若不能，则将贡献收缩为“training-free DLM inference claims 的审计协议与负结果边界”。

### 不是这篇 paper 要做的事

- 不是新 controller family 论文
- 不是 selective-compute policy paper
- 不是 theorem/guarantee paper
- 不是通用 DLM inference improvement 论文
- 不是靠 `n=96` audited slice 去支撑 broader benchmark generality 的论文

### 允许保留的最强 claim

如果 pilot 成功，最强允许 claim 仅为：

> 在 audited controls 下，raw-entropy-guided minimal revision 在 reasoning-oriented slice 上相对 matched-compute 与 random targeting 表现出更可信的净修复证据；该结果支持一种 task-conditional、protocol-bounded 的 controller finding，而不是通用 DLM gain claim。

如果 pilot 失败，允许保留的 claim 为：

> 许多看似合理的 training-free DLM selective revision gain 在 matched-compute 与 sham-controlled 审计下无法站住，因此 future DLM inference papers 需要更严格的 evidence-closure protocol。

## 动机

iteration-2 的核心问题不是方法不够花哨，而是**证据与叙事边界不匹配**。当前 workspace 与项目记忆已经给出几条非常关键的事实：

- H1/H2：DLM 有系统性过度自信，raw entropy 是有效误差信号
- H3/H5：revision 与 `CARD-84` 在 reasoning slice 上看起来有帮助
- H4：calibration 并未超越 raw entropy
- H6：annealing 被 falsify，`T=1.0` 才是最稳配置
- 标准去噪并不单调，说明“多算力是否有用”本身就是需要审计的问题

这些事实共同说明：当前最值得做的不是再发明一个新方法，而是把**最小 reference controller + 最强负对照 + 最严格 runtime contract**组合成一篇 reviewer-friendly 的 protocol paper。

## Evidence-Driven Revisions

相较 iteration-2 的隐含方向，本轮 proposal 做了五个实质收缩：

1. **从“方法增益”收缩为“证据闭环”**  
   不再默认 `CARD-84` 已经足以支持主方法故事，而是把它降格为 reference controller。

2. **从“calibration story”收缩为“observer signal story”**  
   既然 H4 已表明 calibration 没有额外收益，就不再把 calibration 当论文中心；只保留 raw entropy 作为 observer。

3. **从“更多 compute 可能更好”收缩为“必须 matched-compute 才能讲话”**  
   H6 与标准去噪非单调都说明 compute 不是天然正向，因此 `DNB-84` 成为必做基线。

4. **从“aggregate accuracy”收缩为“per-sample repair / harm”**  
   只报平均值已不足以过审，必须报告 `fixed / harmed / unchanged` 以及 code failure anatomy。

5. **从“统一任务故事”收缩为“reasoning / code task dependence”**  
   允许 reasoning 上成立、code 上不成立；但必须把 code 伤害定位清楚，不能用语言吸收负结果。

## Research Questions

1. 在固定 runtime contract 与 matched compute 下，`CARD-84` 相对 `DNB-84` 是否仍保留稳定净收益？
2. `CARD-84` 是否真正优于随机 targeting，而不是 generic revision / 扰动效应？
3. 该收益是否只存在于 reasoning 任务，而无法安全外推到 code？
4. high-entropy 样本是否更集中地产生 `fixed`，从而支持 observer 的排序价值？
5. 若 reasoning gain 存在，它更像“更快稳定变对”，还是“最后一步侥幸修正”？

## Hypotheses

### 主假设

- **H1**：在 GSM8K audited slice 上，`CARD-84` 相对 `DNB-84` 具有正的 `net repaired samples`，且 3 seeds 至少 2 个同号。
- **H2**：在 GSM8K audited slice 上，`CARD-84` 相对 `RAND-84` 具有正的 `net repaired samples`，否则 entropy-targeting 不成立。
- **H3**：高 entropy bucket 中的 `fixed` 富集高于低 entropy bucket；若不成立，observer 只能算 correlation signal，不能算 useful triage signal。
- **H4**：reasoning gain 若存在，应更多体现在 `stable-correct` 提前或 `harmed` 降低；若 code 侧退化存在，必须可定位到 truncation / syntax / semantic failure。

### 否证型假设

- **F1**：若 `CARD-84 <= DNB-84`，则当前故事主要由 extra compute 驱动，mainline 改写为 skeptical audit。
- **F2**：若 `CARD-84 <= RAND-84`，则 entropy targeting 不成立，不能再写 observer-guided controller。
- **F3**：若正负号在 3 seeds 间明显翻转，或 bootstrap CI 广泛覆盖 0，则只能写 audited-control 下 inconclusive。
- **F4**：若 code 伤害无法被定位，不能写 task dependence 解释，只能写 evaluator / output fragility unresolved。

## 决定性 Pilot

### 名称

`pilot_evidence_closure_v1`

### 核心设计

- 数据：
  - `GSM8K` audited slice `n=48`
  - `MBPP` audited slice `n=48`
  - 总计 `n=96`
- seeds：`11 / 22 / 33`
- 运行约束：
  - 最大安全 batch size 先二分探测
  - 左填充批推理
  - `flash_attention_2`
  - `torch.compile`
  - current-only runtime contract
  - 相同 prompt template / tokenizer / post-processing / hardware / seed set

### 必做四个实验臂

1. **`DNB-64`**
   - 标准 64-step denoising
   - `T=1.0`

2. **`DNB-84`**
   - 与 `CARD-84` 对齐总 NFE
   - 不做 selective revision

3. **`CARD-84`**
   - 64-step draft
   - raw entropy top-10% remask
   - 3 revision steps
   - `T=1.0`

4. **`RAND-84`**
   - 与 `CARD-84` 对齐总 NFE、revision 次数、remask 数量
   - 仅将 remask 位置改为随机

### 只在必要时追加的 tie-breaker

若 `CARD-84 > RAND-84` 但 reviewer risk 仍高，则只追加一个 stronger sham：

5. **`PERM-84`**
   - 保留每个样本相同的 intervention budget
   - 将 entropy ranking 在样本间打乱

`HEUR-84`（长度/格式脆弱位点等 cheap heuristic）不进入最小闭环，只在 core pilot 正且需要进一步回答“entropy 是否特殊”时再做。

## Baselines

### publication-critical baselines

- `DNB-64`
- `DNB-84`
- `RAND-84`

### 本文 reference controller

- `CARD-84`

### 明确不进入第一阶段的 baseline

- `Prophet`
- `Prism`
- `Just on Time`
- `DPad`

理由不是这些方法不重要，而是当前第一优先级不是和外部 controller frontier 打榜，而是先证明**我们自己的最小增益到底是不是假的**。如果 core pilot 过关，再进入 phase-2 外部对标。

## 评测与产物

### 主指标

- `fixed / harmed / unchanged-correct / unchanged-wrong`
- `net repaired samples = fixed - harmed`
- `CARD-84 - DNB-84` 的 paired bootstrap CI
- `CARD-84 - RAND-84` 的 paired bootstrap CI
- 3-seed sign consistency

### 轻量 trajectory 指标

只吸收 `interdisciplinary` 中最有用、争议最小的三个指标：

- `first-correct`
- `stable-correct`
- `relapse`

不把 SDT / survival analysis 升格成论文主语言，不引入沉重理论债务。

### 必交付 artifact

- `sample_manifest.json`
- `runtime_contract.json`
- `batch_probe.json`
- `per_seed_outputs/*.jsonl`
- `per_sample_audit.csv`
- `transition_matrix.csv`
- `code_failure_modes.md`
- `claim_to_asset_map.json`
- `summary.md`

## 成功标准

满足以下 4 条中的至少 3 条，才允许把 mainline 继续推进为正向稿件：

1. `CARD-84 > DNB-84` 在 GSM8K 上为正的 `net repaired samples`，且 3 seeds 至少 2 个同号。
2. `CARD-84 > RAND-84` 在 GSM8K 上为正，且 paired bootstrap CI 至少有一个不覆盖 0。
3. 高 entropy bucket 明显富集 `fixed`，而不是全局随机分布。
4. 所有主张都能在 current-only artifact 中独立复核，不依赖旧 run 口头解释。

## 失败标准

出现以下任一情况，正向 controller story 立即降级：

1. `CARD-84 <= DNB-84`
2. `CARD-84 <= RAND-84`
3. 3 seeds 符号翻转明显，主结论不稳
4. code 伤害无法被 failure localization 解释
5. summary 仍混入历史 regime、绝对路径或不可 join 的 artifact

## 拒绝的替代路线

### 1. 拒绝把 `innovator` 当主线

拒绝理由：

- “compute eligibility / selective compute policy” 太容易重新滑回 method-forward narrative
- 当前证据尚不足以支撑一个新的 policy problem definition
- 最多只能作为 discussion 语言，在 core evidence 成立后帮助包装结果

### 2. 拒绝把 `theoretical` 当主线

拒绝理由：

- observer / controller / runtime 分解非常有用，但 theorem-like ambition 超出当前资产
- `near-oracle observer`、强 identifiability 语言会抬高 reviewer 预期
- 应只保留为 claim hygiene 框架，而非独立论文中心

### 3. 拒绝把 `interdisciplinary` 当主线

拒绝理由：

- SDT / survival analysis 目前更像解释层，而不是识别层
- 如果放到主轴，会让 reviewer 觉得我们在用分析框架替代决定性负对照
- 只保留最轻的 trajectory-aware measurement

### 4. 拒绝把 `contrarian` 当独立论文

拒绝理由：

- 它是必要的守门员，但单独作为主轴只会告诉读者“哪些话不能说”，不够告诉读者“剩下什么可信”
- 最佳位置是 mainline 的强制负对照层

## 为什么这能修复 iteration-2 review blockers

### blocker 1：`+3pp` 只来自 audited slice，外推过度

修复方式：

- proposal 明确把 audited slice 结论定义为 **claim validity under audited controls**
- 不允许把 `n=96` pilot 写成 broader benchmark generality

### blocker 2：observer / controller / runtime 混写

修复方式：

- observer 只保留 raw entropy signal
- controller 只保留 `CARD-84` 这一个 reference controller
- runtime 单独写入 `runtime_contract.json`

### blocker 3：runtime fairness 不 reviewer-friendly

修复方式：

- 所有结论只使用 current-only regime
- 强制记录 max-safe batch、FA2、compile、wall-clock、NFE

### blocker 4：artifact 不自包含

修复方式：

- 强制 `claim_to_asset_map`
- 强制样本 manifest / per-sample audit / 可 join schema
- 禁止依赖绝对路径与历史 run 口头补充

### blocker 5：缺少会杀死伪结论的负对照

修复方式：

- `DNB-84` 负责排除 extra compute
- `RAND-84` 负责排除 generic perturbation
- `PERM-84` 作为必要时的 stronger sham

### blocker 6：负结果被语言吸收

修复方式：

- 提前写死 failure criteria
- code 侧若不能定位伤害，直接降级主张，不许用 “task dependence” 兜底

## 预期贡献

1. 一套适用于 training-free DLM revision claim 的最小 evidence-closure protocol。
2. 一个 reviewer-friendly 的 reference controller 报告范式：不是吹方法，而是先关掉 compute、randomness、runtime confound。
3. 一个 per-sample repair/harm 审计框架，帮助区分 reasoning gain 与 code fragility。
4. 一个清晰的 claim ceiling：当前结果若成立，最多支持 task-conditional finding；若不成立，能形成可信的 skeptical audit negative result。

## 备选方向与保留范围

我只保留两个认真备选，且都明确从属于主线结果：

1. **Backup A：skeptical audit paper**  
   如果 `CARD-84` 败给 `DNB-84` 或 `RAND-84`，直接转写成“哪些 DLM selective revision claim 在 sham-controlled matched-compute 下失效”。

2. **Backup B：trajectory-aware mechanism add-on**  
   如果 core pilot 成立，再补 `first-correct / stable-correct / relapse`，解释 reasoning 与 code 差异，但不把跨学科框架升格成主张中心。

## 视角加权说明

本轮 synthesis 的权重分配是明确倾斜的：

- **最高权重：`pragmatist + empiricist`**  
  因为二者最贴近当前证据边界，也最能直接修复 iteration-2 的 reviewer blocker。

- **第二权重：`contrarian`**  
  因为它提供了必须跨过的 falsification gate，能防止 narrative overreach 再次发生。

- **第三权重：`theoretical + interdisciplinary`**  
  只吸收其最能增强识别力、但不会明显增加 scope 的部分。

- **最低权重：`innovator`**  
  只保留语言资源，不允许驱动实验设计或主张范围。

## 一句话结论

iteration-3 的最佳提案不是“再造一个更强 controller”，而是**用 `CARD-84` 这一个最小 reference controller，把 DLM 小增益先放进 sham-controlled、matched-compute、artifact-complete 的证据闭环里；证明它还站得住，再谈更大的故事。**


## 当前可检验假设
# Iteration-3 可检验假设

## 主假设

### H1：matched-compute 下仍有 reasoning 净收益

在 `GSM8K` audited slice 上，`CARD-84` 相对 `DNB-84` 的 `net repaired samples` 为正，且 3 seeds 至少 2 个同号。

### H2：targeted revision 优于随机 revision

在 `GSM8K` audited slice 上，`CARD-84` 相对 `RAND-84` 的 `net repaired samples` 为正；否则 entropy-targeting 不成立。

### H3：observer 只在高风险区域真正有用

`fixed` 样本应在高 entropy bucket 富集；若高低 bucket 差异不明显，则 raw entropy 不能被描述为 useful triage signal。

### H4：task dependence 必须以失败定位为前提

若 reasoning gain 成立而 code 侧不成立，则 code 侧负结果必须可定位到 truncation、syntax、semantic failure 或 evaluator fragility；否则不能写 task dependence explanation。

### H5：trajectory 指标只用于解释，不用于替代主结论

若 `first-correct / stable-correct / relapse` 提供额外解释，它们只能增强 task dependence 的机制说明，不能替代 `DNB-84` / `RAND-84` 的决定性比较。

## 否证条件

### F1：extra compute 假说胜出

若 `CARD-84 <= DNB-84`，则当前 mainline 改写为 skeptical audit，不再保留正向 controller claim。

### F2：generic perturbation 假说胜出

若 `CARD-84 <= RAND-84`，则 entropy targeting 失效，不再保留 observer-guided wording。

### F3：不稳定性过高

若 3 seeds 的结论符号明显翻转，或 paired bootstrap CI 广泛覆盖 0，则只能写 audited-control 下 inconclusive。

### F4：artifact closure 失败

若 `runtime_contract.json`、`per_sample_audit.csv`、`claim_to_asset_map.json` 之间无法 join，或 summary 依赖历史 run 解释，则当前实验不能作为主文证据。


## 当前候选 idea 池（保留 2-3 个候选，必要时淘汰或替换）
{
  "candidates": [
    {
      "candidate_id": "cand_audit_mainline",
      "title": "Sham-Controlled Auditable Gain for Training-Free DLM Revision",
      "status": "front_runner",
      "summary": "以 CARD-84 作为唯一 reference controller，核心贡献是 protocol-first 的 evidence closure，而不是新的 controller family。通过 DNB-64 / DNB-84 / CARD-84 / RAND-84 的 audited-control pilot 判断增益是否真实存在并可被克制表述。",
      "hypotheses": [
        "CARD-84 在 GSM8K audited slice 上相对 DNB-84 有正的 net repaired samples",
        "CARD-84 在 GSM8K audited slice 上相对 RAND-84 有正的 net repaired samples",
        "高 entropy bucket 富集 fixed cases"
      ],
      "pilot_focus": "n=96 audited slice, 3 seeds, matched-compute, current-only runtime contract, artifact closure",
      "claim_ceiling": "最多支持 task-conditional controller evidence，不支持 general DLM gain"
    },
    {
      "candidate_id": "cand_negative_audit_pivot",
      "title": "When Training-Free DLM Revision Gains Fail Under Audited Controls",
      "status": "backup",
      "summary": "若 CARD-84 败给 DNB-84 或 RAND-84，则将论文转写为 skeptical audit negative result，贡献转向 claim hygiene、runtime fairness 与 sham-controlled falsification。",
      "hypotheses": [
        "部分看似正向的 revision gain 主要来自 extra compute 或随机扰动",
        "审计协议能比 aggregate accuracy 更早暴露伪结论"
      ],
      "pilot_focus": "保留同一 pilot，只改写论文中心问题",
      "claim_ceiling": "协议与负结果边界论文，不写正向 controller claim"
    },
    {
      "candidate_id": "cand_trajectory_addon",
      "title": "Trajectory-Aware Mechanism Add-On for Reasoning vs Code",
      "status": "backup",
      "summary": "若 core pilot 成立，则补充 first-correct、stable-correct、relapse 三个 trajectory 指标，用于解释 reasoning gain 与 code fragility 的差异。",
      "hypotheses": [
        "reasoning gain 体现在更快 stable-correct",
        "code 退化更多体现为 relapse 或输出脆弱性"
      ],
      "pilot_focus": "在 core pilot 的 per-step logs 上追加轻量分析",
      "claim_ceiling": "机制解释配件，不替代主结论"
    },
    {
      "candidate_id": "cand_selective_compute_policy",
      "title": "CARE-DLM / Compute Eligibility Policy",
      "status": "dropped",
      "summary": "把谁值得额外 compute 作为新 policy 主线的想法过早，会重新滑回 method-forward narrative。",
      "hypotheses": [
        "prospective compute triage 能形成新的 paper object"
      ],
      "pilot_focus": "需要在 core evidence 之后才能考虑",
      "claim_ceiling": "目前只允许作为 discussion 语言资源"
    },
    {
      "candidate_id": "cand_formal_identifiability",
      "title": "Observer-Controller-Runtime Formal Identifiability",
      "status": "dropped",
      "summary": "理论分层很有价值，但 theorem-like 主张与 near-oracle observer 超出当前 iteration-3 资产与 scope。",
      "hypotheses": [
        "当前实验资产足以支持 formal identifiability claim"
      ],
      "pilot_focus": "不进入当前闭环",
      "claim_ceiling": "仅保留为 conceptual framing"
    }
  ]
}


## 上一轮 validation 决策意见
# Idea Validation Decision

- Decision: `PIVOT`
- Selected candidate: `cand_negative_audit_pivot`
- Confidence: `0.90`

## 为什么

- `CARD-84` 确实击败了 compute-matched `DNB-84`，但没有通过更严格的 sham-control gate：`card84_vs_rand84` 在 GSM8K 上的 `net_repaired = 1`，低于预注册门槛 `>= 2`。
- 本轮最重要的成功不是 hero-method 胜利，而是 artifact closure 成功：四个 arm 已经 sample-level 可 join，runtime 漂移已被修复，negative control 也补齐了。
- `MBPP` 没有把 run 打坏，但它同样没有把 `CARD-84` 和 `RAND-84` 拉开；两者都停在 `0.04`，因此更安全的解释是 skeptical / audited negative result。

## 下一步

- 将 iteration 3 主线切换到 `cand_negative_audit_pivot`。
- 将 trajectory addon 视为因 negative pivot 被显式跳过，而不是“漏做”。
- 带着 `runtime_contract.json`、`per_sample_audit.csv`、`transition_matrix.csv`、`claim_to_asset_map.json` 和 `code_failure_modes.md` 继续进入后续综合/写作阶段。
- 本轮不再新开 controller family 或额外 pilot 分支，直接用 negative-control framing 推进。


## 上一轮 validation 结构化决策
{
  "decision": "PIVOT",
  "selected_candidate_id": "cand_negative_audit_pivot",
  "confidence": 0.9,
  "reasons": [
    "CARD-84 clearly beat the compute-matched DNB-84 gate on GSM8K, but it failed the stricter sham-control gate: `card84_vs_rand84` net repaired was only 1, below the required threshold of 2.",
    "The artifact-closure goal succeeded: all four arms are joinable at sample level, runtime drift was repaired, and the pilot now supports a clean skeptical read rather than a method-forward win.",
    "MBPP did not invalidate the run, but it also did not separate CARD-84 from RAND-84; both landed at 0.04, so the safer narrative is an audited negative-control result."
  ],
  "next_actions": [
    "Adopt `cand_negative_audit_pivot` as the active lane for iteration 3 and write the pilot as a skeptical audited-control result.",
    "Treat trajectory analysis as explicitly skipped by the negative pivot rather than as a missing experiment.",
    "Carry forward the repaired runtime contract, per-sample audit, transition matrix, claim-to-asset map, and code failure taxonomy into the next synthesis/writing stages.",
    "Do not reopen a new controller family or extra pilot branch inside this iteration; move the workflow forward with the negative-result framing."
  ],
  "dropped_candidates": [
    "cand_audit_mainline",
    "cand_trajectory_addon"
  ]
}
