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
