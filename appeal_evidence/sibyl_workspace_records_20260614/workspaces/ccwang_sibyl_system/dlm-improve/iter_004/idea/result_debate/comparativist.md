# Comparativist

## 说明

- 本轮本应做至少两次 arXiv 直接查询，但 arXiv API 当前返回 `HTTP 429`，因此我改用两类替代证据：
  - workspace 内已有的 `context/literature.md`
  - Web 上的官方/一手页面（主要是 arXiv、OpenReview、项目页）
- 这不会改变核心判断，但意味着“最新竞争工作”部分是基于一手页面与本地 literature note 的保守定位。

## 1. SOTA 对比

- 如果把比较对象设成**整个 GSM8K SOTA**，当前结果并不接近前沿。`cand_espd=0.4041` 在现代 AR reasoning 模型面前没有 headline 竞争力，因此这条线不能按“benchmark leader”来包装。
- 但如果把比较对象收窄到**training-free / discrete diffusion LM 的 test-time intervention 与 compute-routing**，定位就不同了。workspace 的 literature note 已经列出几条 2025-2026 邻近工作：
  - `SOAR` (2025): diffusion LM 的 test-time scaling / reasoning加速线
  - `Prism` (2026): hierarchical search + self-verification for discrete diffusion LMs
  - `CD4LM` (2026): confidence-guided direct decoding for diffusion language models
  - `MetaState` (2026): 为 discrete diffusion LM 注入 persistent working memory
  - `Sampler-Centric Evaluation` (2026): 强调 sampler/runtime attribution 不能和方法增益混写
- 在这组邻近工作里，当前结果最像的不是“新 SOTA”，而是**一条经过 matched sham-control 检验的 modest but honest compute-routing result**。

## 2. Contribution Margin

- 当前真实 contribution margin 不在于“绝对分数”，而在于**归因边界**：
  - `cand_espd` 相对 shared controls 只高 `+0.61pp ~ +0.76pp`
  - 但相对 matched sham `ESPD-FixedFrontier` 同时保住了 `+0.53pp` 质量差与 `+18.69 tok/s` 速度优势
- 这意味着它的贡献不是“我们比所有 baseline 都快、都准”，而是：
  - 在统一 runtime contract 下，entropy-routed compute 相比 fixed-frontier sham 展现出可信但温和的 gain
  - 这个 gain 小，但经过了 reviewer 更关心的 sham-control 检验
- 换句话说，贡献边际更像**method attribution hygiene + bounded positive result**，而不是 benchmark-dominating method paper。

## 3. Concurrent Work Check

- 最近的 discrete diffusion LM literature 明显在向 test-time scaling / search / guided decoding 方向移动。`Prism`、`SOAR`、`CD4LM` 都说明 reviewer 不会把“inference-time compute reallocation for dLLMs”视为冷门题。
- 但这些工作大多追求更强任务收益、更复杂 search 或更重 guidance；而当前 workspace 的价值在于，它把 claim 收得更窄，并且更重视 sham control 与 runtime contract。
- `MetaState` 代表了另一个重要对照方向：通过 persistent memory 或训练式增强去提升 dLLM 表现。相较之下，当前工作如果坚持 training-free framing，就必须明确自己的优势不是“结果更强”，而是“代价更轻、归因更干净”。
- `Sampler-Centric Evaluation` 这一类工作对当前 narrative 非常关键：它们实际上在强化本文最值得讲的点，即 **small-gain dLLM 结果必须把 sampler/runtime/method attribution 分开**。

## 4. Novelty

- 若把 novelty claim 写成“我们提出了新的强力 dLLM revision method”，当前证据不够。
- 若把 novelty claim 写成“我们首次证明 entropy 在 dLLM 里是正确的 semantic controller”，当前证据也不够，因为 shared controls 内 `CARD-84` 并未稳定赢过 `RAND-84`。
- 当前最稳的 novelty 写法应是：
  - **在 training-free discrete diffusion LM setting 下，提出并 full-scale 验证了一条 entropy-routed compute 的 modest positive line；同时通过 fixed-frontier sham 证明 gain 不能被简单 frontier budget 复制。**
- 也就是：novelty 不在“大效果”，而在**把 observer signal 从 semantic controller 重新定位为 compute router，并用 honest controls 把这个边界说清楚。**

## 5. Publication Readiness

- 以当前证据强度，我不认为它已经是标准 NeurIPS / ICML main-track 的强方法稿。
- 更现实的定位有三种：
  - 一篇强调 **bounded positive result + strong attribution discipline** 的 systems-for-ML / evaluation-oriented paper
  - 一篇把结果写得足够诚实的 workshop / spotlight-scale paper
  - 更大的主稿中的一条 speed-line subsection，而不是唯一 headline
- 若要冲更高层级 venue，至少还需要：
  - 一个额外 benchmark 或模型上的外推验证
  - 更完整的 runtime-lineage artifact
  - 更强的机制 ablation，拆开 routing 与 stopping

## 6. 缺失对比

- 缺少和更近邻的 dLLM test-time scaling 工作的**同 benchmark、同指标、同 compute 口径**对比。当前 literature positioning 有了，但 quantitative bridge 还不够。
- 缺少 plain draft / no-revision 对照，导致 contribution margin 无法锚定“revision family 整体到底值不值得”。
- 缺少跨任务对比。proposal 明确提了 `MBPP`，而当前只有 `GSM8K`，这会让 reviewer 怀疑结果是否只是算术文本任务上的特例。
- 缺少对 object-level line 的直接比较。由于 proposal 把 `cand_bsr` 设为 front-runner，而 full-scale 只展示 `cand_espd`，读者自然会问：为什么当前最完整的证据没有落在原 front-runner 上？

## 结论

当前结果**不具备 benchmark-level SOTA 意义**，但它具备一个更真实、也更容易被严肃 reviewer 接受的价值：  
**在 dLLM test-time scaling 已经进入竞争期的背景下，当前工作给出了一条经 sham-control 审查后仍站得住的 entropy-routed compute 证据链。**

如果后续再补一个外推验证和一份 reviewer-friendly runtime-lineage artifact，这条线就能从“诚实的小结果”升级成“有明确 contribution margin 的 submission-grade speed-line”。
