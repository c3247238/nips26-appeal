# Methodologist - Round 1

## 总判断

从方法论角度看，`cand_diag` 当前最大的优点是：它已经不再试图用脆弱的 method-win 叙事硬撑，而是把这轮真正被证据支持的内容收敛成一个 **compute-normalized diagnostic benchmark**。这使得内部效度比上一轮 `TIGER/CARD` 主叙事明显更强。

但要把它写成能说服审稿人的结果链，还需要非常谨慎地处理四个边界：

1. 这些 full-cycle 结果目前仍然是 `GSM8K 100 / MATH500 100 / HumanEval 50` 级别，而不是完整 benchmark；
2. `diag_compute_curve_gsm8k` 混合了不同 backend / batch size / compile 配置，结论是可信的，但不能被写成“纯算法差异”；
3. `diag_signal_gap_audit` 里一部分“observer vs controller”结论来自不同资产拼接，而不是单一统一 protocol 下的同分布对照；
4. `diag_math500_shortlist` 和 `diag_humaneval_guard_boundary` 更适合支持“task-dependent / boundary-only”结论，不足以单独支撑广泛外推。

换句话说：这条线现在**可以写，但必须诚实地写成 diagnostic evidence，而不是 universal law**。

## 1. 基线公平性

### 正面评价

- `diag_compute_curve_gsm8k` 明确把 `Standard / DNB / Prophet / CORE-proxy / Entropy / TIGER` 放到统一的 `actual_nfe / latency / tokens_per_sec` 口径下比较，这比只报 nominal steps 公平得多。
- 结果里保留了 `batch_size / attention_backend / compile_enabled`，这是很重要的，因为它承认了系统实现差异本身就是“真实 compute”的一部分。
- `CORE-proxy-64` 被明确标注为 `proxy`，而不是假装成完全同类方法，这在 framing 上是诚实的。

### 方法论风险

- 当前 baseline 并不是“同工程栈公平”。`Standard-64` 和 `DNB-84` 使用了 `batch_size=115 + compile_enabled=true`，而 `Entropy/TIGER` 是 `batch_size=32 + compile_enabled=false`，`CORE-proxy` 更是 `batch_size=1 + compile_enabled=false`。  
  这意味着 `latency` 和 `tokens/sec` 同时反映了：
  - 方法本身的 compute 结构
  - 实现成熟度差异
  - 工程优化是否启用
- 因此 `diag_compute_curve_gsm8k` 非常适合支持：
  - “headline nominal compute 不足以代表真实成本”
  - “真实 Pareto 排序会变”

  但不适合写成：
  - “某方法 intrinsically 更慢/更快”

- `CORE-proxy` 的最优精度来自 `accuracy=0.46`，但它的 `latency_sec=482.953` 且 `batch_size=1`。如果论文只用它做“accuracy upper envelope”是合理的；如果拿它和 revision family 做 production-level 公平比较，就会引来实现偏置质疑。

### 建议

- 论文正文必须区分两类公平性：
  - `algorithmic fairness`: matched actual compute
  - `systems fairness`: backend / batch / compile 一致化程度
- 最好在 appendix 放一张 runtime config 表，把每个方法的 `batch_size / compile / backend / peak_vram` 全部列出来。

## 2. 指标是否真正测到我们声称的东西

### 当前指标是合理的部分

- 对 reasoning：
  - `accuracy`
  - `actual_nfe`
  - `latency_sec`
  - `tokens_per_sec`

  这组指标足够支撑 “quality-speed-compute tradeoff”。

- 对 code boundary：
  - `pass@1`
  - `syntax_failure_rate`
  - `runtime_failure_rate`

  这组指标正好能支撑 proposal 里的 “shallow vs deep failure”。

- 对 observer/control gap：
  - `diagnostic_score`
  - `control_effectiveness`
  - `control_delta_vs_standard`

  这组指标足够支撑 “observer 不等于 controller” 的核心论点。

### 指标边界

- `diag_signal_gap_audit` 中的 `diagnostic_score` 不是统一定义：
  - calibration 用的是 held-out `|entropy_error_corr|`
  - entropy / instability 用的是 signal screen correlation

  这在分析上可以接受，但不能被包装成完全同尺度、同实验协议的 apples-to-apples score。

- `control_effectiveness` 也不是统一实验条件下的 end-to-end policy gain：
  - calibration 没有 deployed control policy，因此是 `0.0`
  - entropy / instability 来自 signal screen / shortlist 资产复用

  所以“calibration strongest observer”可以说，但“calibration controller failed”应当写成：
  - 目前未形成可验证的直接控制策略

- `diag_math500_shortlist` 的主要价值是排序变化和 task dependence，而不是绝对精度水平本身。因为 100-sample shortlist 还不足以给出稳健 benchmark ranking。

### 建议

- 正文里的 claim 应该写成：
  - “the diagnostics and controls are misaligned”
  
  而不是：
  - “we measure the exact causal gap between observation and intervention”
- 对 `signal_audit` 的表格脚注里应明确说明不同 signal 的 diagnostic score 来源并不完全同构。

## 3. 评测协议审计

### 当前协议强项

- `methodology.md` 已经明确把本轮目标限定为 “最小必要新增实验”，避免了上一轮那种为了保方法叙事而无边界扩展的风险。
- `diag_compute_curve_gsm8k` 和 `diag_signal_gap_audit` 大量复用了已有资产，这是好事，因为它减少了重复计算，并让诊断论文更聚焦。
- `HumanEval` 被降级成 boundary / appendix 证据，避免 code 负结果继续污染 headline claim，这是协议层面的正确收缩。

### 当前协议薄弱点

- `summary.md` 缺失，说明 full-cycle 结果并没有被整理成单一规范摘要文件。对自动化 pipeline 来说这不是致命问题，但对可复核性不利。
- `diag_signal_gap_audit` 使用多份异质资产拼接：
  - `diagnostic_calibration_heldout.json`
  - `tiger_signal_screen.json`
  - `gsm8k_main_shortlist.json`

  这使结论更像 synthesis 而非单协议实验。如果正文不明确写出这一点，审稿人会质疑是否存在 selection bias。
- 当前结果仍然是单 seed。虽然这轮主张比上一轮更偏诊断、较少强调小差值 superiority，但：
  - `Entropy = TIGER = 0.39`
  - `Standard = 0.36`
  - `MATH500 CORE = 0.21 vs Standard = 0.23`

  这些小差值都不适合被过度解释。

### 结论边界

从协议角度，我支持以下写法：

- “we observe stable qualitative patterns”
- “we use the full-cycle slice to establish diagnostic mismatches”
- “the benchmark is designed to reveal ranking reorders and failure boundaries, not to claim final leaderboard numbers”

我不支持以下写法：

- “method X definitively beats method Y on reasoning”
- “the observer/controller law is established generally”

## 4. Ablation 完整性

### 已经足够的部分

- 这轮不再要求像 method paper 那样做大量组件消融，因为 `cand_diag` 的主贡献不是新方法。
- `diag_humaneval_guard_boundary` 已经把 gating 的“浅层修复 vs 深层失败”拆开，足以支持 appendix 定位。

### 仍不完整的部分

- Proposal 里强调的 `benefit buckets` 目前还没有真正落地成结果文件。  
  这意味着：
  - “draft-correct-then-harmed”
  - “draft-wrong-then-fixed”
  - “revision-no-effect”

  还只是写作计划，不是现有证据。

- `diag_signal_gap_audit` 已经初步支持 H3，但没有真正把 `signal -> benefit bucket` 做成统一表格或 confusion-style breakdown。
- `diag_math500_shortlist` 只回答了 transfer / ranking change，还没有回答“why this benchmark differs”。

### 建议

- 在写作前至少再补一张或一个 JSON：
  - per-sample benefit bucket summary
  - 按 signal quantile 的 bucket 分布

如果没有这一步，`Revision phase diagram` 会停留在 framing，而不是 evidence。

## 5. 可复现性评估

### 正面

- 结果 JSON 记录得相对完整，尤其是：
  - `actual_nfe`
  - `latency_sec`
  - `tokens_per_sec`
  - `batch_size`
  - `attention_backend`
  - `compile_enabled`

  这对复现实验非常关键。

- `methodology.md` 对 batch probing、backend fallback、multi-GPU task parallel 的约束写得很清楚。

### 不足

- 当前 workspace 缺少一个统一的 `summary.md / summary.json`，导致读者需要手动拼接多个结果文件。
- `diag_signal_gap_audit` 的资产依赖关系虽然写在 JSON 里，但如果没有更明确的“asset lineage”说明，外部复现者很难知道应该先跑哪些前置任务。
- `current/plan/task_plan.json` 仍然保留 pilot 字段和 100-sample 设定，这会让人误解 full-cycle 与 pilot-cycle 的边界。

### 结论

这组结果对“项目内部复现”已经够用了，但对“外部读者一键复现”还不够。论文里如果想要提高 reproducibility score，必须再补：

1. 结果资产依赖图
2. runtime config 总表
3. full-cycle 与 pilot-cycle 的明确区分

## 6. 我给主系统的具体建议

### 可以强写进主文的

- honest compute accounting 改变排序与 Pareto 叙事
- observer 与 controller 之间存在稳定错位
- code guard 更像 shallow safeguard，而不是 general recovery mechanism
- revision effect 是 task-dependent 的

### 只能谨慎写的

- 某个具体 revision 方法优于另一方法
- 某个 signal 在所有任务上都是更好 controller
- MATH500 的绝对 method ranking

### 我建议马上补的最小方法论增强

1. 做一个 `benefit_bucket_audit.json`，把 proposal 里的 phase diagram 从 framing 变成证据。
2. 生成一张 runtime appendix table，统一列 `batch_size / backend / compile / peak_vram / actual_nfe / latency`。
3. 在主文明确声明：
   - 本轮 full-cycle 仍是 diagnostic slice，不是最终 leaderboard-scale benchmark。
4. 如果还有一点预算，优先补一个 seed sensitivity spot-check，而不是再发明新 revision policy。

## 最终裁决

从方法论立场，我支持继续沿 `cand_diag` 前进，因为它现在的主张与现有证据是基本一致的；但前提是我们必须把论文写成：

**“诚实 compute + 任务依赖 + observer/controller 错位”的诊断论文**

而不是再次滑回：

**“我们提出了一个普适更强的 revision 方法”**。

## Round 2

在新的 diagnostic framing 下，我认为下面这些 claim 已经足够方法论安全，可以进入主文主叙事：

- `honest compute accounting changes the story`：这一点已经有直接证据支持，因为 `diag_compute_curve_gsm8k.json` 明确展示了 nominal 与 actual compute 排序不一致，且 `CORE-proxy-64` 相对 `Entropy-Revise-64+3` 发生了关键重排。
- `observer != controller`：这一点也已经基本安全，因为 `diag_signal_gap_audit.json` 里 calibration 与 entropy 都呈现出“诊断强、控制弱”的稳定错位，这个 claim 不依赖某一个方法必须赢。
- `code guard is boundary evidence, not a recovered method story`：这一点已经足够安全，因为 `diag_humaneval_guard_boundary.json` 显示 syntax failure 明显下降，但 `pass@1` 仍未超过 `Standard`，runtime failure 甚至更差。
- `revision effects are task-dependent enough that a single benchmark is misleading`：这条可以写，但建议措辞保持“task-dependent evidence exists”而不是“we fully mapped the response surface”，因为我们目前只有 `GSM8K-100` 和 `MATH500-100` 两个 reasoning slices。

下面这些 claim 目前还只能作为 working hypothesis，不能写成已经被严格建立的结论：

- `we have a full revision phase diagram`：现在还没有，因为 proposal 中承诺的 benefit buckets 还没真正落盘成结果文件。
- `specific signals predict revision benefit better than final correctness under a unified protocol`：方向是对的，但现有 `diag_signal_gap_audit` 仍然是异质资产拼接，尚未在统一 per-sample protocol 下完全闭环。
- `MATH500 validates the same mechanism as GSM8K`：现在只能说 MATH500 提供了 informative transfer evidence，而不能说它已经稳健复现了同一个机制。
- `CORE / Entropy / TIGER` 的精确相对排名已经稳定：不能这么写，因为当前仍是 100-sample slice、单 seed，且工程配置并不统一。

我认为最小必补的证据项只有 4 个，而且都非常具体：

1. `benefit_bucket_audit.json`：把 `draft-correct-then-harmed / draft-wrong-then-fixed / no-effect` 真正做成 per-sample 统计，这是把 diagnostic framing 从叙事升级成证据的最关键一步。
2. `runtime_config_table.json` 或同等 appendix 表：统一列出每个方法的 `batch_size / compile / backend / peak_vram / actual_nfe / latency / throughput`，防止 compute 结论被误读成纯算法差异。
3. 一个最小 `seed_sensitivity_spotcheck`：不需要全任务三 seed，只要对 1-2 个最关键比较做 spot-check，就能显著降低“小差值被单 seed 放大”的方法论风险。
4. 一个 `asset_lineage_summary`：把 `diag_signal_gap_audit` 依赖哪些前置资产、每个 score 来自哪个 protocol 说清楚，否则 observer/controller 的论点会被质疑为 synthesis artifact。

## Round 3

### 1. 现在最安全的 claim

最安全的 claim 是：`honest compute accounting` 会改变 DLM revision 方法的排序叙事，而 `observer` 与 `controller` 之间存在稳定错位。这个结论已经同时被 `diag_compute_curve_gsm8k` 和 `diag_signal_gap_audit` 直接支持，而且不依赖某个新方法必须取胜。

### 2. 现在仍然只能作为 working hypothesis 的 claim

仍然只能作为 working hypothesis 的是：我们已经完整刻画了 revision 的 phase diagram，或者某个 signal 在统一协议下稳定预测 revision benefit。现有证据已经指向这个方向，但还没有把 per-sample benefit buckets 真正做成闭环证据。

### 3. 你要求的唯一必补证据

唯一必补的是一个 `benefit_bucket_audit` 结果文件。只要把 `draft-correct-then-harmed / draft-wrong-then-fixed / no-effect` 三类样本真正统计出来，这篇 diagnostic framing 才算方法论上闭环。
