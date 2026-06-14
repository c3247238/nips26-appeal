# 实验方法规划：REVISION ATLAS（Iteration 1, Refinement Round 2）

## 1. 本轮目标

本轮 planning 不再继续放大 `TIGER` 的方法主张，而是围绕已经被证据支持的主候选 `cand_diag` 组织下一轮实验：

- 主线：`REVISION ATLAS`
- 贡献类型：`compute-normalized diagnostic benchmark`
- 核心问题：
  - revision 何时有用
  - revision 何时有害
  - 为什么某些 signal 适合做诊断，却不足以做控制

## 2. 直接证据基础

本轮 planning 不是从零开始，而是建立在以下已完成结果上：

1. `gsm8k_main_shortlist`
   - `TIGER = Entropy-Revise` on GSM8K shortlist
   - `CORE-proxy > TIGER > Prophet / DNB`
2. `tiger_gating_boundary`
   - gating 降低 syntax failure
   - 但没有恢复到 Standard code baseline，且 reasoning 略掉点
3. `diagnostic_calibration_heldout`
   - calibration / entropy 具有诊断价值
   - 但未形成可信的方法增益来源
4. `final_pareto_synthesis`
   - 已明确建议 pivot 到 `cand_diag`

因此，本轮只设计**最小必要新增实验**，不再重开大规模方法搜索。

## 3. 技术约束落实

### 3.1 Batch / throughput 约束

用户要求所有实验尽量使用最大 batch size。下一轮所有任务统一要求：

- `max_batch_size_hint = auto-detect`
- 先做 batch probing，再正式跑 benchmark
- 默认左填充批量推理
- 正式实验不接受 `batch_size = 1`，除非明确记录 blocker

### 3.2 Flash Attention 约束

根据 `flash-attention` 技能与本项目已有经验，本轮采用如下顺序：

1. 优先尝试 `flash_attention_2`
2. 若安装或数值兼容失败，不回退到 `sdpa`
3. 直接回退到项目已验证可用的 `eager`

必须记录：

- `attention_backend`
- `compile_enabled`
- `tokens_per_sec`
- `peak_vram_mb`

### 3.3 lm-evaluation-harness 约束

根据 `lm-evaluation-harness` 技能与本轮 `protocol_repair` 结果：

- 优先复用 `GSM8K / MATH500 / HumanEval` 的标准任务定义、prompt 约定和 metric 口径
- 不强行把当前 DLM 自定义采样器包装成 `lm_eval` adapter
- 若后续能低成本接入，再作为 bonus；当前默认路径仍是自定义 evaluator + 标准 benchmark conventions

### 3.4 Multi-GPU 规划

尽管单个 DLM 评测任务通常单卡即可，但本轮应通过**任务级并行**来充分利用多卡：

- GPU0: `diag_compute_curve_gsm8k`
- GPU1: `diag_signal_gap_audit`
- GPU2: `diag_math500_shortlist`
- GPU3: `diag_humaneval_guard_boundary`

每个任务 `gpu_count = 1`，`multi_gpu_strategy = single`，但由调度器并行派发。

## 4. Benchmarks 与方法集合

### 4.1 Reasoning benchmarks

- `GSM8K`
  - 主 benchmark
  - 用于 matched-compute 排序与 signal/control gap 分析
- `MATH500`
  - 第二 reasoning benchmark
  - 用于验证 task dependence 不是 GSM8K 特例

### 4.2 Code boundary benchmark

- `HumanEval`
  - 仅作为 code fragility / syntax-guard appendix 证据

### 4.3 方法集合

主表优先保留：

- `Standard-64`
- `DNB-84`
- `Prophet-64`
- `CORE-proxy-64`
- `Entropy-Revise-64+3`
- `TIGER-64+3`

本轮新增实验不再以“找最强方法”为目的，而以“解释排序变化与失败边界”为目的。

## 5. 新一轮最小实验集

### Task A: `diag_compute_curve_gsm8k`

目标：

- 直接回答 honest compute accounting 是否改变结论
- 补上上一轮被批评最严重的缺口：名义 NFE 与实际 compute 的错配

核心输出：

- matched-compute GSM8K 表
- nominal vs actual NFE 差异
- 速度-质量 Pareto 图

### Task B: `diag_signal_gap_audit`

目标：

- 定量展示 `predictive signal != effective intervention`
- 比较 entropy / instability / calibration 的：
  - error correlation
  - revision-benefit correlation
  - control success gap

核心输出：

- correlation 表
- signal-vs-benefit 散点图
- 失败案例切片

### Task C: `diag_math500_shortlist`

目标：

- 在第二 reasoning benchmark 上确认：
  - GSM8K 的排序是否迁移
  - diagnostic story 是否更稳固

核心输出：

- MATH500 shortlist 表
- GSM8K vs MATH500 排序对比

### Task D: `diag_humaneval_guard_boundary`

目标：

- 把 code 结果固定为边界证据
- 检查 guard 是否只是减少 syntax failure，而非真正恢复整体性能

核心输出：

- Standard / Entropy / TIGER / Gated TIGER 的 boundary comparison
- parse/runtime failure breakdown

## 6. 决策规则

### 继续沿 `cand_diag` 写论文的条件

满足以下三条中的两条即可：

1. `diag_compute_curve_gsm8k` 显示 matched actual compute 的确改变至少一个关键比较结论
2. `diag_signal_gap_audit` 显示至少一个 signal 存在“诊断强、控制弱”的明显错位
3. `diag_math500_shortlist` 或 `diag_humaneval_guard_boundary` 强化了“task-dependent revision”叙事

### 重新打开方法主线的条件

只有在新增实验出现以下强信号时才重新考虑 method-forward 叙事：

1. 某个 signal-guided revision 明显超过 `Entropy-Revise`
2. 该优势在 `MATH500` 上也成立
3. code boundary 不再明显弱于 `Standard`

否则保持 `cand_diag`。

## 7. Expected Visualizations

- Table 1: matched-compute reasoning comparison (`GSM8K`, `MATH500`)
- Figure 2: quality-speed Pareto frontier under actual NFE / latency
- Figure 3: diagnostic-control gap scatter (`signal quality` vs `revision gain`)
- Figure 4: code boundary failure breakdown (`syntax`, `runtime`, `pass@1`)
- Appendix table: backend / compile / batch-size / throughput configuration

## 8. 风险与缓解

### 风险 1：诊断论文过散

缓解：

- 主文只围绕三条 claim 写作
- 所有图表都服务 `predictive signal != effective intervention`

### 风险 2：又被 code appendix 拖偏

缓解：

- HumanEval 只保留为 boundary
- 不再把 code 结果作为 headline

### 风险 3：新增实验过多

缓解：

- 只保留 4 个高信息量任务
- 默认复用已有结果，不重跑已足够明确的 baseline
