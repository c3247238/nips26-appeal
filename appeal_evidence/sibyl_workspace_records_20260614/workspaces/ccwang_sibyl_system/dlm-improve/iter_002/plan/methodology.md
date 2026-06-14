# Iteration 2 Methodology

## 目标

本轮不是继续寻找新的 hero controller，而是把论文收束为一篇 **compute-normalized diagnostic / protocol paper**，并用最小新增实验封口当前最关键的 reviewer 风险：

1. `benefit buckets` 缺失  
2. headline comparison 的最小不确定性封口缺失  
3. runtime fairness / asset lineage artifact 缺失  
4. `d(s)` / `g(s)` 定义和 observer-controller split 不够显式

## 候选方向与规划决策

### `cand_protocol`

- 目标：把 honest compute fairness、observer/controller split、canonical asset lineage 做成主贡献。
- 主要交付：
  - `canonical_asset_manifest.json`
  - `runtime_fairness_appendix_inputs.json`
  - `signal_audit_protocol.md`

### `cand_bucket`

- 目标：把 revision 的总体收益拆解为 `fixed / harmed / no-effect`，并与 reasoning / code / structure boundary 连接。
- 主要交付：
  - `benefit_bucket_audit.json`
  - `benefit_bucket_summary.md`

### `cand_min_controller`

- 目标：只在必要时做一个最小 diagnostic decoupling probe，验证 observer quality 不自动推出 controller gain。
- 地位：可选补强，不是主线，不允许扩写成新方法论文。

## 输入资产

- 主要沿用 `iter_001` 已产出的诊断与边界结果，尤其是：
  - `diag_compute_curve_gsm8k.json`
  - `diag_signal_gap_audit.json`
  - `diag_math500_shortlist.json`
  - `diag_humaneval_guard_boundary.json`
  - `final_pareto_synthesis.json`
- 本轮新增实验默认只服务于封口，而不是重新开辟大规模 benchmark 宽度。

## 实验策略

### 1. Runtime-first setup

- 在任何新实验前先做 runtime setup probe：
  - batch size auto-detect
  - `attn_implementation="flash_attention_2"` 优先，若环境不支持则记录 fallback 到 PyTorch SDPA
  - `torch.compile` on/off 对比
  - 双卡拆分可行性检查
- 目标不是追求极限优化论文，而是避免再次出现 `batch=1 / compile=false` 造成的比较污染。

### 2. Bucket-first evidence

- 优先在最关键的 GSM8K headline pair 上产出 per-sample bucket 审计。
- 核心字段：
  - `sample_id`
  - `draft_correct`
  - `revised_correct`
  - `bucket`
  - `task_slice`
  - `controller_budget`
  - `runtime_metadata_ref`
- 如果 bucket 结果已经足够解释 aggregate gain，则不扩更多 controller family。

### 3. Minimal uncertainty closure

- 不做 multi-seed cross-validation，也不追求传统显著性检验。
- 只做一个 **narrow seed spot-check**：
  - 仅在 headline GSM8K pairwise comparison
  - 仅检查 gain 方向与 ranking 是否稳定
  - 若方向不稳定，则同步下调 paper claim

### 4. Optional decoupling probe

- 仅当 bucket audit + runtime appendix 完成后仍无法清晰支撑 observer/controller split 时，才执行。
- 要求：
  - matched compute
  - same runtime path
  - minimal implementation delta

## Baselines

- `Standard-64`
- `Entropy-Revise-64+3`
- `CORE-proxy-64`
- 可选最小 controller baseline（仅在 `cand_min_controller` 中启用）

## Benchmarks

- 主 benchmark：GSM8K
- 边界复核：HumanEval / MATH500 只用于已有结论的 boundary positioning，不优先扩样本

## Metrics

- Accuracy / pass@1 / syntax / runtime（视任务）
- `actual_nfe`
- latency
- batch size
- compile / backend / attention mode
- bucket counts and proportions
- gain direction stability in seed spot-check

## 技术实现约束（来自相关技能）

### Flash Attention / memory efficiency

- 优先启用 `flash_attention_2`
- 若不可用，退回 PyTorch SDPA，并在 artifact 中显式记录
- 对 attention backend 做一次正确性 smoke check，确保与 baseline 数值偏差在可接受范围内

### Evaluation harness

- `lm-evaluation-harness` 作为 benchmark task bookkeeping 和 dataset standardization 参考
- 由于 DLM 自定义 inference loop 不一定能直接接入 harness，主执行路径仍允许保留现有评估代码
- 但 task naming、benchmark split 和输出 schema 应尽量对齐标准 benchmark 习惯

## 风险与应对

- 如果 runtime probe 显示 flash / compile 不稳定，则保留 fallback，但必须把 fallback 写入 manifest。
- 如果 seed spot-check 显示 headline direction 不稳定，则 planning 后续自动收紧 abstract / conclusion claim。
- 如果 bucket audit 只给出很弱的机制信号，则 paper 中将 “failure taxonomy” 降级为 “bucket-level audit”。

## Expected Visualizations

- Architecture diagram: observer / controller / runtime path split
- Table 1: compute-normalized headline comparison with runtime metadata
- Figure 2: benefit bucket stacked bar (`fixed / harmed / no-effect`)
- Table 2: runtime fairness / asset lineage appendix table
- Figure 3: optional seed spot-check direction stability summary
