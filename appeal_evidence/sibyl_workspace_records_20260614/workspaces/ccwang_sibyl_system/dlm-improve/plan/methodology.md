# 实验方法规划：TIGER（Iteration 1）

## 1. 本轮目标

本轮不再延续旧 `CARD` 主叙事，而是执行新的 reasoning-first 方案：

- 方法主线：`TIGER = Task-Gated Instability-Guided Editing`
- 研究重点：`instability / context brittleness` 是否比 `raw entropy` 更适合做 revision target selection
- scope 控制：代码任务只作为边界验证，不再充当主结论支柱
- calibration 角色：仅保留为 **held-out diagnostic**，不再作为主要增益来源

## 2. 迭代硬约束

上一轮和文献迭代给出的硬约束，本轮全部写死：

1. **先吞吐优化，再做科学判断**
   - 先做 batch-size probing
   - 优先启用 `flash_attention_2`；若不可用，退回 `sdpa`
   - 对 stable forward 启用 `torch.compile`
   - 左填充批量推理必须默认开启
2. **held-out calibration split 是强制要求**
   - calibration split 不能和 headline evaluation 样本重合
3. **actual compute accounting 是强制要求**
   - 每个结果必须记录 `actual_nfe`, `latency_sec`, `tokens_per_sec`, `batch_size`, `attention_backend`, `compile_enabled`
4. **强基线顺序固定**
   - required: `CORE`, `Prophet`, `DNB`
   - method sanity baseline: `raw-entropy revision`
   - optional after main line is stable: `DCD`
5. **第二 reasoning benchmark 必须做**
   - 优先 `MATH500`
6. **HumanEval 仅作为 boundary/gating pilot**
   - 只有 cheap syntax guard 成本足够低时，才允许把 guarded revision 放进 code pilot

## 3. 模型、数据与现有资产

### 3.1 主模型

- `LLaDA-8B-Instruct`
  - 主力模型
  - 单卡 97GB VRAM 足够承载大 batch 推理

### 3.2 Benchmarks

- `GSM8K`
  - 主 benchmark
  - 用于方法筛选和 headline result
- `MATH500`
  - 第二 reasoning benchmark
  - 用于验证是否不是单 benchmark 偶然收益
- `HumanEval`
  - 边界验证
  - 只验证 gating 是否阻断 code 退化

### 3.3 必须复用的现有代码

- `iter_001/exp/scripts/batched_dlm_utils.py`
  - 左填充批量推理
  - batch-size binary search
  - `flash_attention_2` / fallback
  - optional `torch.compile`
- `iter_001/exp/scripts/setup_env_verify.py`
  - 环境、显存、模型加载、attention backend 验证
- `iter_001/exp/scripts/full_llada_gsm8k.py`
  - 可复用 GSM8K 全量评测框架与 answer extraction
- `iter_001/exp/scripts/full_llada_humaneval.py`
  - 可复用 HumanEval 边界 pilot 与 execution-based evaluation

## 4. 技术约束落实

### 4.1 来自 `flash-attention` 技能的具体落地

- 优先顺序：
  1. `flash_attention_2`
  2. PyTorch `scaled_dot_product_attention` (`sdpa`)
  3. eager fallback
- 必做验证：
  - attention backend 记录到结果文件
  - 与 baseline 输出数值一致性做一次 smoke test
  - profiling 至少确认吞吐或显存确有改善

### 4.2 来自 `lm-evaluation-harness` 技能的具体落地

- 优先复用 benchmark 的标准任务定义、prompt 约定、metric 口径
- 由于 DLM 自定义采样循环不一定能直接接入 `lm_eval`，本轮策略为：
  - **优先复用 benchmark/task conventions**
  - 若能干净包装为 harness model adapter，再接 `lm_eval`
  - 若不能，则保留自定义 evaluator，但需在 `protocol_repair` 中明确记录“不直接使用 harness 的原因”

## 5. 评测协议修复

本轮在任何 full campaign 前，先修复协议：

1. **held-out calibration split**
   - 从 `GSM8K train` 划出固定 calibration subset
   - headline `GSM8K test` 不得参与 calibrator 拟合
2. **answer extraction audit**
   - GSM8K exact-match 抽取规则单独审计
   - HumanEval completion extraction 保持 execution-safe
3. **compute logging wrapper**
   - 所有方法统一输出：
     - `actual_nfe`
     - `latency_sec`
     - `tokens_per_sec`
     - `batch_size`
     - `peak_vram_mb`
     - `attention_backend`
     - `compile_enabled`
4. **batching contract**
   - 所有推理实验默认 `max_batch_size_hint = auto-detect`
   - 不允许 `batch_size=1` 进入正式 benchmark，除非明确记录 OOM blocker

## 6. 基线集合

### 6.1 Required baselines

1. `Standard-64`
2. `DNB-84`
   - 作为 honest compute-matched baseline
3. `Prophet`
   - 作为早停 / early-commit 强基线
4. `CORE`
   - 作为 strongest revision-style competitor
5. `Raw-Entropy Revision`
   - 作为旧方向的直接 sanity baseline

### 6.2 Optional baseline

6. `DCD`
   - 若前述 required baselines 已稳定复现，再追加

## 7. TIGER 方法定义（当前最小可行版本）

TIGER 的最小实现不追求花哨，而是聚焦三个部件：

1. **instability score**
   - 通过局部上下文扰动或轨迹脆弱性近似，给出 token ranking
2. **selective revision**
   - 只在中后期对少量 token 触发 revision
3. **task gating**
   - reasoning 允许 revision
   - code 默认关闭 revision；只有 cheap syntax guard 通过 smoke test 才允许 guarded revision

calibration 仅用于：

- 分析 raw entropy 为什么失效
- 分析 reasoning / code 差异
- 分析 instability 与 confidence ranking 的差别

## 8. 实验阶段

### Phase 0: Throughput Optimization

目标：把单卡 97GB VRAM 真正吃满，避免再次出现 `batch_size=1`。

输出：

- safe max batch size
- 可用 attention backend
- compile on/off 对吞吐的影响
- 统一推理 runtime contract

### Phase 1: Protocol Repair

目标：在开始 baseline/method 前修好所有会污染结论的协议问题。

输出：

- held-out calibration split manifest
- answer extraction audit
- compute logging schema

### Phase 2: Strong Baseline Reproduction

目标：先知道强基线在当前 pipeline 里是什么水平，再决定 TIGER 是否值得继续。

顺序：

1. `Standard-64 + DNB-84`
2. `Prophet`
3. `CORE`
4. `Raw-Entropy Revision`
5. `DCD`（optional）

### Phase 3: TIGER Screening

目标：用最小 pilot 检查新信号是否真的优于旧 entropy 线。

关键问题：

- instability 是否优于 raw entropy
- task gating 是否保住 reasoning、阻断 code 坍塌

### Phase 4: Full Reasoning Evaluation

目标：只在 TIGER 通过 screening 后，进入 full reasoning benchmarks。

必须完成：

- GSM8K full/near-full
- MATH500 full
- 与 `Prophet + CORE + DNB + raw entropy` 的 matched-compute 对比

### Phase 5: Boundary Validation

目标：用 HumanEval 小规模验证 scope 控制是否成立。

只允许比较：

- `Standard-64`
- `Prophet`
- `Raw-Entropy Revision`
- `TIGER (gated)`
- `TIGER + syntax guard`（仅当 smoke test 通过）

## 9. 主要指标

### 9.1 质量指标

- `GSM8K`: exact match
- `MATH500`: exact match
- `HumanEval`: pass@1

### 9.2 诊断指标

- `ECE`
- entropy / instability 与错误或 revision benefit 的相关性
- syntax / runtime failure rate（HumanEval）

### 9.3 效率指标

- `actual_nfe`
- `latency_sec`
- `tokens_per_sec`
- `peak_vram_mb`
- `safe_batch_size`

## 10. 风险与决策规则

### 10.1 继续推进 TIGER 的条件

满足以下三条中的两条，才继续 method-forward 路线：

1. `instability-guided revision` 在 GSM8K pilot 上优于 `raw-entropy revision`
2. TIGER 在 GSM8K full 上击败 `Prophet` 或 `DNB` 至少一个
3. MATH500 上出现可复现的非零收益

### 10.2 触发 pivot 的条件

出现以下任一情况，转向 `cand_diag`：

- instability 对 raw entropy 无优势
- TIGER 同时输给 `Prophet` 和 `DNB`
- gating 之后 code 仍显著退化

## 11. 预期可视化

## Expected Visualizations

- Throughput table: backend × compile × batch size × tokens/sec
- Figure: batch-size / throughput scaling curve on RTX PRO 6000
- Table: reasoning main results (`Standard`, `DNB`, `Prophet`, `CORE`, `Raw`, `TIGER`)
- Figure: compute-normalized Pareto plot (accuracy vs actual NFE / latency)
- Figure: instability score vs raw entropy ranking overlap / disagreement
- Heatmap: token-level brittleness across denoising steps
- Table: HumanEval boundary results with syntax/runtime failure rate
- Figure: calibration diagnostic plots on held-out split

## 12. 软件与版本要求

- `torch >= 2.2`
- `transformers >= 4.46`
- `datasets >= 2.18`
- `accelerate >= 0.30`
- `scikit-learn >= 1.4`
- `flash-attn` optional but preferred
- `lm-eval` optional for benchmark standardization / adapter integration

## 13. 备注

- 本轮默认单 seed 主跑，不做统计显著性检验
- 若 GSM8K full 上 `TIGER` 相对 `Prophet/DNB` 的优势超过 1.5%，可追加少量 confirmatory reruns 作为附加稳健性检查，但不把它写成硬阻塞
- `Dream-7B` 暂不进入主计划，除非 LLaDA-8B 上出现清晰正信号
