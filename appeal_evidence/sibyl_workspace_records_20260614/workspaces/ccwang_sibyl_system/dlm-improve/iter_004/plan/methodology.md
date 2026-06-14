# Iteration 4 Methodology

## Objective

本轮 planning 服务于一次已经发生 `PIVOT` 的 refine round。  
真实目标不是继续验证 `MGCD/DSG`，而是把下一轮 screening pilot 收紧成两个可直接 falsify 的主问题：

1. **repair object 是否选错了？**
   - 用 `cand_bsr_v1` 检验 uncertainty island / span / boundary 是否比 token-wise revision 更对题。
2. **entropy 是否更适合做 routing / stopping？**
   - 用 `cand_espd` 检验 compute reallocation 是否比 semantic controller 更符合现有证据。

`cand_ugr` 保留，但本轮只作为 **conditional backup**，不进入 phase-1 调度。

## Evidence Boundary

本轮 planning 必须显式继承以下负证据：

- `cand_mgcd` 与 `cand_dsg` 已在 iteration 4 screening 中 failed to separate from `RAND-84`
- 当前 proposal 只能宣称 **new candidate program**, 不能宣称机制已经成立
- 所有 full benchmark 都必须等新的 screening signal 出现后再实例化

## Candidate Ladder

### `cand_bsr`

- 角色：唯一 quality front-runner
- 冻结版本：`cand_bsr_v1`
- 核心机制：
  - uncertainty island diagnosis
  - span aggregation
  - boundary-stable repair
  - local revision acceptance
- 研究命题：
  - 当前 training-free DLM 的主要错误在于 repair object，而不是单纯缺一个更复杂 controller

### `cand_espd`

- 角色：并行 speed line
- 核心机制：
  - entropy-routed active frontier
  - risk-bounded stopping
  - compute budget reallocation
- 研究命题：
  - `raw entropy` 的高价值用途是 routing / stopping，而不是直接 semantic revision

### `cand_ugr`

- 角色：conditional backup
- 核心机制：
  - 在已识别 island 上做 cheap uplift / benefit estimation
- 激活条件：
  - `cand_bsr` 或 `cand_espd` 至少一条在 phase-1 screening 中出现真实 signal

## Benchmarks And Split

### Phase 1 Scheduled Benchmark

- `GSM8K audited slice`
- 样本预算：`n=100`
- 默认 split：
  - `50 design`
  - `50 confirm`

### Phase 2 Conditional Benchmark

- `MBPP audited slice`
- 只在 `cand_bsr_v1` 通过 `GSM8K` gate 后执行
- 重点不是 headline accuracy，而是：
  - `syntax-valid`
  - `exec-valid`
  - `repair/harm` 分解

## Controls

### Shared Controls

- `CARD-84`
- `RAND-84`

### Mechanism Controls For `cand_bsr_v1`

- `RandSpan-84`
- `EntropySpan-NoBoundary`
- `BoundaryLock-RandomSpan`

### Mechanism Control For `cand_espd`

- `ESPD-FixedFrontier`

### Conditional Code Control

- `SyntaxGuard-only`

这些 control 的作用不是凑 baseline 数量，而是把下面几个机制拆开：

- detection 是否有效
- span aggregation 是否有效
- boundary protection 是否有效
- frontier routing 是否真的依赖 entropy
- 代码任务上的 gain 是否只是 generic syntax constraint 造成

## Runtime Contract

### Batch / Attention / Compile

- 所有 GPU 任务先做 `max-safe batch size` probe，再跑正式 pilot
- 优先 `flash_attention_2`
  - 首选 PyTorch SDPA / flash backend
  - 若不可用，回落到 matched fallback，并明确记录原因
- `torch.compile` 要求 parity：
  - 如果 baseline 用 compile
  - candidate 与 sham control 必须同样用 compile
- 记录：
  - backend
  - compile_enabled
  - safe_batch_size
  - tokens_per_sec
  - wall_clock_sec
  - peak_vram_mb

### Auxiliary Compute Ledger

以下任何辅助逻辑都必须单独记账：

- extra forward passes
- frontier-selection overhead
- syntax guard overhead
- uplift estimation overhead

### Harness Alignment

根据 `lm-evaluation-harness` 的最佳实践：

- benchmark 名称、prompt framing、metric 命名尽量对齐社区标准
- 若 DLM 自定义 sampler 无法直接接入 `lm-eval`
  - 仍要复用其 task definition / answer normalization 约定
  - 避免自造 GSM8K / MBPP 评分逻辑

## GPU Allocation Rationale

本轮 phase-1 任务全部设计为 `gpu_count=1`、`multi_gpu_strategy=single`，原因不是偷懒，而是：

- `LLaDA-8B` inference 在单张 RTX PRO 6000 上可运行
- 当前更高效的并行方式是 **task-level parallelism**
  - control 任务占一张卡
  - candidate 任务占另一张卡
- 相比把单个 inference task 强行做多卡，跨任务并行更容易保持归因清晰、日志简单、恢复更稳

若进入 full benchmark，再重新评估 2-4 GPU 的多卡 DataParallel / task-sharding 策略。

## Candidate-Specific Experimental Setup

### `cand_bsr_v1`

#### 冻结合同

planning 结束前必须冻结以下 5 项，不允许在 pilot 中途变更：

1. island score
2. span merge rule
3. boundary lock width
4. max revision steps
5. accept / reject rule

#### 主要比较

- `cand_bsr_v1`
- `CARD-84`
- `RAND-84`
- `RandSpan-84`
- `EntropySpan-NoBoundary`
- `BoundaryLock-RandomSpan`

#### 关键诊断

- repair_count
- harm_count
- harmed_stable_tokens
- mean_span_length
- touched_token_ratio

### `cand_espd`

#### 冻结合同

- active frontier scoring rule
- frontier retention ratio
- stopping threshold
- maximum retained steps

#### 主要比较

- `cand_espd`
- `CARD-84`
- `RAND-84`
- `ESPD-FixedFrontier`

#### 关键诊断

- active_frontier_ratio
- stopped_step_histogram
- equal_compute_quality
- equal_quality_speed

### `cand_ugr`

- 本轮不调度
- 只保留 methodology 约束：
  - predicted uplift must be logged
  - observed net benefit must be logged
  - repair/harm ratio must be compared to plain `cand_bsr_v1`

## Metrics

### Unified Table A: Quality @ Equal Compute

- benchmark score
- repair_count
- harm_count
- harmed_stable_tokens

### Unified Table B: Speed @ Equal Quality Band

- wall_clock_sec
- tokens_per_sec
- active_frontier_ratio
- extra_forward_count

### Code Task Metrics

- `syntax-valid`
- `exec-valid`
- `pass@1`

## Promotion And Failure Criteria

### `cand_bsr_v1` Promotion

仅当以下条件同时满足，才允许把 `cand_bsr` 升级成下一轮 mainline:

1. `cand_bsr_v1 > RAND-84`
2. 主要收益来自 harm containment，而非大范围 rewrite
3. mechanism controls 中至少一个被清楚击败

### `cand_espd` Promotion

仅当以下条件同时满足，才允许把 `cand_espd` 升级成正式 speed contribution:

1. 在 unified runtime contract 下存在明确速度收益
2. 不出现对 `RAND-84` 的明显 collapse
3. `ESPD-FixedFrontier` 无法复现同样收益

### No-Go Rule

若 `cand_bsr_v1` 与 `cand_espd` 都无法提供对 `RAND-84` 的可信 signal，则：

- 不进入 full benchmark
- 不启用 `cand_ugr`
- 直接回到 `idea_debate`

## Expected Visualizations

- Table 1: `GSM8K` screening main results (`Method × quality @ equal compute`)
- Table 2: runtime contract summary (`backend / compile / safe_batch / tokens_per_sec / peak_vram`)
- Figure 1: repair vs harm decomposition for `cand_bsr_v1` and its sham controls
- Figure 2: active frontier shrink / stopped-step histogram for `cand_espd`
- Figure 3: `syntax-valid / exec-valid` comparison on `MBPP` conditional phase
