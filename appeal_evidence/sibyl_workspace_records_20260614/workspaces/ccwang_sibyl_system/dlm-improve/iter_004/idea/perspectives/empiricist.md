# Empiricist 视角

## 结论先行

这份意见对应 **iteration 4 的 PIVOT 后 idea_debate**。我不支持继续沿用旧的 `MGCD front-runner` 叙事。根据 `current/exp/results/pilot_summary.json` 与 `current/supervisor/idea_validation_decision.json`，当前最稳妥的 empirical 结论是：

- `cand_mgcd` 与 `cand_dsg` 都已经被本轮 screening gate 筛掉，不能再作为 serious candidate pool 的存活方法。
- 这轮证据仍然只是 `n=50` 的 audited screening slice，足够做 **kill decision**，但不足以写成 benchmark-level population claim。
- 下一轮 serious candidate pool 应收缩为 **2 个主候选 + 1 个必做代码安全 ablation**：
  - `BSR`：Boundary-Stable Revision
  - `EBR`：Entropy-Budget Reallocation
  - `Syntax-Guard`：不是独立主候选，而是 `BSR/EBR` 在 `MBPP` 上的必做 ablation

如果 synthesizer 还保留 `MGCD` 或 `DSG`，我只接受把它们放进 **archived negative controls**，不接受继续把它们写成待晋级主线。

## 真实 pilot 证据底线

### 来自 workspace 的直接证据

- `current/exp/results/pilot_summary.json`
  - `cand_mgcd`：
    - `GSM8K`: `accuracy=0.22`, `vs_rand84_net_repaired=-4`, `vs_card84_net_repaired=-5`
    - `MBPP`: `accuracy=0.00`, `vs_rand84_net_repaired=-2`, `vs_card84_net_repaired=-2`
  - `cand_dsg`：
    - `GSM8K`: `accuracy=0.30`, `vs_rand84_net_repaired=0`, `vs_card84_net_repaired=-1`
    - `MBPP`: `accuracy=0.02`, `vs_rand84_net_repaired=-1`, `vs_card84_net_repaired=-1`
- `current/supervisor/idea_validation_decision.json`
  - 已明确给出 `PIVOT`
  - 已明确要求把 `MGCD` 与 `DSG` 从 front-runner 叙事中降级

### 来自原始结果文件的额外诊断

- `current/exp/results/mgcd_gsm8k_pilot.remote.json`
  - `latency_sec=112.92`
  - `avg_nfe=70.22`
  - `compile_enabled=false`
  - `fixed=5, harmed=9` 相对 `RAND-84`
- `current/exp/results/dsg_gsm8k_pilot.json`
  - `latency_sec=75.16`
  - `avg_nfe=68.0`
  - `compile_enabled=true`
  - `fixed=4, harmed=4` 相对 `RAND-84`
- `current/exp/results/mgcd_mbpp_pilot.remote.json`
  - `latency_sec=197.03`
  - `accuracy=0.00`
  - 可见失败样本以 `SyntaxError`、`IndentationError`、`NameError` 为主
- `current/exp/results/dsg_mbpp_pilot.remote.json`
  - `latency_sec=106.32`
  - `accuracy=0.02`
  - 同样可见大量 `SyntaxError` / `NameError` / `TypeError`

### empiricist 解读

1. `MGCD` 不是“差一点成功”，而是 **更慢且更 harmful**。
2. `DSG` 比 `MGCD` 更轻，但它的最好结果也只是 `vs_rand84_net_repaired=0`，没有形成真实 separation。
3. 当前 wall-clock 还带有明显 runtime confound：
   - 两个方法都跑在 `attention_backend=eager`
   - `MGCD` 与 `DSG` 的 `compile_enabled` 不一致
   - `probed_safe_batch_size=182`，但实际只用 `batch_size=50`
4. `MBPP` 的代表性错误不是“答案接近但错了”，而是 **程序结构被 revision 破坏**。这说明下一轮若还做 code benchmark，必须把 `syntax-validity` 与 `exec-validity` 单独拉出来做一等指标。
5. 这轮 screening `sample_count=50`，低于项目当前对 pilot 的 `n>=100` 要求，因此它应被视为 **负例筛查证据**，而不是最终有效性结论。

## 文献与方法学锚点

### 检索说明

- 已完成 2 次 arXiv 定向检索：
  - discrete diffusion / sampler / test-time scaling / evaluation
  - confidence calibration / failure prediction / entropy-guided decoding
- `Google Scholar` MCP 在当前 runtime 不可用，因此用 **直接 paper source + web lookup** 替代。
- 已补充 web source 用于 benchmark / evaluation tooling / best-practice 锚点。

### 对本轮最有用的外部证据

1. **Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models**  
   arXiv: https://arxiv.org/abs/2602.01842  
   启发：对 dLLM，真正值得保留的是“早中期窗口内做 compute reallocation，并保留高置信 token”，而不是盲目增加全局 state 复杂度。

2. **Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models**  
   arXiv: https://arxiv.org/abs/2602.19619  
   启发：dLLM 的 headline metric 很容易把 denoiser、sampler、runtime 三种误差混在一起。下一轮必须把 `algorithmic NFE`、`wall-clock latency`、`sampler-side harm` 分开报。

3. **Rethinking Confidence Calibration for Failure Prediction**  
   arXiv: https://arxiv.org/abs/2303.02970  
   启发：observer-side 信号有用，不等于它已经是有效 controller。这与本项目已有经验完全一致：`raw entropy` 有预测性，但“把它直接变成 revision controller”并未自动成立。

4. **Training Verifiers to Solve Math Word Problems**  
   arXiv: https://arxiv.org/abs/2110.14168  
   启发：`GSM8K` 适合继续保留为 reasoning-side benchmark，但必须把它当作 **可审计 slice -> full benchmark** 的两阶段流程，而不是拿小样本 pilot 直接宣布方法成立。

5. **Program Synthesis with Large Language Models**  
   arXiv: https://arxiv.org/abs/2108.07732  
   启发：`MBPP` 的本质是程序合成与单元测试通过率，因此仅报 pass@1 不够；要同时记录 `syntax-valid` 与 `runnable`，否则 revision-induced code breakage 会被混在“推理错误”里。

6. **lm-evaluation-harness**
   GitHub: https://github.com/EleutherAI/lm-evaluation-harness
   启发：若 DLM wrapper 能兼容，应尽量复用标准 task/prompt/eval plumbing；若不能兼容，也应显式记录与 harness 默认设置的差异。

## 我支持保留的 serious candidate pool

## 候选 1: `BSR`（Boundary-Stable Revision）

### 核心假说

真正该修复的对象是 **uncertainty island 的 span/boundary**，不是更复杂的 memory graph。若这个假说成立，一个比 `MGCD` 更简单的 boundary-stable repair 应该能在 matched compute 下超过 `CARD-84`，并且在 `MBPP` 上显著减少 syntax breakage。

### 方法草图

- 用 `raw entropy + cross-step instability` 标出 contiguous uncertainty islands
- 锁住 island 外侧低熵 token
- 只对 island 内部做局部 remask / re-denoise
- 不引入 dual-draft consensus，不引入 tri-state memory graph

### 精确评测协议

- 模型：沿用当前 DLM 主线模型，不额外训练
- 数据：
  - `GSM8K audited slice`: `n=100`
  - `MBPP audited slice`: `n=100`
- 基线：
  - `CARD-84`
  - `RAND-84`
  - `draft-only 64-step`
  - `BSR-68` 与 `BSR-84`
- 主指标：
  - `GSM8K`: `accuracy`, `net_repaired vs CARD`, `net_repaired vs RAND`
  - `MBPP`: `pass@1`, `syntax-valid rate`, `exec-valid rate`, `net_repaired`
- 次指标：
  - `avg_nfe`
  - `wall-clock latency`
  - `avg_tokens_changed`
  - `mean island length`
  - `locked_token_ratio`

### 必做 ablations

- 去掉 boundary locking
- 去掉 cross-step instability，只保留 raw entropy
- `BSR-68` vs `BSR-84`
- `BSR` + `Syntax-Guard` vs `BSR` 原版（仅 `MBPP`）

### 明确 falsification 条件

- `BSR-84` 在 `GSM8K` 上仍然不能超过 `RAND-84`
- 或 `BSR-84` 在 `MBPP` 上的 `syntax-valid rate` 不优于 `CARD-84`
- 或 gain 只能靠显著更高 latency / NFE 才出现

### 主要 confounders 与控制

- Confound 1：runtime stack 不一致  
  控制：所有方法都在同一 backend matrix 下运行，首选 `flash_attention_2 + torch.compile`，不可用时至少保证 compile parity。
- Confound 2：batch size 未吃满  
  控制：每个数据集先做一次 max-safe probe，然后用该上限或其稳定近邻，不允许无理由退回 `50`。
- Confound 3：code breakage 被混成 reasoning 错误  
  控制：单独报 `syntax-valid` 与 `exec-valid`。

### `<1 GPU-hour` pilot 设计

- 单卡，`GSM8K n=100 + MBPP n=100`
- 预计：
  - `GSM8K`: 约 `4-6` 分钟
  - `MBPP`: 约 `8-12` 分钟
  - 总计单候选约 `12-18` 分钟
- 若双卡并行 `GSM8K/MBPP`，则更低

### 成功概率

`0.35`

我给它保守但非零的概率，因为它直接对准了“repair object 错了”这个当前最像真的问题，同时比 `MGCD` 更容易归因。

## 候选 2: `EBR`（Entropy-Budget Reallocation）

### 核心假说

`raw entropy` 更适合做 **compute routing / active-set shrinking**，而不是直接做 semantic controller。若这个假说成立，我们应该能在不训练模型的情况下，把 compute 从“全序列均匀 denoise”转移到“高熵 frontier”，从而实现：

- 同等质量下更低 latency
- 或同等 latency 下更高 accuracy

### 方法草图

- 在 early-to-mid denoising window 统计 token entropy
- 对低熵且连续稳定的 token 进行 freeze
- 仅对高熵 frontier 保留后续更新预算
- 保持总 NFE budget 明确可计，不允许隐藏搜索

### 精确评测协议

- 数据：
  - `GSM8K audited slice`: `n=100`
  - `MBPP audited slice`: `n=100`
- 基线：
  - `CARD-64`
  - `CARD-84`
  - `RAND-84`
  - `BSR-68` 或 `BSR-84` 的 matched-wall-clock 版本
- 被测版本：
  - `EBR-48`
  - `EBR-64`
  - `EBR-64 + Syntax-Guard`（仅 `MBPP`）
- 主指标：
  - `accuracy` / `pass@1`
  - `latency_sec`
  - `avg_nfe`
  - `active_token_ratio` 随 step 的变化
- 速度轨 hard target：
  - `>=20%` wall-clock 降幅
  - 同时 `accuracy` 绝对下降不超过 `1` 个百分点

### 必做 ablations

- 只做 entropy freeze，不做 budget reallocation
- 只做 budget reallocation，不做 freeze
- early window 长短（例如 `16/24/32` step）
- active frontier 阈值（如 top `5%/10%/15%`）

### 明确 falsification 条件

- 无法在统一 runtime stack 下拿到 `>=20%` latency 改善
- 或 wall-clock 改善主要来自 backend 差异而非方法本身
- 或 `MBPP` 上 freeze 导致 syntax / runnable 指标进一步恶化

### 主要 confounders 与控制

- Confound 1：把系统加速误写成算法速度提升  
  控制：先做 backend-normalized baseline，再报方法增益。
- Confound 2：active-set shrinking 造成表面 NFE 降低但语义质量塌陷  
  控制：必须同时报 `net_repaired` 与 harmed count。
- Confound 3：frontier 阈值只在某一数据集上偶然奏效  
  控制：`GSM8K` 与 `MBPP` 都跑同一阈值网格。

### `<1 GPU-hour` pilot 设计

- 单卡运行 `EBR-48/64` 于 `GSM8K n=100 + MBPP n=100`
- 预计单候选约 `10-15` 分钟
- 与 `BSR` 并行时，总 screening 仍可控制在 `<1 GPU-hour`

### 成功概率

`0.45`

我给它比 `BSR` 略高的概率，因为它更贴近当前已被 pilot 支持的正信号：`raw entropy` 有信息量、标准 denoising 非单调、速度目标本身就是项目主任务之一。

## `Syntax-Guard` 不是第三主候选，而是必做 ablation

### 为什么必须做

当前 `MBPP` 的可见失败样本大量表现为：

- `SyntaxError`
- `IndentationError`
- `NameError`
- `TypeError`

这说明 revision 很可能在 code task 上破坏了 **局部程序结构**。如果不把这类 failure mode 独立出来，我们会错误地把“结构损坏”解释成“方法不具备 reasoning gain”。

### 实现建议

- 锁定 `def` 签名、括号配对、缩进前缀、引号闭合等 syntax-critical token
- 仅允许在函数体内部的高熵 island 做 revision
- 单独报：
  - `syntax-valid rate`
  - `exec-valid rate`
  - `unit-test pass@1`

### 判定规则

- 如果 `Syntax-Guard` 能显著减少 syntax failure，但总体 pass@1 仍不提升，说明问题更接近 **repair object / control signal 无效**。
- 如果 `Syntax-Guard` 明显修复 `MBPP`，那它应成为 `BSR` 与 `EBR` 的固定组件，而不是另起第三条主线。

## 下一轮共享实验合同

### 我认为必须写死的规则

1. **`MGCD` 与 `DSG` 不再进入 serious pool**
   - 只保留为 archived negative controls
   - 不再作为晋级 full benchmark 的候选

2. **pilot 最小规模回到 `n>=100`**
   - 当前 `n=50` 结果可用于 kill，不可用于 final effectiveness claim

3. **统一 runtime stack**
   - 首选：`flash_attention_2 + torch.compile`
   - 若环境不支持，则所有候选统一退回同一 fallback
   - 禁止出现 `MGCD compile=false`、`DSG compile=true` 这种不对称对比

4. **尽可能使用最大 batch size**
   - 先 probe，再跑正式 pilot
   - 报告 `probed_safe_batch_size` 与 `actual_batch_size`

5. **速度与质量必须双轨报告**
   - 质量：`accuracy / pass@1 / net_repaired / harmed`
   - 速度：`latency_sec / avg_nfe / active_token_ratio`

6. **代码任务必须拆开三层指标**
   - `syntax-valid`
   - `exec-valid`
   - `unit-test pass@1`

### 进入 full benchmark 的最小门槛

- 至少一个新候选在 `GSM8K n=100` 上 **明确超过 `RAND-84`**
- 同一个候选在 `MBPP` 上 **不出现 syntax-valid / exec-valid 崩塌**
- 速度型候选若主张加速，必须在统一 runtime stack 下拿到 **可复现 wall-clock 改善**

## 给 synthesizer 的一句话建议

把这一轮写成：

> “`MGCD/DSG` 已被真实 screening pilot 否证；下一轮 empirical program 不再追逐更重 state，而是围绕 `boundary-stable repair` 与 `entropy-driven compute reallocation` 两条更容易证伪、也更符合现有信号的 training-free 路线重建候选池。”

如果 proposal 仍然把 `MGCD` 保留为 primary idea，我会给出反对票。

## Sources

### Workspace evidence

- `current/exp/results/pilot_summary.json`
- `current/supervisor/idea_validation_decision.json`
- `current/exp/results/mgcd_gsm8k_pilot.remote.json`
- `current/exp/results/dsg_gsm8k_pilot.json`
- `current/exp/results/mgcd_mbpp_pilot.remote.json`
- `current/exp/results/dsg_mbpp_pilot.remote.json`

### External sources

- Prism: https://arxiv.org/abs/2602.01842
- Sampler-centric evaluation: https://arxiv.org/abs/2602.19619
- Rethinking Confidence Calibration for Failure Prediction: https://arxiv.org/abs/2303.02970
- GSM8K paper: https://arxiv.org/abs/2110.14168
- MBPP paper: https://arxiv.org/abs/2108.07732
- lm-evaluation-harness: https://github.com/EleutherAI/lm-evaluation-harness
