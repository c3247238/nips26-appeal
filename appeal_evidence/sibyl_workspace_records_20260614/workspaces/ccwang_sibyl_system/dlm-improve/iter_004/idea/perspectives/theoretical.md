# Theoretical Perspective

## 核心判断

这轮 `idea_debate` 不应该再延续旧的 `MGCD front-runner` 叙事。  
从理论角度看，当前最可靠的更新不是“再发明一个更复杂的 controller”，而是把问题重写为：

1. **何时应该停止继续 denoise**，以在几乎不掉点的前提下换取真实速度收益。
2. **哪些 token/span 值得被修**，以在 fixed compute 下把 revision 预算集中到真正高风险的位置。
3. **局部不确定区域之间是否需要轻量 message passing**，而不是一上来就引入全局 memory graph。

换句话说，pilot 证据支持“**uncertainty 是好 observer**”，但不支持“**复杂 controller 已成立**”。  
因此理论主线应从 `state construction` 转向 **risk-bounded stopping + set-valued repair + local structured inference**。

## 证据约束

以下判断必须以本轮真实 pilot 为起点，而不是旧 proposal 的先验：

- `current/exp/results/pilot_summary.json` 显示 `cand_mgcd` 与 `cand_dsg` 都已实际运行，而且都 **没有** 穿透 `RAND-84` 的 sham-control gate。
- `current/supervisor/idea_validation_decision.json` 已明确给出 `PIVOT`，并要求把 `cand_mgcd` 从 front-runner 降级为已筛掉候选。
- 当前真正保留下来的正信号是：
  - `H1` miscalibration pass: `SC-ECE=0.22`
  - `H2` entropy signal pass: `Pearson r=0.44`
  - `H3` revision helps pass: `+5.3%`
  - `H4` calibration >= raw 但结论平凡: raw entropy 已够用
  - `H5` `CARD-84 > DNB-84`
  - `H6` annealing falsified: `T=1.0` 优于退火
  - 标准去噪非单调: `128 steps < 64 steps`

这组证据组合在一起，理论上更像是在说：

- **风险信号存在**
- **继续 denoise 的边际收益并不单调**
- **错误修复对象更可能是稀疏/成段/局部的**
- **复杂 controller 很容易把 observer signal 消耗成无效动作**

## 理论上应放弃什么

### 1. 放弃把 `MGCD-lite` 继续当作 training-free 主线

`MetaState` 这类工作说明“跨步 memory”在 dLLM 中可能有价值，但它们依赖训练或微调来学出稳定记忆写入与读取机制，而我们的约束是 **training-free**。  
如果 `MGCD-lite` 在 training-free 条件下已经未能超过 `RAND-84`，那么理论上更自然的解释不是“memory idea 本身一定错”，而是：

- **memory 可能需要学习才能稳定**
- 或者 **当前 repair object 根本不需要那样的 memory**

所以 memory-enhanced controller 不该继续占据 serious pool 的默认第一位。

### 2. 放弃“再做更花的 calibration”作为主要研究对象

当前 pilot 已经说明 `raw entropy` 足够强，而 calibration 没带来额外收益。  
这与 `Rethinking Confidence Calibration for Failure Prediction` 的结论是同向的：**更好的 calibrated score 不自动带来更好的 downstream intervention**。

因此理论问题不应再表述为“如何进一步校准 score”，而应表述为：

- 如何把已有 score 变成 **risk-controlled stopping rule**
- 如何把已有 score 变成 **coverage-controlled repair set**

## 新的 serious candidate pool

| Candidate | 类型 | 目标 | 理论对象 | 预计工程成本 | 成功概率 |
| --- | --- | --- | --- | --- | --- |
| `cand_srts` | improve existing | 在保持性能时显著提速 | sequential stopping / conformal risk budget | 低 | 0.55 |
| `cand_cbr` | new method | 在 fixed compute 下提升修复质量 | conformal set selection over uncertain spans | 中 | 0.40 |
| `cand_bpir` | cross-domain transfer | 用局部结构推断减少 harm、提升修复效率 | belief propagation / syndrome-style local inference | 中高 | 0.25 |

我建议本轮理论视角把 serious pool 固定为这 3 个对象，不再保留 `MGCD-lite` / `DSG` 作为 front-runner 残影。

## Candidate 1: `cand_srts`

### 名称

`SRTS: Sequential Risk-Test Stopping`

### 核心想法

把 DLM 的迭代去噪过程看成一个 **sequential decision problem**：  
每多做一步 denoise，都会付出额外 NFE/latency，但收益 `Δ_t` 不是单调的。  
既然 pilot 已经显示 entropy 对错误有信息量，且更长 denoising 轨迹并不必然更好，那么最自然的理论转向就是：

> 不再问“如何让每一步更激进地改”，而是问“何时继续 denoise 已经不值得”。

### 数学框架

定义：

- `x_t`: 第 `t` 步的部分去噪状态
- `φ_t`: 由 span entropy、margin、稳定 token 比例、最近两步变化率组成的统计量
- `R_t = E[L(x_T) - L(x_t) | φ_t]`: 从 `t` 到终点的剩余风险改进

训练-free 做法不是学习一个新 network，而是在小型校准集上构造一个上置信界 `U_t(φ_t)`，然后在最早满足

`U_t(φ_t) <= λ * cost(t+1:T)`

时停止。

也可以用 conformal 版本写成：

`P(L(x_tau) - L(x_T) > ε) <= α`

其中 `τ` 是 stopping time，`α` 是允许的掉点风险预算。

### 潜在理论保证

- 在 exchangeability 近似成立时，可给出 **distribution-free finite-sample risk control**：
  早停后相对 full denoise 的性能损失超过 `ε` 的概率不超过 `α`。
- 若 `R_t` 随去噪步数近似形成 supermartingale，则可得到一个 **optional stopping** 风格的 compute-quality bound。
- 若 entropy decay 在 instruction-tuned dLLM 上更快，则能解释为什么 `SchED` 一类方法能在几乎不掉点的条件下获得 3-4x speedup。

### 为什么它和当前证据最一致

- `H2 PASS`: entropy 已被证明是强 observer
- `H6 FALSIFIED`: annealing 无益，说明“继续花 compute”本身并不可靠
- `128 < 64`: 说明 stopping 问题是真问题，不是细枝末节
- `MGCD/DSG` 失败：继续在 action 复杂度上加码没有证据支撑

### 预期收益

- 主目标：**同等性能下提速**
- 次目标：减少无效后期 denoise，释放 compute 给更有价值的局部 revision

### 计算成本

- 额外开销接近 `O(1)` per step，只需收集少量统计量并查表/阈值判断
- 几乎不增加显存
- 和当前最大 batch size 约束完全兼容

### 失败模式

- entropy 只对“错没错”敏感，不对“再做一步值不值”敏感
- calibration slice 太小，导致 risk budget 漂移
- 长文本任务存在局部高风险岛，但全局 stopping rule 过早停止

### 小模型验证建议

- 先在最小 dLLM 或可替代的 masked-LM proxy 上验证 `U_t` 的校准质量
- 若当前没有小型 dLLM，可用 `BERT-base` 做 span uncertainty proxy、`Qwen-0.5B` 做 verifier-style loss proxy，只验证 stopping statistic 本身是否稳定
- 通过后再移植到现有 LLaDA runtime

## Candidate 2: `cand_cbr`

### 名称

`CBR: Conformal Boundary Revision`

### 核心想法

这一路径是对 `BSR` 的**重立题版本**，不是 `MGCD` 的附属物。  
核心命题是：

> 真正需要控制的不是 token-by-token controller，而是一个 **set-valued repair set**。

也就是说，我们不是给每个 token 一个“改/不改”的激进 controller，而是先预测：

- 哪些位置构成一个高风险 island
- 这些 island 的边界是否稳定
- revision budget 应该投给哪些 island，而不是撒到全序列

### 数学框架

令 `I_t = {span_k}` 为由 token entropy、boundary disagreement、局部 drift 聚合得到的候选 island 集合。  
对每个 island 定义一个 score：

`s(span_k) = max_entropy + β * boundary_instability + γ * internal_disagreement`

然后用 conformal calibration 选择阈值 `q_(1-α)`，只 remask 满足 `s(span_k) > q_(1-α)` 的 island，边界 token 锁定。

### 潜在理论保证

- 在交换性假设下，可以给出 **repair-set coverage guarantee**：
  真正错误 token 落在被选中 revision set 外的概率不超过 `α`。
- 若错误过程近似 block-sparse，则 island-level selection 相比 token-level gating 可证明有更低的 expected collateral damage。
- 若边界稳定而内部不稳定，则 boundary-locked revision 比全 span 重写更接近最小扰动修复。

### 为什么它和当前证据最一致

- `H3 PASS`: revision 本身是有用的
- `H4 PASS but trivial`: raw entropy 已足够做 observer，不需要更复杂 score
- `CARD > DNB`: 说明更合理的 repair allocation 比朴素 baseline 更重要
- supervisor 已明确提示：如果保留 `cand_bsr`，应把它作为 **新的 object-level candidate** 重新立题

### 预期收益

- 主目标：**在相同 NFE 下提高 repair/harm ratio**
- 次目标：如果 island 很稀疏，可把未被选中的区域直接冻结，从而间接省 compute

### 计算成本

- 需要一次小型校准，之后推理阶段只做 island 构造和阈值判断
- 比 `cand_srts` 略高，但远低于 memory-graph 或多轨 search

### 失败模式

- 错误并不呈 block-sparse，而是高度离散
- island 构造不稳定，导致 boundary locking 误伤真正需修位置
- conformal 阈值在不同任务间转移性弱

### 小模型验证建议

- 用 `BERT-base` 或其他 bidirectional masked LM 先验证 island score 是否比 token entropy 更好地覆盖真实错误位置
- 若 coverage gain 不成立，则应直接淘汰，不必进入大模型 DLM pilot

## Candidate 3: `cand_bpir`

### 名称

`BPIR: Belief-Propagation Island Repair`

### 核心想法

这是 cross-domain transfer 路线。  
来自 error-correcting codes 与图模型的直觉是：如果错误主要集中在少量 uncertain islands，那么更自然的对象不是“对每个 token 各自下命令”，而是让相邻约束之间做几轮 **局部 message passing**。

它和 `CBR` 的区别是：

- `CBR` 先做 set selection，再做 revision
- `BPIR` 进一步把 island 与左右上下文之间的相容性显式写成局部 factor graph

### 数学框架

把每个 uncertain island 视为一个局部变量块 `z_k`，左右稳定上下文提供兼容性势函数：

`ψ_k(z_k; left_context, right_context)`

相邻 island 之间若存在语义/语法耦合，则再加 pairwise term `ψ_(k,l)`。  
在这个小图上做 2-3 轮 loopy BP 或近似最小和更新，只在 island 内重算候选。

### 潜在理论保证

- 若局部图近似 tree-structured，则 BP 给出精确边缘分布
- 若相互作用满足 Dobrushin-style contraction，可得到收敛保证
- 复杂度从“全序列多轮 controller”转为“只在少数 islands 上做局部推断”

### 为什么它值得保留但不应排第一

- `U-Nets as Belief Propagation` 提供了“denoising 近似消息传递”的理论类比
- diffusion-based ECC 工作说明 iterative denoising 与 syndrome-style correction 确实能对接
- 但它比 `cand_srts` / `cand_cbr` 更依赖错误结构假设，因此不应先吃大预算

### 预期收益

- 可能提高局部修复质量，并减少 boundary spillover
- 若 message passing 只在少数 islands 上运行，也可能保留部分速度优势

### 计算成本

- 比 `cand_cbr` 更高
- 仍低于全局 multi-trajectory search，但实现复杂度明显更大

### 失败模式

- 文本错误结构不像 code syndrome 那样局部可分解
- factor graph 近似过粗，消息传递变成噪声放大器
- island 数量一多，局部推断不再局部

### 小模型验证建议

- 先在合成 span corruption 任务上验证局部图模型是否真能恢复边界一致性
- 若在 toy task 上都没有比简单 island thresholding 更强，就不应进入正式 DLM pilot

## 理论排序与执行建议

### 推荐排序

1. `cand_srts`
2. `cand_cbr`
3. `cand_bpir`

### 原因

- `cand_srts` 最直接吃到当前最强正信号：entropy 有效、长轨迹非单调、annealing 失败。
- `cand_cbr` 是对 supervisor 建议“围绕 repair object 重组候选池”的最直接响应。
- `cand_bpir` 有理论美感，但需要更强结构假设，应该是 stretch candidate，不是新 front-runner。

## 我建议写入下一轮 synthesis 的一句话

> Theoretically, the pilot evidence no longer supports a richer training-free controller story; it supports a risk-allocation story. The next candidate pool should therefore center on stopping rules, repair-set coverage, and local structured inference, not on reviving MGCD-style global memory.

## 参考文献锚点

- Bai et al., 2026, *Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models*  
  https://arxiv.org/abs/2602.01842
- Xia et al., 2026, *MetaState: Persistent Working Memory for Discrete Diffusion Language Models*  
  https://arxiv.org/abs/2603.01331
- Rout et al., 2025, *Test-Time Anchoring for Discrete Diffusion Posterior Sampling*  
  https://arxiv.org/abs/2510.02291
- Tejaswi et al., 2026, *EntRGi: Entropy Aware Reward Guidance for Diffusion Language Models*  
  https://arxiv.org/abs/2602.05000
- Chen et al., 2025, *Optimizing Decoding Paths in Masked Diffusion Models by Quantifying Uncertainty*  
  https://arxiv.org/abs/2512.21336
- Mohamed et al., 2025, *Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules*  
  https://arxiv.org/abs/2512.02892
- Mei, 2024, *U-Nets as Belief Propagation*  
  https://arxiv.org/abs/2404.18444
- Choukroun and Wolf, 2022, *Denoising Diffusion Error Correction Codes*  
  https://arxiv.org/abs/2209.13533
- Lei et al., 2025, *Syndrome-Flow Consistency Model Achieves One-step Denoising Error Correction Codes*  
  https://arxiv.org/abs/2512.01389
- Ravfogel et al., 2023, *Conformal Nucleus Sampling*  
  https://arxiv.org/abs/2305.02633
- Deutschmann et al., 2023, *Conformal Autoregressive Generation: Beam Search with Coverage Guarantees*  
  https://arxiv.org/abs/2309.03797
- Zhu et al., 2023, *Rethinking Confidence Calibration for Failure Prediction*  
  https://arxiv.org/abs/2303.02970

## Bottom line

理论上，本轮最值得下注的不是“更复杂的 training-free controller”，而是：

- **用 uncertainty 决定何时停**
- **用 coverage 决定哪里修**
- **只在必要时对局部结构做 message passing**

这三者都比继续维护旧的 `MGCD front-runner` 叙事更符合当前真实 pilot 证据，也更容易形成下一轮可证伪的 candidate program。
