# Interdisciplinary 视角：把 revision 从“高熵即重写”改成“量测-诊断-局部纠错”

我先明确站位：这轮不能再沿着 `MGCD front-runner` 叙事往下写。`current/exp/results/pilot_summary.json` 已经给出负证据：`MGCD-lite` 在 `GSM8K audited slice` 上相对 `RAND-84` 的 `net_repaired=-4`、在 `MBPP` 上 `-2`；`DSG` 在 `GSM8K` 也只到 `0`、在 `MBPP` 是 `-1`。`current/supervisor/idea_validation_decision.json` 也已经把结论写死了：继续按旧 front-runner framing 追加 GPU 预算，只会放大错误对象。

从跨学科角度看，这些证据说明的不是“还不够聪明的 entropy controller”，而是一个更基础的问题：

- `entropy / drift` 更像 **风险量测信号**，不是可直接执行的 **干预指令**。
- 当前失败很像脑科学里的 `prediction error != action policy`，也很像纠错码里的 `soft information != correction itself`。
- `annealing` 有害、`128 steps < 64 steps` 的非单调现象，则更像统计物理里的 **metastability / hysteresis**：系统不是缺全局热度，而是被困在少数局部坏盆地里。

所以我建议这轮保留一个新的 2-3 候选池，但候选的核心不再是“更复杂的 memory graph”，而是 **把 sensing、diagnosis、local repair 和 compute allocation 分开**。

## 文献搜索记录

说明：本环境没有可用的 `bioRxiv` / `Google Scholar` MCP，因此按 role prompt 的 fallback 走了 `arXiv + Web`。

本次至少完成了 3 组定向 arXiv 搜索，并补了 Web：

- `predictive coding / active inference / precision weighting`
- `belief propagation / turbo decoding / error correction / factor graph`
- `metastability / nucleation / hysteresis / cluster move`

这轮最有用的文献锚点不是“再来一个 DLM 方法名”，而是下面三条跨学科结构：

- **认知神经科学 / predictive coding**
  - `Predictive Coding: a Theoretical and Experimental Review` (Millidge et al., 2021, arXiv:2107.12979)
  - `CogDPM: Diffusion Probabilistic Models via Cognitive Predictive Coding` (Chen et al., 2024, arXiv:2405.02384)
  - `Active inference on discrete state-spaces: a synthesis` (Da Costa et al., 2020, arXiv:2001.07203)
- **信息论 / error correction / belief propagation**
  - `Good Error-Correcting Codes based on Very Sparse Matrices` (MacKay, 1999, IEEE TIT)
  - `Turbo Decoding as an Instance of Pearl's Belief Propagation Algorithm` (McEliece et al., 1998, Caltech technical report / Web)
  - `Quantum Error Correction` (Brun, 2019, arXiv:1910.03672)
- **统计物理 / metastability / nucleation**
  - `Gas-liquid Nucleation at Large Metastability` (Santra et al., 2010, arXiv:1012.4746)
  - `Nucleation of magnetisation reversal, from nanoparticles to bulk materials` (Vogel et al., 2006, arXiv:cond-mat/0610840)

再结合当前 workspace 里的 dLLM 近邻工作：

- `Prism` (Bai et al., 2026, arXiv:2602.01842) 说明 dLLM 端确实值得做 inference-time compute reallocation。
- `MetaState` (Xia et al., 2026, arXiv:2603.01331) 说明跨步记忆是有效方向，但它依赖训练，不满足我们当前 training-free 主约束。
- `Sampler-Centric Evaluation` (Tang et al., 2026, arXiv:2602.19619) 提醒我们不要把 sampler-induced gain 和真实 object-level gain 混在一起。

## 我的核心判断

这轮最值得保留的跨学科对象，不是“更强 state”，而是 **更像纠错系统的 revision contract**：

1. 先测量哪里真的像 error syndrome，而不是只看高熵。
2. 再在局部 span / island 内做有边界的修复，而不是全局加热。
3. 最后把 compute 当成有限生理资源，优先供给 persistent contested regions，而不是对所有 token 一视同仁。

换句话说，下一轮 serious candidates 应该围绕：

- `measurement -> diagnosis -> repair`
- `locality -> boundary protection -> reversibility`
- `fixed budget -> adaptive allocation`

来设计，而不是继续做 `entropy score++`。

## 候选 1：`cand_sgr`，Syndrome-Gated Repair

这是我最支持的新主线，优先级最高。

### 借用的外部结构

- 来自 **predictive coding / active inference**：prediction error 只有在被 precision-weighting 之后，才应该驱动更新。
- 来自 **纠错码 / belief propagation**：soft confidence 不是 correction，本体是 `syndrome + local message passing`。

### 结构对应

- token entropy / draft disagreement = noisy error signal
- span-level consistency checks = syndrome measurement
- locked boundary = codeword 中暂时可信的比特
- island 内多次局部更新 = belief propagation / iterative decoding

### 为什么它比 MGCD / DSG 更对路

`MGCD` 和 `DSG` 的共同问题，是把“high-entropy / high-drift”过快地升级成 action trigger。pilot 已经说明，这种从 observation 直接跳到 intervention 的链条会被 `RAND-84` 打平。

`SGR` 的关键改动，是在 action 之前加一个 **diagnosis layer**：

- 只有当一个 span 同时满足 `high uncertainty` 和 `high structural inconsistency` 时才修。
- 如果只是“难”但没有 syndrome，就不动它。
- 如果 syndrome 消失，即使 entropy 还高，也停止继续修。

这比单纯 memory 化更接近一个可证伪的 object-level hypothesis。

### 最小 training-free 实现

- 从当前 DLM 连续两步或两个 cheap drafts 中提取 `entropy / logit variance / token disagreement`
- 在 span 级别构造 `syndrome score`
  - 邻域一致性
  - 数学题里的式子/数值约束
  - 代码里的括号、缩进、语法闭合
  - 可选地加一个极轻量 verifier，例如规则检查或 `Qwen-0.5B` 打分器，只做 diagnosis 不做生成
- 只对 `high-uncertainty + high-syndrome` 的 island 重掩码
- island 外围 `1-2 token halo` 锁边界
- 在 island 内跑 `2-3` 个局部修复子步
- 若 local consistency 没提升，则回滚到前一步

### 可检验预测

- 相比 `RAND-84`，`cand_sgr` 应在 `GSM8K audited slice` 上第一次产生真正的 sham-control separation。
- 相比 `MGCD-lite / DSG`，它的 improvement 如果成立，应该主要来自 `harmed stable tokens` 减少，而不是来自更大范围重写。
- 相比纯 entropy gate，加入 syndrome 后，收益应主要出现在 `MBPP` 的语法闭合错误和 `GSM8K` 的局部算式错误上。

### 计算成本与成功率

- 成本：比 `CARD-84` 高约 `10%-20%`，但仍应明显低于“多 draft + 大范围回写”的 memory-heavy 方案
- 目标 NFE：`68-76`
- 成功概率：
  - 若目标是“超过 `RAND-84`”：`0.40-0.50`
  - 若目标是“大幅超过 `CARD-84`”：`0.20-0.30`

## 候选 2：`cand_cnr`，Cluster-Nucleation Repair

这是我认为最有辨识度的第二候选，主打把失败解释成 metastable bad islands，而不是 state 缺失。

### 借用的外部结构

- 来自 **统计物理 / nucleation**：系统在高 metastability 区域里，不是每次加一点温度都能出去，往往需要 cluster-level barrier crossing。
- 来自 **hysteresis**：一旦外围区域已经稳定，反复全局改写只会制造额外副损伤。

### 结构对应

- contiguous wrong span = metastable droplet
- boundary confidence = surface tension
- repeated failed token-wise updates = barrier-trapped dynamics
- whole-island remask = collective nucleation move

### 为什么它贴合当前 pilot 证据

- `annealing` 有害，说明“给系统更多全局热量”不是答案。
- `128 steps < 64 steps`，说明更多 global denoising 甚至会把系统重新带进坏盆地。
- `CARD > DNB` 但 `MGCD / DSG` 又过不了 `RAND`，说明 revision 本身并非无效，但 **token-wise / signal-only 的修法** 可能错了。

`CNR` 直接押注：真正要改的是 **collective object**，而不是 token controller。

### 最小 training-free 实现

- 用连续 `2-3` 步 persistent disagreement 标出 contested islands
- 对 island 外围做强锁定，不允许外扩污染
- 对 island 内部做一次 cluster remask，而不是 token-by-token remask
- 用局部 free-energy proxy 作为 accept rule
  - 例如 island 内 NLL proxy 下降
  - consistency penalty 下降
  - boundary mismatch 不升高
- 若不满足 accept rule，直接撤销 cluster move

### 可检验预测

- 若 `CNR` 成立，它在 `GSM8K` 上应优先修复连续 reasoning fragments，而不是零散 token。
- 它应显著降低 island 外 harm rate。
- 它未必在总体 accuracy 上大胜，但若 object hypothesis 对，它应优于任何“更聪明的单 token gate”。

### 计算成本与成功率

- 成本：有机会和 `CARD-84` 持平，甚至略低，因为它减少了无效全局步
- 目标 NFE：`52-68`
- 成功概率：
  - 若目标是“以相近 NFE 达到 `CARD-84` 水平并减少 harm”：`0.35-0.45`
  - 若目标是“直接在 audited slice 上显著超越所有基线”：`0.20-0.25`

## 候选 3：`cand_hbr`，Homeostatic Budget Reallocation

这是速度线最值得保留的 fallback。它的目标不是更高 accuracy，而是 **维持当前性能同时降 NFE**。

### 借用的外部结构

- 来自 **生理学 / homeostasis**：资源不应平均投喂给所有部位，而应优先供给异常持续的区域。
- 来自 **active inference**：系统应在不确定性已经足够低时停止采样，把预算转投到还能带来 expected information gain 的区域。

### 结构对应

- fixed global step schedule = 不区分需求的固定代谢供给
- stable spans = 已回到稳态的组织
- persistent contested spans = 继续耗能才有价值的部位
- early freeze + budget reallocation = homeostatic control

### 最小 training-free 实现

- 维护每个 span 的 `stability age`
- 连续 `k` 步稳定的 span 进入 freeze 状态
- 把省下来的 revision budget 只用于 contested spans
- 当 unresolved mass 低于阈值时提前停止整个 sample

### 可检验预测

- 这条线最容易在 speed objective 上成功。
- 它不一定提升 accuracy，但应能在不明显伤害 `CARD-84` 的前提下，把有效 NFE 降低 `20%-35%`。
- 若这条线成功、而 `SGR / CNR` 不成功，那么项目也仍然有一条 reviewer-friendly 结果线：`performance-neutral acceleration under training-free adaptive allocation`。

### 计算成本与成功率

- 成本：最低
- 目标 NFE：`40-60`
- 成功概率：
  - 若目标是“维持性能并提速”：`0.50-0.60`
  - 若目标是“顺便提 accuracy”：`<0.20`

## 我建议直接淘汰或降级的东西

- 不要再把 `cand_mgcd` 当成 default winner；它已经被真实 pilot 证据否掉了。
- 不要再为 `entropy recalibration` 本身投入新预算；`current/context/idea_context.md` 已经写明 calibration gain 很可能是 trivial 的，observer signal informative 不等于 controller valid。
- 不要再做全局 annealing 叙事；`H6` 已经被否证。
- 若保留旧 `BSR`，也只能把它重写成 `CNR` 的简化 control，而不是 `MGCD-lite companion`。

## 我建议给 synthesizer 的候选池排序

1. `cand_sgr`
   - 最像一个真正回答了 pilot 失败机制的新对象
   - 兼顾 accuracy 线和 reviewer 可解释性
2. `cand_hbr`
   - 最适合 speed 线，成功率最高
   - 即使 accuracy 不涨，也可能成为可靠 fallback
3. `cand_cnr`
   - 学术辨识度最高
   - 但实现和 accept rule 设计更敏感，适合作为第三候选而不是唯一主线

如果下一轮只保留两个 serious candidates，我的建议是：

- accuracy-first：`cand_sgr`
- speed-first：`cand_hbr`

`cand_cnr` 可以作为 object-level 解释更强、但工程风险更高的第三候选。

## 最小 screening pilot 建议

为了让这轮 idea debate 真正服务下一轮 pilot，而不是再生一个概念名词，我建议 screening 直接这样设计：

### Pilot A：`SGR-72` vs `RAND-72` vs `CARD-72`

- 数据：沿用当前 `GSM8K audited slice`，再过 `MBPP audited slice`
- 重点日志：
  - syndrome-trigger rate
  - local rollback rate
  - harmed stable tokens
  - island mean size
  - accepted local repair ratio

### Pilot B：`HBR-56` vs `CARD-84`

- 目标不是赢 accuracy，而是证明 `performance-neutral acceleration`
- 重点日志：
  - effective NFE
  - wall-clock latency
  - freeze ratio
  - unresolved mass trajectory

### Pilot C：`CNR-64` vs `CARD-64`

- 只在 `Pilot A` 没有 clear winner 时再做
- 否则先别把第三个高风险对象拉进来烧 GPU

## 最终态度

支持继续 `idea_debate` 后的新跨学科 candidate pool，但不支持任何形式的 `MGCD comeback`。  
我给这轮最值得推进的方向打分如下：

- `cand_sgr`: `8.8/10`
- `cand_hbr`: `8.1/10`
- `cand_cnr`: `7.6/10`

一句话总结：**下一轮不该再问“怎样把 entropy controller 变得更复杂”，而该问“怎样把 DLM revision 变成一个更像诊断驱动的局部纠错系统，或者更像 homeostatic budget allocation 的自适应采样系统”。**

## 参考锚点

- Millidge, Seth, Buckley. `Predictive Coding: a Theoretical and Experimental Review`. arXiv:2107.12979. https://arxiv.org/abs/2107.12979
- Chen et al. `CogDPM: Diffusion Probabilistic Models via Cognitive Predictive Coding`. arXiv:2405.02384. https://arxiv.org/abs/2405.02384
- Da Costa et al. `Active inference on discrete state-spaces: a synthesis`. arXiv:2001.07203. https://arxiv.org/abs/2001.07203
- McEliece et al. `Turbo Decoding as an Instance of Pearl's “Belief Propagation” Algorithm`. Caltech technical report. https://authors.library.caltech.edu/records/478gh-bc410/files/MACieeetit98.pdf
- Brun. `Quantum Error Correction`. arXiv:1910.03672. https://arxiv.org/abs/1910.03672
- Santra, Singh, Bagchi. `Gas-liquid Nucleation at Large Metastability`. arXiv:1012.4746. https://arxiv.org/abs/1012.4746
- Vogel, Moritz, Fruchart. `Nucleation of magnetisation reversal, from nanoparticles to bulk materials`. arXiv:cond-mat/0610840. https://arxiv.org/abs/cond-mat/0610840
- Bai et al. `Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models`. arXiv:2602.01842. https://arxiv.org/abs/2602.01842
- Tang et al. `Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models`. arXiv:2602.19619. https://arxiv.org/abs/2602.19619
- Xia et al. `MetaState: Persistent Working Memory for Discrete Diffusion Language Models`. arXiv:2603.01331. https://arxiv.org/abs/2603.01331
