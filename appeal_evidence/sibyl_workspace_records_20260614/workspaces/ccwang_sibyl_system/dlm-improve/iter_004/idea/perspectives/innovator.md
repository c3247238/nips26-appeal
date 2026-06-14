# Innovator Perspective

## 先把坏消息写死

这轮 `innovator` 不能再沿用旧的 `MGCD front-runner` 叙事，因为真实 pilot 证据已经把那条线打断了：

- `MGCD-lite` 在 `GSM8K audited slice` 上 `accuracy=0.22`，相对 `RAND-84` 的 `net_repaired=-4`，相对 `CARD-84` 的 `net_repaired=-5`
- `MGCD-lite` 在 `MBPP audited slice` 上 `accuracy=0.00`，相对 `RAND-84` 与 `CARD-84` 都是负收益
- `DSG` 在 `GSM8K audited slice` 上 `accuracy=0.30`，相对 `RAND-84` 的 `net_repaired=0`
- `DSG` 在 `MBPP audited slice` 上 `accuracy=0.02`，相对 `RAND-84` 的 `net_repaired=-1`
- 因此本轮 screening gate 的结论只能是：`H1 / Sham-Control Separation` 失败，`cand_mgcd` 和 `cand_dsg` 都不能继续被当作 serious front-runner

我认为这里最重要的解释不是“observer signal 不存在”，而是：

- `raw entropy` 仍然是有信息量的 observer
- 但“高风险 token/step”不等于“对这些位置做 revision 一定有净收益”
- 我们当前失败的不是 `error detection`，而是 `intervention targeting`

这和已有 workspace 证据是一致的：

- 早期 pilot 已支持 `entropy` 能预测错误，且 `revision helps` 有弱正信号
- 但 `calibration >= raw entropy` 没有成立，说明更复杂的 score 并没有自动变成更好的控制器
- `annealing` 已被否证，`T=1.0` isothermal 更稳
- 标准 denoising 非单调，说明“所有 token 同步做满所有步数”很可能本身就是错的 compute allocation

因此我支持的 pivot 不是“再发明一个更复杂的 memory graph”，而是把下一轮 serious ideas 重写为：

- 更好的 `repair object`
- 更好的 `compute allocation`
- 更好的 `edit acceptance criterion`

## 写作前文献检查

按本轮 shared workflow，我先做了定向文献搜索；这里仅保留会改变 idea 方向的部分。

### arXiv 检索 1：dLLM test-time scaling / guidance / revision

检索式：

- `"discrete diffusion" AND ("test-time scaling" OR guidance OR revision OR search)`
- `("self-correction" OR remasking) AND ("masked diffusion" OR "discrete diffusion")`

对我影响最大的结果：

- [Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models (2026)](https://arxiv.org/abs/2602.01842)
  已经把 `hierarchical search + partial remasking + self-verification` 这条线做得很明确，所以“泛泛而谈的 branching / self-verification”不再新。
- [Remasking Discrete Diffusion Models with Inference-Time Scaling (ReMDM, 2025)](https://arxiv.org/abs/2503.00307)
  说明“remasking + more inference-time compute”已经是一条成熟路线，下一轮不能只把 remasking 包装成新 idea。
- [Reversible Diffusion Decoding for Diffusion Language Models (RDD, 2026)](https://arxiv.org/abs/2602.00150)
  说明“backtracking + confidence-guided remasking”也已经被直接提出；所以 generic reversible/backtracking 不能再当主创新点。
- [Simple Guidance Mechanisms for Discrete Diffusion Models (2024)](https://arxiv.org/abs/2412.10193)
  提醒我 guidance 可以成立，但它主要回答“如何更强地推分布”，不自动回答“如何避免 intervention harm”。
- [Debiasing Guidance for Discrete Diffusion with Sequential Monte Carlo (2025)](https://arxiv.org/abs/2502.06079)
  说明如果我们引入 search / branching / importance reweighting，reviewer 会自然追问 sampling bias 与 hidden compute confound。

### arXiv 检索 2：adaptive compute / locality / cross-domain transfer

检索式：

- `(diffusion OR "discrete diffusion") AND (adaptive OR "early exit" OR multigrid OR "adaptive mesh refinement" OR local)`
- `("adaptive computation" OR "event-triggered" OR "early exiting") AND (language model OR diffusion)`

对我影响最大的结果：

- [Adaptive mesh refinement ... multigroup neutron diffusion equations (2025)](https://arxiv.org/abs/2503.11800)
- [Dynamic Implicit 3D Adaptive Mesh Refinement for Non-Equilibrium Radiation Diffusion (2013)](https://arxiv.org/abs/1302.6303)

它们不是 NLP 论文，但给了一个很强的跨领域启发：

- diffusion 类系统未必应该全域、同精度、同步求解
- 合理的做法是：只在高残差区域细化，只在平稳区域冻结或粗化

这和我们当前 DLM 负结果非常契合，因为 `MGCD/DSG` 的问题都像是在“对全局做了太多并不值得的干预”。

### uncertainty / failure prediction 线

检索式：

- `("failure prediction" OR calibration OR uncertainty) AND (decoding OR guidance OR revision)`

最关键的基础锚点：

- [Rethinking Confidence Calibration for Failure Prediction (2023)](https://arxiv.org/abs/2303.02970)

它强化了一个对本项目非常关键的判断：

- `risk signal != intervention value`

也就是说，某个位置“看起来危险”，并不代表“现在改它会更好”。这几乎就是我们当前 `entropy strong, controller weak` 的理论镜像。

### Web 补充

我还用 Web 查了最新实现/项目页与近月 paper 线索，用来避免重复已有方向：

- ReMDM project page: <https://guanghanwang.com/remdm>
- Prism arXiv / code release line: <https://arxiv.org/abs/2602.01842>
- 最新 training-free/self-correction 线在 2025-2026 很密集，generic remask/backtrack 已经明显拥挤

说明下一轮真正值得押注的，不是“再做一个更会 remask 的 sampler”，而是：

- 让 revision 更局部
- 让 compute 更不均匀
- 让 accept/reject 更接近 intervention gain 而不是 raw uncertainty

注：

- `Google Scholar` MCP 在本 runtime 不可用；我用 arXiv 主源 + Web 精确题名检索代替，足够判断 novelty 风险与近邻工作密度。

## 我的创新立场

我建议把 serious candidate pool 收缩到 3 个，而且每个都要显式回答一个不同问题：

1. `怎样把 revision 变得更局部，而不是更复杂？`
2. `怎样把 compute 从“全局同速”改成“局部细化”？`
3. `怎样把 gating 从“风险检测”升级到“干预收益估计”？`

下面是我支持的三个角度。

## Angle 1: SIR

### 名称

`SIR: Stability-Island Routing`

### 类型

改进现有方向；也是我认为最适合接在 `BSR` 后面的具体化版本。

### 核心假设

真正有价值的 repair object 不是“高 entropy token 列表”，而是：

- 连续的、不稳定的、彼此耦合的 `island`
- 且这些 island 的边界一旦稳定，就应该被立刻锁死

如果我们把 revision 单位从 token 改成 `stability island`，并允许每个 island 独立 early stop，那么有机会：

- 减少对稳定区域的副损伤
- 在更低 NFE 下保留甚至提升 `net_repaired`

### 方法草图

1. 用现有最稳的 draft 配置起步：`64-step`, `isothermal T=1.0`
2. 在中后段收集三类无需训练的局部信号：
   - raw entropy
   - temporal volatility（连续两步 token 分布/argmax 变化）
   - boundary tension（高风险 token 两侧的稳定-不稳定差）
3. 将相邻高风险 token 聚合为 `island`
4. 仅对 top-k islands 做局部 remask / revision
5. island 外全部 `hard freeze`
6. 每个 island 单独满足 early-stop 条件时立即关闭，不等全局同步

### 为什么它比 `MGCD/DSG` 更对题

- 比 `MGCD` 少了多轨 memory graph 与 dual-draft hidden compute
- 比 `DSG` 少了 token-wise top-p 切点带来的碎片化修改
- 比已有 `CARD-84` 更像“局部修理器”，而不是“再次全局扫描”

### 与相关工作的差异

- 不做 `Prism` 式 branching/search
- 不做 `RDD` 式 backtracking
- 不做 `ReMDM` 式更一般的 remasking compute scaling
- 重点是：`局部 object + 独立停止 + harm containment`

### 最小 pilot

- 模型：优先最小可用 dLLM checkpoint，不先上大模型
- 数据：`GSM8K audited slice >=100`，通过后再进 `MBPP audited slice >=100`
- 对照：
  - `RAND-84`
  - `CARD-84`
  - `SIR-56`
  - `SIR-64`
  - `SIR-84`
- 必记指标：
  - accuracy
  - `net_repaired`
  - harm count
  - mean island size
  - revised-token ratio
  - avg NFE
  - latency / sample

### 计算成本

- 实现复杂度：低到中
- 单次 pilot：单卡可完成
- 若按项目约束吃满 batch，建议先探测到 `128-182` 再跑，不要再停在 `batch_size=50`
- 100 样本 slice 预计可压在 `30-45 min`

### 成功概率

`0.45`

### 失败模式

- entropy island 仍然不是 causal edit object
- 边界锁定过早，导致数学推导链无法连带修复
- 只减少 harm，但不能穿透 `RAND-84`

## Angle 2: AMR-DLM

### 名称

`AMR-DLM: Adaptive Mesh Refinement for Diffusion Language Models`

### 类型

跨领域迁移；主打“在维持性能时提速”。

### 核心假设

当前 DLM sampler 的一个隐含错误是：把整条序列都当成同样难、同样值得继续求解的对象。  
但从数值 diffusion / PDE 求解器的经验看，这通常是浪费的。更合理的方式是：

- 先粗粒度推进全局
- 只在高残差区域细化
- 在平稳区域降精度或停止更新

如果这个直觉迁移到 DLM 成立，我们可能在不明显掉点的情况下拿到显著 NFE 下降。

### 方法草图

1. 先用 coarse block schedule 快速走到中段
2. 对每个 block 计算局部 residual proxy：
   - entropy slope
   - token flip rate
   - answer-slot sensitivity
3. 只对高 residual blocks 细化为更小粒度 step
4. 低 residual blocks 则：
   - coarse hold
   - skip update
   - 或直接 freeze
5. 当 block residual 降到阈值以下时允许 coarsen

### 为什么它值得保留

- 这是一个真正 `speed-first` 的 training-free 方向
- 它不押注“更强纠错器”，而押注“更好的 compute geometry”
- 与项目里“标准 denoising 非单调”这一证据强一致

### 与相关工作的差异

- `Prism` 是 hierarchical search；这里是 solver-style local refinement
- `Block Diffusion / FAST-DLLM` 强调 block-level generation；这里强调 block-level `adaptive accuracy budget`
- 数值 AMR 文献给的是方法论启发，不是 DLM 上已存在的标准答案

### 最小 pilot

- 只先做 `GSM8K audited slice >=100`
- 对照：
  - fixed `CARD-64`
  - fixed `CARD-84`
  - `AMR-avg56`
  - `AMR-avg64`
- 判据：
  - 是否能在 `avg NFE` 下降 `20-35%` 时维持 accuracy 不显著恶化
  - 是否减少对稳定区域的无效更新

### 计算成本

- 实现复杂度：中
- 单次 pilot：单卡可做
- 关键不是多跑，而是把 batch 撑满、把 compile/attention stack 打开
- 100 样本 slice 目标是 `<=1h`

### 成功概率

`0.30`

### 失败模式

- block residual proxy 不够可靠
- coarse/fine 边界引入额外 artifact
- 省下来的步数被调度开销吃掉

## Angle 3: UGR

### 名称

`UGR: Uplift-Gated Revision`

### 类型

全新方法；我认为这是本轮最有研究味道、也最符合 pivot 逻辑的一条线。

### 核心假设

我们当前最大误判是把 `uncertainty` 当成了 `edit utility`。  
更合理的问题不是：

- “哪里危险？”

而是：

- “改哪里会带来正的净收益？”

`UGR` 的核心思想是：不要只按 raw entropy 选 revision 位置，而要对候选 island 做一个极便宜的 `uplift test`。

### 方法草图

对每个候选 island，只做一个受预算约束的成对微试验：

1. `freeze branch`：该 island 不改，继续走 1-2 个微步
2. `revise branch`：该 island 局部 remask/revise，其他位置冻结，走 1-2 个微步

然后比较两支的 cheap utility proxy：

- local entropy drop
- answer extractor stability
- equation / bracket / syntax consistency
- 与前后文的一致性分数

只有当 `revise branch - freeze branch > threshold` 时才真正提交这次 edit。

### 为什么我觉得它是最“对症”的创新

因为它正面回答了当前最关键的失败原因：

- `MGCD/DSG` 都说明“检测到风险”还不够
- 下一步要学会估计的是 `intervention effect`

这条线与 [Rethinking Confidence Calibration for Failure Prediction (2023)](https://arxiv.org/abs/2303.02970) 的教训也高度一致：

- failure predictor 可以有用
- 但 downstream decision rule 需要另一个层次的目标

### 与相关工作的差异

- 不是 generic backtracking
- 不是 generic self-verification
- 不是 search for best-of-N
- 而是一个严格 budgeted 的、局部的、近似因果的 `edit veto / accept` 机制

### 最小 pilot

- 只在 top-1 或 top-2 island 上运行微试验，严格限制 hidden compute
- 对照：
  - `CARD-84`
  - `SIR-84`
  - `UGR-lite-84`
  - `UGR-lite-84-sham`
- 关键不是绝对 accuracy，而是：
  - 是否比等算力 sham 分支更高 `net_repaired`
  - 是否明显减少 harm

### 计算成本

- 实现复杂度：中到高
- 单次 pilot：若微分支严格限在 top-1/top-2 island，仍可落在 `<=1h`
- 必须记录额外 branch NFE，否则 reviewer 会直接判 hidden-compute confound

### 成功概率

`0.25`

### 失败模式

- utility proxy 太 noisy
- uplift 估计不稳定，反而让 gating 更脆弱
- 小分支试验带来的额外 compute 抵消所有收益

## 我建议保留的 serious pool

如果只能留 `2-3` 个 serious ideas，我的排序是：

1. `SIR`
2. `UGR`
3. `AMR-DLM`

原因很简单：

- `SIR` 最贴近当前证据，最容易快速证伪
- `UGR` 最可能把“observer != controller”这个核心矛盾真正推前一步
- `AMR-DLM` 是最值得保留的 speed-first 分支，但更像系统/求解器创新，不一定最先给质量正增益

我不建议继续把下面两个对象留在 serious pool：

- `cand_mgcd`
- `cand_dsg`

它们现在更适合作为：

- 已筛掉的负例证据
- 或者最多作为某些局部机制的灵感来源

而不是继续吃主线预算。

## 最重要的实验纪律

无论下一轮押哪条线，都必须补上这几个纪律，否则 idea 再新也会被 execution envelope 吃掉：

1. 下一轮 pilot 回到项目约束，`sample_count >= 100`
2. 先做 batch probe，再尽量接近 safe batch 上限；当前已知 safe probe 是 `182`，不要继续保守跑 `50`
3. `flash_attention_2` / `torch.compile` / 多卡拆分如果能开就开
4. 所有新候选都必须带等算力 sham/control
5. 在 `H1` 重新通过前，不进入 full benchmark

## 一句话结论

这轮 `innovator` 不该再争论“MGCD 该不该再给一次机会”，而该把主问题改写成：

> 我们能不能把 DLM 的 revision 从“检测高风险位置”升级成“只在真正值得干预的局部区域上花 compute”？

如果答案是能，那么最值得先试的不是更大的 memory graph，而是：

- `SIR`：局部 island 级修理
- `UGR`：干预收益门控
- `AMR-DLM`：求解器式局部细化
