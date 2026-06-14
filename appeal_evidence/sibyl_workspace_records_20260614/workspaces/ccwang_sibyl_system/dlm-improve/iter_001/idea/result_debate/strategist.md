# Strategist Round 1

## 一句话战略判断

当前最优路径是 **PROCEED with cand_diag**，但必须把项目明确定位成一篇“honest-compute 下的 DLM revision diagnostic benchmark / mechanism paper”，而不是继续包装成一个新 revision 方法论文。

## 为什么这是当前最值钱的方向

### 1. 证据已经支持诊断型主张，而不再支持方法型主张

- `diag_compute_curve_gsm8k.json` 显示名义 compute 与实际 compute 会改变关键排序：
  - `CORE-proxy-64` 的 `compute_gap_pct = 7.81%`
  - 它在 nominal 排序领先 entropy revision，但在 actual compute 排序里落到后面
- `diag_signal_gap_audit.json` 显示 observer/controller gap 已经成形：
  - calibration `diagnostic_score = 0.6225`
  - 但 `control_effectiveness = 0.0`
  - entropy 也有 `diagnostic_score = 0.414`，但相对 random revision 的额外收益仍然接近 0
- `diag_math500_shortlist.json` 说明 GSM8K 叙事并不直接平移到第二 reasoning benchmark：
  - MATH500 排名变成 `Standard > CORE-proxy > TIGER > Entropy`
  - 这正好支撑“revision gain is task-dependent / benchmark-dependent”
- `diag_humaneval_guard_boundary.json` 表明 code 侧只能做 boundary evidence：
  - `Gated TIGER` 把 syntax failure 从 `0.48` 压到 `0.28`
  - 但 `pass@1 = 0.04` 仍低于 `Standard = 0.06`
  - runtime failure 反而更高

战略上，这四条证据指向同一个结论：
**我们已经拥有一篇更诚实、更不依赖单点方法胜利的论文核心。**

### 2. 继续押注 TIGER 的边际回报已经很低

当前数据没有支持 “instability-guided controller > entropy controller”：

- GSM8K shortlist 里 `TIGER = 0.39`，`Entropy = 0.39`
- compute-normalized 分析里，TIGER 相对 Entropy 没有质量优势，反而多 `1` 个 actual NFE、慢约 `3.14s`
- HumanEval gating 也没有把方法故事救回来

如果这时还继续往 TIGER 投 GPU 预算，战略上是在用算力为一个已经失去 headline 资格的子命题续命。

### 3. cand_diag 的论文定位更容易形成 reviewer 可接受的贡献闭环

最可发表的 framing 不是“我们发明了更强 revision policy”，而是：

1. **honest compute accounting** 会改变 DLM inference 方法的公平排序
2. **revision 的收益具有任务依赖性**
3. **强观测信号不等于强控制策略**
4. **浅层 syntax guard 不能恢复深层 semantic / runtime correctness**

这四点放在一起，比单独证明某个 revision heuristic 多赢 1-2 个点更有战略价值，也更符合这轮真实证据。

## 推荐的下一步动作（按优先级）

### P0. 立即锁定论文叙事，不再让方法主张回流

需要马上冻结三件事：

- 主标题和摘要不能再出现 method-forward 口吻
- TIGER 降级为 supporting condition / failed controller case
- HumanEval 只保留 appendix/boundary 定位，不再作为“泛化成功”的证据

如果这一步不先锁，后面写作很容易又滑回“明明是 benchmark/diagnostic 论文，却写成方法论文”的旧问题。

### P1. 把 full paper 的核心图表与表格定义为“机制证据包”

我建议主文至少保证以下几类核心资产：

1. `matched-compute ranking table`
   - Standard / DNB / Prophet / CORE-proxy / Entropy / TIGER
   - 报 `accuracy`, `actual NFE`, `latency`, `TPS`
2. `observer vs controller table`
   - calibration / entropy / instability
   - 报 `diagnostic score` 与 `control effectiveness`
3. `GSM8K vs MATH500 ranking change`
   - 直接展示 benchmark-dependent ordering
4. `failure taxonomy table for code boundary`
   - syntax / runtime / pass@1 拆开报

这套资产一旦齐，论文就不再依赖某个单一 headline result。

### P2. 优先补 benefit buckets / failure buckets，而不是继续扩更多新方法

proposal 里已经写了 `Benefit Buckets`，这是现在最值得投时间的地方，因为它能把“均值故事”变成“机制故事”：

- draft correct -> revision harmed
- draft wrong -> revision fixed
- revision no effect

如果我们补出这一步，cand_diag 就从“结果整理”升级成“可解释机制图谱”。

### P3. 用最少预算保留 backup，而不是切换主线

- `cand_minimal`：保留为 backup sanity check
- `cand_factorization`：保留为下一轮理论升级方向

但当前不建议切过去。原因很简单：
现在最短可发表路径已经不是“找新方法”，而是“把已有证据组织成更硬的机制论文”。

## 资源分配建议

### 1. 计算预算

- **70%** 用于 cand_diag 主线的 full evidence packaging
  - compute-normalized tables
  - benchmark transfer
  - observer/controller diagnostics
  - failure buckets
- **20%** 用于极少量 backup sanity checks
  - 只做最能威胁 cand_diag 主张的那种 minimal baseline 对照
- **10%** 保留给写作期发现的缺口修补

### 2. 写作预算

写作上要尽快前置：

- framing 固化
- contribution list 重写
- result section 骨架先行

否则实验做得越多，后面越容易在 narrative 上返工。

### 3. 风险缓冲

不要再把大量预算花在：

- TIGER 的进一步 controller 微调
- code benchmark 扩张
- 更复杂 revision signal 的新发明

这些方向现在都不是最短价值路径。

## 风险评估

### 方向 A：继续 cand_diag

风险：

- 如果后续只报表格，不给 mechanism buckets，会像“benchmark 整理”
- 如果不明确说明为什么 MATH500 与 GSM8K 排序不同，reviewer 会觉得只是噪声

回报：

- 最接近当前证据形态
- 最少依赖额外 miracle result
- 最容易把负结果变成贡献

### 方向 B：回切 cand_tiger

风险：

- headline 已经被 honest compute 和 CORE-proxy 压住
- 再投算力很可能只是继续证明它“不够强”

回报：

- 很低，除非出现一个非常窄但极强的新 win condition

### 方向 C：切到 cand_minimal / cand_factorization

风险：

- 要重新建立 pilot 闭环
- 会延长投稿路径

回报：

- 可能形成下一轮更锋利的故事
- 但不是当前这轮最短的高价值路径

## 是否 PIVOT 还是 PROCEED

我的战略结论是：

**PROCEED，但对象不是“revision 方法开发”，而是 “cand_diag 的 full diagnostic paper 路线”。**

更准确地说，不是继续推进旧主张，而是：

- 对方法主张执行收缩
- 对诊断主张执行放大

这不是防守姿态，而是把当前最难反驳、最能转化为论文贡献的证据放到舞台中央。

## 给主系统的执行建议

1. 锁定 `cand_diag` 为唯一 front-stage candidate
2. 明确把 TIGER 改写为对照/失败案例，而不是候选主方法
3. 下一轮优先补 benefit buckets 和 failure taxonomy
4. 写作时先做 result-first outline，再决定是否需要追加少量验证实验

## Round 2

skeptic、methodologist、comparativist 的担忧基本指向同一个战略事实：这条线不是不能发，而是**必须继续收缩**。如果我们还想把它写成“大而全的 DLM benchmark / 方法比较论文”，会立刻暴露三个弱点：

1. 样本规模仍偏小，很多差值只是方向性证据，不足以支撑大而广的 superiority claim。
2. compute story 里混有系统实现差异，因此主张必须落在“honest accounting changes the story”，而不是“我们证明某算法本质更快/更慢”。
3. code 侧和 MATH500 侧更适合做边界与迁移证据，不适合抬成 headline。

所以 Round 2 的战略建议不是“再补很多实验把论文做大”，而是“把论文压缩成最短可发表路径”。

### 必须进主文的

这些内容构成最小但完整的贡献闭环，少一个都会让论文失去中心：

1. **Honest-compute reordering**
   - 主文必须保留 `diag_compute_curve_gsm8k` 的核心表格。
   - 重点不是所有方法排名大洗牌，而是：
     - nominal vs actual compute 不等价
     - 关键 headline 会因 actual compute 改写
     - `CORE-proxy-64` 是最典型例子
   - 写法必须克制：强调 “ranking/pareto narrative changes”，不要写成“existing papers are broadly invalidated”。

2. **Observer vs Controller mismatch**
   - 主文必须保留 `diag_signal_gap_audit`。
   - 这是最像“机制贡献”的部分，因为它把 calibration / entropy / instability 从“更强信号”重写成“更像 observer 的 signal family”。
   - 这里要把 claim 压在：
     - `diagnostic value != control value`
     - calibration / entropy 有观察价值，但当前不自动变成更好控制律
   - 不要写成更强的因果宣言。

3. **一条 task-dependent 迁移证据**
   - 主文里要有 GSM8K 与 MATH500 的对照，但目的不是吹 MATH500 排名本身，而是证明：
     - revision response 不该被当成单 benchmark universal truth
   - 换句话说，MATH500 在主文里是“支持 task-dependence 的第二坐标轴”，不是独立 headline。

4. **Benefit buckets / failure buckets**
   - 这是现在最应该补、也最该进主文的新增资产。
   - 因为 skeptic 和 methodologist 都指出：如果没有 bucket 级证据，整篇论文会像结果整理而不是机制论文。
   - 战略上，这比继续补更多 baseline 更值钱。

### 只能进 appendix 的

这些内容有价值，但放进主文会稀释主叙事，甚至引来不必要攻击：

1. **HumanEval guard boundary**
   - 必须降到 appendix。
   - 它现在最合适的作用只有一个：
     - 证明 syntax guard 更像 shallow safeguard，而不是 general recovery mechanism
   - 因为 `n=50`、`pass@1` 太低，放主文只会被 reviewer 抓住样本噪声和 absolute weakness。

2. **详细 runtime config 表**
   - `batch_size / backend / compile / peak_vram`
   - 很重要，但更适合作为 appendix fairness table
   - 主文只保留一句明确声明：actual compute comparison includes real systems/runtime effects

3. **TIGER / DNB / Prophet 的细碎对比**
   - 这些可以留在 appendix 里作为 supporting comparisons
   - 主文里只需要保留它们在核心表中的位置，不需要展开各自小差值

4. **MATH500 的细节 per-method 展开**
   - appendix 可放完整表
   - 主文只需要“与 GSM8K 排序不完全一致”这一层证据

### 最好不再写的

如果目标是最短可发表路径，下面这些内容最好从主叙事里主动删除，否则只会拉长战线：

1. **“我们提出了更强 revision 方法”**
   - 这条线已经结束了。
   - 再写只会正面撞上 CoRe 一类 concurrent work，也会被自己的结果反驳。

2. **“TIGER 是主方法贡献”**
   - 最多保留为 failed-but-informative controller case。
   - 不要再给它单独的方法章节中心位置。

3. **“code 结果证明泛化成功”**
   - 不能再写。
   - 现在 code 只证明 boundary/failure taxonomy，不证明方法泛化。

4. **“校准/熵/不稳定性是更好 controller”**
   - 现在最稳的写法恰恰相反：
     - 它们首先是 observer
     - controller 价值需要另证

5. **过度展开 external leaderboard 叙事**
   - comparativist 已经指出，我们不该和通用 SOTA 比绝对分数。
   - 主文只需要把自己放在 DLM revision/decode literature 的纠偏位置，不需要追逐更大的 benchmark 口号。

### 最短可发表路径

如果我要把路线压到最短，我会建议主系统只保留下面这条骨架：

1. **主问题**
   - DLM revision literature often compares nominal steps, conflates observers with controllers, and overgeneralizes from narrow wins.

2. **主证据 A**
   - honest-compute accounting changes key ranking/Pareto conclusions

3. **主证据 B**
   - observer/controller mismatch is real

4. **主证据 C**
   - revision response is task-dependent, not universal

5. **主证据 D**
   - code-side guards fix shallow syntax issues but not deep execution correctness

6. **机制增强**
   - benefit buckets / failure buckets

这条骨架已经足够形成一篇紧凑、诚实、可 defend 的论文。再往外扩，只会增加 reviewer 可攻击面。

### 最终收敛建议

我的 Round 2 结论是：

- **主文只讲三件事**：honest compute、observer/controller mismatch、task-dependent revision response
- **appendix 承接两件事**：code boundary、runtime fairness details
- **彻底删除三种诱惑**：方法胜利叙事、code 泛化叙事、controller 神化叙事

如果这样收缩，论文会从“一个想证明太多事情的项目”变成“一个命题明确、证据闭环、攻击面较小的分析论文”。

## Round 3

### 1) 主文必须讲的 3 个点

1. honest compute 会改写 DLM revision 方法的排序与 Pareto 叙事，名义步数不够诚实。  
2. entropy / calibration / instability 更像 observer，而不是自动成立的 controller。  
3. revision gain 是 task-dependent 的：reasoning 与 code 不在同一个 response regime。

### 2) appendix 承接什么

appendix 只承接两类内容：code boundary 证据，以及 runtime / batch / compile / backend 的公平性细节表。  
TIGER、DNB、Prophet 的细碎对比和 MATH500 的完整展开也放 appendix，不占主文叙事带宽。

### 3) 接下来唯一的 P0 执行动作

立刻补出 `benefit buckets / failure buckets` 的统一结果资产，并据此重写主文 Results 骨架。  
这是把论文从“结果整理”变成“机制论文”的最短一步，也是现在唯一真正值得优先投入的动作。
