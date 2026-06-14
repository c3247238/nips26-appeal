# Skeptic Round 1

## 结论先行

我不反对把 `cand_diag` 作为当前最合理的 front-runner，但如果现在就把它包装成“已经成立的 benchmark paper”，我会明确反对。当前证据更像是一个**有潜力但仍然脆弱的 diagnostic story**，离稳健主张还有几步。

## 1. 统计担忧

1. 当前所有关键结论几乎都建立在很小的样本上：
   - `diag_compute_curve_gsm8k.json` 是 `n=100`
   - `diag_signal_gap_audit.json` 是 `n=100`
   - `diag_math500_shortlist.json` 是 `n=100`
   - `diag_humaneval_guard_boundary.json` 是 `n=50`
   这些规模足以做方向性判断，但不足以支撑强论文叙事。
2. 没有 seed-to-seed 方差，没有 bootstrap 置信区间，也没有 McNemar 或 permutation test。现在所有“排序变化”“gap 成立”“task-dependent”都还是点估计。
3. `MATH500` 上方法间差异太小，且整体准确率都很低：
   - `Standard-64 = 0.23`
   - `CORE-proxy-64 = 0.21`
   在这种低准确率区间里，`2pp` 很可能就是噪声或 extractor 偶然性，不足以支撑“迁移后排序变化”的强主张。
4. `HumanEval` 上更严重：
   - `Standard pass@1 = 0.06`
   - `Gated TIGER pass@1 = 0.04`
   - `Ungated revision pass@1 = 0.02`
   在 `n=50` 上，这分别只对应 3、2、1 个样本。这里任何 1 个样本的波动都会改写叙事。
5. `diag_compute_curve_gsm8k.json` 里最核心的 honest-compute 证据其实只有 `1` 个 pairwise reorder，`max_abs_compute_gap_pct = 7.81`。这说明信号存在，但远没有强到“旧 ranking 普遍失真”的程度。

## 2. 可能的混杂因素与替代解释

1. `CORE-proxy-64` 的实际 compute / latency 劣势，可能部分来自实现选择，而不一定来自方法本质：
   - `batch_size = 1`
   - `compile_enabled = false`
   - `latency_sec = 482.953`
   如果 reviewer 认为这是 engineering handicap，而不是 algorithmic cost，那么 honest-compute 叙事会被削弱。
2. observer-vs-controller gap 也可能只是“控制器实现太弱”，不一定说明“observer 天生不适合做 controller”：
   - `calibration diagnostic_score = 0.6225, control_effectiveness = 0.0`
   - `entropy diagnostic_score = 0.414, control_effectiveness = 0.0`
   这也可以被解释成：你们没有找到好的 control law，而不是 observer/controller 真有结构性分离。
3. `MATH500` 的“排序变化”也有另一种解释：不是 task-dependent regime，而是**所有方法都在低准确率区间聚团**，导致微小波动足以翻转排序。
4. `HumanEval` 的 guard 结果未必证明“浅层修复无法恢复深层能力”，也可能说明当前 gate 过于保守：
   - `gate_open_rate = 0.14`
   只放开 14% 的样本，本来就很难指望 pass@1 超过 Standard。

## 3. Proxy Metric Gaming 检查

1. `actual_nfe` 可能被过度神化。论文若把 honest compute 主要压在 `actual_nfe` 上，reviewer 会追问：为什么 `latency`、`TPS`、`batchability`、`compile/offload` 不应被视为 equally first-class metrics？
2. 当前 compute story 混合了 algorithmic cost 与 systems cost：
   - `CORE-proxy-64` 的 `actual_nfe = 69`
   - 但它的 `latency_sec` 远高于 entropy/TIGER
   这更像一个“系统实现与方法耦合”的故事，而不是纯算法排序重排。
3. signal-gap 分析里的 `control_effectiveness` 定义也可能被 reviewer 认为太依赖你们挑选的 baseline：
   - entropy / instability 都是对 `Random-Revise` 的 delta
   - shortlist 里又报告对 `Standard` 的 delta
   这会被质疑为 metric framing 可以改写结论。
4. `cand_diag` 说要做 failure taxonomy 和 benefit buckets，但当前结果文件里还没有真正把 “draft-correct-but-harmed / draft-wrong-but-fixed / no-effect” 系统拆出来。现在更多还是 aggregate-level proxy。

## 4. 还缺什么关键证据

1. 缺多 seed。没有这个，所有 task-dependent claim 都不稳。
2. 缺正式显著性检验。尤其是：
   - `GSM8K` 上 `0.36 -> 0.39`
   - `MATH500` 上 `0.23 vs 0.21`
   - `HumanEval` 上 `0.06 vs 0.04`
   这些差值都需要 paired test，而不是肉眼判断。
3. 缺 full-scale matched-compute replication。现在最核心的 honest-compute 与 signal-gap 结论都还是 100-sample slice。
4. 缺真正的 benefit buckets / failure taxonomy 结果表。proposal 里把它写成核心贡献，但现在证据还没到位。
5. 缺对外部最强相关工作的更正面对齐。现在内部比较有 `Standard / DNB / Prophet / CORE-proxy / Entropy / TIGER`，但 reviewer 仍会问：为什么这已经足以代表“revision in DLMs under honest compute”。

## 5. 我要求的追加实验

1. `GSM8K-100` 和 `MATH500-100` 至少做 `3` 个 seeds，并报告 mean/std 与 paired significance。
2. 把 `diag_compute_curve_gsm8k` 扩到更大样本，验证那个 `1` 个 reorder 不是偶然。
3. 把 observer-vs-controller gap 改成更严格的 paired bucket analysis：
   - 高 signal 样本里，revision 到底修了多少
   - 低 signal 样本里，revision 到底伤了多少
4. 在 code 侧补真正的 semantic failure 分析，而不是只停在 syntax/runtime 二分。
5. 对 `CORE-proxy` 做最公平的 systems normalization：
   - 明确说明 `batch_size = 1` 是方法限制还是实现限制
   - 说明 `compile_enabled = false` 是否不可避免

## 当前怀疑判断

如果今天就写论文，我会给出这样一句话：

`cand_diag` 已经摆脱了 `cand_tiger` 的方法失败包袱，但它目前仍然只是“有说服力的 pilot narrative”，还不是“经得起 reviewer 拆解的 full evidence package”。

## Round 2

如果我们不再主打 `TIGER`，而改主打 `diagnostic benchmark / honest compute / observer-controller gap`，最危险的地方并没有自动消失，只是从“方法没赢”变成了“benchmark 主张可能仍然不够硬”。

### 1. 仍然没有消失的拒稿风险

1. **benchmark 规模仍然太小，像 pilot report，不像 benchmark paper。**
   现在的核心证据仍然是：
   - `GSM8K 100`
   - `MATH500 100`
   - `HumanEval 50`
   这对方法探索够了，但对一个自称 benchmark / diagnostic framework 的论文来说明显不够。reviewer 很容易说：这不是 benchmark，只是小规模 case study。
2. **honest compute 的 claim 还没有强到足以支撑 headline。**
   现在最直接的证据是 `diag_compute_curve_gsm8k.json` 里只有 `1` 个 pairwise reorder，`max_abs_compute_gap_pct = 7.81`。这可以支持“需要更诚实地报告 compute”，但还不足以支持“existing conclusions materially change under honest compute”这种更大的 claim。
3. **observer-controller gap 还没有被严格区分为“结构性现象”而不是“当前控制器太弱”。**
   目前我们看到的是：
   - calibration 有强 diagnostic score
   - entropy 有中等 diagnostic score
   - 控制增益接近零
   但 reviewer 完全可以说：这只说明你们当前设计的 control policy 不够好，而不是 observer/controller 存在更普遍的角色分离。
4. **benchmark framing 仍然可能被质疑为“把负结果重新包装成贡献”。**
   如果论文写法不够克制，审稿人会直接说：TIGER 没赢，code 也失败，于是作者改讲 benchmark 与 diagnostics。这个转向可以成立，但必须靠更扎实的证据避免被看成 post-hoc reframing。
5. **外部有效性仍然不足。**
   当前几乎全部证据还锁定在：
   - 单模型族
   - 单轮实验配置
   - 很少的 benchmark 切片
   这会让 reviewer 怀疑：你们到底发现了 DLM 的一般规律，还是只是在这套 LLaDA-8B + 这几个 slice 上观察到一个局部现象？

### 2. 现在仍然太大的 claim

1. **“RIDE-Bench”这个命名本身就偏大。**
   以目前证据，更准确的叫法更像是：
   - diagnostic study
   - compute-normalized evaluation protocol
   - revision failure analysis
   而不是一个已经成型、可复用、足以成为社区基线的 benchmark。
2. **“honest compute 改变排名”这个 claim 仍然太大。**
   更稳妥的说法应该是：
   - honest compute can change some pairwise comparisons
   - nominal-step reporting can misstate Pareto position
   而不是暗示大面积 ranking inversion。
3. **“observer 比 controller 更有用”这个 claim 仍然太大。**
   现在最多能说：
   - 在当前 tested policies 下，observer quality does not reliably translate into control gains
   还不能说成一个更一般的机制定律。
4. **“revision gain is task-dependent”这个 claim 也还偏大。**
   因为当前 reasoning 侧的增益本身并不大，code 侧的样本又过少。现在最多能说“there are early signs of task dependence”，还不能写成已经稳定建立的 phase diagram。

### 3. 最低必需证据：不是加分项，而是门槛

下面这些不是“有的话更好”，而是如果没有，我会建议不要把论文主标题切到 benchmark / diagnostic framework：

1. **full-scale replication of the core diagnostic claims**
   至少要把 honest-compute 与 observer-controller gap 的核心结果扩到明显大于 `n=100` 的规模，尤其是 GSM8K 主结果。
2. **multi-seed uncertainty**
   不是所有实验都要 10 个 seed，但最关键的排序与 gap 结论必须有 seed-level 稳健性。没有方差，benchmark claim 站不住。
3. **paired statistical test for every headline comparison**
   尤其是：
   - Standard vs Entropy
   - Entropy vs TIGER
   - CORE vs revision family
   - Standard vs Gated TIGER on code
   否则 reviewer 会说你们把几道题的偶然波动讲成机制。
4. **明确区分 algorithmic compute 与 systems compute**
   现在 honest compute 混合了：
   - actual NFE
   - latency
   - throughput
   - batchability
   论文必须明确：哪些是方法内生代价，哪些是实现后果。否则“honest compute”会被批成口径混杂。
5. **真正完成 benefit buckets / failure taxonomy**
   这是当前新 framing 最像贡献的地方，但现在还主要停在 aggregate 指标。没有 bucket-level breakdown，diagnostic paper 会显得空心。
6. **更清楚的 benchmark positioning**
   必须回答：
   - 这是 benchmark 吗？
   - 还是 protocol？
   - 还是 diagnostic case study？
   如果这点不收紧，标题和贡献会被 reviewer 认为 oversell。

### 4. 我对新 framing 的最尖锐质疑

我最担心的一点是：我们现在把“方法没有形成压倒性优势”转译成“诊断框架才是真贡献”，但诊断框架本身还没有完成到足以独立站住的程度。

换句话说，旧风险没有消失，只是换了形式：

- 旧风险：`TIGER` 不够强
- 新风险：`cand_diag` 不够大、不够稳、不够像 benchmark

如果不补最低必需证据，这篇稿子最容易收到的评价会是：

`This is an honest and interesting pilot study, but the benchmark / framework framing is still overstated relative to the amount of evidence provided.`

## Round 3

1. 目前最危险的过度声称是什么  
最危险的是把这篇稿子叫成一个已经成立的 `diagnostic benchmark`，因为现有证据更像小规模 diagnostic study，而不是足以定义社区 benchmark 的完整包。`honest compute 会系统性改写排名` 和 `observer-controller gap 是一般规律` 这两个说法现在也都偏大。

2. 哪个缺口如果不补最容易被拒  
最容易导致拒稿的是核心 claim 没有 full-scale + uncertainty 支撑：目前没有多 seed、没有成体系显著性检验、核心 honest-compute / gap 结果仍主要停在 `n=100`。如果这一点不补，reviewer 很容易把整篇文章降格为“有意思但不稳的 pilot”。

3. 你认可的最小可接受论文定位  
我认可的最小可接受定位是：**一个诚实的 compute-normalized diagnostic study，展示 revision in DLMs 的早期机制证据与失败 taxonomy**。不要把自己写成 benchmark standard-setter，而要写成“提出一个更严谨的分析协议，并用两类任务展示它为什么重要”。 
