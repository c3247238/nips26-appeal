# Comparativist Round 1

## 结论先行

如果把这组结果放到 2026 年初的 DLM decoding 文献版图里看，`cand_diag` 的正确定位**不是**“我们提出了一个更强的新 revision 方法”，而是：

1. 我们证明了 **honest compute accounting 会改变 DLM decoding 方法排序**。
2. 我们把 **observer vs controller** 的错位说清楚了。
3. 我们给出了 **reasoning / code 两类任务对 revision 响应并不相同** 的证据。

这三点合起来，比较像一篇“机制澄清 + benchmark framing + failure taxonomy”论文，而不是一篇“新 decoding 算法 SOTA”论文。

## 1. 相对同类 DLM 方法，当前结果站在哪里

### 1.1 相对 `CORE-proxy`

从本轮自己的结果看，`CORE-proxy-64` 仍然是最强的 method-forward 对手：

- 在 `diag_compute_curve_gsm8k.json` 里，`CORE-proxy-64` 的 GSM8K accuracy 是 `0.46`，高于：
  - `Entropy-Revise-64+3 = 0.39`
  - `TIGER-Instability-64+3 = 0.39`
  - `Standard-64 = 0.36`
- 但 `CORE-proxy-64` 的实际代价也明显更高：
  - `actual_nfe = 69`
  - `latency_sec = 482.953`
  - `batch_size = 1`

这意味着我们的 strongest comparative point 不是“打败 CORE”，而是：

- **CORE 的 headline 胜利在 honest compute / latency 口径下不再那么干净**
- 它仍然 accuracy 最强，但其 Pareto 位置比 nominal-step 叙事更差

这点很重要，因为它把论文贡献从“我也做了一个 revision method”转成了“现有 DLM decoding paper 普遍缺 honest compute comparison”。

### 1.2 相对 revision baselines

对 revision-family 内部比较，本轮证据非常明确：

- `TIGER` 没有比 `Entropy revision` 更强
  - GSM8K matched-compute: `0.39 vs 0.39`
  - `actual_nfe`: `69 vs 68`
  - latency 也几乎持平
- `DNB-84` 也没有给出比 `Standard-64` 更好的 accuracy
  - `0.36 vs 0.36`
- `Prophet-64` 反而在 compute 上略快于 `Standard-64`，但 accuracy 更低
  - `0.34 vs 0.36`

外部定位上，这说明我们**不拥有**一个新的强 method margin。相反，我们拥有的是：

- 一个对 revision family 内部“谁真的更值”更诚实的重排序框架

## 2. 与最新 concurrent work 的关系

我补了最新文献/网页检索，最值得警惕的两个 concurrent 方向是：

### 2.1 `CoRe: Context-Robust Remasking for Diffusion Language Models`

arXiv: https://arxiv.org/abs/2602.04096

从检索摘要看，`CoRe` 的主张是：

- 现有 revision strategy 依赖静态 confidence signal，信号本身是 myopic 的
- 在 reasoning 和 code benchmark 上都能稳定优于 compute-matched baselines
- 在 MBPP 上有显著提升

这对我们意味着两件事：

1. 如果我们的论文还想主打“我们有一个更好的 revision signal / revision policy”，会被 `CoRe` 正面对撞。
2. 但如果我们主打“为什么这些 signal 常常更像 observer 而不是 controller”，我们与 `CoRe` 就不是完全重叠，而是形成一种更上位的解释框架。

换句话说，`CoRe` 抢走了“更强 remasking method”这条叙事，但**没有直接抢走 honest-compute diagnostic benchmark** 这条叙事。

### 2.2 `Improving Diffusion Language Model Decoding through Joint Search in Generation Order and Token Space`

arXiv: https://arxiv.org/abs/2601.20339

从检索摘要看，这篇工作报告：

- 在 `GSM8K / MATH500 / Countdown / HumanEval` 上都比 backbone 明显更强
- 甚至可匹敌或超过某些 post-trained diffusion reasoning model

这说明 2026 年的外部环境已经出现更 aggressive 的 decoding/search 方法。对我们很不利的一点是：

- 如果只看 raw benchmark score，我们这轮 `cand_diag` 的数字完全不是同一量级的 headline

但反过来，这也强化了我们的潜在差异化：

- 别人在做“怎么把 decoding method 做得更强”
- 我们可以回答“这些方法到底是靠什么赢、代价是否被诚实计量、哪些收益只是 task-specific 幻觉”

## 3. 相对 benchmark SOTA，贡献边界在哪里

无论是 GSM8K、MATH500 还是 HumanEval，**通用 LLM / reasoning model 的公开 SOTA 远高于我们当前数字**。这一点必须正视：

- 我们自己的 `MATH500` 最好也只有 `0.23`
- `HumanEval` 标准线只有 `0.06`
- GSM8K 最好 `0.46`

而公开 benchmark 生态里，不论闭源还是强开源 reasoning/code model，GSM8K / MATH500 / HumanEval 的绝对分数通常都高得多。

因此这篇工作的可发表点**绝不能**写成：

- “我们在 benchmark 上取得了强性能”

更可信的写法只能是：

- “在 DLM / diffusion decoding 这一受限子领域中，我们重新定义了更诚实的比较口径，并揭示了 revision 的真实收益边界”

## 4. Novelty assessment

### 仍然新颖的部分

我认为还有新意，而且是可以 defend 的新意，主要在三点：

1. **Honest compute benchmark framing**
   - 本轮 `diag_compute_curve_gsm8k.json` 直接显示 nominal ranking 与 actual ranking 不一致
   - `CORE-proxy-64` 从 nominal rank 3 掉到 actual rank 5
   - 这说明已有工作很可能在“方法名义步数”上说故事，却没有把真实 compute / throughput 摆平

2. **Observer vs Controller framing**
   - `diag_signal_gap_audit.json` 里 calibration / entropy 的诊断得分明显高于控制收益
   - calibration gap 甚至达到 `0.6225`
   - 这不是常规 baseline 表格里会自然出现的发现，而是一个很适合写成 paper message 的机制命题

3. **Task-dependent revision response**
   - `diag_math500_shortlist.json` 给出 reasoning benchmark 间的排序变化
   - `diag_humaneval_guard_boundary.json` 说明 code side 的 syntax fix 并不能恢复 runtime / pass@1
   - 这支持“revision 不是统一增益器，而是 task-structure-dependent intervention”

### 已经不新颖甚至危险的部分

1. “我们有更强 revision method” 这条线不新了，而且容易输给 `CoRe` / joint search 类工作。
2. “TIGER 是主方法贡献” 这条线已经被自己的 pilot 否掉。
3. “code 结果证明泛化成功” 这条线也站不住，因为 gated TIGER 仍然没超过 `Standard`。

## 5. Publication readiness

如果现在就投稿：

- 顶会 method track：不够
- 顶会 benchmark / analysis / negative-results-friendly positioning：有机会，但必须重写 framing

更准确地说，这组结果**不适合**包装成“新 DLM decoding algorithm paper”，但**有潜力**包装成：

- 一个 DLM decoding 的 diagnostic / benchmarking / analysis paper
- 强调 honest compute、observer-controller mismatch、task-dependent failure taxonomy

如果写得好，比较像：

- workshop spotlight / findings-style paper：比较现实
- 顶会主会 analysis track：有机会，但前提是把机制图景和 comparison 做得更扎实

## 6. 还缺哪些比较，论文才会更稳

最缺的不是再发明一个新方法，而是把对标链补完整：

1. **显式对标 `CoRe`**
   - 即使只做 proxy-level discussion 或复现有限表格，也比完全不提强得多
   - 否则审稿人会说你忽略最直接 concurrent work

2. **把 honest compute 比较扩成论文主表**
   - 统一列 `accuracy / actual NFE / latency / tokens/sec / batch_size`
   - 不再允许只写 nominal steps

3. **补 benefit buckets**
   - 现在我们知道 revision gain 不是 uniform，但还没把 gain/harm/no-effect 三类样本分桶讲清楚

4. **对 code 增加 failure taxonomy**
   - syntax / runtime / semantic 三层拆开
   - 这是我们相对一般 benchmark paper 最容易形成差异化的地方

5. **如果可能，补一个更简单的 minimal baseline**
   - 因为外部会自然追问：如果复杂 revision 不稳，是否简单 policy 已经足够？

## 7. 对 cand_diag 的最终外部定位

一句话说，`cand_diag` 最好的外部定位是：

**不是 another better decoder，而是对 DLM revision literature 的一篇“纠偏论文”。**

它的贡献边界是：

- 不争 raw benchmark SOTA
- 不争最强 method margin
- 争的是：现有 DLM decoding paper 在 compute accounting、signal interpretation 和 task transfer 上讲得不够诚实、不够机制化

如果按这个方向写，我认为这条线还有真实贡献空间。

## Round 2

如果主线明确改成 **honest-compute diagnostic benchmark**，我认为这条线相对 `CoRe`、joint search 和更广义 revision literature 会**明显更稳**，原因不是我们 suddenly 变强了，而是我们终于把自己放到了一个更难被直接替代的位置上。

### 1. 为什么会更稳

#### 1.1 相对 `CoRe` 会更稳

如果论文主张还是“我们找到一个更好的 revision signal / remasking strategy”，那几乎就是主动走进 `CoRe` 的主战场，因为 `CoRe` 本身就在讲：

- 静态 signal 不够好
- 更 context-robust 的 remasking 才能带来更稳健收益

在这条线上，我们现在没有 margin。  
但如果改成 honest-compute diagnostic benchmark，关系就变成：

- `CoRe` 是被比较和被解释的对象之一
- 我们回答的是：**像 `CoRe` 这样的工作到底在什么口径下看起来更强，真实 compute 下它的优势该怎么解读**

这会把我们从“method loser”转成“evaluation/framing provider”。

#### 1.2 相对 joint search 会更稳

joint search 类工作追求的是更高 raw score、更强搜索、更激进的 decoding。  
我们不可能在当前证据下和它拼 headline result。

但 diagnostic benchmark 叙事可以说：

- aggressive search 方法也应该接受 honest-compute accounting
- 不同任务上的 gain 需要拆开看，不应只看 aggregate score
- revision / search 的收益可能依赖 recoverability，而不是来自统一的“更聪明搜索”

也就是说，joint search 不再是“把我们打败的人”，而是“证明为什么 benchmark framing 必须升级的例子”。

#### 1.3 相对一般 revision literature 会更稳

大多数 revision literature 默认的叙事是：

- 找到更好 signal
- 设计更好 policy
- 在几个 benchmark 上赢 baseline

而我们现在的 strongest message 是：

- signal 强，不代表 control 强
- nominal step 相同，不代表 actual compute 公平
- reasoning 上的小 gain，不能自动外推到 code

这实际上是在 revision literature 之上补一个“审计层”。  
这个层级比单个方法更不容易过时，也更不容易被新 concurrent work 立即替代。

### 2. 但这种改法不是自动安全，还要避开几个对标陷阱

#### 2.1 不能把自己写成“我们比 `CoRe` 更懂机制”

这是最危险的陷阱之一。  
如果我们没有直接重跑 `CoRe`，就不能写出一种口气，好像我们已经证明 `CoRe` 的 gain 只是 compute illusion，或者它的方法解释不成立。

更安全的写法应该是：

- `CoRe` 代表 method-forward 路线的最新强工作
- 我们的结果提示：这类工作需要放到 honest-compute 与 task-dependent response 框架下再理解
- 这是一个 evaluation/framing 补充，不是对其经验结果的推翻

#### 2.2 不能把 benchmark 论文写成“我们重新发明了 baseline 表”

如果只是把几个方法分数、NFE、latency 摆在一起，审稿人会觉得这只是整理结果，不是论文。

所以必须避免这个陷阱：

- 不要把 honest-compute benchmark 写成“更完整表格”

必须把它上升为机制命题：

- 排序为什么会变
- observer/controller 为什么错位
- reasoning 与 code 为什么处在不同 response regime

没有这三层解释，外部定位就会塌回“benchmarking but shallow”。

#### 2.3 不能偷换比较口径

如果我们一边批评别人只报 nominal step，一边自己在某些地方又只报 aggregate accuracy，不把 latency / TPS / batch size 一起报，就会被抓住双标。

因此这条线要稳，必须在全文保持同一标准：

- method comparison 一律报 `actual NFE + latency + throughput + batch size`
- code comparison 一律拆 `syntax / runtime / semantic`
- benchmark transfer 一律讨论“排序是否迁移”，而不是只看某个单点提升

#### 2.4 不能暗示“我们已经覆盖全部 literature”

revision literature 现在发展很快，尤其是 2025-2026 的 concurrent work。  
如果我们口气太大，说自己定义了整个领域的统一评价标准，也容易显得 overstated。

更稳妥的定位是：

- 我们针对 **training-free DLM revision/search** 这个狭义子问题提出一个更诚实的比较框架
- 这是对当前一类论文习惯的纠偏，不是对所有 decoding literature 的总规范

### 3. Round 2 的最终判断

我的判断是：

**主线改成 honest-compute diagnostic benchmark 后，和 `CoRe` / joint search / 一般 revision literature 的关系会更稳，而且这是当前最有机会存活的定位。**

但前提是我们必须守住三条边界：

1. 不把自己伪装成新的 strongest method paper  
2. 不把 benchmark framing 降级成“结果汇总表”  
3. 不对未直接复现的 concurrent work 做过强因果判断

如果这三条守住，`cand_diag` 的外部定位就会从：

- “一个没有赢下 method battle 的方案”

转成：

- “一篇告诉领域应该如何更诚实比较 DLM revision/search 方法的论文”

这条线比继续硬保 `TIGER` 主叙事稳得多。

## Round 3

### 1) 最稳的外部定位

最稳的定位是：这不是一篇“新 decoder 打赢基线”的论文，而是一篇针对 training-free DLM revision/search 的 honest-compute diagnostic benchmark 论文。它的价值在于纠正比较口径，并解释 revision gain 为什么会随任务和 failure mode 改变。

### 2) 最危险的对标陷阱

最危险的陷阱是拿自己去和 `CoRe`、joint search 这类 method-forward 工作正面比“谁更强”。我们现在没有那个 method margin，一旦这样写，整篇论文会立刻落到下风。

### 3) 你建议主动回避的叙事

我建议主动回避三种叙事：`TIGER 是主方法贡献`、`我们接近 benchmark SOTA`、`我们已经证明 concurrent work 的收益主要来自 compute 偏差`。这些说法都超出了现有证据，风险很高。
