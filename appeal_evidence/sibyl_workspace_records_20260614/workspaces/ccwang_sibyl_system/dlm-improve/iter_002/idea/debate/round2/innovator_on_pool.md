# Innovator on Candidate Pool Round 2

## 总裁决

round1 之后，candidate pool 其实已经不适合再按“三个并列 idea”来讨论了。  
更准确的结构应该是：

- 一个**主线命题**
- 一个**主证据层**
- 一个**支撑协议层**

如果继续把 A/B/C 写成三个平行 serious candidates，讨论会越来越虚，因为它们本来就在服务同一篇 paper，而不是三篇不同论文。

## 1. A/B/C 谁是主线，谁是支撑层

### A: Observer-Controller Split

定位：**主线**

原因：

1. 它定义了论文真正研究的问题，而不是只定义一个补件任务。
2. 它最能把我们和 generic DLM improvement、generic scheduler paper 拉开。
3. 它能自然统摄：
   - 为什么 bucket 必须做
   - 为什么 runtime fairness 不能只是 appendix hygiene

但要注意：  
A 只能写成 **低承诺、可证伪的主命题**，不能写成过满的理论总规律。

最安全的版本是：

- 好 observer 不自动变成好 controller
- diagnostic quality 与 realized control gain 必须分开定义、分开报告

### B: Benefit-Bucket / Recoverability Analysis

定位：**主证据层**

原因：

1. B 是最强的机制证据来源，没有它，A 会像一句聪明的话。
2. B 直接把 aggregate gain 打散成：
   - fixed
   - harmed
   - no-effect
3. B 还能把 reasoning 与 code boundary 连接起来，让论文不只停在 GSM8K aggregate table。

因此，B 不该被降级成普通 error analysis。  
它应该是 **A 的主要证据引擎**。

### C: Runtime-Lineage / Honest-Compute Protocol

定位：**支撑层**

原因：

1. C 非常重要，但它的角色更像 credibility shield，而不是论文主问题。
2. 如果把 C 升格成平行主线，论文会显得像 protocol hygiene paper。
3. C 的最佳位置是：
   - 保护 A 不被说成 narrative selection
   - 保护 B 不被说成 implementation confound

结论很明确：

- **A = 主线**
- **B = 主证据层**
- **C = 支撑协议层**

## 2. 是否应把 serious candidates 从 3 个压到 2 个

我的答案是：**应该。**

但这里的“压到 2 个”不是说 C 不做，而是说：

- C 不再作为独立 serious candidate 存在
- C 变成 A+B 的协议底座和 reviewer-facing 护城河

也就是说，真正保留为 serious candidates 的只剩：

1. `A: Observer-Controller Split`
2. `B: Benefit-Bucket / Recoverability Analysis`

而 `C` 的状态应调整为：

- `supporting protocol layer`
- `mandatory artifact package`
- `non-headline contribution`

这是更健康的收缩方式，因为：

1. 它避免三个抽象概念互相抢标题位。
2. 它让论文结构更清楚：
   - 问题是什么
   - 证据是什么
   - 为什么这个证据可信
3. 它也最符合当前证据成熟度。

## 3. 标题级 / abstract 级最值得保留的命题

### 标题级最值得保留的命题

如果只保留一句最像标题级的命题，我建议是：

**在 training-free DLM revision 中，observer quality 与 controller gain 并不等价。**

这句话的优点是：

1. 足够新，不是 generic improvement 口吻。
2. 足够稳，因为我们已经有 signal-gap 证据。
3. 足够能展开，因为 bucket 与 runtime fairness 都能挂在这句话下面。

### abstract 级第二命题

**aggregate revision gain 必须被 fixed / harmed / no-effect 的 recoverability buckets 分解，否则机制解释会失真。**

这句话负责把 B 抬到 abstract 层，而不是留在 appendix。

### abstract 级第三命题

**这些结论必须在 realized compute 与 runtime-lineage 对齐后报告。**

注意，这一句我建议只保留在 abstract 后半段或 contribution 列表里，  
不要让它抢主标题，因为它更像条件约束而不是核心科学问题。

## 4. 明确 DROP 的方向

### DROP 1：把 C 当成独立 headline candidate

原因：

- 它重要，但不够像主问题
- 单独拔高会让论文变成“我们更诚实地报告 runtime”
- 这会削弱论文锋利度

### DROP 2：任何独立成线的 Minimal Controller / Decoupling Probe

原因：

- 一旦独立成线，就会重新滑回 method-adjacent paper
- ROI 低
- 会引入新的 fairness confound

如果以后真做，只能作为 appendix-level sanity probe。

### DROP 3：任何 generic new controller / scheduler / hero method 方向

这条必须继续明确排除：

- 不回到 `TIGER hero`
- 不重启 `Calibration-Aware`
- 不写“another stronger DLM controller”

### DROP 4：把 seed spot-check 当成独立 candidate

`seed_sensitivity_spotcheck` 必须做，但它不是 candidate。  
它是 quality gate artifact，应该并入 C，而不是单列成研究方向。

## 最终建议

round2 之后，我建议 candidate pool 正式压缩成下面这个结构：

### Serious Candidate 1

`Observer-Controller Split`

### Serious Candidate 2

`Benefit-Bucket / Recoverability Analysis`

### Supporting Layer

`Runtime-Lineage / Honest-Compute Protocol`

## 一句话裁决

这轮该做的不是在 A/B/C 里再投票，而是承认它们已经形成层级关系：  
**A 提问题，B 给证据，C 保可信度；因此 serious candidates 应从 3 个压到 2 个。**
