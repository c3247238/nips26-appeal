# Revisionist Round 1

## 核心判断

这一轮数据要求我们做的，不是“为 revision 再找一个更聪明的控制律”，而是**彻底修正我们对 revision 问题本身的理解**。旧心智模型默认：

1. 更强的观测信号应该导向更强的控制收益
2. revision gain 主要是方法设计问题
3. code failure 可以通过轻量 guard 延展成通用修复故事

这三条现在都必须降级或删除。更贴近数据的新心智模型是：

**DLM revision 的关键矛盾不是“信号够不够强”，而是“任务是否可恢复、compute 是否诚实、错误是否属于浅层可修补失效”。**

## 1. 假设审计

### H1: Honest Compute Reorders the Ranking

- 判定：**确认**
- 证据：
  - `diag_compute_curve_gsm8k.json` 明确给出 `qualitative_ranking_change = true`
  - `CORE-proxy-64` 的 `compute_gap_pct = 7.81%`
  - 在 nominal 排名里，`CORE-proxy-64` 在 `Entropy-Revise-64+3` 前；在 actual compute 排名里反而落后
- 修正：
  - 原假设里“复杂 revision 方法的优势会缩小”只说对了一半
  - 真正更重要的不是 revision 是否缩小，而是**headline ranking 本身会因为 honest compute accounting 被重写**

### H2: Revision Benefit Depends on Task Recoverability

- 判定：**部分确认，但需要改写**
- 证据：
  - `diag_math500_shortlist.json` 显示 MATH500 排名与 GSM8K 不同，说明 reasoning benchmark 内部就已经不是单一响应曲线
  - `diag_humaneval_guard_boundary.json` 显示 code 上 `Gated TIGER` 虽然降低 syntax failure `-0.20`，但 `pass@1` 仍低于 `Standard`，且 runtime failure 反而 `+0.18`
- 修正：
  - 原表述把问题写成“reasoning vs code”
  - 数据更支持**recoverability spectrum**，而不是粗粒度任务二分
  - MATH500 说明 reasoning 内部也存在不同恢复结构，不能把 reasoning 看成单一“revision 可帮助”的桶

### H3: Observers Are More Useful Than Controllers

- 判定：**强确认**
- 证据：
  - `diag_signal_gap_audit.json` 中 calibration 的 `diagnostic_score = 0.6225`，但 `control_effectiveness = 0.0`
  - entropy 的 `diagnostic_score = 0.414`，但相对 random revision 的 `control_effectiveness = 0.0`
  - instability 的 `control_effectiveness = 0.01`，而 signal-error 相关性本身也很弱
- 修正：
  - 旧假设仍然太温和，只说“observer 比 controller 更有用”
  - 新表述应该更锋利：**高诊断价值与高干预价值在 DLM 中是可系统性解耦的**

### H4: Syntax Guard Fixes Shallow Errors, Not Deep Ones

- 判定：**确认**
- 证据：
  - `diag_humaneval_guard_boundary.json` 中 `Gated TIGER` 的 syntax failure 从 `0.48` 降到 `0.28`
  - 但 runtime failure 从 `0.50` 升到 `0.68`
  - `pass@1 = 0.04`，仍低于 `Standard = 0.06`
- 修正：
  - 这里不该再写成 “guard can help code a bit”
  - 更准确的命题是：**guard 只能修浅层合法性，不能修深层执行结构**

## 2. 真正意外的结果

### 意外 1：最强方法并不是我们最初想讲的方法

`diag_compute_curve_gsm8k.json` 里，`CORE-proxy-64` 仍然有最高 accuracy `0.46`，但 latency 极高。这说明问题不在于“我们还没把 TIGER 调好”，而在于：

- 真实 Pareto 前沿不是单一 accuracy 排名
- 方法命名中的 step 数并不能代表真实 compute 位置
- 我们之前默认“更精细的 revision 机制会比 proxy control 更优雅且更值钱”这一点没有被支持

### 意外 2：reasoning 内部就已经不稳定

`diag_math500_shortlist.json` 最重要的不是哪个方法赢，而是**GSM8K 的排序不能直接外推到第二个 reasoning benchmark**。这打掉了一个隐含前提：

- 我们原来把 reasoning 看成相对统一的任务族
- 现在更像是不同 benchmark 对 revision 的响应结构不同

这意味着主线不该叫“reasoning 上 revision 有帮助”，而该叫：

**revision response is benchmark- and structure-dependent even within reasoning.**

### 意外 3：最强 observer 几乎没有控制收益

如果 calibration / entropy 只是“弱信号”，那我们可以继续找更强信号。但 `diag_signal_gap_audit.json` 说明不是这么回事：

- calibration 很强，但没有 deployed control gain
- entropy 有中等相关性，但对 random revise 没额外收益

这迫使我们更新理论：**瓶颈不只是 signal quality，而是 actionability mismatch。**

## 3. 心智模型更新

建议把旧模型：

`better signal -> better revision policy -> better task accuracy`

改成新模型：

`observer quality` 和 `controller usefulness` 之间存在一个被任务结构、错误类型与可恢复性共同调节的映射断层

更具体地说：

1. **Compute layer**
   - 先问比较是否 honest
   - 若 compute accounting 不诚实，方法排序本身就不可信

2. **Observer layer**
   - entropy / calibration / instability 可以是好诊断器
   - 但它们只回答“哪里危险”，不自动回答“怎么改才有益”

3. **Controller layer**
   - revision 是 intervention，不是 observation 的自动延伸
   - 同一个高风险点，改动后可能修复，也可能破坏原本局部正确的结构

4. **Failure-structure layer**
   - syntax failure 属于浅层错误
   - runtime / semantic failure 属于深层结构错误
   - reasoning benchmark 内部也存在不同结构恢复力

## 4. 更好的研究重框架

原 framing：

- “我们能否找到一个更强的 training-free revision policy？”

建议改成：

- “在 DLM 中，revision 何时是有效控制，何时只是高风险干预？”
- “为什么 honest compute、recoverability 与 failure depth 比控制信号本身更决定最终收益？”

也就是说，主文不该再以“方法发明”开场，而应该以：

1. `honest compute`
2. `observer-controller gap`
3. `recoverability / failure-depth taxonomy`

这三层逻辑来组织。

## 5. 应该删掉的旧假设

以下旧假设或暗含前提应从主线里删除：

1. **“更强信号自然导向更强控制”**
   - 已被 calibration / entropy gap 直接否定

2. **“reasoning 是 revision gain 的统一正例”**
   - 已被 MATH500 与 GSM8K 的排序差异削弱

3. **“code guard 是方法泛化成功的萌芽”**
   - 已被 HumanEval 的 runtime / pass@1 结果否定

4. **“TIGER 仍有希望通过更多 benchmark 变成主方法”**
   - 这轮诊断结果已经表明，把它继续放在前台只会拖累叙事诚实度

## 6. 应该进入写作主线的新假设

### 新假设 A

**actual compute mismatch 会系统性扭曲 training-free DLM 方法的 headline ranking。**

### 新假设 B

**observer value 与 controller value 在 DLM 中是结构性解耦的。**

### 新假设 C

**revision gain 主要由样本可恢复性与错误深度决定，而不主要由 signal complexity 决定。**

### 新假设 D

**code-side shallow repair 与 end-to-end correctness improvement 之间存在稳定断层。**

## 7. 对写作主线的直接建议

如果后续进入论文主文，这一轮结果最值得保留的不是某个方法名，而是以下四个句子：

1. honest compute 会改变方法排序
2. 高诊断价值不等于高控制价值
3. revision 收益取决于 recoverability，而不是只取决于 signal 强弱
4. 浅层合法性修复不能替代深层结构恢复

## 8. Pivot 还是 iterate

- 结论：**不是继续 method-forward 迭代，而是继续 diagnostic-forward iterate**
- 具体建议：
  - 对 `cand_diag` 继续推进
  - 对 `cand_tiger` 停止前台扩展
  - 把下一轮所有动作都服务于 mechanism articulation，而不是再寻找“谁再多赢 1-2 个点”

## 一句话总结

这轮最重要的修正不是“换一个更强方法”，而是承认：**DLM revision 的核心问题是可恢复性与干预有效性的错配，而不是我们还没找到足够聪明的控制信号。**

## Round 2

基于其他角色的意见，尤其是 `skeptic`、`methodologist`、`strategist` 和 `comparativist` 的反馈，我认为上一轮修正还不够彻底。我们不只是要把旧命题“降级”，而是要把其中几条**彻底删除**，否则写作时它们会继续污染 message。

### 必须彻底删除的旧命题

1. **“我们接近找到一个更强的 revision 方法，只是还差一点工程调优。”**
   - 这个命题现在必须删除，而不是保留成含蓄希望。
   - `optimist` 也承认 strongest story 不是新方法胜利；`comparativist` 明确指出 method-forward 叙事会被 `CoRe` 一类 concurrent work 正面对撞；`skeptic` 则指出当前 small-sample 结果根本不足以支撑 method superiority。
   - 因此，主文里不能再给读者留下 “TIGER 只差一点就能翻盘” 的印象。

2. **“reasoning 是一个统一的 revision gain 场景。”**
   - 这个命题也必须删除。
   - `diag_math500_shortlist.json` 已经说明 GSM8K 与 MATH500 的排序不一致，而 `methodologist` 进一步指出这更适合被写成 diagnostic slice，而不是统一规律。
   - 所以我们不能再写“revision 对 reasoning 有帮助”，只能写“revision response 在 reasoning benchmark 内部也会变化”。

3. **“更好的 observer 最终应该能导出更好的 controller。”**
   - 这条不只是被削弱，而是应当从理论默认项里移除。
   - `diag_signal_gap_audit.json` 已经反复说明 calibration / entropy 的诊断价值和控制收益不共变，`skeptic` 也提醒这更像 actionability mismatch，而不是简单控制器未调好。
   - 以后写作不能再把 controller 失败说成“暂时没做好 mapping”，而应把错位本身视为需要解释的对象。

4. **“code guard 的积极结果可以作为方法泛化成功的前奏。”**
   - 这条必须彻底删除。
   - `diag_humaneval_guard_boundary.json` 给出的结构非常清楚：syntax 修复显著，但 runtime / pass@1 没有恢复。
   - `strategist` 和 `methodologist` 都一致认为这只能服务于 appendix-only boundary claim，不能再被包装成主线成功信号。

### 可以升级成写作核心句的新命题

下面这 4 句是目前最值得升级为论文核心句的内容。它们比上一轮更短，也更硬。

1. **Honest compute changes the story.**
   - 同一批 DLM decoding 方法在 nominal steps 下的排序，和在 actual compute 下的排序并不相同。

2. **Good observers are not automatically good controllers.**
   - calibration / entropy / instability 可以可靠地标记风险，但不能因此被直接视为有效干预律。

3. **Revision gain is governed by recoverability, not by signal complexity alone.**
   - 决定 revision 是否有益的关键不是信号看起来多聪明，而是样本结构是否可恢复、错误是否会被局部改动修复。

4. **Shallow repair does not imply deep recovery.**
   - syntax guard 可以减少表层非法输出，但不能自动恢复 runtime 或 semantic correctness。

### 进一步压缩后的最小 message

如果论文 message 必须压到最少，我建议主文只保留下面 3 句，第四句放成结果段或副标题：

1. **在 DLM revision 里，actual compute 而不是 nominal steps 才决定公平排序。**
2. **高诊断价值不等于高控制价值，observer 和 controller 必须分开建模。**
3. **revision 的净收益由可恢复性决定，而不是由控制信号复杂度决定。**

可选的第 4 句：

4. **浅层合法性修复不能代表深层任务恢复。**

### 对写作结构的最终修正

如果按照这些句子反推论文结构，那么标题、摘要和 contribution list 都应避免方法导向措辞，改成：

1. 先讲 honest compute
2. 再讲 observer-controller gap
3. 再讲 recoverability / shallow-vs-deep failure

也就是说，论文不该回答“哪个 revision policy 最强”，而该回答：

**为什么 DLM revision 文献当前对收益、代价和可控性的理解是混在一起的，以及我们如何把它们拆开。**

## Round 3

### 1) 必须彻底删除的旧命题

不要再写“我们快找到更强 revision 方法了”以及“reasoning 是统一正例”。也不要再暗示“更强 observer 终将导出更强 controller”或“code guard 是泛化成功前兆”。

### 2) 应升级成写作核心句的新命题

应升级的核心句只有三条：honest compute 会改变 DLM 方法排序；good observers are not automatically good controllers；revision gain 取决于 recoverability，而不是 signal complexity 本身。code 侧补一句：浅层修复不等于深层恢复。

### 3) 最终 message 压缩版

这篇论文不该再回答“哪个 revision policy 最强”，而该回答：在 DLM 中，公平比较首先取决于 honest compute，而 revision 是否有效取决于可恢复性与 observer-controller 的错位。更短一点：**honest compute 改写排序，recoverability 决定 revision 是否真的有用。**
