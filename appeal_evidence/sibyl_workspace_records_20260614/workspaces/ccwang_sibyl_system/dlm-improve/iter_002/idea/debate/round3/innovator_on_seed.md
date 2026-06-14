# Innovator on Proposal Seed Round 3

## 1. 这份提案是否已经足够锋利

我的判断是：**已经接近足够锋利，但还差最后一次收口。**

现在这份 seed 的优点很明确：

1. 已经彻底摆脱了 generic DLM improvement / TIGER hero / Calibration-Aware 旧叙事。
2. 已经形成了清楚的层级：
   - 主问题
   - 主证据
   - 协议护城河
3. 已经不再把 `Minimal Controller` 当 serious candidate，这一点是对的。

但它还不够锋利的地方也同样明确：

1. `serious execution candidates` 里写成了 `Benefit-Bucket + Runtime-Lineage`，会让人误读成：
   - bucket 是内容
   - protocol 是并列内容
   - observer/controller split 只是背景句
2. 这会削弱最有辨识度的标题级命题，也就是：
   - **observer quality 与 controller gain 不等价**
3. 换句话说，现在这份 seed 在执行层是清楚的，在**标题层与 abstract 层还不够尖**。

所以我的结论是：

- **方向已经对**
- **结构已经基本对**
- **但最后还要把“主命题”重新抬回最前面**

## 2. abstract / contribution bullets 应保留哪 2-3 条

我建议 abstract / contribution bullets 最多保留 **3 条**，不要再多。

### 必保留 1：Observer-Controller Split

这是最该保留的标题级命题，也是 abstract 第一条。

推荐表述强度：

- 在当前 tested policies 与 compute-aligned setting 下，`observer quality` 不自动转化为 `controller gain`
- 因此，training-free DLM revision 不能只按 aggregate gain 报告

这是整篇 paper 最像“我们真正发现了什么”的一句话。

### 必保留 2：Benefit-Bucket / Recoverability Evidence

这是 abstract 第二条，也是最强证据层。

推荐表述强度：

- aggregate revision gain 必须被分解为 `fixed / harmed / no-effect`
- 否则机制解释会失真

注意这里要坚持：

- 写成 **recoverability evidence**
- 不写成 **full failure taxonomy**
- 不写成 **跨任务 regime law**

### 可保留 3：Realized Compute / Runtime-Lineage Fairness

这是第三条，可以放在 abstract 后半段或 contribution list 最后一条。

推荐表述强度：

- 上述结论必须在 realized compute fairness 与 auditable runtime-lineage 下报告

这条很重要，但它不该抢前两条的锋利度。  
它更像“为什么我们的前两条可信”，而不是“第三个并列科学发现”。

## 3. 哪些措辞最该再 sharpen

### 需要 sharpen 1：`compute-normalized diagnostic / protocol paper`

这个措辞目前是对的，但还不够利。  
它容易让人感觉这是一篇“做 protocol 的论文”，而不是“提出一个更强 diagnostic claim 的论文”。

更好的用法应该是：

- 主句强调：`observer quality != controller gain`
- 然后补充：我们用 `compute-normalized diagnostic protocol` 去验证它

也就是说，`protocol` 是方法论身份，不是最前面的 headline。

### 需要 sharpen 2：`Observer-Controller Split` 只是 framing / falsification-style claim

“framing”这个词可以内部用，但对外不够强。  
建议 sharpen 成：

- **主命题**
- **标题级 claim**
- **可证伪发现**

因为如果继续叫 framing，reviewer 会自然把它降到 discussion。

### 需要 sharpen 3：`Benefit-Bucket / Recoverability Analysis`

这个说法已经不错，但还可以更明确一点：

- 它不是普通 error analysis
- 它是**主证据层**

所以写作时最好把它写成：

- `recoverability decomposition`
- 或 `bucket-level mechanism evidence`

而不是平淡的 `analysis`

### 需要 sharpen 4：`Runtime-Lineage / Honest-Compute Protocol`

这个说法要继续降温，避免它抢主线。

最该 sharpen 的不是把它写得更大，而是把它写得更准：

- 它是 `credibility shield`
- 它是 `auditable protocol layer`
- 它不是标题级并列发现

### 需要 sharpen 5：`serious execution candidates`

这一段当前最容易引起结构误读。  
如果保留这个标题，我建议改成：

- `execution priorities`
- 或 `artifact priorities`

因为现在真正的 serious candidate 不是两条执行项，而是一条主命题加一条主证据。

## 4. 最终推荐结构

我建议最终 proposal / abstract / intro 全部按下面这个结构收口：

### 第一层：标题级主命题

**在 training-free DLM revision 中，observer quality 与 controller gain 并不等价。**

### 第二层：主证据层

我们用 `fixed / harmed / no-effect` 的 bucket-level recoverability decomposition 来说明：

- revision 修了谁
- 伤了谁
- 对谁无效

### 第三层：可信度条件

这些结论只在下面的报告纪律下才成立：

- realized compute fairness
- runtime-lineage auditability
- canonical asset mapping

### 第四层：planning / execution 落地

执行优先级应写成：

1. `benefit_bucket_audit`
2. `seed_sensitivity_spotcheck`
3. `canonical_asset_manifest + runtime_fairness_matrix`

注意：

- 这里不需要再把 `Observer-Controller Split` 写成执行候选
- 它应该出现在标题、摘要、引言、discussion
- bucket 和 protocol 则负责把它撑起来

## 最终裁决

这份提案已经足够接近定稿版本。  
最后一刀不是再改方向，而是做一个更锋利的前后分层：

- **前面更尖**：把 `observer quality != controller gain` 明确抬成标题级命题
- **中间更实**：把 `benefit buckets` 明确写成主证据层
- **后面更准**：把 `runtime-lineage` 明确写成可信度条件，而不是并列主发现

一句话说：

**这份 seed 已经不需要再找新 idea，只需要把“主命题、主证据、可信度条件”三层关系写得更绝对、更前后一致。**
