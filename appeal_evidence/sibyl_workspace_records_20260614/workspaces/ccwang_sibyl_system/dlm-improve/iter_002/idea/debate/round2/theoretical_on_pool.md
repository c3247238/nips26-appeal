# theoretical on pool

## 总判断

从 theoretical 视角看，round2 的核心不是“三选一”，而是把三个候选放到不同层级：

- **真正的研究对象**：必须能提出一个可证伪、可被现有资产直接验证的命题
- **supporting mechanism evidence**：能解释主命题，但自身不宜单独 headline
- **protocol / credibility scaffold**：决定结论是否可信，但通常不该独立充当最锋利的研究 claim

按这个标准，目前三个候选并不对称。

---

## 1) 三个候选里，谁最像真正研究对象

### 最像真正研究对象的是 Candidate A：Observer-Controller Split

原因不是它最容易做，而是它最像一个真正的 research object：

1. 它定义了一个明确的对象区分：
   - `observer quality`
   - `controller gain`
2. 它对应一个明确可证伪命题：
   - `good signal != guaranteed good intervention`
3. 它可以被现有资产直接检验：
   - signal audit
   - bucket audit
   - matched policy 下的 outcome decomposition

更重要的是，A 不是单纯“写法升级”，而是在重新规定这篇 paper 到底在研究什么。  
如果没有 A，B 会沦为一组经验分类，C 会沦为一组 fairness hygiene；只有 A 先成立，B 和 C 才知道自己在服务什么主问题。

### Candidate B 更像机制证据，不是顶层研究对象

B 很重要，而且可能是最急需补齐的证据件，但它本身更像：

- 对 A 的机制化支撑
- 对 aggregate gain 的分解工具
- 对“谁受益、谁受害”的实证回答

它非常关键，但它回答的是 **“主命题如何被看见”**，而不是 **“主命题本身是什么”**。

### Candidate C 更像评价原则，不是最中心的科学问题

C 的地位也很高，但它更像：

- comparison contract
- reviewer-facing credibility shield
- protocol backbone

它解决的是“结论能不能被相信”，而不是“现象本身的核心机制是什么”。  
所以 C 很重要，但作为 headline claim 会偏外层。

---

## 2) 哪些只能作为 support、不能做 headline claim

### Candidate B 不能单独做 headline claim

B 不能单独 headline，不是因为它不重要，而是因为：

- “fixed / harmed / no-effect” 本身只是分解语法
- 如果没有更高层命题，它容易被 reviewer 理解为更细的 error analysis
- 一旦写成 “failure taxonomy” 或 “recoverability law”，就很容易 evidence outrun

因此 B 最合适的定位是：

> **mechanism evidence for the observer/intervention mismatch claim**

可以很强，但不宜单独变成标题最前面的研究问题。

### Candidate C 也不宜单独做 headline claim

C 如果单独 headline，最容易滑成：

- “我们更诚实地报告 compute”
- “我们做了一个更规范的 runtime fairness appendix”

这类内容当然正确，但如果缺少 A 这样的主问题，它会显得“对，但不锋利”。  
所以 C 更适合作为：

> **protocol contribution / credibility infrastructure**

它能保护整篇 paper，但不宜独自充当 paper 的最强前台叙事。

### 只有 A 可以做 headline claim，但必须降承诺

A 可以 headline，但不能写成太满的理论宣言。  
最稳的写法应该是：

> 在当前 tested policies 和 current evidence 下，observer quality 与 controller gain 不应被默认等同，且需要分开定义与报告。

而不是：

> 我们已经建立了所有 DLM revision 的一般理论。

前者可守，后者会 evidence outrun。

---

## 3) 是否应压缩到 2 个 serious candidates

### 是，应该压缩到 2 个 serious candidates

我建议 serious candidates 压缩为：

1. **Candidate A：Observer-Controller Split**
2. **Candidate B：Benefit-Bucket / Recoverability Analysis**

### Candidate C 不应再作为并列 serious candidate

不是因为 C 不做，而是因为它更适合“并入式存在”：

- 并入 A，作为其 protocol backbone
- 并入 B，作为其 credibility shield

换句话说：

- A = 主问题
- B = 主证据
- C = 可信度条件

这比三条并列主线更稳，也更不容易让 paper 看起来像拼盘。

### 为什么不是保留 A+C、降级 B

因为如果没有 B，A 会太像 formal claim，C 会太像 protocol discipline。  
二者都缺一个真正“碰到样本层机制”的中层证据。  
B 正好承担这个角色，因此不能降。

---

## 4) 最终主张应如何避免 evidence outrun

这是本轮最关键的问题。我的建议是把所有主张都压到 **scoped, conditional, asset-backed** 三个层次。

### 原则一：把 A 写成 scoped falsification claim，不写成普适理论

推荐写法：

- 在当前 tested DLM revision setup 中，observer quality 与 controller gain 不应默认等价
- calibration 改善 signal 并不自动带来更高 revision utility

不推荐写法：

- 我们提出了 DLM revision 的统一理论
- 我们证明了 observer 与 controller 在一般情形下必然分离

前者是 evidence-backed，后者会立刻 outrun。

### 原则二：把 B 写成 mechanism evidence，不写成完整 taxonomy

推荐写法：

- 我们通过 `fixed / harmed / no-effect` 机制桶说明 aggregate gain 的来源结构
- 我们在 reasoning / code 边界上观察到不同的 recoverability pattern

不推荐写法：

- 我们建立了完整 failure taxonomy
- 我们发现了跨任务稳定的 revision regime law

因为当前 cross-task evidence 仍然明显不够厚。

### 原则三：把 C 写成 necessity-of-evaluation claim，不写成“额外贡献秀肌肉”

推荐写法：

- realized compute fairness 是解释 inference-time gain 的必要条件
- runtime lineage 必须被显式记录，否则 method-vs-implementation 无法区分

不推荐写法：

- 我们提出了一套全新的 benchmark standard
- 我们解决了 DLM inference comparison 的普遍公平性问题

当前资产可以支持 “necessary protocol discipline”，还支撑不了 “new universal standard setter”。

### 原则四：所有 headline 句子都必须能回指到具体 artifact

也就是说，最终主张必须满足：

- A 能回指到 `observer_controller_protocol` / signal audit / matched comparison
- B 能回指到 `benefit_bucket_audit` / examples / boundary slices
- C 能回指到 `runtime_fairness_matrix` / manifest / lineage table

如果一句 headline claim 找不到直接 artifact，就不要进摘要。

---

## 最终建议的收束方式

如果要我从 theoretical 视角给出最稳的 round2 结论，我会这样收束：

### serious candidate 只保留 2 个

1. **Observer-Controller Split**
   - 作为真正研究对象
   - 但写成低承诺、可证伪的主命题
2. **Benefit-Bucket / Recoverability Analysis**
   - 作为最核心机制证据
   - 但不膨胀成 full taxonomy

### Runtime-Lineage / Honest-Compute Protocol 的定位

- 必做
- 很重要
- 但不再单列为并列主线
- 应该被吸收为 A 和 B 的必要评价条件

---

## 一句话结论

最稳的写法不是 “A/B/C 三线并举”，而是：

> **用 A 定义真正研究对象，用 B 提供机制证据，用 C 防止结论被实现条件冲掉。**

这样既不会回到 generic method paper，也最能避免 evidence outrun。
