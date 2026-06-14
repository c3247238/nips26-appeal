# Innovator Round 1 Summary

## 总裁决

这一轮其实没有出现 5 条彼此竞争的新路。真实情况是：  
大家围绕同一个主轴在不同层面施压。

因此我不建议把 5 份稿件当成 5 个并列候选，而建议直接收缩成 **2-3 个 serious candidates**。

## 我建议保留的 candidate pool

### Candidate 1：Observer-Controller Split as the Main Question

状态：KEEP

这是最值得保留的主线，因为它不是“补材料”，而是在改写论文真正研究的问题：

- observer 看到错误，不等于 controller 能把错误修回来
- diagnostic quality，不等于 intervention quality
- nominal budget，不等于 realized gain

这条线主要吸收：

- `theoretical`
- `innovator`
- `contrarian` 的边界约束

如果要留下一个最像“论文核心 claim”的 candidate，就是它。

### Candidate 2：Benefit-Bucket / Recoverability Analysis

状态：KEEP

这是最强的机制证据候选，也是当前最不能再拖的硬缺口：

- fixed
- harmed
- no-effect
- shallow vs deep failure

它的价值不只是补洞，而是把论文从 aggregate gain 拉到 mechanism-aware diagnostic。  
这条线主要吸收：

- `empiricist`
- `pragmatist`
- `innovator`

如果不做它，所谓 diagnostic paper 还是会显得空。

### Candidate 3：Runtime-Lineage / Honest-Compute Protocol

状态：MERGE

我支持保留，但不支持把它单独当成唯一主故事。  
更准确的定位应该是：

- 作为 Candidate 1 的 protocol backbone
- 作为 Candidate 2 的 credibility shield

也就是说，它重要，但更像**护城河贡献**，不是唯一主角。  
这条线主要吸收：

- `pragmatist`
- `empiricist`
- `contrarian`

## 我明确反对的方向

### 反对 1：任何 generic new controller 叙事

不值得进 serious pool。原因很简单：

- 会把项目重新拖回拥挤赛道
- 和当前最独特的证据不对齐
- 极易重新引入 fairness confound
- ROI 最低

所以 `Minimal Controller for Decoupling` 只能保留为**小型 probe / add-on**，不能升格成主候选。

### 反对 2：把 protocol paper 写成“我们只是更诚实”

这条写法也不够。  
如果只有 runtime metadata、asset manifest、appendix discipline，而没有更强的主问题和 bucket 证据，那论文会显得正确但不锋利。

### 反对 3：跨学科 framing 单独成候选

测量学、医学 responder bucket、认知科学类比都可以保留为写作语言资源，  
但它们不是 serious candidate，本身不构成研究方向。

## 最终建议

我建议下一轮 serious candidates 只保留下面这组：

1. `Observer-Controller Split`
2. `Benefit-Bucket / Recoverability Analysis`
3. `Runtime-Lineage Protocol` 作为并入式第三候选

更强的说法是：

- 前两者是主问题与主证据
- 第三者是可信度与公平性护城河

## 一句话判断

真正值得进入下一轮的，不是“一个新 controller”，而是这套组合：

**用 observer/controller split 提问题，用 benefit buckets 给机制证据，用 runtime-lineage protocol 保证结论不会被实现条件冲掉。**
