# theoretical round1 summary

## 总判断

从 theoretical 视角看，这一轮最该保留的不是某个“新 controller 候选”，而是一个 **diagnostic-theory candidate pool**：它必须同时满足三条条件：

1. 有清晰研究对象，而不是泛泛叙事
2. 有可证伪命题，而不是只说“这很重要”
3. 能直接用现有资产验证，而不是再开一条方法线

据此，我建议保留下面 3 个 candidate pool。

## 建议保留的 candidate pool

### Pool 1：Observer / Controller Split 作为主协议对象

- 来源整合：
  - theoretical 的候选 A
  - innovator 的三层拆分主张
  - contrarian 对 method drift 的边界约束
- 为什么保留：
  - 这是当前最清晰的理论对象定义
  - 它直接回答“为什么好的 signal 不自动意味着好的方法”
  - 还能把后续 bucket 和 fairness 都组织到同一框架下
- 理论价值：
  - 高
  - 因为它定义了两个不能混报的量：diagnostic quality 与 realized control gain
- 需要警惕：
  - 一旦引入“最小 controller”而没写清只是辨识工具，就会滑回 generic method paper

### Pool 2：Signal vs Intervention Mismatch + Bucket Mechanism

- 来源整合：
  - theoretical 的候选 B（扩展后）
  - empiricist / pragmatist 对 bucket audit 的强调
  - contrarian 对 boundary-sensitive bucket 的要求
- 为什么保留：
  - 它把 bucket 从“结果拆分类别”提升为“机制分解对象”
  - 能解释为什么 entropy 对 error 有信息量，但 calibration 不一定转化为 gain
  - 能自然容纳 fixed / harmed / no-effect 以及 reasoning / code boundary
- 理论价值：
  - 高
  - 因为它对应的是一个真正可证伪的命题：error signal 与 intervention value 不等价
- 需要警惕：
  - 如果 bucket 最后只停留在三分类计数，没有连到净收益、伤害风险和结构边界，那就会退化成浅层 error analysis

### Pool 3：Realized Compute Fairness 作为 metareasoning 预算问题

- 来源整合：
  - theoretical 的候选 C
  - innovator 对 runtime stack 的强调
  - empiricist / pragmatist 对 runtime lineage 和 manifest 的需求
  - interdisciplinary 的 “realized protocol rather than nominal budget” 表述
- 为什么保留：
  - 它能把 honest compute 从 appendix hygiene 提升为理论上必须的比较条件
  - 它解释为什么 nominal steps 或 actual NFE 仍不足以定义真正可比的 inference cost
- 理论价值：
  - 中高
  - 它更像外层评价原则，不是最核心机制命题，但对整篇 paper 的可信度非常关键
- 需要警惕：
  - 如果只写成“我们要更诚实地报告 latency / batch size / backend”，那就只是 protocol hygiene，不够成研究贡献

## 哪些命题有理论价值

下面这些我认为有明确理论价值：

1. **observer quality 与 controller gain 必须分开定义、分开报告**
2. **error likelihood 不等于 intervention value**
3. **aggregate gain 必须分解到 fixed / harmed / no-effect 机制桶才能解释**
4. **inference-time gain 必须在 realized compute 下解释，而不是只在 nominal budget 下解释**

它们共同特征是：

- 有清晰对象
- 有可检验预测
- 与现有资产直接对接

## 哪些更像 prose 包装

下面这些目前主要还是表达增强，不应单独当主贡献：

1. **measurement discipline**
   - 如果没有落到 observer/controller split 的形式定义，只是漂亮措辞
2. **response buckets**
   - 如果没有净收益分解与 boundary linkage，只是分类叙述
3. **realized protocol rather than nominal budget**
   - 如果没有 runtime-lineage / cost accounting，只是 fairness 口号
4. **最小 controller**
   - 如果没有严格限定为辨识实验，而被写成“一个轻量新方法”，就会直接退回 generic method paper

## 对其余 5 份稿件的总体判断

- contrarian：
  - 价值在于守边界
  - 不足在于批评强、正向理论对象定义还不够
- empiricist：
  - 价值在于指出证据短板
  - 不足在于更像实验计划，不像理论主张
- innovator：
  - 价值最高，已经接近可用的主叙事
  - 但必须更严厉地切断 method drift
- interdisciplinary：
  - 价值在于类比帮助理解
  - 不足在于容易停在修辞层
- pragmatist：
  - 价值在于执行顺序清楚
  - 不足在于 ROI 不能替代理论区分度

## 最终建议

本轮不应形成 “A/B/C 三个平行 idea 再投票” 的结构，而应收束为一个主线、两个支撑层：

1. 主线：
   - **Observer / Controller Split**
2. 机制层：
   - **Signal vs Intervention Mismatch + Bucket Mechanism**
3. 评价层：
   - **Realized Compute Fairness**

这样写，最能避免退回 generic method paper，也最能把现有资产转化成真正有理论含量的 diagnostic 论文。
