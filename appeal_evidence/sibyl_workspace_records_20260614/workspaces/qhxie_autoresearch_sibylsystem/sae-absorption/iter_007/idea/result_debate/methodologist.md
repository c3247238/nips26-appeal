# 方法论审查：实验方法的内外部效度与可复现性

## 角色

方法论者（Methodologist）：系统审查实验设计的内部效度（结论是否由数据支撑）、外部效度（结论是否可推广）、以及可复现性（他人能否重现这些结果）。

---

## 一、总体方法论评估

本研究在方法论层面表现出明显的二元性。一方面，控制实验设计（C1-C4）、多L0交叉验证、bootstrap置信区间等统计实践达到了高标准。另一方面，几个关键实验的操作化定义存在严重问题，导致核心声明的解释力受到根本性限制。综合评判：**方法论严谨度中上，但关键环节存在定义滑移和效度缺口**。

---

## 二、内部效度审查

### 2.1 最严重问题：Hedging 定义的宽严落差颠覆核心叙事

**问题描述**：论文的标题性发现"98.6%的吸收是hedging而非hierarchy-driven"基于一个极其宽松的操作化定义——任何在L0=176时不再是false negative的token都被归类为hedging。但tightened hedging实验揭示，严格定义（特定parent latent在L0=176时激活）下的hedging率仅为6.2%（41/656），与宽松定义的98.6%之间存在**92.4个百分点的差距**。

**方法论判定**：这不是统计误差，而是**构念效度（construct validity）失败**。宽松定义将两种截然不同的现象——"信息扩散到其他latent（真正的hedging）"和"信息根本未被parent-associated latent编码"——混为一谈。93.8%的FN token在L0=176时parent latent仍然不激活，这意味着问题不是"信息被hedge到了别处"，而可能是"这些parent latent在该SAE配置下根本不编码首字母信息"。

**对内部效度的影响**：宽松hedging率（98.6%）不能作为论文的标题性发现。论文必须以strict率（6.2%）为主要报告数值，并明确讨论宽严落差的含义。当前叙事需要根本性修正。

**建议**：(1) 将strict hedging率作为primary报告数值；(2) 对93.8%的non-hedging FN进行进一步分类——它们是"parent latent从未对该token编码首字母信息"还是"存在某种未被捕获的补偿机制"；(3) 检查这些non-hedging FN在L0=22和L0=176时的parent latent activation分布。

### 2.2 Activation Patching：0/8恢复的解释力受限

**实验设计评价**：三种patching方法（decode-reencode、residual subtraction、all-children zeroing）加上10个control features的设计是合理的。0/8恢复、0/65 control恢复，结果一致且稳健。

**但解释需要谨慎**：0/8恢复有两种可能解释：
- 解释A：不存在competitive exclusion，child feature没有抑制parent feature
- 解释B：SAE的decode-reencode过程引入了额外的非线性变换（JumpReLU门控），使得即使底层存在竞争排斥，patching也无法恢复parent activation

论文选择了解释A，但**未对解释B进行系统排除**。JumpReLU的硬阈值特性意味着，即使child zeroing增加了parent方向的投影，如果增量不足以超过threshold，parent仍不会激活。这是JumpReLU特有的问题，在L1-ReLU SAE上可能不存在。

**建议**：(1) 报告patching后parent方向的投影值（而非只看activation是否>0）——如果投影增加但未过阈值，这支持"阈值效应"而非"无竞争排斥"；(2) 在论文中明确讨论JumpReLU硬阈值对patching实验的影响。

### 2.3 控制失败诊断：结论成立但存在循环风险

**方法论强项**：在R^2304中1000个随机向量的分析清楚地展示了cosine>=0.025阈值下的candidate explosion（23%的decoder columns被识别为candidate）。P(至少1个candidate激活)=1.0的计算有力解释了为什么shuffled control高于measured absorption。

**循环风险**：控制失败诊断解释了metric为什么失败（高维空间中cosine阈值近乎无效），但这与absorption是否存在是两个不同的问题。论文存在将"metric失败"等同于"现象不存在"的滑移倾向。metric可能确实失败了，但absorption作为一种理论构念可能仍然存在——只是当前metric无法可靠检测。

**建议**：在讨论中明确区分三个层次：(1) metric效度（已证伪）；(2) 现象存在性（未决）；(3) 现象严重程度（无法评估）。

### 2.4 CMI诊断：统计效力不足且存在多重比较问题

| 分析 | rho | p值 | 样本量 | 判定 |
|------|-----|-----|--------|------|
| Raw CMI-absorption | -0.383 | 0.059 | 25 | 边缘 |
| Partial (控制F1) | -0.328 | 0.118 | 25 | 不显著 |
| Restricted (F1>0.85) | -0.113 | 0.757 | 10 | 无关联 |
| Bonferroni corrected | -0.383 | 0.236 | 25 | 不显著 |

**方法论判定**：
- N=25（26字母减去X）的样本量提供的统计效力严重不足。检测rho=-0.4的效应需要N>47（alpha=0.05, power=0.80）。当前设计的统计效力约为0.40，即有60%的概率即使效应真实存在也无法检测到。
- d'=10的选择未预注册。在4个d'值（10, 20, 30, 50）中，仅d'=10显示负相关，其余三个为正或接近零。这是典型的researcher degrees of freedom问题。
- restricted分析（F1>0.85, N=10）的rho=-0.113几乎为零，进一步削弱了CMI作为诊断工具的证据。

**建议**：(1) 明确报告post-hoc power analysis；(2) 将CMI结果定位为"hypothesis-generating"而非"partially supported"；(3) L0=22的CMI复制实验是正确的改进方向，但N=25的根本限制仍然存在。

### 2.5 阈值敏感性：稳健性验证充分

**方法论强项**：5x4参数网格（cosine: 0.01-0.05, magnitude gap: 0.5-2.0）全面覆盖了阈值空间。CV=0.077表明absorption rate对阈值选择高度稳健。Mean Kendall tau=0.977表明字母排名高度稳定。

这是本研究中方法论最扎实的部分。结论——"控制失败是结构性的，无法通过调整阈值解决"——有充分的数据支撑。

---

## 三、外部效度审查

### 3.1 模型泛化性：单一模型+单一SAE族

所有核心实验仅在Gemma 2 2B + Gemma Scope JumpReLU SAE上执行。GPT-2 Small的结果仅用于Section 5.3的架构比较，且该比较因模型规模、训练数据、SAE架构全部不同而被论文自身承认为"fully confounded"。

**效度限制**：
- 结论是否适用于其他JumpReLU SAE（如DeepMind的SAE）？
- 结论是否适用于其他模型规模（7B, 13B）？
- 结论是否适用于不同层（仅测试了L10, L12, L20）？
- 控制失败是JumpReLU特有的还是所有SAE共有的？

**缓解因素**：跨层稳定性（L10/L12/L20的CV<10%）部分支持层间泛化。但这三层都在Gemma 2 2B的中间层范围内，不能代表早期层或最终层。

### 3.2 任务泛化性：首字母拼写的代表性问题

首字母拼写任务的层级结构极其特殊：
- 深度固定为2（word -> first letter）
- 分支因子约40（每个字母约40-70个词）
- 近乎完美的共现（word出现时其首字母100%共现）
- 弱图形学特征（模型不需要"看到"字母，而是从语义推断）

这些特性使得首字母拼写成为一个**极端案例而非代表性案例**。真实世界中的特征层级（如"is a cat" -> "is an animal" -> "is a living thing"）具有更深的层级、不完美的共现、以及更强的语义基础。

**跨域实验的失败加剧了这个问题**：五个域的控制都失败了，这意味着论文事实上只在一个极端任务上有可解释的结果。

### 3.3 L0操作点的覆盖不充分

L0 phase transition是最稳健的发现，但仅有4个数据点（L0=22, 41, 82, 176）。声称存在L0=40-80区间的"相变"（phase transition），但该区间内仅有L0=41和L0=82两个数据点。相变的精确位置和锐度无法从4个点确定。

**建议**：将"phase transition"的措辞降级为"sharp decline"或"regime change"，除非能在L0=40-80区间内增加更密采样（如L0=50, 60, 70）。

---

## 四、可复现性审查

### 4.1 正面因素

- **全部training-free**：无需训练任何模型或SAE，仅使用预训练权重
- **公开基础设施**：SAELens v6, TransformerLens, sae-spelling, Gemma Scope SAE全部公开可用
- **数据验证**：validate_integration.py确认85个数值声明中84个匹配（integrity score 0.988）
- **随机种子**：seed=42贯穿所有实验

### 4.2 复现障碍

- **词汇表不一致**：三个不同的词汇表大小（1204, 1196, 1092）用于不同实验，tokenization细节未完整记录
- **probe训练细节**：k-sparse probe的训练过程（学习率、epoch数、early stopping标准）未在论文中完整报告
- **CMI估计器**：k-NN互信息估计器的k值选择、convergence criteria未完整记录
- **9→8 core words的不一致**：confound decomposition声称9个persistent core words，activation patching找到8个。第9个词（wrong/W）在L0=82时恢复但在其他3个L0值上都是FN。这种不一致需要在论文中明确解释。
- **656 vs 657 FN**：不同实验的FN计数存在1个token的差异，源于vocabulary tokenization的微小差别。虽然影响可忽略，但在复现时可能造成困惑。

### 4.3 最关键的复现风险

**tightened hedging的操作化定义**：strict hedging要求"至少1个parent-associated latent（k=5中的任一个）在L0=176时激活"。但不同的k值、不同的parent latent selection方法（cosine vs probe weight ranking）可能产生不同的strict rate。论文应报告k值敏感性分析。

---

## 五、统计方法论审查

### 5.1 多重比较处理：部分到位

- CMI的4个d'值正确应用了Bonferroni校正（4x）
- 但跨域的5个domain comparisons未做校正
- 字母级的25个absorption rates未做校正（虽然这里校正未必必要）

### 5.2 效应量报告：表现良好

- Cohen's d=-0.924（CMI分组比较）
- Bootstrap 95% CI覆盖所有主要发现
- L0 phase transition的cross-layer CV<10%
- Spearman rho贯穿全文

### 5.3 缺失的分析

- **无power analysis**：N=25的CMI分析缺乏统计效力评估
- **无effect size for patching**：activation patching仅报告0/8二元结果，缺少parent方向投影的连续量度
- **无confound decomposition的置信区间**：98.6% hedging（宽松）和6.2% hedging（严格）都应报告CI
- strict hedging的CI已报告（4.4%-8.2%），这是好的

---

## 六、对核心声明的效度判定

| 声明 | 内部效度 | 外部效度 | 置信度 |
|------|---------|---------|--------|
| H1: Metric不迁移到JumpReLU | 高：5个域一致失败 | 中：仅1个模型+1族SAE | 高 |
| H2: 98.6% hedging dominance | **低**：宽严落差92.4 pp | 低：单一任务 | **低** |
| 修正：6.2% strict hedging | 高：有shuffled control (3.4%) | 低：单一任务 | 中 |
| H3: L0 phase transition | 高：4点单调+3层交叉 | 中：仅中间层 | 高 |
| H4: CMI诊断 | 低：效力不足，不过校正 | 低：单一d'值有效 | 低 |
| Patching 0/8恢复 | 中：解释B未排除 | 低：仅8个词 | 中 |
| 控制失败是结构性的 | 高：1000随机向量分析 | 中：仅cosine>=0.025 | 高 |

---

## 七、总体建议

### 必须修正（阻断性）

1. **98.6% hedging叙事必须修正**：以strict rate 6.2%为主要数值，讨论宽严差异的含义
2. **CMI语言必须降级**：从"predicts/diagnostic"降至"exploratory association"
3. **Activation patching需讨论JumpReLU阈值效应**：当前仅考虑解释A，需讨论解释B
4. **"Phase transition"措辞需谨慎**：4个数据点不足以确立相变

### 建议改进（非阻断性）

5. 增加post-hoc power analysis
6. 报告patching后parent方向投影的连续值
7. 对strict hedging进行k值敏感性分析
8. 统一词汇表（选择1196词为canonical）
9. 在方法论节完整记录probe训练和CMI估计细节

### 方法论亮点（应在论文中突出）

10. 四级控制（C1-C4）的设计是同类研究中最全面的
11. 阈值敏感性分析（5x4网格）提供了极强的稳健性证据
12. 数据完整性验证（85/85 match）是自动化研究的最佳实践
13. 7个假设中4个被证伪并诚实报告，这是研究透明度的典范
