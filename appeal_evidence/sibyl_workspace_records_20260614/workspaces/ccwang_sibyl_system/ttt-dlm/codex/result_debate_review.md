# Codex 独立评审 - result_debate

**评审时间**: 2026-03-10
**模型**: Codex (GPT-5)

## 评审意见

**总评**

我给这场 `result debate` 的整体质量打 **6.5/10**。
如果给项目当前这组**结果本身**打分，我会给 **4.0/10**，比他们的 `4.5/10` 略低一点。

核心判断很明确：这场 debate 的优点是敢于否定自己的主假设，尤其是对 `RACFG` 的处理比较果断；但它仍然有两个明显问题：一是**对 n=16 pilot 的解释过多**，二是**对评估稳定性这个根问题重视还不够**。所以我同意 `PROCEED`，但只能是**有条件地继续**，而不是按原 Iteration 4 叙事继续。

### 1. Result debate quality：6 个视角是否充分？

总体上，这 6 个视角已经覆盖了大部分常见讨论维度，尤其有三点做得不错：

- 他们没有继续保护 `BSD + RACFG` 这个原始主叙事，而是接受了"核心创新可能不成立"。
- 他们注意到了 `novelty overlap`、`compute fairness`、`statistical insignificance`、`baseline inconsistency`，这比很多内部讨论成熟。
- 他们没有把 `A-CFG` 的 pilot 小胜直接包装成大突破，这一点是加分项。

但我认为还有几个明显盲点：

- **缺少"evaluation forensics"视角**。`Vanilla` 在同样的 16 个样本上从 `0%` 到 `18.8%` 摆动，这已经不是普通噪声，而是应该优先排查的系统性风险。这里需要一个专门的"是不是 pipeline / decoding / parser / seed / batching 出问题了"的视角。
- **缺少"power analysis / decision theory"视角**。`0/42` 不显著，在 `n=16` 下几乎是预期结果，不能同时拿来否定方法、又把 pilot 小优势当作方向依据。
- **缺少"mechanism falsification"视角**。他们说 `RACFG` 死了，但更准确的说法应该是：`JSD-based selector on Dream-7B` 被证伪了，不等于"reasoning-aware adaptive guidance"这个大类概念完全死亡。
- **缺少"top-tier reviewer simulation"视角**。外部评审不会只问"有没有信号"，而会问：这到底是新方法、负结果论文、还是一个现有方法的复现加诊断？这三种稿件的标准完全不同。

所以，6 个视角已经不错，但还不够"外部评审化"。

### 2. Synthesis fairness：`4.5/10` 和 `PROCEED` 是否公允？

我的看法是：

- `4.5/10` **略偏高**，我会给 **4.0/10**。
- `PROCEED` **方向正确**，但前提必须更严格。

为什么我会把分数再压低一点：

- `A-CFG` 在 `Countdown-16` 的 `12.5%` 其实只是 **2/16**。
- `GSM8K-16` 的 `37.5% vs 25.0%` 其实只是 **6/16 vs 4/16**，也就是多对了 **2 题**。
- `BSD+A-CFG` 的所谓"sub-additive"也是建立在 **1 题和 2 题的差别**上，证据强度很弱。
- `BSD` 的 novelty 已经很危险，`RACFG` 的核心 selector 又被击穿，Iter 4 的原始论文主线实际上已经崩了。

但我仍然同意 `PROCEED`，因为还有两块东西没死：

- `DMI` 仍然是唯一有 full-scale 支撑的真实信号。
- 如果把稿子重构成"对 MDLM training-free inference-time scaling 的系统诊断"，这仍然可能形成一篇有价值的 empirical paper。

换句话说，不是"继续原方案"，而是"继续抢救可发表资产"。

### 3. Debate 忽略了哪些风险？

我认为至少漏了 6 个重要风险：

- **评估管线风险**：`Vanilla` 在同一 16 样本上剧烈摆动，首先要怀疑 `evaluation harness`、`decoding nondeterminism`、`answer extraction`、`batch-order sensitivity`，而不是先讨论方法优劣。
- **pilot overfitting 风险**：20+ variants 迭代后再看 `n=16`，非常容易把超参数和叙事过拟合到极小样本。
- **selection bias 风险**：这 16 个样本是否代表整体难度分布？如果不是，任何结论都可能翻转。
- **single-backbone 风险**：很多结论可能只是 `Dream-7B` 的特性，尤其是 `JSD stability ~0.997`，未必能泛化到别的 MDLM。
- **novelty cliff 风险**：如果 `BSD` 接近已发表工作，`A-CFG` 又不是你们的方法，那么稿件的 novelty 可能比 debate 里估计的还低。
- **claim calibration 风险**：现在最容易犯的错误是把"一个 selector 失败"写成"一个研究方向失败"，或者把"pilot 有 2 题增益"写成"方法有效"。

### 4. Action plan critique：先做 `A-CFG full-scale`，再做 `DMI+A-CFG`，对吗？

我认为**不完全对**。
在当前状态下，最优先的不是直接 full-scale，而是先做**稳定性闸门实验**。

我建议的优先级是：

1. **先做 stability audit**
   - 固定一个 `128` 或 `256` 样本开发集。
   - 跑 `Vanilla`、`DMI`、`A-CFG`，统一代码、统一 parser、统一 seed protocol。
   - 先确认同配置重复运行是否可复现，确认方差到底来自 sampling 还是 pipeline。

2. **再做 confirmatory run**
   - 如果 `A-CFG` 在中等规模上仍稳定优于 `Vanilla`，再上 `Countdown-500 full-scale`。
   - 这一步要预先冻结超参数，不要再边看边调。

3. **然后再做 `DMI x A-CFG` 的 2x2 factorial**
   - 先证明 `A-CFG` 主效应存在，再测 interaction。
   - 否则组合实验只会把噪声放大。

4. **`BSD` 应立即降级**
   - 除非你们能拿出表征层面的独立证据证明它真的改变了 belief dynamics，否则不应该继续占主线资源。

所以，我不同意"`A-CFG full-scale` 然后立刻 `DMI+A-CFG`"这条线作为默认方案；我更赞成"**稳定性审计 -> confirmatory A-CFG -> 2x2 interaction**"。

### 5. Publication strategy：`30-40% main conference` 现实吗？

以**当前材料**看，我觉得 **偏乐观**。
我会给：

- **当前状态**：`main conference 15-25%`
- **如果补齐稳定性审计 + full-scale confirmatory evidence + 更强 framing**：`25-35%`
- **"any venue" 85%**：如果把 workshop、较宽口径 venue 都算进去，这个判断可以接受

现在的 framing "`systematic diagnostic study + A-CFG confirmation on Dream-7B`"方向是对的，但还不够强。更好的 framing 应该是：

- 不是"我们提出了新架构"
- 而是"我们系统检验了 MDLM 上几类 training-free inference-time scaling 机制，发现多数路径不稳或无效，`DMI` 是少数稳健例外，`JSD-based reasoning-aware selection` 在 Dream-7B 上失效，并给出失败原因"

这个 framing 比"确认 `A-CFG` 有点用"更像一篇能过评审的论文。因为单独的 `A-CFG confirmation` 太像复现，不像主要贡献。

### 6. Methodological concerns：实验设计、baseline、结论有什么问题？

这里的问题其实不少：

- **`n=16` 太小**，不足以支持方向性结论，更不足以支持 interaction 结论。
- **absence of significance != evidence of no effect**。`0/42` 不显著在这里更多说明"实验没 power"，不能当成强否定证据。
- **组合失败的结论过强**。`BSD+A-CFG` 的"sub-additive"现在证据不够。
- **baseline 还不够完整**。既然你们自己已经怀疑 `vanilla step-doubling` 可能是 Pareto-optimal，就应该把它升级为正式主 baseline，而不是附带观察。
- **compute fairness 需要更严格**。不要只报 FLOPs 倍数，要报 matched-compute curve，至少给出 accuracy-vs-FLOPs。
- **缺少 paired analysis**。应该用每道题的 paired outcome 做 `bootstrap` 或 `McNemar-style` 分析，而不只是看 aggregate mean。
- **缺少机制指标**。如果 `BSD` 是 representation-layer 方法，那就不该只用最终 exact accuracy 来判断生死；至少要看 entropy、rank、calibration、stepwise recovery。
- **探索性与验证性混在一起**。20+ methods 迭代后，如果还用同一批小样本做方向判断，很容易产生 researcher degrees of freedom。

### 7. 具体、可执行，而且你们大概率还没认真做过的建议

下面这些是我认为最值得立刻加进去的动作：

- **做一个"稳定性闸门"实验**
  固定 `128/256` 样本，冻结 parser 和 decode protocol，跑 `Vanilla/DMI/A-CFG` 多 seed，输出逐题结果哈希。若同配置结果不稳定，暂停一切新方法实验。

- **做 `DMI x A-CFG` 的 2x2 factorial，而不是只比 best number**
  四组：`Vanilla`、`DMI`、`A-CFG`、`DMI+A-CFG`，用相同样本、相同 seed、matched compute。这样才能真正回答"是否 orthogonal"。

- **给 `RACFG` 做一个 oracle upper bound**
  不要只测 `JSD-based selector`。直接用 post-hoc oracle 关键位置，或随机位置对照，看看"位置选择"这件事是否有理论上限。
  如果 oracle 都没帮助，整个方向可以彻底终止；如果 oracle 有帮助，那死的是 selector，不是问题本身。

- **把 `vanilla step scaling` 正式化成主 baseline**
  例如 `1.0x / 1.5x / 2.0x` compute 下的 accuracy 曲线。否则任何"training-free scaling method"都缺少最关键的对照。

- **为 `BSD` 增加机制诊断，而不是只看最终 accuracy**
  至少报告每步 entropy、top-1 stability、ground-truth token rank、embedding norm drift。
  如果这些都没有稳定改善，就不要再把 `BSD` 当作贡献点。

- **加一个跨 backbone 的最小泛化检查**
  哪怕只加 1 个额外 MDLM，也能显著提升论文说服力。否则很多结论都会被评审打成"Dream-7B artifact"。

- **把"evaluation instability itself"变成论文中的一个结果**
  如果小样本 ranking 极不稳定，这本身就是对该研究方向很重要的 methodological finding，甚至比某个 pilot 提升更有发表价值。

**结论**

这场 debate 不是差，而是"已经足够诚实，但还不够硬"。
它成功拆掉了一个站不住的主叙事，但还没有完全建立一个足够稳的替代叙事。

我的最终建议是：

- **保留 `PROCEED`，但改成 conditional proceed**
- **先审计评估稳定性，再做 confirmatory A-CFG**
- **把 `BSD` 降级，把 `RACFG` 限定为 `JSD-based selector` 的失败案例**
- **把论文主线改成：系统诊断 + 稳健例外 (`DMI`) + 失败机制分析**

如果你愿意，我还可以下一步直接帮你把这份评审进一步压缩成一版"适合放进内部 result debate 总结文档"的中文正式评语。

## 评分

6.5/10（debate 质量） / 4.0/10（结果质量）
