# Codex 独立评审 - result_debate

**评审时间**: 2026-03-09
**模型**: Codex (GPT-5)

## 评审意见

**总体判断**

我不会支持当前的综合结论原样进入论文叙事。现有证据更像是一个"现象学发现"而不是一个已被验证的方法学结论：你们很可能发现了"浅层状态/输入干预优于参数干预"的信号，但还没有证明这是因为"去噪时学习"本身有效。当前最大问题不是 DTA 暂时没跑完，而是证据链的内部效度还没闭合。

**1. 实验设计与统计严谨性**

1. 目前设计不够充分。`3 seeds` 只能说明结果方向有一定稳定性，但对 `Countdown-500` 这类逐题二元成败任务，真正的统计单位是题目而不是 seed 均值。你们需要逐题 paired 结果，做 `McNemar`、paired bootstrap/permutation，最好再做带 `item` 和 `seed` 随机效应的 logistic mixed model。
2. DMI 的"2x 提升"在相对值上醒目，但绝对值只是 `4.7% -> 9.3%`，约等于每 500 题多答对 23 题。这个量级值得重视，但也非常容易被评测脚本、长度控制、答案抽取规则放大。
3. 最大红旗是 `Vanilla 4.7%` 对 `Dream 论文 16.0%` 的差距。除非你们能严格证明 benchmark split、prompt、temperature、steps、max length、answer extraction、stop condition、evaluation protocol 完全不同，否则这会直接动摇所有相对改进。
4. 仓库里的过程记录已经说明过，之前一轮"方法差异"后来被发现主要是温度和输出长度驱动，而不是方法本身。这说明你们的 pipeline 已经出现过一次系统性混淆，当前所有新结果都必须重新接受 length-matched / temperature-matched 审查。
5. 还存在明显的多重比较和 winner's curse 风险。你们试了多种干预、多个任务、pilot/full-scale 多轮切换后才出现 DMI 正信号，这种情况下如果没有 hold-out benchmark，DMI 的效应通常会被高估。

**2. 理论框架的有效性**

1. 我倾向于认为：`VDTA` 的 ELBO 单调性证明即使数学上成立，也未必对当前实验现象有解释力。它更像是一个关于 surrogate objective 的局部性质，而不是关于任务准确率的保证。
2. 理论与实验脱节至少有三层。第一，DTA 在采样中在线改参数，破坏了固定 reverse kernel 的平稳假设。第二，你们优化的是模型自己"已揭示 token"的 masked LM 损失，这本质上是自举伪标签，可能强化早期错误而不是纠正错误。第三，局部提高自一致似然，不等于提高 Countdown/GSM8K 的最终正确率。
3. `LoRA delta norm ~1e-5` 是非常不健康的信号，但要说清楚：它未必低于 `bf16` 动态范围，却很可能低于"有效参数分辨率"，尤其如果更新或回写发生在低精度上。换言之，理论可能在一个"理想可更新"的 regime 上成立，但你们实际跑在"几乎没真正更新"的 regime。
4. 如果最终 DTA 仍失败，最合理的解释不是"理论错了"，而是"理论证明了一个与任务目标弱相关的量"，同时参数更新给去噪过程引入了非平稳性，伤害大于收益。

**3. DMI 结果的可靠性**

1. 我不认为 `9.3% vs 4.7%` 可以被轻易视为噪声。三组 seed 都是正向，幅度也不小，这足以把 DMI 视为"真实信号候选"。
2. 但我不接受"这已经证明 inference-time learning 有效"的解释。更大的可能性是替代解释。我主观上会给"替代解释成立"一个高于"真实在线学习成立"的概率，粗略说是 `60-70%` 对 `30-40%`。
3. 最强的替代解释有三类：`隐式温度/熵调节`、`额外跨步信息传递`、`只改善格式/约束满足而非推理本身`。第三点尤其危险，因为 Countdown 的准确率可能被"表达式合法性、数字使用完整性、长度校准"显著影响。
4. 你们必须做对照来拆机制：同等幅度但不学习的 embedding 扰动、随机或冻结 adapter、entropy-matched temperature baseline、只看"合法表达式条件下的正确率"。如果这些对照也能复现大部分增益，那么 DMI 的贡献应改写为"轻量状态注入"而不是"显式推理时学习"。

**4. 竞争定位**

1. 你们现在面对的不是一个空白赛道。`ReMDM` 已经占了 training-free remasking 与 inference-time scaling；`CORE` 占了上下文鲁棒 remasking；`Soft-Masked DLMs` 和 `MetaState` 则直接占了"跨去噪步保留信息/工作记忆"这条主线。
2. 这对你们不利的地方在于：DMI 很可能被审稿人理解为一种"弱版本的跨步记忆机制"，而不是新的 test-time learning 范式。`MetaState` 甚至已经把"冻结 backbone + 极少参数 + 持久工作记忆"说得很完整了。
3. 所以你们的差异化不能再写成"我们也在去噪过程中注入更多信息"，而必须更尖锐：`在不继续预训练、不改 backbone 的前提下，浅层状态化干预比参数更新更稳、更便宜、更适合推理时使用`。如果做不到这点，竞争定位会很弱。
4. 你们还必须正面解释自己与 `CORE` 的分歧。若 `CORE` 在 reasoning/code 上有稳定提升，而你们的 remasking 系列几乎无效，审稿人首先会怀疑实现或 protocol，而不是接受"remasking 无效"这一科学结论。

**5. 叙事策略**

1. "Why Deeper Isn't Better" 这个 pivot 不是不能做，但现在证据还不够，容易显得像为负结果找借口。
2. 这个叙事成立的前提不是"DTA 失败了"，而是"你们系统地证明了干预深度越深，reverse process 越不稳；而浅层/状态式干预在 matched compute 下更可靠"。没有这条因果链，叙事会显得事后合理化。
3. 如果 DTA full-scale 继续负向，我建议不要再把 DTA 作为主角。更可信的主线是：`在冻结 DLM 中，参数空间适配破坏去噪平稳性，而输入/状态空间干预能保留跨步信息且更稳。`
4. 你们目前给出的"发表概率 95%"过于乐观。以当前证据，我会认为"值得继续做"的概率高，但"已接近稳定论文"的概率并不高。

**6. 被忽略的风险和盲点**

1. 六位辩论者没有充分强调"基线失真"是头号风险。`4.7% vs 16.0%` 如果解释不清，整篇论文都会被视为建立在不稳基线之上。
2. 他们也没有充分讨论"DMI 可能主要修复格式而非推理"。在 Countdown 上，这个替代解释非常强。
3. `ReMDM Correction Precision = 31.3%` 只讲了 precision，没有 recall、net utility、calibration curve。低 precision 并不自动推出 remasking 无效。
4. 没有人真正把"自举伪标签导致错误自强化"当作 DTA 的核心机制风险。这其实比"LoRA 太小"更本质。
5. 还有一个被低估的风险是 protocol drift。pilot 与 full-scale 不一致，未必只是小样本噪声，更可能是实验设置在变，导致你们在比较不同 pipeline 而不是比较同一方法。

**7. 接下来最重要的 5 个实验/分析**

1. 先做 `baseline audit`。把 Dream 论文设置逐项对齐：数据版本、prompt、temperature、steps、length、answer extraction、stop rule。这个问题不解决，其他实验都只能算探索。
2. 保存逐题日志，做 `paired` 统计。至少记录 correctness、表达式合法性、数字使用约束、输出长度、compute cost，然后对 `Vanilla vs DMI vs DTA` 做 McNemar/paired bootstrap。
3. 做 `DMI 机制拆解对照`。最关键的是"不学习但加同等状态信息"的 control；再加 entropy-matched temperature control。这个实验会直接决定 DMI 是"学习"还是"状态注入"。
4. 做 `DTA 数值与稳定性 sweep`。测试 `fp32 master weights`、loss scaling、不同 lr/rank/step、update clipping，并记录 delta norm、logit drift、step-wise KL。如果找不到"有实质更新但不崩"的窗口，DTA 应当被降级为负结果主角。
5. 做 `跨任务与 compute-matched` 验证。至少把 DMI 和最关键 control 跑到 `GSM8K`，并与额外 denoising steps、parallel vote、最好还有 faithful `CORE/ReMDM` 设置作 compute-matched 比较。

## 评分

**4.8/10**

理由：项目有一个值得追的经验信号（DMI），也有勇于接受负结果的研究态度。但从严格学术标准看，当前项目的内部效度、机制识别、竞争定位都还不够稳。现在更像"一个可能很有价值的研究转折点"，还不是"一个已经成立的方法论文"。
