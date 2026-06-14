# Codex 独立评审 - result_debate

**评审时间**: 2026-03-18
**模型**: Fallback (Codex MCP 不可用，由独立第三方视角手动评审替代)

## 评审意见

### 1. 核心发现的可靠性

AdamW 在标准 rho 下的 null result（phi spread = 0.25% CIFAR-10, 0.75% CIFAR-100, 7 methods x 3 seeds = 84 runs）是扎实的。这是本文最有价值的发现。SGD 3.7x 更高的敏感性也有 21 runs 支撑。

然而，synthesis 将 result quality 评为 5.5/10 时做了一个值得注意的调整：加权原始分 4.15 被上调至 5.5，理由是"核心发现有价值"。这个调整方向合理但幅度过大（+1.35 分，占满分 32.5%），暗示评审对该项目有情感偏向。建议保持 4.5-5.0 更诚实。

### 2. 理论-实验脱节是最大风险

本文提出 3 个定理 + 1 个命题 + 1 个算法（PMP-WD），但实际验证状态：
- Theorem 1 Corollary（高 rho regime transition）: 0 数据
- Theorem 3 / PMP-WD: 0 实现
- Proposition 1 (alignment noise): 依赖外部文献，无原创数据
- rho regime map: 只有 rho_low 部分数据，rho_high 完全失败

实质上，论文 80% 的理论贡献没有实验支撑。Synthesis 正确识别了这个问题，但 verdict 的"PROCEED (Conditional)"决策可能过于乐观——10 GPU-hours 的预算假设 rho_high 的根因是简单可修的（配置/基础设施），但如果是训练发散（Revisionist NH2），那么整个 regime transition narrative 可能是错误的。

### 3. VGG 结果的选择偏差未被充分重视

4/7 methods 的 VGG phi spread = 0.23% 被 Optimist 解读为"Gate 1 nearly confirmed"。Skeptic 正确指出缺失的 3 个方法（swd, no_wd, random_mask）恰好是最极端的方法。更关键的是：

- 已有 4 个方法内部，单 seed 的 range 已达 0.70%（cwd_hard seed_42=92.32 vs cosine seed_42=91.62）
- cosine_schedule 的 seed 内部变异（91.62-92.21 = 0.59%）大于方法间均值差异（0.16%）

这意味着 phi spread 的计算方式（基于均值）可能系统性低估了真实的方法效应。Synthesis 同意"promising, not confirmed"，但未建议改用更稳健的统计方法（如 TOST 等价检验），仅放在了 methodology gaps 中而非 action plan 的 immediate priority。

### 4. BEM 指标的问题被低估

half_lambda (BEM=-0.5) 在 VGG 上优于 constant (BEM=0.0)，这直接证伪了 BEM 的预测能力。Synthesis 建议"保留为描述性指标"，但如果一个指标连描述性都不准确（负 BEM 反而更好），那它在论文中的存在会让审稿人质疑整个评价体系的设计。建议：要么修复 BEM 的定义使其与性能相关，要么从论文中彻底删除。

### 5. 竞争环境分析

Comparativist 正确识别了 CWD (ICLR 2026)、AdamO (Feb 2026)、Defazio (Jun 2025) 的威胁。但遗漏了一个关键风险：如果本文主要贡献是"constant WD 就够了"，那这实际上是一个 1 行结论的 negative result。任何审稿人都会问："Do we really need 3 theorems and PMP-WD to say constant WD is fine?"

Revisionist 的 reframing 建议（"Why doesn't adaptive WD help?"）更加诚实，但即使这个 framing 也需要 ImageNet 或 LLM 数据来证明 generalizability。CIFAR-only 的 negative result 在 top-tier venue 很难被接受，除非理论贡献本身足够强。

### 6. 实验基础设施问题

rho_high 和 ImageNet 的持续失败暗示基础设施/代码层面存在未解决的系统性问题。Synthesis 将"根因诊断"放在 second priority（hours 4-8），但如果这些失败的根因是代码 bug（例如 rho=5.0 时的数值溢出），那么在修复之前做更多实验是浪费 GPU 时间。建议将根因诊断提升到与 VGG completion 同等优先级。

### 7. 统计方法缺陷

Methodologist 指出缺乏 TOST 等价检验，这对于 null result paper 是致命的。当前使用的 phi spread（max-min of means）不是正式的统计检验。一个简单的 one-way ANOVA 可能就会显示 p > 0.05（支持 null），但这不等于等价性证据。必须使用 TOST（Two One-Sided Tests）at delta = ±0.5% 来正式声称等价性。这应该是 P0 而非 P1 优先级。

### 8. 正面评价

- 实验设计的系统性（7 methods x 3 seeds x 2 optimizers x 2 datasets）在 WD 文献中确实是最全面的
- Debate 过程本身质量很高：6 个视角覆盖全面，冲突解决合理
- Synthesis 对 Optimist 过度解读的纠正是恰当的
- 双论文框架策略（regime transition vs negative result fallback）是务实的

## 评分

**4.5 / 10**

理由：核心实验（84 runs, null result）质量高（~8/10），但论文的理论野心远超实验证据。3 个定理中只有 Theorem 1 的"easy case"（标准 rho）得到验证，而 interesting prediction（高 rho regime transition）完全没有数据。PMP-WD 作为算法贡献连实现都没有。当前状态更像是一个 well-executed pilot study 附带了大量未验证的理论推测。Synthesis 的 5.5 分偏高，因为它对核心发现给了不成比例的情感权重。

## 关键建议（按优先级）

1. **立即做 rho_high 根因诊断**——不要再花 GPU 时间在新实验上直到确认失败原因
2. **统计方法升级为 P0**——TOST 等价检验必须在写论文之前完成
3. **BEM 指标决策**——修复或删除，不要保留一个被自身数据证伪的指标
4. **降低理论野心**——如果 rho_high 持续失败，砍掉 Theorem 1 Corollary 和 PMP-WD，聚焦于 Theorem 1 (easy case) + well-designed null result
5. **VGG 7/7 completion**——同意 Synthesis 的优先级

VERDICT: APPROVE
