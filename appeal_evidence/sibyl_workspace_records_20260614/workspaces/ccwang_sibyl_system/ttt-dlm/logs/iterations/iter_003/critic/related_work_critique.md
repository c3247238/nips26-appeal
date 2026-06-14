## 评分: 7/10

## 优点

1. **结构清晰，三轴分类合理**。按"掩码扩散语言模型 / 推理时计算扩展 / RL 推理增强"三轴组织文献，逻辑清晰。最后的"Positioning of This Work"段落明确了本文在文献版图中的位置，对审稿人友好。

2. **对每个方法的描述准确且有区分度**。ReMDM 的四种采样器、RemeDi 的双流架构、Prism 的层次搜索+SVF、PG-DLM 的粒子 Gibbs 等，都抓住了核心技术贡献，没有流于表面。

3. **关键洞察的提炼到位**。"所有成功方法都引入了外部信息源"这一观察被反复强调（Section 2.2 末段和 2.4），为本文的核心假说提供了文献层面的支撑。

4. **"Positioning of This Work"段落有效**。明确区分了本文与 ReMDM（规模差异）、RemeDi/Prism/PG-DLM（外部信息源）的差异，避免了审稿人提出"你和 XX 有什么区别"的质疑。

## 不足

1. **与引言内容高度重叠**。引言第 1 段已详细介绍 MDLM → LLaDA → Dream → LLaDA 1.5/MoE/Dream-Coder 的发展脉络；Related Work 的 Section 2.1 几乎逐字重述了相同内容，只是换了引用格式（`\citep` → 行内）。同样，引言第 3 段和 Related Work Section 2.2 对 ReMDM、RemeDi、Prism、UnMaskFork、TReASURe、PG-DLM 的描述也有大量重复。**这是两个章节之间最严重的一致性问题**。建议引言只做高层概述，Related Work 负责详细技术描述。

2. **缺少大纲要求的 Section 3.3（评估陷阱）**。大纲明确列出了 "3.3 推理时扩展的评估陷阱——代理指标（PPL）的局限性、小样本评估的过度乐观偏差"，但当前 Related Work 完全没有涉及这一小节。这部分内容是本文的核心方法论贡献之一，在 Related Work 中缺席会削弱论文的独特性。需要补充关于 NLG 评估指标局限性、小样本偏差的已有文献（如 Dodge et al. 2020 on instability of evaluation, Post 2018 on BLEU pitfalls 等）。

3. **AR 模型推理时扩展的相关工作几乎没有独立覆盖**。大纲 Section 3.2 提到"AR 模型：Best-of-N、树搜索、CoT（综述 Zhang et al. 2025; Ji et al. 2025）"，但当前文本仅在引言中一句带过，Related Work 中完全没有对 AR 模型 TTS 方法的独立讨论。考虑到本文的核心动机之一是"AR 模型的 TTS 成功能否迁移到 DLM"，对 AR TTS 文献的综述不可或缺。

4. **时间线/发展脉络感不足**。Section 2.1 按方法名称罗列，但缺少明确的时间线叙事。读者难以感知这个领域的发展速度（MDLM 2024 → LLaDA/Dream 2025 → 各种扩展 2025-2026）。建议在段首加入"Within less than two years..."之类的时间框架语句。

5. **RL 方法（Section 2.3）的覆盖深度不均**。d1 和 wd1 获得了详细的技术描述（具体数字如 44.2% MATH500、84.5% GSM8K），但 DiSPO 和 SPG 只有一句话。如果要保持对称性，要么都给详细描述，要么都只给一句话概括。

6. **缺少对 DLM 与 continuous diffusion（图像领域）的关联讨论**。推理时计算扩展在图像扩散模型中也有丰富文献（如 classifier guidance、DPO for diffusion 等）。简要提及这些工作可以帮助读者理解 DLM 推理时扩展的更广泛背景。

7. **"Positioning"段落过长**。当前约 150 词，部分内容（如"This finding complements the positive results..."）属于讨论性质，更适合放在 Analysis/Discussion 部分。定位段应简洁，聚焦于"我们做了什么不同的事"。

## 具体修改建议

1. **消除与引言的重叠**。具体操作：
   - 引言第 1 段压缩为：仅提 MDLM/LLaDA/Dream 三个里程碑，各一句话，不展开技术细节。
   - 引言第 3 段压缩为：按 remasking/tree-search/RL 三类各一句概括，不列方法名。
   - Related Work 负责展开所有技术细节。
   - 或者反过来：Related Work 的 2.1 开头加一句"As outlined in Section 1, ..."然后聚焦于引言未涉及的技术细节。

2. **补充 Section 2.3.5 或 Section 2.4（评估方法论陷阱）**。至少覆盖：
   - 机器翻译/NLG 中关于 BLEU/PPL 局限性的经典文献
   - 深度学习实验中小样本评估不稳定性的讨论
   - DLM 特有的评估挑战（非自回归生成的指标适用性）

3. **添加 AR 模型 TTS 简要综述**。建议在 Section 2.2 开头加一段（约 100-150 词），涵盖：
   - Best-of-N 和 rejection sampling
   - 树搜索（MCTS for LLMs）
   - Chain-of-thought 和 process reward models
   - 点出"这些方法在 AR 模型上的成功激发了对 DLM 的类似探索"

4. **均衡 RL 方法的描述深度**。建议统一为每个方法 1-2 句：核心技术 + 主要结果。去掉 wd1 的具体数字（44.2%、84.5%），改为定性描述（"significant gains on mathematical reasoning benchmarks"）。

5. **Positioning 段落精简至 80-100 词**。保留核心定位（轻量级 + 大规模 + 负面结果），将推测性讨论（"internal signals do not contain sufficient information..."）移至 Analysis 部分。

6. **在 Section 2.1 段首加入时间线语句**。例如："Masked diffusion language models have progressed from small-scale proof-of-concept (MDLM, mid-2024) to competitive billion-parameter models (LLaDA, Dream, early 2025) within approximately 18 months, an unusually rapid trajectory."

7. **考虑加入一张文献分类表格**。将所有相关方法按维度（需要训练？需要外部验证器？计算开销？模型规模？）制成表格，视觉化本文的定位优势。这也能减少正文中的方法罗列篇幅。
