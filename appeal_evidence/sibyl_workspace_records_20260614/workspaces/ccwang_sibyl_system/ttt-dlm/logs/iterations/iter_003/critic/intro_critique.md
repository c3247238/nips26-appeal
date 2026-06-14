## 评分: 8/10

## 优点

1. **研究问题定义清晰且有价值**。"轻量级、无训练推理时策略能否提升 DLM 生成质量"这一问题被精确定义（无训练、无外部验证器、无架构修改、不超过 2x 计算开销），边界条件明确，便于读者快速理解研究范围。

2. **文献覆盖全面且组织有序**。从 MDLM 到 LLaDA、Dream，再到 ReMDM、RemeDi、Prism、UnMaskFork、TReASURe、d1/wd1/DiFFPO/DiSPO/PG-DLM，几乎涵盖了截至 2026 年初 DLM 推理时扩展的所有主要工作。引言在介绍这些工作时自然地区分了"需要外部信息源"与"纯内部信号"两类方法，为本文的定位做了铺垫。

3. **负面结果的呈现方式有力**。七类策略的失败被逐一列出，每项都有具体数字（PPL 变化百分比、p 值），而非笼统地说"都没用"。这种精确的呈现增强了可信度。

4. **三个 deeper insights 结构精炼**。小样本假象（small-sample mirage）、PPL 不可靠性、规模依赖失败模式——这三点将纯粹的负面结果升华为有建设性的方法论贡献，避免了"我们试了很多东西都没用"的乏味叙事。

5. **统计严谨性突出**。101 prompts x 3 seeds = 303 samples、paired t-tests、Wilcoxon signed-rank tests、Bonferroni-corrected bootstrap CI——这些在引言中就被明确交代，体现了对负面结果论文尤为重要的严谨态度。

6. **贡献列表具体且可验证**。五条贡献都有明确的实验支撑，没有空泛的 claim。

## 不足

1. **引言过长，信息密度过高**。当前引言约 1100 词（不含 LaTeX 标记），第一段就密集罗列了 MDLM、LLaDA、Dream、LLaDA 1.5、LLaDA-MoE、Dream-Coder 共 6 个模型/工作。第三段更是一口气列出 ReMDM、RemeDi、Prism、UnMaskFork、TReASURe、d1、wd1、DiFFPO、DiSPO、PG-DLM 共 10 个方法。这些内容与 Related Work 严重重叠（见下方交叉一致性检查），应将详细介绍移至 Section 2，引言仅保留高层脉络。

2. **缺少 Figure 1 的引用**。大纲明确要求在引言中放置"方法全景图"（图 1），但当前引言文本中没有引用任何图表。一张"计算开销 vs 是否需要训练/验证器"的二维定位图可以大幅减少文字叙述，让读者一目了然。

3. **"free lunch"的表述需要谨慎**。第四段使用了"free lunch"一词，但本文最终证明这种 free lunch 不存在。虽然这是修辞手法，但在审稿人看来可能显得不够严谨。建议改为更中性的表述，例如"a cost-effective plug-and-play enhancement"。

4. **Itemized 负面结果中的表述不够统一**。七项结果有的报告 PPL 变化百分比+p 值（TCR、entropy、temperature），有的只报告现象描述（ReMask-Retry、parallel voting），有的报告 PPL 百分比但没有 p 值（Best-of-N）。建议统一格式：每项都报告 PPL delta%、p 值（若适用）、核心失败机制。

5. **"definitive negative result"的 claim 可能过强**。贡献第 2 条使用了"definitive"一词，但论文仅测试了 3 个模型规模、开放文本生成任务，且未涉及推理任务（数学、代码）。大纲的局限性部分也承认了这些不足。建议将"definitive"改为"strong"或"robust"，并在贡献描述中加入适用范围限定。

6. **对 ReMDM 小模型成功的讨论不够**。引言提到 ReMDM 在 170M 模型上有效，本文在 0.6B-8B 上失败，但没有在引言中明确提出"规模依赖"这一关键张力。虽然在 deeper insights 中有提及，但引言的叙事逻辑可以更早地引入这个对比来增强研究动机。

7. **缺少对读者的路线图提示**。末尾的 paper organization 段落虽然标准，但可以更有引导性——例如提示读者"如果你对方法论贡献最感兴趣，可以直接跳到 Section 5.1 的小样本偏差分析"。

## 具体修改建议

1. **压缩第 1 段和第 3 段**。第 1 段保留 MDLM → LLaDA → Dream 的主线即可，LLaDA 1.5/MoE/Dream-Coder 移至 Related Work。第 3 段按"remasking methods / tree search / RL methods"三类各用一句话概括，具体方法名在 Related Work 中展开。预计可缩减 200-300 词。

2. **添加 Figure 1 引用**。在第三段末尾或第四段开头加入"Figure 1 positions our work..."，并确保图已准备好。

3. **统一 itemized list 格式**。建议格式：`**策略名**: PPL delta (p-value if applicable); 失败机制一句话描述。` 例如：
   - `**ReMask-Retry** (0.6B): -X% PPL (p=Y), but driven entirely by text degeneracy (bigram diversity 0.479 → 0.260).`
   - `**Parallel Voting** (K=4): +114% PPL; 66% of positions show no consensus across trajectories.`

4. **将 "definitive" 改为 "strong empirical"**，并在贡献第 2 条中加入限定语"on open-ended text generation tasks"。

5. **在第 2 段或第 3 段早期引入 ReMDM 的规模张力**。例如："While ReMDM demonstrated that remasking improves quality on ~170M models, whether these gains persist at modern scales (0.6B--8B) remains untested---a gap we directly address."

6. **考虑将 deeper insights 的三个段落改为 numbered list 或小节标题**，与贡献列表形成更清晰的对应关系（insight 1 ↔ contribution 3, insight 2 ↔ contribution 4, insight 3 ↔ contribution 5）。

7. **"free lunch"改为"plug-and-play enhancement applicable to any off-the-shelf DLM"**，去掉引号和 scare-quote 感。
