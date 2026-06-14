# Lessons from Iteration 6

**Date**: 2026-04-15 | **Score**: 6.5/10 | **Trajectory**: Improving (5.5 x3 -> 6.5 -> 6.0 -> 6.5)

---

## Must Improve

- **[DATA INTEGRITY -- BLOCKING] 6 迭代数据管道失败必须终结**: 论文报告 absorbed mean CMI=0.687，源数据为 0.6492（5.9% 误差）。Mann-Whitney U=41 vs 源数据 U=28。三种不同词汇量（1,092/1,196/1,204）产生两个不同的 L0=82 吸收率（14.39% vs 15.96%）。这是连续第 6 次迭代出现数据管道完整性问题（iter 1-3: PILOT 传播; iter 5: Sobel z 错误路径; iter 6: CMI 分区不一致）。根因始终相同：无自动化 source-to-summary 交叉验证。**下一迭代 P0 任务：实现 validate_integration.py**，在写作修订后自动对比论文数值与源 JSON。这是零成本零 GPU 的投资，可消除 6 迭代的系统性弱点。

- **[EXPERIMENT -- BLOCKING] 必须执行 3 个阻断实验**: (1) **Activation patching on 9 core words**（0.5-1 GPU-hour）：论文关于"真正竞争排斥"的唯一因果证据，zero child feature -> check parent recovery。正面结果确认小规模竞争排斥存在；负面结果加强"全是 hedging"叙事；任何结果都加强论文。(2) **Tightened hedging classification**（1 GPU-hour）：当前 98.6% hedging 是上界——不检查特定 parent latent 是否在更高 L0 激活，仅检查 token 是否停止为 FN。在 L0=176，99.2% token 平凡地解决。检查 parent-specific 激活以区分真正 hedging 和补偿性解决。(3) **CMI replication at L0=22**（1 GPU-hour）：所有 probes 在 L0=22 达到 F1=1.0，消除探针质量混杂。吸收方差最大（0%-66%）。pre-register d'=10，如果复制成功则理论支柱稳固。

- **[ANALYSIS -- BLOCKING] CMI 理论声明必须降级**: CMI-absorption 相关在 d'=10 为 rho=-0.383（p=0.059 未校正, p=0.236 Bonferroni 校正），在 d'>=20 符号反转。符号反转是定性失败。Section 6 标题 "CMI Predicts Absorption Susceptibility" 从 p=0.236 证据不能成立。'predicts' -> 'shows directionally correct negative association with'; 'criterion' -> 'candidate diagnostic'。所有引用 CMI 结果的位置必须同时报告未校正和 Bonferroni 校正 p 值。

- **[ANALYSIS -- HIGH] CMI 探针质量混杂必须控制**: Absorption rate 与 probe F1 相关 rho=-0.67（p<0.001），但 CMI 分析从未控制此混杂。必须计算 partial Spearman rho(CMI, absorption | probe_F1) 和限制于 F1>0.85 的 10 个字母的子集相关。如果 partial rho 下降至 -0.2 以下，CMI-absorption 关系实质上被探针质量混杂。零 GPU，30 分钟。

- **[EXPERIMENT -- HIGH] 控制失败必须被诊断**: 论文识别了中心发现（shuffled > measured across 所有 5 个域），但从未解释 WHY。阈值敏感性数据已计算（141KB ablation_threshold_sensitivity.json）但未报告——这是零成本遗漏。分析性计算：随机单位向量与 16,384 decoder columns 的余弦相似度分布，在 tau_cos=0.025 时的 expected n_candidate_features。比较 shuffled vs true probes。

---

## Watch Out

- **[ANALYSIS] Letter S 是显著离群值**: CMI=0.961（高，预测抵抗吸收）但 absorption rate=31.43%（高）。n=25 和边际显著性下，单个高杠杆观测可能驱动结果。必须运行 leave-one-out sensitivity analysis 并讨论 S 的特殊性（最高词数 n=70，可能影响 CMI 估计）。

- **[ANALYSIS] 两种解释的歧义必须明确**: 论文的 "metric does not transfer to JumpReLU" 混淆了两个解释：(a) JumpReLU SAEs 真的没有层级驱动吸收（metric 正确测量近零），(b) metric 阈值对 JumpReLU 几何结构校准不当。数据与两者一致。Activation patching 是区分检验。在执行之前，使用中性语言："metric 的输出与 JumpReLU SAEs 上的层级驱动吸收不一致"。

- **[WRITING] 跨域新颖性声明自相矛盾**: 不能同时声称"首次跨域吸收表征"的新颖性，又承认"所有速率不可解释"。重新框架为"首次跨架构和跨域验证 Chanin metric on JumpReLU SAEs，证明普遍失败"。

- **[WRITING] 标题暗示过强结论**: "Beyond Competitive Exclusion" 暗示已证伪竞争排斥。论文实际证明 Chanin METRIC 在 JumpReLU SAEs 上混淆 hedging 和竞争排斥；竞争排斥是否存在未回答（待 activation patching）。

- **[PLANNING] 必须实施 gate-first 执行**: 决策门（"如果 shuffled > 20%，调查后再继续"）存在于文档中但未作为执行阻塞。跨域实验在控制校准确认前运行。下一迭代：控制实验必须是测量实验的前置依赖。非阻塞决策门等同于无决策门。

- **[PLANNING] 深度优先优于广度优先**: 7 个假设中 4 个被证伪（57% 失败率）。~3.5 GPU-hours 花在低价值实验上（不可解释的跨域测量、零匹配的无监督检测），而 3 个最高信息增益实验（activation patching、control diagnosis、CMI L0=22 replication）未执行。下一迭代：先执行最高信息增益实验，再考虑范围扩展。

- **[PIPELINE] validate_integration.py 必须在 iter 7 第一天实现**: 这已经是第 3 次被建议但从未实现。将其作为 P0 阻塞任务处理，不允许任何科学任务在此之前执行。

---

## Keep Doing

- **战略枢转能力**: 系统成功地从失败方向（观察数据上的因果中介）转向更强方向（metric audit）。枢转由证据驱动（within-width Gamma=1.0, control failure），而非恐慌。这是健康的研究判断力，必须保留。

- **综合控制套件设计**: 四控制（随机探针、打乱标签、稠密探针上界、未训练 SAE）跨五个域部署，是任何吸收研究中最全面的控制套件。所有五个域的普遍失败是毁灭性的且呈现清晰。这应成为未来所有 metric validation 工作的标准。

- **诚实负面结果报告（连续 6 轮）**: H2 证伪（96.9% pilot -> 1.4% full），H4 证伪（零匹配对），H6 功效不足，H7 证伪（两者均双峰）——全部以具体的预期 vs 实际值和清晰解释报告。这是论文在 ALL reviews 中始终获得最高评价的方面。**绝不能妥协。**

- **L0 相变（最稳健发现）**: 42.9% -> 0.8% 跨四个 L0 值，跨层 CV < 10%，bootstrap CIs。单调、跨层稳定、直接可操作。无审查质疑此发现。

- **零实验失败（连续 3 轮，50/50 任务）**: 基础设施可靠性已是已解决的问题。

- **Per-letter 粒度跟踪**: 每字母的 probe F1、absorption rate、CMI 估计。这一粒度使得发现 probe quality confound (rho=-0.67) 成为可能——这本身是重要的方法论观察。

- **Bootstrap CIs 贯穿全文**: 所有吸收率报告 95% bootstrap CI（10,000 resamples, seed=42）。这是最佳实践。

- **Cross-layer validation**: L0=82 在 layers 10、12、20 上测试，CV < 10% 确认 L0 相变非层特异性。

---

## 下一迭代最高优先级

### P0 -- 阻塞项（零 GPU, ~3.25 小时）
1. 修复所有数据完整性错误 + 实现 validate_integration.py（1.5h）
2. 降级理论过度声明 + 添加 Bonferroni p（0.75h）
3. CMI partial correlation + leave-one-out sensitivity（0.5h）
4. 报告已计算的 threshold sensitivity results（0.5h）

### P1 -- 高价值实验（~3 GPU-hours）
5. Activation patching on 9 core words（1h GPU）
6. Tightened hedging classification（1h GPU）
7. CMI replication at L0=22 with perfect probes（1h GPU）

### P2 -- 写作和诊断（零 GPU, ~3.5 小时）
8. Control failure analytical diagnosis（1h）
9. 写作修复：冗余压缩、标题修改、摘要精简、notation 修复（1.5h）
10. CMI estimation diagnostics: bootstrap CIs, convergence, k-sensitivity（1h）

**完成 P0 后**: 预计 7.0 (Weak Accept)
**完成 P0+P1 后**: 预计 7.5 (Accept，取决于 activation patching 结果)
**完成全部后**: 预计 8.0 (Strong Accept)
