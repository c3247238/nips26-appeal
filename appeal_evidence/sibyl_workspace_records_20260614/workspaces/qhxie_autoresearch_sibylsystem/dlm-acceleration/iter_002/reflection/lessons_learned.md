# Lessons from Iteration 2

## Must Improve（下次迭代必须改正）

- **[EXPERIMENT] Pairwise composition 实验必须在 full-scale 运行（N>=1319, 3 seeds）才可作为论文核心证据**: 两个迭代都将 pairwise 停留在 pilot 规模（iter_001: 200/2-seed, iter_002: 100/1-seed）。iter_001 M1+M3 的 pilot/full 4.4x Ortho 差异证明 pilot 估计不可靠。规则：pairwise Ortho 声称需要 N>=500 且 >=2 seeds；基于 N<500 的声称必须包含 bootstrap CIs 并标记为 "pilot estimate"。

- **[ANALYSIS] IGSD 贡献必须诚实反映 null ablation 结果**: 两次迭代均确认 tau=0.0（无置信度门）与 tau=0.9 在 T_draft=32 产生相同准确率（49.5%）。摘要和贡献项必须从"基于 KL 散度的自适应步骤调度器"改写为"简单的 draft-then-truncate 方案；置信度分区机制在主要操作点不提供可测量的准确率改善"。将 null 结果作为 finding 报告，而非隐藏。

- **[EXPERIMENT] M1 涉及的所有 Ortho 声称必须附带 1.16x speedup caveat**: M1 是 no-op。M1+IGSD Ortho=0.96 衡量的是"加入 M1 是否伤害 IGSD"，而非"两种加速方法正交组合"。每次提到 M1+IGSD composition 时必须附加："M1 贡献 1.16x 实测加速（本质为开销）。"

- **[EXPERIMENT] M3 speedup 测量必须分离 per-token speed 和 output-length effect**: M3 原始 TPS ~50 vs baseline ~58.5，per-token 更慢。1.68x 可能是输出长度缩减效果。报告时需同时给出：(a) per-token latency, (b) 平均输出长度, (c) per-sample wall-clock 时间。

- **[PLANNING] 关键 ablation 必须前置到 Phase 1**: IGSD vs naive truncation 是 IGSD 贡献的基础对比，不应在 Phase 7 才运行。规则：每个声称有算法创新的方法，其 null 对比（创新 vs 最简基线）必须在任何 downstream 实验之前完成。

- **[PLANNING] Baseline TPS 必须在 Phase 0 标准化**: 单一规范化测量，固定协议（generation-only, post-warmup, batch=1），写入共享配置文件。所有 speedup 计算使用同一参考。

## Watch Out（下次迭代需警惕）

- **Ortho 指标在 QAS 接近 1.0 时退化**: 当一个方法的 QAS ~1.0 时，Ortho 本质上衡量的是"添加这个 no-op 是否伤害另一个方法"，不是有意义的 composition 分析。考虑设置 QAS > 1.2 的最低门槛。

- **M3 AccRet > 100% 可能是 baseline 选择假象**: 103.9% 使用 3-seed 均值 baseline（71.2%），但 same-seed pilot baseline 可能是 73%，这意味着 AccRet 实际为 ~100%。总是使用 same-seed, same-subset baseline 报告 AccRet。

- **Dream-7B AccRet=125% 因 36% 低 baseline 膨胀**: 不是"可迁移组合模式"的证据，而是"Dream-7B 晚期去噪步骤有害"的证据。分开报告两个模型的结论。

- **MATH500 baseline 11.1% 使所有 MATH500 指标不可靠**: 3 个样本的波动在 100 样本评估上改变 AccRet ~27%。将 MATH500 降级为补充指标，GSM8K 作为主要指标。

- **三方组合 CV 8%**: 每 seed Ortho 范围 [0.91, 1.11] 跨越"近正交"和"协同"两个分类。0.8/1.0 的分类阈值粒度过细。报告连续 Ortho + CI 而非分类标签。

- **Batch sensitivity 无 batch 特定 baseline 不可解读**: 永远不要在没有对应 baseline 的情况下报告加速数据。

- **编辑遗留物和引用占位符是提交阻断项**: 两次迭代都未完成 Figure 2 和 [CITE:xxx] 替换。必须在下次迭代中作为 blocking task 优先处理。

## Keep Doing（下次迭代保持）

- **诚实报告负面结果**: M2 NO_GO、d2Cache 15.2x 失败、AR gap（QAS 3.08 vs 1.07）、IGSD null ablation、M1+M3 干扰——全部透明报告。Supervisor 和 Reviewer 均赞赏。这种知识诚信建立可信度，是论文最大的非技术优势。

- **实验覆盖面全面**: 15 个实验组的因子设计是结构完善的。覆盖面不是问题——深度才是。

- **双轮 Supervisor + Critic 独立审查**: 两轮独立审查有效检测关键问题（iter_001 round 2 发现 alpha=0.52；iter_002 Supervisor 和 Critic 独立发现 IGSD 门、M1 no-op、pilot 规模三个核心问题）。

- **CHR_refine 从分析推导升级为实测验证（0.943）**: 消除数据伪造疑虑。继续对每个定量声称使用实测数据。

- **QAS 公式统一**: iter_001 的 0.5x 隐藏惩罚已消除。所有方法使用相同的 QAS = Speedup x AccRet。维护此一致性。

- **Dream-7B 跨模型验证成功**: 定性模式一致（M1+IGSD 最优、M3 反效果）。跨模型转移的方向性结论可信，但量级需分开报告。

- **Well-structured related work**: Table 1 对已发表 DLM 加速协议的比较是对社区有用的独立贡献。
