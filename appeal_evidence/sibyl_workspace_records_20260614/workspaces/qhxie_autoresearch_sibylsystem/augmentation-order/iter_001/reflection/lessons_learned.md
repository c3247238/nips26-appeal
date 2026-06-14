# Lessons from Iteration 0

## Must Improve

- **运行 Full Tier 1 是唯一的阻塞性任务**：在 Full Tier 1（6 orderings × 2 arch × 2 datasets × 5 seeds × 200 epochs，约 20 GPU-h）完成并产出 paired t-tests + Bonferroni 校正结果之前，不得撰写任何假设判定，不得提交论文，不得移动到下一迭代阶段。Pilot 结果只能用于"方向性信号"，不能用于假设确认/证伪。

- **Pilot 不等于实验完成**：`gpu_progress.json` 标注 "completed" 的任务若处于 pilot 模式，只意味着 pilot pass criteria 满足，**不**意味着可以推进到写作。写作阶段的触发条件必须是 full-scale 实验完成，不是 pilot 完成。

- **分析脚本必须拒绝在 n=1 时产生统计量**：Cohen's d、配对 t 检验、Bonferroni 校正、假设 "confirmed/falsified" 判定——这些都需要 n>=2 seed 的数据。分析脚本应在 n=1 时返回 null 或显式错误，而不是产生数值上可信但实际无效的统计量。

- **NC_2 代理必须在足够规格下计算**：SWD 在 3072 维空间中需要 10k+ 样本和 1000+ 随机投影才能产生可靠估计。100 样本/100 投影的结果与随机数无区别。H3 判定应更改为"inconclusive (proxy underpowered)"直到完整规格计算完成。

- **InfoNCE MI 估计必须使用收敛质量的编码器**：使用 10 epoch 随机质量编码器的 MI 估计不承载信号。在 Full Tier 1 完成后，使用 epoch 100–200 checkpoint 重新运行 Tier 4b。

- **理论叙事必须重建**：论文不能将已被证伪的 NC_2 和 DPI 框架作为正面贡献宣传。正确框架：以实证发现为主线（排序影响精度；ViT 比 CNN 更敏感；交错 > 块状），将 NC/DPI 定位为"测试并发现不足的理论候选"，将像素空间分布测量无法预测优化中介学习结果定为新颖理论发现。

- **假设编号必须锁定**：H2 在 pipeline 中途从"架构差异敏感性"被重定义为"可逆性排序获胜"，预注册的 ANOVA 测试从未执行。假设定义应在 proposal 阶段锁定，任何修改必须显式版本化并在所有 artifact 中一致更新。

- **Tier 3 best/worst 选择标准必须在实验前预定义**：从单一 block（CIFAR-100/ResNet-18）选出的"最佳"排序在其他 block 中可能是"最差"。应基于所有 Tier 1 block 的聚合排名来定义 best/worst，并在 Tier 3 启动前将选择标准写入 task_plan。

## Watch Out

- **Tier 2 的 9.01 pp 差异极易缩水**：在 50 epoch 确认 Pilot 之前，这个数字不能作为任何实践建议的依据。在 Tier 2 确认 Pilot 结果出来前，Introduction 和 Discussion 中的 9.01 pp 引用必须添加"pilot-scale, single seed"限定语。

- **M14 下两个排序精度完全相同（0.245）是可疑信号**：需要检查是否存在训练崩溃、RNG 状态意外共享、或极端增强导致训练退化。在 Full Tier 3 启动前，先验证 M14 pilot 的训练曲线和损失值是否独立收敛。

- **ResNet-18/CIFAR-10 和 ViT/CIFAR-100 的 Pilot block 是无效的**：这两个 block 的精度在随机猜测基准附近，任何排序差异都无法被解释为信号。在报告 H1 支持的 block 数量时，只有精度显著高于随机猜测的 block 才应被计入。

- **Introduction 现在时态声明是高风险**："We find that ordering produces accuracy spreads up to 2.32%" 在 Pilot 阶段呈现为已确立事实。在 Full Tier 1 完成之前，所有这类声明都应替换为"In our pilot study, we observe..."。

- **RandomHorizontalFlip 的可逆性分类需要明确定义**：当前分类（"medium reversibility"）与对其"perfectly invertible"的描述矛盾。在 Method 章节修订时，必须选择信息论可逆性（Flip 应为 high，eta=1）或任务相关可逆性（需要新的形式化定义），并保持全文一致。

## Keep Doing (Success Patterns)

- **继续诚实报告负面结果**：将 NC_2 失败（rho=-0.20）和 H5 证伪（非单调量级效应）明确呈现，这本身是有价值的科学贡献，是论文诚信的有力体现。

- **保留预注册假设设计**：每个假设有明确的指标、阈值和证伪标准——这是本项目的核心优势，继续在 Full Tier 1 结果上应用 pre-registered 分析协议（paired t-test, Bonferroni, ANOVA）。

- **保留四层实验结构**：Tier 1→2→3→4 的层次覆盖逻辑合理，在 Full 实验中继续沿用，必要时根据 Tier 2 确认 Pilot 的门控决策调整 Tier 2 预算。

- **保留假设判定汇总表**：Table 3（H1-H5 vs 预注册阈值 vs 观测值）是强有力的组织工具，在 Full Tier 1 结果基础上更新后继续保留。

- **保留"零成本超参数"框架**：这是传达排序研究实践价值的有效钩子，在 Discussion 的实践建议部分（以 Full 实验结果为依据后）继续使用。

- **保留配对 Seed 设计**：在 Full Tier 1 中严格执行配对 seed（每个排序使用相同 seed 集合），这是执行配对 t 检验和降低 within-seed 方差的关键设计。

- **继续并行化独立实验任务**：Tier 1 的四个子任务完全独立，在 4 个 GPU 上并行运行可将墙钟时间从 ~40h 缩短至 ~10h。
