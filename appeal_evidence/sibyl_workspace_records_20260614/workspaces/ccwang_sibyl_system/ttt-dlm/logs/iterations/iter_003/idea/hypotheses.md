# 可检验假设

## 核心假设（DTA 方法）

### H1：DTA 在 Countdown 上显著有效
- **陈述**：DTA 在 Dream-7B-Instruct 上的 Countdown 准确率显著优于 vanilla 去噪
- **预期效应量**：+5-10pp（绝对提升）
- **检验方法**：McNemar test，α=0.05，500 题 x 3 seeds
- **证伪条件**：如果 DTA 的准确率提升 < 2pp 且 p > 0.1，则 H1 被证伪——参数级推理时适应对 DLM 去噪无效
- **预期结果**：DTA 通过在去噪过程中积累已揭示 token 的模式信息（如 Countdown 中的数字约束关系），使后续步骤的预测更准确

### H2：DTA + Remasking 组合优于纯 Remasking
- **陈述**：DTA + ReMDM-conf 在 GSM8K 上显著优于纯 ReMDM-conf
- **预期效应量**：+3-5pp
- **检验方法**：Paired McNemar test，1319 题 x 3 seeds
- **证伪条件**：如果组合方法的准确率 < 纯 ReMDM-conf + 1pp，则 DTA 与 remasking 不互补
- **预期结果**：DTA 提供参数级"记忆"使 remasking 后的重预测更有信息，两者正交互补

### H3：DTA 的推理时扩展曲线不饱和
- **陈述**：DTA 的准确率随去噪步数 T 的增加持续改善，不出现 remasking 在 T > 2L 后的饱和现象
- **预期结果**：在 T = 64, 128, 256, 512 步的扫描中，DTA 的准确率呈现对数增长趋势，而 ReMDM-conf 在 T > 256 后趋于平坦
- **证伪条件**：如果 DTA 的准确率在 T > 2L 后也饱和（与 ReMDM-conf 相同的饱和行为），则 H3 被证伪——参数适应未提供超越 remasking 的扩展能力
- **理论预测**：DTA 的收益来自参数空间的梯度优化，理论上具有指数收敛保证（VDTA 命题），不同于 remasking 在 token 空间的离散搜索

## 消融假设（信息增强谱系）

### H4：跨步信息传递的价值单调递增
- **陈述**：信息增强谱系中，Level 3 (DTA) > Level 2 (SCP) > Level 1 (DMI) > Level 0 (vanilla) 在准确率上严格递增
- **检验方法**：在 Countdown 上四个 level 的 pairwise McNemar test
- **证伪条件**：如果任意相邻 level 之间无显著差异，则该 level 的额外信息传递未带来价值
- **特别关注**：如果 Level 1 (DMI, ~0 额外开销) 的效果接近 Level 3 (DTA, ~2.5x 开销)，则 DMI 是更优的工程选择

### H5：DMI 的 logit carry-over 改善 remasking 后的预测质量
- **陈述**：被 remask 的 token 在使用 DMI（带前步 logit 信息的软 embedding）时的预测准确率高于使用纯 mask embedding
- **检验方法**：token 级分析——对比 remask 位置在有/无 DMI 时的 top-1 预测正确率
- **证伪条件**：如果 DMI 的 token 预测准确率 < 纯 mask + 1pp，则 logit carry-over 未提供有用信息

### H6：SCP 的自矛盾检测比置信度更精准地定位错误 token
- **陈述**：SCP 的 Correction Precision（被标记为自矛盾的 token 中确实是错误的比例）高于 ReMDM-conf 的 Correction Precision
- **检验方法**：在 Countdown 500 题上计算两种方法的 Correction Precision 和 Recall
- **证伪条件**：如果 SCP 的 Correction Precision < ReMDM-conf 的 Correction Precision，则 leave-one-out probing 不比静态置信度更准确

## 理论假设

### H7：DLM 去噪过程中存在可测量的信息积累
- **陈述**：DTA 的 LoRA 参数 Δθ 关于目标序列 x_0 的互信息 I(Δθ^(t); x_0) 随去噪步 t 单调增加
- **检验方法**：用 MINE（互信息神经估计器）或变分下界估计 I(Δθ^(t); x_0) 在不同 t 的值
- **证伪条件**：如果 I(Δθ^(t); x_0) 在 t 增大过程中出现下降或停滞，则信息积累假设不成立
- **替代检验**：测量 DTA 模型在步 t 对"未来将揭示的 token"的预测准确率是否随 t 提升

### H8：DTA 的 LoRA 参数衰减因子 γ 存在最优范围
- **陈述**：γ 过小（<0.8）导致遗忘过快，γ=1.0 导致参数漂移，最优 γ 在 [0.9, 0.99] 范围内
- **检验方法**：γ 消融实验（0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0）
- **证伪条件**：如果 γ=1.0（无衰减）的效果最好，则遗忘机制不必要；如果 γ=0.0（每步重置）最好，则跨步累积无价值

## 诊断假设

### H9：当前 remasking 方法的 Correction Precision 低于 50%
- **陈述**：ReMDM-conf 在 Countdown 上 remask 的 token 中，不到一半是真正的错误 token
- **检验方法**：记录每步 remask 的位置，与最终正确答案对比
- **预期结果**：大量"正确但低置信"的 token 被错误地 remask，浪费了计算资源
- **意义**：如果 H9 成立，则说明所有基于内部信号的 remasking 方法存在根本性的信号质量问题，DTA 通过参数适应而非 token 选择来规避这一问题

### H10：DTA 不引入文本退化
- **陈述**：DTA 生成的文本在 n-gram 重复率和词汇多样性上不劣于 vanilla 去噪
- **检验方法**：对比 DTA 和 vanilla 生成的 distinct-1/2/3 和 rep-2/3 指标
- **证伪条件**：如果 DTA 的 rep-3 > vanilla + 20%，则参数更新导致了模式坍缩
- **意义**：前 18 轮的核心教训——PPL 改善可能由退化驱动。H10 确保 DTA 的准确率提升是真实的
