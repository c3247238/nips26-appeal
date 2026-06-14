# 迭代 1 反思报告

## 本轮迭代总结

这一轮最重要的进展不是又多做了一个方法，而是把论文真正纠正成了它应该成为的样子：从旧的 method-forward / CARD hero 叙事，转成了 compute-normalized diagnostic study。这个方向变化不是包装，而是证据驱动的。`final_pareto_synthesis.json`、`diag_compute_curve_gsm8k.json`、`diag_signal_gap_audit.json`、`diag_math500_shortlist.json` 和 `diag_humaneval_guard_boundary.json` 现在共同支持的是一个更诚实也更稳定的主张：honest compute 会改变关键比较，observer quality 不等于 controller gain，而 revision 对任务结构高度敏感。

同时，这一轮把写作链真正收口到了可运行状态。LaTeX 模板、图表复制、protocol figure、bibliography 和 PDF 编译已经完成，`writing_latex` 不再是半成品阶段。`review` 阶段也成功形成了 critic / supervisor / codex 三路独立审查，并给出了相对一致的判断：论文方向是对的，但证据封口还不够。

## 各类问题分析

### ANALYSIS

1. **benefit-bucket audit 仍缺失（HIGH）**  
   这是当前最实质的证据缺口。论文已经在平均结果层面说明“revision 有时帮助、有时伤害”，但还没有把 fixed / harmed / no-effect 样本分离出来，因此机制性主张仍停留在 aggregate-only 层面。

2. **pilot-to-current shrinkage narrative 仍未显式化（MEDIUM）**  
   外部读者仍看不到“为什么 earlier pilot story 更像 method win，而 current story 更像 diagnostic study”的完整解释。虽然内部已经做了正确战略转向，但正文还没把这件事说透。

3. **honest-compute fairness philosophy 仍不够显式（MEDIUM）**  
   论文现在报告了 actual NFE、latency、batch、backend、compile，但还没有把“为何 actual NFE 是主归一化轴，以及如何看待 implementation-level confounds”讲成 reviewer-friendly 的方法学声明。

### EXPERIMENT

1. **seed-sensitivity 仍未封口（HIGH, RECURRING）**  
   关键对比依然缺最小不确定性检查。这已经不是第一轮被指出的问题，属于 recurring issue。

2. **cross-task evidence 仍偏薄（MEDIUM）**  
   GSM8K/MATH500 为 n=100，HumanEval 为 n=50。对于 boundary evidence 足够，但不足以支持强 regime 结论。

### WRITING

1. **observer-controller audit 定义仍偏隐式（HIGH）**  
   监督审查指出的核心问题很准确：paper 里说 calibration 是 strongest observer，但底层 score 实际来自 held-out correlation，而不是 reviewer 可能自然联想到的直接 calibration statistic。这个定义如果不写透，会被质疑为 metric conflation。

2. **markdown / LaTeX 双轨不同步（MEDIUM）**  
   `paper.md` 仍保留 protocol figure TODO，而 `latex/main.tex` 已有正式 Figure 2。这会让 review trail 看起来像 pipeline 没同步好。

### PIPELINE

1. **版本目录卫生问题仍在（MEDIUM）**  
   用户多次关心为什么没有进入 `iter_2`，以及为什么没有把旧内容归档到 `iter_000`。从控制面看，本轮尚处 iteration 1 的 reflection，因此没有自然进入 `iter_002`。但这个版本卫生问题确实应该在下一个真实迭代边界解决，而不是继续拖延。

### EFFICIENCY

1. **batching / flash-attn 优化再次未落地（HIGH, RECURRING）**  
   这是最明显的 recurring issue。项目记忆里已经明确要求“下一轮必须应用”，但在本轮中依然没有真正先做。它不是边缘问题，而是直接卡住了 seed spot-check 和更大样本机制分析的根因。

## 修复追踪

由于 `prev_action_plan.json` 缺失，本轮的 FIXED / RECURRING / NEW 主要依据上一轮保留下来的 `reflection/lessons_learned.md` 推断。

### FIXED

- 旧的 method-forward / Calibration-Aware 论文定位已经被修正。
- LaTeX 与正式 PDF 打包已完成，不再是“图表和附录未完成、无法提交”的状态。
- HumanEval 负面结果已进入主叙事，不再被继续淡化。

### RECURRING

- 单 seed / 缺 uncertainty closure
- batching / efficiency optimization 未先行落地

### NEW

- benefit-bucket audit 缺失成为新的第一优先级 blocker
- observer-controller audit 的定义性风险被审查显式暴露
- iter_000 / iter_002 的版本目录卫生需要在迭代边界处理

## 质量趋势判断

**Improving。**

相较上一轮保留下来的 6.0/10 method-paper 风险图景，本轮已经把论文转成更稳的 diagnostic framing，并完成 LaTeX 收口。当前 supervisor score 仍为 `6.4/10`，未达到通过线，但这是“方向已对、证据待补”的 revise，而不再是“核心叙事错误”的 revise。

## 资源效率评估

`exp/gpu_progress.json` 仅记录了 completed task 列表，没有 timings，因此无法精确计算 GPU utilization 或 idle minutes。这本身就是一个效率分析盲点。

尽管缺少 timings，瓶颈依旧明确：

1. 实验阶段仍是主要瓶颈；
2. batching / flash-attn / compile 未先行实现，意味着后续任何 seed 或 bucket 工作都会再次被速度约束；
3. 可以并行的小型验证任务（seed spot-check、bucket audit、artifact appendix）尚未按小任务策略拆开。

### 调度改进建议

- 下一轮先做 `P0 = batching optimization`，否则所有实验性收尾都会被再次拖慢。
- 把 seed spot-check 设计为最小集合任务，而不是重新开整轮大实验。
- 在运行时日志中补 timings；否则 reflection 无法给出真正量化的效率复盘。

## 根因分析

本轮的根因已经和上一轮不同。

上一轮的主要根因是 **论文定位错误**。  
这一轮的主要根因变成了 **证据封口不足**。

具体来说：

1. 战略层已经纠正，front-runner 与证据一致；
2. 工程收口（LaTeX）已经完成；
3. 现在卡住质量门的是“最后一层诊断 deliverables 没真正做出来”，尤其是 benefit buckets、seed spot-check、protocol appendix 和 asset-lineage。

这意味着下一轮不应该再大规模发散，而应该围绕这些封口项做集中补强。

## 系统自检响应

`logs/self_check_diagnostics.json` 不存在，本轮无需额外回应。

## 成功模式提取

1. **证据驱动 pivot 成功**  
   当方法主线站不住时，没有强行硬写，而是切换到 diagnostic paper，这是本轮最大的成功。

2. **三路 review 非常有效**  
   critic、supervisor、independent codex 三个视角高度互补，而且都把问题收束到“证据封口”而非“方向错误”。

3. **LaTeX 收口质量明显提升**  
   这一轮真正把 paper package 拉到了可评审状态，为下一轮做 focused revision 提供了稳定基础。

## 下一轮建议优先级

1. `P0`: 先做 benefit-bucket audit
2. `P1`: 做 seed-sensitivity spot-check
3. `P2`: 写 observer-controller protocol note
4. `P3`: 补 runtime fairness / asset-lineage appendix
5. `P4`: 同步 `paper.md` 与 `latex/main.tex`
6. `P5`: 在下一个真实 iteration 边界处理 `iter_000` 归档和 `iter_002` 目录卫生
