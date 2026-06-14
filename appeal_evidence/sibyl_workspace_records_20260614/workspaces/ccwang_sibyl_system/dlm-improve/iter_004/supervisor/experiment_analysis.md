# 实验结果分析

## 实验结果概要

- 当前 full-scale 主结果来自 `cand_espd`
- `GSM8K` 全量 `1319` 样本上：
  - `cand_espd`: accuracy `0.4041`, equal-quality speed `124.42 tok/s`
  - `ESPD-FixedFrontier`: accuracy `0.3988`, equal-quality speed `105.73 tok/s`
  - `CARD-84`: accuracy `0.3965`, speed `126.08 tok/s`
  - `RAND-84`: accuracy `0.3980`, speed `128.00 tok/s`
- bundle verdict 为 `ADVANCE`
- 当前最关键的机制证据是：在相同 `frontier_ratio=0.1211` 下，fixed-frontier sham 无法复现 candidate 的质量/速度组合

## 各方观点总结

- 乐观者：`cand_espd` 已从 speed-line backup 升级为 full-scale 支持的 serious line；当前最值得珍惜的是 candidate 相对 sham 与 shared controls 的小而真实的优势。
- 怀疑论者：当前增益幅度小、统计不稳、sham 仍不够 matched，不能把结果写成 mechanism breakthrough。
- 战略家：应当 `PROCEED`，但必须是 narrowed proceed；先把 `cand_espd` 做成一条 bounded speed-line，再决定 object-level line 是否重新上位。

## 分析

### 1. 方法可行性

核心方法已经按“当前最小可发表标准”工作起来了。`cand_espd` 至少满足两点：

1. 没有相对 shared controls 发生质量崩塌
2. 相对 matched fixed-frontier sham 保留了质量与速度双重优势

这说明当前方向不是伪阳性空壳，而是一条真实存在的可继续优化线路。

### 2. 性能表现

性能表现属于 **弱到中等强度的正信号**，不是压倒性胜利。

- 优于 `RAND-84` 与 `CARD-84`，但幅度只有 `+0.61pp ~ +0.76pp`
- 绝对吞吐并不高于 shared controls
- 但相对 fixed-frontier sham 的 `+18.69 tok/s` 与 `+0.53pp` 很关键

因此，最诚实的表述应是：

> entropy-routed compute 在当前 unified runtime contract 下展现出可信但幅度不大的 trade-off gain。

### 3. 改进空间

当前方向有明确、可执行、低风险的改进路径：

1. 做一个关键外推验证：第二个 benchmark 或第二个 DLM
2. 补 reviewer-friendly runtime-lineage artifact
3. 做 routing/stopping 的拆分 ablation
4. 对 fixed-frontier sham 做更 matched 的实现

这些改进都不要求重新发明方法，只要求把现有信号做扎实，因此继续优化的边际收益是可预期的。

### 4. 时间成本

继续当前方向明显比重新开始更高效。

- 若现在 `PIVOT`，将放弃 iteration 4 中唯一已经拿到 full-scale backing 的 serious line
- 若继续 `cand_espd`，只需补 1 个外推验证、1 份 runtime artifact、1 组机制拆分实验，就可以判断这条线是否具有 submission-grade 稳定性

从时间成本看，这是一个典型的“先把已有正信号做扎实，再决定是否重开新主线”的局面。

### 5. 怀疑论者的批评是否致命

不是致命，但必须被显式吸收进后续计划。

怀疑论者指出的几项问题都成立：

- 效果幅度小
- 显著性不足
- fixed-frontier sham 仍不够 matched
- 缺少跨任务验证

但这些问题的性质是 **限制 claim scope**，不是直接否定当前方向。  
只要后续阶段把 claim 收窄，并优先补最小关键缺口，这些问题不会强迫当前项目回到 `PIVOT`。

## 决策理由

我选择 `PROCEED`，原因如下：

1. 当前已经出现真实的 full-scale 正信号，且该信号通过了最关键的 sham-control 检查。
2. 该信号虽弱，但有明确、低成本、可执行的验证与补强路径。
3. 继续推进的时间成本明显低于重新开始。
4. 怀疑论者提出的是“先缩口径、再补证据”，而不是“当前主线已被否定”。
5. 当前最该做的不是换方向，而是把 proposal 的排序与最新证据对齐：`cand_espd` 升为当前主线，`cand_bsr` 下调为 challenger。

## 后续执行约束

1. 所有 paper-facing 叙事必须改为 bounded contribution，不得写成 benchmark-level breakthrough。
2. 在进入更大规模 narrative 扩张前，必须优先完成：
   - 一个外推验证
   - 一份 runtime-lineage artifact
   - 一个 routing/stopping 拆分 ablation
3. `cand_bsr` 只保留低成本 continuation，不开启新的 full-scale object-line。

DECISION: PROCEED
