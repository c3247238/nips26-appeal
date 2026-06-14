# 本轮反思报告

## 本轮总结

这一轮最重要的变化不是再做出一个更大的 headline，而是把 iteration 4 的论文对象彻底收紧成一个 reviewer-safe 的 bounded contribution。`paper.md`、`review.md`、`writing/latex/main.tex` 和 `writing/latex/main.pdf` 现在已经在同一条叙事线上：`cand_espd` 的 full-scale GSM8K 结果支持一个关于 entropy-routed compute 的有限机制性结论，但不支持 broader benchmark-dominance claim。

质量上，本轮已经明显优于上一轮。终审从 6.8 提升到 7.3，supervisor review 给出 7.6，说明 submission-facing packaging gap 基本被补上。推动这一提升的关键动作不是新实验，而是三类 reviewer-facing 修补：`primary endpoint`、`claim boundary`、`runtime lineage / artifact-release`。

## 问题分类

### SYSTEM

- Feishu/Lark 同步仍被本地 OAuth refresh `invalid_grant` 阻塞，虽然 registry 已安全，但 pending queue 继续积累。

### EXPERIMENT

- 当前 full-scale 证据仍只覆盖 GSM8K；缺少外部 benchmark。
- `cand_espd` vs `ESPD-FixedFrontier` 的 sham 仍是 partially matched，机制解释仍未完全封口。

### WRITING

- reviewer-facing packaging gap 已大幅修复，但 proposal object-line front-runner 与 final paper speed-line mainline 的转向还可以再写得更显性。

### ANALYSIS

- strongest claim 依然是 bounded attribution signal，而不是 clean mechanism proof。
- uncertainty treatment 诚实但偏轻量，更多是在阻止 overclaim，而不是支撑强 superiority。

### EFFICIENCY

- `exp/gpu_progress.json` 没有 timings，无法对 GPU 利用率和 idle gap 做可信复盘。
- 背景同步在收尾阶段反复因认证失败而重试，形成非研究性阻塞。

## 修复追踪

### FIXED

- runtime-lineage artifact 已进入正文和 Figure 4，不再只是 reviewer 需要手动拼的隐式信息。
- artifact-release / replication statement 已补入正文。
- References 与真实 Figure 1 已补齐。
- LaTeX、BibTeX、图表复制与 PDF 编译链全部跑通。

### RECURRING

- sham 仍未 fully matched。
- external validation 仍缺失。

### NEW

- Feishu OAuth refresh token 已失效，后台同步必须先重新授权。
- GPU telemetry 仍不完整，导致效率分析只能做定性判断。
- proposal-to-paper 主线转移需要更显性的 reviewer-facing 说明。

## 质量趋势判断

本轮质量趋势可以判为 `improving`。最直接的证据是：

- final review 从 6.8 提升到 7.3；
- supervisor review 为 7.6；
- LaTeX 与 PDF 收尾没有再引入新的叙事性倒退。

这说明当前 package 已经从“还在修补逻辑缺口”进入“围绕剩余证据缺口做精修”的阶段。

## 资源效率评估

`exp/gpu_progress.json` 只记录了 completed task 列表，没有 timings，因此无法给出可信的 GPU 利用率百分比或 idle 分钟统计。也正因为这个缺口，本轮效率分析的主要结论不是“GPU 用得怎么样”，而是“我们缺少足够的 telemetry 去判断 GPU 用得怎么样”。

能确认的事实是：

- 本轮实验任务已经全部完成，没有新的 GPU 侧运行阻塞；
- 当前瓶颈已经从 experiment compute 转移到 writing/review polish 与 background sync auth；
- 若下轮继续做 full benchmark follow-up，必须优先补全 timings 和 idle gap 记录，否则无法判断多 GPU 调度是否真的达到项目约束中的高利用率目标。

## 根因分析

1. 当前剩余问题的核心根因不再是“没有结果”，而是“最强结果仍然不够完全归因”。
2. 后期 writing quality 的提升主要依赖 reviewer-facing scope discipline，而不是继续拓展 candidate story。
3. 系统侧辅助链路在研究收尾时再次成为瓶颈，说明自动化系统在长期运行后的认证恢复与 telemetry 完整性仍需补强。

## 系统自检响应

当前未发现 `logs/self_check_diagnostics.json`，因此没有额外的自检诊断需要逐项响应。下一轮如果继续以系统可靠性和效率为重点，建议把自检结果作为标准 artifact 固化下来。

## 下一轮重点

1. 保持当前 bounded framing，不要重新膨胀成 broader methods claim。
2. 若继续 research iteration，优先补 stronger sham 或 routing/stopping split。
3. 其次补一个外部 benchmark，提升 external validity。
4. 修复 Feishu OAuth 与 GPU telemetry，减少非研究性阻塞。

