# 本轮迭代教训

## 必须改进

- [sham 仍未 fully matched]: 下一轮最值得花实验预算的不是新候选，而是 stronger sham 或 routing/stopping split ablation。
- [external validation 仍缺失]: 当前 full-scale 只覆盖 GSM8K，若还要继续提升 package，优先补一个外部 benchmark。
- [Feishu OAuth 已失效]: 进入下一轮前先完成浏览器重新授权，否则后台同步会持续空转。
- [效率 telemetry 不完整]: `exp/gpu_progress.json` 必须记录 timings 与 idle gap，否则所有效率复盘都不可信。

## 需要注意

- [bounded framing 是当前最强护城河]: 这篇 paper 的价值来自 attribution discipline，不来自 benchmark domination。
- [proposal 到 final paper 的主线转移要写透]: `cand_bsr` 到 `cand_espd` 的排序更新必须对 reviewer 明说，避免 narrative selection 质疑。
- [equal-quality speed 不能脱离 wall-clock 单独讲]: 否则 reviewer 会把它读成被重定义过的速度优势。

## 做得好的（继续保持）

- [evidence-first 收缩叙事]: 这轮没有再强撑旧主线，最终 paper 与证据对齐了。
- [reviewer-facing 三件套有效]: `primary endpoint`、`claim boundary`、`runtime lineage` 是本轮最有价值的写作修复。
- [LaTeX 链路跑通]: 模板、BibTeX、图和 PDF 已不再是风险项。
- [主线不停顿]: 从 writing_integrate 到 final review、latex、review、reflection 一路推进，没有在状态汇报处停机。
