# pragmatist round1 summary

## 总判断

这一轮我建议保留的不是“更多候选方法”，而是一个极小的 candidate pool，专门服务于最短补证据路径。标准只有两个：

1. 能不能直接补当前 paper 的硬缺口
2. 能不能在不引入新 fairness confound 的前提下快速落地到 planning / pilot

## 建议保留的 candidate pool

### Pool A：Bucket-First 机制补证据

- 来源整合：`pragmatist + empiricist + contrarian`
- 保留理由：
  - 这是当前对 paper score 提升最大的单项补件
  - 不需要新模型、不需要新 controller、主要是离线分析和样本级对齐
  - 能把 aggregate story 变成真正的 mechanism evidence
- 执行要求：
  - 第一批只做 GSM8K headline main pair
  - 必须带代表样本，不接受只给 aggregate count
  - 写法上不要过早扩成 full failure taxonomy

### Pool B：Minimal Seed Spot-Check

- 来源整合：`empiricist + pragmatist`
- 保留理由：
  - 这是最便宜的质量门封口
  - 目标不是做 full replication，而是检查 headline delta 的方向稳不稳
  - 对 planning 很友好，可以直接做成一个小实验包
- 执行要求：
  - 只做 headline pairwise comparison
  - 只做最小 `>=100` 样本、`3 seeds`
  - 强制走现有 batched runner，不再接受默认慢路径

### Pool C：Runtime Fairness / Asset-Lineage Artifact

- 来源整合：`innovator + theoretical + pragmatist`
- 保留理由：
  - 这是 honest compute 叙事能否站稳的 reviewer-facing 资产
  - 它不一定最“亮”，但最能减少 fairness confound 质疑
  - 同时还能让后续 planning/pilot 有统一 runtime contract
- 执行要求：
  - 第一版只覆盖 shortlist 核心方法
  - 必须绑定 canonical source JSON
  - 目标是 appendix-grade artifact，不是再写一段抽象原则

## 不建议单独保留的方向

### DROP：任何独立成线的 Minimal Controller 候选

- 原因不是它完全没价值，而是当前 ROI 太低
- 只要它进入 candidate pool，就很容易重新把讨论拖回 “再做一个 trick”
- 除非前面三个 pool 都完成且还有算力余量，否则不应进入本轮 planning 主线

## 我对其他视角稿的综合吸收

- `contrarian` 提醒了边界：不要把 bucket 写得过满，不要把小 slice 写成稳定规律
- `empiricist` 给了最稳的证据优先级
- `innovator` 提供了论文层面的新意解释，但执行上必须去掉 C 的主线地位
- `interdisciplinary` 的测量学 framing 可以保留到写作层，不该抢 planning 主线
- `theoretical` 的 protocol object 划分很有用，但必须放在 artifact 落地之后，而不是之前

## 最终建议的执行顺序

1. `Bucket-First 机制补证据`
2. `Minimal Seed Spot-Check`
3. `Runtime Fairness / Asset-Lineage Artifact`

## 一句话结论

本轮只该保留 2-3 个直接补缺口的候选池：`bucket audit`、`minimal seed spot-check`、`runtime fairness manifest`；任何新 controller 都不应进入主线，因为它会破坏最短补证据路径。
