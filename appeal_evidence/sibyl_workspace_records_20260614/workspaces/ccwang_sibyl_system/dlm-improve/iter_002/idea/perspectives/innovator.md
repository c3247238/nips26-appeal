# 创新者视角

## 总判断

这一轮不该再发明一个更复杂的 remasking controller，而应该把“比较纪律本身”提升成贡献对象。真正有新意的地方，是把 DLM inference 论文里默认混在一起的三层因素拆开：`observer signal`、`controller policy`、`realized runtime stack`。

## 我主张保留的 3 个候选方向

### 候选 A：Protocol-First Diagnostic Paper

- 核心想法：把 `honest compute fairness + observer/controller split + asset lineage` 做成论文主轴。
- 新意来源：现有工作大多在提新 scheduler，很少把“结论为什么会被实现条件改写”当主贡献。
- 最关键产物：
  - `benefit_bucket_audit.json`
  - `seed_sensitivity_spotcheck.json`
  - `canonical_asset_manifest.json`
  - runtime fairness appendix

### 候选 B：Bucket-Mechanism Analysis

- 核心想法：把 revision 的效果拆成 `fixed / harmed / no-effect` 三类，并进一步对 reasoning / code / structure-sensitive samples 做细分。
- 新意来源：从 average gain 推进到 mechanism-aware failure taxonomy。
- 亮点：这能把当前 HumanEval boundary 和 GSM8K 正收益统一到一个更强解释框架里。

### 候选 C：Minimal Controller for Decoupling

- 核心想法：只引入“最小控制器”，目的不是追求 SOTA，而是构造一个更干净的实验，证明 `strong observer != guaranteed controller gain`。
- 允许形式：
  - observer-only ranking
  - fixed-budget threshold controller
  - matched-compute two-arm revision gate
- 禁止形式：重新包装成 “我们提出了更强方法”

## 排名

1. 候选 A
2. 候选 B
3. 候选 C

## 为什么不是新方法论文

- 文献已经显示 2025-2026 的主流都在 controller / scheduler 上堆技巧。
- 我们手里最独特的证据不是“某方法又赢 2pp”，而是“同一个 headline ordering 会被 realized pipeline 条件改写”。
- 这比再做一个 entropy 变种更难被替代。

## 对 planning 的直接建议

- 先把 A+B 当成联合主线，C 只作为可选补强。
- 不要再扩 benchmark 宽度优先；先把 GSM8K 主结果的证据封口做深。
- 所有新实验必须强制实装 batching / flash attention / compile / 多卡拆分，否则后面 bucket audit 会再次被速度拖死。
