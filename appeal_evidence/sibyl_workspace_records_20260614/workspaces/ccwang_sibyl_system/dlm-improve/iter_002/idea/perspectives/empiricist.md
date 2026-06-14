# 实验主义者视角

## 总判断

从证据角度看，最值得推进的是“补齐当前结论最短板的实验”，而不是再开新 hypothesis。当前短板非常明确：bucket、seed、runtime lineage。

## 候选方向

### 候选 A：Protocol-First Diagnostic Paper

- 证据价值：高
- 需要的数据：
  - 每个 headline result 的 runtime metadata
  - 统一的 asset-to-claim 映射
- 能直接降低 reviewer 对 fairness confound 的攻击面。

### 候选 B：Bucket-Mechanism Analysis

- 证据价值：最高
- 需要的数据：
  - per-sample draft correctness
  - per-sample revised correctness
  - revision 是否修复/伤害/无效
- 这是从 aggregate result 走向 mechanism claim 的最短路径。

### 候选 C：Minimal Controller for Decoupling

- 证据价值：中
- 如果做，必须限定成 very small spot-check：
  - matched-compute
  - same runtime path
  - 只为验证 observer/controller split，不为追 headline gain

## 排名

1. 候选 B
2. 候选 A
3. 候选 C

## 对 planning 的硬要求

- 先做 bucket audit，再做 seed spot-check。
- 任何新增实验必须先完成 batch-size 探测并尽量吃满显存。
- 如果无法在短期内扩 cross-task sample，就主动收紧语言，不再扩 claim。
