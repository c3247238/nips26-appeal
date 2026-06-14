# 实用主义者视角

## 总判断

最值得做的是“最小新增实验、最大论文增益”的方案。我们当前已经有可投稿骨架，缺的不是更多故事，而是 4 个 reviewer 一眼会追问的封口件。

## 候选方向评估

### 候选 A：Protocol-First Diagnostic Paper

- 工程可行性：最高
- 需要新增的主要工作：
  - 统一 runtime metadata 采集
  - 生成 artifact manifest
  - 重写相关 appendix / method note
- 风险最低，因为最大程度复用现有结果资产。

### 候选 B：Bucket-Mechanism Analysis

- 工程可行性：高
- 需要新增的主要工作：
  - 逐样本对齐 draft / revised / oracle outcome
  - 分类 fixed / harmed / no-effect
  - 结合任务类型做 bucket summary
- 性价比很高，是当前最缺但也最容易补出的机制证据。

### 候选 C：Minimal Controller for Decoupling

- 工程可行性：中
- 主要风险：
  - 很容易滑回 method paper
  - 需要额外实现和 compute，可能又引入 fairness confound
- 只能做成 small add-on，不能做成主线。

## 排名

1. 候选 B
2. 候选 A
3. 候选 C

## 原因

- `benefit_bucket_audit` 是最直接的 blocker，对 paper score 的边际提升最大。
- `canonical_asset_manifest` 和 runtime appendix 能快速修复 reviewer 对 fairness/confound 的质疑。
- 新 controller 的 ROI 明显最低。

## 建议的执行顺序

1. 先产出 bucket audit。
2. 紧接着做 seed spot-check 和 manifest。
3. 最后视时间决定是否做最小 controller decoupling probe。
