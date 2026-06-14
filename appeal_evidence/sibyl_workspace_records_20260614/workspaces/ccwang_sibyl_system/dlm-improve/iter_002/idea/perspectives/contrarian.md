# 反对者视角

## 总判断

如果我们不克制，本轮最容易犯的错有两个：

1. 把 diagnostic paper 又偷偷写回 method paper  
2. 在证据还不够厚时，把 “failure taxonomy” 写得过满

## 我对三个候选的质疑

### 候选 A：Protocol-First Diagnostic Paper

- 风险：容易停留在“我们报告得更诚实”这种看起来不够强的层面。
- 过关条件：必须拿出清晰、可复用、可审计的 artifact，而不是只写几段 discussion。

### 候选 B：Bucket-Mechanism Analysis

- 风险：如果 bucket 只做到 fixed / harmed / no-effect 三分，而没有和任务边界、样本类型或结构风险建立联系，仍然可能被 reviewer 认为浅。
- 过关条件：至少要把 bucket 与 reasoning / code / structure-sensitive boundary 联系起来。

### 候选 C：Minimal Controller for Decoupling

- 风险最大：
  - 会把我们拉回“又一个 trick”
  - 一旦结果不明显，既花 compute 又伤 narrative
- 我倾向只在 A/B 完成后再决定是否做。

## 排名

1. 候选 A
2. 候选 B
3. 候选 C

## 我的底线要求

- abstract 不要提前承诺 full failure taxonomy，除非 bucket artifact 真出来。
- 不要把 MATH500/HumanEval 的小 slice 写成跨任务稳定规律。
- 任何新 controller 实验都必须 compute-matched 且 runtime metadata 完整，否则宁可不做。
