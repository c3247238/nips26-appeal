# Backup Ideas for the Reopened Candidate Pool

## Alternative A: `cand_minimal`

### 标题

**Simple-Enough Beats Clever: Minimal Intervention Baselines for DLM Sampling**

### 核心命题

如果复杂 revision policy 在 honest compute 下并没有稳定优势，那么下一轮最值得测试的不是“再复杂一点”，而是“能不能更简单一点”。

### 为什么值得保留

- 实现代价最低
- 与反对者视角强一致
- 一旦成立，论文 punchline 会非常锋利

### 最小 pilot

- `GSM8K 100`
- 一个极简干预 baseline
- 对比 `DNB / Prophet / Entropy`

### 风险

- 可能只得到“也差不多”，但没有足够强的 headline

## Alternative B: `cand_factorization`

### 标题

**Factorization Error Audit and Dependency-Aware Grouping**

### 核心命题

revision 失败并不一定因为选错了 token，而可能因为 token 独立更新本身破坏了结构一致性。若如此，就该先做 factorization audit，再决定是否要实现 dependency-aware grouping。

### 为什么值得保留

- 理论味道最强
- 能解释 code 边界上的结构性失败
- 是本轮唯一像样的方法创新备胎

### 最小 pilot

- offline dependency proxy
- 对照 revision harm buckets
- 不急着实现新 sampler

### 风险

- proxy 可能很弱，最后既不够理论也不够工程

## Alternative C: Code-First Stress Test Companion

### 标题

**Revision under Structural Rigidity: Code Tasks as a Boundary Condition**

### 核心命题

如果主线最终仍以 reasoning 为中心，可以保留一个 companion 角度：code 不是泛化成功的加分项，而是暴露 revision 结构脆弱性的边界任务。

### 为什么值得保留

- 与现有 HumanEval 结果直接对接
- 能把 gating 的价值放在 appendix / boundary，而不是主 claim

### 风险

- 不能单独成为主线，只能作为附属贡献
