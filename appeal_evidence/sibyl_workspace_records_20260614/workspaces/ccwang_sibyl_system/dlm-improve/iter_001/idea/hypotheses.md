# Testable Hypotheses for `cand_diag`

## Scope

当前 front-runner 不再是假设某个新 revision policy 必然取胜，而是验证一个更强也更诚实的命题：

**revision gain 在 DLM 中是 task-dependent、compute-dependent、failure-mode-dependent 的。**

## H1: Honest Compute Reorders the Ranking

**Statement**:
当比较从 nominal NFE 改为 actual compute 后，training-free DLM 方法的排名会发生稳定重排。

**Measurement**:

- 对 `Standard / DNB / Prophet / CORE-proxy / Entropy / TIGER` 统一记录：
  - actual NFE
  - latency
  - tokens/sec
  - accuracy / pass@1
- 对比 nominal ranking 与 compute-normalized ranking

**Expected outcome**:

- `CORE-proxy` 继续保持优势
- 复杂 revision 方法的领先幅度缩小，甚至消失

**Falsification**:

- 如果 compute-normalized ranking 与 nominal ranking 基本一致，这条主张就不成立

## H2: Revision Benefit Depends on Task Recoverability

**Statement**:
revision 在 reasoning 与 code 上的净效应不同，原因是两类任务的结构恢复力不同。

**Measurement**:

- Reasoning: `GSM8K`
- Code boundary: `HumanEval`
- 比较同一方法族在两类任务上的 gain / harm
- 额外拆分 syntax failure、runtime failure、semantic failure

**Expected outcome**:

- reasoning 上存在有限但可见的 revision benefit
- code 上更容易出现结构性 harm，尤其是 runtime / semantic 层面

**Falsification**:

- 如果 reasoning 与 code 对 revision 的响应几乎同构，这条任务差异命题就站不住

## H3: Observers Are More Useful Than Controllers

**Statement**:
entropy / instability / calibration 对 revision benefit 的预测力，高于它们直接作为控制信号时带来的性能提升。

**Measurement**:

- 比较三类信号与以下量的相关性：
  - final accuracy
  - revision benefit bucket
  - draft wrong -> revised right 的概率
- 同时比较这些信号驱动的 policy 实际增益

**Expected outcome**:

- 三类信号对 benefit buckets 有解释力
- 但把它们直接变成控制律时，收益并不稳定

**Falsification**:

- 如果某个信号既强解释又强控制，那就应该重新回到 method-forward 路线

## H4: Syntax Guard Fixes Shallow Errors, Not Deep Ones

**Statement**:
轻量 syntax guard 能降低 parse failure，但不能单独恢复 runtime / semantic correctness。

**Measurement**:

- 比较 `Standard`、`Ungated revision`、`Gated revision`
- 指标：
  - syntax failure
  - runtime failure
  - pass@1

**Expected outcome**:

- syntax failure 下降
- runtime failure 不一定同步下降，甚至可能反弹
- 总体 pass@1 不一定超过 Standard

**Falsification**:

- 如果 syntax guard 同时提升 syntax / runtime / pass@1，那么它就值得升级为真正方法贡献

## Decision Rule

继续 `cand_diag` 作为主线，前提是至少满足以下两条：

1. H1 成立：compute normalization 确实改变排序或叙事重点
2. H2/H4 至少一条成立：task-dependent harm 有清晰模式
3. H3 成立：观测信号更像诊断器而非控制器

如果这些都不成立，则优先转向：

- `cand_minimal`
- 或 `cand_factorization`
