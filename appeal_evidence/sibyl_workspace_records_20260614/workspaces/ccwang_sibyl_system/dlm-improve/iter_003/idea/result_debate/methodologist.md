# Methodologist View

## 方法学结论

从方法论视角看，这一轮最有价值的地方不是性能数字本身，而是它把“什么样的对照和资产，才足以让一个 small-gain claim 站住”这件事讲清楚了。当前 negative-case lane 的方法学价值，主要来自 **baseline fairness + sham control + sample-level audit + current-only artifact closure** 的组合。

## 1. baseline fairness 审计

`main_results_table.csv` 给出了四个臂：

- `dnb64`
- `dnb84`
- `card84`
- `rand84`

这四个臂的设计在当前 audited slice 内部是有逻辑闭环的：

1. `DNB-64` 提供更低 compute 的参考点
2. `DNB-84` 提供 compute-matched active control
3. `CARD-84` 是被审计对象
4. `RAND-84` 提供 budget-matched sham control

其中最关键的方法学升级是 `RAND-84`。没有它，我们最多只能知道 `CARD-84` 不只是“多跑几步”的结果；有了它，我们才开始知道该信号是否超出 budget-matched random targeting。

## 2. sham control 对识别性的影响

`decision.json` 与 `repair_harm_table.csv` 一起表明：

- `CARD-84` 相对 `DNB-84` 的 GSM8K `net_repaired = +7`
- `CARD-84` 相对 `RAND-84` 的 GSM8K `net_repaired = +1`

这组数字的方法学含义非常直接：

1. compute-matched baseline 解决的是 **compute fairness**
2. sham control 进一步逼近的是 **identification of targeting value**

也就是说，当前 negative case 之所以成立，不是因为 `CARD-84` 没有任何局部作用，而是因为在更强识别条件下，这种局部作用不再足以支撑正向方法语言。

## 3. sample-level audit 与 current-only artifact closure 的价值

`validation_report.json` 明确写出：

- `joinable = true`
- `current_only_bundle = true`
- required artifacts 全部存在

这意味着当前结论不是凭 aggregate summary 生出来的，而是可以回到：

1. `per_sample_audit.csv`
2. `transition_matrix.csv`
3. 各 arm 的 `metrics.json`
4. `claim_to_asset_map.json`
5. `code_failure_modes.md`

这种 asset closure 对 negative-case 尤其重要，因为 reviewer 最容易怀疑“是不是你们只是因为没有更好的结果，才临时把故事改写成负案例”。有了 current-only joinable bundle，这个怀疑会被大幅削弱。

## 4. 仍存在的内外部效度限制

### 4.1 内部效度限制

1. 当前仍是单 seed、单 slice，统计稳健性有限。
2. trajectory addon 被跳过，因此我们没有更细粒度的 revision 轨迹解释。
3. 当前 runtime contract 是 logged stable eager path，不构成性能优化贡献。

### 4.2 外部效度限制

1. 当前证据范围被 `proposal.md` 明确限定为 audited slice。
2. 不能把当前发现直接外推成一般性的 DLM revision law。
3. minimal audit template 的价值目前是 supporting contribution，而不是 protocol paradigm。

## 5. 方法学上的最安全结论

如果只保留一条最稳的结论，我会写成：

**在 training-free DLM revision 的 small-gain setting 中，仅做 compute-matched active control 不足以支撑正向方法解释；引入 budget-matched sham control、sample-level audit 和 current-only artifact closure 后，原本看似 promising 的局部增益被改写成 audited negative case。**

这条结论的价值不在于夸大当前结果，而在于它展示了什么样的最小证据结构，足以防止我们把 small gain 误写成 validated method。
