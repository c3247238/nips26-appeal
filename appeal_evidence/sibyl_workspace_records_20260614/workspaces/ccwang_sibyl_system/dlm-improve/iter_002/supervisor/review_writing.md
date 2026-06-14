# Supervisor Review

## 总评

评分：`6.4/10`  
结论：`revise`

这版稿子最大的优点是收缩得足够诚实。`writing/paper.md` 已经不再强行包装新 controller，而是把贡献聚焦到 bucket decomposition、runtime-lineage audit 和 minimal seed closure，这个方向是对的。基于当前 workspace，我能核到的核心链条也基本成立：`benefit_bucket_audit_pilot.json` 支持 `+3pp` 与 `7 fixed / 4 harmed / 89 no_effect`，`seed_sensitivity_spotcheck.json` 支持三种子同向但不支持 full robustness，`min_controller_decoupling_probe.json` 也与“不要重新打开 hero-method 叙事”的写法一致。

但这版还不能判 `done`。最关键的原因不是 headline 数字错了，而是交付形态还不够可审计：`exp/results/summary.md` 缺失，我只能改用代表性资产做交叉验证；同时 `canonical_asset_manifest.json` 和 `runtime_fairness_matrix.json` 里的多个主资产仍指向当前 workspace 之外的绝对路径，导致“claim-to-asset lineage 可审计”这个元主张在本次实际交付里没有完全闭环。

## 主要问题

1. `summary.md` 缺失，canonical evidence bundle 不完整。  
本次审查明确发现 `current/exp/results/summary.md` 不存在。我因此改用 `runtime_probe_iter2.json`、`benefit_bucket_audit_pilot.json`、`seed_sensitivity_spotcheck.json`、`runtime_fairness_matrix.json`、`observer_controller_protocol.json`、`canonical_asset_manifest.json` 和 `min_controller_decoupling_probe.json` 作为代表性结果资产。这个替代是可行的，但它本身说明交付还没收口。

2. claim lineage 在 `current` workspace 内不自洽。  
`current/exp/results/canonical_asset_manifest.json` 里若干 `primary_artifact` 指向 `/home/ccwang/...`，在当前机器上不可达；而这些资产实际上出现在 `iter_001/exp/results/`。这意味着 reviewer 若只拿到 `current` workspace，无法复核 paper 提到的部分 boundary / protocol 证据。

3. honest-compute 论证仍混杂异质 runtime 条件。  
`runtime_probe_iter2.json` 给出的推荐执行路径是 `eager|compile=True` 且 safe batch 为 `57`。但 `iter_001/exp/results/diag_compute_curve_gsm8k.json` 与 `current/exp/results/runtime_fairness_matrix.json` 中的方法比较混用了不同 compile 状态和 batch size，例如 `Standard-64` 是 `compile=True, batch_size=115`，`Entropy-Revise-64+3` 是 `compile=False, batch_size=32`，`CORE-proxy-64` 甚至是 `batch_size=1`。如果正文不更明确地区分“runtime realization audit”和“受控方法对比”，reviewer 会质疑这是不是 implementation artifact。

4. 同一 headline pair 的数字在不同资产间漂移，但正文没有解释。  
`diag_compute_curve_gsm8k.json` 是 `0.36 -> 0.39`，`benefit_bucket_audit_pilot.json` 是 `0.34 -> 0.37`，`seed_sensitivity_spotcheck.json` 的 reference seed 是 `0.36 -> 0.37`。这些都能支持“方向没变、幅度不大”，但现在稿子没有明确说明它们属于不同 seed / slice / runtime contract，读者很容易认为这是口径不稳定。

5. scholarly packaging 还没完成。  
`paper.md` 的 Related Work 没有 citation 或 bibliography。对内部迭代来说这不稀奇，但对 supervisor 质量门来说，这是明确的未完成项。

## 风险判断

- 最大风险是“可复核性被高估”。稿子反复强调 auditable protocol，但当前交付并没有把所有关键资产真正打包到同一 workspace 里。
- 第二个风险是“honest compute 被误读为算法结论”。如果不把 runtime confound 写得更直白，外部读者会把排序变化当成方法本身的证据，而不是 realized execution 的证据。
- 第三个风险是“narrative selection”质疑。当前收缩方向是正确的，但如果不补上 artifact reconciliation，reviewer 仍可能质疑为什么这里报 `0.34 -> 0.37`，那里又报 `0.36 -> 0.39`。

## 建议动作

1. 补出 `exp/results/summary.md`，并让它明确指向本轮 paper 真正依赖的资产。
2. 把 `canonical_asset_manifest.json`、`runtime_fairness_matrix.json` 里的路径统一成 workspace-relative 或 iteration-relative，可直接在当前交付里打开。
3. 在正文新增一段 artifact reconciliation，解释 compute curve、bucket audit、seed spot-check 三组数字分别对应什么数据切片。
4. 在摘要与结论直接写出 `n=100`、`3 seeds`、`sign consistency only`，并给 `+3pp` 增加最小不确定性刻画。
5. 补齐 related work 的最小引用集。

## 本次审查使用的证据

- 论文：`current/writing/paper.md`
- 当前 iteration 结果：`current/exp/results/runtime_probe_iter2.json`
- 当前 iteration 结果：`current/exp/results/benefit_bucket_audit_pilot.json`
- 当前 iteration 结果：`current/exp/results/benefit_bucket_examples_pilot.json`
- 当前 iteration 结果：`current/exp/results/seed_sensitivity_spotcheck.json`
- 当前 iteration 结果：`current/exp/results/runtime_fairness_matrix.json`
- 当前 iteration 结果：`current/exp/results/observer_controller_protocol.json`
- 当前 iteration 结果：`current/exp/results/canonical_asset_manifest.json`
- 当前 iteration 结果：`current/exp/results/min_controller_decoupling_probe.json`
- 直接相关 supporting artifacts：`iter_001/exp/results/diag_compute_curve_gsm8k.json`
- 直接相关 supporting artifacts：`iter_001/exp/results/diag_signal_gap_audit.json`
- 直接相关 supporting artifacts：`iter_001/exp/results/diag_math500_shortlist.json`
- 直接相关 supporting artifacts：`iter_001/exp/results/diag_humaneval_guard_boundary.json`
- 直接相关 supporting artifacts：`iter_001/exp/results/final_pareto_synthesis.json`
