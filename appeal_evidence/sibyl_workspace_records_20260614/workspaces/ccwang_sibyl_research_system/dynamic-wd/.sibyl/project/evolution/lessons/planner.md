# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][EXPERIMENT] Statistical power insufficient for null-result claims. N=3 gives ~15-20% power for 0.5% effects. TOST equivalence confirmed in only 6/12 comparisons at delta=1.0%. A null-result paper requires higher evidentiary burden than a positive-result paper. (出现 3 次, 权重 2.14)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM][EXPERIMENT] NoBN ablation data from iter_005 exists (constant: 87.74, CWD: 87.62) but only 2/7 methods tested. Section 6.2 presents these as evidence for the BN mechanism hypothesis, but spread 0.11pp WITHOUT BN is narrower than 0.25pp WITH BN, contradicting the hypothesis. Presenting 2/7 as evidence for a 7-method claim is misleading. (出现 2 次, 权重 1.43)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。

## 继续保持
- Full 42-experiment design is clean, reproducible, and well-documented; serves as a reusable benchmark infrastructure. (出现 1 次)
- NoBN experiment design correct: constant 87.74+/-0.20 vs CWD 87.62+/-0.12 shows larger absolute drop than BN versions, suggesting BN does contribute to invariance. (出现 1 次)
- Three-stage review pipeline catches distinct issues: supervisor (experimental completeness, cross-iteration consistency), critic (V_t contradiction, Theorem 2 gap, PMP-WD misleading framing), writing review (Figure 1 placeholder, missing appendix, notation). Each stage adds unique value. (出现 1 次)
