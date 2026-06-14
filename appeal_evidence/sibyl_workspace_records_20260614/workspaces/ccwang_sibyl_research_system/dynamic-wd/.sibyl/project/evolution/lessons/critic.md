# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][ANALYSIS] CSI listed as a primary contribution (Section 1.3, item 2) despite zero predictive value (rho<0.3, p>0.3) and arbitrary component weights (0.4, 0.3, 0.3). Sensitivity analysis tests only 3 nearly-identical weight sets. Flagged for 5+ iterations without demotion. (出现 3 次, 权重 2.14)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM][ANALYSIS] Section 5.7 draws directional conclusions from two non-significant correlations (rho=-0.379, p=0.121; rho=0.045, p=0.858). The claim 'cumulative alignment may be more informative' is statistically improper without a formal Steiger's z-test. Flagged since iter_007. (出现 2 次, 权重 1.43)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。

## 继续保持
- Result debate synthesis with six perspectives produces well-balanced strategic decisions. (出现 1 次)
- Rigorous statistical treatment: paired t-tests with Bonferroni correction, Cohen's d, TOST equivalence testing, and explicit power analysis. This is above the community norm for null-result papers. (出现 1 次)
- Well-framed falsifiable conjecture with explicit scope narrowing to match evidence — the Phi Invariance Conjecture is a model of how to handle null results scientifically. (出现 1 次)
