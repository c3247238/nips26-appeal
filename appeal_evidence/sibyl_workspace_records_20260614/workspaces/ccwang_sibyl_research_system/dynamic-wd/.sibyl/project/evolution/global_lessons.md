# 西比拉全局经验总结 (自动生成)


## ANALYSIS 类问题

影响 agent: supervisor, critic, skeptic, reflection

- [MEDIUM] CSI listed as a primary contribution (Section 1.3, item 2) despite zero predictive value (rho<0.3, p>0.3) and arbitrary component weights (0.4, 0.3, 0.3). Sensitivity analysis tests only 3 nearly-identical weight sets. Flagged for 5+ iterations without demotion. (出现 3 次, 权重 2.14)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。
- [MEDIUM] Section 5.7 draws directional conclusions from two non-significant correlations (rho=-0.379, p=0.121; rho=0.045, p=0.858). The claim 'cumulative alignment may be more informative' is statistically improper without a formal Steiger's z-test. Flagged since iter_007. (出现 2 次, 权重 1.43)
  建议: 深化分析：不要 cherry-pick 结果、补充 ablation 和 baseline 对比、讨论局限性。

## EXPERIMENT 类问题

影响 agent: experimenter, server_experimenter, planner

- [MEDIUM] Statistical power insufficient for null-result claims. N=3 gives ~15-20% power for 0.5% effects. TOST equivalence confirmed in only 6/12 comparisons at delta=1.0%. A null-result paper requires higher evidentiary burden than a positive-result paper. (出现 3 次, 权重 2.14)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。
- [MEDIUM] NoBN ablation data from iter_005 exists (constant: 87.74, CWD: 87.62) but only 2/7 methods tested. Section 6.2 presents these as evidence for the BN mechanism hypothesis, but spread 0.11pp WITHOUT BN is narrower than 0.25pp WITH BN, contradicting the hypothesis. Presenting 2/7 as evidence for a 7-method claim is misleading. (出现 2 次, 权重 1.43)
  建议: 加强实验设计：在公认 benchmark 上评估、确保有 baseline 对比、做 ablation study。

## WRITING 类问题

影响 agent: sequential_writer, section_writer, editor, codex_writer

- [MEDIUM] Abstract implies full factorial design ('7 methods, 2 optimizers, 2 architectures, 2 datasets') suggesting 168 runs. Actual: 84 (ResNet-20) + 21 (VGG-16-BN SGD/CIFAR-10 only) = 105. Reviewers expecting VGG+AdamW or VGG+CIFAR-100 will be disappointed. (出现 2 次, 权重 1.43)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM] Paper's Phi Invariance Conjecture abstract claims 105 experiments across 2 optimizers, 2 architectures -- but paper body has only ResNet-20/AdamW. The abstract-body disconnect will be flagged by any reviewer reading past the first page. (出现 2 次, 权重 1.42)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。

## 成功模式 (继续保持)

- Rigorous statistical treatment: paired t-tests with Bonferroni correction, Cohen's d, TOST equivalence testing, and explicit power analysis. This is above the community norm for null-result papers. (出现 1 次)
- Well-framed falsifiable conjecture with explicit scope narrowing to match evidence — the Phi Invariance Conjecture is a model of how to handle null results scientifically. (出现 1 次)
- Three-stage review pipeline (supervisor + final_critic + Codex independent) consistently identifies issues before writing finalization. (出现 1 次)
- Visual audit with cell-by-cell data verification catches data integrity errors (e.g., SGD p-value inflation). (出现 1 次)
- Result debate synthesis with six perspectives produces well-balanced strategic decisions. (出现 1 次)
- Full 42-experiment design is clean, reproducible, and well-documented; serves as a reusable benchmark infrastructure. (出现 1 次)
- VGG-16-BN execution: all 21 runs completed successfully with consistent results (Phi spread 0.16%), confirming cross-architecture robustness of null result. (出现 1 次)
- NoBN experiment design correct: constant 87.74+/-0.20 vs CWD 87.62+/-0.12 shows larger absolute drop than BN versions, suggesting BN does contribute to invariance. (出现 1 次)
- Supervisor cross-validation of VGG-16-BN data against summary.json confirms Table 3 is accurate. Data integrity pipeline working for this configuration. (出现 1 次)
- Statistical honesty consistently maintained across 5 iterations: explicit power limitations, data gap disclosure, hedged language. (出现 1 次)
