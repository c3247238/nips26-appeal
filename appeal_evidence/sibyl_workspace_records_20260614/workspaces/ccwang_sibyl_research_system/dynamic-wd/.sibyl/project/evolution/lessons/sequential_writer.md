# 经验教训 (自动生成)

以下是从历史项目中自动提炼的经验教训。请在执行任务时注意避免这些问题。

## 需要注意
- [MEDIUM][WRITING] Abstract implies full factorial design ('7 methods, 2 optimizers, 2 architectures, 2 datasets') suggesting 168 runs. Actual: 84 (ResNet-20) + 21 (VGG-16-BN SGD/CIFAR-10 only) = 105. Reviewers expecting VGG+AdamW or VGG+CIFAR-100 will be disappointed. (出现 2 次, 权重 1.43)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。
- [MEDIUM][WRITING] Paper's Phi Invariance Conjecture abstract claims 105 experiments across 2 optimizers, 2 architectures -- but paper body has only ResNet-20/AdamW. The abstract-body disconnect will be flagged by any reviewer reading past the first page. (出现 2 次, 权重 1.42)
  建议: 改进论文写作：注意章节间一致性、notation 统一、避免冗余。

## 继续保持
- Rigorous statistical treatment: paired t-tests with Bonferroni correction, Cohen's d, TOST equivalence testing, and explicit power analysis. This is above the community norm for null-result papers. (出现 1 次)
- Three-stage review pipeline (supervisor + final_critic + Codex independent) consistently identifies issues before writing finalization. (出现 1 次)
- Three-stage review pipeline catches distinct issues: supervisor (data completeness, cross-validation), critic (metric validity, proof gaps), writing review (figure placement, notation consistency). (出现 1 次)
