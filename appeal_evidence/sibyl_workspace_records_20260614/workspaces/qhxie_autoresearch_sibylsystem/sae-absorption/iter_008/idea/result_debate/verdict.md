# Result Debate Verdict

**Project:** SAE Feature Absorption Cross-Domain Characterization
**Date:** 2026-04-15
**Overall Quality Score:** 7.0 / 10

---

## Key Conclusion

This project has produced genuinely novel empirical results on SAE feature absorption that no prior work has reported: a 15x variation in absorption across model layers, statistically significant differences across hierarchy types, the first interventional causal evidence for feature suppression in SAEs, and a methodological critique showing that the standard hedging metric is near-tautological. Three proposed unsupervised detectors (GAS, CMI, Absorption Tax) fail definitively, contributing honest negative results.

The evidence base is strong enough to support a top-venue publication, with one critical caveat: the cross-domain comparison is confounded by differential probe quality (rho=-0.756 between probe F1 and false negative rate). A degraded-probe ablation experiment (1-2 GPU hours) would resolve this ambiguity and is strongly recommended before submission.

---

## Score Breakdown

| Dimension | Score | Key Evidence |
|-----------|-------|-------------|
| Novelty | 8/10 | First cross-domain measurement, first layer-dependence characterization, first interventional evidence |
| Statistical rigor | 7/10 | Activation patching d=1.33, p<0.001. Cross-domain ANOVA p=0.005 but confounded by probe quality |
| Evidence completeness | 6/10 | Probe quality ablation missing. Cross-domain causal evidence absent. Architecture comparison underpowered |
| Negative results | 9/10 | GAS, CMI, Tax cleanly refuted. H2' honestly reported as refuted |
| Practical impact | 6/10 | Shows SAEBench single-task evaluation is insufficient. No mitigation proposed |

---

## Decision: PROCEED to Writing

### Three Must-Do Actions (Zero GPU Cost)

1. **Report probe-only FN baselines** -- separates probe error from SAE-induced absorption
2. **Run restricted Wilcoxon** on high-confidence words (raw accuracy >= 0.50) -- validates causal evidence on representative tokens
3. **Restructure narrative** -- from "semantic > syntactic" to "layer-dependent x hierarchy-dependent interaction"

### One Strongly Recommended Action (1-2 GPU Hours)

4. **Degraded-probe ablation** -- degrade first-letter probe to F1~0.84, remeasure absorption. Resolves the single biggest reviewer concern.

### Target Venue

**ICLR 2027** (primary) | **COLM 2027 / EMNLP 2026** (backup) | **arXiv preprint** (immediate upon writing completion)
