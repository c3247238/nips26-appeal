# Supervisor Review: Iteration 004

## Overall Score: 7.0 / 10 (Weak Accept)

**Verdict: CONTINUE**

This iteration successfully addresses the previous review's critical concerns by removing the Goodhart's Law overreach, eliminating estimated PCA and low-cooccurrence data, and reverting to a clean construct-validity study. The paper now presents an honest, well-scoped contribution with three research questions (H1-H3) and real empirical data throughout.

---

## Dimension Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Novelty & Significance | 8 | The first construct-validity study of the SAEBench absorption metric is genuinely novel. The hierarchy-specificity failure and Random-SAE anomaly are important negative results for the SAE community. |
| Technical Soundness | 7 | The core methodology (adapt SAEBench protocol to semantic hierarchies, add non-hierarchy control, include Random-SAE baseline) is sound. Statistical tests are appropriate. But a critical data error in Table 1 undermines the central Random-SAE claim. |
| Experimental Rigor | 6 | The hierarchy-specificity test (H2) is robust (t=-4.748, p=0.003, d=-1.794). But H1 is underpowered (n=7, CI spans full range), and the GPT-2 replication is too thin to support confident conclusions. Single model, single layer, 10 shallow hierarchies. |
| Reproducibility | 7 | All raw data is available and matches paper claims (after correcting the Random-SAE error). The evaluation protocol is described precisely enough to reimplement. Code release location is missing. |

---

## Executive Summary

The paper makes a defensible, important contribution: it is the first empirical test of whether the dominant SAE absorption metric generalizes from artificial first-letter hierarchies to real semantic hierarchies. The answer, based on 8 SAE architectures on Pythia-160M, is "inconclusive for construct validity, but the metric fails hierarchy specificity."

The three main findings are:

1. **Construct validity is inconclusive** (H1): r=0.463, 95% CI [-0.389, 0.981]. The paper correctly labels this inconclusive, though the abstract could emphasize the uncertainty more.

2. **Hierarchy specificity fails** (H2): Non-hierarchy correlated features show significantly higher absorption than semantic hierarchies (t=-4.748, p=0.003, d=-1.794). This is the paper's strongest, most statistically decisive finding. All 8 architectures show the reversal.

3. **Random-SAE anomaly** (H3/Control): A randomized SAE achieves semantic-hierarchy absorption that is in the same range as trained SAEs. HOWEVER, the paper's central claim that Random=Standard (0.352) is contradicted by the actual data, which shows Random=0.175. This is a serious data error that must be fixed.

The paper is honest about negative results, well-structured, and appropriately scoped. But the experimental breadth is thin (one model, one layer, 10 shallow hierarchies), and the data error in Table 1 is a credibility risk.

**Bottom line**: A solid paper with a defensible negative result. Fix the data error, add sensitivity analyses, and consider expanding the SAE cohort or hierarchy depth to strengthen the contribution.

---

## Critical Issues

### 1. [CRITICAL] Random-SAE data error in Table 1

The paper reports Random SAE semantic-hierarchy = 0.352 and non-hierarchy = 0.416, identical to Standard SAE. But the raw data shows Random semantic-hierarchy = 0.175 and non-hierarchy = 0.233. This appears to be a copy-paste error where Standard's scores were duplicated into the Random row.

**Impact**: The paper's central claim---"a Random-SAE control yields semantic-hierarchy absorption of 0.352, identical to the Standard SAE"---is factually wrong. This undermines the most striking finding.

**Fix**: Update Table 1 with correct values (Random: SH=0.175, NH=0.233). Reframe the finding: "The Random SAE achieves semantic-hierarchy absorption of 0.175, below the Standard SAE's 0.352 but within the range of trained architectures (0.064-0.359). The first-letter task correctly distinguishes trained from random (0.030 vs 0.026), but the semantic adaptation shows reduced sensitivity to learned structure."

### 2. [CRITICAL] Cohen's d inconsistency

The paper reports d=-1.68 throughout, but e3_ttest_results.json shows d=-1.794.

**Fix**: Use the correct value (-1.794) consistently.

### 3. [MAJOR] H1 underpowered

With n=7, the bootstrap CI spans nearly the full correlation range. The abstract features the point estimate (r=0.463) without sufficient emphasis on uncertainty.

**Fix**: Lead with inconclusiveness in the abstract. Add power analysis.

### 4. [MAJOR] GPT-2 replication interpretation

Near-zero scores on GPT-2 could mean model-specific behavior OR metric adaptation failure.

**Fix**: Add discussion of alternative explanations. Acknowledge exploratory nature.

### 5. [MAJOR] Perfect probe ceiling effect

All hierarchies achieve AUROC=1.0, which may invalidate the metric adaptation.

**Fix**: Add sensitivity analysis for lower residual AUROC values.

### 6. [MAJOR] Hierarchy vs. non-hierarchy boundary

Pairs like "tree-wood" and "river-water" involve meronymy, blurring the distinction.

**Fix**: Discuss this limitation explicitly.

### 7. [MAJOR] No multiple comparison correction

~8 tests reported; hierarchy specificity survives Bonferroni but this is not stated.

**Fix**: Apply and report corrected p-values.

---

## What Would Raise the Score

**To 7.5-8.0**:
1. Fix Random-SAE data error and reframe honestly
2. Add sensitivity analysis for probe ceiling effect
3. Apply multiple comparison correction
4. Add 2-3 more SAE architectures
5. Test 1-2 deeper hierarchies (3-4 levels)

**To 8.5+**:
6. Add second base model with non-zero scores (e.g., Gemma-2-2B)
7. Include causal ablation validation (activation patching)
8. Test with natural-language templates

---

## Risks

- The Random-SAE data error, if discovered by reviewers, would severely damage credibility
- Small n=7 for H1 invites "why not collect more data before writing?" criticism
- GPT-2 near-zero scores may be interpreted as metric adaptation failure
- Perfect probe ceiling effect may question whether metric has discriminative power at all
