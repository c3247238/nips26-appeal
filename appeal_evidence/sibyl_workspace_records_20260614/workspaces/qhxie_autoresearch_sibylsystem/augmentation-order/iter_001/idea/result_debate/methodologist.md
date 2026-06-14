# Methodologist Audit: Does Augmentation Operation Ordering Matter?

**Auditor**: Methodologist Agent
**Date**: 2026-04-02
**Data source**: `exp/results/full/final_summary.json`, `exp/results/full/tier1_analysis.json`, `exp/results/full/tier4_correlation.json`, `exp/results/full/tier3_results.json`, `exp/results/full/tier4a_nc.json`
**Mode**: All results are from PILOT mode (limited epochs, reduced dataset subsets, single seeds for most experiments)

---

## 1. Baseline Fairness Audit

### Critical Issue: Asymmetric training budgets between orderings and baselines

The ordering experiments (Tier 1) used **10 epochs on 100-sample subsets** (pilot mode), producing accuracies in the range of 10-20% (CIFAR-10 ResNet-18: 10.01-10.97%). Meanwhile, the baselines ran for **30 epochs on the full 50k training set**, achieving 91-93% (CIFAR-10 ResNet-18). This is a **9x epoch difference** and a **500x data size difference**.

**Evidence**:
- Tier 1 ordering accuracies: 10.01% - 10.97% (CIFAR-10 RN18), from `tier1_analysis.json` note: "PILOT mode: 10 epochs, 100-sample subsets"
- Baseline conventional: 91.91% (CIFAR-10 RN18), from `baselines_cifar10_pilot.json`: "epochs: 30" on full dataset

**Implication**: The ordering results and baseline results exist in entirely different performance regimes. The 0.96% spread between orderings on CIFAR-10 ResNet-18 (10.01% to 10.97%) occurs at near-random-chance accuracy. It is methodologically unsound to compare these results in the same table (Table 1 of the final summary) or draw conclusions about practical relevance from them.

### Secondary Issue: Baselines not run under same pilot conditions

The baselines were trained with standard augmentation pipelines at full scale, while the orderings were tested at pilot scale. A fair comparison requires running both under identical conditions (same epoch count, same data size). The current setup cannot answer whether any ordering matches or exceeds baseline performance.

**Severity**: HIGH. This asymmetry makes the main results table (Table 1) misleading.

---

## 2. Metric-Claim Alignment

### Claim 1: "Augmentation ordering significantly affects accuracy" (H1)

**Metric used**: Max-min spread across 6 orderings.
**Assessment**: The metric is appropriate for detecting *whether* differences exist. However, at 10 epochs on 100 samples, the measured accuracies are dominated by stochastic initialization effects, not by augmentation ordering. The spread of 0.96% on CIFAR-10 RN18 translates to roughly 1 correctly classified image difference out of 100, at an overall accuracy below random baseline for some orderings.

**Measurement gap**: No confidence intervals, no paired t-tests (t_stat and p_value are null for all blocks due to n_seeds=1). The methodology document specifies "5 seeds [42..46], paired t-test" but the pilot used only 1 seed. Cohen's d values reported (2.27-2.71) are computed from single-seed runs and are therefore meaningless as effect size estimates.

### Claim 2: "Reversibility-sorted ordering outperforms conventional" (H2)

**Metric used**: Binary win/loss across 4 arch-dataset blocks.
**Assessment**: At 2/4 blocks, this barely meets the threshold of "50% of blocks." The differences are tiny in absolute terms:
- RN18-CIFAR10: +0.78% (but at 10% accuracy level)
- ViT-CIFAR10: +0.13%
- RN18-CIFAR100: -0.38% (conventional wins)
- ViT-CIFAR100: -0.16% (conventional wins)

The claim "confirmed" is premature. A 2/4 win rate is indistinguishable from random chance without statistical testing.

### Claim 3: "NC_2 predicts accuracy ranking" (H3)

**Metric used**: Spearman rho between NC_2 proxy and accuracy difference.
**Assessment**: The metric is appropriate. The result (rho=-0.20, p=0.68) correctly leads to "falsified." However, the NC_2 was computed on only 100 samples with 100 projections. The methodology specified 10k samples and 1000 projections. The pilot NC_2 values may be unreliable due to this 100x reduction in sample size and 10x reduction in projections.

### Claim 4: "InfoNCE MI correlates with accuracy" (H4)

**Assessment**: The contradictory sign between CIFAR-10 (rho=+0.54) and CIFAR-100 (rho=-0.66) is a strong signal that the MI estimation procedure is unreliable at pilot scale, not that the hypothesis is "inconclusive." The paired comparison results show suspicious anomalies: `conventional_acc: 0.14, reversibility_sorted_acc: 0.14` in the CIFAR-10 block, which are different from the Tier 1 accuracies (0.1019 vs. 0.1097). This suggests different data was used for the MI estimation vs. the accuracy measurement, or a data processing bug.

---

## 3. Validity Threats Checklist

- [x] **Data leakage**: No evidence of data leakage. Standard CIFAR train/test splits used.
- [ ] **Contamination**: Not applicable for CIFAR-10/100 classification.
- [x] **Selection bias**: The "best" and "worst" orderings for Tier 3 were selected from Tier 1 pilot results (order_2 and order_5). With n=1 seed and 10 epochs, this selection is likely dominated by noise. The Tier 3 results then test these two orderings, inheriting any selection bias.
  - **Evidence**: In the Tier 1 pilot, order_5 (CJ->Flip->Crop) is the BEST ordering on CIFAR-10 RN18 (0.1097) but the WORST on CIFAR-100 RN18 (0.4575). The best/worst ordering is inconsistent across blocks, suggesting the selection for Tier 3 may be capturing noise rather than signal.
- [x] **Overfitting to evaluation**: The 100-sample training subsets for Tier 1 create extreme overfitting risk. With only 100 training samples and 10k validation samples, the training dynamics are in a regime unrepresentative of practical use. Augmentation effects at 100 samples may differ fundamentally from effects at 50k samples.
- [x] **Pilot-to-full extrapolation risk**: All "confirmed" verdicts rest on pilot data. The Tier 0 pilot (5k samples, 10 epochs, 2 seeds) showed a 2.68% spread with CJ-first orderings winning. But the Tier 1 pilot (100 samples, 10 epochs, 1 seed) shows a different winner on CIFAR-100 (Flip->Crop->CJ). The reversal between Tier 0 and Tier 1 pilot winners undermines confidence that pilot results will hold at full scale.

---

## 4. Ablation Gap Analysis

### Proposed Components and Ablation Coverage

| Component | Has Ablation? | Notes |
|---|---|---|
| Ordering permutation (3 ops) | YES (Tier 1) | All 6 permutations tested, but at pilot scale only |
| Category-level ordering (6 ops) | PARTIAL | Tier 2 pilot exists but only for ResNet-18 on CIFAR-10 5k subset; no ViT, no CIFAR-100 |
| Magnitude interaction | YES (Tier 3) | But only 1 seed, 10 epochs; spread at M14 = 0.00 undermines conclusions |
| Architecture differential (CNN vs. ViT) | PARTIAL | Both architectures run, but no statistical interaction test possible with 1 seed |
| NC_2 theoretical measure | YES (Tier 4a) | But severely underpowered (100 samples, 100 projections vs. planned 10k/1000) |
| InfoNCE MI estimation | YES (Tier 4b) | Computed, but contradictory signs across datasets |

### Missing Ablations

1. **Augmentation vs. no augmentation at pilot scale**: The "no augmentation" baseline was run at full scale (30 epochs, full dataset) but not at pilot scale (10 epochs, 100 samples). We cannot isolate whether ordering effects persist relative to no-augmentation at matched training budgets.
2. **Random ordering at pilot scale**: The random-per-image baseline was similarly run only at full scale. A matched pilot comparison would strengthen the H1 finding.
3. **Number of operations ablation**: Only K=3 and K=6 tested. No test with K=2 (pairwise ordering is the simplest meaningful case and would directly validate the pairwise NC_2 predictions).

---

## 5. Reproducibility Score: 2/5

| Criterion | Score | Justification |
|---|---|---|
| Random seeds fixed | 1/1 | Seed 42 used consistently; but only 1 seed means no variance estimation |
| All hyperparameters specified | 1/1 | Detailed in methodology.md (lr, momentum, wd, batch size, etc.) |
| Code/data availability | 0.5/1 | Training scripts in exp/scripts/; datasets are standard CIFAR |
| Hardware requirements documented | 0.5/1 | "RTX PRO 6000" mentioned; but actual pilot may have used different GPU |
| Reproducible within 10% | 0/1 | Cannot assess with n=1 seed; no variance reported for core results |

**Key reproducibility concern**: All Tier 1 ordering results, all Tier 3 results, and all Tier 4 results are from **single seed runs**. The methodology plan specifies 5 seeds, but only 1 was used. Without multi-seed runs, we cannot distinguish genuine ordering effects from initialization-dependent noise.

---

## 6. Top-3 Recommendations (ordered by effort-to-credibility ratio)

### Recommendation 1: Run Tier 1 at full scale with 5 seeds (HIGH PRIORITY)

**What**: Execute the planned 120-run Tier 1 experiment (6 orderings x 2 architectures x 2 datasets x 5 seeds x 200 epochs).
**Why**: This is the single highest-impact improvement. The entire H1/H2 argument rests on the ordering spread being real, but with n=1 seed we cannot compute p-values (all t_stat fields are null). The Cohen's d values reported are mathematically vacuous with a single observation. A finding of ~1% spread could easily be within seed-level variance.
**Effort**: ~10 hours on 2 GPUs (per the methodology plan). The infrastructure is already built and validated.
**What changes**: If the spread persists with 5 seeds and p < 0.05, H1 is genuinely confirmed. If it disappears, the pilot detected noise, saving the project from publishing a spurious result.

### Recommendation 2: Match training conditions between orderings and baselines

**What**: Either (a) run baselines at pilot scale (10 epochs, 100 samples) to create a matched comparison, or (b) run orderings at full scale (200 epochs, full dataset). Preferably option (b) since it also addresses Recommendation 1.
**Why**: The current Table 1 juxtaposes 10% ordering accuracies with 92% baseline accuracies. A reviewer will immediately notice this mismatch and question the validity of all comparisons. The "practical recommendations" suggesting "Use CJ->Flip->Crop (acc=0.1097)" are meaningless at 10.97% accuracy -- this is random chance on CIFAR-10.
**Effort**: Subsumed by Recommendation 1 if orderings are run at full scale.
**What changes**: Creates a fair, reviewable comparison between proposed orderings and established baselines.

### Recommendation 3: Increase NC_2 sample size from 100 to 10,000

**What**: Re-run the Tier 4a NC_2 computation with 10k samples and 1000 projections (as originally planned).
**Why**: The SWD proxy with 100 samples in 32x32x3 = 3072 dimensions is statistically unreliable. The NC_2 values are very close together (0.035-0.051), and the ranking could flip entirely with more samples. The H3 "falsified" verdict may be an artifact of the severely underpowered NC_2 estimation.
**Effort**: <30 minutes (CPU only, no training). Minimal effort for potentially changing the H3 verdict.
**What changes**: If NC_2 ranking changes with 10k samples and now correlates with accuracy, H3 moves from "falsified" to a potential confirmation, strengthening the paper's theoretical contribution.

---

## Summary of Methodology Concerns

The experimental infrastructure is well-designed and the methodology plan is sound. The problem is that the results presented are all from a severely underpowered pilot. The core issues are:

1. **All hypothesis verdicts are premature**: No result has the planned statistical power (5 seeds, 200 epochs). Every "confirmed" or "falsified" verdict should be labeled "provisional pilot finding."
2. **The accuracy regime is wrong**: At 10% accuracy on CIFAR-10, the model has not learned meaningful features. Augmentation ordering effects in this regime may not transfer to the 90%+ accuracy regime where practitioners care about ordering.
3. **Contradictory orderings across settings**: CJ->Flip->Crop wins on CIFAR-10 RN18 but loses on CIFAR-100 RN18. This inconsistency, combined with n=1 seed, strongly suggests the results are dominated by stochastic noise rather than a genuine ordering effect.
4. **Tier 3 M14 spread = 0.00 is suspicious but not for the stated reason**: The notes attribute this to "very aggressive augmentation overwhelming any ordering effect." An alternative explanation: at 10 epochs, the model has barely started learning under M14 augmentation (24.5% accuracy on CIFAR-100), so the ordering signal has not had time to manifest. Full 200-epoch runs are needed to distinguish these explanations.

**Bottom line**: The methodology is well-conceived but the execution is at pilot scale only. No finding should be presented as confirmed until the planned full-scale experiments are completed. The pilot successfully demonstrates that the infrastructure works and that there may be a signal worth investigating -- which is exactly what a pilot should do.
