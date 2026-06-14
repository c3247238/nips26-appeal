# Writing Quality Review (Iteration 8)

## Summary

The paper presents the Phi Modulator Framework and Phi Invariance Conjecture with 105 controlled experiments. Iteration 8 addressed several issues from iter 7: BEM half_lambda corrected to 0.5, cumulative alignment analysis reframed as non-significant observation, Figure 3/8 captions updated.

## Reviewer Scores

| Reviewer | Score | Key Concerns |
|----------|:-----:|--------------|
| Critic | ~6.5 | 17 findings (5 critical): PMP-WD ghost method in figures, undefined Lyapunov certificate, experiment count discrepancy, figure-text inconsistencies |
| Supervisor | 6.5 | Data provenance mismatch (0.33pp inter-batch > 0.25pp inter-method), no appendix proofs, no ImageNet, Lyapunov contradiction |

## Dimension Scores (Supervisor)

| Dimension | Score |
|-----------|:-----:|
| Novelty | 7.0 |
| Soundness | 5.5 |
| Experiments | 6.0 |
| Reproducibility | 5.0 |

## Critical Issues Remaining

1. **[Critical] PMP-WD ghost method**: Figures 3, 8, 9 show PMP-WD which is not in the paper's 7-method catalog. Must regenerate figures or add PMP-WD to method list.

2. **[Critical] Lyapunov certificate undefined**: Section 5.7 references a "Lyapunov stability certificate" never defined. Must add formal definition or remove section.

3. **[Critical] Data provenance mismatch**: Paper mixes iter_003 data (Table 2) with iter_006 data (PMP-WD). The 0.33pp inter-batch variance exceeds the 0.25pp inter-method spread claimed as the main finding.

4. **[Major] No appendix with proofs**: Theorems 1-4 stated without proofs for 5 consecutive iterations.

5. **[Major] No ImageNet experiments**: CIFAR-only evaluation limits venue eligibility.

6. **[Major] Experiment count discrepancy**: Abstract says 105, but paper body only presents ~63 explicitly in tables.

7. **[Major] Build manifest in paper**: Lines 432-448 contain internal file listing that must be removed.

## What Works Well

1. Statistical reporting is exemplary (paired t-tests, Bonferroni correction, Cohen's d, TOST)
2. Table 1 method catalog is excellent
3. SGD boundary condition analysis (Section 5.2/5.6) is the strongest section
4. Abstract precision with specific numbers
5. Honest limitations section

## What Would Raise Score to 8.0

Per supervisor: (1) Rerun all methods with consistent data provenance (~8 GPU-hours), (2) Write appendix with proofs (~6 hours), (3) Add ImageNet-100 experiments (~6-8 GPU-hours), (4) Resolve V_t/Lyapunov contradiction, (5) Add more seeds for TOST (~2 GPU-hours).

## Issues for the Editor

1. Remove "Figures and Tables" build manifest (lines 432-448)
2. Specify best vs final test accuracy in Section 4
3. Fix forward reference to Figure 6 before Figures 3-5
4. Regenerate Figures 3, 8, 9 without PMP-WD or add PMP-WD to method catalog
5. Either define certified band formally or remove Section 5.7
6. State N=18 derivation in Section 5.8

SCORE: 6.5
